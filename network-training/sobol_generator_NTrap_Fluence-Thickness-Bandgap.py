"""
25/02/2026 Robin Heumann
Sobol sequence generator for N-trap TRPL simulations (variable trap count).

Generalises the OneTrap / TwoTrap sobol generators to an arbitrary number of
traps.  Set N_TRAPS below and the bounds / filter / HDF5 layout are built
automatically.

Parameter layout (all in log10 space, matching multitrap_rem_agent.rem_agent):
  [n_pulse, Eg, k_rad,
   Ntrap_1, DeltaEtrap_1, taun_1, taup_1,
   Ntrap_2, DeltaEtrap_2, taun_2, taup_2,
   ...
   Ntrap_N, DeltaEtrap_N, taun_N, taup_N]

  DeltaEtrap is the fractional position within the bandgap (0 = VB edge,
  1 = CB edge).  The filter Etrap_1 < Etrap_2 < ... < Etrap_N is applied
  so that ordering does not matter for training and duplicates are removed.

Output: HDF5 file in ./Synthetic-Training-Data/
  Datasets:
    lb, ub          – log10 of lower / upper bounds (1-D, length = n_params)
    par_mat_ln      – Sobol samples in log10 space  (n_samples × n_params)
    Constants       – scalar constants (Nc, Nv, Vt, DECAY_MAGNITUDE,
                      QFLS_POINTS, N_TRAPS)
    Y1_tarr         – time grid used for ODE solver (length = QFLS_STEPS)
    PL, time, QFLS, tau_diff – pre-allocated simulation output arrays
                                (n_samples × QFLS_STEPS)
    solve_time      – pre-allocated per-sample timing (length = n_samples)
"""

import time
start_time = time.monotonic()

import sys
import argparse
import numpy as np
import h5py
from datetime import timedelta, datetime
from pathlib import Path
from scipy.stats import qmc

# ── CLI – allows overriding N_TRAPS and DIM from the command line ─────────────
_parser = argparse.ArgumentParser(
    description="Sobol parameter generator for N-trap TRPL simulations.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
_parser.add_argument("--n-traps", type=int, default=None,
                     help="Number of trap states (overrides N_TRAPS in script)")
_parser.add_argument("--dim",     type=int, default=None,
                     help="Sobol exponent: 2**DIM samples drawn before ordering filter "
                          "(overrides DIM in script)")
# ── shared parameter bounds ───────────────────────────────────────────────────
_parser.add_argument("--n-pulse-min",      type=float, default=None, help="n_pulse lower bound [cm^-3]")
_parser.add_argument("--n-pulse-max",      type=float, default=None, help="n_pulse upper bound [cm^-3]")
_parser.add_argument("--eg-min",           type=float, default=None, help="Eg lower bound [eV]")
_parser.add_argument("--eg-max",           type=float, default=None, help="Eg upper bound [eV]")
_parser.add_argument("--krad-min",         type=float, default=None, help="k_rad lower bound [cm^3/s]")
_parser.add_argument("--krad-max",         type=float, default=None, help="k_rad upper bound [cm^3/s]")
# ── per-trap parameter bounds ─────────────────────────────────────────────────
_parser.add_argument("--ntrap-min",        type=float, default=None, help="Ntrap lower bound [cm^-3]")
_parser.add_argument("--ntrap-max",        type=float, default=None, help="Ntrap upper bound [cm^-3]")
_parser.add_argument("--delta-etrap-min",  type=float, default=None, help="DeltaEtrap lower bound (fraction of Eg)")
_parser.add_argument("--delta-etrap-max",  type=float, default=None, help="DeltaEtrap upper bound (fraction of Eg)")
_parser.add_argument("--taun-min",         type=float, default=None, help="taun lower bound [s]")
_parser.add_argument("--taun-max",         type=float, default=None, help="taun upper bound [s]")
_parser.add_argument("--taup-min",         type=float, default=None, help="taup lower bound [s]")
_parser.add_argument("--taup-max",         type=float, default=None, help="taup upper bound [s]")
_parser.add_argument("--output-dir",       type=str,   default=None,
                     help="Directory to save the HDF5 output "
                          "(default: ./Synthetic-Training-Data)")
_cli_args = _parser.parse_args()

##################################################################
############ Physical constants
Q   = 1.602176462e-19   # [As]  elementary charge
K   = 1.38064852e-23    # [J/K] Boltzmann constant
T   = 300               # [K]   temperature
VT  = K * T / Q        # [V]   thermal voltage (~25.8 mV at 300 K)

############ Effective DOS (cm^-3)
NC = 2.21359e18
NV = 2.21359e18

############ Simulation grid


TMIN       = 1e-11  # time grid for ODE solver [s]
TMAX       = 1e5    # maximum simulation time [s] (must be >> longest trap lifetime to reach equilibrium)
QFLS_STEPS = 256   # number of log-spaced time evaluation points
PL_DECAY_MAGNITUDE = 12
tarr = np.logspace(np.log10(TMIN), np.log10(TMAX), QFLS_STEPS)

##################################################################
############ USER SETTINGS  ######################################
##################################################################

N_TRAPS = 1   # <-- change this to 1, 2, 3, ... as needed

DIM = 14     # Sobol exponent: 2**DIM samples are drawn before filtering
              # Increase DIM if too few samples survive the ordering filter.
              # Rule of thumb: each ordering filter reduces by ~1/N_TRAPS!

# CLI overrides (set by --n-traps / --dim)
if _cli_args.n_traps is not None:
    N_TRAPS = _cli_args.n_traps
if _cli_args.dim is not None:
    DIM = _cli_args.dim

print(f"N_TRAPS (effective): {N_TRAPS}")
print(f"DIM     (effective): {DIM}  →  {2**DIM} Sobol draws")

##################################################################
############ Build parameter bounds
# Shared parameters (always 3):
#   n_pulse : [1e12,  1e18]  cm^-3  (initial photoexcited carrier density)
#   Eg      : [1.15,  2.35]  eV     (bandgap; stored as log10 → actual range
#                                    directly after scaling)
#   k_rad   : [1e-12, 1e-10] cm^3/s (radiative recombination coefficient)
SHARED_L = [1E12,    10**1.15, 1.0E-12]
SHARED_U = [1E18,    10**2.35, 1.0E-10]

# Per-trap parameters (4 per trap):
#   Ntrap      : [1e13, 1e20]  cm^-3  (trap density)
#   DeltaEtrap : [0.5,  0.975] (fraction of Eg; range avoids band edges by ~kT)
#   taun       : [1e-11, 1e-4] s      (electron lifetime → beta_n = 1/(taun*Ntrap))
#   taup       : [1e-11, 1e-4] s      (hole lifetime)
TRAP_L = [1E12,    10**0.5,   1E-12, 1E-12]
TRAP_U = [1E20,    10**0.975, 1E-4,  1E-4 ]

# CLI overrides for bounds
if _cli_args.n_pulse_min     is not None: SHARED_L[0] = _cli_args.n_pulse_min
if _cli_args.n_pulse_max     is not None: SHARED_U[0] = _cli_args.n_pulse_max
if _cli_args.eg_min          is not None: SHARED_L[1] = 10 ** _cli_args.eg_min
if _cli_args.eg_max          is not None: SHARED_U[1] = 10 ** _cli_args.eg_max
if _cli_args.krad_min        is not None: SHARED_L[2] = _cli_args.krad_min
if _cli_args.krad_max        is not None: SHARED_U[2] = _cli_args.krad_max
if _cli_args.ntrap_min       is not None: TRAP_L[0]   = _cli_args.ntrap_min
if _cli_args.ntrap_max       is not None: TRAP_U[0]   = _cli_args.ntrap_max
if _cli_args.delta_etrap_min is not None: TRAP_L[1]   = _cli_args.delta_etrap_min
if _cli_args.delta_etrap_max is not None: TRAP_U[1]   = _cli_args.delta_etrap_max
if _cli_args.taun_min        is not None: TRAP_L[2]   = _cli_args.taun_min
if _cli_args.taun_max        is not None: TRAP_U[2]   = _cli_args.taun_max
if _cli_args.taup_min        is not None: TRAP_L[3]   = _cli_args.taup_min
if _cli_args.taup_max        is not None: TRAP_U[3]   = _cli_args.taup_max

l_bounds = SHARED_L + TRAP_L * N_TRAPS
u_bounds = SHARED_U + TRAP_U * N_TRAPS

n_params = len(l_bounds)   # = 3 + 4 * N_TRAPS

##################################################################
############ Sobol sampling (log10 space)
# Fixed parameters (lb == ub) are excluded from the Sobol sequence and
# inserted back as a constant column after sampling.
lb_arr = np.array(l_bounds, dtype=np.float64)
ub_arr = np.array(u_bounds, dtype=np.float64)
fixed_mask = (lb_arr == ub_arr)
free_idx   = np.where(~fixed_mask)[0]
fixed_idx  = np.where( fixed_mask)[0]

n_free = len(free_idx)
sampler = qmc.Sobol(d=n_free, scramble=False)
sample_free = sampler.random_base2(m=DIM)

# Scale free dimensions
sample_free = qmc.scale(sample_free,
                        np.log10(lb_arr[free_idx]),
                        np.log10(ub_arr[free_idx]))

# Reconstruct full sample matrix
sample = np.empty((sample_free.shape[0], n_params), dtype=np.float64)
sample[:, free_idx]  = sample_free
sample[:, fixed_idx] = np.log10(lb_arr[fixed_idx])   # constant columns

if fixed_idx.size:
    print(f"Fixed parameters : {list(fixed_idx)}  "
          f"(log10 values: {np.log10(lb_arr[fixed_idx]).tolist()})")

##################################################################
############ Ordering filter: Etrap_1 < Etrap_2 < ... < Etrap_N
# DeltaEtrap for trap i sits at column index  3 + i*4 + 1
etrap_cols = [3 + i * 4 + 1 for i in range(N_TRAPS)]

mask = np.ones(len(sample), dtype=bool)
for i in range(N_TRAPS - 1):
    mask &= sample[:, etrap_cols[i]] < sample[:, etrap_cols[i + 1]]

sample = sample[mask]
params = np.array(sample, dtype=np.float64)
n_samples = len(params)
print(f"N_TRAPS          : {N_TRAPS}")
print(f"Parameters/sample: {n_params}  (3 shared + {4*N_TRAPS} trap)")
print(f"Sobol draws      : 2^{DIM} = {2**DIM}")
print(f"After ordering filter: {len(params)}")

##################################################################
############ Constants record array
namesList = ['Nc', 'Nv', 'Vt', 'tmin', 'tmax', 'N_TRAPS', 'QFLS_STEPS', 'PL_DECAY_MAGNITUDE']
ds_dt   = np.dtype({'names': namesList, 'formats': [float] * len(namesList)})
constants = [NC, NV, VT, TMIN, TMAX, float(N_TRAPS), QFLS_STEPS, PL_DECAY_MAGNITUDE]
rec_arr = np.rec.fromarrays(constants, dtype=ds_dt)

##################################################################
############ Save to HDF5
now       = datetime.now()
date_time = now.strftime("%d_%m_%Y")

if _cli_args.output_dir is not None:
    folder = Path(_cli_args.output_dir)
else:
    folder = Path(__file__).parent / 'Synthetic-Training-Data'
folder.mkdir(parents=True, exist_ok=True)

trap_label = f'{N_TRAPS}Trap' if N_TRAPS != 1 else '1Trap'
name_str = folder / (
    f'{trap_label}_TRPL_Simulation_{date_time}_sims_{len(params)}_t_{len(tarr)}.hdf5'
)

simulation_length = len(params)

with h5py.File(name_str, 'w') as hf:

    dt = h5py.vlen_dtype(np.float64)  # variable-length float arrays

    hf.create_dataset('lb',         data=np.log10(l_bounds))
    hf.create_dataset('ub',         data=np.log10(u_bounds))
    hf.create_dataset('par_mat_ln', data=params)
    hf.create_dataset('constants',  data=rec_arr)
    hf.create_dataset('tarr',       data=tarr)       # log-spaced time grid [s]

    hf.create_dataset('pl',
        shape=(n_samples,),
        dtype=dt
    )
    hf.create_dataset('time',
        shape=(n_samples,),
        dtype=dt
    )
    hf.create_dataset('tau_diff',
        shape=(n_samples,),
        dtype=dt
    )
    hf.create_dataset('qfls',
        shape=(n_samples,),
        dtype=dt
    )
    hf.create_dataset('solve_time',
        data=np.zeros(n_samples),   # one float per sample
        dtype=np.float64
    )
end_time = time.monotonic()
print(timedelta(seconds=end_time - start_time))
print(name_str)

##################################################################
############ Save creation hyperparameters
import json

param_names_list = (
    ["n_pulse", "Eg", "k_rad"] +
    [f"{p}_{i+1}" for i in range(N_TRAPS) for p in ["Ntrap", "DeltaEtrap", "taun", "taup"]]
)

creation_params = {
    "timestamp":                now.isoformat(timespec="seconds"),
    "hdf5_file":                str(name_str),
    "n_traps":                  N_TRAPS,
    "dim":                      DIM,
    "sobol_draws":              2 ** DIM,
    "n_samples_after_filter":   n_samples,
    "qfls_steps":               QFLS_STEPS,
    "tmin_s":                   TMIN,
    "tmax_s":                   TMAX,
    "pl_decay_magnitude":       PL_DECAY_MAGNITUDE,
    "parameter_names":          param_names_list,
    "bounds_physical": {
        name: {"lower": float(lb_arr[i]), "upper": float(ub_arr[i])}
        for i, name in enumerate(param_names_list)
    },
    "cli_args": {k: v for k, v in vars(_cli_args).items() if v is not None},
}

creation_params_path = name_str.with_suffix("").with_name(name_str.stem + "_creation_params.json")
with open(creation_params_path, "w") as _f:
    json.dump(creation_params, _f, indent=2)
print(f"Creation params saved: {creation_params_path}")
