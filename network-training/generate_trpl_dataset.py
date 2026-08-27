"""
25/02/2026  Robin Heumann

TRPL dataset generator.

Reads a Sobol-parameter HDF5 file produced by
sobol_generator_NTrap_Fluence-Thickness-Bandgap.py, runs solve_transient for
every parameter set, interpolates all outputs onto the stored reference time
grid (Y1_tarr), and writes PL, QFLS, tau_diff, solve_time and
completion_timestamps back into the same file.

Usage
-----
    python generate_trpl_dataset.py <path_to_hdf5_file>

The script can be re-started after an interruption: samples whose
solve_time > 0 are skipped automatically (resume support).

Parameter decoding convention (sobol_generator stores everything in "log10
space", but Eg and DeltaEtrap are stored directly, not as log10):
  • n_pulse, krad, Ntrap, taun, taup  →  actual = 10 ** par_mat_ln value
  • Eg, DeltaEtrap                    →  actual = par_mat_ln value  (the
    bounds were written as 10^(value) so log10(bound) == the physical value)

"Simulations completed vs. wall time" plot
------------------------------------------
After the run, load completion_timestamps from the HDF5 file:

    import h5py, numpy as np, matplotlib.pyplot as plt
    with h5py.File('path/to/file.hdf5', 'r') as hf:
        ts = hf['completion_timestamps'][:]
    valid = np.isfinite(ts) & (ts > 0)
    plt.plot(np.sort(ts[valid]), np.arange(1, valid.sum() + 1))
    plt.xlabel('Wall time  [s]')
    plt.ylabel('Simulations completed')
    plt.show()
"""

import sys
import time
import numpy as np
import h5py
from pathlib import Path
from scipy.interpolate import PchipInterpolator
import pandas as pd
from numba import njit

# ── make sspl_module importable (required by trpl_module) ────────────────────

from trpl_module import solve_transient  # noqa: E402  (path patched above)

# ─────────────────────────────────────────────────────────────────────────────
# Physical constants  (must match sobol_generator)
# ─────────────────────────────────────────────────────────────────────────────
NC = 2.21359e18   # effective DOS, conduction band  [cm^-3]
NV = 2.21359e18   # effective DOS, valence band      [cm^-3]


# ─────────────────────────────────────────────────────────────────────────────
# Parameter decoding
# ─────────────────────────────────────────────────────────────────────────────
@njit
def decode_params(par, N_TRAPS):
    """
    Convert one log10-space Sobol row to physical simulation parameters.

    Eg and DeltaEtrap are stored as direct values (bounds were written as
    10^(value) so that log10(bound) == the physical value).  All other
    parameters are genuine log10 values and need 10^par.
    """
    npulse = 10 ** par[0]
    Eg     = par[1]                # eV  — direct value, NOT 10^par
    krad   = 10 ** par[2]          # cm³/s

    Nt_arr   = np.empty(N_TRAPS)
    Et_arr   = np.empty(N_TRAPS)   # eV relative to VB edge
    taun_arr = np.empty(N_TRAPS)
    taup_arr = np.empty(N_TRAPS)

    for i in range(N_TRAPS):
        base        = 3 + i * 4
        Nt_arr[i]   = 10 ** par[base]
        delta_E     = par[base + 1]        # fraction of Eg — direct value
        Et_arr[i]   = delta_E * Eg         # eV relative to VB
        taun_arr[i] = 10 ** par[base + 2]
        taup_arr[i] = 10 ** par[base + 3]

    return npulse, Eg, krad, Nt_arr, Et_arr, taun_arr, taup_arr


# ─────────────────────────────────────────────────────────────────────────────
# Interpolation helper
# ─────────────────────────────────────────────────────────────────────────────
def to_grid(t_sim, data, tarr):
    """
    Linearly interpolate *data* (defined on adaptive t_sim) onto the fixed
    reference *tarr*.  Out-of-range points are filled with the nearest
    boundary value.  Returns a NaN array when t_sim has fewer than 2 points.
    """
    if len(t_sim) < 2:
        return np.full(len(tarr), np.nan)
    f = PchipInterpolator(t_sim, data, extrapolate=False)
    return f(tarr)


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────
def main(hdf5_path: Path):
    t0_wall = time.monotonic()

    with h5py.File(hdf5_path, 'r+') as hf:
        par_mat    = hf['par_mat_ln'][:]       # (n_samples, n_params)
        tmin       = hf['constants']['tmin'][()]         # reference time grid [s]
        tmax       = hf['constants']['tmax'][()]  
        n_samples, n_params = par_mat.shape
        N_TRAPS    = int(hf['constants']['N_TRAPS'][()] ) 
        QFLS_STEPS = hf['constants']['QFLS_STEPS'][()]  

        # ── create completion_timestamps dataset if absent ────────────────
        if 'completion_timestamps' not in hf:
            hf.create_dataset('completion_timestamps',
                              data=np.full(n_samples, np.nan),
                              maxshape=(None,))

        solve_ds = hf['solve_time']
        ts_ds    = hf['completion_timestamps']

        print(f"N_TRAPS   : {N_TRAPS}")
        print(f"N samples : {n_samples}")
        print(f"Grid pts  : {QFLS_STEPS}")
        print(f"HDF5      : {hdf5_path.resolve()}\n")

        # ── resume support ────────────────────────────────────────────────
        already_done = int(np.sum(solve_ds[:] > 0))
        if already_done:
            print(f"Resuming: {already_done} done, "
                  f"{n_samples - already_done} remaining.\n")

        completed  = already_done
        failed     = 0
        n_this_run = 0             # samples processed in THIS run
        cum_time = 0
        cum_sim_times_per_sample = []   # per-sample simulation time [s]
        cum_n_curves = []               # cumulative count of simulated curves

        for i in range(n_samples):
            if solve_ds[i] > 0:    # already finished → skip
                continue

            par     = par_mat[i]
            npulse, Eg, krad, Nt, Et, taun, taup = decode_params(par, N_TRAPS)
            t_start = time.monotonic()

            try:
                (t_sim, n_tr, p_tr, nt_tr,
                 QFLS_sim, tau_diff, _tau_n, _tau_p,
                 PL_sim, *_) = solve_transient(
                    [tmin,tmax], npulse, Eg, NC, NV, krad, Nt, Et, taun, taup)

                # normalise PL to peak = 1 and interpolate onto tarr
                PL_norm = PL_sim
                hf['time'][i] = np.log10(t_sim*1e9) #time in ns
                hf['pl'][i]       = np.log10(PL_norm)
                hf['qfls'][i]     = QFLS_sim
                hf['tau_diff'][i] = np.log10(tau_diff)

                completed  += 1

            except Exception as exc:
                print(f"  [FAIL] sample {i:6d}: {exc}")
                failed += 1

            dt       = time.monotonic() - t_start
            wall_now = time.monotonic() - t0_wall

            solve_ds[i] = dt
            ts_ds[i]    = wall_now    # seconds since this run started

            n_this_run += 1
            cum_time += dt
            cum_sim_times_per_sample.append(dt)
            cum_n_curves.append(n_this_run)


            # ── progress report ──────────────────────────────────────────
            if n_this_run % 500 == 0 or n_this_run == 1:
                rate = n_this_run / wall_now if wall_now > 0 else float('nan')
                remaining = n_samples - completed - failed
                eta  = remaining / rate if rate > 0 else float('nan')
                print(f"  [{completed + failed:6d}/{n_samples}]  "
                      f"rate={rate:5.1f} sim/s  "
                      f"elapsed={wall_now:7.0f} s  "
                      f"ETA={eta:7.0f} s  "
                      f"failed={failed}")

            # ── periodic flush every 1000 samples ────────────────────────
            if n_this_run % 1000 == 0:
                hf.flush()
        
        cumulative_times = np.cumsum(cum_sim_times_per_sample)
        data = pd.DataFrame({
            'n_curves': cum_n_curves,
            'cumulative_sim_time_s': cumulative_times,
        })
        csv_path = hdf5_path.parent / (hdf5_path.stem + '_simulation_times.csv')
        data.to_csv(csv_path, index=False)
        print(f"Simulation-times CSV saved: {csv_path}")

    total_time = time.monotonic() - t0_wall
    print(f"\nDone.  completed={completed}  failed={failed}  "
          f"wall time={total_time:.1f} s")
    print("\nTo plot simulations vs. wall time:")
    print("  ts = hf['completion_timestamps'][:]")
    print("  plt.plot(np.sort(ts[np.isfinite(ts)]), "
          "np.arange(1, np.isfinite(ts).sum()+1))")
    



if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    main(Path(sys.argv[1]))
    #main(Path("Synthetic-Training-Data/1Trap_TRPL_Simulation_10_03_2026_sims_32_t_256.hdf5"))