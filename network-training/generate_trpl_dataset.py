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
    python generate_trpl_dataset.py <path_to_hdf5_file> [n_jobs]

Simulations run in parallel across n_jobs worker processes (joblib/loky).
n_jobs defaults to $SLURM_CPUS_PER_TASK (matching --cpus-per-task in
submit_training.sh) when running under SLURM, else all visible CPU cores.

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

import os
import sys
import time
import numpy as np
import h5py
from pathlib import Path
from scipy.interpolate import PchipInterpolator
import pandas as pd
from numba import njit
from joblib import Parallel, delayed

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
# Per-sample worker (runs in a joblib subprocess)
# ─────────────────────────────────────────────────────────────────────────────
def _simulate_one(par, N_TRAPS, tmin, tmax):
    """
    Decode one Sobol row and run solve_transient for it.

    Kept as a top-level function (not nested in main()) so it can be pickled
    and shipped to joblib's worker processes. Does no HDF5 I/O itself — the
    HDF5 file handle is not picklable/fork-safe for concurrent writers, so
    every worker just returns arrays and the parent process (main()) is the
    single writer.

    Returns
    -------
    ok : bool
        True if the simulation succeeded.
    result : dict or None
        {'time_log', 'pl_log', 'qfls', 'tau_log'} on success, else None.
    err : str or None
        Exception message on failure, else None.
    dt : float
        Wall time spent decoding + integrating this sample [s].
    """
    t_start = time.monotonic()
    try:
        npulse, Eg, krad, Nt, Et, taun, taup = decode_params(par, N_TRAPS)
        (t_sim, n_tr, p_tr, nt_tr,
         QFLS_sim, tau_diff, _tau_n, _tau_p,
         PL_sim, *_) = solve_transient(
            [tmin, tmax], npulse, Eg, NC, NV, krad, Nt, Et, taun, taup)

        result = dict(
            # time in ns, shifted by tmin (= 1e-11 s): the solver starts at t = 0 (16/10/2026), so t + tmin is the
            # time axis of the earlier start at tmin (first point exactly tmin, no log10(0))
            time_log=np.log10((t_sim + tmin) * 1e9),
            pl_log=np.log10(PL_sim),
            qfls=QFLS_sim,
            tau_log=np.log10(tau_diff),
        )
        return True, result, None, time.monotonic() - t_start
    except Exception as exc:
        return False, None, str(exc), time.monotonic() - t_start


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
CHUNK_SIZE = 2000   # samples submitted to the worker pool per parallel batch


def main(hdf5_path: Path, n_jobs: int = None):
    t0_wall = time.monotonic()

    if n_jobs is None:
        # SLURM_CPUS_PER_TASK matches --cpus-per-task in submit_training.sh;
        # fall back to all visible cores when run outside SLURM.
        n_jobs = int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count() or 1))

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
        print(f"HDF5      : {hdf5_path.resolve()}")
        print(f"Workers   : {n_jobs} (joblib/loky)\n")

        # ── resume support ────────────────────────────────────────────────
        solve_arr     = solve_ds[:]
        already_done  = int(np.sum(solve_arr > 0))
        todo          = np.flatnonzero(solve_arr <= 0)   # indices still to run
        if already_done:
            print(f"Resuming: {already_done} done, "
                  f"{n_samples - already_done} remaining.\n")

        completed  = already_done
        failed     = 0
        n_this_run = 0             # samples processed in THIS run
        cum_time = 0
        cum_sim_times_per_sample = []   # per-sample simulation time [s]
        cum_n_curves = []               # cumulative count of simulated curves

        # ── parallel simulation loop ────────────────────────────────────
        # Workers only compute (decode_params + solve_transient); the HDF5
        # file itself is written exclusively here in the parent process,
        # since an h5py File is not fork/pickle-safe for concurrent writers.
        with Parallel(n_jobs=n_jobs, backend='loky') as parallel:
            for start in range(0, len(todo), CHUNK_SIZE):
                chunk = todo[start:start + CHUNK_SIZE]

                results = parallel(
                    delayed(_simulate_one)(par_mat[i], N_TRAPS, tmin, tmax)
                    for i in chunk
                )

                for i, (ok, result, err, dt) in zip(chunk, results):
                    wall_now = time.monotonic() - t0_wall

                    if ok:
                        hf['time'][i]     = result['time_log']
                        hf['pl'][i]       = result['pl_log']
                        hf['qfls'][i]     = result['qfls']
                        hf['tau_diff'][i] = result['tau_log']
                        completed += 1
                    else:
                        print(f"  [FAIL] sample {i:6d}: {err}")
                        failed += 1

                    solve_ds[i] = dt
                    ts_ds[i]    = wall_now    # seconds since this run started

                    n_this_run += 1
                    cum_time += dt
                    cum_sim_times_per_sample.append(dt)
                    cum_n_curves.append(n_this_run)

                    # ── progress report ──────────────────────────────────
                    if n_this_run % 500 == 0 or n_this_run == 1:
                        rate = n_this_run / wall_now if wall_now > 0 else float('nan')
                        remaining = n_samples - completed - failed
                        eta  = remaining / rate if rate > 0 else float('nan')
                        print(f"  [{completed + failed:6d}/{n_samples}]  "
                              f"rate={rate:5.1f} sim/s  "
                              f"elapsed={wall_now:7.0f} s  "
                              f"ETA={eta:7.0f} s  "
                              f"failed={failed}")

                    # ── periodic flush every 1000 samples ─────────────────
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
    if len(sys.argv) not in (2, 3):
        print(__doc__)
        sys.exit(1)
    n_jobs_arg = int(sys.argv[2]) if len(sys.argv) == 3 else None
    main(Path(sys.argv[1]), n_jobs=n_jobs_arg)
    #main(Path("Synthetic-Training-Data/1Trap_TRPL_Simulation_10_03_2026_sims_32_t_256.hdf5"))