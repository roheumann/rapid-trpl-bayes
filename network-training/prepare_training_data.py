"""
10/03/2026  Robin Heumann

Preprocess a TRPL simulation HDF5 file into a training-ready HDF5 dataset.

Reads the raw simulation output produced by generate_trpl_dataset.py, filters
invalid samples, splits into train / val / test sets, fits scalers on the
training split only, and writes everything (raw + scaled) to a new HDF5 file.
Two joblib files are saved alongside the HDF5:

    <stem>_param_scaler.joblib   – sklearn StandardScaler for the 11 input
                                   parameters (log10-space par_mat_ln columns)
    <stem>_output_scaler.joblib  – sklearn MinMaxScaler  for the output
                                   (log10 tau_diff time series).  Fitted on a
                                   single global [min, max] so that all QFLS_STEPS
                                   points share the same scale, preserving
                                   the shape of the curves.

Usage
-----
    python prepare_training_data.py <simulation_hdf5> [<output_hdf5>]

If <output_hdf5> is omitted the output is written next to the source file with
the suffix '_training.hdf5'.

HDF5 output layout
------------------
Metadata:
    source_file          – path of the source simulation HDF5
    N_TRAPS              – number of trap states
    lb, ub               – log10 parameter bounds  (n_params,)
    constants            – physical constants record array
    param_names          – ASCII strings for each parameter column  (S64)
    param_scaler_path    – path to the StandardScaler joblib
    output_scaler_path   – path to the MinMaxScaler joblib

Raw (unscaled) splits – log10-space parameters and log10(tau_diff):
    X_train_raw, X_val_raw, X_test_raw   – (n, n_params)     float32
    y_train_raw, y_val_raw, y_test_raw   – (n, QFLS_STEPS)   float32  [log10 tau_diff]

Interpolated curves (all on the per-sample QFLS grid):
    time_train, time_val, time_test      – (n, QFLS_STEPS)   float32  [log10(t / ns), t shifted by tmin]
    pl_train,   pl_val,   pl_test        – (n, QFLS_STEPS)   float32  [log10(PL / PL_max)]
    qfls_train, qfls_val, qfls_test      – (n, QFLS_STEPS)   float32  [eV]

Scaled splits – ready for model training:
    X_train, X_val, X_test               – StandardScaler(X_raw)
    y_train, y_val, y_test               – MinMaxScaler(y_raw)  → [0, 1]
"""

import sys
import json
import numpy as np
import h5py
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.interpolate import PchipInterpolator


# ─────────────────────────────────────────────────────────────────────────────
# Parameter name helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_param_names(N_TRAPS: int) -> list[str]:
    """Return short ASCII parameter names matching the par_mat_ln column order."""
    names = ["log_n_pulse", "Eg_eV", "log_krad"]
    for t in range(1, N_TRAPS + 1):
        names += [
            f"log_Ntrap_{t}",
            f"DeltaEtrap_{t}",
            f"log_taun_{t}",
            f"log_taup_{t}",
        ]
    return names


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(sim_path: Path, out_path: Path) -> None:
    print(f"Source : {sim_path.resolve()}")
    print(f"Output : {out_path.resolve()}")

    # ── Load simulation HDF5 ─────────────────────────────────────────────────
    with h5py.File(sim_path, "r") as hf:
        par_mat    = hf["par_mat_ln"][:]   # (n_samples, n_params), log10 space
        time       = hf["time"][:]
        pl         = hf["pl"][:]
        qfls       = hf["qfls"][:]
        tau_diff   = hf["tau_diff"][:]     # (n_samples, n_time), log10(tau_diff)
        lb         = hf["lb"][:]
        ub         = hf["ub"][:]
        constants  = hf["constants"][()]
        solve_time = hf["solve_time"][:]

    n_samples, n_params = par_mat.shape
    N_TRAPS     = (n_params - 3) // 4
    param_names = make_param_names(N_TRAPS)
    QFLS_STEPS          = int(constants["QFLS_STEPS"][()])
    PL_DECAY_MAGNITUDE  = float(constants["PL_DECAY_MAGNITUDE"][()])
    print(f"\nN_TRAPS     : {N_TRAPS}")
    print(f"n_samples   : {n_samples}")
    print(f"n_params    : {n_params}")
    print(f"QFLS_STEPS  : {QFLS_STEPS}")

    # ── Filter, cap, and interpolate ─────────────────────────────────────────
    par_list  = []
    tau_list  = []
    time_list = []
    pl_list   = []
    qfls_list = []

    removed        = 0
    rm_solve_time  = 0
    rm_too_invalid = 0
    rm_too_short   = 0
    rm_dedup       = 0

    for i in range(n_samples):

        time_i = time[i]
        tau_i  = tau_diff[i]
        qfls_i = qfls[i]
        pl_i   = pl[i]
        par_i  = par_mat[i]

        # basic parameter check
        if not np.all(np.isfinite(par_i)) or solve_time[i] <= 0:
            removed += 1; rm_solve_time += 1
            continue

        # remove samples whose raw solver output is empty (e.g. an ODE run
        # that event-stopped before/at its first step) before doing anything
        # else with them — np.isfinite/np.max on a zero-size array either
        # trivially pass the "too invalid" ratio check below (0 invalid out
        # of 0 points) or crash outright.
        if tau_i.size == 0:
            removed += 1; rm_too_invalid += 1
            continue

        # count invalid tau values
        finite_mask = np.isfinite(tau_i)
        n_invalid   = (~finite_mask).sum()

        # remove whole sample if more than half invalid — compare against
        # this sample's own raw length, not the QFLS_STEPS target grid size
        # (the raw, pre-interpolation length varies per sample).
        if n_invalid > len(tau_i) / 2:
            removed += 1; rm_too_invalid += 1
            continue

        # remove only invalid points
        time_i = time_i[finite_mask]
        pl_i   = pl_i[finite_mask]
        qfls_i = qfls_i[finite_mask]
        tau_i  = tau_i[finite_mask]

        # ------------------------------------------------------------------
        # normalize PL to peak = 1 (in log10 space: shift so max = 0),
        # then cap at PL_DECAY_MAGNITUDE orders of magnitude below peak
        # ------------------------------------------------------------------
        pl_i    = pl_i - np.max(pl_i)          # log10(PL/PL_max): 0 at t=0, negative later
        cutoff  = -PL_DECAY_MAGNITUDE           # e.g. -12

        idx_cap = np.where(pl_i < cutoff)[0]
        if len(idx_cap) > 0:
            cap    = idx_cap[0]
            time_i = time_i[:cap]
            pl_i   = pl_i[:cap]
            tau_i  = tau_i[:cap]
            qfls_i = qfls_i[:cap]

        # need at least 2 points for interpolation
        if len(qfls_i) < 2:
            removed += 1; rm_too_short += 1
            continue

        # ------------------------------------------------------------------
        # interpolate tau_diff, time, pl onto a uniform QFLS grid
        # qfls decreases with time → reverse so x is increasing, then
        # deduplicate to guarantee strictly increasing x for PchipInterpolator
        # ------------------------------------------------------------------
        qfls_rev = qfls_i[::-1]
        tau_rev  = tau_i[::-1]
        time_rev = time_i[::-1]
        pl_rev   = pl_i[::-1]

        _, unique_idx = np.unique(qfls_rev, return_index=True)
        qfls_rev = qfls_rev[unique_idx]
        tau_rev  = tau_rev[unique_idx]
        time_rev = time_rev[unique_idx]
        pl_rev   = pl_rev[unique_idx]

        if len(qfls_rev) < 2:
            removed += 1; rm_dedup += 1
            continue

        qfls_grid  = np.linspace(qfls_rev.min(), qfls_rev.max(), QFLS_STEPS)
        tau_inter  = PchipInterpolator(qfls_rev, tau_rev,  extrapolate=None)(qfls_grid)
        time_inter = PchipInterpolator(qfls_rev, time_rev, extrapolate=None)(qfls_grid)
        pl_inter   = PchipInterpolator(qfls_rev, pl_rev,   extrapolate=None)(qfls_grid)

        par_list.append(par_i)
        tau_list.append(tau_inter)
        time_list.append(time_inter)
        pl_list.append(pl_inter)
        qfls_list.append(qfls_grid)

    # ── Convert to arrays ────────────────────────────────────────────────────
    par_mat  = np.array(par_list,  dtype=np.float32)   # (n_valid, n_params)
    tau_diff = np.array(tau_list,  dtype=np.float32)   # (n_valid, QFLS_STEPS)
    time_arr = np.array(time_list, dtype=np.float32)   # (n_valid, QFLS_STEPS)
    pl_arr   = np.array(pl_list,   dtype=np.float32)   # (n_valid, QFLS_STEPS)
    qfls_arr = np.array(qfls_list, dtype=np.float32)   # (n_valid, QFLS_STEPS)

    n_valid = len(par_mat)
    print(f"\nFiltered out: {removed}/{n_samples} "
          f"({100 * removed / n_samples:.2f} %)")
    print(f"  - solve_time <= 0 or non-finite params : {rm_solve_time}")
    print(f"  - >50% invalid tau values              : {rm_too_invalid}")
    print(f"  - <2 points after PL cap               : {rm_too_short}")
    print(f"  - <2 unique QFLS points after dedup    : {rm_dedup}")
    print(f"Valid samples: {n_valid}")

    if n_valid == 0:
        print("\nERROR: No valid samples found. Did you run generate_trpl_dataset.py on this file first?")
        sys.exit(1)

    # ── Train / Val / Test split  (80 / 10 / 10) ────────────────────────────
    indices = np.arange(n_valid)
    idx_tr, idx_te = train_test_split(indices, test_size=0.10, shuffle=True, random_state=42)
    idx_tr, idx_va = train_test_split(idx_tr,  test_size=0.10 / 0.90, shuffle=True, random_state=42)

    def split(arr): return arr[idx_tr], arr[idx_va], arr[idx_te]

    X_tr, X_va, X_te          = split(par_mat)
    y_tr, y_va, y_te          = split(tau_diff)
    time_tr, time_va, time_te = split(time_arr)
    pl_tr,   pl_va,   pl_te   = split(pl_arr)
    qfls_tr, qfls_va, qfls_te = split(qfls_arr)

    print(f"\nTrain : {len(X_tr)}")
    print(f"Val   : {len(X_va)}")
    print(f"Test  : {len(X_te)}")

    # ── Fit input scaler (StandardScaler) on training split only ─────────────
    param_scaler = StandardScaler()
    X_tr_sc = param_scaler.fit_transform(X_tr).astype(np.float32)
    X_va_sc = param_scaler.transform(X_va).astype(np.float32)
    X_te_sc = param_scaler.transform(X_te).astype(np.float32)

    # ── Fit output scaler (MinMaxScaler) on training split only ──────────────
    # Fit a global [min, max] across all samples AND all QFLS points so that
    # the relative shape of each curve is preserved after scaling.
    y_tr_global_min = float(y_tr.min())
    y_tr_global_max = float(y_tr.max())
    n_time          = y_tr.shape[1]

    output_scaler = MinMaxScaler(feature_range=(0, 1))
    dummy = np.array(
        [[y_tr_global_min] * n_time,
         [y_tr_global_max] * n_time],
        dtype=np.float64,
    )
    output_scaler.fit(dummy)

    y_tr_sc = output_scaler.transform(y_tr).astype(np.float32)
    y_va_sc = output_scaler.transform(y_va).astype(np.float32)
    y_te_sc = output_scaler.transform(y_te).astype(np.float32)

    print(f"\ny range (log10 τ): [{y_tr_global_min:.3f}, {y_tr_global_max:.3f}]")

    # ── Save scalers ─────────────────────────────────────────────────────────
    param_scaler_path  = out_path.parent / (out_path.stem + "_param_scaler.joblib")
    output_scaler_path = out_path.parent / (out_path.stem + "_output_scaler.joblib")

    joblib.dump(param_scaler,  param_scaler_path)
    joblib.dump(output_scaler, output_scaler_path)
    print(f"Param scaler saved : {param_scaler_path}")
    print(f"Output scaler saved: {output_scaler_path}")

    # ── Write training HDF5 ──────────────────────────────────────────────────
    with h5py.File(out_path, "w") as hf:
        # Metadata
        hf.create_dataset("source_file",         data=str(sim_path.resolve()).encode())
        hf.create_dataset("N_TRAPS",             data=np.array(N_TRAPS))
        hf.create_dataset("lb",                  data=lb)
        hf.create_dataset("ub",                  data=ub)
        hf.create_dataset("constants",           data=constants)
        hf.create_dataset("param_names",         data=np.array(param_names, dtype="S64"))
        hf.create_dataset("param_scaler_path",   data=str(param_scaler_path.resolve()).encode())
        hf.create_dataset("output_scaler_path",  data=str(output_scaler_path.resolve()).encode())

        # Raw (unscaled) parameter splits
        hf.create_dataset("X_train_raw", data=X_tr)
        hf.create_dataset("X_val_raw",   data=X_va)
        hf.create_dataset("X_test_raw",  data=X_te)

        # Interpolated tau_diff (log10) splits – unscaled and scaled
        hf.create_dataset("tau_diff_train", data=y_tr)
        hf.create_dataset("tau_diff_val",   data=y_va)
        hf.create_dataset("tau_diff_test",  data=y_te)
        hf.create_dataset("y_train_raw",    data=y_tr)
        hf.create_dataset("y_val_raw",      data=y_va)
        hf.create_dataset("y_test_raw",     data=y_te)
        hf.create_dataset("y_train",        data=y_tr_sc)
        hf.create_dataset("y_val",          data=y_va_sc)
        hf.create_dataset("y_test",         data=y_te_sc)

        # Scaled parameter splits
        hf.create_dataset("X_train",     data=X_tr_sc)
        hf.create_dataset("X_val",       data=X_va_sc)
        hf.create_dataset("X_test",      data=X_te_sc)

        # Interpolated time, PL, QFLS curves (not scaled – used for analysis/plotting)
        hf.create_dataset("time_train",  data=time_tr)
        hf.create_dataset("time_val",    data=time_va)
        hf.create_dataset("time_test",   data=time_te)
        hf.create_dataset("pl_train",    data=pl_tr)
        hf.create_dataset("pl_val",      data=pl_va)
        hf.create_dataset("pl_test",     data=pl_te)
        hf.create_dataset("qfls_train",  data=qfls_tr)
        hf.create_dataset("qfls_val",    data=qfls_va)
        hf.create_dataset("qfls_test",   data=qfls_te)

    print(f"\nTraining HDF5 saved: {out_path}")
    print("Done.")


# ─────────────────────────────────────────────────────────────────────────────
# PL-decade axis (networks from 10/2026)
# ─────────────────────────────────────────────────────────────────────────────
#
# Every curve is put on ONE fixed axis u = log10(PL / PL(t=0)) from -DECADES to 0.
# The axis does not depend on the parameters (n_pulse, p0, Eg), so the network
# output, the measured data and the likelihood share it without any QFLS
# calibration; absolute QFLS = QFLS(0) + kT ln10 * u is stored for plots only.
# Grid points that a curve does not cover, or that lie in a stretch where the
# solver returned no valid tau (points removed there), are NaN = masked in the
# loss.  Nothing is filled or extrapolated.

def curve_on_u_grid(u, tau_log, extra, u_grid, decades):
    """
    One simulated curve on the fixed u grid.

    u       : log10(PL/PL0) at the solver output points (decreasing in time)
    tau_log : log10 tau_diff at the same points (non-finite = invalid)
    extra   : dict name -> array at the same points (time, qfls), interpolated alike
    Returns (y, extra_on_grid, status); status 'ok' | 'short' | 'few'.
    """
    if not np.isfinite(u).any() or np.nanmin(u) > -decades:
        return None, None, "short"
    # strictly decreasing u along the curve (points where PL rises are removed
    # like invalid tau), valid tau only
    valid = np.isfinite(u) & np.isfinite(tau_log)
    u_valid = np.where(valid, u, np.inf)
    prev_min = np.concatenate([[np.inf], np.minimum.accumulate(u_valid)[:-1]])
    keep = np.flatnonzero(valid & (u < prev_min))   # valid points that set a new minimum of u
    if keep.size < 4:
        return None, None, "few"
    uk = u[keep][::-1]                          # increasing
    yk = tau_log[keep][::-1]
    y = np.full(len(u_grid), np.nan)
    inside = (u_grid >= uk[0]) & (u_grid <= uk[-1])
    y[inside] = PchipInterpolator(uk, yk)(u_grid[inside])
    # mask grid points inside stretches where solver points were removed
    removed = np.diff(keep) > 1                 # between keep[j] and keep[j+1] (time order)
    for j in np.flatnonzero(removed):
        hi, lo = u[keep[j]], u[keep[j + 1]]     # u decreases in time
        y[(u_grid > lo) & (u_grid < hi)] = np.nan
    out = {}
    for name, arr in extra.items():
        a = np.full(len(u_grid), np.nan)
        a[inside] = PchipInterpolator(uk, arr[keep][::-1])(u_grid[inside])
        a[~np.isfinite(y)] = np.nan
        out[name] = a
    if np.isfinite(y).sum() < 16:
        return None, None, "few"
    return y, out, "ok"


def main_pl_decades(sim_path: Path, out_path: Path, n_grid: int = 511) -> None:
    print(f"Source : {sim_path.resolve()}")
    print(f"Output : {out_path.resolve()}  (axis: PL decades, {n_grid} grid points)")
    with h5py.File(sim_path, "r") as hf:
        par_mat    = hf["par_mat_ln"][:]
        time       = hf["time"][:]
        time_units = hf["time"].attrs.get("units", "log10(ns)")
        pl         = hf["pl"][:]
        qfls       = hf["qfls"][:]
        tau_diff   = hf["tau_diff"][:]
        lb, ub     = hf["lb"][:], hf["ub"][:]
        constants  = hf["constants"][()]
        solve_time = hf["solve_time"][:]
    if isinstance(time_units, bytes):
        time_units = time_units.decode()
    n_samples, n_params = par_mat.shape
    N_TRAPS = (n_params - 3) // 4
    param_names = make_param_names(N_TRAPS)
    decades = float(constants["PL_DECAY_MAGNITUDE"][()])
    u_grid = np.linspace(-decades, 0.0, n_grid)          # increasing, like QFLS
    print(f"N_TRAPS {N_TRAPS} | n_samples {n_samples} | time units: {time_units}")

    keep_idx, Y, T, Q, umin, q0 = [], [], [], [], [], []
    counts = {"solve_time<=0": 0, "empty": 0, "short": 0, "few": 0}
    for i in range(n_samples):
        if not np.all(np.isfinite(par_mat[i])) or solve_time[i] <= 0:
            counts["solve_time<=0"] += 1
            continue
        if tau_diff[i].size < 2:
            counts["empty"] += 1
            continue
        pl_i = pl[i]
        u = pl_i - pl_i[0]                               # PL0 = PL at t = 0 (first output point)
        t_i = time[i] if time_units.startswith("ns") else 10.0 ** time[i]   # linear ns
        y, ex, status = curve_on_u_grid(u, tau_diff[i], {"time": t_i, "qfls": qfls[i]}, u_grid, decades)
        if status != "ok":
            counts[status] += 1
            continue
        keep_idx.append(i); Y.append(y); T.append(ex["time"]); Q.append(ex["qfls"])
        umin.append(np.nanmin(u)); q0.append(qfls[i][0])
        if (i + 1) % 50000 == 0:
            print(f"  {i + 1}/{n_samples} processed", flush=True)

    par = par_mat[keep_idx].astype(np.float32)
    Y = np.asarray(Y, np.float32); T = np.asarray(T, np.float32); Q = np.asarray(Q, np.float32)
    umin = np.asarray(umin, np.float32); q0 = np.asarray(q0, np.float32)
    n_valid = len(par)
    print(f"\nRemoved {n_samples - n_valid}/{n_samples}: {counts}")
    print(f"Valid samples: {n_valid} | masked grid points: {100 * np.mean(~np.isfinite(Y)):.3f} %")

    indices = np.arange(n_valid)
    idx_tr, idx_te = train_test_split(indices, test_size=0.10, shuffle=True, random_state=42)
    idx_tr, idx_va = train_test_split(idx_tr, test_size=0.10 / 0.90, shuffle=True, random_state=42)

    param_scaler = StandardScaler().fit(par[idx_tr])
    y_min, y_max = float(np.nanmin(Y[idx_tr])), float(np.nanmax(Y[idx_tr]))
    output_scaler = MinMaxScaler(feature_range=(0, 1)).fit(
        np.array([[y_min] * n_grid, [y_max] * n_grid], dtype=np.float64))
    print(f"y range (log10 tau): [{y_min:.3f}, {y_max:.3f}]")

    param_scaler_path  = out_path.parent / (out_path.stem + "_param_scaler.joblib")
    output_scaler_path = out_path.parent / (out_path.stem + "_output_scaler.joblib")
    joblib.dump(param_scaler, param_scaler_path)
    joblib.dump(output_scaler, output_scaler_path)

    with h5py.File(out_path, "w") as hf:
        hf.create_dataset("source_file",        data=str(sim_path.resolve()).encode())
        hf.create_dataset("axis",               data=b"pl_decades")
        hf.create_dataset("u_grid",             data=u_grid)
        hf.create_dataset("decades",            data=decades)
        hf.create_dataset("y_min",              data=y_min)
        hf.create_dataset("y_max",              data=y_max)
        hf.create_dataset("N_TRAPS",            data=np.array(N_TRAPS))
        hf.create_dataset("lb",                 data=lb)
        hf.create_dataset("ub",                 data=ub)
        hf.create_dataset("constants",          data=constants)
        hf.create_dataset("param_names",        data=np.array(param_names, dtype="S64"))
        hf.create_dataset("param_scaler_path",  data=str(param_scaler_path.resolve()).encode())
        hf.create_dataset("output_scaler_path", data=str(output_scaler_path.resolve()).encode())
        hf.create_dataset("removed_counts",     data=json.dumps(counts).encode())
        for split, idx in (("train", idx_tr), ("val", idx_va), ("test", idx_te)):
            hf.create_dataset(f"X_{split}_raw", data=par[idx])
            hf.create_dataset(f"X_{split}",     data=param_scaler.transform(par[idx]).astype(np.float32))
            hf.create_dataset(f"y_{split}_raw", data=Y[idx])                       # NaN = masked
            hf.create_dataset(f"y_{split}",     data=((Y[idx] - y_min) / (y_max - y_min)).astype(np.float32))
            hf.create_dataset(f"time_{split}",  data=T[idx])                       # linear ns
            hf.create_dataset(f"qfls_{split}",  data=Q[idx])                       # eV (absolute, incl. p0)
            hf.create_dataset(f"u_min_{split}", data=umin[idx])
            hf.create_dataset(f"qfls0_{split}", data=q0[idx])
            hf.create_dataset(f"sample_index_{split}", data=np.asarray(keep_idx)[idx])
    print(f"\nTraining HDF5 saved: {out_path}")
    print("Done.")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[1] if __doc__ else None)
    ap.add_argument("sim_hdf5")
    ap.add_argument("out_hdf5", nargs="?", default=None)
    ap.add_argument("--axis", choices=["qfls_window", "pl_decades"], default="qfls_window",
                    help="qfls_window: per-curve QFLS window (networks before 10/2026); "
                         "pl_decades: fixed axis log10(PL/PL0) with mask")
    ap.add_argument("--grid-points", type=int, default=511,
                    help="points of the fixed u grid (pl_decades); 511 -> every 2nd point gives 256 incl. both ends")
    a = ap.parse_args()

    sim_path = Path(a.sim_hdf5)
    if not sim_path.exists():
        print(f"Error: file not found: {sim_path}")
        sys.exit(1)

    out_path = (
        Path(a.out_hdf5) if a.out_hdf5
        else sim_path.parent / (sim_path.stem + "_training.hdf5")
    )
    if a.axis == "pl_decades":
        main_pl_decades(sim_path, out_path, n_grid=a.grid_points)
    else:
        main(sim_path, out_path)
