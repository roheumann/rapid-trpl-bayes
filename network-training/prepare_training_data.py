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
    time_train, time_val, time_test      – (n, QFLS_STEPS)   float32  [s]
    pl_train,   pl_val,   pl_test        – (n, QFLS_STEPS)   float32  [a.u.]
    qfls_train, qfls_val, qfls_test      – (n, QFLS_STEPS)   float32  [eV]

Scaled splits – ready for model training:
    X_train, X_val, X_test               – StandardScaler(X_raw)
    y_train, y_val, y_test               – MinMaxScaler(y_raw)  → [0, 1]
"""

import sys
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

        # count invalid tau values
        finite_mask = np.isfinite(tau_i)
        n_invalid   = (~finite_mask).sum()

        # remove whole sample if more than half invalid
        if n_invalid > QFLS_STEPS / 2:
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


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    sim_path = Path(sys.argv[1])
    if not sim_path.exists():
        print(f"Error: file not found: {sim_path}")
        sys.exit(1)

    out_path = (
        Path(sys.argv[2]) if len(sys.argv) >= 3
        else sim_path.parent / (sim_path.stem + "_training.hdf5")
    )
    main(sim_path, out_path)
