"""
02_trpl_errors.py  —  TRPL error computation (NN-based)
=======================================================
Reads a grid CSV produced by 01_create_grid.py, runs every parameter
combination through the neural network, and writes the MSE and integral-MSE
errors vs. the experimental TRPL data.

Works with full-grid and slice/1-D CSVs (any CSV that has the expected
parameter columns).  Pass different --grid files for slices vs. full grids.

Output
------
  <output_dir>/errors_trpl.csv
      all parameter columns  +  error_trpl_mse  +  error_trpl_integral_mse
      (one row per grid point, in the same order as the input)

Usage
-----
  python 02_trpl_errors.py --config config.json --grid grid.csv
  python 02_trpl_errors.py --config config.json --grid grid_slices.csv
"""

import argparse, json, os, sys, time
import numpy as np
import pandas as pd
from scipy.integrate import trapezoid

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_SCRIPT_DIR, ".."))

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")   # overridden by config

import tensorflow as tf


def _p(msg): print(msg, flush=True)


def _find_col(columns, *keywords):
    """Return the first column whose normalised name contains any keyword."""
    def _norm(s):
        return s.lower().replace(" ", "").replace("_", "").replace("(", "").replace(")", "")
    for col in columns:
        n = _norm(col)
        for kw in keywords:
            if _norm(kw) in n:
                return col
    return None


# ── NN input builder (vectorised, matches nn_gridsearch._build_all_nn_inputs) ─

def _build_nn_inputs(chunk_df, num_traps, Eg, npulse):
    N = len(chunk_df)
    n_feat = 3 + 4 * num_traps
    x = np.empty((N, n_feat), dtype=np.float64)
    x[:, 0] = np.log10(npulse)
    x[:, 1] = Eg
    x[:, 2] = np.log10(chunk_df["krad_cm3s"].values)
    for k in range(1, num_traps + 1):
        col = 3 + 4 * (k - 1)
        x[:, col    ] = np.log10(chunk_df[f"Nt_{k}_cm-3"].values)
        x[:, col + 1] = chunk_df[f"Et_{k}_eV"].values / Eg
        x[:, col + 2] = np.log10(chunk_df[f"tau_n_{k}_s"].values)
        x[:, col + 3] = np.log10(chunk_df[f"tau_p_{k}_s"].values)
    return x


# ── GPU inference (mirrors nn_gridsearch._predict_all_gpu) ────────────────────

def _predict_gpu(x_nn, model, input_scaler, output_scaler, batch_size):
    x_sc = input_scaler.transform(x_nn).astype(np.float32)
    chunks = []
    for s in range(0, len(x_sc), batch_size):
        y = model(tf.constant(x_sc[s: s + batch_size]), training=False)
        chunks.append(y.numpy())
    log_tau   = output_scaler.inverse_transform(np.concatenate(chunks, axis=0))
    tau_batch = 10.0 ** log_tau
    valid     = np.all(np.isfinite(tau_batch) & (tau_batch > 0), axis=1)
    return tau_batch, valid


# ── Error computation (mirrors nn_gridsearch._compute_errors_vectorized) ──────

def _compute_errors(tau_batch, valid_mask, qfls_nn, qfls_exp, tau_exp):
    N = len(tau_batch)
    nan_row = np.full(N, np.nan)
    err_mse  = nan_row.copy()
    err_imse = nan_row.copy()

    if not valid_mask.any():
        return err_mse, err_imse

    tau_v  = tau_batch[valid_mask]
    idx_v  = np.where(valid_mask)[0]

    sort_idx    = np.argsort(qfls_nn)
    qfls_sorted = qfls_nn[sort_idx]
    tau_sorted  = tau_v[:, sort_idx]        # (M, n_qfls)

    # Interpolate simulated tau to experimental QFLS points
    idx_b   = np.searchsorted(qfls_sorted, qfls_exp) - 1
    in_rng  = (idx_b >= 0) & (idx_b < len(qfls_sorted) - 1)
    idx_s   = np.clip(idx_b, 0, len(qfls_sorted) - 2)

    x0 = qfls_sorted[idx_s];  x1 = qfls_sorted[idx_s + 1]
    w  = np.where(in_rng, (qfls_exp - x0) / (x1 - x0 + 1e-30), 0.0)

    #linear interpolation
    tau_fit  = tau_sorted[:, idx_s] * (1.0 - w) + tau_sorted[:, idx_s + 1] * w
    tau_fit[:, ~in_rng] = np.nan

    log_res = np.log10(tau_fit) - np.log10(tau_exp)   # (M, n_exp)

    valid_pts = np.isfinite(log_res)
    n_valid   = valid_pts.sum(axis=1)                  # (M,)
    good      = n_valid > 0

    # MSE in log10 space
    res2 = np.where(valid_pts, log_res ** 2, 0.0)
    mse  = np.where(good, res2.sum(axis=1) / np.maximum(n_valid, 1), np.nan)
    err_mse[idx_v] = mse

    # Integral MSE: trapezoid / QFLS range  (weights by QFLS spacing)
    qfls_exp_v = np.where(in_rng, qfls_exp, np.nan)
    abs_res    = np.where(valid_pts, np.abs(log_res), np.nan)
    q_range    = np.nanmax(qfls_exp_v) - np.nanmin(qfls_exp_v)
    if q_range > 0:
        imse_vals = -trapezoid(abs_res, qfls_exp, axis=1) / q_range
        err_imse[idx_v] = np.where(good, imse_vals, np.nan)

    return err_mse, err_imse


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--grid",   default=None,
                    help="Path to grid CSV (default: <output_dir>/grid.csv)")
    ap.add_argument("--out",    default=None,
                    help="Output CSV path (default: <output_dir>/errors_trpl.csv)")
    args = ap.parse_args()

    cfg_path = args.config if os.path.isabs(args.config) else \
               os.path.join(os.getcwd(), args.config)
    with open(cfg_path) as fh:
        cfg = json.load(fh)

    out_dir = cfg["output_dir"]
    if not os.path.isabs(out_dir):
        out_dir = os.path.join(os.path.dirname(cfg_path), out_dir)
    os.makedirs(out_dir, exist_ok=True)

    grid_mode = cfg["grid"].get("mode", "full")
    default_grid = "grid_slices.csv" if grid_mode == "slice" else "grid.csv"
    grid_csv = args.grid or os.path.join(out_dir, default_grid)
    out_csv  = args.out  or os.path.join(out_dir, "errors_trpl.csv")

    # ── GPU setup ─────────────────────────────────────────────────────────────
    cuda_dev = str(cfg["hardware"].get("cuda_device", "0"))
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_dev
    gpu_frac   = float(cfg["hardware"].get("gpu_memory_frac", 0.5))
    batch_size = int(cfg["hardware"].get("batch_size", 8192))
    chunk_size = int(cfg["grid"].get("chunk_size", 100_000))

    gpus = tf.config.list_physical_devices("GPU")
    _p(f"GPUs: {[g.name for g in gpus]}")
    if gpus and gpu_frac > 0:
        total_mb = 48 * 1024  # fallback
        try:
            import subprocess
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.total",
                 "--format=csv,noheader,nounits", f"--id={cuda_dev}"], text=True)
            total_mb = int(out.strip().splitlines()[0])
        except Exception:
            pass
        lim = int(gpu_frac * total_mb)
        for g in gpus:
            tf.config.set_logical_device_configuration(
                g, [tf.config.LogicalDeviceConfiguration(memory_limit=lim)])
        _p(f"GPU memory limit: {lim} MiB")

    # ── Load NN artifacts ─────────────────────────────────────────────────────
    nn_cfg = cfg["nn"]
    base   = os.path.dirname(cfg_path)
    _p("Loading NN artifacts …")
    sys.path.insert(0, os.path.join(_SCRIPT_DIR, ".."))
    from trpl_fitting import load_nn_artifacts, compute_qfls_network

    def _abs(p):
        return p if os.path.isabs(p) else os.path.join(base, p)

    model, input_scaler, output_scaler = load_nn_artifacts(
        _abs(nn_cfg["model_path"]),
        _abs(nn_cfg["param_scaler_path"]),
        _abs(nn_cfg["output_scaler_path"]),
        training_hdf5_path=_abs(nn_cfg.get("training_hdf5_path", "")),
    )
    ph = cfg["physics"]
    qfls_nn = compute_qfls_network(ph["npulse_cm3"], ph["Eg_eV"])
    if hasattr(model, "_qfls_axis") and model._qfls_axis is not None:
        qfls_nn = model._qfls_axis
    _p(f"  QFLS axis: {qfls_nn.min():.4f}–{qfls_nn.max():.4f} eV ({len(qfls_nn)} pts)")

    # ── Load experimental TRPL data ───────────────────────────────────────────
    trpl_path = cfg["data"]["trpl_csv"]
    if not os.path.isabs(trpl_path):
        trpl_path = os.path.join(base, trpl_path)
    df_trpl = pd.read_csv(trpl_path)
    qfls_col = _find_col(df_trpl.columns, "qfls")
    tau_col  = _find_col(df_trpl.columns, "taudiff", "tau_diff", "tautr", "lifetime")
    if qfls_col is None:
        raise ValueError(f"Cannot find QFLS column in {list(df_trpl.columns)}")
    if tau_col is None:
        raise ValueError(f"Cannot find tau_diff/lifetime column in {list(df_trpl.columns)}")
    _p(f"  Detected columns: QFLS='{qfls_col}', tau='{tau_col}' "
       f"(from {list(df_trpl.columns)})")
    qfls_exp = df_trpl[qfls_col].values
    tau_exp  = df_trpl[tau_col].values
    mask     = np.isfinite(qfls_exp) & np.isfinite(tau_exp) & (tau_exp > 0)
    qfls_exp = qfls_exp[mask]
    tau_exp  = tau_exp[mask]
    _p(f"TRPL data: {len(qfls_exp)} valid points  "
       f"QFLS [{qfls_exp.min():.3f}, {qfls_exp.max():.3f}] eV")

    # ── Process grid CSV in chunks ────────────────────────────────────────────
    num_traps = cfg["physics"]["num_traps"]
    Eg        = cfg["physics"]["Eg_eV"]
    npulse    = cfg["physics"]["npulse_cm3"]

    _p(f"\nGrid CSV : {grid_csv}")
    _p(f"Output   : {out_csv}")
    t0 = time.time()
    rows_done = 0
    first_chunk = True

    for chunk in pd.read_csv(grid_csv, chunksize=chunk_size):
        x_nn = _build_nn_inputs(chunk, num_traps, Eg, npulse)
        tau_batch, valid = _predict_gpu(x_nn, model, input_scaler, output_scaler,
                                        batch_size)
        err_mse, err_imse = _compute_errors(tau_batch, valid, qfls_nn,
                                             qfls_exp, tau_exp)

        out_chunk = chunk.copy()
        # Drop extra label columns if present (from slice grid)
        out_chunk["error_trpl_mse"]          = err_mse
        out_chunk["error_trpl_integral_mse"] = err_imse
        out_chunk.to_csv(out_csv, mode="w" if first_chunk else "a",
                         header=first_chunk, index=False, float_format="%.6e")
        first_chunk = False
        rows_done  += len(chunk)
        rate = rows_done / (time.time() - t0)
        _p(f"  {rows_done:>12,} rows  {rate/1e3:.1f}k rows/s")

    _p(f"\nDone. {rows_done:,} rows in {time.time()-t0:.1f}s → {out_csv}")


if __name__ == "__main__":
    main()
