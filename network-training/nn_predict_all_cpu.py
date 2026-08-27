"""
12/03/2026  Robin Heumann

CPU-only version of nn_predict_all.py — identical behaviour but forces
TensorFlow to ignore all GPUs and run exclusively on CPU.

Usage
-----
    python nn_predict_all_cpu.py <model.keras> <training.hdf5> [--output-dir DIR]
"""

import argparse
import os
import time
import numpy as np
import h5py
import joblib
import pandas as pd
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""   # hide GPUs before TF initialises
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
from tensorflow import keras

# Confirm no GPUs are visible
tf.config.set_visible_devices([], "GPU")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_hdf5(path: Path) -> dict:
    data = {}
    with h5py.File(path, "r") as hf:
        for key in hf.keys():
            ds = hf[key]
            if ds.dtype.kind in ("S", "O"):
                raw = ds[()]
                data[key] = (raw.decode() if isinstance(raw, bytes)
                             else [v.decode() for v in raw])
            else:
                data[key] = ds[()]
    return data


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def predict_all(args: argparse.Namespace) -> None:
    model_path = Path(args.model)
    h5_path    = Path(args.training_hdf5)

    print(f"Device  : CPU (GPU disabled)")
    print(f"Model   : {model_path}")
    print(f"Data    : {h5_path}")

    # ── output directory ─────────────────────────────────────────────────────
    out_dir = Path(args.output_dir) if args.output_dir else model_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── load data ────────────────────────────────────────────────────────────
    print("Loading data …")
    d = load_hdf5(h5_path)

    X_all_sc  = np.concatenate([d["X_train"],     d["X_val"],     d["X_test"]],     axis=0)
    X_all_raw = np.concatenate([d["X_train_raw"],  d["X_val_raw"],  d["X_test_raw"]],  axis=0)
    y_all_raw = np.concatenate([d["y_train_raw"],  d["y_val_raw"],  d["y_test_raw"]],  axis=0)

    param_names = d["param_names"]
    if hasattr(param_names, "__iter__") and not isinstance(param_names, str):
        param_names = [n.decode() if isinstance(n, bytes) else n for n in param_names]

    output_scaler_path = d["output_scaler_path"]
    if isinstance(output_scaler_path, bytes):
        output_scaler_path = output_scaler_path.decode()
    output_scaler = joblib.load(output_scaler_path)

    n = len(X_all_sc)
    print(f"  Total samples : {n}  "
          f"(train={len(d['X_train'])}  val={len(d['X_val'])}  test={len(d['X_test'])})")

    # ── load model ───────────────────────────────────────────────────────────
    print("Loading model …")
    with tf.device("/CPU:0"):
        model = keras.models.load_model(model_path)

    INFER_BATCH = min(2048, n)

    # ── warmup ───────────────────────────────────────────────────────────────
    print("Warming up …")
    with tf.device("/CPU:0"):
        for _ in range(args.n_warmup):
            model.predict(X_all_sc[:INFER_BATCH], batch_size=INFER_BATCH, verbose=0)

    # ── timed batch inference ─────────────────────────────────────────────────
    print("Running full-batch inference …")
    with tf.device("/CPU:0"):
        t0         = time.perf_counter()
        y_pred_sc  = model.predict(X_all_sc, batch_size=INFER_BATCH, verbose=0)
        total_time = time.perf_counter() - t0

    y_pred_raw = output_scaler.inverse_transform(y_pred_sc)

    print(f"  {n} samples  total={total_time:.3f} s  "
          f"mean={total_time / n * 1e3:.3f} ms/curve")

    # ── save timing CSV ───────────────────────────────────────────────────────
    timing_path = out_dir / "nn_inference_times_batch_all_cpu.csv"
    pd.DataFrame({
        "n_curves":          np.arange(1, n + 1),
        "cumulative_time_s": np.linspace(total_time / n, total_time, n),
    }).to_csv(timing_path, index=False)
    print(f"  Timing saved  : {timing_path}")

    # ── save predictions HDF5 ─────────────────────────────────────────────────
    pred_path = out_dir / "nn_predictions_all_cpu.hdf5"
    with h5py.File(pred_path, "w") as hf:
        hf.create_dataset("X_all_raw",    data=X_all_raw,  compression="gzip")
        hf.create_dataset("y_true_raw",   data=y_all_raw,  compression="gzip")
        hf.create_dataset("y_pred_raw",   data=y_pred_raw, compression="gzip")
        hf.create_dataset(
            "param_names",
            data=np.array([n.encode() for n in param_names], dtype="S64"),
        )
        hf.attrs["model_path"]    = str(model_path.resolve())
        hf.attrs["data_path"]     = str(h5_path.resolve())
        hf.attrs["n_samples"]     = n
        hf.attrs["total_time_s"]  = total_time
        hf.attrs["device"]        = "CPU"
    print(f"  Predictions   : {pred_path}")
    print("Done.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run NN inference on the full dataset (train+val+test) — CPU only.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("model",          help="Path to .keras model file")
    p.add_argument("training_hdf5",  help="Path to training HDF5 (from prepare_training_data.py)")
    p.add_argument("--output-dir",   default=None,
                   help="Output directory (default: same folder as the model)")
    p.add_argument("--n-warmup",     type=int, default=5,
                   help="Number of warmup passes before timed inference")
    return p.parse_args()


if __name__ == "__main__":
    predict_all(parse_args())
