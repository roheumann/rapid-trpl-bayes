"""
12/03/2026  Robin Heumann

Run neural-network inference on the full parameter set (train + val + test).

Loads a trained .keras model and a training HDF5 produced by
prepare_training_data.py, concatenates all three splits, predicts tau_diff
for every sample in one batch, and writes:

  nn_predictions_all.hdf5   – param inputs (raw log10-space) + predicted
                               tau_diff (inverse-scaled, log10 space) for
                               every sample in the full dataset
  nn_inference_times_batch_all.csv – same format as the per-split timing CSV
                                     produced by train_nn.py, so it can be
                                     overlaid directly with ODE completion
                                     timestamps

Usage
-----
    python nn_predict_all.py <model.keras> <training.hdf5> [--output-dir DIR]
"""

import argparse
import os
import time
import numpy as np
import h5py
import joblib
import pandas as pd
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
from tensorflow import keras

for gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)


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
    model = keras.models.load_model(model_path)

    # Use a fixed batch size; passing the full dataset as one batch can exceed
    # cuDNN autotuning limits on some GPUs.
    INFER_BATCH = min(8192, n)

    # ── warmup ───────────────────────────────────────────────────────────────
    print("Warming up …")
    for _ in range(args.n_warmup):
        model.predict(X_all_sc[:INFER_BATCH], batch_size=INFER_BATCH, verbose=0)

    # ── timed batch inference ─────────────────────────────────────────────────
    print("Running full-batch inference …")
    t0          = time.perf_counter()
    y_pred_sc   = model.predict(X_all_sc, batch_size=INFER_BATCH, verbose=0)
    total_time  = time.perf_counter() - t0

    y_pred_raw = output_scaler.inverse_transform(y_pred_sc)

    print(f"  {n} samples  total={total_time:.3f} s  "
          f"mean={total_time / n * 1e3:.3f} ms/curve")

    # ── save timing CSV ───────────────────────────────────────────────────────
    timing_path = out_dir / "nn_inference_times_batch_all.csv"
    pd.DataFrame({
        "n_curves":          np.arange(1, n + 1),
        "cumulative_time_s": np.linspace(total_time / n, total_time, n),
    }).to_csv(timing_path, index=False)
    print(f"  Timing saved  : {timing_path}")

    # ── save predictions HDF5 ─────────────────────────────────────────────────
    pred_path = out_dir / "nn_predictions_all.hdf5"
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
    print(f"  Predictions   : {pred_path}")
    print("Done.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run NN inference on the full dataset (train+val+test).",
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
