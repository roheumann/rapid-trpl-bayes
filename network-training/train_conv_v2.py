"""
train_conv_v2.py — retraining of the 3609 Conv1D surrogate (same body) with better training settings
====================================================================================================
14/10/2026  (revision, plan ~/.claude/plans/floofy-purring-fiddle.md, Part 1)

The network BODY is exactly the one of 3609 (baseline_code_2026-10-13/train_nn.py: build_model):
    Dense(128, GELU) -> Dense(128*64, GELU) -> Reshape(64,128) -> 2 x {UpSampling1D, Conv1D} -> Flatten
Only the training procedure and, optionally, the output head change:

  --lr-schedule plateau   baseline behaviour (Adam + ReduceLROnPlateau, EarlyStopping)
  --lr-schedule cosine    linear warm-up + cosine decay lr -> lr*min_lr_factor over all epochs,
                          EarlyStopping(restore_best_weights) with a long patience
  --init-model PATH       warm start from an existing model.keras (layers matched by name)
  --head dense            Dense(256, sigmoid) on the scaled log10 tau axis (baseline)
  --head pod --pod-p P    Dense(P, linear) coefficients x FIXED POD basis (mean + top-P SVD modes of the
                          scaled training curves) -> the output is a combination of P smooth modes, so
                          neighbouring points cannot wiggle independently.  Stored as two standard Dense
                          layers ("pod_coefficients", "pod_basis" non-trainable): model.keras loads
                          without custom objects and inference.surrogate treats it like the baseline.
  --head-epochs N         (pod) first N epochs train only the new head with the body frozen
  --eval-only             only evaluate --init-model on the splits (check of the data path: must
                          reproduce the test MSE in its training_details.json)

The data split is the one stored in the training HDF5 (prepare_training_data.py, seed 42).  Runs on the
3609 training HDF5 therefore use exactly the 3609 test set; the merged B2 file keeps the 3609 test set.

Outputs: same files as the baseline (model.keras, history*.csv, training_details.json, metrics_summary.csv,
confusion_matrices/, test_curves_sample.png, timings) + pod_basis.npz for the POD head.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import h5py
import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent / "baseline_code_2026-10-13"
sys.path.insert(0, str(BASE_DIR))
import train_nn as base                       # noqa: E402  (baseline functions, unchanged)

import tensorflow as tf                       # noqa: E402
from tensorflow import keras                  # noqa: E402
from tensorflow.keras import layers           # noqa: E402

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


# ─────────────────────────────────────────────────────────────────────────────
# data
# ─────────────────────────────────────────────────────────────────────────────
KEYS = ["X_train", "y_train", "X_val", "y_val", "X_test", "y_test",
        "X_train_raw", "y_train_raw", "X_val_raw", "y_val_raw", "X_test_raw", "y_test_raw"]


def load_data(path: Path) -> dict:
    """Only the arrays training needs (the baseline loader also reads pl/time/tau_diff: several GB)."""
    d = {}
    with h5py.File(path, "r") as hf:
        for k in KEYS:
            d[k] = hf[k][()]
        d["qfls_test"] = hf["qfls_test"][()] if "qfls_test" in hf else None
        d["param_names"] = [n.decode() if isinstance(n, bytes) else str(n) for n in hf["param_names"][()]]
        p = hf["output_scaler_path"][()]
        d["output_scaler_path"] = p.decode() if isinstance(p, bytes) else str(p)
    return d


# ─────────────────────────────────────────────────────────────────────────────
# model
# ─────────────────────────────────────────────────────────────────────────────
def pod_basis(y_train_scaled: np.ndarray, p: int, n_max: int = 200_000, seed: int = 0):
    """Mean curve and the top-p right singular vectors of the centred scaled training curves."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(y_train_scaled), size=min(n_max, len(y_train_scaled)), replace=False)
    Y = y_train_scaled[idx].astype(np.float64)
    mu = Y.mean(axis=0)
    _, s, vt = np.linalg.svd(Y - mu, full_matrices=False)
    energy = np.cumsum(s ** 2) / np.sum(s ** 2)
    return mu.astype(np.float32), vt[:p].astype(np.float32), s.astype(np.float64), energy


def build(args, input_dim: int, output_dim: int, y_train: np.ndarray, out_dir: Path) -> keras.Model:
    """Baseline body; dense or POD head; optional warm start (weights copied layer by layer by name)."""
    body = base.build_model(input_dim=input_dim, output_dim=output_dim, num_conv_layers=args.num_conv_layers,
                            kernel_size=args.kernel_size, max_filter=args.max_filter,
                            dropout_rate=args.dropout_rate, use_batch_norm=args.batch_norm, lr=args.lr)
    if args.head == "dense":
        model = body
    else:
        mu, modes, s, energy = pod_basis(y_train, args.pod_p)
        np.savez(out_dir / "pod_basis.npz", mean=mu, modes=modes, singular_values=s, energy=energy)
        print(f"  POD basis: p = {args.pod_p}, captured energy = {energy[args.pod_p - 1]:.8f}")
        feat = body.get_layer("flatten").output
        c = layers.Dense(args.pod_p, activation=None, name="pod_coefficients")(feat)
        basis = layers.Dense(output_dim, activation=None, name="pod_basis", trainable=False)
        out = basis(c)
        basis.set_weights([modes, mu])                      # kernel (p, 256) = modes, bias = mean curve
        model = keras.Model(body.input, out, name="conv_pod")

    if args.init_model:
        src = keras.models.load_model(args.init_model, compile=False)
        copied = []
        for lyr in model.layers:
            if lyr.name in ("pod_basis",):
                continue
            try:
                s = src.get_layer(lyr.name)
            except ValueError:
                continue
            if lyr.weights and [w.shape for w in lyr.get_weights()] == [w.shape for w in s.get_weights()]:
                lyr.set_weights(s.get_weights())
                copied.append(lyr.name)
        print(f"  warm start from {args.init_model}: copied {copied}")
    return model


def compile_model(model, lr, steps_per_epoch: int = 1, epochs: int = 1, schedule: str = "plateau",
                  warmup_epochs: int = 0, min_lr_factor: float = 1e-3):
    if schedule == "cosine":
        warm = warmup_epochs * steps_per_epoch
        lr_t = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=lr * (0.1 if warm else 1.0), decay_steps=max(1, epochs * steps_per_epoch - warm),
            alpha=min_lr_factor, warmup_target=lr if warm else None, warmup_steps=warm)
    else:
        lr_t = lr
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr_t), loss="mse", metrics=["mse", "mae"])
    return model


class LrLogger(keras.callbacks.Callback):
    """Writes the current learning rate into the logs (CSVLogger then records it, also for schedules)."""

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            lr = self.model.optimizer.learning_rate
            logs["lr"] = float(lr.numpy() if hasattr(lr, "numpy") else lr)


# ─────────────────────────────────────────────────────────────────────────────
# training
# ─────────────────────────────────────────────────────────────────────────────
def fit_phase(model, args, X, y, Xv, yv, epochs, lr, out_csv, schedule, patience, batch_size, label):
    steps = max(1, len(X) // batch_size)
    compile_model(model, lr, steps, epochs, schedule, args.warmup_epochs if schedule == "cosine" else 0,
                  args.min_lr_factor)
    cbs = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, min_delta=1e-8,
                                         restore_best_weights=True),
           LrLogger(), keras.callbacks.CSVLogger(str(out_csv))]
    if schedule == "plateau":
        cbs.insert(1, keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                                        patience=max(2, patience // 3), min_delta=1e-8,
                                                        min_lr=1e-8, verbose=1))
    print(f"\n── {label}: {len(X)} train / {len(Xv)} val, epochs ≤ {epochs}, lr {lr:.1e} ({schedule}), "
          f"batch {batch_size}, patience {patience} ──", flush=True)
    t0 = time.time()
    h = model.fit(X, y, batch_size=batch_size, epochs=epochs, shuffle=True, validation_data=(Xv, yv),
                  callbacks=cbs, verbose=2)
    print(f"   {label}: {len(h.history['loss'])} epochs in {(time.time() - t0) / 60:.1f} min, "
          f"best val_loss {min(h.history['val_loss']):.4e}", flush=True)
    return h


def evaluate(model, d, scaler):
    res, pred = {}, {}
    for s in ("train", "val", "test"):
        loss, mse, mae = model.evaluate(d[f"X_{s}"], d[f"y_{s}"], batch_size=2048, verbose=0)
        p = scaler.inverse_transform(model.predict(d[f"X_{s}"], batch_size=2048, verbose=0))
        res.update({f"{s}_loss": float(loss), f"{s}_mse": float(mse), f"{s}_mae": float(mae),
                    f"r2_{s}": float(base.compute_r2(d[f"y_{s}_raw"], p)),
                    # unscaled whole-curve MSE in log10 tau_diff units (enters sigma of the inference)
                    f"{s}_mse_log10": float(np.mean((p - d[f"y_{s}_raw"]) ** 2))})
        pred[s] = p
    return res, pred


def train(args):
    t_start = time.time()                     # wall time of the whole run -> training_details.json
    h5_path = Path(args.training_hdf5)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading {h5_path}", flush=True)
    d = load_data(h5_path)
    scaler = joblib.load(d["output_scaler_path"])
    in_dim, out_dim = d["X_train"].shape[1], d["y_train"].shape[1]
    print(f"  train / val / test = {len(d['X_train'])} / {len(d['X_val'])} / {len(d['X_test'])}, "
          f"dims {in_dim} -> {out_dim}", flush=True)

    # files the inference needs next to the model (inference.surrogate: <dir>/<stem>.hdf5 + scalers)
    stem = args.stem or h5_path.stem
    for src, dst in [(h5_path, out_dir / f"{stem}.hdf5"),
                     (Path(d["output_scaler_path"]), out_dir / f"{stem}_output_scaler.joblib"),
                     (Path(str(d["output_scaler_path"]).replace("_output_scaler", "_param_scaler")),
                      out_dir / f"{stem}_param_scaler.joblib")]:
        if not dst.exists():
            os.symlink(src.resolve(), dst)

    if args.eval_only:
        model = keras.models.load_model(args.init_model, compile=False)
        model.compile(loss="mse", metrics=["mse", "mae"])
        res, _ = evaluate(model, d, scaler)
        print(json.dumps(res, indent=1))
        (out_dir / "eval_only.json").write_text(json.dumps(res, indent=1))
        return

    model = build(args, in_dim, out_dim, d["y_train"], out_dir)
    model.summary(print_fn=lambda s: print(s, flush=True))

    hist_files = []
    if args.head == "pod" and args.head_epochs > 0:           # phase 0: only the new head
        for lyr in model.layers:
            lyr.trainable = lyr.name == "pod_coefficients"
        fit_phase(model, args, d["X_train"], d["y_train"], d["X_val"], d["y_val"], args.head_epochs,
                  1e-3, out_dir / "history_head.csv", "plateau", args.head_epochs, args.batch_size, "head only")
        hist_files.append(out_dir / "history_head.csv")
        for lyr in model.layers:
            lyr.trainable = lyr.name != "pod_basis"

    h = fit_phase(model, args, d["X_train"], d["y_train"], d["X_val"], d["y_val"], args.epochs, args.lr,
                  out_dir / "history.csv", args.lr_schedule, args.patience, args.batch_size, "main")
    hist_files.append(out_dir / "history.csv")
    epochs_trained = len(h.history["loss"])

    # baseline's secondary fine-tune on DeltaEtrap >= threshold (same rule as 3609)
    names = d["param_names"]
    et_col = next((i for i, n in enumerate(names) if "etrap" in n.lower()), None)
    if not args.no_finetune and et_col is not None:
        mt = d["X_train_raw"][:, et_col] >= args.finetune_et_threshold
        mv = d["X_val_raw"][:, et_col] >= args.finetune_et_threshold
        fit_phase(model, args, d["X_train"][mt], d["y_train"][mt], d["X_val"][mv], d["y_val"][mv],
                  args.finetune_epochs, args.lr * args.finetune_lr_factor, out_dir / "history_finetune.csv",
                  "plateau", max(5, args.patience // 2), max(16, args.batch_size // 2), "fine-tune high E_t")
        hist_files.append(out_dir / "history_finetune.csv")

    # merged history (epoch numbers continue across phases)
    frames, off = [], 0
    for f in hist_files:
        df = pd.read_csv(f)
        df["phase"] = f.stem.replace("history_", "").replace("history", "main")
        df["epoch"] = df["epoch"] + off
        off += len(df)
        frames.append(df)
    pd.concat(frames, ignore_index=True).to_csv(out_dir / "history_complete.csv", index=False)

    model.save(out_dir / "model.keras")
    res, pred = evaluate(model, d, scaler)
    res["best_val_loss"] = float(min(h.history["val_loss"]))
    print("\n" + json.dumps(res, indent=1), flush=True)

    for s in ("train", "val", "test"):
        cm = base.compute_confusion_matrices(X_val_raw=d[f"X_{s}_raw"], y_val_raw=d[f"y_{s}_raw"],
                                             y_pred_raw=pred[s], param_names=names, n_bins=args.confusion_bins)
        base.save_confusion_matrices(cm, out_dir / "confusion_matrices", split_label=s)
    base.save_training_details(path=out_dir / "training_details.json", model=model, args=args,
                               n_train=len(d["X_train"]), n_val=len(d["X_val"]), n_test=len(d["X_test"]),
                               input_dim=in_dim, output_dim=out_dim, epochs_trained=epochs_trained, results=res)
    td = json.loads((out_dir / "training_details.json").read_text())   # record what really changed
    td["training"].update({"lr_schedule": args.lr_schedule, "warmup_epochs": args.warmup_epochs,
                           "min_lr_factor": args.min_lr_factor, "init_model": args.init_model,
                           "head": args.head, "pod_p": args.pod_p if args.head == "pod" else None,
                           "head_epochs": args.head_epochs if args.head == "pod" else None,
                           "script": "train_conv_v2.py"})
    td["training"]["wall_time_s"] = time.time() - t_start     # read by inference.benchmark.offline_costs (T8)
    (out_dir / "training_details.json").write_text(json.dumps(td, indent=2))
    base.save_metrics_csv(out_dir / "metrics_summary.csv", res)
    if d["qfls_test"] is not None:
        base.plot_test_curves(y_true_raw=d["y_test_raw"], y_pred_raw=pred["test"], qfls=d["qfls_test"],
                              out_path=out_dir / "test_curves_sample.png")
    n = min(200, len(d["X_test"]))
    base.measure_inference_sequential(model, d["X_test"][:n], out_dir / "nn_inference_times_sequential.csv")
    base.measure_inference_batch(model, d["X_test"][:n], out_dir / "nn_inference_times_batch.csv")
    print(f"Done: {out_dir}", flush=True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("training_hdf5")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--stem", default=None, help="file stem the inference expects (default: HDF5 stem)")
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--patience", type=int, default=60)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr-schedule", choices=["plateau", "cosine"], default="cosine")
    p.add_argument("--warmup-epochs", type=int, default=3)
    p.add_argument("--min-lr-factor", type=float, default=1e-3, help="cosine: final lr = lr * factor")
    p.add_argument("--init-model", default=None)
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--head", choices=["dense", "pod"], default="dense")
    p.add_argument("--pod-p", type=int, default=64)
    p.add_argument("--head-epochs", type=int, default=10)
    # body = 3609 (do not change for the comparison)
    p.add_argument("--num-conv-layers", type=int, default=2)
    p.add_argument("--max-filter", type=int, default=128)
    p.add_argument("--kernel-size", type=int, default=16)
    p.add_argument("--dropout-rate", type=float, default=0.0)
    p.add_argument("--batch-norm", action="store_true")
    p.add_argument("--decay-rate", type=float, default=0.995, help="unused (kept for the baseline JSON)")
    # baseline fine-tune phase
    p.add_argument("--no-finetune", action="store_true")
    p.add_argument("--finetune-et-threshold", type=float, default=0.8)
    p.add_argument("--finetune-lr-factor", type=float, default=0.1)
    p.add_argument("--finetune-epochs", type=int, default=100)
    p.add_argument("--confusion-bins", type=int, default=20)
    return p.parse_args()


if __name__ == "__main__":
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    train(parse_args())
