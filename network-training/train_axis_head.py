"""
train_axis_head.py — axis head for the Conv1D surrogates: q_max and span of each curve's QFLS window
====================================================================================================
16/10/2026 (revision, plan ~/.claude/plans/floofy-purring-fiddle.md)

Every training curve was interpolated onto ITS OWN axis linspace(q_lo, q_hi, 256) of the physical ODE QFLS
(prepare_training_data.py).  The inference so far assumed the analytic high-injection axis
linspace(q_an - 12 Vt ln10, q_an, 256), q_an = 2 Vt ln(n_pulse / n_i), which is wrong where
  * p0 of the traps is not small against n_pulse (low injection): q_hi = QFLS(0) > q_an, and
  * the curve reaches equilibrium before PL has dropped 12 decades: span = q_hi - q_lo < 12 Vt ln10.

The existing network (body + Dense(256) curve head, warm start, unchanged targets) gets a second output "axis":

    [flatten (8192), scaled input] -> Dense(128, GELU) -> Dense(64, GELU) -> d_top (softplus), span (sigmoid x max)

both in units of AXIS_UNIT = 10 meV:  d_top = q_hi - q_an >= 0,  span = q_hi - q_lo in eV (<= 12 Vt ln10).
The axis of a curve is then linspace(q_an + d_top - span, q_an + d_top, 256) — exactly its training axis.

  phase 1 (--head-epochs):  body and curve head frozen, only the axis head is trained -> curve output bit-identical
  phase 2 (--joint-epochs): everything trainable, loss = curve MSE + lambda * axis Huber, small learning rate;
                            kept only if the validation curve MSE is not worse than that of the warm-start network

Targets come from the training HDF5 (qfls_*[:, 0], qfls_*[:, -1], X_*_raw[:, :2], constants): no new simulations.
Outputs: model.keras (outputs [curve, axis]; standard layers only), surrogate_meta.json (axis = "predicted_window"),
axis_metrics.json, training_details.json, offline_provenance.json, history_*.csv, scaler/HDF5 symlinks (<stem>_*).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import h5py
import joblib
import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train_conv_v2 as v2                    # noqa: E402  (load_data, compile schedule, LrLogger, evaluate)

import tensorflow as tf                       # noqa: E402
from tensorflow import keras                  # noqa: E402
from tensorflow.keras import layers           # noqa: E402

AXIS_UNIT = 0.01                              # eV: both axis outputs are in units of 10 meV
COLUMNS = ["d_top", "span_eV"]


# ─────────────────────────────────────────────────────────────────────────────
# targets
# ─────────────────────────────────────────────────────────────────────────────
def constants_of(path) -> dict:
    with h5py.File(path, "r") as hf:
        c = hf["constants"][()]
        return {n: float(c[n]) for n in c.dtype.names}


def q_analytic(X_raw2, c) -> np.ndarray:
    """2 Vt ln(n_pulse / n_i) for rows [log10 n_pulse, Eg] (= inference.surrogate.nn_qfls_axis upper end)."""
    Vt = c["Vt"]
    ni = np.sqrt(c["Nc"] * c["Nv"]) * np.exp(-X_raw2[:, 1] / (2.0 * Vt))
    return 2.0 * Vt * np.log(10.0 ** X_raw2[:, 0] / ni)


def axis_targets(path, split, n=None) -> np.ndarray:
    """(N, 2) = [q_hi - q_an, q_hi - q_lo] / AXIS_UNIT of the stored per-curve axes."""
    c = constants_of(path)
    with h5py.File(path, "r") as hf:
        sl = slice(None) if n is None else slice(0, n)
        X2 = hf[f"X_{split}_raw"][sl, :2].astype(np.float64)
        q = hf[f"qfls_{split}"]
        lo = q[sl, 0].astype(np.float64)
        hi = q[sl, -1].astype(np.float64)
    return (np.stack([hi - q_analytic(X2, c), hi - lo], axis=1) / AXIS_UNIT).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# model
# ─────────────────────────────────────────────────────────────────────────────
def build(src: keras.Model, span_max_units: float, hidden=(128, 64)) -> keras.Model:
    """Warm-start network + axis head; the layers of `src` are reused (shared weights)."""
    x = src.input
    h = layers.Concatenate(name="axis_concat")([src.get_layer("flatten").output, x])
    for k, u in enumerate(hidden):
        h = layers.Dense(u, activation="gelu", name=f"axis_hidden_{k}")(h)
    d_top = layers.Dense(1, activation="softplus", name="axis_dtop",
                         bias_initializer=keras.initializers.Constant(-3.0))(h)          # ~0.5 meV at start
    frac = layers.Dense(1, activation="sigmoid", name="axis_span_frac",
                        bias_initializer=keras.initializers.Constant(5.0))(h)            # ~99 % of the maximum
    span = layers.Rescaling(span_max_units, name="axis_span")(frac)
    axis = layers.Concatenate(name="axis")([d_top, span])
    curve = layers.Identity(name="curve")(src.output)
    return keras.Model(x, [curve, axis], name=f"{src.name}_axis")


def compile_model(model, lr, steps, epochs, warmup_epochs, min_lr_factor, loss_weights, huber_delta):
    warm = warmup_epochs * steps
    sched = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=lr * (0.1 if warm else 1.0), decay_steps=max(1, epochs * steps - warm),
        alpha=min_lr_factor, warmup_target=lr if warm else None, warmup_steps=warm)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=sched), loss={"curve": "mse", "axis": keras.losses.Huber(delta=huber_delta)},
                  loss_weights=loss_weights)


def fit(model, args, d, A, epochs, lr, loss_weights, monitor, out_csv, label):
    steps = max(1, len(d["X_train"]) // args.batch_size)
    compile_model(model, lr, steps, epochs, args.warmup_epochs, args.min_lr_factor, loss_weights, args.huber_delta)
    cbs = [keras.callbacks.EarlyStopping(monitor=monitor, mode="min", patience=args.patience, min_delta=0.0,
                                         restore_best_weights=True),
           v2.LrLogger(), keras.callbacks.CSVLogger(str(out_csv))]
    print(f"\n── {label}: {len(d['X_train'])} train, epochs ≤ {epochs}, lr {lr:.1e} (cosine), "
          f"loss weights {loss_weights}, monitor {monitor} ──", flush=True)
    t0 = time.time()
    h = model.fit(d["X_train"], {"curve": d["y_train"], "axis": A["train"]}, batch_size=args.batch_size,
                  epochs=epochs, shuffle=True, verbose=2, callbacks=cbs,
                  validation_data=(d["X_val"], {"curve": d["y_val"], "axis": A["val"]}))
    dt = time.time() - t0
    print(f"   {label}: {len(h.history['loss'])} epochs in {dt / 60:.1f} min", flush=True)
    return h, dt


# ─────────────────────────────────────────────────────────────────────────────
# evaluation
# ─────────────────────────────────────────────────────────────────────────────
def predict(model, X, bs=4096):
    c, a = model.predict(X, batch_size=bs, verbose=0)
    return np.asarray(c), np.asarray(a)


def axis_metrics(pred, true, Vt) -> dict:
    """Errors in meV of q_max (= d_top), q_lo and the span; vs the analytic assumption (d_top = 0, 12 decades)."""
    dec = Vt * np.log(10.0) / AXIS_UNIT                                      # one PL decade in AXIS_UNIT
    analytic = np.column_stack([np.zeros(len(true)), np.full(len(true), 12.0 * dec)])
    t_dtop, t_span = true[:, 0], true[:, 1]
    subsets = {"all": np.ones(len(true), bool), "d_top>1meV": t_dtop > 0.1,
               "span<11.9dec": t_span < 11.9 * dec}
    out = {}
    for name, m in subsets.items():
        r = {"n": int(m.sum())}
        for lab, P in (("head", pred), ("analytic", analytic)):
            e_hi = (P[m, 0] - t_dtop[m]) * 10.0                               # meV
            e_lo = ((P[m, 0] - P[m, 1]) - (t_dtop[m] - t_span[m])) * 10.0     # meV
            e_sp = (P[m, 1] - t_span[m]) / dec                                # decades
            for k, e in (("q_max_meV", e_hi), ("q_lo_meV", e_lo), ("span_dec", e_sp)):
                a = np.abs(e)
                r[f"{lab}_{k}"] = {"median": float(np.median(a)), "p95": float(np.percentile(a, 95)),
                                   "max": float(a.max())} if len(a) else None
        out[name] = r
    return out


def curve_metrics(curve_pred, d, split, scaler) -> dict:
    y, yr = d[f"y_{split}"], d[f"y_{split}_raw"]
    p = scaler.inverse_transform(curve_pred)
    ss = np.sum((yr - yr.mean(axis=0)) ** 2)
    return {f"{split}_mse": float(np.mean((curve_pred - y) ** 2)),
            f"{split}_mse_log10": float(np.mean((p - yr) ** 2)),
            f"r2_{split}": float(1.0 - np.sum((p - yr) ** 2) / ss)}


# ─────────────────────────────────────────────────────────────────────────────
def main(args):
    t_start = time.time()
    h5 = Path(args.training_hdf5).resolve()
    init_dir = Path(args.init_model).resolve().parent
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = args.stem
    c = constants_of(h5)
    print(f"Loading {h5}", flush=True)
    d = v2.load_data(h5)
    if args.subset:                                         # smoke test
        for k in list(d):
            if isinstance(d[k], np.ndarray) and d[k].ndim >= 1 and k.split("_")[0] in ("X", "y", "qfls"):
                d[k] = d[k][:args.subset]
    n = {s: len(d[f"X_{s}"]) for s in ("train", "val", "test")}
    A = {s: axis_targets(h5, s, n[s]) for s in n}
    for s in A:
        assert len(A[s]) == n[s] and np.all(np.isfinite(A[s])), s
    scaler = joblib.load(init_dir / f"{stem}_output_scaler.joblib")
    span_max = 12.0 * c["Vt"] * np.log(10.0) / AXIS_UNIT
    print(f"  {n}; axis targets (10 meV units): d_top median {np.median(A['train'][:, 0]):.4f} "
          f"max {A['train'][:, 0].max():.2f}, span median {np.median(A['train'][:, 1]):.3f} (max {span_max:.3f})",
          flush=True)

    src = keras.models.load_model(args.init_model, compile=False)
    ref_curve = {s: np.asarray(src.predict(d[f"X_{s}"], batch_size=4096, verbose=0)) for s in ("val", "test")}
    ref = {**curve_metrics(ref_curve["val"], d, "val", scaler), **curve_metrics(ref_curve["test"], d, "test", scaler)}
    print(f"  warm start {args.init_model}: val MSE {ref['val_mse']:.4e}, test MSE {ref['test_mse']:.4e}", flush=True)

    model = build(src, span_max, tuple(args.hidden))
    for lyr in src.layers:
        lyr.trainable = False
    model.summary(print_fn=lambda s: print(s, flush=True), line_length=110)

    # phase 1: axis head only
    h1, dt1 = fit(model, args, d, A, args.head_epochs, args.lr_head, {"curve": 0.0, "axis": 1.0},
                  "val_axis_loss", out / "history_head.csv", "phase 1 (axis head only)")
    model.save(out / "model_phase1.keras")
    cp1, ap1 = predict(model, d["X_test"])
    cv1, _ = predict(model, d["X_val"])
    assert np.array_equal(cp1, ref_curve["test"]), "phase 1 changed the curve output"
    res1 = {**curve_metrics(cv1, d, "val", scaler), **curve_metrics(cp1, d, "test", scaler),
            "axis_test": axis_metrics(ap1, A["test"], c["Vt"])}
    chosen, res, epochs, dt2 = "phase1", res1, len(h1.history["loss"]), 0.0

    # phase 2: joint
    if args.joint_epochs > 0:
        for lyr in model.layers:
            lyr.trainable = True
        h2, dt2 = fit(model, args, d, A, args.joint_epochs, args.lr_joint,
                      {"curve": 1.0, "axis": args.lambda_axis}, "val_loss", out / "history_joint.csv",
                      "phase 2 (joint)")
        model.save(out / "model_phase2.keras")
        cp2, ap2 = predict(model, d["X_test"])
        cv2, _ = predict(model, d["X_val"])
        res2 = {**curve_metrics(cv2, d, "val", scaler), **curve_metrics(cp2, d, "test", scaler),
                "axis_test": axis_metrics(ap2, A["test"], c["Vt"])}
        keep2 = res2["val_mse"] <= ref["val_mse"]
        print(f"  phase 2: val MSE {res2['val_mse']:.4e} vs warm start {ref['val_mse']:.4e} -> "
              f"{'keep phase 2' if keep2 else 'keep phase 1'}", flush=True)
        if keep2:
            chosen, res, epochs = "phase2", res2, epochs + len(h2.history["loss"])
        (out / "phase2_metrics.json").write_text(json.dumps(res2, indent=1))
    (out / "phase1_metrics.json").write_text(json.dumps(res1, indent=1))
    best = keras.models.load_model(out / f"model_{chosen}.keras", compile=False)
    best.save(out / "model.keras")

    # files the inference needs (inference.surrogate.load_network: surrogate_meta.json or <stem>_* in the folder)
    links = {}
    for suffix in ("_param_scaler.joblib", "_output_scaler.joblib", ".hdf5"):
        srcf = (init_dir / f"{stem}{suffix}").resolve()
        dst = out / f"{stem}{suffix}"
        if not dst.exists():
            os.symlink(srcf, dst)
        links[suffix] = str(srcf)
    meta = {"variant": "axis_head", "axis": "predicted_window", "outputs": ["curve", "axis"],
            "axis_columns": COLUMNS, "axis_unit_eV": AXIS_UNIT, "span_max_units": span_max,
            "definition": "q_max = 2 Vt ln(n_pulse/n_i) + d_top; axis = linspace(q_max - span_eV, q_max, 256)",
            "param_scaler_path": links["_param_scaler.joblib"], "output_scaler_path": links["_output_scaler.joblib"],
            "training_hdf5": links[".hdf5"], "init_model": str(Path(args.init_model).resolve()), "phase": chosen}
    (out / "surrogate_meta.json").write_text(json.dumps(meta, indent=1))

    wall = time.time() - t_start
    results = {**{k: v for k, v in res.items() if k != "axis_test"}, "warm_start": ref}
    det = {"timestamp": datetime.now().isoformat(timespec="seconds"),
           "data": {"hdf5_path": str(h5), **{f"n_{s}": n[s] for s in n}},
           "architecture": {"type": "warm-start Conv1D body + curve head + axis head", "hidden": args.hidden,
                            "total_parameters": int(best.count_params())},
           "training": {"head": "dense+axis", "epochs_trained": epochs, "phase": chosen, "script": "train_axis_head.py",
                        "init_model": meta["init_model"], "head_epochs": args.head_epochs,
                        "joint_epochs": args.joint_epochs, "lambda_axis": args.lambda_axis, "lr_head": args.lr_head,
                        "lr_joint": args.lr_joint, "batch_size": args.batch_size, "huber_delta": args.huber_delta, "wall_time_s": wall,
                        "phase_wall_s": {"phase1": dt1, "phase2": dt2}},
           "results": results, "axis_test": res["axis_test"]}
    (out / "training_details.json").write_text(json.dumps(det, indent=2))
    (out / "axis_metrics.json").write_text(json.dumps(res["axis_test"], indent=1))
    pd.DataFrame([{k: v for k, v in results.items() if not isinstance(v, dict)}]).to_csv(out / "metrics_summary.csv",
                                                                                         index=False)
    # offline cost (T8): warm-start network + this run
    try:
        sys.path.insert(0, str(HERE.parent / "rapid-trpl-bayes"))
        from inference.benchmark import offline_costs
        oc = offline_costs(init_dir, init_dir / f"{stem}.hdf5")
        prov = {"network": out.name, "trained_from_scratch": False,
                "training_wall_h_upper": float(oc["training_wall_h_upper"]) + wall / 3600,
                "axis_head_wall_h": wall / 3600, "warm_start_training_wall_h_upper": oc["training_wall_h_upper"],
                "source": f"warm start {init_dir.name} ({oc.get('training_source')}) + train_axis_head.py wall time",
                "simulations": "same simulation files as the warm-start network (no new ODE runs)"}
        (out / "offline_provenance.json").write_text(json.dumps(prov, indent=1))
    except Exception as e:                                                   # noqa: BLE001
        print(f"  offline_provenance.json not written: {type(e).__name__}: {e}", flush=True)
    print("\n" + json.dumps({"chosen": chosen, **results, "axis_test": res["axis_test"]}, indent=1), flush=True)
    print(f"Done: {out} ({wall / 60:.1f} min)", flush=True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("training_hdf5")
    p.add_argument("--init-model", required=True)
    p.add_argument("--stem", required=True, help="file stem of the scalers/HDF5 in the warm-start folder")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--hidden", type=int, nargs="+", default=[128, 64])
    p.add_argument("--head-epochs", type=int, default=60)
    p.add_argument("--joint-epochs", type=int, default=40)
    p.add_argument("--lr-head", type=float, default=1e-3)
    p.add_argument("--lr-joint", type=float, default=1e-4)
    p.add_argument("--lambda-axis", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--patience", type=int, default=15)
    # ~1 % of the curves have very short, irregular spans (they carry 96 % of the span MSE); with an MSE loss the
    # validation loss is dominated by them and early stopping keeps the first epoch -> Huber (quadratic below delta)
    p.add_argument("--huber-delta", type=float, default=0.5, help="axis loss, in units of 10 meV")
    p.add_argument("--warmup-epochs", type=int, default=1)
    p.add_argument("--min-lr-factor", type=float, default=1e-2)
    p.add_argument("--subset", type=int, default=0, help="smoke test: first N rows of every split")
    return p.parse_args()


if __name__ == "__main__":
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    main(parse_args())
