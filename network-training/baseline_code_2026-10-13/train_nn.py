"""
10/03/2026  Robin Heumann

Train a MLP neural network to predict TRPL differential lifetime
curves (τ_diff) from material parameters.

Reads a training HDF5 produced by prepare_training_data.py, trains the
network, and writes:
  - trained model (.keras)
  - training history (history.csv)
  - per-parameter confusion matrices (confusion_matrix_<param>.csv)
  - per-parameter error summary    (confusion_summary_<param>.csv)
  - training_details.json           (architecture + hyperparameters + results)

The "confusion matrix" for each input material parameter is a 2-D count
matrix:
    rows    – N_BINS bins of the parameter value
    columns – N_BINS bins of the per-sample MSE on the validation set
    values  – number of validation samples that fall in each (param, MSE) cell

This reveals which parameter regimes are easy vs. hard for the model.

Optional W&B logging is activated with --wandb.

Usage
-----
    python train_nn.py <training_hdf5>
                       [--output-dir DIR]
                       [--epochs N]
                       [--batch-size N]
                       [--patience N]
                       [--hidden-dim N]
                       [--num-hidden-layers N]
                       [--dropout-rate FLOAT]
                       [--lr FLOAT]
                       [--decay-rate FLOAT]
                       [--confusion-bins N]
                       [--wandb]
                       [--wandb-project PROJECT]
                       [--wandb-run-name NAME]

The training HDF5 must be produced by prepare_training_data.py.
Both scaler joblib files (<stem>_param_scaler.joblib and
<stem>_output_scaler.joblib) must remain next to the HDF5, or the paths
stored inside it must still be valid.
"""

import argparse
import json
import os
import time
import numpy as np
import h5py
import joblib
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import gc

# ── suppress verbose TF logging ──────────────────────────────────────────────
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
gc.collect()
keras.backend.clear_session()


# ─────────────────────────────────────────────────────────────────────────────
# GPU memory growth (avoids OOM on shared GPUs)
# ─────────────────────────────────────────────────────────────────────────────
for gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_training_hdf5(path: Path) -> dict:
    """Load all datasets from a training HDF5 file."""

    def _decode_scalar(v):
        if isinstance(v, (bytes, np.bytes_)):
            return v.decode()
        if isinstance(v, np.ndarray) and v.ndim == 0:
            item = v.item()
            return item.decode() if isinstance(item, (bytes, np.bytes_)) else item
        return v

    data = {}
    with h5py.File(path, "r") as hf:
        for key in hf.keys():
            ds  = hf[key]
            raw = ds[()]
            if ds.dtype.kind not in ("S", "O"):
                data[key] = raw
            elif isinstance(raw, (bytes, np.bytes_)):
                # scalar string returned as plain bytes
                data[key] = raw.decode()
            elif isinstance(raw, np.ndarray) and raw.ndim == 0:
                item = raw.item()
                data[key] = item.decode() if isinstance(item, (bytes, np.bytes_)) else item
            elif isinstance(raw, np.ndarray) and raw.ndim == 1:
                # 1-D string array (e.g. param_names)
                data[key] = [
                    v.decode() if isinstance(v, (bytes, np.bytes_)) else v
                    for v in raw
                ]
            else:
                # 2-D+ string arrays (e.g. constants) – keep as raw bytes
                data[key] = raw
    return data


# ─────────────────────────────────────────────────────────────────────────────
# Model architecture
# ─────────────────────────────────────────────────────────────────────────────

def build_model(
    input_dim: int,
    output_dim: int,
    num_conv_layers: int = 2,
    kernel_size: int = 16,
    max_filter: int = 128,
    dropout_rate: float = 0.0,
    use_batch_norm: bool = False,
    lr: float = 1e-3,
    decay_steps: int = 1000,
    decay_rate: float = 0.995,
) -> keras.Model:
    """
    UpSampling1D + Conv1D surrogate model.

    Architecture (defaults: max_filter=128, num_conv_layers=2, output_dim=256):
        Dense(max_filter, GELU)
        Dense(max_filter × map_size, GELU)      map_size = output_dim / 2^num_conv_layers
        Reshape(map_size, max_filter)            e.g. (64, 128) for output_dim=256
        [BatchNormalization]  (optional)
        num_conv_layers × {
            UpSampling1D(2)
            Conv1D(n_filters, kernel_size, 'same', GELU)
            [BatchNormalization]  (optional)
            [Dropout]             (optional)
        }
        Flatten → Dense(output_dim, sigmoid)
    """
    map_size = max(1, int(round(output_dim / (2 ** num_conv_layers))))

    inp = keras.Input(shape=(input_dim,), name="parameters")
    x = layers.Dense(max_filter, activation="gelu", name="dense_seed_0")(inp)
    x = layers.Dense(max_filter * map_size, activation="gelu", name="dense_seed_1")(x)
    x = layers.Reshape((map_size, max_filter), name="reshape")(x)
    if use_batch_norm:
        x = layers.BatchNormalization(name="bn_seed")(x)

    for i in range(num_conv_layers):
        n_filters = max(1, max_filter // (2 ** (i + 1)))
        x = layers.UpSampling1D(size=2, name=f"upsample_{i}")(x)
        x = layers.Conv1D(
            filters=n_filters,
            kernel_size=kernel_size,
            padding="same",
            activation="gelu",
            name=f"conv_{i}",
        )(x)
        if use_batch_norm:
            x = layers.BatchNormalization(name=f"bn_{i}")(x)
        if dropout_rate > 0.0:
            x = layers.Dropout(dropout_rate, name=f"dropout_{i}")(x)

    x = layers.Flatten(name="flatten")(x)
    out = layers.Dense(output_dim, activation="sigmoid", name="output")(x)

    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=lr,
        decay_steps=decay_steps,
        decay_rate=decay_rate,
        staircase=False,
    )
    model = keras.Model(inputs=inp, outputs=out)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr), #learning_rate=lr_schedule
        loss="mse",
        metrics=["mse", "mae"],
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Architecture registry
# ─────────────────────────────────────────────────────────────────────────────

ARCHITECTURES_DIR = Path(__file__).parent / "architectures"


def save_architecture_config(
    model: keras.Model,
    args: argparse.Namespace,
    input_dim: int,
    output_dim: int,
    timestamp: str,
) -> Path:
    """
    Save a self-contained architecture config to nn-training/architectures/.

    File name: <timestamp>_<arch_name>.json
    The JSON contains all kwargs needed to call build_model() again, plus the
    full Keras model config for exact layer-by-layer reconstruction.

    Returns the path written.
    """
    ARCHITECTURES_DIR.mkdir(parents=True, exist_ok=True)

    # Human-readable name encoding the key hyperparameters
    arch_name = f"UpSampling1D_Conv1D_f{args.max_filter}_k{args.kernel_size}_l{args.num_conv_layers}"
    if args.batch_norm:
        arch_name += "_BN"
    if args.dropout_rate > 0:
        arch_name += f"_do{args.dropout_rate}"

    config = {
        "timestamp":        timestamp,
        "architecture_name": arch_name,
        "architecture_type": "UpSampling1D + Conv1D",
        "build_model_kwargs": {
            "input_dim":       input_dim,
            "output_dim":      output_dim,
            "num_conv_layers": args.num_conv_layers,
            "kernel_size":     args.kernel_size,
            "max_filter":      args.max_filter,
            "dropout_rate":    args.dropout_rate,
            "use_batch_norm":  args.batch_norm,
            "lr":              args.lr,
            "decay_rate":      args.decay_rate,
        },
        "total_params":     int(model.count_params()),
        "trainable_params": int(sum(np.prod(w.shape) for w in model.trainable_weights)),
        "keras_model_config": json.loads(model.to_json()),
    }

    out_path = ARCHITECTURES_DIR / f"{timestamp}_{arch_name}.json"
    with open(out_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Architecture config saved: {out_path}")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# R² metric
# ─────────────────────────────────────────────────────────────────────────────

def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Global R² (coefficient of determination) over all samples and time points.

    R² = 1 - SS_res / SS_tot
         SS_res = Σ (y_true - y_pred)²
         SS_tot = Σ (y_true - mean(y_true))²

    Computed on the *raw* (physical, log10-space) values so the metric is
    independent of the [0,1] normalization used during training.
    """
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot == 0.0:
        return 0.0
    return 1.0 - ss_res / ss_tot


# ─────────────────────────────────────────────────────────────────────────────
# Training details – save to JSON
# ─────────────────────────────────────────────────────────────────────────────

def save_training_details(
    path: Path,
    model: keras.Model,
    args: argparse.Namespace,
    n_train: int,
    n_val: int,
    n_test: int,
    input_dim: int,
    output_dim: int,
    epochs_trained: int,
    results: dict,
) -> None:
    """
    Write a JSON file documenting the model structure and training configuration.

    Sections
    --------
    data         – dataset sizes and dimensionality
    architecture – layer-by-layer breakdown with parameter counts
    training     – all hyperparameters and callbacks used
    results      – final loss / R² on train / val / test splits
    """
    # ── layer-by-layer architecture ──────────────────────────────────────────
    layer_info = []
    for lyr in model.layers:
        cfg = lyr.get_config()
        entry = {
            "name":        lyr.name,
            "type":        type(lyr).__name__,
            "trainable_params": int(lyr.count_params()),
        }
        # Dense layers
        if hasattr(lyr, "units"):
            entry["units"]      = int(lyr.units)
            entry["activation"] = cfg.get("activation", "linear")
        # Dropout layers
        if hasattr(lyr, "rate"):
            entry["dropout_rate"] = float(lyr.rate)
        # Input layers
        if hasattr(lyr, "input_shape"):
            try:
                entry["output_shape"] = list(lyr.output_shape)
            except Exception:
                pass
        layer_info.append(entry)

    details = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "data": {
            "hdf5_path":  str(args.training_hdf5),
            "n_train":    n_train,
            "n_val":      n_val,
            "n_test":     n_test,
            "input_dim":  input_dim,
            "output_dim": output_dim,
        },
        "architecture": {
            "type":                 "UpSampling1D + Conv1D",
            "num_conv_layers":      args.num_conv_layers,
            "max_filter":           args.max_filter,
            "kernel_size":          args.kernel_size,
            "dropout_rate":         args.dropout_rate,
            "use_batch_norm":       args.batch_norm,
            "total_parameters":     int(model.count_params()),
            "trainable_parameters": int(
                sum(np.prod(w.shape) for w in model.trainable_weights)
            ),
            "layers": layer_info,
        },
        "training": {
            "batch_size":        args.batch_size,
            "epochs_max":        args.epochs,
            "epochs_trained":    epochs_trained,
            "patience":          args.patience,
            "optimizer":         "Adam",
            "loss":              "mse",
            "lr_initial":        args.lr,
            "lr_decay_rate":     args.decay_rate,
            "lr_decay_steps":    "steps_per_epoch",
            "callbacks": [
                "EarlyStopping(monitor=val_loss, restore_best_weights=True)",
                "ReduceLROnPlateau(monitor=val_loss, factor=0.5, patience=patience//3)",
                "TensorBoard",
                "CSVLogger",
            ],
        },
        "results": results,
    }

    with open(path, "w") as f:
        json.dump(details, f, indent=2)
    print(f"  Training details saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Confusion-matrix analysis
# ─────────────────────────────────────────────────────────────────────────────

def compute_confusion_matrices(
    X_val_raw: np.ndarray,
    y_val_raw: np.ndarray,
    y_pred_raw: np.ndarray,
    param_names: list[str],
    n_bins: int = 20,
) -> dict:
    """
    For each input parameter, build a 2-D confusion matrix:
        rows    – parameter value bins
        columns – per-sample MSE bins
        values  – count of validation samples in that cell

    Also computes a summary (mean/std MSE per parameter bin).

    Parameters
    ----------
    X_val_raw  : (n_val, n_params)  raw log10-space parameters
    y_val_raw  : (n_val, n_time)    true log10 tau_diff
    y_pred_raw : (n_val, n_time)    predicted log10 tau_diff (inverse-scaled)
    param_names: list of parameter name strings

    Returns
    -------
    dict mapping param_name →
        {
          "matrix": 2-D int array (n_bins × n_bins),
          "param_edges": 1-D array (n_bins+1),
          "mse_edges":   1-D array (n_bins+1),
          "summary":     DataFrame with per-bin error stats,
        }
    """
    # Per-sample MSE in log10(tau_diff) space
    per_sample_mse = np.mean((y_val_raw - y_pred_raw) ** 2, axis=1)
    per_sample_mae = np.mean(np.abs(y_val_raw - y_pred_raw), axis=1)

    # MSE bin edges: use percentiles to avoid extreme outliers dominating bins
    mse_edges = np.percentile(
        per_sample_mse, np.linspace(0, 100, n_bins + 1)
    )
    # Ensure strictly monotone (can be equal at extreme percentiles)
    mse_edges = np.unique(mse_edges)
    if len(mse_edges) < 3:
        mse_edges = np.linspace(per_sample_mse.min(), per_sample_mse.max(), n_bins + 1)

    results = {}
    for j, name in enumerate(param_names):
        param_vals = X_val_raw[:, j]

        # Parameter bin edges (uniform over observed range)
        p_edges = np.linspace(param_vals.min(), param_vals.max(), n_bins + 1)

        # 2-D histogram
        matrix, _, _ = np.histogram2d(
            param_vals, per_sample_mse,
            bins=[p_edges, mse_edges],
        )

        # Summary: mean MSE, std MSE, MAE, count per parameter bin
        bin_idx = np.digitize(param_vals, p_edges) - 1
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)

        rows = []
        for b in range(n_bins):
            mask = bin_idx == b
            count = int(mask.sum())
            if count == 0:
                rows.append(
                    dict(
                        bin_index=b,
                        param_low=p_edges[b],
                        param_high=p_edges[b + 1],
                        param_center=0.5 * (p_edges[b] + p_edges[b + 1]),
                        count=0,
                        mean_mse=np.nan,
                        std_mse=np.nan,
                        rmse=np.nan,
                        mean_mae=np.nan,
                    )
                )
            else:
                m = per_sample_mse[mask]
                rows.append(
                    dict(
                        bin_index=b,
                        param_low=p_edges[b],
                        param_high=p_edges[b + 1],
                        param_center=0.5 * (p_edges[b] + p_edges[b + 1]),
                        count=count,
                        mean_mse=float(m.mean()),
                        std_mse=float(m.std()),
                        rmse=float(np.sqrt(m.mean())),
                        mean_mae=float(per_sample_mae[mask].mean()),
                    )
                )

        results[name] = {
            "matrix":      matrix.astype(int),
            "param_edges": p_edges,
            "mse_edges":   mse_edges,
            "summary":     pd.DataFrame(rows),
        }

    return results


def save_metrics_csv(path: Path, metrics: dict) -> None:
    """
    Write a flat CSV with MAE / MSE / R² for each data split.

    Columns: split, MAE, MSE, R2
    """
    rows = [
        {"split": "train", "MAE": metrics["train_mae"], "MSE": metrics["train_mse"], "R2": metrics["r2_train"]},
        {"split": "val",   "MAE": metrics["val_mae"],   "MSE": metrics["val_mse"],   "R2": metrics["r2_val"]},
        {"split": "test",  "MAE": metrics["test_mae"],  "MSE": metrics["test_mse"],  "R2": metrics["r2_test"]},
    ]
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"  Metrics summary saved: {path}")


def plot_test_curves(
    y_true_raw: np.ndarray,
    y_pred_raw: np.ndarray,
    qfls: np.ndarray,
    out_path: Path,
    n_curves: int = 10,
    seed: int = 42,
) -> None:
    """
    Plot n_curves randomly selected test samples and their NN predictions.

    Each pair shares a colour.  Simulation = dashed, NN prediction = solid.
    x-axis: QFLS (eV),  y-axis: log10(tau_diff / s)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    rng = np.random.default_rng(seed)
    n   = min(n_curves, len(y_true_raw))
    idx = np.sort(rng.choice(len(y_true_raw), size=n, replace=False))

    # CSV output directory: <plot_stem>_data/ next to the plot
    csv_dir = out_path.parent / (out_path.stem + "_data")
    csv_dir.mkdir(parents=True, exist_ok=True)

    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(9, 5))

    for k, i in enumerate(idx):
        color = cmap(k % 10)
        x = qfls[i]
        ax.plot(x, y_true_raw[i], linestyle="--", color=color, linewidth=1.0, alpha=0.85)
        ax.plot(x, y_pred_raw[i], linestyle="-",  color=color, linewidth=1.4, alpha=0.95)

        pd.DataFrame({
            "qfls_eV":             x,
            "log10_tau_diff_sim":  y_true_raw[i],
            "log10_tau_diff_nn":   y_pred_raw[i],
        }).to_csv(csv_dir / f"curve_{k:02d}_sample{i:05d}.csv", index=False)

    legend_handles = [
        Line2D([0], [0], color="k", linestyle="--", linewidth=1.2, label="Simulation (test set)"),
        Line2D([0], [0], color="k", linestyle="-",  linewidth=1.4, label="NN prediction"),
    ]
    ax.legend(handles=legend_handles, fontsize=10)
    ax.set_xlabel("QFLS (eV)")
    ax.set_ylabel(r"$\log_{10}(\tau_\mathrm{diff}\ /\ \mathrm{s})$")
    ax.set_title(f"Test set — {n} randomly selected curves (seed={seed})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Test curves plot saved: {out_path}")
    print(f"  Test curves data saved: {csv_dir}")


def _plot_confusion_matrix(
    matrix: np.ndarray,
    param_centers: np.ndarray,
    col_centers: np.ndarray,
    summary: "pd.DataFrame",
    param_name: str,
    split_label: str,
    out_dir: Path,
    tag: str,
) -> None:
    """
    Save two images per parameter:
      1. Heatmap  – 2-D count matrix (log-scale colour), y=param, x=MSE
      2. Summary  – RMSE and mean MAE vs parameter bin centre
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    # ── 1. Heatmap ────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))

    # log-scale normalisation; guard against zeros
    mat_plot = matrix.astype(float)
    mat_plot[mat_plot == 0] = np.nan
    norm = mcolors.LogNorm(vmin=1, vmax=max(1, np.nanmax(mat_plot)))

    im = ax.imshow(
        mat_plot,
        origin="lower",
        aspect="auto",
        norm=norm,
        cmap="viridis",
        extent=[col_centers[0], col_centers[-1],
                param_centers[0], param_centers[-1]],
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Sample count (log scale)", fontsize=10)

    ax.set_xlabel("Per-sample MSE  (log₁₀ τ_diff space)", fontsize=10)
    ax.set_ylabel(param_name, fontsize=10)
    ax.set_title(
        f"Confusion matrix – {param_name}  [{split_label}]", fontsize=11
    )
    fig.tight_layout()
    fig.savefig(out_dir / f"confusion_matrix_{param_name}{tag}.png", dpi=150)
    plt.close(fig)

    # ── 2. Summary (RMSE + MAE vs parameter bin) ──────────────────────────────
    df = summary.dropna(subset=["rmse", "mean_mae"])
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df["param_center"], df["rmse"],     marker="o", ms=4,
            linewidth=1.4, label="RMSE")
    ax.plot(df["param_center"], df["mean_mae"], marker="s", ms=4,
            linewidth=1.4, label="Mean MAE", linestyle="--")

    # shaded ±1 std band around RMSE
    df2 = summary.dropna(subset=["mean_mse", "std_mse"])
    if not df2.empty:
        lo = np.sqrt(np.maximum(0, df2["mean_mse"] - df2["std_mse"]))
        hi = np.sqrt(df2["mean_mse"] + df2["std_mse"])
        ax.fill_between(df2["param_center"], lo, hi, alpha=0.2, label="±1 std (RMSE)")

    ax.set_xlabel(param_name, fontsize=10)
    ax.set_ylabel("Error  (log₁₀ τ_diff space)", fontsize=10)
    ax.set_title(f"Error summary – {param_name}  [{split_label}]", fontsize=11)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / f"confusion_summary_{param_name}{tag}.png", dpi=150)
    plt.close(fig)


def save_confusion_matrices(results: dict, out_dir: Path, split_label: str = "") -> None:
    """
    Write confusion matrices and summaries to CSV files (OriginLab-friendly)
    and PNG images.

    Confusion matrix CSV layout (importable as matrix worksheet in Origin):
        First column  – param_center  (numeric, use as X axis)
        Other columns – MSE bin centers as numeric float headers

    Summary CSV layout (flat table, importable as worksheet):
        bin_index, param_low, param_high, param_center, count,
        mean_mse, std_mse, rmse, mean_mae

    Images (PNG):
        confusion_matrix_<param>_<split>.png  – log-scale heatmap
        confusion_summary_<param>_<split>.png – RMSE / MAE vs parameter bin
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_{split_label}" if split_label else ""

    for name, res in results.items():
        matrix  = res["matrix"]
        p_edges = res["param_edges"]
        m_edges = res["mse_edges"]

        n_mse_bins    = matrix.shape[1]
        col_centers   = 0.5 * (m_edges[:n_mse_bins]   + m_edges[1:n_mse_bins + 1])
        n_param_bins  = matrix.shape[0]
        param_centers = 0.5 * (p_edges[:n_param_bins] + p_edges[1:n_param_bins + 1])

        # ── CSVs ─────────────────────────────────────────────────────────────
        df_matrix = pd.DataFrame(
            matrix,
            columns=[f"{v:.6e}" for v in col_centers],
        )
        df_matrix.insert(0, "param_center", param_centers)

        df_matrix.to_csv(out_dir / f"confusion_matrix_{name}{tag}.csv", index=False)
        res["summary"].to_csv(out_dir / f"confusion_summary_{name}{tag}.csv", index=False)

        # ── Images ───────────────────────────────────────────────────────────
        _plot_confusion_matrix(
            matrix        = matrix,
            param_centers = param_centers,
            col_centers   = col_centers,
            summary       = res["summary"],
            param_name    = name,
            split_label   = split_label or "val",
            out_dir       = out_dir,
            tag           = tag,
        )

    print(f"  Confusion matrices ({split_label or 'val'}) saved to: {out_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# Inference timing
# ─────────────────────────────────────────────────────────────────────────────

def measure_inference_sequential(
    model: keras.Model,
    X: np.ndarray,
    out_path: Path,
    n_warmup: int = 5,
) -> None:
    """
    Time the model by predicting each sample individually (batch=1).

    Runs n_warmup full-batch passes first so TF graph tracing is complete
    before timing begins.  Each sample is then predicted one-by-one and the
    wall time recorded.

    CSV columns (OriginLab-ready):
        n_curves                – cumulative number of predictions
        per_sample_time_s       – wall time for this single prediction [s]
        cumulative_time_s       – cumulative wall time [s]
    """
    n = len(X)

    # warmup with full batch to compile TF graph
    for _ in range(n_warmup):
        model.predict(X, batch_size=n, verbose=0)

    per_sample = np.empty(n, dtype=np.float64)
    for i in range(n):
        x  = X[i : i + 1]
        t0 = time.perf_counter()
        model.predict(x, batch_size=1, verbose=0)
        per_sample[i] = time.perf_counter() - t0

    cumulative = np.cumsum(per_sample)
    df = pd.DataFrame({
        "n_curves":           np.arange(1, n + 1),
        "per_sample_time_s":  per_sample,
        "cumulative_time_s":  cumulative,
    })
    df.to_csv(out_path, index=False)
    print(f"  Sequential timing : {n} samples  "
          f"total={cumulative[-1]:.3f} s  mean={per_sample.mean()*1e3:.3f} ms/curve")
    print(f"  Saved: {out_path}")


def measure_inference_batch(
    model: keras.Model,
    X: np.ndarray,
    out_path: Path,
    n_warmup: int = 5,
) -> None:
    """
    Time the model predicting the entire parameter set as one batch.

    This is the fair comparison against the ODE solver (which also processes
    all samples in a single run).  Warmup uses the exact same input shape to
    avoid TF retracing overhead on the timed call.

    The cumulative curve is built by linear interpolation from 0 → total_time
    so it can be overlaid directly with the ODE simulation-times CSV.

    CSV columns (OriginLab-ready):
        n_curves          – cumulative number of predictions
        cumulative_time_s – cumulative wall time [s]
    """
    n = len(X)

    for _ in range(n_warmup):
        model.predict(X, batch_size=n, verbose=0)

    t0         = time.perf_counter()
    model.predict(X, batch_size=n, verbose=0)
    total_time = time.perf_counter() - t0

    df = pd.DataFrame({
        "n_curves":          np.arange(1, n + 1),
        "cumulative_time_s": np.linspace(total_time / n, total_time, n),
    })
    df.to_csv(out_path, index=False)
    print(f"  Batch timing      : {n} samples  "
          f"total={total_time:.3f} s  mean={total_time/n*1e3:.3f} ms/curve")
    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main training routine
# ─────────────────────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    # ── optional W&B ─────────────────────────────────────────────────────────
    use_wandb = args.wandb
    if use_wandb:
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name or datetime.now().strftime("%Y%m%d-%H%M%S"),
                config=vars(args),
            )
        except ImportError:
            print("Warning: wandb not installed – continuing without W&B logging.")
            use_wandb = False

    # ── load data ────────────────────────────────────────────────────────────
    h5_path = Path(args.training_hdf5)
    print(f"Loading training data: {h5_path}")
    d = load_training_hdf5(h5_path)

    X_train = d["X_train"]
    y_train = d["y_train"]
    X_val   = d["X_val"]
    y_val   = d["y_val"]
    X_test  = d["X_test"]
    y_test  = d["y_test"]

    X_train_raw = d["X_train_raw"]
    y_train_raw = d["y_train_raw"]
    X_val_raw   = d["X_val_raw"]
    y_val_raw   = d["y_val_raw"]
    X_test_raw  = d["X_test_raw"]
    y_test_raw  = d["y_test_raw"]

    # ── load output scaler ────────────────────────────────────────────────────
    output_scaler_path = (
        d["output_scaler_path"].decode()
        if isinstance(d["output_scaler_path"], bytes)
        else d["output_scaler_path"]
    )
    output_scaler: MinMaxScaler = joblib.load(output_scaler_path)

    # Decode param names (stored as byte strings)
    raw_names = d["param_names"]
    param_names = (
        [n.decode() if isinstance(n, bytes) else n for n in raw_names]
        if hasattr(raw_names, "__iter__") and not isinstance(raw_names, str)
        else [raw_names]
    )

    INPUT_DIM  = X_train.shape[1]
    OUTPUT_DIM = y_train.shape[1]
    STEPS      = max(1, X_train.shape[0] // args.batch_size)

    print(f"  Input dim : {INPUT_DIM}")
    print(f"  Output dim: {OUTPUT_DIM}")
    print(f"  Train / Val / Test : {len(X_train)} / {len(X_val)} / {len(X_test)}")

    # ── output directory ─────────────────────────────────────────────────────
    ts     = datetime.now().strftime("%Y%m%d-%H%M%S")
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = h5_path.parent / f"{ts}_nn_training"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Output dir: {out_dir}")

    # ── build model ──────────────────────────────────────────────────────────
    model = build_model(
        input_dim       = INPUT_DIM,
        output_dim      = OUTPUT_DIM,
        num_conv_layers = args.num_conv_layers,
        kernel_size     = args.kernel_size,
        max_filter      = args.max_filter,
        dropout_rate    = args.dropout_rate,
        use_batch_norm  = args.batch_norm,
        lr              = args.lr,
        decay_steps     = STEPS,
        decay_rate      = args.decay_rate,
    )
    model.summary()

    # ── save architecture config ─────────────────────────────────────────────
    save_architecture_config(
        model     = model,
        args      = args,
        input_dim = INPUT_DIM,
        output_dim= OUTPUT_DIM,
        timestamp = ts,
    )

    # ── callbacks ────────────────────────────────────────────────────────────
    reduce_lr_patience = max(1, args.patience // 3)
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=args.patience,
            min_delta=1e-7,
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=reduce_lr_patience,
            min_delta=1e-7,
            min_lr=1e-7,
            verbose=1,
        ),
        keras.callbacks.TensorBoard(
            log_dir=str(out_dir / "logs"),
            histogram_freq=0,
        ),
        keras.callbacks.CSVLogger(str(out_dir / "history.csv")),
    ]

    if use_wandb:
        import wandb
        from wandb.integration.keras import WandbMetricsLogger
        callbacks.append(WandbMetricsLogger(log_freq="epoch"))

    # ── train ────────────────────────────────────────────────────────────────
    history = model.fit(
        X_train, y_train,
        batch_size      = args.batch_size,
        epochs          = args.epochs,
        shuffle         = True,
        validation_data = (X_val, y_val),
        callbacks       = callbacks,
        verbose         = 1,
    )
    epochs_trained = len(history.history["loss"])

    # ── confusion matrices after main training (pre-fine-tune) ───────────────
    _y_train_pred_main = output_scaler.inverse_transform(
        model.predict(X_train, batch_size=512, verbose=0)
    )
    _y_val_pred_main   = output_scaler.inverse_transform(
        model.predict(X_val,   batch_size=512, verbose=0)
    )
    _y_test_pred_main  = output_scaler.inverse_transform(
        model.predict(X_test,  batch_size=512, verbose=0)
    )
    _confusion_dir = out_dir / "confusion_matrices"
    for _split_label, _X_raw, _y_raw, _y_pred_raw in [
        ("train_main", X_train_raw, y_train_raw, _y_train_pred_main),
        ("val_main",   X_val_raw,   y_val_raw,   _y_val_pred_main),
        ("test_main",  X_test_raw,  y_test_raw,  _y_test_pred_main),
    ]:
        print(f"\nComputing confusion matrices on {_split_label} set …")
        _cm = compute_confusion_matrices(
            X_val_raw   = _X_raw,
            y_val_raw   = _y_raw,
            y_pred_raw  = _y_pred_raw,
            param_names = param_names,
            n_bins      = args.confusion_bins,
        )
        save_confusion_matrices(_cm, _confusion_dir, split_label=_split_label)

    # ── secondary fine-tuning on high-Etrap samples ──────────────────────────
    et_col = next(
        (i for i, n in enumerate(param_names) if "etrap" in n.lower()),
        None,
    )
    if not args.no_finetune and et_col is not None:
        ft_mask_tr  = X_train_raw[:, et_col] >= args.finetune_et_threshold
        ft_mask_val = X_val_raw[:,   et_col] >= args.finetune_et_threshold
        n_ft_train  = ft_mask_tr.sum()
        n_ft_val    = ft_mask_val.sum()

        if n_ft_train < 16:
            print(f"\nFine-tuning skipped: only {n_ft_train} train samples with "
                  f"DeltaEtrap >= {args.finetune_et_threshold}.")
        else:
            print(f"\n── Secondary fine-tuning on DeltaEtrap >= {args.finetune_et_threshold} "
                  f"({n_ft_train} train / {n_ft_val} val samples) ─────────────")

            ft_lr = args.lr * args.finetune_lr_factor
            model.optimizer.learning_rate.assign(ft_lr)
            print(f"  Learning rate reset to {ft_lr:.2e}")

            ft_patience  = max(5, args.patience // 2)
            ft_callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=ft_patience,
                    min_delta=1e-8,
                    restore_best_weights=True,
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=0.5,
                    patience=max(2, ft_patience // 3),
                    min_delta=1e-8,
                    min_lr=1e-8,
                    verbose=1,
                ),
                keras.callbacks.CSVLogger(
                    str(out_dir / "history_finetune.csv")
                ),
            ]
            if use_wandb:
                import wandb
                from wandb.integration.keras import WandbMetricsLogger
                ft_callbacks.append(WandbMetricsLogger(log_freq="epoch"))

            ft_val_data = (
                (X_val[ft_mask_val], y_val[ft_mask_val])
                if n_ft_val >= 4
                else (X_val, y_val)        # fall back to full val set
            )

            model.fit(
                X_train[ft_mask_tr], y_train[ft_mask_tr],
                batch_size      = max(16, args.batch_size // 2),
                epochs          = args.finetune_epochs,
                shuffle         = True,
                validation_data = ft_val_data,
                callbacks       = ft_callbacks,
                verbose         = 1,
            )
            print("Fine-tuning complete.")
    elif et_col is None:
        print("\nFine-tuning skipped: no DeltaEtrap parameter found in param_names.")

    # ── merge training + fine-tuning history ─────────────────────────────────
    hist_main_path = out_dir / "history.csv"
    hist_ft_path   = out_dir / "history_finetune.csv"
    if hist_ft_path.exists():
        df_main = pd.read_csv(hist_main_path)
        df_ft   = pd.read_csv(hist_ft_path)
        epoch_offset = len(df_main)
        df_ft = df_ft.copy()
        df_ft["epoch"] = df_ft["epoch"] + epoch_offset
        df_combined = pd.concat([df_main, df_ft], ignore_index=True)
    else:
        df_combined = pd.read_csv(hist_main_path)
    hist_complete_path = out_dir / "history_complete.csv"
    df_combined.to_csv(hist_complete_path, index=False)
    print(f"  Complete training history saved: {hist_complete_path}")

    # ── save loss history CSV (epoch, loss, val_loss) ─────────────────────────
    loss_cols = [c for c in ["epoch", "loss", "val_loss"] if c in df_combined.columns]
    df_combined[loss_cols].to_csv(out_dir / "loss_history.csv", index=False)
    print(f"  Loss history saved: {out_dir / 'loss_history.csv'}")

    # ── save model ───────────────────────────────────────────────────────────
    model_path = out_dir / "model.keras"
    model.save(model_path)
    print(f"\nModel saved: {model_path}")

    # ── evaluate on all splits ────────────────────────────────────────────────
    test_loss, test_mse, test_mae = model.evaluate(X_test,  y_test,  verbose=0)
    val_loss,  val_mse,  val_mae  = model.evaluate(X_val,   y_val,   verbose=0)
    train_loss,train_mse,train_mae= model.evaluate(X_train, y_train, verbose=0)

    # R² in raw (log10 tau_diff) space
    y_train_pred_raw = output_scaler.inverse_transform(
        model.predict(X_train, batch_size=512, verbose=0)
    )
    y_val_pred_raw   = output_scaler.inverse_transform(
        model.predict(X_val,   batch_size=512, verbose=0)
    )
    y_test_pred_raw  = output_scaler.inverse_transform(
        model.predict(X_test,  batch_size=512, verbose=0)
    )

    r2_train = compute_r2(y_train_raw, y_train_pred_raw)
    r2_val   = compute_r2(y_val_raw,   y_val_pred_raw)
    r2_test  = compute_r2(y_test_raw,  y_test_pred_raw)

    print(f"\nTrain loss={train_loss:.6f}  mse={train_mse:.6f}  mae={train_mae:.6f}  R²={r2_train:.6f}")
    print(f"Val   loss={val_loss:.6f}  mse={val_mse:.6f}  mae={val_mae:.6f}  R²={r2_val:.6f}")
    print(f"Test  loss={test_loss:.6f}  mse={test_mse:.6f}  mae={test_mae:.6f}  R²={r2_test:.6f}")

    if use_wandb:
        import wandb
        wandb.log({
            "train_loss": train_loss, "train_mse": train_mse, "train_mae": train_mae, "train_r2": r2_train,
            "val_loss":   val_loss,   "val_mse":   val_mse,   "val_mae":   val_mae,   "val_r2":   r2_val,
            "test_loss":  test_loss,  "test_mse":  test_mse,  "test_mae":  test_mae,  "test_r2":  r2_test,
        })

    # ── confusion matrices for train / val / test sets ───────────────────────
    confusion_dir = out_dir / "confusion_matrices"

    for split_label, X_raw, y_raw, y_pred_raw in [
        ("train", X_train_raw, y_train_raw, y_train_pred_raw),
        ("val",   X_val_raw,   y_val_raw,   y_val_pred_raw),
        ("test",  X_test_raw,  y_test_raw,  y_test_pred_raw),
    ]:
        print(f"\nComputing confusion matrices on {split_label} set …")
        cm_results = compute_confusion_matrices(
            X_val_raw   = X_raw,
            y_val_raw   = y_raw,
            y_pred_raw  = y_pred_raw,
            param_names = param_names,
            n_bins      = args.confusion_bins,
        )
        save_confusion_matrices(cm_results, confusion_dir, split_label=split_label)

    # ── save training details ─────────────────────────────────────────────────
    best_val_loss = min(history.history["val_loss"])
    results_dict = {
        "best_val_loss": float(best_val_loss),
        "train_loss":    float(train_loss),
        "train_mse":     float(train_mse),
        "train_mae":     float(train_mae),
        "r2_train":      float(r2_train),
        "val_loss":      float(val_loss),
        "val_mse":       float(val_mse),
        "val_mae":       float(val_mae),
        "r2_val":        float(r2_val),
        "test_loss":     float(test_loss),
        "test_mse":      float(test_mse),
        "test_mae":      float(test_mae),
        "r2_test":       float(r2_test),
    }
    save_training_details(
        path          = out_dir / "training_details.json",
        model         = model,
        args          = args,
        n_train       = len(X_train),
        n_val         = len(X_val),
        n_test        = len(X_test),
        input_dim     = INPUT_DIM,
        output_dim    = OUTPUT_DIM,
        epochs_trained= epochs_trained,
        results       = results_dict,
    )

    # ── save metrics CSV ─────────────────────────────────────────────────────
    save_metrics_csv(out_dir / "metrics_summary.csv", results_dict)

    # ── plot sample test curves ───────────────────────────────────────────────
    qfls_test_arr = d.get("qfls_test")
    if qfls_test_arr is not None:
        plot_test_curves(
            y_true_raw = y_test_raw,
            y_pred_raw = y_test_pred_raw,
            qfls       = qfls_test_arr,
            out_path   = out_dir / "test_curves_sample.png",
        )
    else:
        print("  Warning: qfls_test not found in HDF5 – skipping curve plot.")

    # ── inference timing ──────────────────────────────────────────────────────
    timing_seq_path   = out_dir / "nn_inference_times_sequential.csv"
    timing_batch_path = out_dir / "nn_inference_times_batch.csv"
    n_timing = min(200, len(X_test))
    print(f"\nMeasuring inference times on {n_timing} test samples …")
    measure_inference_sequential(model, X_test[:n_timing], timing_seq_path)
    measure_inference_batch(model, X_test[:n_timing], timing_batch_path)

    # ── final summary ────────────────────────────────────────────────────────
    print(f"\nBest val loss      : {best_val_loss:.6f}")
    print(f"R² train / val / test : {r2_train:.4f} / {r2_val:.4f} / {r2_test:.4f}")
    print(f"Confusion matrices : {confusion_dir}")
    print("Done.")

    if use_wandb:
        import wandb
        wandb.finish()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train TRPL Conv1DTranspose surrogate network.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("training_hdf5", help="Path to training HDF5 (from prepare_training_data.py)")
    p.add_argument("--output-dir",         default=None,
                   help="Output directory (default: timestamped folder next to HDF5)")
    p.add_argument("--epochs",             type=int,   default=300)
    p.add_argument("--batch-size",         type=int,   default=128,
                   help="Mini-batch size for training")
    p.add_argument("--patience",           type=int,   default=25,
                   help="Early-stopping patience in epochs (ReduceLROnPlateau uses patience//3)")

    # ── Conv1DTranspose architecture ──────────────────────────────────────────
    p.add_argument("--num-conv-layers",  type=int,   default=2,
                   help="Number of Conv1DTranspose upsampling layers")
    p.add_argument("--max-filter",       type=int,   default=128,
                   help="Filters in the first Dense seed layer (halved each conv layer)")
    p.add_argument("--kernel-size",      type=int,   default=16,
                   help="Conv1D kernel size (larger → smoother output)")
    p.add_argument("--dropout-rate",     type=float, default=0.0,
                   help="Dropout rate applied after each conv layer (0 = disabled)")
    p.add_argument("--batch-norm",       action="store_true", default=False,
                   help="Enable BatchNormalization layers")

    # ── optimizer ─────────────────────────────────────────────────────────────
    p.add_argument("--lr",                 type=float, default=1e-3)
    p.add_argument("--decay-rate",         type=float, default=0.995,
                   help="ExponentialDecay rate per decay_steps (=steps/epoch)")

    # ── secondary fine-tuning ────────────────────────────────────────────────
    p.add_argument("--no-finetune",          action="store_true", default=False,
                   help="Disable secondary fine-tuning phase on high-Etrap samples")
    p.add_argument("--finetune-et-threshold",type=float, default=0.8,
                   help="DeltaEtrap threshold for selecting fine-tuning samples")
    p.add_argument("--finetune-lr-factor",   type=float, default=0.1,
                   help="LR multiplier for fine-tuning relative to --lr (default 0.1 → lr/10)")
    p.add_argument("--finetune-epochs",      type=int,   default=100,
                   help="Max epochs for fine-tuning phase")

    p.add_argument("--confusion-bins",     type=int,   default=20,
                   help="Number of bins per axis in confusion matrices")
    p.add_argument("--wandb",              action="store_true", default=False,
                   help="Enable Weights & Biases logging (default: off)")
    p.add_argument("--no-wandb",           dest="wandb", action="store_false",
                   help="Disable Weights & Biases logging")
    p.add_argument("--wandb-project",      default="trpl-nn-training")
    p.add_argument("--wandb-run-name",     default=None)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
