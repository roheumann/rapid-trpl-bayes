"""
Visualize training results from a train_nn.py output directory.

Produces:
    training_history.png      – train / val loss and MAE vs epoch
    confusion_heatmap_<param>_<split>.png  – 2-D confusion heatmap per parameter
    confusion_summary_<param>.png          – mean RMSE vs parameter value (all splits)

Usage
-----
    python visualize_training.py <run_dir>

<run_dir> is the timestamped output folder created by train_nn.py or
train_pipeline.sh.
"""

import sys
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # no display needed
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import LogLocator, NullFormatter


# ─────────────────────────────────────────────────────────────────────────────
# Style
# ─────────────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.size":        11,
    "axes.titlesize":   12,
    "axes.labelsize":   11,
    "legend.fontsize":  10,
    "figure.dpi":       150,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
})

SPLIT_COLORS = {"train": "#2196F3", "val": "#FF9800", "test": "#4CAF50"}
SPLIT_LABELS = {"train": "Train", "val": "Validation", "test": "Test"}


# ─────────────────────────────────────────────────────────────────────────────
# Training history
# ─────────────────────────────────────────────────────────────────────────────

def plot_history(run_dir: Path) -> None:
    csv_path = run_dir / "history.csv"
    if not csv_path.exists():
        print(f"  history.csv not found – skipping training history plot.")
        return

    df = pd.read_csv(csv_path)

    has_mae = "mae" in df.columns and "val_mae" in df.columns
    n_panels = 2 if has_mae else 1

    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 4))
    if n_panels == 1:
        axes = [axes]

    epochs = df["epoch"] + 1 if "epoch" in df.columns else np.arange(1, len(df) + 1)

    # Loss panel
    ax = axes[0]
    ax.plot(epochs, df["loss"],     label="Train",      color=SPLIT_COLORS["train"])
    ax.plot(epochs, df["val_loss"], label="Validation", color=SPLIT_COLORS["val"], linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Training & Validation Loss")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, which="both", linestyle=":", alpha=0.5)

    # MAE panel
    if has_mae:
        ax = axes[1]
        ax.plot(epochs, df["mae"],     label="Train",      color=SPLIT_COLORS["train"])
        ax.plot(epochs, df["val_mae"], label="Validation", color=SPLIT_COLORS["val"], linestyle="--")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MAE")
        ax.set_title("Training & Validation MAE")
        ax.set_yscale("log")
        ax.legend()
        ax.grid(True, which="both", linestyle=":", alpha=0.5)

    out_path = run_dir / "training_history.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Confusion heatmaps
# ─────────────────────────────────────────────────────────────────────────────

def _param_display_name(name: str) -> str:
    """Convert internal parameter name to a readable label."""
    name = name.replace("log_", "log₁₀ ").replace("_", " ")
    return name


def plot_confusion_heatmap(matrix_csv: Path, run_dir: Path) -> None:
    """Plot a single confusion matrix CSV as a 2-D heatmap."""
    df = pd.read_csv(matrix_csv)
    if "param_center" not in df.columns:
        return

    param_centers = df["param_center"].values
    mse_centers   = np.array([float(c) for c in df.columns[1:]])
    counts        = df.iloc[:, 1:].values.astype(float)

    # normalise each row (param bin) to show probability
    row_sum = counts.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1
    prob = counts / row_sum

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.pcolormesh(
        mse_centers, param_centers, prob,
        cmap="Blues", shading="auto",
        norm=mcolors.PowerNorm(gamma=0.4, vmin=0, vmax=prob.max()),
    )
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("Fraction of samples per param bin")

    ax.set_xscale("log")
    ax.set_xlabel("MSE  (log₁₀ τ_diff)")
    ax.set_ylabel("log₁₀ parameter value")

    # extract param name and split from filename
    stem = matrix_csv.stem                      # confusion_matrix_<param>_<split>
    parts = stem.replace("confusion_matrix_", "").rsplit("_", 1)
    param_name = parts[0] if len(parts) == 2 else stem
    split_name = parts[1] if len(parts) == 2 else ""

    ax.set_title(
        f"{_param_display_name(param_name)}"
        + (f"  [{SPLIT_LABELS.get(split_name, split_name)}]" if split_name else "")
    )

    out_path = run_dir / f"{matrix_csv.stem}.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Confusion summary (RMSE vs parameter value, all splits on one plot)
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_summaries(cm_dir: Path, run_dir: Path) -> None:
    """
    For each parameter, overlay train/val/test RMSE vs param_center curves
    on one figure.
    """
    # collect all summary files, group by parameter name
    summary_files = list(cm_dir.glob("confusion_summary_*.csv"))
    if not summary_files:
        return

    # group by param name (remove _train/_val/_test suffix)
    param_groups: dict[str, dict[str, Path]] = {}
    for f in summary_files:
        stem  = f.stem.replace("confusion_summary_", "")
        parts = stem.rsplit("_", 1)
        if len(parts) == 2 and parts[1] in ("train", "val", "test"):
            param, split = parts
        else:
            param, split = stem, "val"
        param_groups.setdefault(param, {})[split] = f

    for param, split_files in param_groups.items():
        fig, ax = plt.subplots(figsize=(6, 4))

        for split in ("train", "val", "test"):
            if split not in split_files:
                continue
            df = pd.read_csv(split_files[split])
            if "param_center" not in df.columns or "rmse" not in df.columns:
                continue
            mask = df["count"] > 0
            ax.plot(
                df.loc[mask, "param_center"],
                df.loc[mask, "rmse"],
                label=SPLIT_LABELS[split],
                color=SPLIT_COLORS[split],
                marker="o", markersize=3,
                linestyle="-",
            )
            # shaded std band
            if "mean_mse" in df.columns and "std_mse" in df.columns:
                rmse  = df.loc[mask, "rmse"].values
                std   = df.loc[mask, "std_mse"].values
                low   = np.sqrt(np.maximum(0, df.loc[mask, "mean_mse"].values - std))
                high  = np.sqrt(df.loc[mask, "mean_mse"].values + std)
                ax.fill_between(
                    df.loc[mask, "param_center"],
                    low, high,
                    color=SPLIT_COLORS[split], alpha=0.15,
                )

        ax.set_xlabel(f"log₁₀ {_param_display_name(param)}")
        ax.set_ylabel("RMSE  (log₁₀ τ_diff)")
        ax.set_title(f"Prediction error vs {_param_display_name(param)}")
        ax.legend()
        ax.grid(True, linestyle=":", alpha=0.5)

        out_path = run_dir / f"confusion_summary_{param}.png"
        fig.savefig(out_path)
        plt.close(fig)
        print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(run_dir: Path) -> None:
    print(f"Visualising results in: {run_dir.resolve()}")

    # ── training history ─────────────────────────────────────────────────────
    print("\n-- Training history --")
    plot_history(run_dir)

    # ── confusion heatmaps ───────────────────────────────────────────────────
    cm_dir = run_dir / "confusion_matrices"
    if cm_dir.exists():
        print("\n-- Confusion heatmaps --")
        for f in sorted(cm_dir.glob("confusion_matrix_*.csv")):
            plot_confusion_heatmap(f, cm_dir)

        print("\n-- Confusion summaries (RMSE vs parameter) --")
        plot_confusion_summaries(cm_dir, cm_dir)
    else:
        print("  confusion_matrices/ directory not found – skipping.")

    print("\nDone.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    run_dir = Path(sys.argv[1])
    if not run_dir.is_dir():
        print(f"Error: not a directory: {run_dir}")
        sys.exit(1)

    main(run_dir)
