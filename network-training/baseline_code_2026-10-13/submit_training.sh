#!/usr/bin/env bash
# =============================================================================
# submit_training.sh
#
# SLURM job script for the TRPL neural network training pipeline.
# Adjust the #SBATCH directives to match your cluster's partition names,
# GPU type, and time limits before submitting.
#
# Usage
# -----
#   # Recommended: use the wrapper so the logs/ dir is created before sbatch
#   bash submit_training.sh --n-traps 1 --dim 14
#
#   # Or submit directly (requires logs/ to already exist):
#   sbatch submit_training.sh --n-traps 2 --dim 16 --epochs 300 --wandb
#
# All arguments after the script name are forwarded verbatim to
# train_pipeline.sh, so every option documented there works here too.
#
# SLURM stdout/stderr are written to:
#   logs/slurm_<JOBID>.out  /  logs/slurm_<JOBID>.err
# and are also copied into the run directory by train_pipeline.sh at the end.
# =============================================================================

# ── Self-submitting wrapper ───────────────────────────────────────────────────
# When executed as "bash submit_training.sh ..." (i.e. not inside a SLURM job),
# create the logs/ directory and re-submit this script via sbatch.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    mkdir -p "${SCRIPT_DIR}/logs"
    exec sbatch "$0" "$@"
fi

# ── SLURM resource requests ───────────────────────────────────────────────────
#SBATCH --job-name=trpl_nn
#SBATCH --output=/data/home/ro.heumann/simulations/nn-training/logs/slurm_%j.out   # stdout (%j = job ID)
#SBATCH --error=/data/home/ro.heumann/simulations/nn-training/logs/slurm_%j.err    # stderr
#SBATCH --time=12:00:00                # wall-clock limit  HH:MM:SS
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8              # ODE simulation uses multiprocessing
#SBATCH --mem=32G
#SBATCH --gres=gpu:1                   # 1 GPU for training
#SBATCH --partition=gpu                # change to your cluster's GPU partition

# ── environment ──────────────────────────────────────────────────────────────

# Activate the pip environment
source ~/simulations/nn-training/nn_env/bin/activate
# Print environment info for reproducibility
echo "Python  : $(python --version)"
echo "TF      : $(python -c 'import tensorflow as tf; print(tf.__version__)')"
echo "Node    : $(hostname)"
echo "GPUs    : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo "Working : $(pwd)"
echo "Args    : $*"
echo "======================================================="

# ── run pipeline ─────────────────────────────────────────────────────────────
bash "${SCRIPT_DIR}/train_pipeline.sh" "$@"
