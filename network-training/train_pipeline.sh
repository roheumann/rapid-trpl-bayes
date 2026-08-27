#!/usr/bin/env bash
# =============================================================================
# train_pipeline.sh
#
# Full training pipeline: generate parameters → simulate → prepare → train
#
# Usage
# -----
#   bash train_pipeline.sh --n-traps N --dim D [options]
#
# Required
#   --n-traps N    Number of trap states (e.g. 1, 2, 3)
#   --dim D        Sobol exponent: 2^D parameter sets drawn before ordering
#                  filter.  Rule of thumb: DIM = 14 for 1 trap, 16 for 2 traps,
#                  18 for 3 traps (each extra trap costs ~factor N_TRAPS! in
#                  samples surviving the ordering filter).
#
# Pipeline options
#   --skip-sobol      Skip sobol_generator (use existing --sobol-file)
#   --skip-simulate   Skip generate_trpl_dataset.py
#   --skip-prepare    Skip prepare_training_data.py
#   --sobol-file PATH Existing Sobol HDF5 (required with --skip-sobol)
#
# Parameter bounds (forwarded to sobol_generator; omit to use defaults)
#   --n-pulse-min F     n_pulse lower bound [cm^-3]     (default: 1e12)
#   --n-pulse-max F     n_pulse upper bound [cm^-3]     (default: 1e18)
#   --eg-min F          Eg lower bound [eV]             (default: 1.15)
#   --eg-max F          Eg upper bound [eV]             (default: 2.35)
#   --krad-min F        k_rad lower bound [cm^3/s]      (default: 1e-12)
#   --krad-max F        k_rad upper bound [cm^3/s]      (default: 1e-10)
#   --ntrap-min F       Ntrap lower bound [cm^-3]       (default: 1e12)
#   --ntrap-max F       Ntrap upper bound [cm^-3]       (default: 1e20)
#   --delta-etrap-min F DeltaEtrap lower bound          (default: 0.5)
#   --delta-etrap-max F DeltaEtrap upper bound          (default: 0.975)
#   --taun-min F        taun lower bound [s]            (default: 1e-12)
#   --taun-max F        taun upper bound [s]            (default: 1e-4)
#   --taup-min F        taup lower bound [s]            (default: 1e-12)
#   --taup-max F        taup upper bound [s]            (default: 1e-4)
#
# Training options
#   --epochs N        Training epochs            (default: 200)
#   --batch-size N    Mini-batch size            (default: 64)
#   --patience N      Early-stopping patience    (default: 15)
#   --lr F            Initial learning rate      (default: 0.001)
#   --weight-decay F  AdamW weight decay         (default: 0.0001)
#   --wandb                       Enable Weights & Biases logging
#   --tensorboard                 Launch TensorBoard after training
#   --finetune-et-threshold F     DeltaEtrap_1 threshold for fine-tuning subset
#                                 (default: 0.8; Et_1>1.2 eV with Eg=1.625 → 0.7385)
#   --training-hdf5 PATH          Explicit path to the prepared training HDF5;
#                                 use together with --skip-sobol --skip-simulate
#                                 --skip-prepare when the HDF5 is in a prior run dir
#
# Run directory naming
#   The output directory is named:
#     <TIMESTAMP>-<SLURM_JOB_ID>-<N_TRAPS>Trap-<N_DATA>sims
#   where SLURM_JOB_ID is replaced by "local" when run outside SLURM.
#   All outputs (Sobol HDF5, simulation CSV, training HDF5, scalers, model,
#   logs, plots) are written into this single directory.
#   SLURM stdout/stderr logs are copied into the run directory at the end.
#
# Examples
#   # Full pipeline, 1 trap, ~16k samples
#   bash train_pipeline.sh --n-traps 1 --dim 14
#
#   # 2 traps, ~64k Sobol draws, W&B logging
#   bash train_pipeline.sh --n-traps 2 --dim 16 --wandb --epochs 300
#
#   # Skip Sobol generation, use existing file
#   bash train_pipeline.sh --n-traps 1 --dim 14 --skip-sobol \
#       --sobol-file path/to/1Trap_TRPL_Simulation_..._sims_16384_t_256.hdf5
# =============================================================================

set -euo pipefail
export PYTHONUNBUFFERED=1

die() { echo "ERROR: $*" >&2; exit 1; }
log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ── defaults ─────────────────────────────────────────────────────────────────
N_TRAPS=""
DIM=""
SKIP_SOBOL=0
SKIP_SIMULATE=0
SKIP_PREPARE=0
SOBOL_FILE=""
EPOCHS=200
BATCH_SIZE=64
PATIENCE=15
LR=0.001
USE_WANDB=0
LAUNCH_TB=0
FINETUNE_ET_THRESHOLD=""   # empty = use train_nn.py default (0.8)
TRAINING_HDF5_OVERRIDE=""  # explicit path; bypasses the sobol-derived path
NUM_CONV_LAYERS=""         # empty = use train_nn.py default (2)
MAX_FILTER=""              # empty = use train_nn.py default (128)
# parameter bounds (empty = use sobol_generator defaults)
N_PULSE_MIN="" ; N_PULSE_MAX=""
EG_MIN=""      ; EG_MAX=""
KRAD_MIN=""    ; KRAD_MAX=""
NTRAP_MIN=""   ; NTRAP_MAX=""
DELTA_ETRAP_MIN="" ; DELTA_ETRAP_MAX=""
TAUN_MIN=""    ; TAUN_MAX=""
TAUP_MIN=""    ; TAUP_MAX=""

# ── parse arguments ───────────────────────────────────────────────────────────
[[ $# -eq 0 ]] && { grep '^#' "$0" | head -60; exit 1; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --n-traps)      N_TRAPS="$2";       shift 2 ;;
        --dim)          DIM="$2";           shift 2 ;;
        --skip-sobol)   SKIP_SOBOL=1;       shift   ;;
        --skip-simulate)SKIP_SIMULATE=1;    shift   ;;
        --skip-prepare) SKIP_PREPARE=1;     shift   ;;
        --sobol-file)   SOBOL_FILE="$2";    shift 2 ;;
        --epochs)           EPOCHS="$2";           shift 2 ;;
        --batch-size)       BATCH_SIZE="$2";       shift 2 ;;
        --patience)         PATIENCE="$2";         shift 2 ;;
        --lr)               LR="$2";               shift 2 ;;
        --wandb)                USE_WANDB=1;                    shift   ;;
        --tensorboard)          LAUNCH_TB=1;                    shift   ;;
        --finetune-et-threshold) FINETUNE_ET_THRESHOLD="$2";   shift 2 ;;
        --training-hdf5)        TRAINING_HDF5_OVERRIDE="$2";   shift 2 ;;
        --num-conv-layers)      NUM_CONV_LAYERS="$2";          shift 2 ;;
        --max-filter)           MAX_FILTER="$2";               shift 2 ;;
        --n-pulse-min)      N_PULSE_MIN="$2";      shift 2 ;;
        --n-pulse-max)      N_PULSE_MAX="$2";      shift 2 ;;
        --eg-min)           EG_MIN="$2";           shift 2 ;;
        --eg-max)           EG_MAX="$2";           shift 2 ;;
        --krad-min)         KRAD_MIN="$2";         shift 2 ;;
        --krad-max)         KRAD_MAX="$2";         shift 2 ;;
        --ntrap-min)        NTRAP_MIN="$2";        shift 2 ;;
        --ntrap-max)        NTRAP_MAX="$2";        shift 2 ;;
        --delta-etrap-min)  DELTA_ETRAP_MIN="$2";  shift 2 ;;
        --delta-etrap-max)  DELTA_ETRAP_MAX="$2";  shift 2 ;;
        --taun-min)         TAUN_MIN="$2";         shift 2 ;;
        --taun-max)         TAUN_MAX="$2";         shift 2 ;;
        --taup-min)         TAUP_MIN="$2";         shift 2 ;;
        --taup-max)         TAUP_MAX="$2";         shift 2 ;;
        *) die "Unknown option: $1" ;;
    esac
done

# ── validate ──────────────────────────────────────────────────────────────────
[[ -z "$N_TRAPS" ]] && die "--n-traps is required"
[[ -z "$DIM"     ]] && die "--dim is required"
[[ $SKIP_SOBOL -eq 1 && -z "$SOBOL_FILE" ]] && \
    die "--skip-sobol requires --sobol-file <path>"
[[ $SKIP_SOBOL -eq 1 && ! -f "$SOBOL_FILE" ]] && \
    die "Sobol file not found: $SOBOL_FILE"

# ── base paths ────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMESTAMP=$(date '+%Y%m%d-%H%M%S')
SLURM_ID="${SLURM_JOB_ID:-local}"
TRAP_LABEL="${N_TRAPS}Trap"

# ── synthetic data directory (persistent across runs) ─────────────────────────
SYNTH_DATA_DIR="${SCRIPT_DIR}/Synthetic-Training-Data"
mkdir -p "$SYNTH_DATA_DIR"

# ── step 1: generate Sobol parameter set ─────────────────────────────────────
if [[ $SKIP_SOBOL -eq 0 ]]; then
    echo "[$(date '+%H:%M:%S')] === Step 1/5: Generating Sobol parameter set ==="

    SOBOL_ARGS=(--n-traps "$N_TRAPS" --dim "$DIM" --output-dir "$SYNTH_DATA_DIR")
    [[ -n "$N_PULSE_MIN"     ]] && SOBOL_ARGS+=(--n-pulse-min     "$N_PULSE_MIN")
    [[ -n "$N_PULSE_MAX"     ]] && SOBOL_ARGS+=(--n-pulse-max     "$N_PULSE_MAX")
    [[ -n "$EG_MIN"          ]] && SOBOL_ARGS+=(--eg-min          "$EG_MIN")
    [[ -n "$EG_MAX"          ]] && SOBOL_ARGS+=(--eg-max          "$EG_MAX")
    [[ -n "$KRAD_MIN"        ]] && SOBOL_ARGS+=(--krad-min        "$KRAD_MIN")
    [[ -n "$KRAD_MAX"        ]] && SOBOL_ARGS+=(--krad-max        "$KRAD_MAX")
    [[ -n "$NTRAP_MIN"       ]] && SOBOL_ARGS+=(--ntrap-min       "$NTRAP_MIN")
    [[ -n "$NTRAP_MAX"       ]] && SOBOL_ARGS+=(--ntrap-max       "$NTRAP_MAX")
    [[ -n "$DELTA_ETRAP_MIN" ]] && SOBOL_ARGS+=(--delta-etrap-min "$DELTA_ETRAP_MIN")
    [[ -n "$DELTA_ETRAP_MAX" ]] && SOBOL_ARGS+=(--delta-etrap-max "$DELTA_ETRAP_MAX")
    [[ -n "$TAUN_MIN"        ]] && SOBOL_ARGS+=(--taun-min        "$TAUN_MIN")
    [[ -n "$TAUN_MAX"        ]] && SOBOL_ARGS+=(--taun-max        "$TAUN_MAX")
    [[ -n "$TAUP_MIN"        ]] && SOBOL_ARGS+=(--taup-min        "$TAUP_MIN")
    [[ -n "$TAUP_MAX"        ]] && SOBOL_ARGS+=(--taup-max        "$TAUP_MAX")
    python "${SCRIPT_DIR}/sobol_generator_NTrap_Fluence-Thickness-Bandgap.py" "${SOBOL_ARGS[@]}"

    # Find the file just created in Synthetic-Training-Data/
    SOBOL_FILE=$(ls -t "${SYNTH_DATA_DIR}/${TRAP_LABEL}_TRPL_Simulation_"*.hdf5 2>/dev/null | head -1)
    [[ -z "$SOBOL_FILE" ]] && die "Sobol generator did not create an HDF5 file."
    echo "[$(date '+%H:%M:%S')] Sobol file: $SOBOL_FILE"
else
    echo "[$(date '+%H:%M:%S')] === Step 1/5: Skipping Sobol generation (--skip-sobol) ==="
    echo "[$(date '+%H:%M:%S')] Using file: $SOBOL_FILE"
fi

# ── extract N_DATA and create run directory ───────────────────────────────────
# N_DATA is encoded in the filename as "sims_<N>_t_"
N_DATA=$(basename "$SOBOL_FILE" .hdf5 | sed 's/.*_sims_\([0-9]*\)_.*/\1/')
[[ -z "$N_DATA" || "$N_DATA" == "$(basename "$SOBOL_FILE" .hdf5)" ]] && N_DATA="unknown"

RUN_NAME="${TIMESTAMP}-${SLURM_ID}-${TRAP_LABEL}-${N_DATA}sims"
RUN_DIR="${SCRIPT_DIR}/${RUN_NAME}"
mkdir -p "$RUN_DIR"

# ── redirect all remaining output into pipeline.log ───────────────────────────
exec > >(tee -a "${RUN_DIR}/pipeline.log") 2>&1

log "======================================================="
log "  TRPL Neural Network Training Pipeline"
log "======================================================="
log "  N_TRAPS    : $N_TRAPS"
log "  DIM        : $DIM  (2^DIM = $((2**DIM)) Sobol draws)"
log "  N_DATA     : $N_DATA (samples after ordering filter)"
log "  SLURM ID   : $SLURM_ID"
log "  Run dir    : $RUN_DIR"
log "======================================================="
log "  Sobol file : $SOBOL_FILE"

# Derived paths – all outputs land in RUN_DIR
SOBOL_STEM="$(basename "${SOBOL_FILE%.hdf5}")"
if [[ -n "$TRAINING_HDF5_OVERRIDE" ]]; then
    TRAINING_HDF5="$TRAINING_HDF5_OVERRIDE"
elif [[ $SKIP_PREPARE -eq 1 ]]; then
    # Training HDF5 already exists next to the sobol file (old run dir)
    TRAINING_HDF5="$(dirname "${SOBOL_FILE}")/${SOBOL_STEM}_training.hdf5"
else
    TRAINING_HDF5="${RUN_DIR}/${SOBOL_STEM}_training.hdf5"
fi

# ── step 2: run ODE simulations ──────────────────────────────────────────────
if [[ $SKIP_SIMULATE -eq 0 ]]; then
    log "=== Step 2/5: Running ODE simulations ==="
    python "${SCRIPT_DIR}/generate_trpl_dataset.py" "$SOBOL_FILE"
    # generate_trpl_dataset.py writes simulation_times.csv next to the HDF5,
    # which is already inside RUN_DIR.
    log "Simulations complete."
else
    log "=== Step 2/5: Skipping simulations (--skip-simulate) ==="
fi

# ── step 3: prepare training data ────────────────────────────────────────────
if [[ $SKIP_PREPARE -eq 0 ]]; then
    log "=== Step 3/5: Preparing training data ==="
    python "${SCRIPT_DIR}/prepare_training_data.py" \
        "$SOBOL_FILE" \
        "$TRAINING_HDF5"
    # Scaler joblib files are written next to TRAINING_HDF5, i.e. in RUN_DIR.
    log "Training data : $TRAINING_HDF5"
else
    log "=== Step 3/5: Skipping prepare (--skip-prepare) ==="
    [[ -f "$TRAINING_HDF5" ]] || die "Training HDF5 not found: $TRAINING_HDF5"
fi

# ── step 4: train neural network ─────────────────────────────────────────────
log "=== Step 4/5: Training neural network ==="

TRAIN_ARGS=(
    "$TRAINING_HDF5"
    --output-dir    "$RUN_DIR"
    --epochs        "$EPOCHS"
    --batch-size    "$BATCH_SIZE"
    --patience      "$PATIENCE"
    --lr            "$LR"
)
[[ $USE_WANDB -eq 1 ]] && TRAIN_ARGS+=(--wandb)
[[ -n "$FINETUNE_ET_THRESHOLD" ]] && TRAIN_ARGS+=(--finetune-et-threshold "$FINETUNE_ET_THRESHOLD")
[[ -n "$NUM_CONV_LAYERS"       ]] && TRAIN_ARGS+=(--num-conv-layers "$NUM_CONV_LAYERS")
[[ -n "$MAX_FILTER"            ]] && TRAIN_ARGS+=(--max-filter "$MAX_FILTER")

python "${SCRIPT_DIR}/train_nn.py" "${TRAIN_ARGS[@]}"
log "Training complete."

# ── step 5: batch inference on full dataset ───────────────────────────────────
log "=== Step 5/5: Batch inference on full dataset ==="
MODEL_FILE="${RUN_DIR}/model.keras"
python "${SCRIPT_DIR}/nn_predict_all.py" "$MODEL_FILE" "$TRAINING_HDF5" --output-dir "$RUN_DIR"
log "Batch inference complete."

# ── visualize ────────────────────────────────────────────────────────────────
log "=== Generating plots ==="
python "${SCRIPT_DIR}/visualize_training.py" "$RUN_DIR"

# ── copy SLURM logs into run directory ───────────────────────────────────────
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    SLURM_OUT="${SCRIPT_DIR}/logs/slurm_${SLURM_JOB_ID}.out"
    SLURM_ERR="${SCRIPT_DIR}/logs/slurm_${SLURM_JOB_ID}.err"
    [[ -f "$SLURM_OUT" ]] && cp "$SLURM_OUT" "${RUN_DIR}/" \
        && log "Copied SLURM stdout log → ${RUN_DIR}/slurm_${SLURM_JOB_ID}.out"
    [[ -f "$SLURM_ERR" ]] && cp "$SLURM_ERR" "${RUN_DIR}/" \
        && log "Copied SLURM stderr log → ${RUN_DIR}/slurm_${SLURM_JOB_ID}.err"
fi

# ── summary ──────────────────────────────────────────────────────────────────
log ""
log "======================================================="
log "  Pipeline complete"
log "  Results in : $RUN_DIR"
log "======================================================="
log ""
log "  Run directory contents:"
ls -lh "$RUN_DIR"
log ""
log "  To view training in TensorBoard:"
log "      tensorboard --logdir ${RUN_DIR}/logs"
log ""

# ── update training-data-info.json ───────────────────────────────────────────
TRAINING_INFO="${SCRIPT_DIR}/training-data-info.json"
log "Updating training-data-info.json …"
python3 - <<PYEOF
import json, os, sys
from datetime import datetime

entry = {
    "timestamp":    "${TIMESTAMP}",
    "run_name":     "${RUN_NAME}",
    "run_dir":      "${RUN_DIR}",
    "sobol_hdf5":   "${SOBOL_FILE}",
    "sobol_hdf5_name": os.path.basename("${SOBOL_FILE}"),
    "training_hdf5":      "${TRAINING_HDF5}",
    "training_hdf5_name": os.path.basename("${TRAINING_HDF5}"),
    "n_traps":      int("${N_TRAPS}"),
    "n_samples":    "${N_DATA}",
    "slurm_job_id": "${SLURM_ID}",
}

info_path = "${TRAINING_INFO}"
entries = []
if os.path.exists(info_path):
    try:
        with open(info_path) as f:
            entries = json.load(f)
    except Exception:
        entries = []

entries.append(entry)
with open(info_path, "w") as f:
    json.dump(entries, f, indent=2)
print(f"  training-data-info.json updated ({len(entries)} total entries)")
PYEOF

# ── optional: launch TensorBoard ─────────────────────────────────────────────
if [[ $LAUNCH_TB -eq 1 ]]; then
    log "Launching TensorBoard at http://localhost:6006  (Ctrl+C to stop)"
    tensorboard --logdir "${RUN_DIR}/logs" --port 6006
fi
