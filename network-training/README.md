# TRPL Neural Network Training

Four-stage pipeline for training a neural-network surrogate that predicts TRPL
differential lifetime curves (τ_diff vs. QFLS) from physical material parameters
(trap density, energy, lifetimes, bandgap, excitation density).

The trained model and scaler artefacts can be loaded directly into the fitting
and uncertainty-analysis pipelines via `load_nn_artifacts` in `trpl_fitting.py`.

> **For full documentation see [TRAINING_MANUAL.md](TRAINING_MANUAL.md).**

---

## Pipeline overview

```
sobol_generator_NTrap_Fluence-Thickness-Bandgap.py
        ↓  Sobol-sampled parameter space → HDF5 skeleton
generate_trpl_dataset.py
        ↓  ODE physics simulation → populates HDF5
prepare_training_data.py
        ↓  train/val/test split + scalers → training HDF5 + .joblib files
train_nn.py
        ↓  Conv1DTranspose MLP → model.keras + diagnostics
```

---

## Files

| File | Description |
|---|---|
| `sobol_generator_NTrap_Fluence-Thickness-Bandgap.py` | Stage 1 — Sobol quasi-random parameter sampling |
| `generate_trpl_dataset.py` | Stage 2 — SRH ODE simulation for each parameter set (resumable) |
| `prepare_training_data.py` | Stage 3 — preprocessing, 80/10/10 split, StandardScaler + MinMaxScaler |
| `train_nn.py` | Stage 4 — model training with optional W&B logging |
| `visualize_training.py` | Plot training history and confusion matrices after training |
| `nn_predict_all.py` | Batch NN inference on a full parameter grid (GPU) |
| `nn_predict_all_cpu.py` | Same as above, CPU-only variant |
| `trpl_module.py` | SRH transient ODE solver — shared with `fitting/` and `uncertainity_analysis/` |
| `train_pipeline.sh` | Shell wrapper that runs all four stages end-to-end |
| `submit_training.sh` | SLURM job script for HPC cluster submission |

---

## Quick start

```bash
# Stage 1 — generate Sobol samples (edit N_TRAPS and DIM at the top first)
python sobol_generator_NTrap_Fluence-Thickness-Bandgap.py

# Stage 2 — run ODE simulations (resumable; pass the HDF5 path from stage 1)
python generate_trpl_dataset.py Synthetic-Training-Data/<name>.hdf5

# Stage 3 — preprocess into training dataset
python prepare_training_data.py Synthetic-Training-Data/<name>.hdf5

# Stage 4 — train
python train_nn.py Synthetic-Training-Data/<name>_training.hdf5 \
    --output-dir Results/my_run --epochs 200
```

Or run all stages at once on a cluster:

```bash
bash submit_training.sh --n-traps 1 --dim 14
```

---

## Dependencies

```bash
pip install -r requirements.txt
```

W&B logging (`--wandb` flag in `train_nn.py`) requires an optional install:

```bash
pip install wandb
```

---

## Outputs

After training, `--output-dir` contains:

| File | Content |
|---|---|
| `model.keras` | Trained Keras model (load into `fitting/` and `uncertainity_analysis/`) |
| `history.csv` / `history_complete.csv` / `history_finetune.csv` | Loss and metric values per epoch (full run and fine-tune stages) |
| `loss_history.csv` | Condensed loss summary |
| `metrics_summary.csv` | Final train / val / test metrics |
| `training_details.json` | Architecture, hyperparameters, and final results |
| `training_history.png` | Loss curve plot |
| `test_curves_sample.png` | Example predicted vs. ground-truth τ_diff curves |
| `nn_inference_times_batch.csv` | Batch NN inference timing |
| `nn_inference_times_batch_all.csv` | Batch inference timing across the full dataset |
| `nn_inference_times_sequential.csv` | Sequential (single-sample) inference timing |
| `pipeline.log` | Full console log of the training run |

The three artefact files produced by `prepare_training_data.py`
(`<stem>_training.hdf5`, `<stem>_param_scaler.joblib`, `<stem>_output_scaler.joblib`)
must remain in the same directory as `model.keras` and are required at inference time.

---

## Authors

Robin Heumann — 2026
