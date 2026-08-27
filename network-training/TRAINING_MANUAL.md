# TRPL Neural Network Training – Manual

**Author:** Robin Heumann
**Last updated:** 2026-03-10

This document describes the complete pipeline from raw physics simulations to
a trained neural network and hyperparameter search.

---

## Overview

The pipeline has four stages, each with its own script:

```
[1] sobol_generator_NTrap_Fluence-Thickness-Bandgap.py
        ↓  Synthetic-Training-Data/<name>.hdf5
[2] generate_trpl_dataset.py
        ↓  same HDF5 (populated) + <name>_simulation_times.csv
[3] prepare_training_data.py
        ↓  <name>_training.hdf5
           <name>_param_scaler.joblib
           <name>_output_scaler.joblib
[4a] train_nn.py                       ← standard training run
[4b] hyperparameter_search.py          ← Optuna + W&B search
```

---

## Dependencies

Install all required packages into your Python / conda environment:

```bash
pip install numpy scipy h5py scikit-learn joblib pandas tensorflow
pip install wandb optuna optuna-integration[wandb]
```

For GPU support follow the TensorFlow GPU installation guide for your CUDA
version.

---

## Stage 1 – Generate Sobol parameter samples

**Script:** `sobol_generator_NTrap_Fluence-Thickness-Bandgap.py`

Edit the user-settings block at the top of the file:

| Variable | Meaning | Example |
|---|---|---|
| `N_TRAPS` | Number of trap states | `2` |
| `DIM` | Sobol exponent – draws `2^DIM` samples before filtering | `18` → ~130 k valid |

Run:

```bash
python sobol_generator_NTrap_Fluence-Thickness-Bandgap.py
```

**Output:** `Synthetic-Training-Data/2Trap_TRPL_Simulation_<date>_sims_<N>_t_200.hdf5`

The ordering filter (`Etrap_1 < Etrap_2 < … < Etrap_N`) is applied
automatically.  Increase `DIM` if too few samples survive (rule of thumb:
each additional trap level reduces the surviving fraction by ~50 %).

---

## Stage 2 – Run physics simulations

**Script:** `generate_trpl_dataset.py`

```bash
python generate_trpl_dataset.py <path/to/sobol_file.hdf5>
```

The script calls `solve_transient()` (Shockley-Read-Hall ODE) for every
parameter set and writes the results (`pl`, `tau_diff`, `qfls`, `solve_time`)
back into the **same HDF5 file**.

**Resume support:** if interrupted, re-run with the same path. Samples whose
`solve_time > 0` are skipped automatically.

**Outputs written alongside the HDF5:**

| File | Columns | Description |
|---|---|---|
| `<stem>_simulation_times.csv` | `n_curves`, `cumulative_sim_time_s` | Cumulative ODE solver wall time vs number of curves – use this to benchmark against NN inference |

---

## Stage 3 – Preprocess into a training dataset

**Script:** `prepare_training_data.py`

```bash
python prepare_training_data.py <sobol_file.hdf5> [<output_training.hdf5>]
```

If the output path is omitted it is placed next to the source file with the
suffix `_training.hdf5`.

**What it does:**

1. Loads `par_mat_ln` (log₁₀-space parameters) and `tau_diff` (log₁₀(τ))
2. Removes samples with non-finite values or `solve_time == 0`
3. Splits **80 / 10 / 10** train / val / test (random seed 42, reproducible)
4. Fits a **`StandardScaler`** on the training inputs
5. Fits a **`MinMaxScaler`** on the training outputs using a single global
   [min, max] across all samples and time points – the shape of each curve
   is preserved
6. Saves both scalers as `.joblib` files and all splits (raw + scaled) into
   the training HDF5

**Outputs:**

| File | Content |
|---|---|
| `<stem>_training.hdf5` | All splits, metadata, scaler paths |
| `<stem>_param_scaler.joblib` | `StandardScaler` for the 11 input parameters |
| `<stem>_output_scaler.joblib` | `MinMaxScaler` for the 200-point τ_diff output |

Keep the three files in the **same directory** – the HDF5 stores the absolute
paths to the joblib files, and `train_nn.py` loads them from there.

---

## Stage 4a – Train the neural network

**Script:** `train_nn.py`

### Minimal run

```bash
python train_nn.py <stem>_training.hdf5
```

### Full options

```bash
python train_nn.py <stem>_training.hdf5 \
    --output-dir      Results/my_run \
    --epochs          200 \
    --batch-size      64 \
    --patience        15 \
    --num-conv-layers 4 \
    --max-filter      512 \
    --kernel-size     8 \
    --lr              1e-3 \
    --decay-rate      0.995 \
    --confusion-bins  20 \
    --wandb \
    --wandb-project   trpl-nn-training \
    --wandb-run-name  my_first_run
```

| Argument | Default | Description |
|---|---|---|
| `--epochs` | 200 | Maximum training epochs |
| `--batch-size` | 64 | Mini-batch size |
| `--patience` | 15 | Early-stopping patience (epochs without val_loss improvement) |
| `--num-conv-layers` | 4 | Number of Conv1DTranspose upsampling layers |
| `--max-filter` | 512 | Width of the dense stem and first conv layer |
| `--kernel-size` | 8 | Conv1DTranspose kernel size |
| `--lr` | 1e-3 | Initial learning rate (Adam) |
| `--decay-rate` | 0.995 | ExponentialDecay rate applied each epoch |
| `--confusion-bins` | 20 | Bins per axis in the confusion matrices |
| `--wandb` | off | Enable Weights & Biases logging |

### Model architecture

```
Input (n_params,)
  Dense(max_filter, GELU)
  Dense(max_filter × map_size, GELU)
  Reshape(map_size, max_filter)
  BatchNormalization
  [Conv1DTranspose(max_filter//k, kernel, stride=2, GELU) + BN] × (num_conv_layers−1)
  Conv1DTranspose(1, kernel, stride=2, GELU)
  Flatten
  Dense(output_dim=200, sigmoid)      ← decoupled from spatial size
Output (200,)  – MinMax-scaled log₁₀(τ_diff)
```

Loss: MSE  |  Optimizer: Adam + ExponentialDecay

### Outputs written to `--output-dir`

| File | Content |
|---|---|
| `model.keras` | Trained Keras model |
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

### Comparing NN inference vs ODE simulation

Both `<sim>_simulation_times.csv` (from stage 2) and `nn_inference_times_batch_all.csv`
contain cumulative timing data.  Plot them together:

```python
import pandas as pd, matplotlib.pyplot as plt

sim = pd.read_csv("Synthetic-Training-Data/<stem>_simulation_times.csv")
nn  = pd.read_csv("Results/my_run/nn_inference_times_batch_all.csv")

plt.plot(sim["cumulative_sim_time_s"], sim["n_curves"], label="ODE solver")
plt.plot(nn["cumulative_inference_time_s"],  nn["n_curves"],  label="Neural network")
plt.xlabel("Cumulative wall time [s]")
plt.ylabel("Number of curves generated")
plt.legend()
plt.title("Neural network vs. ODE solver throughput")
plt.tight_layout()
plt.savefig("speed_comparison.pdf")
```

---

## Stage 4b – Hyperparameter search (Optuna + W&B)

**Script:** `hyperparameter_search.py`

```bash
python hyperparameter_search.py <stem>_training.hdf5 \
    --n-trials       100 \
    --max-epochs      50 \
    --patience         7 \
    --train-fraction  0.5 \
    --study-name      trpl_2trap \
    --storage         sqlite:///hparam_study.db \
    --wandb \
    --wandb-project   trpl-hparam-search \
    --output-dir      Results/hparam_search
```

| Argument | Default | Description |
|---|---|---|
| `--n-trials` | 100 | Number of Optuna trials to run |
| `--max-epochs` | 50 | Maximum epochs per trial |
| `--patience` | 7 | Early-stopping patience per trial |
| `--train-fraction` | 0.5 | Fraction of training set used per trial (speeds up search) |
| `--study-name` | `trpl_hparam_search` | Optuna study name |
| `--storage` | *(in-memory)* | SQLite (or other) URL for a persistent study – allows resuming across runs |
| `--wandb` | off | Log every trial to W&B |

### Search space

| Hyperparameter | Type | Range |
|---|---|---|
| `num_conv_layers` | int | 3 – 6 |
| `max_filter` | categorical | 64, 128, 256, 512 |
| `kernel_size` | categorical | 4, 8, 16, 32 |
| `lr` | log-float | 1×10⁻⁴ – 1×10⁻² |
| `decay_rate` | float | 0.90 – 0.9999 |
| `batch_size` | categorical | 32, 64, 128, 256 |

Optuna uses **TPE sampling** (tree-structured Parzen estimator) and a
**Median pruner** to stop unpromising trials early.

### Resuming a search

If you used `--storage sqlite:///hparam_study.db`, re-run the same command
with the same `--study-name` and Optuna will continue from where it left off.

### Outputs

| File | Content |
|---|---|
| `best_hyperparameters.json` | Best trial number, val_loss, and all hyperparameter values |
| `all_trials.csv` | Full trial history – useful for plotting parallel-coordinate or importance plots |

### Using the best hyperparameters in a full training run

```bash
python train_nn.py <stem>_training.hdf5 \
    --output-dir    Results/best_model \
    --epochs        200 \
    --num-conv-layers 4 \
    --max-filter    256 \
    --kernel-size    8 \
    --lr            4.2e-4 \
    --decay-rate    0.993 \
    --batch-size     64 \
    --wandb
```

(Fill in the values from `best_hyperparameters.json`.)

---

## Inverting predictions after training

Load both joblib scalers and the model to go from raw parameters back to
physical τ_diff values:

```python
import numpy as np, joblib
from tensorflow import keras

model         = keras.models.load_model("Results/my_run/model.keras")
param_scaler  = joblib.load("<stem>_param_scaler.joblib")
output_scaler = joblib.load("<stem>_output_scaler.joblib")

# par_raw shape: (n_samples, n_params)  –  log10-space values from par_mat_ln
par_scaled  = param_scaler.transform(par_raw)
tau_scaled  = model.predict(par_scaled)                  # [0, 1]
tau_log10   = output_scaler.inverse_transform(tau_scaled) # log10(tau_diff / s)
tau_diff_s  = 10 ** tau_log10                            # physical units [s]
```

---

## Physical parameter layout

Columns of `par_mat_ln` (log₁₀-space, matching `param_names` in the HDF5):

| Index | Name | Physical meaning | Unit | Storage |
|---|---|---|---|---|
| 0 | `log_n_pulse` | Initial carrier density | cm⁻³ | log₁₀ |
| 1 | `Eg_eV` | Bandgap energy | eV | direct |
| 2 | `log_krad` | Radiative rate coefficient | cm³ s⁻¹ | log₁₀ |
| 3 | `log_Ntrap_1` | Trap density (trap 1) | cm⁻³ | log₁₀ |
| 4 | `DeltaEtrap_1` | Trap position (fraction of Eg) | — | direct |
| 5 | `log_taun_1` | Electron lifetime (trap 1) | s | log₁₀ |
| 6 | `log_taup_1` | Hole lifetime (trap 1) | s | log₁₀ |
| 7–10 | *(trap 2)* | Same four quantities for trap 2 | | |

For N_TRAPS > 2 the pattern repeats: indices `3 + i*4` to `3 + i*4 + 3` for
trap `i+1` (0-indexed).
