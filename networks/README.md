# networks/

The two trained surrogates of the publication. Each one maps the parameters
(log10 n_pulse, E_g, log10 k_rad, and log10 N_t, E_t/E_g, log10 τ_n, log10 τ_p for each trap) to
log10 τ_diff on 256 QFLS points. The configs load them with `nn.dir` and `nn.stem`
(`inference.surrogate.load_network`).

| folder | traps | internal name | training set (train / val / test) | test MSE (log10 τ_diff) | σ_nn = √MSE |
|---|---|---|---|---|---|
| `1trap/` | 1 | B2nf | 1,468,006 / 52,429 / 52,429 | 0.01262 | 0.1124 dec |
| `2trap/` | 2 | A3 (with axis head) | 1,887,422 / 104,857 / 104,858 | 0.01560 | 0.1249 dec |

Training box (the prior support of the inference): log10 n_pulse 12–18, E_g 1.15–2.4 eV, log10 k_rad −12 to −9,
log10 N_t 12–18.3, E_t/E_g 0.02–0.98, log10 τ_n and log10 τ_p −12 to −4 (s). The 2-trap training data contain only
E_t,1 < E_t,2.

## Files

| file | content |
|---|---|
| `model.keras` | the Keras model (Keras 3) |
| `<n>trap_training_param_scaler.joblib`, `<n>trap_training_output_scaler.joblib` | input (StandardScaler) and output (MinMaxScaler) scalers |
| `<n>trap_training.hdf5` | **metadata of the training set, not the training curves**: physical constants, bounds, parameter names, raw input parameters of train/val/test, and the first and last QFLS point of every training curve (the other `qfls_train` columns are empty and read as NaN). This is everything the inference reads from the training file. The full training data are in Jülich DATA |
| `surrogate_meta.json` (2trap) | axis-head metadata (q_max and span of every curve are network outputs) and the paths of scalers and HDF5, relative to the repository root |
| `training_details.json`, `history*.csv`, `metrics_summary.csv`, `axis_metrics.json`, `phase*_metrics.json` | architecture, training settings, loss curves, accuracy |
| `offline_provenance.json` | wall time of the simulations and of the training (used by the timing table of `main.ipynb`) |

The absolute paths inside `training_details.json` and `offline_provenance.json` are where the networks were trained;
they are records only.

## How they were trained

Scripts in `network-training/` (see its README).

- **1 trap (`1trap/`)**: 2^19 Sobol samples (data set "3609") + 2^20 further samples with the same bounds, simulated
  with the rate-equation solver of `network-training/baseline_code_2026-10-13/` and merged
  (`merge_training_data.py`). `train_conv_v2.py --head dense --lr 1e-3 --lr-schedule cosine --epochs 200
  --patience 40 --batch-size 256 --no-finetune`: 141 epochs, 3.3 M parameters (dense seed → 2 × UpSampling1D/Conv1D →
  dense sigmoid output).
- **2 traps (`2trap/`)**: 2^20 Sobol samples (data set "3611") + 2^20 further samples with the same bounds
  (`sobol_generator_NTrap_Fluence-Thickness-Bandgap.py --n-traps 2 --dim 21 --scramble-seed 20261017 ...`), simulated
  with `network-training/generate_trpl_dataset.py`, merged, trained with the same `train_conv_v2.py` settings, then an
  axis head was added with `train_axis_head.py --head-epochs 60 --joint-epochs 40 --lambda-axis 1e-3 --huber-delta 0.5`
  (phase 1 kept: the curve output is that of the warm-start network).
