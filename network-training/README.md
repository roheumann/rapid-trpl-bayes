# network-training/

Simulation of the training data and training of the two networks in `networks/`. The inference itself only needs
`trpl_module.py`: `inference/ode.py` loads it for every rate-equation evaluation (best-fit refinement with
`curve_fit`, checks against the network, `ode_inference.ipynb`).

| script | step |
|---|---|
| `sobol_generator_NTrap_Fluence-Thickness-Bandgap.py` | Sobol design of the parameters → HDF5 skeleton |
| `generate_trpl_dataset.py` | solves the rate equations for every sample (`trpl_module.solve_transient`, parallel) |
| `trpl_module.py` | the 0D SRH rate equations with detrapping and radiative recombination (`solve_transient`, `calcequil_general`) |
| `prepare_training_data.py` | τ_diff vs QFLS on 256 points per curve, train/val/test split, scalers → training HDF5 + `.joblib` |
| `merge_training_data.py` | merges two prepared training sets (keeps the val/test split of the first) |
| `train_conv_v2.py` | trains the Conv1D network (body as in `baseline_code_2026-10-13/train_nn.py`, cosine learning rate) |
| `train_axis_head.py` | adds the axis head (q_max offset and span of every curve) to a trained network (2-trap network) |
| `train_nn.py` | original training script (also used by the unit tests for a random-weight model) |
| `baseline_code_2026-10-13/` | the scripts that produced the original 1-trap and 2-trap data sets; `train_conv_v2.py` imports `train_nn.py` from here |

The two copies of `trpl_module.py` differ only in the start of the time integration: `baseline_code_2026-10-13/`
starts at t = t_min = 1e-11 s, the top-level one at t = 0 (`generate_trpl_dataset.py` shifts the time axis by
t_min so that both give the same time axis). The right-hand side does not depend on t explicitly. The 1-trap extra
data were simulated with the baseline copy, the 2-trap extra data with the top-level copy.

Example (2-trap extra data and network, as for `networks/2trap/`):

```bash
python sobol_generator_NTrap_Fluence-Thickness-Bandgap.py --n-traps 2 --dim 21 --scramble-seed 20261017 \
    --output-dir sobol --n-pulse-min 1e12 --n-pulse-max 1e18 --eg-min 1.15 --eg-max 2.4 --krad-min 1e-12 \
    --krad-max 1e-9 --ntrap-min 1e12 --ntrap-max 2e18 --delta-etrap-min 0.02 --delta-etrap-max 0.98 \
    --taun-min 1e-12 --taun-max 1e-4 --taup-min 1e-12 --taup-max 1e-4
python generate_trpl_dataset.py sobol/<name>.hdf5 8
python prepare_training_data.py sobol/<name>.hdf5 <name>_training.hdf5
python merge_training_data.py <first>_training.hdf5 <name>_training.hdf5 merged/<stem>.hdf5
python train_conv_v2.py merged/<stem>.hdf5 --output-dir <net> --head dense --lr 1e-3 --lr-schedule cosine \
    --epochs 200 --patience 40 --batch-size 256 --no-finetune
python train_axis_head.py <net>/<stem>.hdf5 --init-model <net>/model.keras --stem <stem> --output-dir <net_axis> \
    --head-epochs 60 --joint-epochs 40 --lambda-axis 1e-3 --huber-delta 0.5
```

## Full training data (Jülich DATA)

The training files of both networks are archived in a lossless compact form (`1trap_training_compact.h5`,
`2trap_training_compact.h5`, plain HDF5 with gzip). The compact files drop everything that can be recomputed: the
scaled arrays `X_<s>` and `y_<s>` (= scaler.transform of the raw arrays) and the QFLS axes `qfls_<s>` (= linspace
between their first and last point, plus a last-bit correction). The curves `y_<s>_raw` are stored as bit deltas
along the QFLS axis, and both scalers and a SHA-256 checksum of every original dataset are stored inside.

| script | step |
|---|---|
| `compact_training_data.py <training.hdf5> <compact.h5>` | training HDF5 → compact file |
| `restore_training_data.py <compact.h5> <training.hdf5>` | compact file → training HDF5 + `<stem>_param_scaler.joblib`, `<stem>_output_scaler.joblib`; checks every dataset against its checksum |

The restored file is bit-identical to the original in every dataset, except `param_scaler_path` / `output_scaler_path`,
which point to the restored scaler files. It is the input of `train_conv_v2.py`, `train_nn.py` and `train_axis_head.py`:

```bash
python restore_training_data.py 1trap_training_compact.h5 data/1Trap_TRPL_3609plus2e20_training.hdf5
python restore_training_data.py 2trap_training_compact.h5 data/2Trap_TRPL_3611plus2e20_training.hdf5
```

Restoring needs about 5 GB (1 trap) and 7 GB (2 traps) of disk space.
