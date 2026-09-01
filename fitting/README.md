# SSPL + TRPL Multitrap Fitting

Simultaneous fitting of steady-state PLQY (SSPL) and time-resolved
differential lifetime (TRPL) data to a Shockley–Read–Hall (SRH) multitrap
model using **CMA-ES** global optimisation.

---

## Project structure

```
.
├── Figure9.ipynb              ← 2-trap fitting workflow for the 85:15 film transient PL data, corresponding to Figure9 in the main paper
├── Figure10.ipynb             ← 1-trap fits enforcing a high and low trap density for the 85:15 film transient PL data, corresponding to Figure10 in the main paper
├── Figure5.ipynb               ← synthetic 1-trap test workflow, corresponding to Figure5 in the main paper
├── sstrpl_fitting.py           ← main fitting engine (fit_multitrap, plot_fit, plot_corner)
├── trpl_fitting.py             ← TRPL-only fitting (fit_trpl_multitrap); NN loader (load_nn_artifacts)
├── sspl_module.py              ← steady-state SRH solver
├── trpl_module.py              ← transient SRH solver
├── start_point_estimation_module.py  ← heuristic initial-guess estimator
├── data_utils.py               ← preprocessing helpers and export utilities
├── constants.py                ← physical constants (NC, NV, VT, …)
├── network/                    ← pre-trained NN surrogate models
│   └── <run-folder>/
│       ├── model.keras
│       ├── <prefix>_param_scaler.joblib
│       ├── <prefix>_output_scaler.joblib
│       └── <prefix>.hdf5           ← training data; stores QFLS axis & bounds
└── data/                       ← place your input data files here
```

---

## Dependencies

Install with pip (or conda):

```bash
pip install numpy pandas matplotlib scipy corner pymoo joblib
# Optional, required for spline preprocessing:
pip install torch splinetorch
# Required for interactive notebook widgets:
pip install ipywidgets ipympl
# Required for neural-network surrogate fitting:
pip install tensorflow scikit-learn
```

---

## Data format

### SSPL file (whitespace-delimited, one header row)

```
qfls   plqy
0.95   0.012
1.00   0.045
...
```

| Column | Unit | Description |
|--------|------|-------------|
| `qfls` | eV   | Quasi-Fermi level splitting |
| `plqy` | –    | Photoluminescence quantum yield (0–1) |

### TRPL file (whitespace-delimited, 7 header rows)

```
# ... 6 comment lines ...
time   pl   qfls   tau_diff
1e-9   1.0  0.95   4e-6
...
```

| Column     | Unit | Description |
|------------|------|-------------|
| `time`     | s    | Time after excitation pulse |
| `pl`       | –    | Normalised PL intensity |
| `qfls`     | eV   | Quasi-Fermi level splitting at that time |
| `tau_diff` | s    | Differential lifetime |

If your raw data is not yet in this format, use the preprocessing cells in
Step 2 of `main.ipynb`.

---

## Quick start

Open `main.ipynb` and run the cells top-to-bottom.

### Step 1 — Load data

Set the file paths and sample parameters:

```python
sspl_file_path = "data/my_sspl.dat"
trpl_file_path = "data/my_trpl.dat"
Eg     = 1.54       # bandgap [eV]
npulse = 8.9177e17  # laser-excited carrier density [cm^-3]
```

### Step 1b — Load neural-network surrogate *(optional)*

Skip this step if you want to use the ODE solver only.  Run it once at the
top of the notebook; the loaded artefacts are reused for every fit.

```python
import os
from trpl_fitting import load_nn_artifacts

NETWORK_FOLDER     = "network/<run-folder>"
MODEL_PATH         = os.path.join(NETWORK_FOLDER, "model.keras")
PARAM_SCALER_PATH  = os.path.join(NETWORK_FOLDER, "<prefix>_param_scaler.joblib")
OUTPUT_SCALER_PATH = os.path.join(NETWORK_FOLDER, "<prefix>_output_scaler.joblib")
HDF5_PATH          = os.path.join(NETWORK_FOLDER, "<prefix>.hdf5")

model, scaler_in, scaler_out, qfls_axis = load_nn_artifacts(
    MODEL_PATH, PARAM_SCALER_PATH, OUTPUT_SCALER_PATH,
    hdf5_path=HDF5_PATH,   # recommended: loads the canonical QFLS axis
)
```

`load_nn_artifacts` returns four objects:

| Return value | Description |
|--------------|-------------|
| `model` | Loaded `tf.keras.Model` |
| `scaler_in` | `StandardScaler` for the NN inputs |
| `scaler_out` | `MinMaxScaler` for the NN outputs |
| `qfls_axis` | 256-point QFLS axis [eV] read from the HDF5 file (`None` if `hdf5_path` not given) |

Pass all four to `fit_multitrap` via `nn_model`, `nn_input_scaler`,
`nn_output_scaler`, and `nn_qfls_axis` (see Step 4b).

### Step 2 — Preprocessing *(optional)*

Skip this step if your data files already contain QFLS and τ_diff columns.
Otherwise the preprocessing cells:

1. Load raw PL transients.
2. Align to the pulse peak with `hc.trpl_shift`.
3. Fit a spline to smooth noise: `utils.spline_torch`.
4. Convert (t, PL) → (QFLS, τ_diff): `utils.t_pl_to_qfls_tau`.
5. Save the result to CSV.

### Step 3 — Start-point estimation *(optional)*

The start-point estimator analyses the local curvature (ideality factor) of
the τ_diff vs. QFLS curve to detect trap regimes and estimate initial values
for Et, Nt, τ_n, and τ_p.  Providing good initial values makes CMA-ES
converge faster, but a random start also works.

```python
start_point = spe.start_point_estimator(
    trpl_data, qfls_min, qfls_max, Eg,
    trap_type="shallow",   # or "deep"
    Ntrap=1e14,
)
```

### Step 4 — Fitting

```python
results = sstrpl_fitting.fit_multitrap(
    sspl_data, trpl_data,
    Eg, npulse,
    num_traps=2,
    fit_mode='both',        # 'sspl' | 'trpl' | 'both'
    error_type='mse',       # 'mse' | 'mae' | 'integral_mae' | 'integral_mse'
    trials=1,
    maxfevals=50000,
)
```

#### Key keyword arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `fit_mode` | `'both'` | Which dataset(s) to include in the objective. `'sspl'` or `'trpl'` ignores the other dataset entirely. |
| `error_type` | `'mse'` | Error metric. `mse`/`mae` are point-wise; `integral_mae`/`integral_mse` are area-normalised integrals over QFLS — useful for unevenly-spaced data. |
| `n_interp_sspl` | `None` | If set to an integer, both the experiment and simulation are interpolated onto a uniform QFLS grid of this size before computing the SSPL error. |
| `n_interp_trpl` | `None` | Same as above for TRPL. |
| `qfls_range_sspl` | `None` | `(qfls_min, qfls_max)` — restrict the SSPL error to this QFLS window [eV]. |
| `qfls_range_trpl` | `None` | Same as above for TRPL. |
| `w_sspl` / `w_trpl` | `1.0` | Relative weight of each dataset when `fit_mode='both'`. |
| `Et_bounds` | `(Eg/2, Eg)` | Trap energy search range [eV]. Accepts a single `(lo, hi)` tuple (same for all traps) **or** a list of `(lo, hi)` tuples with one entry per trap. |
| `Nt_bounds` | `(1e12, 1e20)` | Trap density search range [cm⁻³]. Same per-trap list syntax as `Et_bounds`. |
| `tau_n_bounds` | `(1e-11, 1e-4)` | Electron SRH lifetime search range [s]. Same per-trap list syntax. |
| `tau_p_bounds` | `(1e-11, 1e-4)` | Hole SRH lifetime search range [s]. Same per-trap list syntax. |
| `krad_bounds` | `(1e-12, 1e-9)` | Radiative rate coefficient search range [cm³/s]. |
| `fixed_params` | `{}` | Fix specific parameters to constant values — they are removed from the CMA-ES search space entirely. See below. |
| `sigma` | `0.25` | CMA-ES initial step size in the normalised parameter space. |
| `maxfevals` | `50000` | Maximum objective function evaluations per trial. |
| `trials` | `1` | Number of independent CMA-ES restarts; the best result is kept. |
| `Ngrid` | `1000` | Number of carrier-density points for the steady-state solver. |
| `tspan` | `[1e-12, 1e5]` | Time span for the TRPL transient solver [s]. |
| `x0_multitrap` | `None` | Manual initial guess. Layout: `[Et_0, Nt_0, τn_0, τp_0, …, krad]`. |
| `nn_model` | `None` | Loaded Keras model from `load_nn_artifacts`. When set, replaces the ODE transient solver. |
| `nn_input_scaler` | `None` | `StandardScaler` for NN inputs (second return value of `load_nn_artifacts`). |
| `nn_output_scaler` | `None` | `MinMaxScaler` for NN outputs (third return value of `load_nn_artifacts`). |
| `nn_qfls_axis` | `None` | 256-point QFLS axis [eV] from `load_nn_artifacts`. Required for correct QFLS mapping when using the NN. |

#### Per-trap bounds

Every bound argument (`Et_bounds`, `Nt_bounds`, `tau_n_bounds`, `tau_p_bounds`) accepts either a **single tuple** (broadcast to all traps) or a **list of tuples** with one entry per trap:

```python
# Same range for both traps (old behaviour, still works)
Et_bounds = (0.5 * Eg, Eg)

# Different range per trap (new)
Et_bounds = [(0.5 * Eg, 0.85 * Eg),   # trap 0 — shallow
             (0.85 * Eg, Eg)]          # trap 1 — deep
```

#### Fixed parameters

Use `fixed_params` to pin specific parameters to known values.  Fixed parameters are **removed from the optimiser's search space** — the CMA-ES dimensionality is reduced accordingly.

```python
fixed_params = {
    0: {'Et': 0.65},                    # fix trap 0's energy level
    1: {'Nt': 1e16, 'tau_p': 2e-8},    # fix trap 1's density and hole lifetime
    'krad': 5e-11,                      # fix the radiative rate
}
```

- Keys are **0-based trap indices** (integers) or `'krad'`.
- Valid parameter names per trap: `'Et'`, `'Nt'`, `'tau_n'`, `'tau_p'`.
- Values in `x0_multitrap` for fixed parameters are accepted but ignored.

#### Initial-guess vector layout

For `num_traps = K` the vector must have length `4*K + 1`:

```
[Et_0, Nt_0, tau_n_0, tau_p_0,  Et_1, Nt_1, tau_n_1, tau_p_1,  …,  krad]
```

### Step 5 — Plot fit

```python
fig, axes = sstrpl_fitting.plot_fit(results, save_path="fits/fit.png")
```

```python
fig = sstrpl_fitting.plot_corner(results, error_threshold=1.0,
                                  save_path="fits/corner.png")
```

The corner plot shows only the **free** (optimised) parameters.  If
`fixed_params` was used, fixed parameters are omitted from the plot axes
automatically.

### Step 6 — Export results

```python
utils.export_fit_results(results, output_dir="fitting_results", prefix="2trap")
```

Files written:

| File | Contents |
|------|----------|
| `{prefix}_trpl.csv` | TRPL data vs. fit |
| `{prefix}_simulation_trpl.csv` | Full TRPL simulation (time-domain) |
| `{prefix}_sspl_fit.csv` | SSPL data vs. fit |
| `{prefix}_sspl_sim.csv` | Full steady-state simulation |
| `{prefix}_parameters.json` | All fitted parameters, bounds, and metadata |
| `{prefix}_history.npz` | CMA-ES evaluation history (X, F arrays) |
| `{prefix}_summary.txt` | Human-readable summary |

---

## Results dictionary keys

`fit_multitrap` returns a Python dictionary with these entries:

| Key | Type | Description |
|-----|------|-------------|
| `params` | DataFrame | Fitted trap parameters (Et, Nt, τ_n, τ_p) per trap |
| `krad_cm3s` | float | Fitted radiative rate coefficient [cm³/s] |
| `fit_mode` | str | Which datasets were fitted |
| `error_type` | str | Error metric used |
| `error_total` | float | Weighted total error |
| `error_sspl` | float or None | SSPL component of the error |
| `error_trpl` | float or None | TRPL component of the error |
| `sspl_sim` | DataFrame or None | Full steady-state simulation at best fit |
| `trpl_sim` | DataFrame or None | Full transient simulation at best fit |
| `sspl_fit` | DataFrame or None | SSPL data vs. fit on the comparison grid |
| `trpl_fit` | DataFrame or None | TRPL data vs. fit on the comparison grid |
| `n_evals` | int | Number of objective evaluations |
| `n_gens` | int | Number of CMA-ES generations |
| `solve_time` | float | Wall-clock optimisation time [s] |
| `history` | list or None | Full CMA-ES population history (for corner plot) |
| `x_opt` | ndarray | Best normalised free-parameter vector |
| `x0` | ndarray | Initial normalised free-parameter vector |
| `bounds` | dict | Per-trap parameter bounds used (lists of `(lo, hi)` tuples) |
| `free_indices` | list[int] | Indices into the full `4K+1` vector that were optimised |
| `fixed_params_physical` | dict | `{full_index: value}` for every fixed parameter |

### SSPL simulation columns (`results['sspl_sim']`)

| Column | Unit | Description |
|--------|------|-------------|
| `n_ss_1/cm3` | cm⁻³ | Electron density |
| `p_ss_1/cm3` | cm⁻³ | Hole density |
| `nt_ss_{i}_1/cm3` | cm⁻³ | Trapped carrier density, trap *i* |
| `Rtot_ss_1/cm3s` | cm⁻³ s⁻¹ | Total recombination rate |
| `G_ss_suns` | suns | Generation rate (= Rtot / G_per_sun) |
| `Rrad_ss_1/cm3s` | cm⁻³ s⁻¹ | Radiative recombination rate |
| `Rsrh_ss_trap{i}_1/cm3s` | cm⁻³ s⁻¹ | SRH recombination rate, trap *i* |
| `qfls_ss_eV` | eV | Quasi-Fermi level splitting |
| `PLQY_ss` | – | PLQY |
| `tau_ss_s` | s | Effective lifetime √(np)/R |
| `tau_n_ss_s` | s | Electron lifetime n/R |
| `tau_p_ss_s` | s | Hole lifetime p/R |
| `nid` | – | Local ideality factor |
| `e_trapping_ss_trap{i}_1/cm3/s` | cm⁻³ s⁻¹ | Electron trapping flux, trap *i* |
| `e_detrapping_ss_trap{i}_1/cm3/s` | cm⁻³ s⁻¹ | Electron detrapping flux, trap *i* |
| `h_trapping_ss_trap{i}_1/cm3/s` | cm⁻³ s⁻¹ | Hole trapping flux, trap *i* |
| `h_detrapping_ss_trap{i}_1/cm3/s` | cm⁻³ s⁻¹ | Hole detrapping flux, trap *i* |

---

## Neural-network surrogate fitting (optional)

`sstrpl_fitting.fit_multitrap` optionally replaces the ODE transient solver
with a pre-trained neural network via four keyword arguments:
`nn_model`, `nn_input_scaler`, `nn_output_scaler`, and `nn_qfls_axis`.
The NN evaluates orders of magnitude faster than the ODE solver and can be
combined with the steady-state SSPL solver in `fit_mode='both'`.

The NN predicts the differential lifetime τ_diff vs. QFLS curve for a given
set of physical parameters (krad, Et, Nt, τ_n, τ_p, Eg, n_pulse).
It must have been trained with compatible Eg and n_pulse values.

### NN input feature layout

| # | Feature | Unit / notes |
|---|---------|--------------|
| 1 | log₁₀(n_pulse) | cm⁻³ |
| 2 | Eg | eV (linear) |
| 3 | log₁₀(krad) | cm³/s |
| 4 | log₁₀(Nt_1) | cm⁻³ |
| 5 | Et_1 / Eg | dimensionless |
| 6 | log₁₀(τ_n1) | s |
| 7 | log₁₀(τ_p1) | s |
| … | *(repeat 4–7 for each additional trap)* | |

### Step 4b — NN fitting (single TRPL curve)

Requires Step 1b to have been run first.

```python
# Load artefacts (Step 1b) — done once per session
model, scaler_in, scaler_out, qfls_axis = load_nn_artifacts(
    MODEL_PATH, PARAM_SCALER_PATH, OUTPUT_SCALER_PATH,
    hdf5_path=HDF5_PATH,
)

# Fit — same function as ODE, just add nn_* kwargs
results = sstrpl_fitting.fit_multitrap(
    None,               # sspl_data — None when fit_mode='trpl'
    trpl_data,          # DataFrame with columns ['qfls', 'tau_diff']
    Eg=1.6,             # bandgap [eV] — must match NN training value
    npulse=1e17,        # excited carrier density [cm^-3]
    num_traps=2,
    fit_mode='trpl',    # 'trpl' or 'both' — 'sspl' raises an error with nn_model
    error_type='mse',
    trials=1,
    sigma=0.25,
    nn_model=model,
    nn_input_scaler=scaler_in,
    nn_output_scaler=scaler_out,
    nn_qfls_axis=qfls_axis,   # maps NN output to the correct QFLS grid
)

print(results['params'])
print(f"krad = {results['krad_cm3s']:.3e} cm³/s")
```

### Step 4b — NN fitting (multiple TRPL curves, simultaneous)

All curves share the same trap parameters; only `npulse` differs between them.

```python
results = sstrpl_fitting.fit_multitrap(
    None,
    [trpl_data_low, trpl_data_high],    # list of DataFrames
    Eg=1.6,
    npulse=[5e16, 2e17],                # one npulse per curve [cm^-3]
    num_traps=2,
    fit_mode='trpl',
    nn_model=model,
    nn_input_scaler=scaler_in,
    nn_output_scaler=scaler_out,
    nn_qfls_axis=qfls_axis,
)

# Per-curve errors
for i, (err, np_i) in enumerate(zip(
        results['error_trpl_per_curve'],
        results['trpl_npulse_list'])):
    print(f"Curve {i+1}: error = {err:.4f}  (npulse = {np_i:.2e})")
```

### NN results dictionary (additional keys)

| Key | Description |
|-----|-------------|
| `error_trpl_per_curve` | List of per-curve errors (length = number of TRPL curves) |
| `trpl_npulse_list` | List of npulse values used per curve |
| `trpl_sim` | DataFrame with NN prediction on its native 256-point QFLS axis |
| `trpl_sim_list` | Always-list version of `trpl_sim` (convenient for multi-curve loops) |
| `trpl_fit_list` | Always-list version of `trpl_fit` |

### Plotting the NN fit

The same `plot_fit` and `plot_corner` from `sstrpl_fitting` work unchanged:

```python
fig, axes = sstrpl_fitting.plot_fit(results)

fig = sstrpl_fitting.plot_corner(results, error_threshold=0.5)
```

---

## Start-point estimation — `determine_trap_regime`

Before calling `start_point_estimator`, use `determine_trap_regime` to
automatically identify the QFLS ranges where deep and shallow traps dominate:

```python
from start_point_estimation_module import determine_trap_regime, start_point_estimator

mask_deep, mask_shallow = determine_trap_regime(
    trpl_data,               # DataFrame with 'qfls' and 'tau_diff' columns
    window_size=3,           # sliding-window average width
    deep_nid_max_min=[7.5, 50],    # ideality-factor range for deep traps
    shallow_nid_max_min=[1.8, 2.2], # ideality-factor range for shallow traps
    contiguous_window_size=5,       # minimum consecutive points to count as a regime
)
```

`determine_trap_regime` computes the local ideality factor n_id from the slope of
log(τ_diff) vs QFLS.  Regions with n_id ∈ [7.5, 50] are flagged as deep-trap
dominated; regions with n_id ∈ [1.8, 2.2] as shallow-trap dominated.  Only
contiguous runs of at least `contiguous_window_size` points are kept.

Pass the returned boolean masks directly to `start_point_estimator`:

```python
krad, Et, Nt, taun, taup = start_point_estimator(
    trpl_data,
    trap_regime_mask=mask_shallow,  # or mask_deep
    Eg=1.6,
    trap_type="shallow",
)
```

---

## TRPL-only fitting — `fit_trpl_multitrap`

`trpl_fitting.fit_trpl_multitrap` is an older, simpler fitting function that
optimises only the TRPL differential-lifetime data using the ODE transient solver.
Use it when SSPL data is unavailable and NN surrogate speed is not needed.

```python
from trpl_fitting import fit_trpl_multitrap

results = fit_trpl_multitrap(
    df_trpl,          # DataFrame with 'qfls' [eV] and 'tau_diff' [s] columns
    Eg=1.6,           # bandgap [eV]
    npulse=1e17,      # excited carrier density [cm^-3]
    num_traps=1,
    trials=3,
    maxfevals=50000,
    Et_bounds=(0.8, 1.6),
    Nt_bounds=(1e12, 1e18),
    tau_n_bounds=(1e-10, 1e-5),
    tau_p_bounds=(1e-10, 1e-5),
    krad_bounds=(1e-12, 1e-9),
)
```

The objective minimises the area-normalised integral of
|log₁₀(τ_sim) − log₁₀(τ_data)| over QFLS.

> For simultaneous SSPL+TRPL fitting, or for neural-network surrogate fitting
> across multiple excitation densities, use `sstrpl_fitting.fit_multitrap` instead.

---

## Data utilities — `data_utils.py`

### Preprocessing

| Function | Description |
|---|---|
| `trim_and_normalize(df)` | Trim TRPL to PL peak, normalise PL to 1, shift time to 0 |
| `normalize(df)` | Same as above but preserves original absolute time values |
| `t_pl_to_qfls_tau(t_pl, n_pulse, Eg)` | Convert (time, PL) → (QFLS, τ_diff) |
| `qfls_tau_to_t_pl(df_data)` | Invert the above: (QFLS, τ_diff) → (time, PL) |
| `spline_torch(df, n_points, degree, n_pred)` | Monotone constrained B-spline smoothing in log-log space |
| `calculate_excited_carrier_concentration(λ, d, E, φ)` | Carrier density from laser pulse parameters |

### Input DataFrame requirements

**`trim_and_normalize` / `normalize`:** DataFrame with time [ns] as index and a `'pl'` column.

**`t_pl_to_qfls_tau`:** DataFrame with time [ns] as index and `'pl'` column (normalised, peak = 1).

**`qfls_tau_to_t_pl`:** DataFrame with `'qfls'` [eV] and `'tau_diff'` [s] columns, ordered from high QFLS to low QFLS.

**`spline_torch`:** DataFrame with time [ns] as index and `'pl'` column (normalised, already trimmed to peak).

### SSPL preprocessing

```python
from data_utils import preprocess_sspl

sspl_data = preprocess_sspl(
    df_raw,              # raw DataFrame with 'intensity_suns' and 'plqy' columns
    Eg=1.6,              # bandgap [eV]
    krad=1e-11,          # bimolecular rate coefficient [cm³/s]
    plqy_in_percent=False,
)
# Returns a DataFrame with 'intensity_suns', 'plqy', and 'qfls' columns
```

### Export

```python
from data_utils import export_fit_results

export_fit_results(results, output_dir="fitting_results", prefix="2trap")
```

See **Step 6 — Export results** above for the list of output files.

---

## Authors

Robin Heumann, Chris Dreessen — 2026
