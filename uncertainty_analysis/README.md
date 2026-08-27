# TRPL/SSPL Bayesian Inference Pipeline

Four-step pipeline for inferring trap parameters from TRPL and SSPL measurements using a neural-network surrogate (for TRPL) and an analytic/numeric physics solver (for SSPL).

```
01_create_grid.py  →  02_trpl_errors.py  →  03_sspl_errors.py  →  04_bayesian.py
   (parameter grid)     (NN TRPL errors)      (physics SSPL errors)   (posterior + plots)
```

Support modules (not run directly):

| File | Purpose |
|---|---|
| `trpl_module.py` | SRH transient ODE solver — shared copy from `fitting/trpl_module.py` |
| `constants.py` | Physical constants (NC, NV, VT, …) — shared copy from `fitting/constants.py` |
| `trpl_fitting.py` | NN artefact loader (`load_nn_artifacts`) — used internally by `02_trpl_errors.py` |
| `config-paper.json` | Example config for a 1-trap real measurement |
| `config-paper-1trap-syn.json` | Example config for a 1-trap synthetic dataset |
| `config-paper-2trap_ye-genghua.json` | Example config for a 2-trap real measurement |

---

## Model error analysis — `model-error-analysis/`

This subfolder contains the calibration analysis that determines the homoskedastic
sigma (`sigma_trpl`) used in the Bayesian log-likelihood (step 04).

The 0D point model at the core of the pipeline (uniform carrier generation)
is an approximation of the true 1D spatial problem.  Real TRPL measurements
are excited by a laser pulse that creates a non-uniform carrier profile through
the film via Beer–Lambert absorption.  The sensitivity of τ_diff to the optical
absorption coefficient α quantifies how large this approximation error is.

### Method

A 1D spatial PDE model (`pdex1_sproul_Chris_frequency.m`, MATLAB) is solved for
five physical scenarios at five α values (10 000 – 1 000 000 cm⁻¹), spanning
realistic absorption regimes:

| Folder | Physical scenario |
|---|---|
| `simulation_results-shallow-plateau_alphas-mu-1` | Shallow trap with τ_diff plateau feature |
| `simulation_results-shallow_photodoping_alphas-mu-1` | Shallow trap with photodoping |
| `simulation_results-no-photodopgin_alphas-mu-1` | Baseline without photodoping |
| `simulation_results-trap-high-Nt_alphas-mu-1` | High trap density |
| `simulation_results-trap-low-Nt_alphas-mu-1` | Low trap density |

Each CSV in these folders contains columns `time_s`, `PL_norm`, `QFLS_eV`,
and `tau_diff_s` for a single α value.

### Sigma derivation (`homoskedastic_sigma_derivative_alpha.m`)

The MATLAB script computes the centred finite-difference derivative
`d log₁₀(τ_diff) / d log₁₀(α)` at every QFLS point on a shared 256-point
grid (QFLS ∈ [0.90, 1.49] eV).  Assuming α is uncertain by a factor of 10
(i.e. `Δ log₁₀(α) = 1`), the pointwise uncertainty in log₁₀(τ_diff) is:

```
σ(QFLS) = |d log₁₀(τ_diff) / d log₁₀(α)| × 1
```

Three estimates of the scalar homoskedastic sigma are computed:

| Method | Formula | Value |
|---|---|---|
| RMS (chosen) | √(mean(σ²)) | **0.4936** |
| Mean | mean(σ) | 0.3287 |
| Sample | √(Σσ²/(N−1)) | 0.4946 |

The RMS value **σ_trpl = 0.4936** is used as `bayesian.sigma_trpl` in all
config files and is plugged into the effective sigma of the Gaussian
log-likelihood in step 04:

```
sigma_eff = sqrt(sigma_trpl² + MSE_grid_min)
```

### Outputs (`derivative_analysis_output/`)

| File | Contents |
|---|---|
| `analysis_summary.txt` / `.csv` | Physical parameters, all three sigma estimates, chosen value |
| `mean_derivative_per_folder.csv` | Per-scenario mean derivative curve on the 256-point QFLS grid |
| `mean_derivative_all_folders.csv` | Overall mean derivative across all scenarios |
| `sigma_per_folder.csv` | Per-scenario σ(QFLS) curves |
| `sigma_all_folders.csv` | Overall σ(QFLS) curve |
| `sigma_homoskedastic_summary.csv` | RMS / Mean / Sample sigma values |
| `deriv_per_file_<folder>.csv` | Per-α-file derivative curves for each scenario |
| `sigma_per_file_<folder>.csv` | Per-α-file σ curves for each scenario |
| `derivatives_<folder>.png` | Derivative plot per scenario |
| `sigma_<folder>.png` | σ plot per scenario |
| `summary_mean_derivative.png` | All-scenario overlay of mean derivatives |
| `summary_sigma.png` | All-scenario overlay of σ curves |

---

## Quick Start

```bash
cd pipeline/

# 1. Create the parameter grid
python 01_create_grid.py --config config.json

# 2. Compute TRPL errors via neural network (GPU)
python 02_trpl_errors.py --config config.json

# 3. Compute SSPL errors via physics solver (CPU)
python 03_sspl_errors.py --config config.json

# 4. Bayesian inference and corner plot
python 04_bayesian.py --config config.json --mode joint
```

All outputs go to `output_dir` specified in config.

---

## Input Data Formats

### TRPL CSV (`data.trpl_csv`)
Must have at minimum these two columns. Either naming convention is accepted:

| Accepted name | Alternative | Description |
|---|---|---|
| `qfls_eV` | `qfls` | Quasi-Fermi level splitting (eV) |
| `tau_diff_s` | `tau_diff` | Differential carrier lifetime (s) |

The script prefers the `_eV` / `_s` suffixed form and falls back to the unsuffixed form automatically. The file is loaded with `index_col=0`, so the first column (e.g. `time_ns`) becomes the row index.

### SSPL CSV (`data.sspl_csv`)
Must have at minimum:

| Column | Description |
|---|---|
| `qfls_eV` | Quasi-Fermi level splitting (eV) |
| `plqy` | Photoluminescence quantum yield (dimensionless, 0–1) |

---

## Config Reference (`config.json`)

### Top-level fields

| Field | Type | Description |
|---|---|---|
| `run_name` | string | Human-readable label, written into metadata files |
| `output_dir` | string | Directory for all outputs. Relative to the config file location |

---

### `physics` — Material parameters

| Field | Example | Description |
|---|---|---|
| `Eg_eV` | `1.625` | Band gap in eV |
| `npulse_cm3` | `1.6e17` | Photoexcited carrier density for TRPL (cm⁻³) |
| `NC_cm3` | `2.21359e18` | Effective density of states in conduction band (cm⁻³) |
| `NV_cm3` | `2.21359e18` | Effective density of states in valence band (cm⁻³) |
| `T_K` | `300` | Temperature (K) |
| `num_traps` | `1` | Number of trap levels. Controls how many trap parameter sets appear in grid/results |

---

### `nn` — Neural network artifacts

All paths are relative to the config file location.

| Field | Description |
|---|---|
| `model_path` | Path to the `.keras` model file |
| `param_scaler_path` | Path to the `.joblib` input (parameter) scaler |
| `output_scaler_path` | Path to the `.joblib` output (TRPL) scaler |
| `training_hdf5_path` | Path to the training HDF5 dataset (used to read the QFLS axis) |

---

### `data` — Experimental data

| Field | Description |
|---|---|
| `trpl_csv` | Path to the TRPL data CSV (see format above) |
| `sspl_csv` | Path to the SSPL data CSV (see format above) |

Paths are relative to the config file location.

---

### `best_fit` — Optimizer result (required for `slice` grid mode)

| Field | Description |
|---|---|
| `json_path` | Path to a `*_parameters.json` file produced by a prior NN optimizer run |

Expected JSON structure:
```json
{
  "traps": [
    { "Et_eV": 1.23, "Nt_cm3": 8.8e16, "tau_n_s": 1.0e-7, "tau_p_s": 1.0e-6 }
  ],
  "krad_cm3s": 1.2e-11
}
```

For `num_traps > 1`, add one object per trap to the `traps` array.

The best-fit values are shown as **orange dotted vertical lines** on the diagonal panels of the corner plot and are used to anchor both the 1-D and 2-D sigma regions:
- **1-D (diagonal panels):** the 1σ interval is the smallest contiguous region that contains the best-fit point and encloses 68.27 % of the posterior mass.
- **2-D (off-diagonal panels and contour CSVs):** each sigma contour threshold is `min(T_hdr, P_bf)`, where `T_hdr` is the standard HDR threshold and `P_bf` is the posterior probability at the best-fit grid point, ensuring the best-fit always falls inside the 1σ contour.

---

### `grid` — Parameter space settings

| Field | Example | Description |
|---|---|---|
| `mode` | `"full"` | `"full"` = Cartesian product over all parameter ranges. `"slice"` = 2-D cross-sections + 1-D profiles anchored at the best-fit point. Can be overridden with `--mode` flag |
| `chunk_size` | `100000` | Rows per chunk when writing/reading the grid CSV. Tune to fit in RAM |
| `slice_n_points` | `80` | Unused by current scripts; kept for legacy compatibility |
| `traps` | array | One object per trap (length must equal `physics.num_traps`) |
| `krad_bounds_cm3s` | `[1e-12, 1e-10]` | Search bounds for radiative recombination coefficient |
| `krad_n_points` | `10` | Number of grid points for `krad` |

**Per-trap fields** (inside each element of `traps`):

| Field | Example | Description |
|---|---|---|
| `Et_bounds_eV` | `[0.8125, 1.625]` | Trap energy level range. Typically `[Eg/2, Eg]` for shallow traps or `[0, Eg]` for full range |
| `Et_n_points` | `50` | Grid points for `Et`. Linear spacing |
| `Nt_bounds_cm3` | `[1e12, 2.21e18]` | Trap density range |
| `Nt_n_points` | `50` | Grid points for `Nt`. Log spacing |
| `taun_bounds_s` | `[1e-12, 1e-4]` | Electron capture lifetime range |
| `taun_n_points` | `50` | Grid points for `tau_n`. Log spacing |
| `taup_bounds_s` | `[1e-12, 1e-4]` | Hole capture lifetime range |
| `taup_n_points` | `50` | Grid points for `tau_p`. Log spacing |

> **Grid size warning:** For 1 trap with 50 points each and 10 krad points, the full grid is 50⁴ × 10 = 62.5 million rows. Reduce `_n_points` if memory or runtime is a concern.

---

### `sspl` — SSPL physics solver settings

| Field | Example | Description |
|---|---|---|
| `n_sweep_lo` | `1e8` | Lower bound of the carrier density sweep (cm⁻³) |
| `n_sweep_hi` | `2.21359e18` | Upper bound, typically `NV_cm3` |
| `n_sweep_n` | `80` | Number of points in the carrier density sweep. Log spacing |
| `solver` | `"analytic"` | `"analytic"` = fast quadratic formula, valid only for `num_traps == 1`. `"numeric"` = fixed-point iteration, works for any number of traps. Automatically falls back to `"numeric"` when `num_traps > 1` |
| `compare` | `"plqy_vs_qfls"` | Which quantity to compare and which x-axis to use. See table below |

**`compare` modes:**

| Value | x-axis | y-axis / error metric | Required CSV columns |
|---|---|---|---|
| `"plqy_vs_qfls"` (default) | QFLS (eV, linear) | log₁₀(PLQY) | `qfls_eV`, `plqy` |
| `"plqy_vs_G"` | log₁₀(G) (cm⁻³s⁻¹) | log₁₀(PLQY) | `G_cm3s`, `plqy` |
| `"qfls_vs_G"` | log₁₀(G) (cm⁻³s⁻¹) | QFLS (eV, linear) | `G_cm3s`, `qfls_eV` |

In all modes the MSE is computed in the y-axis space (log₁₀ for PLQY, linear eV for QFLS). The generation rate `G` is always treated in log₁₀ for interpolation. The simulated generation rate is `G = R_rad + R_SRH` (total recombination rate in steady state).

---

### `plot` — Corner plot appearance

| Field | Example | Description |
|---|---|---|
| `colorscale_levels` | `30` | Number of contour levels in all 2-D panels. More levels = smoother gradient |
| `colorscale_vmin` | `null` | Minimum probability value for the color scale. `null` = use 0 |
| `colorscale_vmax` | `null` | Maximum probability value for the color scale. `null` = use the global maximum across all 2-D panels in the corner plot |

All 2-D panels in a single corner plot share the same color scale and a single colorbar is placed on the right side of the figure. Setting `colorscale_vmax` to a fixed value (e.g. `0.005`) is useful when comparing corner plots across runs, so the color mapping is identical between figures.

---

### `bayesian` — Posterior computation settings

| Field | Example | Description |
|---|---|---|
| `sigma_trpl` | `0.4936` | Base NN uncertainty (σ) for TRPL in log₁₀ space. Added in quadrature with the grid MSE minimum to give an effective σ |
| `sigma_sspl` | `0.0340` | Base NN uncertainty for SSPL in log₁₀ space |
| `chunk_size` | `1000000` | Rows per chunk when streaming the error CSVs in step 04 |

The effective sigma used in the Gaussian log-likelihood is:

```
sigma_eff = sqrt(sigma_nn² + MSE_grid_min)
```

where `MSE_grid_min` is the minimum MSE found across the entire grid. This prevents the posterior from being infinitely sharp when the best-fit grid point has near-zero error.

---

### `hardware` — GPU settings (used by step 02)

| Field | Example | Description |
|---|---|---|
| `cuda_device` | `"1"` | Which GPU to use (`CUDA_VISIBLE_DEVICES`). Use `"0"` for the first GPU |
| `gpu_memory_frac` | `0.5` | Fraction of GPU memory to allocate (0–1) |
| `batch_size` | `8192` | NN inference batch size. Increase for faster throughput if VRAM allows |

---

### `true_params` — Ground-truth values (optional)

Set to `null` for real measurements. For synthetic data where the true parameters are known, provide them here and they will be plotted as **red dashed lines** in the corner plot, and the `credible_intervals.csv` will record whether each true value falls within the 1σ credible interval.

**Keys for `num_traps = 1`:**

```json
"true_params": {
  "Et_1_eV":   1.23,
  "Nt_1_cm-3": 8.83e16,
  "tau_n_1_s": 1.02e-7,
  "tau_p_1_s": 1.04e-6,
  "krad_cm3s": 1.22e-11
}
```

**Keys for `num_traps = 2`** — append a second set with index `_2`:

```json
"true_params": {
  "Et_1_eV":   1.23,  "Nt_1_cm-3": 8.83e16, "tau_n_1_s": 1e-7, "tau_p_1_s": 1e-6,
  "Et_2_eV":   0.90,  "Nt_2_cm-3": 1e15,    "tau_n_2_s": 5e-8, "tau_p_2_s": 5e-7,
  "krad_cm3s": 1.22e-11
}
```

Trap indices are **1-based** (first trap is `_1`, second is `_2`, etc.). Do not use `_0`.

---

## Script Reference

### `01_create_grid.py`

Creates the parameter grid CSV(s).

```bash
python 01_create_grid.py --config config.json [--mode full|slice]
```

| Flag | Default | Description |
|---|---|---|
| `--config` | `config.json` | Path to config file |
| `--mode` | from config | Override `grid.mode` |

**Outputs (`full` mode):**
- `<output_dir>/grid.csv` — all parameter combinations
- `<output_dir>/grid_metadata.json` — axis values and spacing

**Outputs (`slice` mode):**
- `<output_dir>/grid_slices.csv` — 2-D cross-sections (column `slice_pair` identifies each panel)
- `<output_dir>/grid_1d.csv` — 1-D profiles (column `varied_param` identifies each)
- `<output_dir>/grid_metadata.json`

In `slice` mode, the grid is anchored so that the best-fit point (from `best_fit.json_path`) lands exactly on a grid node.

---

### `02_trpl_errors.py`

Runs every grid row through the neural network and computes TRPL MSE vs experiment.

```bash
python 02_trpl_errors.py --config config.json [--grid PATH] [--out PATH]
```

| Flag | Default | Description |
|---|---|---|
| `--config` | `config.json` | Path to config file |
| `--grid` | `<output_dir>/grid.csv` | Input grid CSV. Pass `grid_slices.csv` or `grid_1d.csv` for slice mode |
| `--out` | `<output_dir>/errors_trpl.csv` | Output path |

**Output:** same columns as input grid + `error_trpl_mse` + `error_trpl_integral_mse`.

The MSE is computed in log₁₀(τ) space after interpolating the NN output to the experimental QFLS points.

---

### `03_sspl_errors.py`

Computes SSPL MSE vs experiment using the physics solver (no GPU needed).

```bash
python 03_sspl_errors.py --config config.json [--grid PATH] [--out PATH]
```

| Flag | Default | Description |
|---|---|---|
| `--config` | `config.json` | Path to config file |
| `--grid` | `<output_dir>/grid.csv` | Input grid CSV |
| `--out` | `<output_dir>/errors_sspl.csv` | Output path |

**Output:** same columns as input grid + `error_sspl_mse`.

The MSE is in log₁₀(PLQY) space.

---

### `04_bayesian.py`

Computes the Bayesian posterior from the error CSVs and generates corner plots and export files.

```bash
python 04_bayesian.py --config config.json --mode joint [--grid-mode full|slice] \
    [--trpl-errors PATH] [--sspl-errors PATH] [--trpl-1d PATH] [--sspl-1d PATH]
```

| Flag | Default | Description |
|---|---|---|
| `--config` | `config.json` | Path to config file |
| `--mode` | `joint` | `trpl` = TRPL only. `sspl` = SSPL only. `joint` = both combined |
| `--grid-mode` | from config | Override `grid.mode`. Use `slice` when errors were computed on slice grids |
| `--trpl-errors` | `<output_dir>/errors_trpl.csv` | TRPL error CSV (2-D slices for slice mode) |
| `--sspl-errors` | `<output_dir>/errors_sspl.csv` | SSPL error CSV (2-D slices for slice mode) |
| `--trpl-1d` | none | TRPL 1-D profile errors CSV (slice mode only) |
| `--sspl-1d` | none | SSPL 1-D profile errors CSV (slice mode only) |

#### Posterior probability

The unnormalised log-posterior for a grid point θ is:

```
log P(θ | data) = -MSE / (2 · σ_eff²) − log(σ_eff √2π)
```

Probabilities are normalised so that the sum over all grid points equals 1 (discrete probability mass). Probability **density** (used for 1-D plots and the 2-D colour fill) is obtained by dividing the mass by the grid step size on each axis.

#### Corner plot visuals

**Diagonal panels (1-D marginals):**

| Element | Appearance | Description |
|---|---|---|
| PDF curve | Solid steelblue line | Marginalised posterior probability density |
| 1σ region | Blue shaded span | Smallest contiguous interval containing the optimizer best-fit that encloses 68.27 % of posterior mass |
| 1σ bounds | Dashed steelblue verticals | Lower and upper edges of the 1σ region |
| Posterior mode | Solid black vertical | Peak of the marginalised PDF |
| Optimizer best-fit | Orange dotted vertical | Value from `best_fit.json_path` (if provided) |
| True value | Red dashed vertical | Ground-truth value from `true_params` (if provided) |
| ±σ annotation | Top-right text | Asymmetric half-widths of the 1σ interval in axis units (log₁₀ for log-space parameters) |

**Off-diagonal panels (2-D marginals):**

| Element | Description |
|---|---|
| Colour fill | Probability **density** (not mass); shared viridis colour scale across all panels |
| White contour lines | 1σ / 2σ / 3σ credible regions anchored to the optimizer best-fit point. The threshold for each level is `min(T_hdr, P_bf)` where `T_hdr` is the normal HDR threshold enclosing the target fraction and `P_bf` is the posterior probability at the best-fit grid point. This guarantees the best-fit is always inside the 1σ contour — analogous to the 1-D interval anchoring |
| Red dashed lines | True-parameter cross-hairs (if `true_params` is provided) |

---

## Output Files — Detailed Reference

All files are written to `<output_dir>/bayesian_<mode>/` (e.g. `bayesian_trpl/`, `bayesian_joint/`).

### Corner plots

| File | Description |
|---|---|
| `cornerplot.png` | Full corner plot (full-grid mode) |
| `cornerplot_slice.png` | Slice corner plot (slice mode) |

---

### 1-D marginal credible intervals

#### `hpd_intervals.csv`

Primary output for parameter uncertainties. One row per inferred parameter.

| Column | Description |
|---|---|
| `parameter` | Physical column name (e.g. `Et_1_eV`, `krad_cm3s`) |
| `mode_phys` | Posterior mode in physical units |
| `ci1s_lo_phys` | Lower bound of 1σ credible interval, physical units |
| `ci1s_hi_phys` | Upper bound of 1σ credible interval, physical units |
| `ci1s_lo_axis` | Lower bound on the plot axis (log₁₀ for log-space parameters) |
| `ci1s_hi_axis` | Upper bound on the plot axis |
| `sigma_minus_axis` | Distance from best-fit to lower bound, in axis units |
| `sigma_plus_axis` | Distance from best-fit to upper bound, in axis units |
| `fraction` | Target probability mass enclosed (always 0.6827) |
| `true_value` | Ground-truth value if `true_params` was provided, else blank |
| `true_in_ci` | `True`/`False` whether the true value falls within the 1σ interval |

The 1σ interval is the **smallest contiguous interval** that (a) contains the optimizer best-fit point and (b) encloses at least 68.27 % of the posterior mass. The algorithm checks all possible left boundaries and uses prefix sums to find the minimum-width solution (O(N log N)).

#### `credible_intervals.csv`

Compact version of the same information, including both physical and axis values for the mode and interval bounds. Suitable for quick inspection.

#### `modes.csv`

Minimal table with just the posterior mode and optimizer best-fit value for each parameter.

| Column | Description |
|---|---|
| `parameter` | Physical column name |
| `mode_phys` | Posterior mode in physical units (position of the black vertical line in the corner plot) |
| `best_fit_phys` | Optimizer best-fit value in physical units (position of the orange dotted line), if `best_fit.json_path` was provided |

---

### 2-D panel data

#### `panels_2d/marginal_<X>_vs_<Y>.csv` (full-grid mode)

One file per parameter pair. Contains the 2-D marginal probability table.

| Column | Description |
|---|---|
| `<X>_axis_col` | x-axis value (log₁₀ for log-space parameters) |
| `<Y>_axis_col` | y-axis value |
| `<X>_phys` | Physical x value |
| `<Y>_phys` | Physical y value |
| `probability` | Marginalised probability mass for this cell |
| `probability_density` | `probability / (Δx · Δy)` — integrates to 1 |
| `log10_P` | log₁₀(probability), for visualisation |

#### `panels_2d/slice_<X>_vs_<Y>.csv` (slice mode)

Same column layout as the marginal table above, but computed from the 2-D cross-section grid anchored at the best-fit point rather than from the full marginal.

---

### 1-D panel data

#### `panels_1d/marginal_<param>.csv` (full-grid mode)

| Column | Description |
|---|---|
| `<axis_col>` | Axis value (log₁₀ for log-space parameters) |
| `<phys_col>` | Physical value |
| `probability` | Marginalised probability mass |
| `probability_density` | `probability / Δ` — the y-axis shown in the corner plot diagonal |
| `log10_P` | log₁₀(probability) |
| `in_ci1s` | `True` if this grid point falls within the 1σ credible interval |

#### `panels_1d/margslice_<param>.csv` (slice mode, no `--trpl-1d` override)

Same layout. Produced by integrating each 2-D slice over its complementary axis and averaging across all slices that contain the parameter.

#### `panels_1d/profile_<param>.csv` (slice mode, `--trpl-1d`/`--sspl-1d` provided)

Same layout. Computed directly from the explicit 1-D profile grids instead of marginalisation.

---

### σ-contour paths for OriginLab

#### `contours/contour_<X>_vs_<Y>.csv`

One file per 2-D panel. Contains the x/y coordinates of the 1σ, 2σ, and 3σ posterior mass contour lines, ready for direct import into OriginLab or any plotting tool.

| Column | Description |
|---|---|
| `<X>_phys` | Physical x coordinate of the contour point |
| `<Y>_phys` | Physical y coordinate of the contour point |
| `sigma` | Sigma level: `1`, `2`, or `3` |

The contour levels are anchored to the optimizer best-fit point (when `best_fit.json_path` is provided): the probability threshold for each sigma level is `min(T_hdr, P_bf)`, where `T_hdr` is the HDR threshold and `P_bf` is the posterior probability at the best-fit grid point. This ensures the best-fit always lies on or inside the 1σ contour. Without `best_fit_params` the contours revert to standard HDR (mode-centred).

Disconnected segments of the same sigma level (common when the posterior has multiple lobes) are separated by a row of `NaN` values. In OriginLab, plot x vs y as a line graph grouped by `sigma`; the NaN rows automatically create gaps so segments are not spuriously connected.

---

### Run metadata

#### `summary.json`

Complete machine-readable record of the run.

```json
{
  "run_name": "...",
  "mode": "trpl",
  "grid_mode": "slice",
  "num_traps": 1,
  "sigma": {
    "sigma_nn_trpl": 0.4936,
    "sigma_eff_trpl": 0.4940,
    ...
  },
  "evidence": {
    "log_Z": -12345.6,
    "log10_Z": -5362.1
  },
  "credible_intervals": [ ... ]
}
```

#### `summary.csv`

Flat version of `summary.json` for spreadsheet import. First row contains sigma/evidence values; subsequent rows contain per-parameter credible intervals.

#### `percentile_thresholds.csv`

Records the probability density values corresponding to the best-1 %, best-2 %, best-5 %, … best-90 % percentiles, pooled across all 2-D panels. These thresholds define the colour scale boundaries used in the corner plot colour fill.

---

## Typical Workflows

### Workflow A — Full grid (brute force)

Best for an initial, unbiased search over the full parameter space.

```bash
# Set grid.mode = "full" in config.json
python 01_create_grid.py --config config.json
python 02_trpl_errors.py --config config.json
python 03_sspl_errors.py --config config.json
python 04_bayesian.py    --config config.json --mode joint
```

### Workflow B — Slice grid (refinement around best fit)

Use after a prior optimizer run to visualise the likelihood landscape around the best-fit point with finer resolution at lower cost.

```bash
# Set grid.mode = "slice" in config.json, and fill best_fit.json_path

python 01_create_grid.py --config config.json --mode slice
python 02_trpl_errors.py --config config.json --grid <output_dir>/grid_slices.csv \
    --out <output_dir>/errors_trpl_slices.csv
python 03_sspl_errors.py --config config.json --grid <output_dir>/grid_slices.csv \
    --out <output_dir>/errors_sspl_slices.csv
python 04_bayesian.py --config config.json --mode joint --grid-mode slice \
    --trpl-errors <output_dir>/errors_trpl_slices.csv \
    --sspl-errors <output_dir>/errors_sspl_slices.csv
```

To override the auto-marginalized 1-D panels with explicit 1-D profiles:

```bash
python 02_trpl_errors.py --config config.json --grid <output_dir>/grid_1d.csv \
    --out <output_dir>/errors_trpl_1d.csv
python 03_sspl_errors.py --config config.json --grid <output_dir>/grid_1d.csv \
    --out <output_dir>/errors_sspl_1d.csv
python 04_bayesian.py --config config.json --mode joint --grid-mode slice \
    --trpl-errors <output_dir>/errors_trpl_slices.csv \
    --sspl-errors <output_dir>/errors_sspl_slices.csv \
    --trpl-1d     <output_dir>/errors_trpl_1d.csv \
    --sspl-1d     <output_dir>/errors_sspl_1d.csv
```

### Workflow C — TRPL only (no SSPL data available)

```bash
python 01_create_grid.py --config config.json
python 02_trpl_errors.py --config config.json
python 04_bayesian.py    --config config.json --mode trpl
```

---

## Output Directory Structure

```
<output_dir>/
├── grid.csv                         # full grid (or grid_slices.csv + grid_1d.csv for slice)
├── grid_metadata.json
├── errors_trpl.csv                  # step 02 output
├── errors_sspl.csv                  # step 03 output
└── bayesian_joint/                  # step 04 output (one dir per --mode)
    ├── cornerplot.png               # corner plot (full-grid mode)
    ├── cornerplot_slice.png         # corner plot (slice mode)
    ├── hpd_intervals.csv            # 1σ credible intervals for all parameters
    ├── credible_intervals.csv       # compact version of hpd_intervals
    ├── modes.csv                    # posterior modes and optimizer best-fit values
    ├── percentile_thresholds.csv    # colour scale threshold values
    ├── summary.json                 # complete run metadata
    ├── summary.csv                  # flat version for spreadsheet import
    ├── panels_2d/
    │   ├── marginal_Et1_vs_Nt1.csv  # 2-D marginal probability tables
    │   └── ...
    ├── panels_1d/
    │   ├── marginal_Et1.csv         # 1-D marginal probability tables (with in_ci1s column)
    │   └── ...
    └── contours/
        ├── contour_Et1eV_vs_Nt1.csv # 1σ/2σ/3σ contour paths for OriginLab
        └── ...
```
