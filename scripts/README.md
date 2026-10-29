# scripts/

Command-line helpers around the notebooks. Run them from the repository root (`python scripts/<name>.py --help`).
`RUN` is a run folder written by `main.ipynb` (`<output_dir>/<run_name>_<mode>/`).

## Running the workflow

| script | what it does |
|---|---|
| `run_notebook.py` | runs a notebook headless with `KEY=VALUE` settings that replace the defaults of its first code cell |
| `start_point_study.py` | quality of the physics-informed start: CMA-ES (network) → curve_fit (rate equations) from many Sobol starts |
| `sbc_smc.py` | simulation-based calibration of a run with tempered SMC (any mode) |
| `sbc_compare.py` | SBC with network data vs rate-equation data, all modes, summary table + figures |
| `make_synthetic_1trap_85-15.py` | the synthetic 1-trap case (`data/synthetic_1trap/`, config) from the 1-trap fit of the 85:15 film |
| `sspl_truth_carriers.py` | steady-state carrier densities of the synthetic truth |

## Tables

| script | what it does |
|---|---|
| `best_fit_mse_table.py` | MSE / RMSE of the best fits per mode (tr-PL with network and rate equations, ss-PL) |
| `best_fit_loglik_table.py` | log L of the physics start point and of the best fits |
| `collect_log_evidence.py` | log Z of every SMC run in one or more run folders |
| `credible_intervals_table.py` | 68 % / 95 % credible intervals of the SMC posteriors of a synthetic case |
| `end_to_end_timing.py` | end-to-end time table (network workflow vs rate equations) |
| `timing_repeats.py`, `timing_table_mean_sd.py` | mean ± SD of wall time and evaluations over independent seeds, one table |

## Figures (matplotlib previews + HDF5 / CSV files, e.g. for Origin)

| script | what it does |
|---|---|
| `paper_slices.py` | the sliced corner of the paper through the best fit (or the truth) + simulations at those points |
| `modes_corner.py` | one HDF5 with the SMC posteriors of the tr-PL, ss-PL and joint runs (+ slice panels) |
| `render_corner_modes.py` | matplotlib version of a `corner_modes.h5` in the layout of the paper's 2-trap figure |
| `nn_vs_ode_corner.py`, `nuts_vs_smc_corner.py`, `combined_corner.py` | comparison corners (network vs rate equations, NUTS vs SMC, all samplers) |
| `ppc_ode_overlay.py` | network vs rate-equation posterior predictive in one figure |
| `posterior_point_simulations.py`, `start_point_simulations.py` | single parameter sets (posterior picks, start point) simulated with the rate equations and the ss-PL model |
| `single_trap_decomposition.py` | every trap of a 2-trap best fit on its own and in parallel |
| `raw_trpl_points.py` | the raw measured τ_diff vs QFLS points of a run (no spline) |
| `export_convergence_origin.py`, `export_error_histogram_origin.py` | start-point study as Origin worksheets |
| `export_dd_vs_0d_origin.py` | drift-diffusion vs 0D τ_diff(QFLS) (from `model-error-analysis/sigma_model.py`) for Origin |
| `export_ode_posterior_origin.py` | the rate-equation SMC posterior as Origin worksheets |
