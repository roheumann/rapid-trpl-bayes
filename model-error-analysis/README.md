# model-error-analysis/

σ_model: the error of the 0D rate equations (`network-training/trpl_module.py`) relative to a drift-diffusion model
with the Beer–Lambert excitation profile. In the inference configs, σ_trPL = √(σ_model² + σ_nn²).

σ_model = RMSE of log10 τ_diff between the drift-diffusion model and the 0D rate equations, evaluated at a best fit,
on the measured QFLS points and with the QFLS calibration of the data (μ = 1 cm²/Vs).

| file | content |
|---|---|
| `dd_model.py` | drift-diffusion tr-PL model: Python port of the MATLAB model `pdex1_sproul_Chris_frequency.m` (slab, no Poisson equation) |
| `sigma_model.py` | σ_model at a best fit (`--best-fit <run>/6_best_fit.json`, `--alpha` [1/cm], `--wavelength-nm`, `--dd-only`); also writes the DD and 0D curves |
| `alpha_spectrum.py` | α(E) from the measured absorptance (`data/85-15_film/absorptance.dat`), with and without reflection correction |
| `validate_dd_1trap.py` | validation of `dd_model.py`: 0D limit and the MATLAB results in `simulation_results-*/` |
| `pdex1_sproul_Chris_frequency.m` | MATLAB drift-diffusion model (reference) |
| `homoskedastic_sigma_derivative_alpha.m`, `derivative_analysis_output/`, `alpha_vs_E_SSPL.dat` | the σ_model estimate of the original submission (five scenarios, α = 1e4–1e6 /cm), for comparison |
| `simulation_results-*/` | MATLAB drift-diffusion curves of the five scenarios at five α |

Values used in the publication:

| film | best fit | α | σ_model |
|---|---|---|---|
| 85:15 | 2-trap joint fit | 5.07e4 /cm at 630 nm (reflection-corrected, from `data/85-15_film/absorptance.dat`) | 0.145 dec |
| 77:23 | `configs/2trap_77-23_film_fit.json` | 61624 /cm at 514 nm | 0.353 dec |

```bash
python model-error-analysis/sigma_model.py --best-fit results/2trap_77-23_film/2trap_77-23_film_fit_joint/6_best_fit.json \
    --dd-only --alpha 61624.1675 --wavelength-nm 514 --no-robustness --out results/2trap_77-23_film/sigma_model
```
