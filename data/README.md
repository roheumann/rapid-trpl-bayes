# data/

| file | content | used by |
|---|---|---|
| `85-15_film/trpl_transient.csv` | tr-PL transient of the 85:15 film: time [ns], normalised PL (no header; 90 points) | `configs/1trap_85-15_film_fit.json`, `configs/2trap_85-15_film_*.json` |
| `85-15_film/sspl_suns_qfls.dat` | ss-PL of the 85:15 film: intensity [suns] vs QFLS [eV] (tab separated, header + unit row) | same configs |
| `85-15_film/absorptance.dat` | absorptance spectrum of the 85:15 film: photon energy [eV], absorptance (comma separated) | `model-error-analysis/sigma_model.py`, `alpha_spectrum.py` (α at the laser wavelength for σ_model) |
| `77-23_film/trpl_transient.dat` | tr-PL transient of the PI-treated 77:23 film: time [ns], normalised PL (tab separated, no header; 65 points) | `configs/2trap_77-23_film_*.json` |
| `77-23_film/sspl_suns_qfls.dat` | ss-PL of the 77:23 film: intensity [suns] vs QFLS [eV] (tab separated, header + unit row) | same configs |
| `synthetic_1trap/` | synthetic 1-trap case, made by `scripts/make_synthetic_1trap_85-15.py` from the 1-trap fit of the 85:15 film (table below) | `configs/1trap_synthetic_inference.json` |
| `AM15G.dat` | AM1.5G photon flux: energy [eV], flux [photons cm⁻² s⁻¹ eV⁻¹] (generation rate at 1 sun) | `inference/data.py` |

Film parameters (set in the configs): 85:15 — E_g = 1.625 eV (1-trap case: 1.63 eV), d = 450 nm, absorbed
n_pulse = 1.6e17 × A(630 nm) = 1.130e17 cm⁻³, Voc,rad = 1.333 V, k_rad,ext = 4.793e-11 cm³/s;
77:23 — E_g = 1.67 eV, d = 1000 nm, absorbed n_pulse = 8.838e16 cm⁻³ (4.46 µJ/cm² at 514 nm, A = 0.766),
Voc,rad = 1.362 V, k_rad,ext = 3.3581e-11 cm³/s.

## `synthetic_1trap/`

Ground truth = the rate-equation-refined joint 1-trap fit of the 85:15 film (`configs/1trap_85-15_film_fit.json`):
E_t = 1.4650 eV, N_t = 5.736e16 cm⁻³, τ_n = 4.973e-9 s, τ_p = 7.765e-8 s, k_rad = 4.793e-11 cm³/s, E_g = 1.63 eV,
n_pulse = 1.130e17 cm⁻³. The data are noise-free.

| file | content |
|---|---|
| `synthetic_trpl_qfls_tau.csv` | τ_diff [s] vs QFLS [eV] at the 90 measured QFLS points (rate equations) |
| `synthetic_sspl_suns_qfls.csv` | QFLS [eV] and PLQY at the 10 measured ss-PL intensities (exact steady-state model) |
| `true_parameters.json`, `true_parameters.dat` | the ground truth |
| `ode_full_simulation.csv` | full rate-equation transient of the truth (time, PL, QFLS, τ_diff, densities, rates) |
| `synthetic_sspl_carriers_truth_*.csv` | steady-state carrier densities of the truth (at the data points and as a curve) |
| `figures/synthetic_data.png` | plot of the data |
