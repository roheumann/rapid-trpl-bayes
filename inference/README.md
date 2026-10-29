# `inference/` — full-dimensional Bayesian inference

This package replaces the pairwise-sliced grid posterior of the original submission. It samples the full 4- to 9-D
posterior (one or two traps) of the SRH model with detrapping, using the trained network as a surrogate for the tr-PL
rate equations and the exact steady-state model for the ss-PL data.

Entry points are the notebooks in the repository root:

1. **`main.ipynb`** — import data, preprocess, estimate the start point, fit (CMA-ES on the network, refined with
   `curve_fit` on the rate equations);
2. **`smc.ipynb`** — posterior with tempered SMC (log Z, credible intervals, corner HDF5);
3. **`ppc.ipynb`** — posterior predictive checks;
4. **`ode_inference.ipynb`** — the same inference with the rate equations instead of the network.

Every step plots and exports its results. All fits and the sampling are timed with `timeit`, and every
likelihood evaluation is counted. The configs of the publication are listed in the top-level `README.md`.
Results are written to `<output_dir>/<run_name>_<mode>/` (CSV and JSON files and a `figures/` subfolder).

```bash
pip install -r requirements.txt
python -m pytest tests -q          # unit tests (random weights where a network is not found)
```

## Modules

| module | purpose |
|---|---|
| `surrogate.py` | loads the NN artefacts; θ (B, D) → log10 τ_diff at the measured QFLS (batched, differentiable). `TrplSurrogate`: fixed analytic axis (interpolation matrix). `TrplSurrogateWindow` (axis-head networks, `surrogate_meta.json` axis `predicted_window`): q_max and span per θ, data read at the same PL decade, θ excluded where the curve ends above a data point. QFLS-axis consistency check of the training data |
| `prior.py` | uniform prior on the **training box** of the network, with E_t1 < E_t2 (and optionally N_t > n_1); fixed parameters; bijection to unconstrained u with exact log\|J\| (TF and numpy) |
| `sspl_tf.py` | ss-PL forward model in TF float64 (1 trap: closed form; N traps: bracketed root with implicit-function gradient; interpolation per row) and its numpy twin |
| `likelihood.py` | `LogPosterior`, the **single objective** for the optimiser, samplers, grid and profile likelihood (modes `trpl` / `sspl` / `joint`) |
| `optimize.py` | multi-start CMA-ES (instances in lockstep, one batched call per generation; trajectories recorded) + batched L-BFGS |
| `refine.py` | refinement of every CMA-ES result with `scipy.optimize.curve_fit` on the full ODE (same likelihood; parallel on CPU cores) |
| `samplers.py` | NUTS (TFP, chunked XLA, staged dense-metric warm-up, per-chain step sizes, chain groups for many data sets, stuck-chain reset) and emcee (stretch move, stuck-walker reset); ArviZ diagnostics |
| `smc.py` | gradient-free adaptive importance sampling: likelihood-tempered SMC (adaptive β at ESS ≈ 0.5 N, systematic resampling, random-walk MH with the weighted-particle covariance, independent islands; log Z and log Z(σ')) and AMIS (Student-t mixture in u, started from the eigenvector cylinders + defensive prior draws, weighted EM, balance-heuristic weights, PSIS k̂); every likelihood evaluation is one batched call for the whole population; results are `SamplerResult`s with weights (notebook `smc.ipynb`) |
| `grid.py` | full-dimensional grid posterior (streamed, normalised 1-D/2-D marginals, intervals, plots) |
| `cylinder.py` | eigenvector-cylinder importance sampling around the best fit; Laplace covariance: curve_fit (mode `trpl`) or Gauss–Newton at θ* (`laplace_cov`, modes `sspl` / `joint`) |
| `corner_export.py`, `corner_plot.py` | one HDF5 file per corner plot (posterior-mass bands, 1σ/2σ contours) (also readable by Origin) and the matplotlib preview; comparison corner of several samplers (`write_comparison_corner`, `scripts/combined_corner.py`) |
| `runs.py` | rebuilds the likelihood of a finished main.ipynb run (any mode) for `smc.ipynb`, `ppc.ipynb` and the scripts |
| `multi.py` | tr-PL likelihood for many data sets in one batch (SBC, coverage) |
| `sbc.py` | T5: computational simulation-based calibration (ranks, histograms, ECDF, uniformity tests) |
| `coverage.py` | T6: ODE decays + measured noise + real preprocessing → full workflow → interval coverage |
| `validate.py` | T7: ODE re-evaluation of posterior samples with Pareto-smoothed importance sampling |
| `dynamics.py` | T9: trap occupations / net capture rates across posterior samples (Figure 8) |
| `profile.py` | T3: profile likelihood in one parameter (all others re-optimised, batched, warm-started) |
| `ode.py` | rate-equation evaluation (training-data solver + post-processing); comparisons with data at the same PL decade (`axis='nominal'`), physical QFLS for plots (`axis='physical'`); full ODE output for export (both QFLS columns) |
| `synthetic.py` | noise-free synthetic use case (tr-PL + ss-PL) from a parameter set |
| `report.py` | equal-tailed intervals with prior-limited flags, Eq. 10/11 derived quantities, plots (incl. optimiser trajectories), `.dat`/CSV/JSON export |
| `data.py`, `start_points.py` | loaders (any separator, header detection) and wrappers around `preprocessing/` (start values may assume E_t / τ_n: config keys `Et_eV`, `taun_s`; τ_p then follows from τ_0) |
| `benchmark.py` | `timeit_call`, `EvalCounter` (counts correctly inside XLA), throughput benchmark, offline costs of the surrogate |

## Conventions

* θ = `[log10 k_rad, (log10 N_t, E_t/E_g, log10 τ_n, log10 τ_p) per trap]`. These are the NN inputs without log10 n_pulse and E_g.
  Plots and tables use display coordinates, i.e. E_t in eV.
* Likelihood: `log L_d = −MSE_d/(2σ_d²) − log(σ_d √2π)`. In this MSE form each curve counts as one effective data point.
* σ_eff,d² = σ_d² + MSE_min,d. Here MSE_min,d is the best-fit MSE of data set d fitted **alone**, as in the legacy convention.
  So σ_trpl is the same in the tr-PL-only run and in the joint run.
* Optimisers minimise −log L(θ(u)) without the Jacobian, so the best fit is the maximum likelihood inside the support.
  Samplers use log p(u) = log L + log π + log\|J\|.

## Things to know

* QFLS axis (16/10/2026).  Each training curve lies on its own axis linspace(q_lo, q_hi, 256) of the physical ODE QFLS.
  q_hi = kT ln((n0 + n_pulse)(p0 + n_pulse)/n_i²) exceeds the high-injection value 2 Vt ln(n_pulse/n_i) when p0 of
  the traps is not small (19.6 % of the B2nf curves > 1 meV), and 12 % of the curves reach equilibrium before 12 PL
  decades.  Networks with an axis head (`network-training/train_axis_head.py`; the shipped 2-trap network `networks/2trap` has one) predict
  q_max and the span; older networks use the analytic axis.  Measured QFLS stays calibrated with the high-injection
  formula; model and data are always compared at the same PL decade u = log10(PL/PL0).
* Legacy ODE solver (`network-training/trpl_module.py`, the solver that generated the training data): `atol_vec` (a scalar atol sometimes fails) and integration from t = 0 (16/10/2026,
  Robin).  The right-hand side has no explicit t, so the start only shifts the time axis; the 3609 and B2 training
  data were generated with atol_vec.
* SMC-only workflow (16/10/2026, 85:15 film v2): config `nuts.run = false` makes `main.ipynb` stop after the fits
  (T8 rows of the fits only); `smc.ipynb` with `NUTS_REFERENCE = None`, `RUN_AMIS = False`, `RUN_CYL_IS = False` gives
  the posterior (credible intervals, corner HDF5, T8 row); `ppc.ipynb` works with SMC samples alone and checks the
  ss-PL term too (exact model); `scripts/sbc_smc.py` runs SBC with SMC for any mode (`sbc.run_sbc_smc`, data swapped
  with `LogPosterior.set_data`, one compiled evaluator).
* `sigma_rule = "model_plus_nn"`: σ_trpl = sqrt(σ_model² + σ_nn²), σ_nn² = whole-curve test MSE (log10) of the
  network, no MSE_min term; σ_sspl as given.  The sliced grid uses the same σ for such runs.
* `sspl.voc_rad_V` (needs k_rad fixed): the 1-sun generation is chosen so that the model's radiative limit
  kT ln(1 + G/(k_rad n_i²)) equals the measured Voc,rad (`data.sspl_g_per_sun`, check stored in 6_best_fit.json).
* The 2-trap training data contain **only** E_t1 < E_t2, and E_t/E_g ≤ 0.975.
  Inside the prior the NN is never evaluated outside its training box.
* Interpolation: `TrplSurrogate(interp_space='log10')` interpolates log10 τ, which is the default.
  `'linear'` reproduces `02_trpl_errors.py` (τ interpolated linearly) to about 1e-6 relative (unit test).
* emcee: use the default stretch move.  The mixture DEMove + DESnookerMove gave ~4 % too narrow
  intervals in a test on the prior alone (exact answer known; `tests/test_samplers.py`).
* SMC / AMIS (`smc.py`, `smc.ipynb`): no gradients and no sequential chains, so ridges, plateaus at the prior bounds
  and several modes do not stall them; both return log Z (model comparison).  AMIS works in the unconstrained u, not in
  decades: in decades 2/3 of the t draws of the near-prior 2-trap 85:15 posterior fell outside the box and log Z was
  biased by −0.03.  Use the Kish ESS / k̂ of the weights; the ArviZ ESS of SMC particles is computed along the
  particle index (systematic resampling keeps descendants next to each other).
* NUTS: `target_accept = 0.95` is needed for these ridge-shaped posteriors. With 0.8 about 5 % of the transitions
  diverge in the 1-trap synthetic case.
  Chains that are stuck in regions of negligible posterior mass after warm-up stage 1 or 2 are moved onto good chains.
  This happens only during warm-up, and the number of resets is reported in `info['chains_reset']`.
