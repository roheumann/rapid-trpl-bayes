"""
sbc.py — computational simulation-based calibration (task T5)
============================================================

Checks that the sampler + likelihood + prior are computationally correct
(Talts et al. 2018, arXiv:1804.06788):

1. theta_true ~ prior (incl. E_t ordering);
2. data y = NN(theta_true) at fixed QFLS points + noise of the curve-level
   sigma (the whole-curve error, FIXED, no sigma_eff, never rescaled), drawn
   point by point as iid N(0, N sigma^2): the per-point equivalent of the
   MSE likelihood -MSE/(2 sigma^2), i.e. EXACTLY its generative model;
3. posterior sampling with the production NUTS, all data sets in one batch
   (``chains_per_dataset`` chains each, grouped metric/step size);
4. rank of theta_true among L roughly independent posterior draws.

If everything is correct the ranks are uniform on {0, ..., L} for every
parameter (and for derived quantities).  Rank histograms and ECDF-difference
plots are produced; chi^2 p-values quantify the uniformity.  Fixed parameters
are skipped.

Data from another model (19/10/2026)
------------------------------------
``run_sbc_smc(..., trpl_model=...)`` takes the tr-PL data from another forward model while the inference keeps the
NN likelihood, e.g. the legacy rate equations (``OdeTrplData``; variant B of the 1-trap SBC).  The ranks are then
uniform only if the NN can stand in for the ODE; data sets whose ODE curve does not cover every data point are
redrawn from the prior (the same rule as for axis-head networks), from a SEPARATE random stream, so theta_true, the
noise and the SMC seeds stay identical to the NN-data run of the same seed wherever no redraw was needed
(pair-by-pair comparison, ``paired_comparison``).

Additional outputs (19/10/2026): coverage of central posterior intervals with binomial standard errors, posterior
contraction 1 - var_post / var_prior and z-scores (``calibration_table``), and ECDF-difference plots with
SIMULTANEOUS 95 % bands (Säilynoja, Bürkner & Vehtari 2022, Stat. Comput. 32:32; ``ecdf_band``), with the
pointwise band as an option.
"""
from __future__ import annotations

import functools
import math
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

LEVELS = (0.5, 0.68, 0.95)


@dataclass
class SBCResult:
    theta_true: np.ndarray            # (n, D)
    ranks: np.ndarray                 # (n, D) — NaN for fixed parameters
    ranks_derived: pd.DataFrame       # (n, K)
    n_keep: int                       # L: draws per data set used for the ranks
    names: list
    diagnostics: pd.DataFrame         # per data set: max R-hat, min ESS, divergences
    info: dict = field(default_factory=dict)
    posterior: pd.DataFrame = field(default_factory=pd.DataFrame)   # per data set: true value, posterior mean/sd,
                                                                    # inside the central intervals (display units)


def simulate_data(surrogate, prior, qfls, sigma, n, rng):
    """theta_true ~ prior and noisy NN data at the QFLS points (implied noise model)."""
    import tensorflow as tf
    from .surrogate import linear_interp_matrix
    theta = prior.sample(n, rng=rng)
    if hasattr(surrogate, "values_at_u"):
        # axis-head networks: the curve of theta must reach every point (exclusion rule of the likelihood);
        # draws that do not are replaced by new prior draws (= the prior restricted to the covered region)
        u = np.minimum((np.asarray(qfls, np.float64) - surrogate.qfls_axis[-1]) / (surrogate.Vt * np.log(10.0)), 0.0)
        val = lambda th: surrogate.values_at_u(tf.constant(th), tf.constant(u[None]))[0].numpy()
        y = val(theta)
        for _ in range(100):
            bad = ~np.all(np.isfinite(y), axis=1)
            if not bad.any():
                break
            theta[bad] = prior.sample(int(bad.sum()), rng=rng)
            y[bad] = val(theta[bad])
        else:
            raise RuntimeError("could not draw covered parameter sets for the SBC data")
    else:
        y_axis = surrogate.predict_numpy(theta)                   # (n, 256) on the NN axis
        W = linear_interp_matrix(surrogate.qfls_axis, qfls)
        y = y_axis @ W.T
    noise_sd = sigma * math.sqrt(len(qfls))
    return theta, y + noise_sd * rng.standard_normal(y.shape)


def _thin_indices(n_chains, n_draws, n_keep):
    """n_keep draw indices spread evenly over all chains (chain, draw)."""
    per = int(math.ceil(n_keep / n_chains))
    d = np.linspace(0, n_draws - 1, per).astype(int)
    pairs = [(c, k) for c in range(n_chains) for k in d]
    return pairs[:n_keep]


def run_sbc(surrogate, prior, qfls, sigma: float, n_datasets: int = 100, chains_per_dataset: int = 4,
            n_warmup: int = 500, n_draws: int = 1000, n_keep: int = 100, batch_datasets: int = 100,
            target_accept: float = 0.95, seed: int = 0, verbose: bool = True) -> SBCResult:
    """
    Run SBC.  ``batch_datasets`` data sets are sampled together in one NUTS
    run (batch_datasets x chains_per_dataset chains).
    """
    from .multi import MultiTrplLikelihood
    from .report import derived_samples
    from .samplers import NutsSampler, SamplerResult, diagnostics, init_dispersed

    rng = np.random.default_rng(seed)
    qfls = np.asarray(qfls, np.float64)
    theta_true, Y = simulate_data(surrogate, prior, qfls, sigma, n_datasets, rng)
    D = prior.dim
    ranks = np.full((n_datasets, D), np.nan)
    der_true = derived_samples(theta_true, prior)
    ranks_der = pd.DataFrame(np.nan, index=range(n_datasets), columns=der_true.columns)
    diag_rows, t0, n_evals = [], time.perf_counter(), 0
    for b0 in range(0, n_datasets, batch_datasets):
        ids = np.arange(b0, min(n_datasets, b0 + batch_datasets))
        nb = len(ids)
        multi = MultiTrplLikelihood(surrogate, prior, [(qfls, Y[d]) for d in ids], sigma)
        groups = np.repeat(np.arange(nb), chains_per_dataset)
        sampler = NutsSampler(multi.bind(groups), n_chains=len(groups), groups=groups,
                              target_accept=target_accept)
        res = sampler.run(init_dispersed(D, len(groups), rng), n_warmup=n_warmup, n_draws=n_draws,
                          seed=int(rng.integers(1 << 30)), verbose=verbose)
        n_evals += res.info["evals_total_points"]
        for g, d in enumerate(ids):
            ch = np.nonzero(groups == g)[0]
            th = res.theta[ch]                                     # (chains, draws, D)
            pick = _thin_indices(len(ch), th.shape[1], n_keep)
            draws = np.stack([th[c, k] for c, k in pick])
            for i in prior.free_idx:
                ranks[d, i] = np.sum(draws[:, i] < theta_true[d, i])
            dd = derived_samples(draws, prior)
            ranks_der.loc[d] = (dd.values < der_true.loc[d].values[None, :]).sum(axis=0)
            sub = SamplerResult("nuts", theta=th, u=res.u[ch],
                                sample_stats={"diverging": res.sample_stats["diverging"][ch]})
            dg = diagnostics(sub, prior)
            free_keys = [prior.display_keys[i] for i in prior.free_idx]
            diag_rows.append({"dataset": int(d), "max_r_hat": float(dg.loc[free_keys, "r_hat"].max()),
                              "min_ess_bulk": float(dg.loc[free_keys, "ess_bulk"].min()),
                              "divergences": int(res.sample_stats["diverging"][ch].sum())})
        if verbose:
            print(f"SBC: {ids[-1] + 1}/{n_datasets} data sets done ({time.perf_counter() - t0:.0f} s)")
    info = {"n_datasets": n_datasets, "chains_per_dataset": chains_per_dataset, "n_warmup": n_warmup,
            "n_draws": n_draws, "n_keep": n_keep, "sigma": sigma, "n_qfls_points": len(qfls),
            "sigma_curve": sigma, "noise_sd_per_point": sigma * math.sqrt(len(qfls)),
            "noise": "curve-level sigma (whole-curve error) drawn per point as iid N(0, N sigma^2)", "time_s": time.perf_counter() - t0,
            "evals_total_points": n_evals}
    return SBCResult(theta_true=theta_true, ranks=ranks, ranks_derived=ranks_der, n_keep=n_keep,
                     names=list(prior.display_keys), diagnostics=pd.DataFrame(diag_rows), info=info)


class OdeTrplData:
    """
    Noise-free tr-PL data from the legacy rate equations for every tr-PL term of a LogPosterior (SBC variant B).

    The ODE is evaluated exactly as in the ODE-vs-NN comparison (``ode_likelihood.OdeLogPosterior.ode_curves``):
    ``network-training/trpl_module.solve_transient`` unchanged (atol_vec, integration from t = 0), training-data
    post-processing, log10 tau_diff at the used points on the NOMINAL QFLS axis 2 Vt ln(n_pulse/n_i) + Vt ln(PL/PL_max)
    (the calibration of the data and of ``likelihood.predict_trpl``), NaN where the curve does not reach a point.
    The solves run in ``n_jobs`` worker processes.  ``theta (B, D) -> [log10 tau (B, n_k) per tr-PL term]``.
    """

    def __init__(self, log_post, prior, constants: dict, T: float = 300.0, n_jobs: int = 8):
        from .ode_likelihood import OdeLogPosterior
        self.olps = [OdeLogPosterior(prior, "trpl", constants, t.surrogate.npulse, T, t.surrogate.qfls_used,
                                     n_jobs=n_jobs) for t in log_post.trpl_terms]
        self.n_jobs, self.T = int(n_jobs), float(T)
        self.n_solves, self.n_calls = 0, 0

    def warm_up(self) -> float:
        """Start the worker processes and compile numba (untimed elsewhere); returns seconds."""
        return float(sum(o.warm_up() for o in self.olps))

    def __call__(self, theta):
        theta = np.atleast_2d(np.asarray(theta, np.float64))
        ys = [o.ode_curves(theta) for o in self.olps]
        self.n_solves += len(theta) * len(self.olps)
        self.n_calls += 1
        return ys

    @property
    def info(self) -> dict:
        return {"model": "legacy ODE: network-training/trpl_module.solve_transient (atol_vec, t from 0.0), "
                         "training-data post-processing, nominal QFLS axis (inference.ode.simulate_trpl)",
                "n_solves": self.n_solves, "n_calls": self.n_calls, "n_jobs": self.n_jobs,
                "time_ode_s": float(sum(o.time_ode_s for o in self.olps)),
                "npulse_cm3": [float(o.npulse) for o in self.olps], "T_K": self.T}


def simulate_mode_data(log_post, prior, n: int, rng, max_tries: int = 200, trpl_model=None, redraw_rng=None,
                       return_info: bool = False):
    """
    theta_true ~ prior and synthetic data for every term of a LogPosterior (any mode), with the noise model implied by
    its curve-level likelihood -MSE/(2 sigma^2): y = model(theta) + iid N(0, N sigma^2) per point (the per-point
    equivalent of the whole-curve sigma; sigma = the term's current sigma, not rescaled).
    tr-PL: the term's surrogate at the used points (axis-head networks: theta whose curve does not reach every point is
    redrawn — the exclusion rule of the likelihood), or ``trpl_model`` (a callable theta (B, D) -> [y (B, n_k) per
    tr-PL term], e.g. ``OdeTrplData``; NaN where its curve does not cover a point -> redrawn); ss-PL: the exact model
    at the measured points (draws for which the model is not finite are redrawn).

    ``redraw_rng``: generator of the replacement draws (default: ``rng``, as before).  A separate generator keeps the
    stream of ``rng`` (the noise and everything drawn after it) independent of the number of redraws.
    Returns (theta (n, D), [y_trpl (n, n_k) per tr-PL term], y_sspl (n, m) | None) and, with ``return_info``, a dict:
    redrawn (n,) bool, n_redrawn, n_replacement_draws, theta_first (the first draws), y_trpl_clean (noise-free data).
    """
    import tensorflow as tf
    sig_t = [float(s.numpy()) for s in log_post.sigma_trpl]
    sig_s = None if log_post.sspl_term is None else float(log_post.sigma_sspl.numpy())
    redraw_rng = rng if redraw_rng is None else redraw_rng

    def model(th):
        th_t = tf.constant(th)
        if trpl_model is None:
            yt = [t.surrogate.log10_tau(th_t).numpy() for t in log_post.trpl_terms]
        else:
            yt = [np.asarray(y, np.float64) for y in trpl_model(th)]
        ys = None if log_post.sspl_term is None else log_post.sspl_term.observation.predict(th_t).numpy()
        ok = np.ones(len(th), bool)
        for y in yt + ([ys] if ys is not None else []):
            ok &= np.all(np.isfinite(y), axis=1)
        return yt, ys, ok

    theta = prior.sample(n, rng=rng)
    theta_first = theta.copy()
    yt, ys, ok = model(theta)
    redrawn, n_repl = ~ok, 0
    for _ in range(max_tries):
        if ok.all():
            break
        bad = np.nonzero(~ok)[0]
        n_repl += len(bad)
        theta[bad] = prior.sample(len(bad), rng=redraw_rng)
        yt_b, ys_b, ok_b = model(theta[bad])
        for k in range(len(yt)):
            yt[k][bad] = yt_b[k]
        if ys is not None:
            ys[bad] = ys_b
        ok[bad] = ok_b
    if not ok.all():
        raise RuntimeError("could not draw parameter sets with finite model predictions")
    y_clean = [y.copy() for y in yt]
    yt = [y + s * math.sqrt(y.shape[1]) * rng.standard_normal(y.shape) for y, s in zip(yt, sig_t)]
    if ys is not None:
        ys = ys + sig_s * math.sqrt(ys.shape[1]) * rng.standard_normal(ys.shape)
    if not return_info:
        return theta, yt, ys
    return theta, yt, ys, {"redrawn": redrawn, "n_redrawn": int(redrawn.sum()), "n_replacement_draws": int(n_repl),
                           "theta_first": theta_first, "y_trpl_clean": y_clean}


def _posterior_summary(th, prior, true_disp, true_der, levels) -> dict:
    """Posterior mean, sd and 'inside the central interval' flags of the free parameters and derived quantities."""
    from .report import derived_samples
    disp = prior.to_display(th)
    cols = {prior.display_keys[i]: (disp[:, i], true_disp[i]) for i in prior.free_idx}
    dd = derived_samples(th, prior)
    cols.update({c: (dd[c].to_numpy(), float(true_der[c])) for c in dd.columns})
    row = {}
    for name, (x, x0) in cols.items():
        row[f"true_{name}"] = x0
        row[f"mean_{name}"] = float(np.mean(x))
        row[f"sd_{name}"] = float(np.std(x))
        q = np.quantile(x, np.concatenate([[(1 - a) / 2, (1 + a) / 2] for a in levels]))
        for j, a in enumerate(levels):
            row[f"in{int(round(100 * a))}_{name}"] = bool(q[2 * j] <= x0 <= q[2 * j + 1])
    return row


def prior_variances(prior, n: int = 200_000, seed=0) -> dict:
    """Prior variance of the free parameters (display units) and derived quantities (Monte Carlo, n draws)."""
    from .report import derived_samples
    th = prior.sample(n, rng=np.random.default_rng(seed))
    disp = prior.to_display(th)
    out = {prior.display_keys[i]: float(np.var(disp[:, i])) for i in prior.free_idx}
    dd = derived_samples(th, prior)
    out.update({c: float(dd[c].var(ddof=0)) for c in dd.columns})
    return out


def run_sbc_smc(log_post, prior, n_datasets: int = 200, n_keep: int = 100, smc_kwargs: dict | None = None,
                seed: int = 0, evaluator=None, max_batch: int = 16384, verbose: bool = True,
                trpl_model=None, data_model: str = "nn", levels=LEVELS) -> SBCResult:
    """
    Computational SBC with tempered SMC as the sampler, for any mode (tr-PL, ss-PL, joint).

    For every data set: theta_true ~ prior, data from ``simulate_mode_data`` (implied noise model, sigma FIXED at
    the current sigmas of ``log_post``), the data are put into ``log_post`` with ``set_data`` (one compiled
    population evaluator for all data sets), SMC is run and the rank of theta_true among ``n_keep`` draws picked at
    random from the equally weighted final particles is recorded.  The original data of ``log_post`` are restored at
    the end.  Diagnostics per data set: max R-hat over the islands, min bulk ESS, Kish ESS, log Z, time.
    Per data set also (``SBCResult.posterior``): posterior mean / sd and whether theta_true lies inside the central
    ``levels`` intervals of ALL final particles (free parameters in display units and derived quantities).

    ``trpl_model``: noise-free tr-PL data from another model (e.g. ``OdeTrplData``) instead of the NN; the inference
    still uses the NN.  Its redraws (curve not covering the points) use a separate generator, so theta_true, noise
    and SMC seeds equal those of the NN-data run with the same seed wherever no redraw was needed.
    """
    from .report import derived_samples
    from .samplers import diagnostics
    from . import smc as smc_mod
    smc_kwargs = dict(smc_kwargs or {})
    rng = np.random.default_rng(seed)
    redraw_rng = None if trpl_model is None else np.random.default_rng([seed, 1])
    y0_t = [np.array(t.y_obs, copy=True) for t in log_post.trpl_terms]
    y0_s = None if log_post.sspl_term is None else np.array(log_post.sspl_term.observation.y_obs, copy=True)
    t_sim = time.perf_counter()
    theta_true, Yt, Ys, sim = simulate_mode_data(log_post, prior, n_datasets, rng, trpl_model=trpl_model,
                                                 redraw_rng=redraw_rng, return_info=True)
    t_sim = time.perf_counter() - t_sim
    rms_nn = None
    if trpl_model is not None:
        import tensorflow as tf
        y_nn = [t.surrogate.log10_tau(tf.constant(theta_true)).numpy() for t in log_post.trpl_terms]
        dif = np.concatenate([a - b for a, b in zip(y_nn, sim["y_trpl_clean"])], axis=1)
        rms_nn, max_nn = np.sqrt(np.mean(dif ** 2, axis=1)), np.max(np.abs(dif), axis=1)
    ev = evaluator or smc_mod.PopulationEvaluator(log_post, max_batch=max_batch, fixed_shape=True)
    D = prior.dim
    ranks = np.full((n_datasets, D), np.nan)
    der_true = derived_samples(theta_true, prior)
    true_disp = prior.to_display(theta_true)
    ranks_der = pd.DataFrame(np.nan, index=range(n_datasets), columns=der_true.columns)
    free_keys = [prior.display_keys[i] for i in prior.free_idx]
    rows, post_rows, t0, n_evals = [], [], time.perf_counter(), 0
    try:
        for d in range(n_datasets):
            log_post.set_data(y_trpl=[y[d] for y in Yt] if Yt else None, y_sspl=None if Ys is None else Ys[d])
            td = time.perf_counter()
            res = smc_mod.run_smc(log_post, **smc_kwargs, seed=int(rng.integers(1 << 30)), evaluator=ev, verbose=False)
            dt = time.perf_counter() - td
            n_evals += res.info["evals_total_points"]
            th = res.flat_theta()
            draws = th[rng.choice(len(th), size=n_keep, replace=False)]
            for i in prior.free_idx:
                ranks[d, i] = np.sum(draws[:, i] < theta_true[d, i])
            dd = derived_samples(draws, prior)
            ranks_der.loc[d] = (dd.values < der_true.loc[d].values[None, :]).sum(axis=0)
            dg = diagnostics(res, prior)
            row = {"dataset": d, "max_r_hat": float(dg.loc[free_keys, "r_hat"].max()),
                   "min_ess_bulk": float(dg.loc[free_keys, "ess_bulk"].min()), "ess_kish": float(res.ess),
                   "n_unique_final": int(np.sum(res.info.get("n_unique_final", -1))), "log_z": float(res.log_evidence),
                   "n_stages": int(np.max(res.info["n_stages"])), "evals": int(res.info["evals_total_points"]),
                   "time_s": dt, "redrawn": bool(sim["redrawn"][d])}
            if rms_nn is not None:
                row.update({"rms_nn_minus_data_model": float(rms_nn[d]), "max_abs_nn_minus_data_model": float(max_nn[d])})
            rows.append(row)
            post_rows.append({"dataset": d, **_posterior_summary(th, prior, true_disp[d], der_true.loc[d], levels)})
            if verbose and ((d + 1) % 10 == 0 or d == n_datasets - 1):
                print(f"SBC (SMC): {d + 1}/{n_datasets} data sets, {time.perf_counter() - t0:.0f} s, "
                      f"last max R-hat {rows[-1]['max_r_hat']:.3f}", flush=True)
    finally:
        log_post.set_data(y_trpl=y0_t if y0_t else None, y_sspl=y0_s)
    info = {"sampler": "tempered SMC", "n_datasets": n_datasets, "n_keep": n_keep, "smc": smc_kwargs,
            "sigma_trpl": [float(s.numpy()) for s in log_post.sigma_trpl],
            "sigma_sspl": None if log_post.sspl_term is None else float(log_post.sigma_sspl.numpy()),
            "n_points_trpl": [int(y.shape[1]) for y in Yt], "n_points_sspl": None if Ys is None else int(Ys.shape[1]),
            "noise": "curve-level sigma (whole-curve error, not rescaled) drawn per point as iid N(0, N sigma^2), the per-point equivalent of -MSE/(2 sigma^2)", "time_s": time.perf_counter() - t0,
            "evals_total_points": n_evals, "seed": seed,
            "data_model": data_model,
            "data_model_info": getattr(trpl_model, "info", None), "time_simulate_data_s": t_sim,
            "n_redrawn": sim["n_redrawn"], "n_replacement_draws": sim["n_replacement_draws"],
            "redraw_stream": "rng" if trpl_model is None else "separate (seed, 1)",
            "levels": list(levels), "prior_var": prior_variances(prior, seed=[seed, 2])}
    return SBCResult(theta_true=theta_true, ranks=ranks, ranks_derived=ranks_der, n_keep=n_keep,
                     names=list(prior.display_keys), diagnostics=pd.DataFrame(rows), info=info,
                     posterior=pd.DataFrame(post_rows))


# ─────────────────────────────────────────────────────────────────────────────
# summaries
# ─────────────────────────────────────────────────────────────────────────────
def rank_columns(sbc: SBCResult) -> dict:
    """{quantity: ranks (n,)} of the free parameters (finite ranks) and the derived quantities."""
    cols = {n: sbc.ranks[:, i] for i, n in enumerate(sbc.names) if np.isfinite(sbc.ranks[:, i]).any()}
    cols.update({c: sbc.ranks_derived[c].to_numpy(float) for c in sbc.ranks_derived.columns})
    return cols


def subset(sbc: SBCResult, mask) -> SBCResult:
    """The SBC restricted to the data sets where ``mask`` is True (e.g. converged ones)."""
    mask = np.asarray(mask, bool)
    idx = np.nonzero(mask)[0]
    pick = lambda df: df.iloc[idx].reset_index(drop=True) if len(df) else df
    info = dict(sbc.info, n_datasets_subset=int(mask.sum()))
    return SBCResult(theta_true=sbc.theta_true[mask], ranks=sbc.ranks[mask], ranks_derived=pick(sbc.ranks_derived),
                     n_keep=sbc.n_keep, names=sbc.names, diagnostics=pick(sbc.diagnostics), info=info,
                     posterior=pick(sbc.posterior))


def uniformity_table(sbc: SBCResult, n_bins: int = 20) -> pd.DataFrame:
    """
    chi^2 test of uniform ranks per parameter / derived quantity, plus the p-value of the simultaneous ECDF test
    (Säilynoja et al. 2022: the fraction of uniform rank samples whose ECDF deviates more, ``p_ecdf_sim``).
    """
    from scipy import stats
    rows = []
    for name, r in rank_columns(sbc).items():
        r = r[np.isfinite(r)]
        if len(r) == 0:
            continue
        h, _ = np.histogram(r, bins=n_bins, range=(-0.5, sbc.n_keep + 0.5))
        chi2, p = stats.chisquare(h)
        rows.append({"quantity": name, "n": len(r), "chi2": chi2, "p_value": p,
                     "mean_rank_rel": float(np.mean(r) / sbc.n_keep),
                     "p_ecdf_sim": ecdf_test_p(r, sbc.n_keep)})
    return pd.DataFrame(rows).set_index("quantity")


def calibration_table(sbc: SBCResult, levels=None) -> pd.DataFrame:
    """
    Per quantity: coverage of the central ``levels`` posterior intervals (fraction of data sets with theta_true
    inside, binomial SE sqrt(c (1 - c) / n)), posterior contraction 1 - var_post / var_prior (median and mean over
    the data sets) and the posterior z-score (mean - true) / sd (mean and sd; N(0, 1) if calibrated).
    """
    post = sbc.posterior
    if post is None or len(post) == 0:
        return pd.DataFrame()
    levels = levels or sbc.info.get("levels", LEVELS)
    pv = sbc.info.get("prior_var", {})
    rows = []
    for name in [c[len("mean_"):] for c in post.columns if c.startswith("mean_")]:
        n = len(post)
        row = {"quantity": name, "n": n}
        for a in levels:
            tag = int(round(100 * a))
            c = float(post[f"in{tag}_{name}"].astype(float).mean())
            row[f"cov{tag}"], row[f"cov{tag}_se"] = c, math.sqrt(c * (1 - c) / n)
        if name in pv:
            con = 1.0 - post[f"sd_{name}"].to_numpy() ** 2 / pv[name]
            row["contraction_median"], row["contraction_mean"] = float(np.median(con)), float(np.mean(con))
        z = (post[f"mean_{name}"] - post[f"true_{name}"]) / post[f"sd_{name}"]
        row["z_mean"], row["z_sd"] = float(z.mean()), float(z.std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows).set_index("quantity")


def paired_comparison(sbc_a: SBCResult, sbc_b: SBCResult, n_bins: int = 20) -> pd.DataFrame:
    """
    Variant A vs B on the SAME theta_true (data sets where both runs drew the same parameters; with the same seed
    they also share the noise and the SMC seeds).  Per quantity: mean (rank_B - rank_A) / L, Wilcoxon signed-rank
    p of the paired rank differences, chi^2 homogeneity p of the two rank histograms (all data sets, unpaired),
    posterior-mean shift (mean_B - mean_A) / sd_A (median |.|, mean, max |.|), median sd_B / sd_A.
    """
    from scipy import stats
    if sbc_a.theta_true.shape != sbc_b.theta_true.shape:
        raise ValueError("paired comparison needs two SBC runs with the same number of data sets")
    same = np.all(np.isclose(sbc_a.theta_true, sbc_b.theta_true, rtol=0, atol=1e-12), axis=1)
    ra, rb = rank_columns(sbc_a), rank_columns(sbc_b)
    pa, pb = sbc_a.posterior, sbc_b.posterior
    rows = []
    for name in ra:
        if name not in rb:
            continue
        a, b = ra[name][same], rb[name][same]
        diff = b - a
        row = {"quantity": name, "n_paired": int(same.sum()), "mean_rank_diff_rel": float(np.mean(diff) / sbc_a.n_keep),
               "frac_rank_diff_zero": float(np.mean(diff == 0))}
        try:
            row["wilcoxon_p"] = float(stats.wilcoxon(b, a).pvalue) if np.any(diff != 0) else 1.0
        except ValueError:
            row["wilcoxon_p"] = float("nan")
        ha, _ = np.histogram(ra[name][np.isfinite(ra[name])], bins=n_bins, range=(-0.5, sbc_a.n_keep + 0.5))
        hb, _ = np.histogram(rb[name][np.isfinite(rb[name])], bins=n_bins, range=(-0.5, sbc_b.n_keep + 0.5))
        keep = (ha + hb) > 0
        row["homogeneity_chi2_p"] = float(stats.chi2_contingency(np.stack([ha[keep], hb[keep]]))[1])
        if len(pa) and len(pb) and f"mean_{name}" in pa and f"mean_{name}" in pb:
            sh = ((pb[f"mean_{name}"] - pa[f"mean_{name}"]) / pa[f"sd_{name}"]).to_numpy()[same]
            row.update({"mean_shift_sd_median_abs": float(np.median(np.abs(sh))), "mean_shift_sd_mean": float(np.mean(sh)),
                        "mean_shift_sd_max_abs": float(np.max(np.abs(sh))),
                        "sd_ratio_median": float(np.median((pb[f"sd_{name}"] / pa[f"sd_{name}"]).to_numpy()[same]))})
        rows.append(row)
    return pd.DataFrame(rows).set_index("quantity")


# ─────────────────────────────────────────────────────────────────────────────
# ECDF bands (Säilynoja, Bürkner & Vehtari 2022)
# ─────────────────────────────────────────────────────────────────────────────
def _ecdf_counts(r, n_keep: int) -> tuple[np.ndarray, np.ndarray]:
    """z_i = i / (L + 1), i = 1..L, and the counts #(rank <= i - 1): Binomial(n, z_i) under uniform ranks."""
    z = np.arange(1, n_keep + 1) / (n_keep + 1)
    r = np.sort(np.asarray(r, float)[np.isfinite(r)])
    return z, np.searchsorted(r, np.arange(n_keep) + 0.5, side="left")


def _gamma(counts, n: int, z) -> np.ndarray:
    """2 min_i min(P(K <= c_i), P(K >= c_i)), K ~ Binomial(n, z_i): the pointwise level the ECDF just touches."""
    from scipy import stats
    counts = np.atleast_2d(counts)
    lo = stats.binom.cdf(counts, n, z[None, :])
    hi = stats.binom.sf(counts - 1, n, z[None, :])
    return 2.0 * np.min(np.minimum(lo, hi), axis=1)


@functools.lru_cache(maxsize=64)
def _gamma_null(n: int, n_keep: int, n_sim: int = 10000, seed: int = 0) -> np.ndarray:
    """gamma of n_sim samples of n uniform ranks on {0..L} (the null distribution of the simultaneous test)."""
    rng = np.random.default_rng(seed)
    r = rng.integers(0, n_keep + 1, size=(n_sim, n))
    z = np.arange(1, n_keep + 1) / (n_keep + 1)
    h = np.bincount((r + (n_keep + 1) * np.arange(n_sim)[:, None]).ravel(), minlength=n_sim * (n_keep + 1))
    counts = np.cumsum(h.reshape(n_sim, n_keep + 1), axis=1)[:, :n_keep]          # #(rank <= i - 1), i = 1..L
    return np.sort(_gamma(counts, n, z))


def ecdf_band(n: int, n_keep: int, alpha: float = 0.05, method: str = "simultaneous", n_sim: int = 10000,
              seed: int = 0) -> dict:
    """
    Band of ECDF(z) - z for n uniform ranks on {0..L} at z_i = i / (L + 1).

    method = "simultaneous": Säilynoja et al. (2022), simulation method — the pointwise binomial level gamma is
             chosen such that the ECDF of n uniform ranks lies inside ALL pointwise (1 - gamma) intervals with
             probability 1 - alpha;
    method = "pointwise":    the former band, +-1.96 sqrt(z (1 - z) / n) (normal approximation, each z on its own).
    Returns dict z, lo, hi (ECDF-difference units), gamma.
    """
    from scipy import stats
    z = np.arange(1, n_keep + 1) / (n_keep + 1)
    if method == "pointwise":
        h = stats.norm.ppf(1 - alpha / 2) * np.sqrt(z * (1 - z) / n)
        return {"z": z, "lo": -h, "hi": h, "gamma": alpha, "method": method}
    if method != "simultaneous":
        raise ValueError("method must be 'simultaneous' or 'pointwise'")
    g = float(np.quantile(_gamma_null(int(n), int(n_keep), n_sim, seed), alpha))
    lo = stats.binom.ppf(g / 2, n, z) / n - z
    hi = stats.binom.ppf(1 - g / 2, n, z) / n - z
    return {"z": z, "lo": lo, "hi": hi, "gamma": g, "method": method}


def ecdf_test_p(r, n_keep: int, n_sim: int = 10000) -> float:
    """p-value of the simultaneous ECDF test: P(gamma_uniform <= gamma_observed) (small = non-uniform)."""
    r = np.asarray(r, float)[np.isfinite(r)]
    z, c = _ecdf_counts(r, n_keep)
    g_obs = float(_gamma(c, len(r), z)[0])
    null = _gamma_null(len(r), int(n_keep), n_sim)
    return float(np.searchsorted(null, g_obs, side="right") / len(null))


def ecdf_difference(r, n_keep: int) -> tuple[np.ndarray, np.ndarray]:
    """(z, ECDF(z) - z) of the ranks at z_i = i / (L + 1), with the end points z = 0 and 1 (difference 0)."""
    z, c = _ecdf_counts(r, n_keep)
    n = int(np.isfinite(np.asarray(r, float)).sum())
    return np.concatenate([[0.0], z, [1.0]]), np.concatenate([[0.0], c / n - z, [0.0]])


# ─────────────────────────────────────────────────────────────────────────────
# plots
# ─────────────────────────────────────────────────────────────────────────────
def plot_rank_histograms(sbc: SBCResult, prior, n_bins: int = 20, derived: bool = False):
    """Rank histograms with the 99 % band of a uniform histogram (binomial)."""
    import matplotlib.pyplot as plt
    from scipy import stats
    if derived:
        names = list(sbc.ranks_derived.columns)
        R = sbc.ranks_derived.to_numpy(float)
    else:
        names = [prior.display_keys[i] for i in prior.free_idx]
        R = sbc.ranks[:, prior.free_idx]
    n = int(np.isfinite(R[:, 0]).sum())
    ncol = min(len(names), 5)
    nrow = int(math.ceil(len(names) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.6 * nrow))
    axes = np.atleast_1d(axes).ravel()
    lo, hi = stats.binom.ppf([0.005, 0.995], n, 1.0 / n_bins)
    for a, name in enumerate(names):
        ax = axes[a]
        ax.axhspan(lo, hi, color="0.85")
        ax.axhline(n / n_bins, color="0.5", lw=0.8)
        ax.hist(R[:, a][np.isfinite(R[:, a])], bins=n_bins, range=(-0.5, sbc.n_keep + 0.5),
                color="C0", edgecolor="w")
        ax.set_title(name, fontsize=9)
        ax.set_yticks([])
    for ax in axes[len(names):]:
        ax.axis("off")
    fig.suptitle(f"SBC rank histograms ({n} data sets, L = {sbc.n_keep}; grey: 99 % band of uniform ranks)")
    fig.tight_layout()
    return fig


def plot_ecdf_difference(sbc: SBCResult, prior, band: str = "simultaneous"):
    """
    ECDF of the normalised ranks minus the uniform CDF, all free parameters in one panel, with the 95 % band:
    band = "simultaneous" (Säilynoja et al. 2022, default) or "pointwise" (the former band).
    """
    import matplotlib.pyplot as plt
    names = [prior.display_keys[i] for i in prior.free_idx]
    R = sbc.ranks[:, prior.free_idx]
    n = int(np.isfinite(R[:, 0]).sum())
    b = ecdf_band(n, sbc.n_keep, method=band)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.fill_between(b["z"], b["lo"], b["hi"], color="0.85", step="mid", label=f"95 % band ({band})")
    for a, name in enumerate(names):
        z, e = ecdf_difference(R[:, a], sbc.n_keep)
        ax.plot(z, e, lw=1.2, label=name)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("normalised rank")
    ax.set_ylabel("ECDF - uniform")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def _panel_grid(n_panels: int, ncol: int = 4, w: float = 3.2, h: float = 2.7):
    import matplotlib.pyplot as plt
    ncol = min(n_panels, ncol)
    nrow = int(math.ceil(n_panels / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(w * ncol, h * nrow), squeeze=False)
    axes = axes.ravel()
    for ax in axes[n_panels:]:
        ax.axis("off")
    return fig, axes


def plot_ecdf_panels(results: dict, names=None, band: str = "simultaneous", colors=None, title: str | None = None):
    """
    ECDF-difference plot, one panel per quantity, the SBC runs of ``results`` ({label: SBCResult}) overlaid, with
    the 95 % band of the first run (band = "simultaneous" | "pointwise"; dashed: the band of another run if its
    number of data sets differs).
    """
    labels = list(results)
    cols = {lab: rank_columns(results[lab]) for lab in labels}
    names = names or list(cols[labels[0]])
    colors = colors or {lab: f"C{i}" for i, lab in enumerate(labels)}
    fig, axes = _panel_grid(len(names))
    for ax, name in zip(axes, names):
        drawn = set()
        for j, lab in enumerate(labels):
            if name not in cols[lab]:
                continue
            r = cols[lab][name]
            n, L = int(np.isfinite(r).sum()), results[lab].n_keep
            if (n, L) not in drawn:
                b = ecdf_band(n, L, method=band)
                if not drawn:
                    ax.fill_between(b["z"], b["lo"], b["hi"], color="0.85", step="mid", lw=0)
                else:
                    ax.step(b["z"], b["lo"], color=colors[lab], ls=":", lw=0.8, where="mid")
                    ax.step(b["z"], b["hi"], color=colors[lab], ls=":", lw=0.8, where="mid")
                drawn.add((n, L))
            z, e = ecdf_difference(r, L)
            ax.plot(z, e, color=colors[lab], lw=1.3, label=f"{lab} (n = {n})")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("normalised rank", fontsize=8)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel("ECDF - uniform", fontsize=8)
    axes[0].legend(fontsize=7)
    fig.suptitle(title or f"SBC ECDF difference (grey: 95 % {band} band of uniform ranks)", fontsize=10)
    fig.tight_layout()
    return fig


def plot_rank_histogram_panels(results: dict, names=None, n_bins: int = 20, colors=None, title: str | None = None):
    """Rank histograms (step lines) of the runs of ``results`` overlaid per quantity, 99 % band of uniform ranks."""
    from scipy import stats
    labels = list(results)
    cols = {lab: rank_columns(results[lab]) for lab in labels}
    names = names or list(cols[labels[0]])
    colors = colors or {lab: f"C{i}" for i, lab in enumerate(labels)}
    fig, axes = _panel_grid(len(names))
    for ax, name in zip(axes, names):
        for j, lab in enumerate(labels):
            if name not in cols[lab]:
                continue
            r = cols[lab][name]
            r = r[np.isfinite(r)]
            L = results[lab].n_keep
            if j == 0:
                lo, hi = stats.binom.ppf([0.005, 0.995], len(r), 1.0 / n_bins)
                ax.axhspan(lo, hi, color="0.88")
                ax.axhline(len(r) / n_bins, color="0.5", lw=0.8)
            h, e = np.histogram(r, bins=n_bins, range=(-0.5, L + 0.5))
            ax.stairs(h, e, color=colors[lab], lw=1.4, label=lab)
        ax.set_title(name, fontsize=9)
        ax.set_yticks([])
        ax.tick_params(labelsize=7)
    axes[0].legend(fontsize=7)
    fig.suptitle(title or f"SBC rank histograms ({n_bins} bins; grey: 99 % band of uniform ranks)", fontsize=10)
    fig.tight_layout()
    return fig


def plot_coverage(results: dict, names=None, levels=None, colors=None, title: str | None = None):
    """Coverage minus nominal level of the central intervals (+-2 binomial SE), runs side by side, one panel each."""
    labels = list(results)
    tabs = {lab: calibration_table(results[lab], levels) for lab in labels}
    names = names or list(tabs[labels[0]].index)
    levels = levels or results[labels[0]].info.get("levels", LEVELS)
    colors = colors or {lab: f"C{i}" for i, lab in enumerate(labels)}
    fig, axes = _panel_grid(len(names))
    off = np.linspace(-0.12, 0.12, len(labels)) if len(labels) > 1 else [0.0]
    for ax, name in zip(axes, names):
        for j, lab in enumerate(labels):
            t = tabs[lab]
            if name not in t.index:
                continue
            for k, a in enumerate(levels):
                tag = int(round(100 * a))
                c, se = t.loc[name, f"cov{tag}"], t.loc[name, f"cov{tag}_se"]
                ax.errorbar(k + off[j], c - a, yerr=2 * se, fmt="o", ms=4, color=colors[lab], capsize=2,
                            label=lab if k == 0 else None)
        ax.axhline(0, color="k", lw=0.7)
        ax.set_xticks(range(len(levels)), [f"{int(round(100 * a))} %" for a in levels], fontsize=8)
        ax.set_title(name, fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("coverage - nominal", fontsize=8)
    axes[0].legend(fontsize=7)
    fig.suptitle(title or "Coverage of the central posterior intervals (error bars: +-2 binomial SE)", fontsize=10)
    fig.tight_layout()
    return fig


def plot_zscore_contraction(results: dict, names=None, colors=None, title: str | None = None):
    """Posterior z-score (mean - true) / sd vs contraction 1 - var_post / var_prior, every data set a point."""
    labels = list(results)
    colors = colors or {lab: f"C{i}" for i, lab in enumerate(labels)}
    first = results[labels[0]].posterior
    names = names or [c[len("mean_"):] for c in first.columns if c.startswith("mean_")]
    fig, axes = _panel_grid(len(names))
    for ax, name in zip(axes, names):
        for lab in labels:
            p, pv = results[lab].posterior, results[lab].info.get("prior_var", {})
            if f"mean_{name}" not in p or name not in pv:
                continue
            z = (p[f"mean_{name}"] - p[f"true_{name}"]) / p[f"sd_{name}"]
            con = 1.0 - p[f"sd_{name}"] ** 2 / pv[name]
            ax.scatter(con, z, s=7, alpha=0.6, color=colors[lab], label=lab, lw=0)
        ax.axhline(0, color="k", lw=0.6)
        for s in (-2, 2):
            ax.axhline(s, color="0.6", lw=0.6, ls="--")
        ax.set_xlim(min(0.0, ax.get_xlim()[0]), 1.0)
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("contraction", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("posterior z-score", fontsize=8)
    axes[0].legend(fontsize=7, markerscale=2)
    fig.suptitle(title or "Posterior z-score vs contraction (dashed: |z| = 2)", fontsize=10)
    fig.tight_layout()
    return fig
