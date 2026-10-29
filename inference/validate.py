"""
validate.py — does the surrogate change the posterior? (task T7)
================================================================

Thinned posterior samples (NN likelihood) are re-evaluated with the
rate-equation ODE.  The importance weights

    log w_s = log L_ODE(theta_s) - log L_NN(theta_s)

(only the tr-PL term differs; the ss-PL term is exact physics in both) turn
the NN posterior into the ODE posterior.  The weights are Pareto-smoothed
(PSIS, Vehtari et al. 2024, JMLR 25(72)); the shape parameter k-hat of the
fitted generalised Pareto tail is the reliability diagnostic:

    k-hat < 0.5   reliable;   0.5–0.7  usable;   > 0.7  the two posteriors
    differ too much for reweighting (then the NN posterior itself is suspect).

Caveat: reweighting can only redistribute mass among the NN posterior
samples; regions the NN posterior does not visit cannot be recovered.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.special import logsumexp


# ─────────────────────────────────────────────────────────────────────────────
# PSIS (algorithm of Vehtari et al.; generalised Pareto fit of Zhang & Stephens 2009)
# ─────────────────────────────────────────────────────────────────────────────

def _gpd_fit(x):
    """Zhang–Stephens estimate (k, sigma) of a generalised Pareto fit to sorted x > 0."""
    n = len(x)
    prior_bs, prior_k = 3.0, 10.0
    m = 30 + int(math.sqrt(n))
    b = 1.0 - np.sqrt(m / (np.arange(1, m + 1) - 0.5))
    b /= prior_bs * x[int(n / 4 + 0.5) - 1]
    b += 1.0 / x[-1]
    k = np.log1p(-b[:, None] * x).mean(axis=1)
    L = n * (np.log(-(b / k)) - k - 1.0)
    w = 1.0 / np.exp(L - L[:, None]).sum(axis=1)
    keep = w >= 10 * np.finfo(float).eps
    w, b = w[keep] / w[keep].sum(), b[keep]
    b_post = float(np.sum(b * w))
    k_post = float(np.log1p(-b_post * x).mean())
    sigma = -k_post / b_post
    k_post = (n * k_post + prior_k * 0.5) / (n + prior_k)        # weakly informative prior on k
    return k_post, sigma


def _gpd_inv(p, k, sigma):
    if abs(k) < np.finfo(float).eps:
        return -sigma * np.log1p(-p)
    return sigma * np.expm1(-k * np.log1p(-p)) / k


def psis(log_w, r_eff: float = 1.0):
    """
    Pareto-smoothed importance sampling.

    Returns (normalised smoothed log weights, k_hat).
    """
    lw = np.asarray(log_w, np.float64).copy()
    S = len(lw)
    lw -= lw.max()
    M = int(math.ceil(min(0.2 * S, 3.0 * math.sqrt(S / r_eff))))
    order = np.argsort(lw)
    cutoff = max(lw[order[-M - 1]], np.log(np.finfo(float).tiny))
    tail = np.nonzero(lw > cutoff)[0]
    k = np.inf
    if len(tail) > 4:
        tail_sorted = tail[np.argsort(lw[tail])]
        x = np.exp(lw[tail_sorted]) - math.exp(cutoff)
        k, sigma = _gpd_fit(x)
        if np.isfinite(k):
            p = (np.arange(1, len(tail) + 1) - 0.5) / len(tail)
            smoothed = np.log(_gpd_inv(p, k, sigma) + math.exp(cutoff))
            lw[tail_sorted] = np.minimum(smoothed, 0.0)          # truncate at the largest raw weight
    return lw - logsumexp(lw), float(k)


# ─────────────────────────────────────────────────────────────────────────────
# ODE re-evaluation of posterior samples
# ─────────────────────────────────────────────────────────────────────────────

def ode_reweight(log_post, art, npulse: float, theta_samples, n_samples: int = 1000,
                 n_jobs: int = 16, seed: int = 0, T: float = 300.0) -> dict:
    """
    Re-evaluate ``n_samples`` thinned posterior samples with the ODE, PSIS.

    Returns dict with theta (used samples), log_w (raw), log_w_psis
    (normalised), k_hat, ess (of the smoothed weights), frac_covered and the
    ODE timing.  Samples whose ODE curve covers < 90 % of the tr-PL points are
    dropped (reported in n_dropped).
    """
    from .ode import ode_mse
    rng = np.random.default_rng(seed)
    th_all = np.asarray(theta_samples)
    th = th_all[rng.choice(len(th_all), size=min(n_samples, len(th_all)), replace=False)]
    term = log_post.trpl_terms[0]
    sigma = log_post.sigmas()["trpl"][0]
    r = ode_mse(th, log_post.prior, npulse, art.constants, term.surrogate.qfls_used, term.y_obs,
                n_jobs=n_jobs, T=T)
    mse_nn = log_post.np_mse_terms(th)[term.name]
    ok = np.isfinite(r["mse"]) & (r["frac_covered"] >= 0.9)
    log_w = -(r["mse"][ok] - mse_nn[ok]) / (2.0 * sigma ** 2)
    lw_s, k = psis(log_w)
    w = np.exp(lw_s)
    return {"theta": th[ok], "log_w": log_w, "log_w_psis": lw_s, "k_hat": k,
            "ess": float(1.0 / np.sum(w ** 2)), "n_used": int(ok.sum()), "n_dropped": int((~ok).sum()),
            "mse_nn": mse_nn[ok], "mse_ode": r["mse"][ok], "ode_time_s": r["time_s"],
            "ode_time_per_solve_s": r["time_per_solve_s"]}


def weighted_quantiles(x, w, qs):
    o = np.argsort(x)
    cw = np.cumsum(w[o])
    cw /= cw[-1]
    return np.interp(qs, cw, x[o])


def reweight_table(rw: dict, prior) -> pd.DataFrame:
    """Quantiles (16/50/84 %) of NN samples vs ODE-reweighted samples per parameter."""
    disp = prior.to_display(rw["theta"])
    w = np.exp(rw["log_w_psis"])
    rows = []
    for i in prior.free_idx:
        q_nn = np.percentile(disp[:, i], [16, 50, 84])
        q_ode = weighted_quantiles(disp[:, i], w, [0.16, 0.5, 0.84])
        rows.append({"parameter": prior.display_keys[i], "q16_nn": q_nn[0], "median_nn": q_nn[1],
                     "q84_nn": q_nn[2], "q16_ode": q_ode[0], "median_ode": q_ode[1], "q84_ode": q_ode[2],
                     "shift_in_sd": (q_ode[1] - q_nn[1]) / max(1e-12, 0.5 * (q_nn[2] - q_nn[0]))})
    return pd.DataFrame(rows).set_index("parameter")


def plot_reweighted(rw: dict, prior, bins: int = 30):
    """Marginals of the NN posterior samples vs the ODE-reweighted ones; inset: log w."""
    import matplotlib.pyplot as plt
    free = list(prior.free_idx)
    disp = prior.to_display(rw["theta"])
    w = np.exp(rw["log_w_psis"])
    D = len(free) + 1
    ncol = min(D, 5)
    nrow = int(math.ceil(D / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.6 * nrow))
    axes = np.atleast_1d(axes).ravel()
    bnd = prior.display_bounds()
    for a, i in enumerate(free):
        axes[a].hist(disp[:, i], bins=bins, range=tuple(bnd[i]), density=True, histtype="step",
                     lw=1.5, color="C0", label="NN posterior")
        axes[a].hist(disp[:, i], bins=bins, range=tuple(bnd[i]), weights=w, density=True,
                     histtype="step", lw=1.5, color="C3", label="ODE-reweighted (PSIS)")
        axes[a].set_xlabel(prior.labels[i])
        axes[a].set_yticks([])
    ax = axes[len(free)]
    ax.hist(rw["log_w"] - np.max(rw["log_w"]), bins=40, color="0.5")
    ax.set_xlabel(r"$\log w$ (raw, shifted)")
    ax.set_title(f"k̂ = {rw['k_hat']:.2f}, ESS = {rw['ess']:.0f}/{rw['n_used']}", fontsize=9)
    for ax in axes[D:]:
        ax.axis("off")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    return fig
