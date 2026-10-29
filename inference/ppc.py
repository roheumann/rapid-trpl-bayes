"""
ppc.py — posterior predictive checks of the tr-PL fit
=====================================================
15/10/2026 (revision).  Works on equally weighted posterior draws θ_s of any sampler (NUTS, emcee, SMC, …).

Model curve and replicated data
-------------------------------
  f_s       = log10 τ_diff(θ_s) at the measured QFLS points (NN surrogate; or the rate equations, ``ode_curves``)
  y_rep,s   = f_s + s_pt ε_s,  ε ~ N(0, 1) iid per point

σ is the error of the WHOLE curve: the likelihood of the workflow is log L = −MSE/(2σ²) − log(σ√2π) per curve
(likelihood.py), σ is never rescaled.  Replicated data must be drawn point by point; the per-point noise that gives
exactly this curve likelihood is iid N(0, Nσ²), i.e. s_pt = σ√N (the per-point equivalent of the curve-level σ, not
a change of σ).  Two noise scales are checked (the keys stay as before, the output file names depend on them):

  'likelihood'  s_pt = σ√N  — curve-level σ (each curve = 1 observation), drawn per point: the generative model of
                              the posterior
  'per point'   s_pt = σ    — the stricter reading: σ as the RMS residual of every single point

Checks (per sampler and noise scale)
------------------------------------
  * bands of f (parameter uncertainty only) and of y_rep (+ noise): 2.5 / 16 / 50 / 84 / 97.5 % per point
  * coverage of the data by the central 68 % / 95 % y_rep intervals (0.68 / 0.95 if calibrated)
  * PIT u_i = P(y_rep,i ≤ y_obs,i) — uniform if calibrated; piled at 0.5: predictive too wide;
    piled at 0 and 1: too narrow or systematic misfit
  * discrepancies T(y, θ) with posterior predictive p-values p = P(T(y_rep, θ) ≥ T(y_obs, θ)):
      T_mse  = mean_i (y_i − f_i(θ))²                 (the statistic of the likelihood)
      T_ac1  = lag-1 autocorrelation of y − f(θ) along QFLS (iid assumption: smooth misfit → T_ac1 ≫ 0)
    p ≈ 0.5: consistent; p → 0: misfit / correlated residuals; p → 1: noise model wider than the residuals.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

QS = (0.025, 0.16, 0.5, 0.84, 0.975)


def pick_draws(theta, n: int, seed: int = 0) -> np.ndarray:
    """n equally weighted draws (without replacement if possible) from (S, D) posterior samples."""
    theta = np.asarray(theta, np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(theta), size=min(n, len(theta)), replace=False)
    return theta[idx]


def nn_curves(log_post, theta, term: int = 0, batch: int = 4096) -> dict:
    """NN predictions for the draws: log10 τ at the data points (S, n) and on the NN QFLS axis (S, 256)."""
    used, axis = [], []
    for i in range(0, len(theta), batch):
        p = log_post.predict_trpl(theta[i:i + batch], term)
        used.append(p["log10_tau_used"]); axis.append(p["log10_tau_axis"])
    return {"f": np.concatenate(used), "f_axis": np.concatenate(axis), "qfls": p["qfls_used"],
            "qfls_axis": p["qfls_axis"], "y_obs": p["log10_tau_obs"]}


def sspl_curves(log_post_ss, theta, batch: int = 4096) -> dict:
    """ss-PL model (exact, no surrogate) at the measured points for the draws: y (S, n), x_exp, y_obs."""
    ys = []
    for i in range(0, len(theta), batch):
        p = log_post_ss.predict_sspl(theta[i:i + batch])
        ys.append(p["y_model"])
    return {"f": np.concatenate(ys), "x": p["x_exp"], "y_obs": p["y_obs"]}


def ode_curves(theta, prior, npulse: float, constants: dict, qfls, n_jobs: int = 8, T: float = 300.0) -> np.ndarray:
    """Rate-equation log10 τ at the data points for every draw (S, n); NaN outside the simulated QFLS range."""
    from joblib import Parallel, delayed
    from . import ode
    ode._ode_module()                                  # compile the numba functions once in the parent
    rows = Parallel(n_jobs=n_jobs)(delayed(ode.ode_log10_tau)(th, prior, npulse, constants, qfls, T) for th in theta)
    return np.asarray(rows, np.float64)


def _ac1(r) -> np.ndarray:
    """Lag-1 autocorrelation of every row of r (NaN-free rows)."""
    r = r - r.mean(axis=1, keepdims=True)
    den = np.sum(r * r, axis=1)
    return np.where(den > 0, np.sum(r[:, 1:] * r[:, :-1], axis=1) / np.where(den > 0, den, 1.0), 0.0)


def check(f, y_obs, s_pt: float, seed: int = 0) -> dict:
    """
    Posterior predictive check for model curves f (S, n) at the data y_obs (n,) with iid noise s_pt per point.
    Draws whose curve does not cover every data point (NaN, ODE) are dropped and counted.
    """
    f = np.asarray(f, np.float64)
    ok = np.all(np.isfinite(f), axis=1)
    f = f[ok]
    rng = np.random.default_rng(seed)
    y_rep = f + s_pt * rng.standard_normal(f.shape)
    r_obs, r_rep = y_obs[None, :] - f, y_rep - f
    t_mse_obs, t_mse_rep = np.mean(r_obs ** 2, axis=1), np.mean(r_rep ** 2, axis=1)
    t_ac_obs, t_ac_rep = _ac1(r_obs), _ac1(r_rep)
    qf = np.quantile(f, QS, axis=0)
    qy = np.quantile(y_rep, QS, axis=0)
    pit = np.mean(y_rep <= y_obs[None, :], axis=0)
    bands = pd.DataFrame({"point": np.arange(len(y_obs)), "y_obs": y_obs,
                          **{f"f_q{100 * q:g}": qf[k] for k, q in enumerate(QS)},
                          **{f"yrep_q{100 * q:g}": qy[k] for k, q in enumerate(QS)}, "pit": pit})
    inside68 = (y_obs >= qy[1]) & (y_obs <= qy[3])
    inside95 = (y_obs >= qy[0]) & (y_obs <= qy[4])
    f_inside95 = (y_obs >= qf[0]) & (y_obs <= qf[4])
    summary = {"s_pt": float(s_pt), "n_draws": int(ok.sum()), "n_dropped": int((~ok).sum()), "n_points": int(len(y_obs)),
               "coverage68": float(inside68.mean()), "coverage95": float(inside95.mean()),
               "data_inside_f95_band": float(f_inside95.mean()),
               "pit_mean": float(pit.mean()), "pit_sd": float(pit.std()),
               "pit_frac_0.25_0.75": float(np.mean((pit > 0.25) & (pit < 0.75))),
               "mse_obs_median": float(np.median(t_mse_obs)), "mse_rep_median": float(np.median(t_mse_rep)),
               "p_mse": float(np.mean(t_mse_rep >= t_mse_obs)),
               "ac1_obs_median": float(np.median(t_ac_obs)), "ac1_rep_median": float(np.median(t_ac_rep)),
               "p_ac1": float(np.mean(t_ac_rep >= t_ac_obs)),
               "rmse_median_curve": float(np.sqrt(np.mean((qf[2] - y_obs) ** 2))),
               "max_abs_resid_median_curve": float(np.max(np.abs(qf[2] - y_obs)))}
    return {"summary": summary, "bands": bands, "t": pd.DataFrame({"mse_obs": t_mse_obs, "mse_rep": t_mse_rep,
                                                                   "ac1_obs": t_ac_obs, "ac1_rep": t_ac_rep}),
            "pit": pit, "y_rep": y_rep, "f": f}


def plot_bands(ax, qfls, y_obs, res: dict, color="C0", label="", show_rep=True, ylog10: bool = True):
    """
    Data, median curve, 95 % band of f (dark) and of y_rep (light) (upper panel).  ylog10: the values are log10
    (tr-PL: log10 τ, drawn as τ on a log axis); False: drawn as they are (ss-PL QFLS in eV).
    """
    b = res["bands"]
    tr = (lambda v: 10 ** np.asarray(v)) if ylog10 else (lambda v: np.asarray(v))
    if show_rep:
        ax.fill_between(qfls, tr(b["yrep_q2.5"]), tr(b["yrep_q97.5"]), color=color, alpha=0.15, lw=0,
                        label=f"{label} y_rep 95 %")
    ax.fill_between(qfls, tr(b["f_q2.5"]), tr(b["f_q97.5"]), color=color, alpha=0.45, lw=0, label=f"{label} f 95 %")
    ax.plot(qfls, tr(b["f_q50"]), "-", color=color, lw=1.0)
    ax.plot(qfls, tr(y_obs), "o", ms=2.5, color="k", zorder=5)
    if ylog10:
        ax.set_yscale("log")


def plot_residuals(ax, qfls, y_obs, res: dict, color="C0", label=""):
    """Data minus the median curve with the 95 % bands of f and y_rep (relative to the median curve)."""
    b = res["bands"]
    m = b["f_q50"].values
    ax.fill_between(qfls, b["yrep_q2.5"] - m, b["yrep_q97.5"] - m, color=color, alpha=0.15, lw=0)
    ax.fill_between(qfls, b["f_q2.5"] - m, b["f_q97.5"] - m, color=color, alpha=0.45, lw=0)
    ax.plot(qfls, y_obs - m, "o", ms=2.5, color="k", label=label)
    ax.axhline(0, color="0.3", lw=0.6)
