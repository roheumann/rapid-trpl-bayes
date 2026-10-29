"""
report.py — credible intervals, derived quantities, plots and export
====================================================================

Intervals (REVISION_PLAN.md, Section 2 item 6)
----------------------------------------------
* Equal-tailed (central quantile) intervals at 68 % and 95 %; unlike HPD
  intervals they do not change between log and linear axes.
* "Prior-limited": if the marginal density in the histogram bin adjacent to a
  prior bound exceeds ``edge_frac`` of its maximum, the data do not constrain
  that side; the table then reports "< x" (or "> x") instead of a two-sided
  interval (R1 comment 6).

Derived quantities (task T4, Eqs. 10/11 of the paper), per trap k and sample:
  n_1 = N_C exp((E_t - E_g)/kT),  log10(tau_p sqrt(n_1 N_t)),
  log10(tau_p n_1),  log10(N_t / n_1)
(tau_SRH,p is taken as the hole capture time constant tau_p of that trap.)

Plots
-----
plot_trpl_fit, plot_sspl_fit, plot_optimizer_history, plot_corner,
plot_trace, plot_marginals_compare, plot_profile, plot_start_sensitivity.

Export
------
save_json (numpy-safe), export_samples (CSV in theta + physical units).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

KB_EV = 8.617333262e-5


# ─────────────────────────────────────────────────────────────────────────────
# intervals
# ─────────────────────────────────────────────────────────────────────────────

def credible_intervals(samples_disp: np.ndarray, bounds_disp: np.ndarray, keys: list,
                       levels=(0.68, 0.95), edge_frac: float = 0.2, n_bins: int = 40,
                       truth=None, best=None) -> pd.DataFrame:
    """
    Equal-tailed intervals with prior-limited flags.

    Parameters
    ----------
    samples_disp : (N, D) samples in display coordinates
    bounds_disp : (D, 2) prior bounds in display coordinates
    keys : parameter names
    edge_frac : threshold for the prior-limited flag (density in the edge bin
                relative to the maximum bin)
    truth, best : optional (D,) vectors added as columns
    """
    rows = []
    for i, k in enumerate(keys):
        x = samples_disp[:, i]
        lo_b, hi_b = bounds_disp[i]
        if hi_b - lo_b < 1e-6:                       # fixed parameter
            row = {"parameter": k, "median": float(np.median(x)), "mean": float(np.mean(x)), "sd": 0.0,
                   "prior_lo": lo_b, "prior_hi": hi_b, "prior_limited_low": False, "prior_limited_high": False}
            for lev in levels:
                tag = f"{int(round(lev * 100))}"
                row.update({f"lo{tag}": lo_b, f"hi{tag}": hi_b, f"width{tag}": 0.0, f"interval{tag}": "fixed"})
                if truth is not None:
                    row[f"truth_in{tag}"] = bool(abs(truth[i] - x[0]) < 1e-6)
            if truth is not None:
                row["truth"] = float(truth[i])
            if best is not None:
                row["best_fit"] = float(best[i])
            rows.append(row)
            continue
        h, _ = np.histogram(x, bins=n_bins, range=(lo_b, hi_b))
        hmax = max(h.max(), 1)
        lim_lo = h[0] / hmax > edge_frac
        lim_hi = h[-1] / hmax > edge_frac
        row = {"parameter": k, "median": float(np.median(x)), "mean": float(np.mean(x)),
               "sd": float(np.std(x)), "prior_lo": lo_b, "prior_hi": hi_b,
               "prior_limited_low": bool(lim_lo), "prior_limited_high": bool(lim_hi)}
        for lev in levels:
            a = 100 * (1 - lev) / 2
            ql, qh = np.percentile(x, [a, 100 - a])
            tag = f"{int(round(lev * 100))}"
            row[f"lo{tag}"], row[f"hi{tag}"] = float(ql), float(qh)
            row[f"width{tag}"] = float(qh - ql)
            if lim_lo and lim_hi:
                txt = "prior-limited (unconstrained)"
            elif lim_lo:
                txt = f"< {qh:.4g}"
            elif lim_hi:
                txt = f"> {ql:.4g}"
            else:
                txt = f"{ql:.4g} – {qh:.4g}"
            row[f"interval{tag}"] = txt
            if truth is not None:
                row[f"truth_in{tag}"] = bool(ql <= truth[i] <= qh)
        if truth is not None:
            row["truth"] = float(truth[i])
        if best is not None:
            row["best_fit"] = float(best[i])
        rows.append(row)
    return pd.DataFrame(rows).set_index("parameter")


def derived_samples(theta: np.ndarray, prior) -> pd.DataFrame:
    """Eq. 10/11 parameter combinations per sample (log10 units), see module doc."""
    th = np.atleast_2d(theta)
    kT = KB_EV * prior.T
    out = {}
    for k in range(1, prior.n_traps + 1):
        c = 1 + 4 * (k - 1)
        logNt, et, logtaup = th[:, c], th[:, c + 1], th[:, c + 3]
        log_n1 = math.log10(prior.Nc) + (et * prior.Eg - prior.Eg) / (kT * math.log(10))
        out[f"log10_n1_{k}"] = log_n1
        out[f"log10_taup_sqrt_n1Nt_{k}"] = logtaup + 0.5 * (log_n1 + logNt)
        out[f"log10_taup_n1_{k}"] = logtaup + log_n1
        out[f"log10_Nt_over_n1_{k}"] = logNt - log_n1
    return pd.DataFrame(out)


def derived_intervals(theta: np.ndarray, prior, levels=(0.68, 0.95)) -> pd.DataFrame:
    """Equal-tailed intervals of the derived quantities (no prior-limited flag)."""
    d = derived_samples(theta, prior)
    rows = []
    for k in d.columns:
        x = d[k].to_numpy()
        row = {"quantity": k, "median": float(np.median(x)), "sd": float(np.std(x))}
        for lev in levels:
            a = 100 * (1 - lev) / 2
            ql, qh = np.percentile(x, [a, 100 - a])
            tag = f"{int(round(lev * 100))}"
            row[f"lo{tag}"], row[f"hi{tag}"], row[f"width{tag}"] = ql, qh, qh - ql
        rows.append(row)
    return pd.DataFrame(rows).set_index("quantity")


def compare_widths(ci_a: pd.DataFrame, ci_b: pd.DataFrame, label_a="trpl", label_b="joint") -> pd.DataFrame:
    """Width ratios b/a of 68 % and 95 % intervals (task T4 table)."""
    common = ci_a.index.intersection(ci_b.index)
    out = pd.DataFrame(index=common)
    for tag in ("68", "95"):
        out[f"width{tag}_{label_a}"] = ci_a.loc[common, f"width{tag}"]
        out[f"width{tag}_{label_b}"] = ci_b.loc[common, f"width{tag}"]
        out[f"ratio{tag}_{label_b}/{label_a}"] = out[f"width{tag}_{label_b}"] / out[f"width{tag}_{label_a}"]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_trpl_fit(log_post, theta_best, theta_samples=None, ode_curve=None, term: int = 0,
                  qfls_all=None, tau_all=None, title=None, ax=None):
    """
    tr-PL data vs best fit (NN) and, optionally, a posterior-predictive band
    (5–95 % of NN predictions for posterior samples) and the ODE curve of the
    best fit.  Lower panel: residuals in log10 tau.
    """
    import matplotlib.pyplot as plt
    p = log_post.predict_trpl(theta_best, term)
    if ax is None:
        fig, ax = plt.subplots(2, 1, figsize=(6, 6), sharex=True,
                               gridspec_kw={"height_ratios": [3, 1]})
    else:
        fig = ax[0].figure
    # show model curves only over the data range (+ margin): at the ends of the
    # NN QFLS axis the network output is not constrained by any data point
    x_lo, x_hi = p["qfls_used"].min() - 0.02, p["qfls_used"].max() + 0.02
    sel = (p["qfls_axis"] >= x_lo) & (p["qfls_axis"] <= x_hi)
    if qfls_all is not None:
        ax[0].semilogy(qfls_all, tau_all, ".", color="0.75", ms=4, label="data (not used)")
    ax[0].semilogy(p["qfls_used"], 10 ** p["log10_tau_obs"], "o", ms=4, color="k", label="data")
    if theta_samples is not None:
        ys = log_post.predict_trpl(theta_samples, term)["log10_tau_axis"]
        lo, hi = np.nanpercentile(ys, [5, 95], axis=0)          # NaN: axis-head nets beyond a curve's end
        ax[0].fill_between(p["qfls_axis"][sel], 10 ** lo[sel], 10 ** hi[sel], color="C0", alpha=0.3,
                           label="posterior 5–95 %")
    ax[0].semilogy(p["qfls_axis"][sel], 10 ** p["log10_tau_axis"][0][sel], "-", color="C3", lw=2,
                   label="best fit (NN)")
    if ode_curve is not None:
        so = (ode_curve[0] >= x_lo) & (ode_curve[0] <= x_hi)
        ax[0].semilogy(ode_curve[0][so], 10 ** ode_curve[1][so], "--", color="C2", lw=1.5, label="best fit (ODE)")
    ax[0].set_xlim(x_lo, x_hi)
    ax[0].set_ylabel(r"$\tau_\mathrm{diff}$ [s]")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)
    if title:
        ax[0].set_title(title)
    res = p["log10_tau_used"][0] - p["log10_tau_obs"]
    ax[1].plot(p["qfls_used"], res, "o", ms=3, color="C3")
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_xlabel(r"$\Delta E_\mathrm{F}$ [eV]")
    ax[1].set_ylabel(r"$\Delta\log_{10}\tau$")
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    return fig, ax


def plot_sspl_fit(log_post, theta_best, theta_samples=None, ax=None, title=None):
    """ss-PL data vs model (best fit and optional posterior band), in the comparison space."""
    import matplotlib.pyplot as plt
    obs = log_post.sspl_term.observation
    p = log_post.predict_sspl(theta_best)
    if ax is None:
        fig, ax = plt.subplots(2, 1, figsize=(6, 6), sharex=True,
                               gridspec_kw={"height_ratios": [3, 1]})
    else:
        fig = ax[0].figure
    is_plqy = obs.mode.startswith("plqy")
    xlog = obs.mode != "plqy_vs_qfls"
    y_obs = 10 ** p["y_obs"] if is_plqy else p["y_obs"]
    ax[0].plot(p["x_exp"], y_obs, "o", color="k", ms=5, label="data")
    c = p["curve"]
    if theta_samples is not None:
        curves = obs.model_curve(theta_samples)
        # band on a common x grid
        xg = np.logspace(np.log10(p["x_exp"].min()) - 0.3, np.log10(p["x_exp"].max()) + 0.3, 100) if xlog \
            else np.linspace(p["x_exp"].min() - 0.02, p["x_exp"].max() + 0.02, 100)
        ys = []
        for i in range(len(theta_samples)):
            xi, yi = curves["x"][i], curves["y"][i]
            o = np.argsort(xi)
            xx = np.log10(xi[o]) if xlog else xi[o]
            yy = np.log10(yi[o]) if is_plqy else yi[o]
            ys.append(np.interp(np.log10(xg) if xlog else xg, xx, yy))
        lo, hi = np.percentile(np.array(ys), [5, 95], axis=0)
        if is_plqy:
            lo, hi = 10 ** lo, 10 ** hi
        ax[0].fill_between(xg, lo, hi, color="C0", alpha=0.3, label="posterior 5–95 %")
    o = np.argsort(c["x"][0])
    ax[0].plot(c["x"][0][o], c["y"][0][o], "-", color="C3", lw=2, label="best fit")
    if xlog:
        ax[0].set_xscale("log")
    if is_plqy:
        ax[0].set_yscale("log")
    xlabel = {"suns": "intensity [suns]", "G": r"G [cm$^{-3}$s$^{-1}$]", "qfls": "QFLS [eV]"}[obs.mode.split("_vs_")[1]]
    ax[0].set_xlim(p["x_exp"].min() / (1.5 if xlog else 1) - (0 if xlog else 0.02),
                   p["x_exp"].max() * (1.5 if xlog else 1) + (0 if xlog else 0.02))
    yr = (y_obs.min(), y_obs.max())
    pad = (yr[1] - yr[0]) * 0.3 if not is_plqy else 0
    ax[0].set_ylim((yr[0] / 3, yr[1] * 3) if is_plqy else (yr[0] - pad, yr[1] + pad))
    ax[0].set_ylabel("PLQY" if is_plqy else "QFLS [eV]")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)
    if title:
        ax[0].set_title(title)
    ax[1].plot(p["x_exp"], p["y_model"][0] - p["y_obs"], "o", color="C3", ms=4)
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_xlabel(xlabel)
    ax[1].set_ylabel("residual [" + obs.residual_unit + "]")
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    return fig, ax


def plot_optimizer_history(results: list, title="CMA-ES convergence", highlight: str | None = "physics"):
    """Best -log L vs evaluations for every CMA-ES start."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    fmin = min(r.neg_log_lik for r in results)
    for r in results:
        if r.history_evals is None:
            continue
        hl = highlight is not None and highlight in r.label
        ax.plot(r.history_evals, r.history_best - fmin + 1e-3, color="C3" if hl else "0.6",
                lw=2 if hl else 0.8, zorder=3 if hl else 1, label=r.label if hl else None)
    ax.set_yscale("log")
    ax.set_xlabel("evaluations")
    ax.set_ylabel(r"$-\log L - \min(-\log L) + 10^{-3}$")
    ax.set_title(title)
    if highlight:
        ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig, ax


def plot_corner(samples_disp: np.ndarray, prior, truth=None, best=None, overlay=None,
                overlay_label="emcee", label="NUTS", range_mode: str = "prior", bins: int = 40):
    """
    Corner plot (``corner`` package) in display coordinates.

    range_mode 'prior' shows the full prior range (makes prior-limited
    parameters obvious); 'data' zooms to the samples.
    """
    import corner
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    free = list(prior.free_idx)                      # fixed parameters are not plotted
    samples_disp = np.asarray(samples_disp)[:, free]
    overlay = np.asarray(overlay)[:, free] if overlay is not None else None
    truth = np.asarray(truth)[free] if truth is not None else None
    best = np.asarray(best)[free] if best is not None else None
    rng = [tuple(b) for b in prior.display_bounds()[free]] if range_mode == "prior" else None
    fig = corner.corner(samples_disp, labels=[prior.labels[i] for i in free], range=rng, bins=bins, color="C0",
                        smooth=1.0, plot_datapoints=False, fill_contours=True,
                        levels=(0.68, 0.95), hist_kwargs={"density": True})
    if overlay is not None:
        corner.corner(overlay, fig=fig, range=rng, bins=bins, color="C1", smooth=1.0,
                      plot_datapoints=False, plot_density=False, levels=(0.68, 0.95),
                      hist_kwargs={"density": True})
    handles = [Line2D([0], [0], color="C0", lw=2, label=label)]
    if overlay is not None:
        handles.append(Line2D([0], [0], color="C1", lw=2, label=overlay_label))
    if truth is not None:
        corner.overplot_lines(fig, truth, color="k", lw=1.2, ls="--")
        handles.append(Line2D([0], [0], color="k", ls="--", label="true value"))
    if best is not None:
        corner.overplot_points(fig, np.asarray(best)[None], marker="*", color="C3", ms=12)
        handles.append(Line2D([0], [0], marker="*", color="C3", ls="", ms=10, label="best fit"))
    fig.legend(handles=handles, loc="upper right", fontsize=10)
    return fig


def plot_trace(result, prior, max_chains: int = 16):
    """Trace plots (display coordinates) for the first ``max_chains`` chains."""
    import matplotlib.pyplot as plt
    disp = prior.to_display(result.theta.reshape(-1, prior.dim)).reshape(result.theta.shape)
    free = list(prior.free_idx)
    fig, axes = plt.subplots(len(free), 1, figsize=(8, 1.4 * len(free)), sharex=True)
    axes = np.atleast_1d(axes)
    for a, i in enumerate(free):
        for c in range(min(max_chains, disp.shape[0])):
            axes[a].plot(disp[c, :, i], lw=0.4, alpha=0.7)
        axes[a].set_ylabel(prior.labels[i], fontsize=8)
        axes[a].set_ylim(prior.display_bounds()[i])
    axes[-1].set_xlabel("draw")
    fig.tight_layout()
    return fig


def plot_marginals_compare(samples: dict, prior, truth=None, bins: int = 50):
    """1-D marginal histograms of several sample sets (e.g. NUTS vs emcee, trpl vs joint)."""
    import matplotlib.pyplot as plt
    free = list(prior.free_idx)
    D = len(free)
    ncol = min(D, 5)
    nrow = int(math.ceil(D / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.6 * nrow))
    axes = np.atleast_1d(axes).ravel()
    bnd = prior.display_bounds()
    for a, i in enumerate(free):
        for j, (name, s) in enumerate(samples.items()):
            axes[a].hist(s[:, i], bins=bins, range=tuple(bnd[i]), density=True, histtype="step",
                         lw=1.5, color=f"C{j}", label=name)
        if truth is not None:
            axes[a].axvline(truth[i], color="k", ls="--", lw=1)
        axes[a].set_xlabel(prior.labels[i])
        axes[a].set_yticks([])
    for ax in axes[D:]:
        ax.axis("off")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_profile(prof: pd.DataFrame, prior, mcmc_samples_disp=None, label=None, ax=None, color="C0"):
    """-2 Delta log L of a profile with chi^2_1 thresholds; optional MCMC marginal."""
    import matplotlib.pyplot as plt
    j = prior.names.index(prof.attrs.get("param", prof.columns[3]))
    scale = prior.Eg if prior.names[j].startswith("Et_rel") else 1.0
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.figure
    ax.plot(prof["value"] * scale, prof["delta_2nll"], "o-", color=color, ms=3, label=label or "profile")
    for thr, ls in ((1.0, ":"), (3.84, "--")):
        ax.axhline(thr, color="k", ls=ls, lw=0.8)
    ax.text(ax.get_xlim()[0], 1.0, " 68 %", va="bottom", fontsize=8)
    ax.text(ax.get_xlim()[0], 3.84, " 95 %", va="bottom", fontsize=8)
    ax.set_xlabel(prior.labels[j])
    ax.set_ylabel(r"$-2\,\Delta\log L$ (profile)")
    ax.set_ylim(-0.2, max(10, min(30, prof["delta_2nll"].max() * 1.05)))
    if mcmc_samples_disp is not None:
        ax2 = ax.twinx()
        ax2.hist(mcmc_samples_disp[:, j], bins=40, range=tuple(prior.display_bounds()[j]),
                 density=True, color=color, alpha=0.2)
        ax2.set_ylabel("MCMC marginal density")
    ax.legend(loc="upper center", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig, ax


def plot_start_sensitivity(table: pd.DataFrame, prior, fmin: float, tol: float = 0.5):
    """
    Optimiser sensitivity to the start values (task T2):
    left: final -log L - min per start; right: evaluations per start.
    """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    d = table["neg_log_lik"].to_numpy() - fmin
    is_phys = table["label"].str.contains("physics").to_numpy()
    order = np.argsort(d)
    ax[0].semilogy(np.arange(len(d)), d[order] + 1e-4, "o", color="0.5", ms=4)
    ax[0].semilogy(np.nonzero(is_phys[order])[0], d[order][is_phys[order]] + 1e-4, "*", color="C3",
                   ms=12, label="physics-informed")
    ax[0].axhline(tol, color="k", ls="--", lw=0.8, label=f"Δ(-log L) = {tol}")
    ax[0].set_xlabel("start (sorted)")
    ax[0].set_ylabel(r"$-\log L - \min + 10^{-4}$")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)
    ax[1].hist(table.loc[~is_phys, "n_evals"], bins=20, color="0.6", label="random (Sobol) starts")
    for v in table.loc[is_phys, "n_evals"]:
        ax[1].axvline(v, color="C3", lw=2, label="physics-informed")
    ax[1].set_xlabel("evaluations until termination")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    return fig, ax


# ─────────────────────────────────────────────────────────────────────────────
# export
# ─────────────────────────────────────────────────────────────────────────────

def _to_builtin(o):
    if isinstance(o, dict):
        return {str(k): _to_builtin(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_to_builtin(v) for v in o]
    if isinstance(o, np.ndarray):
        return _to_builtin(o.tolist())
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, float):
        return None if not math.isfinite(o) else o
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, pd.DataFrame):
        return o.reset_index().to_dict(orient="records")
    return o


def save_json(obj, path) -> Path:
    """Write a (nested) dict with numpy/pandas content as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(_to_builtin(obj), fh, indent=2)
    return path


def export_samples(result, prior, path) -> Path:
    """Posterior samples as CSV: chain, draw, theta columns and physical units."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    C, N, D = result.theta.shape
    flat = result.theta.reshape(-1, D)
    df = pd.DataFrame(flat, columns=prior.names)
    df.insert(0, "draw", np.tile(np.arange(N), C))
    df.insert(0, "chain", np.repeat(np.arange(C), N))
    df = pd.concat([df, prior.to_physical(flat)], axis=1)
    if "lp" in result.sample_stats:
        df["lp"] = result.sample_stats["lp"].reshape(-1)
    df.to_csv(path, index=False, float_format="%.8g")
    return path


def save_figure(fig, path, dpi: int = 200) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


def write_parameters_dat(path, prior, theta, title: str = "", extra: dict | None = None,
                         cov_theta=None) -> Path:
    """
    Human-readable parameter file (.dat): one parameter per line with value,
    unit, the theta coordinate and — if a covariance is given (e.g. from
    curve_fit) — a 1-sigma uncertainty in theta units.  Fixed parameters are
    marked.  ``extra`` adds "key = value" lines (errors, sigmas, method, ...).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    th = np.asarray(theta, np.float64)
    phys = prior.to_physical(th[None]).iloc[0]
    sd = np.sqrt(np.clip(np.diag(cov_theta), 0, None)) if cov_theta is not None else None
    units = {"krad": "cm^3/s", "Et": "eV", "Nt": "cm^-3", "tau_n": "s", "tau_p": "s"}
    lines = [f"# {title}".rstrip(), f"# written {pd.Timestamp.now().isoformat(timespec='seconds')}",
             f"# bandgap Eg = {prior.Eg} eV, T = {prior.T} K, traps = {prior.n_traps}",
             "# theta = log10(value) for k_rad, N_t, tau; theta = E_t/Eg for E_t.",
             "# +-1sd: Laplace (curve_fit) standard deviation — in decades (log10) for k_rad, N_t, tau and in eV for E_t.",
             "#",
             f"# {'parameter':<12s} {'value':>14s}  {'unit':<8s} {'theta':>12s}  {'+-1sd':>13s}  note"]
    order = [("krad", None)] + [(f, k) for k in range(1, prior.n_traps + 1) for f in ("Et", "Nt", "tau_n", "tau_p")]
    for field, k in order:
        key = "krad_cm3s" if field == "krad" else {"Et": f"Et_{k}_eV", "Nt": f"Nt_{k}_cm-3",
                                                     "tau_n": f"tau_n_{k}_s", "tau_p": f"tau_p_{k}_s"}[field]
        i = 0 if field == "krad" else {"Nt": 1, "Et": 2, "tau_n": 3, "tau_p": 4}[field] + 4 * (k - 1)
        name = field if k is None else f"{field}_{k}"
        val = phys[key]
        vtxt = f"{val:14.6f}" if field == "Et" else f"{val:14.6e}"
        note = "fixed" if prior.fixed_mask[i] else ""
        sd_i = (sd[i] * prior.Eg if field == "Et" else sd[i]) if sd is not None else None
        etxt = f"{sd_i:13.4g}" if sd is not None and not prior.fixed_mask[i] else f"{'':13s}"
        lines.append(f"  {name:<12s} {vtxt}  {units[field]:<8s} {th[i]:12.6f}  {etxt}  {note}")
    if extra:
        lines.append("#")
        for k_, v in extra.items():
            lines.append(f"# {k_} = {v}")
    path.write_text("\n".join(lines) + "\n")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# optimiser trajectories (CMA-ES on the NN, then curve_fit on the ODE)
# ─────────────────────────────────────────────────────────────────────────────

def _start_colors(labels, highlight="physics"):
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap("viridis")
    n = max(1, len(labels) - 1)
    cols, k = [], 0
    for lb in labels:
        if highlight and highlight in lb:
            cols.append("C3")
        else:
            cols.append(cmap(k / n))
            k += 1
    return cols


def plot_fit_histories(cma_results: list, ode_results: list | None = None,
                       title: str = "fit convergence", highlight: str = "physics"):
    """
    Best -log L so far for every start, in two panels with a common y axis.

    Left: CMA-ES on the NN likelihood vs NN evaluations.  Right: the
    subsequent curve_fit with the ODE vs ODE solves.  The switch from CMA-ES
    to curve_fit is the boundary between the panels; it is marked by a
    black-edged circle at the end of every CMA-ES curve (NN likelihood of the
    CMA-ES result) and at the start of every curve_fit curve (ODE likelihood
    of the same point) — the vertical offset between the two is the
    surrogate error at that point.  y = -log L - (overall minimum) + 1e-3.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    labels = [r.label for r in cma_results]
    cols = _start_colors(labels, highlight)
    vals = [r.neg_log_lik for r in cma_results] + ([r.neg_log_lik for r in ode_results] if ode_results else [])
    fmin = min(vals)
    if ode_results:
        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True,
                                      gridspec_kw={"width_ratios": [2, 1], "wspace": 0.04})
    else:
        fig, ax = plt.subplots(figsize=(7, 4.2))
        ax2 = None
    for i, r in enumerate(cma_results):
        hl = bool(highlight and highlight in r.label)
        lw, z, ms = (2.2, 5, 7) if hl else (0.9, 2, 4)
        ax.plot(r.history_evals, r.history_best - fmin + 1e-3, "-", color=cols[i], lw=lw, zorder=z)
        ax.plot([r.history_evals[-1]], [r.history_best[-1] - fmin + 1e-3], "o", mfc=cols[i], mec="k",
                ms=ms, zorder=z + 1)
        if ax2 is not None:
            o = ode_results[i]
            x = np.concatenate([[0], o.history_evals]) if o.history_evals is not None else np.array([0])
            y = np.concatenate([[o.start_neg_log_lik], o.history_best]) if o.history_best is not None \
                else np.array([o.start_neg_log_lik])
            ax2.plot(x, y - fmin + 1e-3, "--", color=cols[i], lw=lw, zorder=z)
            ax2.plot([0], [o.start_neg_log_lik - fmin + 1e-3], "o", mfc=cols[i], mec="k", ms=ms, zorder=z + 1)
            ax2.plot([x[-1]], [y[-1] - fmin + 1e-3], "*", color=cols[i], mec="k", mew=0.4, ms=ms + 4, zorder=z + 2)
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel("NN evaluations (CMA-ES)")
    ax.set_ylabel(r"$-\log L - \min + 10^{-3}$")
    ax.grid(alpha=0.3, which="both")
    handles = [Line2D([0], [0], color="0.4", lw=1.5, label="CMA-ES (NN likelihood)"),
               Line2D([0], [0], marker="o", color="w", mfc="0.6", mec="k", label="switch to curve_fit")]
    if ax2 is not None:
        ax2.set_xlabel("ODE solves (curve_fit)")
        ax2.grid(alpha=0.3, which="both")
        ax2.axvline(0, color="k", lw=1.0)
        handles += [Line2D([0], [0], color="0.4", lw=1.5, ls="--", label="curve_fit (ODE likelihood)"),
                    Line2D([0], [0], marker="*", color="w", mfc="0.6", mec="k", ms=10, label="end")]
    if highlight and any(highlight in lb for lb in labels):
        handles.append(Line2D([0], [0], color="C3", lw=2.2, label="physics-informed start"))
    (ax2 or ax).legend(handles=handles, fontsize=8, loc="upper right")
    fig.suptitle(title)
    fig.tight_layout()
    return fig, (ax, ax2)


def plot_trajectory_corner(prior, cma_results: list, ode_results: list | None = None, truth=None,
                           highlight: str = "physics", title: str = "optimiser trajectories"):
    """
    Corner plot of the optimiser paths of all starts (display coordinates,
    free parameters only).

    Lower panels: best point per generation as a connected curve for every
    start (solid = CMA-ES, dashed = curve_fit with the ODE), start (open
    circle), switch to curve_fit (black-edged dot) and end point (star).
    Diagonal: histograms of the start values (grey) and of the final values
    (colour).  The physics-informed start is drawn in red.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    free = list(prior.free_idx)
    nd = len(free)
    labels = [r.label for r in cma_results]
    cols = _start_colors(labels, highlight)
    bnd = prior.display_bounds()
    disp = lambda th: prior.to_display(np.atleast_2d(th))
    fig, axes = plt.subplots(nd, nd, figsize=(2.3 * nd, 2.3 * nd))
    axes = np.atleast_2d(axes)
    finals = []
    for s, r in enumerate(cma_results):
        hl = highlight and highlight in r.label
        lw, z = (1.8, 5) if hl else (0.7, 2)
        path_c = disp(r.history_theta) if r.history_theta is not None else disp(r.theta)
        path_c = np.vstack([disp(r.start_theta), path_c])
        path_o = None
        if ode_results:
            o = ode_results[s]
            path_o = np.vstack([disp(o.start_theta), disp(o.history_theta)]) if o.history_theta is not None else disp(o.theta)
        finals.append(disp(ode_results[s].theta if ode_results else r.theta)[0])
        for a in range(nd):
            for b in range(nd):
                ax = axes[a, b]
                if b > a:
                    continue
                if a == b:
                    continue
                i, j = free[b], free[a]           # x = column parameter, y = row parameter
                ax.plot(path_c[:, i], path_c[:, j], "-", color=cols[s], lw=lw, alpha=0.9, zorder=z)
                ax.plot(path_c[0, i], path_c[0, j], "o", mfc="none", mec=cols[s], ms=4, zorder=z)
                if path_o is not None:
                    ax.plot(path_o[:, i], path_o[:, j], "--", color=cols[s], lw=lw, zorder=z)
                    ax.plot(path_o[0, i], path_o[0, j], "o", mfc=cols[s], mec="k", ms=4, zorder=z + 1)
                    ax.plot(path_o[-1, i], path_o[-1, j], "*", color=cols[s], mec="k", mew=0.4, ms=9, zorder=z + 2)
                else:
                    ax.plot(path_c[-1, i], path_c[-1, j], "*", color=cols[s], mec="k", mew=0.4, ms=9, zorder=z + 2)
    finals = np.array(finals)
    starts = disp(np.stack([r.start_theta for r in cma_results]))
    for a in range(nd):
        for b in range(nd):
            ax = axes[a, b]
            if b > a:
                ax.axis("off")
                continue
            i, j = free[b], free[a]
            ax.set_xlim(bnd[i])
            if a == b:
                ax.hist(starts[:, i], bins=20, range=tuple(bnd[i]), color="0.75", label="start")
                ax.hist(finals[:, i], bins=20, range=tuple(bnd[i]), color="C0", alpha=0.7, label="final")
                if truth is not None:
                    ax.axvline(truth[i], color="k", ls=":", lw=1)
                ax.set_yticks([])
            else:
                ax.set_ylim(bnd[j])
                if truth is not None:
                    ax.plot(truth[i], truth[j], "x", color="k", ms=8, mew=2, zorder=10)
            if a == nd - 1:
                ax.set_xlabel(prior.labels[i], fontsize=9)
            else:
                ax.set_xticklabels([])
            if b == 0 and a != b:
                ax.set_ylabel(prior.labels[j], fontsize=9)
            elif a != b:
                ax.set_yticklabels([])
            ax.tick_params(labelsize=7)
    handles = [Line2D([0], [0], color="0.4", lw=1.2, label="CMA-ES (NN), best per generation"),
               Line2D([0], [0], marker="o", color="w", mfc="none", mec="0.4", label="start")]
    if ode_results:
        handles += [Line2D([0], [0], color="0.4", lw=1.2, ls="--", label="curve_fit (ODE)"),
                    Line2D([0], [0], marker="o", color="w", mfc="0.6", mec="k", label="switch to curve_fit")]
    handles += [Line2D([0], [0], marker="*", color="w", mfc="0.6", mec="k", ms=10, label="end")]
    if highlight and any(highlight in lb for lb in labels):
        handles.append(Line2D([0], [0], color="C3", lw=1.8, label="physics-informed start"))
    if truth is not None:
        handles.append(Line2D([0], [0], marker="x", color="k", ls="", label="true value"))
    fig.legend(handles=handles, loc="upper right", fontsize=9)
    fig.suptitle(title)
    fig.tight_layout()
    return fig
