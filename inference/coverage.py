"""
coverage.py — realistic parameter recovery and interval coverage (task T6)
=========================================================================

Unlike SBC (sbc.py), this test mimics the real analysis chain:

1. theta_true ~ prior (incl. E_t ordering);
2. PL(t) from the rate-equation ODE (the solver that generated the training
   data) on the time grid of the measured transient;
3. noise at the PL level: multiplicative (log-normal) noise with the scatter
   of the measured transient around its spline, and the detection floor
   (dynamic range) of the measurement — points below it are removed;
4. the REAL preprocessing (trim at the peak, monotone spline, conversion to
   tau_diff vs QFLS, preprocessing/data_utils.py);
5. optionally a systematic model-error term s(QFLS) * delta, delta ~ N(0, 1)
   per decay, with s from model-error-analysis/ (sigma_all_folders.csv);
6. the complete workflow: multi-start CMA-ES (NN), sigma_eff per decay,
   NUTS;
7. coverage = fraction of decays whose true parameter lies inside the 68 % /
   95 % equal-tailed interval.  With the conservative MSE likelihood and
   sigma_eff, coverage >= nominal is expected (report it as coverage, not as
   SBC).

The ODE refinement (curve_fit) of the main workflow is not part of this test
(the posterior uses the NN likelihood and sigma_eff the NN best fit).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import REPO_ROOT


# ─────────────────────────────────────────────────────────────────────────────
# noise model from the measured transient
# ─────────────────────────────────────────────────────────────────────────────

def estimate_noise(raw_df: pd.DataFrame, n_knots: int = 5, n_pred: int = 256) -> dict:
    """
    Noise model of a measured transient (time [ns] index, 'pl' column):
    robust scatter (1.4826 MAD) of log10 PL around the monotone spline, the
    detection floor (smallest positive PL after normalisation) and the time
    grid after trimming at the peak.
    """
    import contextlib
    import io
    from . import add_legacy_paths
    add_legacy_paths()
    import data_utils as utils
    trimmed = utils.trim_and_normalize(raw_df.copy())
    with contextlib.redirect_stdout(io.StringIO()):
        spline = utils.spline_torch(trimmed, n_points=n_knots, degree=3, n_pred=n_pred)
    t = trimmed.index.to_numpy(float)
    pl = trimmed["pl"].to_numpy(float)
    ts, ps = spline.index.to_numpy(float), spline["pl"].to_numpy(float)
    ok = (t > 0) & (pl > 0)
    pred = np.interp(np.log10(t[ok]), np.log10(np.maximum(ts, 1e-6)), np.log10(ps))
    res = np.log10(pl[ok]) - pred
    rel = float(1.4826 * np.median(np.abs(res - np.median(res))))
    return {"rel_sd_log10": rel, "floor": float(pl[pl > 0].min()), "t_grid_ns": t,
            "n_points": int(len(t)), "dynamic_range_decades": float(-np.log10(pl[pl > 0].min()))}


def load_model_error(path=None):
    """s(QFLS) [log10 tau] of the diffusion/model-error analysis (callable, 0 outside)."""
    if path is None:
        path = REPO_ROOT / "model-error-analysis" / "derivative_analysis_output" / "sigma_all_folders.csv"
    df = pd.read_csv(path).dropna()
    q, s = df.iloc[:, 0].to_numpy(float), df.iloc[:, 1].to_numpy(float)
    return lambda x: np.interp(x, q, s, left=0.0, right=0.0)


# ─────────────────────────────────────────────────────────────────────────────
# synthetic decays
# ─────────────────────────────────────────────────────────────────────────────

def _simulate_one(theta, prior, npulse, constants, t_grid_ns, rel_sd, floor, n_knots, n_pred,
                  seed, T=300.0):
    """ODE -> noisy PL(t) on the measured grid -> real preprocessing (runs in a worker)."""
    import contextlib
    import io
    import sys
    import warnings
    warnings.filterwarnings("ignore")
    from . import add_legacy_paths
    add_legacy_paths()
    import data_utils as utils
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass
    rng = np.random.default_rng(seed)
    from .ode import solve
    phys = prior.to_physical(np.asarray(theta)[None]).iloc[0]
    K = prior.n_traps
    get = lambda key: np.array([phys[f"{key}_{k}" + ("_cm-3" if key == "Nt" else "_eV" if key == "Et" else "_s")]
                                for k in range(1, K + 1)])
    try:
        with np.errstate(all="ignore"):
            out = solve(constants, npulse, prior.Eg, phys["krad_cm3s"],
                        get("Nt"), get("Et"), get("tau_n"), get("tau_p"), T)
        t_s, pl = out[0], out[8]
        ok = np.isfinite(pl) & (pl > 0) & (t_s >= 0)
        t_s, pl = t_s[ok], pl[ok] / np.max(pl[ok])
        tg = np.asarray(t_grid_ns, float) * 1e-9
        lt = np.log10(np.maximum(t_s, 1e-15))
        lpl = np.interp(np.log10(np.maximum(tg, 1e-15)), lt, np.log10(pl), right=-np.inf)
        lpl[tg > t_s.max()] = -np.inf
        noisy = lpl + rel_sd * rng.standard_normal(len(lpl))
        keep = np.isfinite(noisy) & (noisy >= math.log10(floor)) & (lpl >= math.log10(floor))
        if keep.sum() < 15:
            return {"ok": False, "reason": f"only {int(keep.sum())} points above the floor"}
        df = pd.DataFrame({"time": np.asarray(t_grid_ns)[keep], "pl": 10 ** noisy[keep]}).set_index("time")
        trimmed = utils.trim_and_normalize(df.copy())
        with contextlib.redirect_stdout(io.StringIO()):
            spline = utils.spline_torch(trimmed, n_points=n_knots, degree=3, n_pred=n_pred)
        proc = utils.t_pl_to_qfls_tau(spline, npulse, prior.Eg)
        q, tau = proc["qfls"].to_numpy(float), proc["tau_diff"].to_numpy(float)
        good = np.isfinite(q) & np.isfinite(tau) & (tau > 0)
        if good.sum() < 10:
            return {"ok": False, "reason": "preprocessing left < 10 points"}
        o = np.argsort(q[good])
        return {"ok": True, "qfls": q[good][o], "tau": tau[good][o], "n_raw": int(keep.sum())}
    except Exception as e:                     # ODE or spline failure
        return {"ok": False, "reason": f"{type(e).__name__}: {e}"}


def make_synthetic_datasets(prior, npulse: float, constants: dict, noise: dict, n: int,
                            n_knots: int = 5, n_pred: int = 256, model_error=None,
                            n_jobs: int = 16, seed: int = 0, T: float = 300.0) -> list:
    """
    n synthetic decays (theta ~ prior).  Returns a list of dicts with
    theta_true, qfls, tau (preprocessed) or ok=False and a reason.
    ``model_error``: callable s(QFLS) or None.
    """
    from joblib import Parallel, delayed
    rng = np.random.default_rng(seed)
    thetas = prior.sample(n, rng=rng)
    seeds = rng.integers(0, 2**31 - 1, size=n)
    res = Parallel(n_jobs=n_jobs)(delayed(_simulate_one)(
        thetas[i], prior, npulse, constants, noise["t_grid_ns"], noise["rel_sd_log10"], noise["floor"],
        n_knots, n_pred, int(seeds[i]), T) for i in range(n))
    for i, r in enumerate(res):
        r["theta_true"] = thetas[i]
        if r["ok"] and model_error is not None:
            delta = rng.standard_normal()
            r["model_error_delta"] = float(delta)
            r["tau"] = r["tau"] * 10 ** (model_error(r["qfls"]) * delta)
    return res


# ─────────────────────────────────────────────────────────────────────────────
# workflow on all decays + coverage
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CoverageResult:
    table: pd.DataFrame               # one row per decay
    summary: pd.DataFrame             # coverage per parameter / derived quantity
    info: dict = field(default_factory=dict)


def _wilson(k, n, z=1.0):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def run_coverage(surrogate, prior, datasets: list, sigma_base: float, use_sigma_eff: bool = True,
                 n_starts: int = 4, cma_max_evals: int = 5000, chains_per_dataset: int = 4,
                 n_warmup: int = 500, n_draws: int = 1000, batch_datasets: int = 100,
                 target_accept: float = 0.95, seed: int = 0, verbose: bool = True) -> CoverageResult:
    """Full workflow on every usable synthetic decay and the interval coverage."""
    from .multi import MultiObjective, MultiTrplLikelihood
    from .optimize import cmaes_multistart
    from .report import derived_samples
    from .samplers import NutsSampler, init_dispersed

    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    use = [d for d in datasets if d.get("ok")]
    n = len(use)
    multi = MultiTrplLikelihood(surrogate, prior, [(d["qfls"], np.log10(d["tau"])) for d in use], sigma_base)
    theta_true = np.stack([d["theta_true"] for d in use])
    # 1) fit: n_starts Sobol starts per decay, all CMA-ES instances in lockstep
    starts = np.tile(prior.sobol(n_starts, seed=seed + 1), (n, 1))
    set_of_start = np.repeat(np.arange(n), n_starts)
    opt = cmaes_multistart(MultiObjective(multi, set_of_start), starts, max_evals=cma_max_evals,
                           verbose=False)
    nll = np.array([r.neg_log_lik for r in opt]).reshape(n, n_starts)
    best = np.argmin(nll, axis=1)
    theta_fit = np.stack([opt[d * n_starts + best[d]].theta for d in range(n)])
    mse_min = multi.np_mse_theta(theta_fit, np.arange(n))
    sig = np.sqrt(sigma_base ** 2 + mse_min) if use_sigma_eff else np.full(n, sigma_base)
    multi.set_sigma(sig)
    if verbose:
        print(f"coverage: fits done for {n} decays ({time.perf_counter() - t0:.0f} s); "
              f"median sigma_eff = {np.median(sig):.4f}")
    # 2) NUTS, batch of decays per run
    rows = []
    D = prior.dim
    der_true = derived_samples(theta_true, prior)
    for b0 in range(0, n, batch_datasets):
        ids = np.arange(b0, min(n, b0 + batch_datasets))
        groups = np.repeat(np.arange(len(ids)), chains_per_dataset)
        bound = multi.bind(ids[groups])
        sampler = NutsSampler(bound, n_chains=len(groups), groups=groups, target_accept=target_accept)
        res = sampler.run(init_dispersed(D, len(groups), rng), n_warmup=n_warmup, n_draws=n_draws,
                          seed=int(rng.integers(1 << 30)), verbose=verbose)
        for g, d in enumerate(ids):
            ch = np.nonzero(groups == g)[0]
            draws = res.theta[ch].reshape(-1, D)
            row = {"decay": int(d), "sigma_eff": float(sig[d]), "mse_fit": float(mse_min[d]),
                   "n_points": int(multi.n_points[d]),
                   "divergences": int(res.sample_stats["diverging"][ch].sum())}
            disp_draws, disp_true = prior.to_display(draws), prior.to_display(theta_true[d])[0]
            for i in prior.free_idx:
                k = prior.display_keys[i]
                q = np.percentile(disp_draws[:, i], [2.5, 16, 50, 84, 97.5])
                row.update({f"{k}_true": disp_true[i], f"{k}_median": q[2],
                            f"{k}_in68": bool(q[1] <= disp_true[i] <= q[3]),
                            f"{k}_in95": bool(q[0] <= disp_true[i] <= q[4]),
                            f"{k}_pit": float(np.mean(disp_draws[:, i] < disp_true[i]))})
            dd = derived_samples(draws, prior)
            for c in dd.columns:
                q = np.percentile(dd[c], [2.5, 16, 84, 97.5])
                v = der_true.loc[d, c]
                row.update({f"{c}_in68": bool(q[1] <= v <= q[2]), f"{c}_in95": bool(q[0] <= v <= q[3]),
                            f"{c}_pit": float(np.mean(dd[c] < v))})
            rows.append(row)
        if verbose:
            print(f"coverage: NUTS {ids[-1] + 1}/{n} decays ({time.perf_counter() - t0:.0f} s)")
    table = pd.DataFrame(rows)
    names = [prior.display_keys[i] for i in prior.free_idx] + list(der_true.columns)
    summ = []
    for k in names:
        r = {"quantity": k}
        for lev in ("68", "95"):
            hits = table[f"{k}_in{lev}"].to_numpy(bool)
            lo, hi = _wilson(hits.sum(), len(hits))
            r.update({f"coverage{lev}": hits.mean(), f"ci{lev}_lo": lo, f"ci{lev}_hi": hi})
        summ.append(r)
    info = {"n_decays_used": n, "n_decays_total": len(datasets),
            "failed": [d.get("reason") for d in datasets if not d.get("ok")],
            "sigma_base": sigma_base, "use_sigma_eff": use_sigma_eff, "n_starts": n_starts,
            "time_s": time.perf_counter() - t0}
    return CoverageResult(table=table, summary=pd.DataFrame(summ).set_index("quantity"), info=info)


def plot_coverage(cov: CoverageResult, title: str = "coverage of the credible intervals"):
    """Coverage per quantity (with 68 % Wilson intervals) against the nominal levels."""
    import matplotlib.pyplot as plt
    s = cov.summary
    x = np.arange(len(s))
    fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(s) + 2), 4))
    for lev, col, off in (("68", "C0", -0.15), ("95", "C1", 0.15)):
        ax.errorbar(x + off, s[f"coverage{lev}"], yerr=[s[f"coverage{lev}"] - s[f"ci{lev}_lo"],
                                                         s[f"ci{lev}_hi"] - s[f"coverage{lev}"]],
                    fmt="o", color=col, label=f"{lev} % interval")
        ax.axhline(int(lev) / 100, color=col, ls="--", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(s.index, rotation=60, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("fraction of decays covered")
    ax.set_title(f"{title} (n = {cov.info['n_decays_used']})")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_pit(cov: CoverageResult, prior, n_bins: int = 10):
    """Histograms of the PIT values (posterior CDF at the true value) per parameter."""
    import matplotlib.pyplot as plt
    keys = [prior.display_keys[i] for i in prior.free_idx]
    ncol = min(len(keys), 5)
    nrow = int(math.ceil(len(keys) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.6 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for a, k in enumerate(keys):
        axes[a].hist(cov.table[f"{k}_pit"], bins=n_bins, range=(0, 1), color="C0", edgecolor="w")
        axes[a].axhline(len(cov.table) / n_bins, color="0.4", lw=0.8)
        axes[a].set_title(k, fontsize=9)
        axes[a].set_yticks([])
    for ax in axes[len(keys):]:
        ax.axis("off")
    fig.suptitle("PIT of the true values (uniform = calibrated, hump = conservative)")
    fig.tight_layout()
    return fig
