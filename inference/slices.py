"""
slices.py — the sliced grid of the original submission, on the current network and likelihood
=============================================================================================
15/10/2026 (revision).  Re-implements `uncertainty_analysis/01_create_grid.py --mode slice` and the slice part of
`04_bayesian.py` (paper version, git HEAD of the simulations repository):

* one axis per free parameter with n points (paper: 100) spanning the prior range, shifted so that the best fit is
  an exact node (`make_anchored_grid`); axes in decades (E_t -> log10 n1, linear in E_t, as in every corner here);
* for every PAIR of free parameters a 2-D slice n x n through the best fit, all other parameters fixed at the best
  fit (no marginalisation: a slice, not a projection);
* log L = sum_d [-MSE_d / (2 sigma_eff,d^2) - log(sigma_eff,d sqrt(2 pi))] with
  sigma_eff,d = sqrt(sigma_d^2 + MSE_min,d) and MSE_min,d the smallest MSE over all slice points (paper convention);
* per panel: probability normalised inside the slice (sum = 1) and log10 of the likelihood ratio to the best-fit
  node (`relative_prob` of the paper);
* colour (default since 19/10/2026, Robin): rel_drop = (p(best fit) - p) / p(best fit) = 1 - exp(log L - log L_ref),
  one global scale (the best-fit node is the reference of every panel);
* colour of the paper figure (`_percentile_levels(col="relative_prob")`, option --colour rank): the rank of relative_prob among ALL slice
  points with relative_prob > 0 in float64 (exp underflows below about -745 nats; those points are not ranked),
  pooled over the panels: rank 0.01 = "best 1 %" ... 0.9 = "best 90 %".  Stored as `rank` (0 = best, 1 = not ranked).
* iso-tau_0 lines (19/10/2026, Robin) in the n1_k - tau_p,k panels: tau_0 = tau_p a / (1 + k_rad tau_p n1) with
  a = sqrt(n1 N_t)/n_i for N_t >= 10 n1 (photodoping) and a = n1/n_i for N_t <= 0.1 n1, tau_0 of the shallow regime
  of the data (`tau0_shallow`), N_t and k_rad at the best fit (the values of the slice).
"""
from __future__ import annotations

import math
import time
from itertools import combinations

import numpy as np
from scipy.special import logsumexp

LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)


def anchored_axis(lo: float, hi: float, n: int, anchor: float):
    """n points with the spacing of linspace(lo, hi, n), shifted so that `anchor` is a node (paper)."""
    step = (hi - lo) / (n - 1)
    idx = int(np.clip(round((anchor - lo) / step), 0, n - 1))
    return anchor + step * (np.arange(n) - idx), idx


def run_slices(log_post, prior, theta_best, n: int = 100, sigma_base: dict | None = None, add_mse_min: bool = True):
    """
    2-D slices through theta_best for every pair of free parameters (decade coordinates, see cylinder.DecadeMap).

    log_post   : likelihood.LogPosterior of the mode (its MSE terms are used; its sigmas are NOT)
    sigma_base : base sigma per term {'trpl': .., 'sspl': ..} (paper: bayesian.sigma_trpl / sigma_sspl)
    add_mse_min: paper convention sigma_eff = sqrt(sigma_base^2 + MSE_min over the slice points); False = sigma_base
                 as it is (runs with sigma_rule 'model_plus_nn': sigma = sqrt(sigma_model^2 + sigma_nn^2))
    Returns dict(keys, axes {key: values}, anchor_idx, panels {(kx, ky): dict(x, y, logL (ny, nx), prob, log10_rel)},
                 sigma (per term: base, mse_min, eff), n_evals, time_s, logL_ref).
    """
    from .cylinder import DecadeMap
    t0 = time.perf_counter()
    dmap = DecadeMap(prior)
    keys = dmap.keys
    x_best = dmap.to_x(np.asarray(theta_best, float)[None])[0]
    xlo, xhi = dmap.to_x(prior.lo[None])[0], dmap.to_x(prior.hi[None])[0]
    axes, anchor = {}, {}
    for i, k in enumerate(keys):
        axes[k], anchor[k] = anchored_axis(min(xlo[i], xhi[i]), max(xlo[i], xhi[i]), n, x_best[i])

    blocks, X_all = [], []
    for i, j in combinations(range(len(keys)), 2):
        gx, gy = np.meshgrid(axes[keys[i]], axes[keys[j]])          # (ny, nx): rows = y
        X = np.repeat(x_best[None], gx.size, axis=0)
        X[:, i], X[:, j] = gx.ravel(), gy.ravel()
        blocks.append((i, j, gx.shape))
        X_all.append(X)
    X_all = np.vstack(X_all)
    TH = dmap.to_theta(X_all, theta_best)
    # tol: the decade round trip moves a best fit ON a bound by ~1e-16 outside it (77:23 F6, E_t2 = 0.98 E_g), which
    # made every slice through it -inf (27/10/2026).  Ties E_t1 = E_t2 (a best fit on the ordering constraint, 85:15
    # tr-PL fit) count as ordered: trap k's E_t is shifted by k x 1e-12 for the support test only.
    TH_s = TH.copy()
    if getattr(prior, "order_traps", False):
        TH_s[:, prior.idx_Et] += 1e-12 * np.arange(len(prior.idx_Et))
    ok = prior.in_support(TH_s, tol=1e-11)
    mse = {k: np.full(len(TH), np.nan) for k in ([t.name for t in log_post.trpl_terms]
                                                + ([log_post.sspl_term.name] if log_post.sspl_term else []))}
    if ok.any():
        m = log_post.np_mse_terms(TH[ok])
        for k in mse:
            mse[k][ok] = m[k]
    sig = {}
    logL = np.zeros(len(TH))
    for k, v in mse.items():
        fin = v[np.isfinite(v)]
        mse_min = float(fin.min()) if len(fin) else 0.0
        s_eff = math.sqrt(float(sigma_base[k]) ** 2 + (mse_min if add_mse_min else 0.0))
        sig[k] = {"base": float(sigma_base[k]), "mse_min": mse_min, "eff": s_eff}
        logL += -v / (2.0 * s_eff ** 2) - math.log(s_eff) - LOG_SQRT_2PI
    logL = np.where(np.isfinite(logL) & ok, logL, -np.inf)

    panels, start, refs = {}, 0, []
    for (i, j, shape) in blocks:
        L = logL[start:start + shape[0] * shape[1]].reshape(shape)
        start += shape[0] * shape[1]
        refs.append(L[anchor[keys[j]], anchor[keys[i]]])
        panels[(keys[i], keys[j])] = {"x": axes[keys[i]], "y": axes[keys[j]], "logL": L}
    logL_ref = float(np.max(refs))                                 # the best-fit node (identical in every panel)
    for p in panels.values():
        L = p["logL"]
        fin = np.isfinite(L)
        lz = logsumexp(L[fin]) if fin.any() else 0.0
        p["prob"] = np.where(fin, np.exp(L - lz), 0.0)
        p["log10_rel"] = np.where(fin, (L - logL_ref) / math.log(10.0), np.nan)
        # (p(best fit) - p) / p(best fit): 0 at the best-fit node, the same reference in every panel (Robin, 19/10/2026)
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            p["rel_drop"] = np.where(fin, np.clip(1.0 - np.exp(np.minimum(L - logL_ref, 0.0)), 0.0, 1.0), 1.0)
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            p["ranked"] = fin & (np.exp(np.where(fin, L - logL_ref, -np.inf)) > 0)     # paper: relative_prob > 0
    pool = np.sort(np.concatenate([p["log10_rel"][p["ranked"]] for p in panels.values()]))[::-1]
    for p in panels.values():                    # fraction of the pooled points with a higher relative probability
        v = p["log10_rel"]
        r = np.searchsorted(-pool, -np.where(p["ranked"], v, -np.inf), side="right") / max(1, len(pool))
        p["rank"] = np.where(p["ranked"], r, 1.0)
    return {"keys": keys, "axes": axes, "anchor_idx": anchor, "panels": panels, "sigma": sig,
            "n_evals": int(ok.sum()), "n_points": len(TH), "time_s": time.perf_counter() - t0, "logL_ref": logL_ref,
            "n_ranked": int(len(pool)),
            "dmap": dmap, "x_best": x_best}


def layout_slices(keys, extra_panels=("tau_qfls", "pl_t")):
    """
    Lower triangle WITHOUT diagonal: (D-1) x (D-1) 2-D panels, panel (r, c) = x keys[c], y keys[r+1]; x tick labels on
    the bottom row, y on the first column; E_t - E_g axis on top of every n1 column (its topmost panel, row = col).
    The empty upper-right triangle holds the curve panels (kind 'curve', kx = panel name, extra 'span:<rows>x<cols>').
    """
    from .corner_export import LAYOUT_DTYPE
    D = len(keys)
    m = D - 1
    rows = []
    for r in range(m):
        for c in range(r + 1):
            et = b"top:Et-Eg" if (r == c and keys[c].startswith("log10_n1")) else b""
            rows.append((r, c, b"2d", keys[c].encode(), keys[r + 1].encode(),
                         b"bottom" if r == m - 1 else b"none", b"left" if c == 0 else b"none", et))
    free = [(r, c) for r in range(m) for c in range(r + 1, m)]           # upper-right cells
    extra = list(extra_panels)
    if not free or not extra:
        return np.array(rows, dtype=LAYOUT_DTYPE)
    if m == 2:                                                            # 3 parameters: one free cell
        rows.append((0, 1, b"curve", extra[0].encode(), b"", b"bottom", b"right", b"span:1x1"))
        return np.array(rows, dtype=LAYOUT_DTYPE)
    if len(extra) == 2:            # wide tau(QFLS) panel over the top row, PL(t) below at the right edge
        rows.append((0, 1, b"curve", extra[0].encode(), b"", b"bottom", b"right", f"span:1x{m - 1}".encode()))
        rows.append((1, m - 1, b"curve", extra[1].encode(), b"", b"bottom", b"right", b"span:1x1"))
    else:                          # one panel per free cell of the top rows
        cells = [(0, c) for c in range(1, m)] + [(1, c) for c in range(2, m)]
        for (r, c), name in zip(cells, extra):
            rows.append((r, c, b"curve", name.encode(), b"", b"bottom", b"right", b"span:1x1"))
    return np.array(rows, dtype=LAYOUT_DTYPE)


def tau0_shallow(qfls, tau_diff, kT: float, mask=None) -> float:
    """
    tau_0 of the shallow-trap regime tau_diff = tau_0 exp(-QFLS / (2 kT)): exp(mean(ln tau_diff + QFLS / (2 kT))) over
    the points in `mask` (legacy start-point estimator, preprocessing/start_point_estimation_module.py).
    """
    q, tau = np.asarray(qfls, float), np.asarray(tau_diff, float)
    m = np.isfinite(q) & np.isfinite(tau) & (tau > 0) & (np.ones(len(q), bool) if mask is None else np.asarray(mask, bool))
    if m.sum() < 2:
        raise ValueError(f"tau_0: only {int(m.sum())} data points in the shallow regime")
    return float(np.exp(np.mean(np.log(tau[m]) + q[m] / (2.0 * kT))))


def tau0_of_taup(n1, taup, Nt, krad, ni, photodoping: bool):
    """Robin, 19/10/2026: tau_0 = tau_p a / (1 + k_rad tau_p n1), a = sqrt(n1 N_t)/n_i (photodoping, N_t >= 10 n1) or
    n1/n_i (N_t <= 0.1 n1)."""
    n1, taup = np.asarray(n1, float), np.asarray(taup, float)
    a = np.sqrt(n1 * Nt) / ni if photodoping else n1 / ni
    return taup * a / (1.0 + krad * taup * n1)


def taup_of_tau0(n1, tau0, Nt, krad, ni, photodoping: bool):
    """Inverse of tau0_of_taup: tau_p = tau_0 / (a - k_rad n1 tau_0); NaN where a <= k_rad n1 tau_0 (no tau_p gives tau_0)."""
    n1 = np.asarray(n1, float)
    a = np.sqrt(n1 * Nt) / ni if photodoping else n1 / ni
    den = a - krad * n1 * tau0
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, tau0 / den, np.nan)


TAU0_REGIMES = {"photodoping": "N_t >= 10 n1", "non_photodoping": "N_t <= 0.1 n1"}


def tau0_lines(x_range, y_range, tau0, Nt, krad, ni, n: int = 1000):
    """
    Iso-tau_0 lines in the (log10 n1, log10 tau_p) plane of a slice (N_t, k_rad: the slice's fixed values), each only
    inside its regime: 'photodoping' for n1 <= N_t / 10, 'non_photodoping' for n1 >= 10 N_t.  Returns
    {regime: (m, 2) array [log10 n1, log10 tau_p]}, points outside y_range NaN (Origin draws gaps), (0, 2) if empty.
    """
    lx = math.log10(Nt)
    edges = {"photodoping": (x_range[0], min(x_range[1], lx - 1.0)),
             "non_photodoping": (max(x_range[0], lx + 1.0), x_range[1])}
    out = {}
    for reg, (a, b) in edges.items():
        if b <= a:
            out[reg] = np.zeros((0, 2))
            continue
        x = np.linspace(a, b, n)
        y = np.log10(taup_of_tau0(10.0 ** x, tau0, Nt, krad, ni, reg == "photodoping"))
        y = np.where((y >= y_range[0]) & (y <= y_range[1]), y, np.nan)
        ok = np.flatnonzero(np.isfinite(y))
        out[reg] = (np.column_stack([x, y])[ok[0]:ok[-1] + 1] if len(ok) else np.zeros((0, 2)))
    return out


def nn_time_axis(qfls, tau_diff, t_at, q0: float, kT: float, q_max: float | None = None):
    """
    t and PL/PL_0 of a tau_diff(QFLS) curve: dt = -tau_diff/(2 kT) dQFLS (tau_diff = -2 / (d ln PL / dt) and
    PL ~ np ~ exp(QFLS/kT)); the first point (highest QFLS <= q_max) is put at t_at(QFLS) — the time of the ODE
    solution of the same parameter set at that QFLS — and PL/PL_0 = exp((QFLS - q0)/kT), q0 = QFLS at t = 0 of
    that ODE solution.  Returns (qfls, tau, t, pl) sorted by time.
    """
    q = np.asarray(qfls, float)
    tau = np.asarray(tau_diff, float)
    keep = np.isfinite(q) & np.isfinite(tau) & (tau > 0)
    if q_max is not None:
        keep &= q <= q_max + 1e-12
    q, tau = q[keep], tau[keep]
    o = np.argsort(-q)                                                    # decreasing QFLS = increasing time
    q, tau = q[o], tau[o]
    dt = 0.5 * (tau[1:] + tau[:-1]) / (2.0 * kT) * (q[:-1] - q[1:])
    t = float(t_at(q[0])) + np.concatenate([[0.0], np.cumsum(dt)])
    return q, tau, t, np.exp((q - q0) / kT)
