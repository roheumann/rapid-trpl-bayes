"""
grid.py — full-dimensional grid evaluation of the posterior
===========================================================

A deterministic reference for the samplers: the SAME likelihood is evaluated
on every point of a D-dimensional Cartesian grid (all parameters vary at
once — not the pairwise slices of the original submission), and normalised
1-D and 2-D marginal masses are accumulated.

* Grid: ``n_points`` values per parameter (default ~10, user setting),
  either in a window around a centre (the physics-informed fit) or over the
  full prior box.  Fixed parameters get a single grid value.
* The prior is uniform in theta, so every grid point has the same prior
  weight; points outside the support (E_t ordering, photodoping) get weight 0
  and are not evaluated.
* The grid is streamed in chunks; nothing of size n^D is stored.  Marginals
  are accumulated with a running maximum of log L (numerically stable).
* Masses on a grid are probabilities of grid CELLS, not densities.  Credible
  intervals are read from the cumulative 1-D mass with linear interpolation
  between grid values, so their resolution is the grid spacing.
* ``edge_mass`` reports the 1-D mass on the outermost grid values: a large
  value means the window truncates the posterior.

Cost: n^D likelihood evaluations (10^5 for one trap, 10^9 for two traps at
n = 10; the notebook prints an estimate before running).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class GridResult:
    axes: list                      # D arrays of grid values (theta coordinates)
    log_z: float                    # log sum_grid L (uniform prior weights)
    marg1: list                     # D arrays: normalised 1-D masses
    marg2: dict                     # (i, j) -> normalised 2-D masses (n_i, n_j), i < j
    theta_max: np.ndarray           # grid point with the largest log L
    log_lik_max: float
    n_points: int                   # total grid points
    n_evaluated: int                # points inside the support (likelihood evaluated)
    time_s: float
    edge_mass: np.ndarray = field(default=None)   # (D, 2) mass on the first/last grid value


def grid_axes(prior, center_theta, n_points: int = 7, mode: str = "window",
              halfwidth_log10: float = 2.0, halfwidth_Et_eV: float = 0.1) -> list:
    """
    Grid values per parameter (theta coordinates).

    mode 'window': centre +- half-width (log10 decades for k_rad, N_t, tau;
    eV for E_t), clipped to the prior box.  mode 'prior': the full prior box.
    Fixed parameters get their single value.
    """
    axes = []
    c = np.asarray(center_theta, np.float64)
    for i, name in enumerate(prior.names):
        lo, hi = prior.lo[i], prior.hi[i]
        if prior.fixed_mask[i]:
            axes.append(np.array([0.5 * (lo + hi)]))
            continue
        if mode == "window":
            hw = halfwidth_Et_eV / prior.Eg if name.startswith("Et_rel") else halfwidth_log10
            a, b = max(lo, c[i] - hw), min(hi, c[i] + hw)
        elif mode == "prior":
            a, b = lo, hi
        else:
            raise ValueError("mode must be 'window' or 'prior'")
        axes.append(np.linspace(a, b, int(n_points)))
    return axes


def grid_size(axes) -> int:
    return int(np.prod([len(a) for a in axes]))


def grid_points(prior, axes) -> tuple[np.ndarray, np.ndarray]:
    """
    All grid points inside the prior support, in the flat order that
    :func:`grid_posterior` uses: (theta (M, D), grid indices (M, D)).
    For expensive likelihoods (ODE) whose values are computed beforehand.
    """
    shape = tuple(len(a) for a in axes)
    idx = np.stack(np.unravel_index(np.arange(grid_size(axes)), shape), axis=1)
    theta = np.stack([axes[d][idx[:, d]] for d in range(len(axes))], axis=1)
    inside = prior.in_support(theta)
    return theta[inside], idx[inside]


class TabulatedLogLik:
    """
    Precomputed log L on a grid, with the interface :func:`grid_posterior`
    needs (``prior``, ``counter``, ``np_log_lik_theta``).

    Used for the ODE grid: the ODE predictions are computed once
    (ode.ode_predictions), log L for a given sigma is tabulated, and the
    marginals, intervals and plots then come from the same code as the NN grid.
    Points that are not tabulated return -inf (grid_posterior skips them).
    """

    def __init__(self, prior, axes, idx, log_lik):
        from .benchmark import PyCounter
        self.prior = prior
        self.axes = [np.asarray(a, np.float64) for a in axes]
        self.table = np.full(tuple(len(a) for a in axes), -np.inf)
        self.table[tuple(np.asarray(idx).T)] = log_lik
        self.counter = PyCounter("table")

    def np_log_lik_theta(self, theta) -> np.ndarray:
        theta = np.atleast_2d(theta)
        # nearest grid index per axis (theta comes from the same axes, so this is exact)
        idx = tuple(np.abs(theta[:, [d]] - a[None, :]).argmin(axis=1) for d, a in enumerate(self.axes))
        self.counter.record(len(theta))
        return self.table[idx]


def grid_posterior(log_post, axes, chunk: int = 2 ** 20, verbose: bool = True) -> GridResult:
    """
    Evaluate log L on the full grid and accumulate normalised marginals.

    log_post : likelihood.LogPosterior (its log_lik is used; uniform prior)
    axes : list of D arrays (see :func:`grid_axes`)
    chunk : grid points per streamed chunk
    """
    from .benchmark import count_evals
    prior = log_post.prior
    D = len(axes)
    shape = tuple(len(a) for a in axes)
    N = grid_size(axes)
    pairs = [(i, j) for i in range(D) for j in range(i + 1, D)]
    M = -np.inf                                   # running maximum of log L
    s1 = [np.zeros(n) for n in shape]             # sum of exp(logL - M) per 1-D index
    s2 = {(i, j): np.zeros(shape[i] * shape[j]) for i, j in pairs}
    tot = 0.0
    best = (-np.inf, None)
    n_eval = 0
    t0 = time.perf_counter()
    report_every = max(1, int(N / chunk / 20))
    with count_evals(log_post.counter):
        for c_idx, start in enumerate(range(0, N, chunk)):
            flat = np.arange(start, min(N, start + chunk))
            idx = np.stack(np.unravel_index(flat, shape), axis=1)          # (m, D)
            theta = np.stack([axes[d][idx[:, d]] for d in range(D)], axis=1)
            inside = prior.in_support(theta)
            if not inside.any():
                continue
            idx, theta = idx[inside], theta[inside]
            ll = log_post.np_log_lik_theta(theta)
            n_eval += len(ll)
            ok = np.isfinite(ll)
            if not ok.any():
                continue
            idx, theta, ll = idx[ok], theta[ok], ll[ok]
            m = float(ll.max())
            if m > best[0]:
                best = (m, theta[np.argmax(ll)].copy())
            if m > M:                                  # rescale the accumulators
                scale = math.exp(M - m) if np.isfinite(M) else 0.0
                tot *= scale
                for a in s1:
                    a *= scale
                for k in s2:
                    s2[k] *= scale
                M = m
            w = np.exp(ll - M)
            tot += float(w.sum())
            for d in range(D):
                s1[d] += np.bincount(idx[:, d], weights=w, minlength=shape[d])
            for (i, j) in pairs:
                s2[(i, j)] += np.bincount(idx[:, i] * shape[j] + idx[:, j], weights=w,
                                          minlength=shape[i] * shape[j])
            if verbose and (c_idx % report_every == 0):
                done = min(N, start + chunk)
                el = time.perf_counter() - t0
                print(f"  grid: {done:,}/{N:,} points ({100 * done / N:.0f} %), {el:.0f} s, "
                      f"ETA {el / done * (N - done):.0f} s", flush=True)
    dt = time.perf_counter() - t0
    marg1 = [a / tot for a in s1]
    marg2 = {k: (v / tot).reshape(shape[k[0]], shape[k[1]]) for k, v in s2.items()}
    edge = np.array([[m[0], m[-1]] if len(m) > 1 else [np.nan, np.nan] for m in marg1])
    return GridResult(axes=axes, log_z=float(M + math.log(tot)), marg1=marg1, marg2=marg2,
                      theta_max=best[1], log_lik_max=best[0], n_points=N, n_evaluated=n_eval,
                      time_s=dt, edge_mass=edge)


def grid_quantiles(values, mass, qs, lo=None, hi=None) -> np.ndarray:
    """
    Quantiles from a 1-D grid marginal (piecewise-linear cumulative mass).
    Cells extend half a spacing beyond the outermost grid values, clipped to
    [lo, hi] (the prior bounds) so that no quantile lies outside the prior.
    """
    values = np.asarray(values, np.float64)
    mass = np.asarray(mass, np.float64)
    if len(values) == 1:
        return np.full(len(qs), values[0])
    # cell boundaries halfway between grid values; mass uniform inside a cell
    edges = np.concatenate([[values[0] - 0.5 * (values[1] - values[0])],
                            0.5 * (values[1:] + values[:-1]),
                            [values[-1] + 0.5 * (values[-1] - values[-2])]])
    if lo is not None:
        edges[0] = max(edges[0], lo)
    if hi is not None:
        edges[-1] = min(edges[-1], hi)
    cdf = np.concatenate([[0.0], np.cumsum(mass)])
    cdf /= cdf[-1]
    return np.interp(qs, cdf, edges)


def grid_intervals(res: GridResult, prior, levels=(0.68, 0.95)) -> pd.DataFrame:
    """Median and equal-tailed intervals per parameter (display coordinates) from the grid."""
    rows = []
    bnd = prior.display_bounds()
    for i, k in enumerate(prior.display_keys):
        scale = prior.Eg if prior.names[i].startswith("Et_rel") else 1.0
        vals = res.axes[i] * scale
        lo_b, hi_b = bnd[i]
        row = {"parameter": k, "median": float(grid_quantiles(vals, res.marg1[i], [0.5], lo_b, hi_b)[0]),
               "edge_mass_low": float(res.edge_mass[i, 0]), "edge_mass_high": float(res.edge_mass[i, 1]),
               "grid_lo": float(vals[0]), "grid_hi": float(vals[-1]), "grid_points": len(vals)}
        for lev in levels:
            a = (1 - lev) / 2
            ql, qh = grid_quantiles(vals, res.marg1[i], [a, 1 - a], lo_b, hi_b)
            tag = f"{int(round(lev * 100))}"
            row[f"lo{tag}"], row[f"hi{tag}"], row[f"width{tag}"] = float(ql), float(qh), float(qh - ql)
        rows.append(row)
    return pd.DataFrame(rows).set_index("parameter")


def plot_grid_vs_samples(res: GridResult, prior, samples: dict, truth=None, bins: int = 40):
    """
    1-D marginals: grid masses (converted to densities, markers + line) vs
    sample histograms restricted to the grid range (each renormalised to that
    range, so that all curves describe the same conditional distribution).
    """
    import matplotlib.pyplot as plt
    free = list(prior.free_idx)
    D = len(free)
    ncol = min(D, 5)
    nrow = int(math.ceil(D / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.3 * ncol, 2.7 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for a, i in enumerate(free):
        ax = axes[a]
        scale = prior.Eg if prior.names[i].startswith("Et_rel") else 1.0
        vals = res.axes[i] * scale
        lo, hi = vals[0], vals[-1]
        width = (vals[1] - vals[0]) if len(vals) > 1 else 1.0
        for j, (name, s) in enumerate(samples.items()):
            x = s[:, i]
            x = x[(x >= lo - 0.5 * width) & (x <= hi + 0.5 * width)]
            if len(x):
                ax.hist(x, bins=bins, range=(lo - 0.5 * width, hi + 0.5 * width), density=True,
                        histtype="step", lw=1.4, color=f"C{j}", label=f"{name} (in window)")
        ax.plot(vals, res.marg1[i] / width, "o-", color="k", ms=4, lw=1.2, label="grid")
        if truth is not None:
            ax.axvline(truth[i], color="k", ls="--", lw=1)
        ax.set_xlabel(prior.labels[i])
        ax.set_yticks([])
    for ax in axes[D:]:
        ax.axis("off")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    return fig


def plot_grid_corner(res: GridResult, prior, samples_disp=None, truth=None):
    """
    2-D grid marginals (filled colour) with the 68/95 % mass contours of the
    grid; optionally the sample points (e.g. NUTS) on top, thinned.
    """
    import matplotlib.pyplot as plt
    free = list(prior.free_idx)
    nd = len(free)
    fig, axes = plt.subplots(nd, nd, figsize=(2.3 * nd, 2.3 * nd))
    axes = np.atleast_2d(axes)
    sc = [prior.Eg if n.startswith("Et_rel") else 1.0 for n in prior.names]
    if samples_disp is not None and len(samples_disp) > 3000:
        samples_disp = samples_disp[np.random.default_rng(0).choice(len(samples_disp), 3000, replace=False)]
    for a in range(nd):
        for b in range(nd):
            ax = axes[a, b]
            i, j = free[b], free[a]
            if b > a:
                ax.axis("off")
                continue
            xi = res.axes[i] * sc[i]
            if a == b:
                ax.plot(xi, res.marg1[i], "o-", color="k", ms=3)
                ax.set_yticks([])
            else:
                key = (min(i, j), max(i, j))
                m2 = res.marg2[key] if i < j else res.marg2[key].T       # rows: i (x), cols: j (y)
                yj = res.axes[j] * sc[j]
                ax.pcolormesh(xi, yj, m2.T, shading="nearest", cmap="Greys")
                # mass levels enclosing 68 / 95 % (highest-mass cells first)
                flat = np.sort(m2.ravel())[::-1]
                cum = np.cumsum(flat)
                lev = [flat[min(np.searchsorted(cum, q), len(flat) - 1)] for q in (0.95, 0.68)]
                if len(xi) > 1 and len(yj) > 1 and lev[0] < lev[1]:
                    ax.contour(xi, yj, m2.T, levels=lev, colors=["C1", "C3"], linewidths=1.0)
                if samples_disp is not None:
                    ax.plot(samples_disp[:, i], samples_disp[:, j], ",", color="C0", alpha=0.3)
                if truth is not None:
                    ax.plot(truth[i], truth[j], "x", color="C2", ms=7, mew=2)
                ax.set_ylim(yj[0], yj[-1])
            ax.set_xlim(xi[0], xi[-1])
            if a == nd - 1:
                ax.set_xlabel(prior.labels[i], fontsize=9)
            else:
                ax.set_xticklabels([])
            if b == 0 and a != b:
                ax.set_ylabel(prior.labels[j], fontsize=9)
            elif a != b:
                ax.set_yticklabels([])
            ax.tick_params(labelsize=7)
    fig.suptitle("full-dimensional grid: 2-D marginal masses (68 % red, 95 % orange)"
                 + ("; dots: NUTS samples" if samples_disp is not None else ""))
    fig.tight_layout()
    return fig
