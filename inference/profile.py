"""
profile.py — profile likelihood in one parameter (task T3, R1 comment 5)
========================================================================

For a grid of fixed values of one parameter theta_j (e.g. log10 N_t of the
dominant shallow trap) all OTHER parameters are re-optimised inside the prior
support:

    PL(v) = max_{theta: theta_j = v} log L(theta)

and  -2 [PL(v) - max_v PL(v)]  is plotted with the thresholds 1 (68 %) and
3.84 (95 %) of the chi^2_1 distribution.  The difference between the profile
and the MCMC marginal of theta_j shows volume effects.

Optimisation strategy (fully batched)
-------------------------------------
* pass 0: every grid value starts from (a) the best fit with theta_j replaced
  by the grid value and (b) ``n_random`` prior draws — all rows of all grid
  points form ONE batched L-BFGS run;
* sweeps 1..n_sweeps: every grid value is restarted from the current optima
  of its two neighbours (warm start), again in one batch; the best result per
  grid value is kept.
The fixed parameter is held by ``Prior.forward(u, fixed=(j, value))``; for an
ordered trap energy the other energies stay ordered around the fixed value.

Note: with the MSE likelihood (one effective data point per curve) the chi^2
thresholds refer to that likelihood; they are therefore conservative in the
same sense as the credible intervals.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .optimize import lbfgs_polish


def _make_objective(log_post, j: int):
    """Batched value-and-gradient of -log L with theta_j fixed per row."""
    import tensorflow as tf
    import tensorflow_probability as tfp

    fixed_vals = tf.Variable(np.zeros(1), dtype=tf.float64, shape=tf.TensorShape([None]),
                             trainable=False)

    def nll(u):
        theta, _ = log_post.prior.forward(u, fixed=(j, fixed_vals))
        return -log_post.log_lik(theta)

    @tf.function(jit_compile=True, reduce_retracing=True)
    def vg(u):
        return tfp.math.value_and_gradient(nll, u)

    return vg, fixed_vals


def profile_likelihood(log_post, param: str | int, grid, theta_best, n_random: int = 8,
                       n_sweeps: int = 2, max_iter: int = 300, tolerance: float = 1e-8,
                       seed: int = 0, verbose: bool = True) -> pd.DataFrame:
    """
    Profile likelihood of one parameter.

    Parameters
    ----------
    log_post : likelihood.LogPosterior (its current sigmas are used)
    param : theta name (e.g. 'log10_Nt_2') or index
    grid : values of theta_j (in theta coordinates, i.e. E_t/Eg for energies)
    theta_best : best-fit theta (warm start)
    n_random : random prior starts per grid value in pass 0

    Returns
    -------
    DataFrame with columns value, neg_log_lik, delta_2nll (= -2 Delta log L),
    n_evals and the optimised theta components.  ``attrs`` holds the
    total number of evaluations.
    """
    from .benchmark import count_evals

    prior = log_post.prior
    j = prior.names.index(param) if isinstance(param, str) else int(param)
    grid = np.asarray(grid, np.float64)
    G = len(grid)
    rng = np.random.default_rng(seed)
    vg, fixed_vals = _make_objective(log_post, j)

    def solve(theta_starts, values):
        """Batched L-BFGS from theta starts (rows) with theta_j = values (rows)."""
        th = prior.project(theta_starts)
        th[:, j] = values
        U = prior.inverse(th)
        fixed_vals.assign(values)
        res = lbfgs_polish(log_post, U, max_iter=max_iter, tolerance=tolerance, value_and_grad=vg)
        f = np.array([r.neg_log_lik for r in res])
        # re-map u -> theta with the SAME fixed value (lbfgs_polish maps without it)
        import tensorflow as tf
        theta = prior.forward(tf.constant(np.stack([r.u for r in res])),
                              fixed=(j, tf.constant(values)))[0].numpy()
        return f, theta

    best_f = np.full(G, np.inf)
    best_theta = np.zeros((G, prior.dim))
    with count_evals(log_post.counter) as cnt:
        # pass 0: best fit + random starts
        starts = [np.repeat(np.asarray(theta_best)[None], G, axis=0)]
        for _ in range(n_random):
            starts.append(prior.sample(G, rng=rng))
        starts = np.concatenate(starts)
        values = np.tile(grid, 1 + n_random)
        f, th = solve(starts, values)
        for g in range(G):
            rows = np.arange(g, len(values), G)
            k = rows[np.argmin(f[rows])]
            best_f[g], best_theta[g] = f[k], th[k]
        if verbose:
            print(f"  profile pass 0: {len(values)} optimisations")
        # warm-start sweeps from neighbours
        for sweep in range(n_sweeps):
            idx_from = np.concatenate([np.clip(np.arange(G) - 1, 0, G - 1),
                                       np.clip(np.arange(G) + 1, 0, G - 1)])
            f, th = solve(best_theta[idx_from], np.tile(grid, 2))
            improved = 0
            for g in range(G):
                for k in (g, g + G):
                    if f[k] < best_f[g] - 1e-9:
                        best_f[g], best_theta[g] = f[k], th[k]
                        improved += 1
            if verbose:
                print(f"  profile sweep {sweep + 1}: {improved} grid values improved")
            if improved == 0:
                break

    out = pd.DataFrame({"value": grid, "neg_log_lik": best_f})
    out["delta_2nll"] = 2.0 * (out["neg_log_lik"] - out["neg_log_lik"].min())
    for i, n in enumerate(prior.names):
        out[n] = best_theta[:, i]
    out.attrs["param"] = prior.names[j]
    out.attrs["n_evals"] = cnt.counts["n_points"]
    return out


def profile_interval(df: pd.DataFrame, threshold: float = 3.84) -> tuple:
    """
    Range of grid values with -2 Delta log L <= threshold (1 -> 68 %, 3.84 -> 95 %).
    Returns (lo, hi, lo_at_grid_edge, hi_at_grid_edge).
    """
    ok = df["delta_2nll"].to_numpy() <= threshold
    if not ok.any():
        return (np.nan, np.nan, False, False)
    v = df["value"].to_numpy()
    idx = np.nonzero(ok)[0]
    return (float(v[idx[0]]), float(v[idx[-1]]), bool(idx[0] == 0), bool(idx[-1] == len(v) - 1))
