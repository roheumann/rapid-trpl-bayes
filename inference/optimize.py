"""
optimize.py — best fit on the SAME log-likelihood the samplers use
==================================================================

Two optimisers, both working in the unconstrained coordinates u of
``prior.Prior`` (no walls, ordering built in) and both minimising
-log L(theta(u)):

* :func:`cmaes_multistart` — CMA-ES (pycma), gradient-free, the method of the
  original submission.  Several independent CMA-ES instances (one per start
  point) run in LOCKSTEP: in every generation the populations of all active
  instances are stacked and evaluated in ONE batched NN call.  Every instance
  keeps its own evaluation count, so "evaluations to convergence" can be
  compared between physics-informed and random starts (task T2 / R1 comment 7).
* :func:`lbfgs_polish` — batched L-BFGS (TFP) with autodiff gradients; used to
  polish CMA-ES results and inside the profile likelihood.

:func:`fit` combines them: CMA-ES from the given starts, then L-BFGS from the
best CMA-ES solutions; the overall best is returned.

Note on CMA-ES step size: near the centre of the box a step of 1 in u
corresponds to 1/4 of the parameter range, so ``sigma0 = 1.0`` in u is
comparable to the legacy ``sigma = 0.25`` in the normalised unit cube.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class OptimResult:
    """Result of one optimiser run (one start point)."""
    u: np.ndarray
    theta: np.ndarray
    neg_log_lik: float
    n_evals: int
    n_iter: int
    stop_reason: str
    start_theta: np.ndarray | None = None
    history_evals: np.ndarray | None = None     # cumulative evaluations per generation
    history_best: np.ndarray | None = None      # best -log L so far per generation
    history_theta: np.ndarray | None = None     # theta of the best point so far, per generation
    method: str = "cmaes"
    label: str = ""

    def evals_to_reach(self, target: float) -> float:
        """Evaluations until the best value first dropped below ``target`` (NaN if never)."""
        if self.history_best is None:
            return np.nan
        hit = np.nonzero(self.history_best <= target)[0]
        return float(self.history_evals[hit[0]]) if len(hit) else np.nan


@dataclass
class FitResult:
    """Combined result of :func:`fit`."""
    best: OptimResult
    cmaes: list = field(default_factory=list)
    lbfgs: list = field(default_factory=list)
    n_evals_cmaes: int = 0
    n_evals_lbfgs: int = 0
    n_grad_evals_lbfgs: int = 0

    def table(self, prior=None) -> pd.DataFrame:
        """One row per CMA-ES start (+ polish), for export / plots."""
        rows = []
        for r in self.cmaes:
            row = {"label": r.label, "method": r.method, "neg_log_lik": r.neg_log_lik,
                   "n_evals": r.n_evals, "n_iter": r.n_iter, "stop": r.stop_reason}
            names = prior.names if prior is not None else [f"theta_{i}" for i in range(len(r.theta))]
            row.update({f"start_{n}": v for n, v in zip(names, r.start_theta)})
            row.update({n: v for n, v in zip(names, r.theta)})
            rows.append(row)
        return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# CMA-ES (lockstep multi-start)
# ─────────────────────────────────────────────────────────────────────────────

def cmaes_multistart(log_post, theta_starts, sigma0: float = 1.0, max_evals: int = 20000,
                     tolfun: float = 1e-7, tolx: float = 1e-6, popsize: int | None = None,
                     seed: int = 1, labels=None, verbose: bool = True) -> list:
    """
    Run one CMA-ES instance per start point, all evaluated together.

    Parameters
    ----------
    log_post : likelihood.LogPosterior (or multi.MultiObjective: every start
        then belongs to its own data set)
    theta_starts : (S, D) start points in theta coordinates (inside the support)
    sigma0 : initial step size in u coordinates
    max_evals : evaluation budget per instance
    tolfun, tolx : pycma termination tolerances (-log L units / u units)
    popsize : population size (default pycma: 4 + 3 ln D)
    seed : base seed (instance s uses seed + s)
    labels : optional names of the starts (e.g. 'physics', 'sobol_3')

    Returns
    -------
    list of OptimResult (same order as the starts)
    """
    import cma

    theta_starts = np.atleast_2d(np.asarray(theta_starts, np.float64))
    U0 = log_post.prior.inverse(theta_starts)
    S = len(U0)
    labels = list(labels) if labels is not None else [f"start_{s}" for s in range(S)]
    opts = {"verbose": -9, "maxfevals": int(max_evals), "tolfun": tolfun, "tolx": tolx,
            "tolflatfitness": 10, "CMA_stds": None}
    es_list = []
    for s in range(S):
        o = dict(opts, seed=int(seed) + s)
        if popsize is not None:
            o["popsize"] = int(popsize)
        es_list.append(cma.CMAEvolutionStrategy(U0[s], sigma0, o))
    hist_e = [[] for _ in range(S)]
    hist_f = [[] for _ in range(S)]
    hist_u = [[] for _ in range(S)]
    best_f = np.full(S, np.inf)
    best_u = [U0[s].copy() for s in range(S)]
    gen = 0
    while True:
        active = [s for s in range(S) if not es_list[s].stop()]
        if not active:
            break
        sols = {s: es_list[s].ask() for s in active}
        X = np.concatenate([np.asarray(sols[s]) for s in active])
        if hasattr(log_post, "np_neg_log_lik_u_rows"):
            # multi-data-set objective (multi.MultiObjective): start s -> its data set
            ids = np.concatenate([np.full(len(sols[s]), s) for s in active])
            F = log_post.np_neg_log_lik_u_rows(X, ids)
        else:
            F = log_post.np_neg_log_lik_u(X)
        pos = 0
        for s in active:
            k = len(sols[s])
            f_s = F[pos:pos + k]
            pos += k
            es_list[s].tell(sols[s], f_s.tolist())
            k_best = int(np.argmin(f_s))
            if f_s[k_best] < best_f[s]:
                best_f[s], best_u[s] = float(f_s[k_best]), np.asarray(sols[s][k_best]).copy()
            hist_e[s].append(es_list[s].countevals)
            hist_f[s].append(best_f[s])
            hist_u[s].append(best_u[s])
        gen += 1
        if verbose and gen % 50 == 0:
            print(f"  CMA-ES generation {gen:5d}: {len(active):3d} active, "
                  f"best -logL = {np.min(best_f):.6g}")

    out = []
    for s, es in enumerate(es_list):
        res = es.result
        u = np.asarray(res.xbest)
        theta = log_post.prior.forward(u[None])[0].numpy()[0]
        out.append(OptimResult(
            u=u, theta=theta, neg_log_lik=float(res.fbest), n_evals=int(res.evaluations),
            n_iter=int(res.iterations), stop_reason=",".join(es.stop().keys()),
            start_theta=theta_starts[s], history_evals=np.asarray(hist_e[s]),
            history_best=np.asarray(hist_f[s]),
            history_theta=log_post.prior.forward_np(np.asarray(hist_u[s])) if hist_u[s] else None,
            method="cmaes", label=labels[s]))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# L-BFGS (batched, autodiff)
# ─────────────────────────────────────────────────────────────────────────────

_LBFGS_CACHE = {}

def lbfgs_polish(log_post, U0, max_iter: int = 500, tolerance: float = 1e-8,
                 value_and_grad=None, labels=None) -> list:
    """
    Batched L-BFGS from the rows of U0 (unconstrained coordinates).

    ``value_and_grad`` may be supplied (e.g. a profile objective); by default
    -log L(theta(u)) of ``log_post`` with its gradient is used.  All rows are
    optimised simultaneously; TFP freezes rows that have converged.

    Returns list of OptimResult.  ``n_evals`` is the number of batched
    objective+gradient evaluations (identical for all rows of the batch).
    """
    import tensorflow as tf
    import tensorflow_probability as tfp

    U0 = np.atleast_2d(np.asarray(U0, np.float64))
    vg = value_and_grad or log_post.tf_neg_log_lik_and_grad_u

    # one traced optimiser per (objective, settings): avoids retracing on every call
    key = (id(vg), int(max_iter), float(tolerance))
    if key not in _LBFGS_CACHE:
        @tf.function(jit_compile=False, reduce_retracing=True)
        def _run(u0):
            return tfp.optimizer.lbfgs_minimize(
                vg, initial_position=u0, max_iterations=max_iter, tolerance=tolerance,
                num_correction_pairs=10)
        _LBFGS_CACHE[key] = (_run, vg)       # keep vg alive so that id(vg) stays unique
    r = _LBFGS_CACHE[key][0](tf.constant(U0))
    pos = r.position.numpy()
    f = r.objective_value.numpy()
    theta = log_post.prior.forward(pos)[0].numpy()
    conv = r.converged.numpy()
    failed = r.failed.numpy()
    n_eval = int(r.num_objective_evaluations.numpy())
    n_it = int(r.num_iterations.numpy())
    theta0 = log_post.prior.forward(U0)[0].numpy()
    labels = labels or [f"lbfgs_{i}" for i in range(len(U0))]
    out = []
    for i in range(len(U0)):
        stop = "converged" if conv[i] else ("failed" if failed[i] else "max_iter")
        out.append(OptimResult(u=pos[i], theta=theta[i], neg_log_lik=float(f[i]), n_evals=n_eval,
                               n_iter=n_it, stop_reason=stop, start_theta=theta0[i],
                               method="lbfgs", label=labels[i]))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Combined fit
# ─────────────────────────────────────────────────────────────────────────────

def fit(log_post, theta_starts, labels=None, n_polish: int = 3, cmaes_kwargs=None,
        lbfgs_kwargs=None, verbose: bool = True) -> FitResult:
    """
    CMA-ES from every start, then L-BFGS from the ``n_polish`` best CMA-ES
    solutions.  Returns the overall best and all individual runs.

    Evaluation counts are read from the log-posterior's EvalCounter, i.e. they
    are the number of parameter sets for which the likelihood was evaluated
    (L-BFGS evaluations additionally include a gradient).
    """
    from .benchmark import count_evals

    cmaes_kwargs = dict(cmaes_kwargs or {})
    lbfgs_kwargs = dict(lbfgs_kwargs or {})
    cmaes_kwargs.setdefault("verbose", verbose)
    with count_evals(log_post.counter) as c1:
        cm = cmaes_multistart(log_post, theta_starts, labels=labels, **cmaes_kwargs)
    order = np.argsort([r.neg_log_lik for r in cm])
    pol, n_pol = [], 0
    n_lb_rows = 0
    if n_polish > 0:
        U = np.stack([cm[i].u for i in order[:n_polish]])
        with count_evals(log_post.counter) as c2:
            pol = lbfgs_polish(log_post, U, labels=[f"lbfgs<-{cm[i].label}" for i in order[:n_polish]],
                               **lbfgs_kwargs)
        n_pol = c2.counts["n_points"]
        n_lb_rows = n_pol
    candidates = cm + pol
    best = min(candidates, key=lambda r: r.neg_log_lik)
    if verbose:
        print(f"Best -log L = {best.neg_log_lik:.6g} ({best.method}, {best.label}); "
              f"CMA-ES evaluations: {c1.counts['n_points']}, L-BFGS evaluations: {n_pol}")
    return FitResult(best=best, cmaes=cm, lbfgs=pol, n_evals_cmaes=c1.counts["n_points"],
                     n_evals_lbfgs=n_pol, n_grad_evals_lbfgs=n_lb_rows)
