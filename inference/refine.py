"""
refine.py — local refinement of a fit with the FULL rate equations (ODE)
========================================================================

CMA-ES on the NN likelihood can stall in flat valleys or stop on a slightly
wrong optimum because of surrogate error.  After CMA-ES, every candidate is
refined with ``scipy.optimize.curve_fit`` using the rate-equation ODE (the
solver and post-processing that generated the training data, see ode.py):

    model prediction  = [log10 tau_ODE(QFLS_i) for the tr-PL points,
                         ss-PL model at the measured x (exact physics, numpy)]
    sigma per point   = sigma_d * sqrt(N_d)      (d = tr-PL, ss-PL; the per-point equivalent of the
                                                 curve-level sigma_d of -MSE_d/(2 sigma_d^2), not a rescaling)

so that chi^2 = sum_d MSE_d / sigma_d^2 = -2 log L + const, i.e. curve_fit
maximises the same likelihood as the rest of the workflow — only with the ODE
instead of the surrogate.

* The fit runs in the unconstrained coordinates u of ``prior.Prior`` (no
  bounds; the E_t ordering is built in).  Fixed parameters are not fitted.
* The Jacobian is a forward finite difference in u; its D ODE solves can run
  in parallel (``n_jobs_jac``).  ``refine_many`` instead runs several starts
  in parallel worker processes (serial Jacobian in each).
* Every residual evaluation is recorded, so the trajectory (best -log L and
  theta vs number of ODE solves) can be plotted after the CMA-ES trajectory.
* Points the ODE curve does not reach (outside its simulated QFLS range) get
  a fixed residual of ``penalty`` (in units of sigma*sqrt(N)).

All code here is numpy/scipy/numba only (no TensorFlow), so it can run in
CPU worker processes next to a GPU session.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from .ode import ode_log10_tau


@dataclass
class OdeFitProblem:
    """Everything curve_fit needs; picklable (numpy only)."""
    prior: object                      # prior.Prior (numpy parts only are used)
    npulse: float
    constants: dict                    # training constants (Nc, Nv, tmin, tmax, PL_DECAY_MAGNITUDE)
    T: float
    qfls: np.ndarray                   # tr-PL QFLS points used in the likelihood
    y_trpl: np.ndarray                 # observed log10 tau_diff at those points
    sigma_trpl: float
    sspl: object | None = None         # sspl_tf.SsplSpecNumpy or None
    y_sspl: np.ndarray | None = None
    sigma_sspl: float | None = None
    penalty: float = 5.0
    # sigma_delta of the marginalised ss-PL offset (likelihood.SsplTerm.offset_sigma; 24/10/2026).  In least-squares form:
    # ss-PL prediction shifted by delta_hat = sum r / (N + s^2/sigma_delta^2) plus one pseudo-observation delta_hat / sigma_delta;
    # since delta_hat is the optimum this equals the analytic marginal (MSE - w m^2)/(2 sigma^2) of the NN likelihood exactly
    offset_sigma: float | None = None

    @property
    def y_obs(self) -> np.ndarray:
        if self.sspl is None:
            return self.y_trpl
        y = np.concatenate([self.y_trpl, self.y_sspl])
        return np.append(y, 0.0) if self.offset_sigma else y

    @property
    def sigma_points(self) -> np.ndarray:
        s = np.full(len(self.y_trpl), self.sigma_trpl * math.sqrt(len(self.y_trpl)))
        if self.sspl is not None:
            s = np.concatenate([s, np.full(len(self.y_sspl), self.sigma_sspl * math.sqrt(len(self.y_sspl)))])
            if self.offset_sigma:
                s = np.append(s, self.offset_sigma)
        return s

    def sspl_offset(self, y_s) -> float:
        """Most probable ss-PL offset delta_hat (model - data) for the ss-PL prediction y_s (0 without offset)."""
        if not self.offset_sigma:
            return 0.0
        n = len(self.y_sspl)
        s2 = self.sigma_sspl ** 2 * n
        return float(np.sum(np.asarray(y_s) - self.y_sspl) / (n + s2 / self.offset_sigma ** 2))

    def predict(self, theta) -> tuple[np.ndarray, float]:
        """Model prediction (uncovered tr-PL points replaced by penalty values) and covered fraction."""
        y_t = ode_log10_tau(theta, self.prior, self.npulse, self.constants, self.qfls, self.T)
        ok = np.isfinite(y_t)
        s = self.sigma_trpl * math.sqrt(len(self.y_trpl))
        y_t = np.where(ok, y_t, self.y_trpl + self.penalty * s)
        if self.sspl is None:
            return y_t, float(ok.mean())
        y_s = self.sspl.predict(np.atleast_2d(theta))[0]
        if self.offset_sigma:
            d = self.sspl_offset(y_s)
            return np.concatenate([y_t, y_s - d, [d]]), float(ok.mean())
        return np.concatenate([y_t, y_s]), float(ok.mean())

    def neg_log_lik(self, theta) -> float:
        """-log L with the ODE (same constants as likelihood.LogPosterior)."""
        pred, _ = self.predict(theta)
        return self._nll_from_pred(pred)

    def neg_log_lik_many(self, y_trpl_pred, sigma_trpl: float | None = None) -> np.ndarray:
        """
        -log L (tr-PL only) for many stored ODE predictions at once, optionally
        with another sigma.  Same value as :meth:`neg_log_lik` for the same
        sigma: uncovered points (NaN) get the residual ``penalty``.

        y_trpl_pred : (N, n_points) log10 tau_diff, e.g. from ode.ode_predictions
        """
        if self.sspl is not None:
            raise ValueError("neg_log_lik_many covers the tr-PL-only likelihood")
        s = float(self.sigma_trpl if sigma_trpl is None else sigma_trpl)
        y = np.atleast_2d(np.asarray(y_trpl_pred, np.float64))
        r = (y - self.y_trpl) / (s * math.sqrt(len(self.y_trpl)))
        r = np.where(np.isfinite(r), r, self.penalty)
        return 0.5 * np.sum(r * r, axis=1) + math.log(s) + 0.5 * math.log(2 * math.pi)

    def _nll_from_pred(self, pred) -> float:
        r = (pred - self.y_obs) / self.sigma_points
        const = math.log(self.sigma_trpl) + 0.5 * math.log(2 * math.pi)
        if self.sspl is not None:
            const += math.log(self.sigma_sspl) + 0.5 * math.log(2 * math.pi)
        return float(0.5 * np.sum(r * r) + const)


@dataclass
class OdeFitResult:
    """Result of one ODE refinement."""
    theta: np.ndarray
    u: np.ndarray
    neg_log_lik: float
    start_theta: np.ndarray
    start_neg_log_lik: float
    n_ode_solves: int
    nfev: int
    success: bool
    message: str
    cov_theta: np.ndarray | None = None          # Laplace covariance from curve_fit (theta coords)
    history_evals: np.ndarray | None = None      # cumulative ODE solves at each residual evaluation
    history_best: np.ndarray | None = None       # best -log L so far
    history_theta: np.ndarray | None = None      # theta of the best point so far
    time_s: float = 0.0
    label: str = ""
    method: str = "curve_fit(ODE)"


def _fd_jacobian(f, p, f0, h, n_jobs):
    """Forward-difference Jacobian of f at p (optionally parallel over columns)."""
    cols = [p + h * np.eye(len(p))[k] for k in range(len(p))]
    if n_jobs and n_jobs > 1:
        from joblib import Parallel, delayed
        vals = Parallel(n_jobs=n_jobs)(delayed(f)(c) for c in cols)
    else:
        vals = [f(c) for c in cols]
    return np.stack([(v - f0) / h for v in vals], axis=1)


def ode_curve_fit(problem: OdeFitProblem, theta0, max_nfev: int = 100, fd_step: float = 1e-3,
                  n_jobs_jac: int = 1, label: str = "") -> OdeFitResult:
    """
    Refine one start with scipy.optimize.curve_fit (method 'trf') on the ODE.

    Parameters
    ----------
    problem : OdeFitProblem
    theta0 : start (theta coordinates), typically a CMA-ES result
    max_nfev : maximum number of residual evaluations of curve_fit
    fd_step : finite-difference step in u
    n_jobs_jac : parallel ODE solves for the Jacobian columns
    """
    import warnings
    from scipy.optimize import curve_fit

    prior = problem.prior
    free = np.asarray(prior.free_idx)
    theta0 = np.asarray(theta0, np.float64)
    u0 = prior.inverse(theta0[None])[0]
    y_obs, sig = problem.y_obs, problem.sigma_points
    rec = {"evals": [], "nll": [], "theta": []}
    counter = {"solves": 0}

    def theta_of(p):
        u = u0.copy()
        u[free] = p
        return prior.forward_np(u[None])[0]

    def pred_of(p):
        try:
            return problem.predict(theta_of(p))[0]
        except Exception:
            return y_obs + problem.penalty * sig

    def model(_x, *p):
        p = np.asarray(p, np.float64)
        pred = pred_of(p)
        counter["solves"] += 1
        rec["evals"].append(counter["solves"])
        rec["nll"].append(problem._nll_from_pred(pred))
        rec["theta"].append(theta_of(p))
        return pred

    def jac(_x, *p):
        p = np.asarray(p, np.float64)
        f0 = pred_of(p)
        J = _fd_jacobian(pred_of, p, f0, fd_step, n_jobs_jac)
        counter["solves"] += len(p) + 1
        return J

    start_nll = problem.neg_log_lik(theta0)
    t0 = time.perf_counter()
    ok, msg, pcov, popt = True, "", None, u0[free]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            popt, pcov, info, msg, ier = curve_fit(
                model, np.zeros(1), y_obs, p0=u0[free], sigma=sig, absolute_sigma=True,
                method="trf", jac=jac, max_nfev=max_nfev, x_scale="jac", full_output=True)
            ok = ier in (1, 2, 3, 4)
        except RuntimeError as e:          # max_nfev reached etc.
            ok, msg = False, str(e)
    dt = time.perf_counter() - t0
    # best point seen (curve_fit returns its last accepted point; normally the same)
    nll = np.asarray(rec["nll"])
    if len(nll):
        i_best = int(np.argmin(nll))
        theta_best = rec["theta"][i_best] if nll[i_best] < problem.neg_log_lik(theta_of(popt)) else theta_of(popt)
    else:
        theta_best = theta_of(popt)
    best_nll = problem.neg_log_lik(theta_best)
    cov_theta = None
    if pcov is not None and np.all(np.isfinite(pcov)):
        # covariance in u (free dims) -> theta via the Jacobian of the forward map
        J = np.zeros((prior.dim, len(free)))
        for k, i in enumerate(free):
            e = np.zeros(prior.dim)
            e[i] = 1e-6
            J[:, k] = (prior.forward_np((prior.inverse(theta_best[None])[0] + e)[None])[0] - theta_best) / 1e-6
        cov_theta = J @ pcov @ J.T
    run_best = np.minimum.accumulate(nll) if len(nll) else np.array([best_nll])
    best_idx = [int(np.argmin(nll[:k + 1])) for k in range(len(nll))]
    return OdeFitResult(
        theta=theta_best, u=prior.inverse(theta_best[None])[0], neg_log_lik=best_nll,
        start_theta=theta0, start_neg_log_lik=start_nll, n_ode_solves=counter["solves"],
        nfev=len(nll), success=bool(ok), message=str(msg), cov_theta=cov_theta,
        history_evals=np.asarray(rec["evals"]), history_best=run_best,
        history_theta=np.asarray([rec["theta"][i] for i in best_idx]) if len(nll) else theta_best[None],
        time_s=dt, label=label)


def _refine_one(problem, theta0, max_nfev, fd_step, label):
    import warnings
    warnings.filterwarnings("ignore")
    return ode_curve_fit(problem, theta0, max_nfev=max_nfev, fd_step=fd_step, n_jobs_jac=1, label=label)


def refine_many(problem: OdeFitProblem, theta_starts, labels=None, max_nfev: int = 100,
                fd_step: float = 1e-3, n_jobs: int = 8) -> list:
    """Refine several starts in parallel worker processes (one curve_fit each)."""
    from joblib import Parallel, delayed
    theta_starts = np.atleast_2d(theta_starts)
    labels = list(labels) if labels is not None else [f"start_{i}" for i in range(len(theta_starts))]
    if len(theta_starts) == 1 or n_jobs <= 1:
        return [ode_curve_fit(problem, th, max_nfev=max_nfev, fd_step=fd_step,
                              n_jobs_jac=n_jobs, label=lb) for th, lb in zip(theta_starts, labels)]
    return Parallel(n_jobs=min(n_jobs, len(theta_starts)))(
        delayed(_refine_one)(problem, th, max_nfev, fd_step, lb) for th, lb in zip(theta_starts, labels))


def make_problem(log_post, art, npulse: float, T: float = 300.0, penalty: float = 5.0) -> OdeFitProblem:
    """OdeFitProblem with the data and CURRENT sigmas of a LogPosterior (mode trpl or joint)."""
    if not log_post.trpl_terms:
        raise ValueError("The ODE refinement needs a tr-PL term")
    term = log_post.trpl_terms[0]
    sig = log_post.sigmas()
    sspl, y_s, s_s = None, None, None
    if log_post.sspl_term is not None:
        sspl = log_post.sspl_term.observation.numpy_spec()
        y_s = np.asarray(log_post.sspl_term.observation.y_obs)
        s_s = sig["sspl"]
    return OdeFitProblem(prior=log_post.prior, npulse=float(npulse), constants=dict(art.constants),
                         T=float(T), qfls=np.asarray(term.surrogate.qfls_used),
                         y_trpl=np.asarray(term.y_obs), sigma_trpl=sig["trpl"][0],
                         sspl=sspl, y_sspl=y_s, sigma_sspl=s_s, penalty=penalty,
                         offset_sigma=log_post.sspl_term.offset_sigma if log_post.sspl_term is not None else None)
