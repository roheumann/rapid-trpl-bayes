"""
ode_likelihood.py — the likelihood of the workflow with the rate equations instead of the NN surrogate
======================================================================================================
15/10/2026 (revision).  Reference for the surrogate speed-up: the SAME likelihood as likelihood.LogPosterior,

    log L = Σ_d [ −MSE_d / (2σ_d²) − log(σ_d √2π) ],

with the tr-PL model curve log10 τ_diff(θ) from the legacy ODE solver (``network-training/trpl_module.py``,
``solve_transient``, unchanged; post-processing as for the training data, see ode.py) instead of the NN.
The ss-PL term has no surrogate (sspl_tf is the steady-state physics itself) and is taken over unchanged.

* tr-PL points outside the simulated QFLS range of an ODE curve cannot be predicted: such θ get log L = −∞
  (the decay never reaches the measured QFLS, so θ cannot produce the measurement).  ``n_uncovered`` counts them.
* The ODE solves run in parallel worker processes (joblib/loky, ``n_jobs``); one call = one batch of θ.
* ``counter`` (benchmark.PyCounter) counts batched calls and ODE solves — the evaluation count of the timing
  table.  The ss-PL term counts its own TF evaluations in ``sspl_log_post.counter``.

``OdeLogPosterior`` has the interface the samplers use (``prior``, ``counter``, ``np_log_prob_u``,
``np_log_lik_theta``), so ``samplers.run_emcee`` works unchanged.  ``OdePopulationEvaluator`` is the drop-in for
``smc.PopulationEvaluator`` (``smc.run_smc(..., evaluator=...)``).  NUTS is not offered: it needs gradients, which
the legacy solver does not provide.
"""
from __future__ import annotations

import math
import time

import numpy as np

from .benchmark import PyCounter
from .smc import LOG4, PopulationEvaluator

LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)


def _ode_row(theta, prior, npulse, constants, qfls, T):
    from .ode import ode_log10_tau
    return ode_log10_tau(theta, prior, npulse, constants, qfls, T)


class OdeLogPosterior:
    """
    log L / log p with the tr-PL term from the rate equations.

    Parameters
    ----------
    prior : prior.Prior (the prior of the NN run)
    mode : 'trpl' | 'sspl' | 'joint'
    constants, npulse, T : physics of the ODE (``art.constants`` of the network, as for the training data)
    qfls, y_obs, sigma_trpl : tr-PL data (the points used by the NN run) and its σ (σ_eff of the NN run)
    sspl_log_post : likelihood.LogPosterior of mode 'sspl' (σ already set) for modes 'sspl' / 'joint'
    n_jobs : ODE worker processes
    progress_path : optional text file; one line (time, calls, evaluations, rate) at most every
                    ``progress_every_s`` seconds, so that long runs can be followed in a log
    """

    def __init__(self, prior, mode: str, constants=None, npulse=None, T: float = 300.0, qfls=None, y_obs=None,
                 sigma_trpl=None, sspl_log_post=None, n_jobs: int = 8, progress_path=None,
                 progress_every_s: float = 60.0):
        if mode not in ("trpl", "sspl", "joint"):
            raise ValueError("mode must be 'trpl', 'sspl' or 'joint'")
        self.prior, self.mode, self.dim = prior, mode, prior.dim
        self.has_trpl = mode in ("trpl", "joint")
        self.has_sspl = mode in ("sspl", "joint")
        if self.has_sspl and sspl_log_post is None:
            raise ValueError(f"mode '{mode}' needs sspl_log_post")
        self.constants, self.npulse, self.T = constants, npulse, float(T)
        self.qfls = None if qfls is None else np.asarray(qfls, np.float64)
        self.y_obs = None if y_obs is None else np.asarray(y_obs, np.float64)
        self.sigma_trpl = None if sigma_trpl is None else float(sigma_trpl)
        self.sspl_log_post = sspl_log_post
        self.n_jobs = int(n_jobs)
        self.counter = PyCounter(f"ode_{mode}")
        self.n_uncovered = 0
        self.time_ode_s = 0.0                 # wall time spent in the parallel ODE calls
        self.progress_path, self.progress_every_s = progress_path, float(progress_every_s)
        self._t_start = self._t_last = time.perf_counter()

    # ------------------------------------------------------------------
    def ode_curves(self, theta) -> np.ndarray:
        """ODE log10 τ at the tr-PL points for every row of theta (B, n); NaN outside the simulated range."""
        from joblib import Parallel, delayed
        from . import ode
        ode._ode_module()
        theta = np.atleast_2d(np.asarray(theta, np.float64))
        t0 = time.perf_counter()
        rows = Parallel(n_jobs=self.n_jobs)(delayed(_ode_row)(th, self.prior, self.npulse, self.constants,
                                                               self.qfls, self.T) for th in theta)
        self.time_ode_s += time.perf_counter() - t0
        return np.asarray(rows, np.float64)

    def trpl_terms(self, theta) -> dict:
        """tr-PL MSE, coverage flag and log L term for every row of theta."""
        Y = self.ode_curves(theta)
        ok = np.all(np.isfinite(Y), axis=1)
        mse = np.full(len(Y), np.nan)
        mse[ok] = np.mean((Y[ok] - self.y_obs[None, :]) ** 2, axis=1)
        s = self.sigma_trpl
        ll = np.where(ok, -mse / (2.0 * s * s) - math.log(s) - LOG_SQRT_2PI, -np.inf)
        return {"mse": mse, "covered": ok, "log_lik": ll, "curves": Y}

    def np_log_lik_theta(self, theta) -> np.ndarray:
        """log L(θ) for rows of theta (numpy); one ODE solve per row and/or the ss-PL term."""
        theta = np.atleast_2d(np.asarray(theta, np.float64))
        ll = np.zeros(len(theta))
        if self.has_trpl:
            t = self.trpl_terms(theta)
            ll += t["log_lik"]
            self.n_uncovered += int((~t["covered"]).sum())
        if self.has_sspl:
            ll += self.sspl_log_post.np_log_lik_theta(theta)
        self.counter.record(len(theta))
        self._progress()
        return np.where(np.isfinite(ll), ll, -np.inf)

    def _progress(self) -> None:
        if self.progress_path is None:
            return
        now = time.perf_counter()
        if now - self._t_last < self.progress_every_s:
            return
        self._t_last = now
        c = self.counter.snapshot()
        with open(self.progress_path, "a") as fh:
            fh.write(f"{time.strftime('%H:%M:%S')} {self.mode}: {c['n_calls']} calls, {c['n_points']:,} evaluations, "
                     f"{c['n_points'] / max(now - self._t_start, 1e-9):.1f} evaluations/s since start, "
                     f"{self.n_uncovered} not covering the data\n")

    def np_log_prob_u(self, U) -> np.ndarray:
        """log p(u) = log L(θ(u)) − log V + log|J(u)| (as LogPosterior.log_prob_u); −inf outside the support."""
        U = np.atleast_2d(np.asarray(U, np.float64))
        theta, logdet = self.prior.forward(U)
        theta, lp = theta.numpy(), logdet.numpy() - self.prior.log_volume
        out = np.full(len(U), -np.inf)
        ok = np.isfinite(lp) & self.prior.in_support(theta)
        if ok.any():
            out[ok] = self.np_log_lik_theta(theta[ok]) + lp[ok]
        return out

    def sigmas(self) -> dict:
        out = {}
        if self.has_trpl:
            out["trpl"] = [self.sigma_trpl]
        if self.has_sspl:
            out["sspl"] = float(self.sspl_log_post.sigmas()["sspl"])
        return out

    def warm_up(self) -> float:
        """Start the worker processes and compile numba in each of them (untimed elsewhere); returns seconds."""
        if not self.has_trpl:
            return 0.0
        t0 = time.perf_counter()
        mid = 0.5 * (np.asarray(self.prior.lo) + np.asarray(self.prior.hi))
        self.ode_curves(np.tile(mid, (self.n_jobs, 1)))
        self.time_ode_s = 0.0
        return time.perf_counter() - t0


class OdePopulationEvaluator(PopulationEvaluator):
    """
    smc.PopulationEvaluator for the ODE likelihood: ev(u_free) → (log L, log p0(u)) of a whole population in one
    parallel batch.  Same coordinates and prior density as the NN evaluator (fixed parameters at u = 0).
    """

    def __init__(self, ode_log_post: OdeLogPosterior, max_batch: int | None = None):
        self.log_post = ode_log_post
        self.prior = ode_log_post.prior
        self.free = np.asarray(self.prior.free_idx, int)
        self.d = len(self.free)
        self.D = self.prior.dim
        self.n_fixed = self.D - self.d
        self.counter = ode_log_post.counter
        self.max_batch = None if max_batch is None else int(max_batch)
        self.fixed_shape = False
        self._const = -self.prior.log_volume + self.n_fixed * LOG4

    def __call__(self, U) -> tuple[np.ndarray, np.ndarray]:
        U = np.atleast_2d(np.asarray(U, np.float64))
        theta, logdet = self.prior.forward(self.full_u(U))
        theta, lp = theta.numpy(), logdet.numpy() + self._const
        return self.log_post.np_log_lik_theta(theta), lp

    def precompile(self, sizes) -> float:
        return self.log_post.warm_up()
