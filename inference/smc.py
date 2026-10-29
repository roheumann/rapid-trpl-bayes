"""
smc.py — adaptive importance sampling for ridge-shaped and multimodal posteriors
===============================================================================
15/10/2026  (revision; alternative to NUTS/emcee in samplers.py and to the fixed eigenvector-cylinder IS in
cylinder.py)

Why
---
The tr-PL posteriors have curved ridges (E_t – τ_p – N_t) and plateaus down to the prior bounds.  NUTS needs
target_accept 0.95 and stuck-chain resets, and on hard cases its R-hat stays > 1.1.  The fixed cylinder IS uses
the linearised (Laplace) covariance and misses curved ridges.  The two samplers here need only batched likelihood
values (no gradients and no sequential chain).  Both also return the log evidence log Z, which is needed for model
comparison (1 trap vs 2 traps).

(a) Likelihood-tempered SMC  (``run_smc``)
------------------------------------------
Target sequence π_β(u) ∝ L(θ(u))^β p0(u), 0 = β_0 < β_1 < … < β_T = 1.  The coordinates u are the unconstrained
coordinates of ``prior.Prior`` (the same as NUTS/emcee), restricted to the FREE parameters (fixed parameters sit
at the centre of their degenerate interval, u = 0).  p0 is the prior density in u (π(θ(u))|J|, normalised).

  1. N particles per island from the prior (θ ~ π, u = prior.inverse(θ)).
  2. Next β by bisection so that the ESS of the incremental weights w_i = L_i^{Δβ} is ``ess_target`` · N
     (adaptive tempering).  log Z += log mean_i w_i  (the weights are uniform after every resampling).
  3. Proposal covariance Σ = weighted covariance of the WEIGHTED particles (before resampling).
  4. Systematic resampling.
  5. Random-walk Metropolis moves u' = u + s·chol(Σ)·ε, s adapted towards ``target_accept`` after every step;
     with probability ``p_small`` the step is shrunk by ``small_factor`` (symmetric mixture → exact MH), which
     helps in narrow ridge directions.  The number of moves per stage is adaptive: moves continue until a
     particle has moved with probability ≥ ``p_move`` (n ≈ log(1 − p_move)/log(1 − acc)), within [min_mh, max_mh].
  6. Repeat 2–5 until β = 1; the last stage ends with resampling + moves, so the returned particles are
     equally weighted.  The weighted particles BEFORE the last resampling are also returned (``weighted``).

Every likelihood evaluation is ONE batched call for all particles of all islands (the whole population; it is
only split if it exceeds ``max_batch`` rows).  There is no sequential chain anywhere.

Islands: ``n_islands`` independent SMC populations run in the same batch.  They act as independent "chains" for
R-hat (``samplers.diagnostics``).  The spread of their log Z gives the Monte Carlo error of log Z.
Combined: log Z = log mean_i Z_i, and the islands are combined with equal weight.

σ path: SMC estimates log Z_β = log ∫ L^β p0 for every β of the schedule.  For the MSE likelihood
log L = −MSE/(2σ²) − log(σ√2π), L^β is (up to a constant) the likelihood with σ' = σ/√β, so the schedule gives the
evidence for a whole range of σ' ≥ σ at no extra cost (``sigma_path``).

(b) AMIS refinement  (``run_amis``)
-----------------------------------
Adaptive multiple importance sampling (Cornuet et al. 2012) in the same unconstrained coordinates u as the SMC
(free parameters only).  In u every draw lies inside the prior support, and a posterior that is close to the
uniform prior on the box is smooth.  (A first version worked in decades x: for the near-prior 2-trap 85:15
posterior, 2/3 of the t draws then fell outside the box, the ESS was 8 % and log Z was 0.03 too low.)

  * iteration 0: the scrambled-Sobol points of the eigenvector cylinders (exact cylinder mixture density,
    cylinder.mixture_log_q, in decades), plus ``alpha_def`` · N draws from the prior (defensive component);
  * iteration t ≥ 1: proposal q_t = (1 − α) T_t + α p0, T_t a Student-t mixture (ν = ``nu``) in u, fitted by
    weighted EM to ALL draws so far (covariances × ``cov_inflation`` for sampling; a new component at the
    heaviest draw if its weight exceeds ``birth_threshold`` × the mean weight).  The first fit is initialised from
    the cylinders: one component per cylinder segment (``n_split`` segments along its axis), with the moments of
    the weighted cylinder points.  The component counts are deterministic (stratified mixture sampling);
  * every component c enters through r_c = q_c / p0, its density relative to the prior.  Such ratios do not depend
    on the coordinates, so the cylinder density in decades gives r = q_x(x(u)) / π_x(x(u));
  * balance heuristic: every draw, also the old ones, gets the weight w = L / r̄ with r̄ = Σ_t N_t r_t / Σ_t N_t
    (deterministic-mixture weights, Owen & Zhou 2000), recomputed after every iteration;  Z = mean(w);
  * PSIS k̂ (validate.psis) of the final weights is the reliability diagnostic (< 0.7 usable, < 0.5 good).

Result
------
``WeightedSamplerResult`` is a ``samplers.SamplerResult``.  ``theta``/``u`` hold EQUALLY weighted draws
(chains, draws, D): for SMC the final particles of every island, for AMIS a systematic resample of the weighted
draws.  So ``samplers.to_arviz``, ``samplers.diagnostics`` and ``report.export_samples`` work unchanged.
``weighted`` holds the weighted samples (θ, normalised weights); ``corner_group`` writes them in the HDF5 corner
format of corner_export.py (Origin script and corner_plot preview).

Note on ESS: for SMC particles the ArviZ ESS is computed along the particle index.  Systematic resampling keeps
descendants of the same ancestor next to each other, so duplicates show up as autocorrelation.  The
importance-sampling ESS (Kish, 1/Σw²) and the number of distinct particles are given in ``info``.  For AMIS the
Kish ESS of the weights and k̂ are the relevant numbers.  The ArviZ ESS of the resampled draws is only indicative.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.linalg import solve_triangular
from scipy.special import gammaln, logsumexp

from .samplers import SamplerResult

LOG4 = math.log(4.0)


# ─────────────────────────────────────────────────────────────────────────────
# result
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class WeightedSamplerResult(SamplerResult):
    """
    SamplerResult with importance weights, log evidence and the adaptation history.

    theta, u      (chains, draws, D) equally weighted draws (see module docstring)
    weighted      dict: theta (M, D), w (M,) normalised, log_w (M,) unnormalised, log_lik (M,), group (M,)
                  (island for SMC, iteration for AMIS)
    log_evidence  log Z = log ∫ L(θ) π(θ) dθ  (π normalised on the prior support)
    stages        DataFrame, one row per SMC stage and island / per AMIS iteration
    """
    weighted: dict = field(default_factory=dict)
    log_evidence: float = float("nan")
    log_evidence_se: float = float("nan")
    stages: pd.DataFrame = field(default_factory=pd.DataFrame)
    history: list = field(default_factory=list)

    def flat_weighted(self):
        """(theta (M, D), normalised weights (M,)) of the weighted samples."""
        return self.weighted["theta"], self.weighted["w"]

    @property
    def ess(self) -> float:
        """Kish ESS of the weighted samples."""
        w = self.weighted["w"]
        return float(1.0 / np.sum(w ** 2))


def ess_kish(log_w) -> float:
    """Kish effective sample size of (unnormalised) log weights; -inf entries are zero weights."""
    lw = np.asarray(log_w, float)
    lw = lw[np.isfinite(lw)]
    if len(lw) == 0:
        return 0.0
    return float(np.exp(2.0 * logsumexp(lw) - logsumexp(2.0 * lw)))


def normalise_log_weights(log_w) -> np.ndarray:
    lw = np.asarray(log_w, float)
    fin = np.isfinite(lw)
    w = np.zeros_like(lw)
    if fin.any():
        w[fin] = np.exp(lw[fin] - logsumexp(lw[fin]))
    return w


def systematic_resample(w, n: int, rng) -> np.ndarray:
    """Systematic resampling: indices (sorted, n) with E[count_i] = n w_i and |count_i − n w_i| < 1."""
    w = np.asarray(w, float)
    cs = np.cumsum(w)
    cs /= cs[-1]
    pos = (rng.random() + np.arange(n)) / n
    return np.minimum(np.searchsorted(cs, pos, side="right"), len(w) - 1)


def weighted_cov(X, w, ridge: float = 1e-10) -> tuple[np.ndarray, np.ndarray]:
    """Weighted mean and (bias-corrected) covariance of the rows of X; w normalised."""
    w = np.asarray(w, float)
    mu = w @ X
    Xc = X - mu
    denom = max(1.0 - float(np.sum(w ** 2)), 1e-12)
    cov = (w[:, None] * Xc).T @ Xc / denom
    cov = 0.5 * (cov + cov.T)
    cov += ridge * max(float(np.trace(cov)) / len(mu), 1e-12) * np.eye(len(mu))
    return mu, cov


def _chol(cov):
    """Cholesky factor with jitter fallback."""
    d = len(cov)
    jit = 0.0
    for _ in range(8):
        try:
            return np.linalg.cholesky(cov + jit * np.eye(d))
        except np.linalg.LinAlgError:
            jit = max(1e-12, 10 * jit) if jit else 1e-10 * max(np.trace(cov) / d, 1e-12)
    w, V = np.linalg.eigh(cov)
    return V @ np.diag(np.sqrt(np.clip(w, 1e-12, None)))


# ─────────────────────────────────────────────────────────────────────────────
# batched evaluation of the whole population
# ─────────────────────────────────────────────────────────────────────────────
class PopulationEvaluator:
    """
    log L(θ(u)) and log p0(u) of a whole particle population in ONE batched XLA call.

    u are the free-parameter coordinates (B, d); fixed parameters are inserted at u = 0 (the centre of their
    degenerate interval).  log p0 is the normalised prior density of the free coordinates in u:
    log p0(u) = −log V + log|J(u)| + n_fixed·log 4  (the last term removes the logistic density at u = 0 of the
    fixed coordinates, so p0 integrates to 1 over the free coordinates).

    The batch is padded to a power of two (≥ 64) so that only a few XLA programs are compiled; padded rows are
    removed from the output and from the evaluation counter.  Batches larger than ``max_batch`` are split
    (GPU memory); None = never split.  ``fixed_shape=True`` (needs max_batch) pads EVERY chunk to max_batch:
    one XLA program for all calls, so timings never include a compilation.
    """

    def __init__(self, log_post, max_batch: int | None = None, fixed_shape: bool = False):
        import tensorflow as tf
        self.log_post = log_post
        self.prior = log_post.prior
        self.free = np.asarray(self.prior.free_idx, int)
        self.d = len(self.free)
        self.D = self.prior.dim
        self.n_fixed = self.D - self.d
        self.counter = log_post.counter
        self.max_batch = None if max_batch is None else int(max_batch)
        self.fixed_shape = bool(fixed_shape) and self.max_batch is not None
        E = np.zeros((self.d, self.D))
        E[np.arange(self.d), self.free] = 1.0
        E = tf.constant(E, tf.float64)
        const = -self.prior.log_volume + self.n_fixed * LOG4

        def f(u_free):
            u = tf.matmul(u_free, E)
            theta, logdet = self.prior.forward(u)
            return log_post.log_lik(theta), logdet + const

        self._fn = tf.function(f, jit_compile=True, reduce_retracing=True)

    # ------------------------------------------------------------------
    def full_u(self, u_free) -> np.ndarray:
        u_free = np.atleast_2d(u_free)
        u = np.zeros((len(u_free), self.D))
        u[:, self.free] = u_free
        return u

    def theta(self, u_free) -> np.ndarray:
        return self.prior.forward_np(self.full_u(u_free))

    def u_free_from_theta(self, theta) -> np.ndarray:
        return self.prior.inverse(theta)[:, self.free]

    @staticmethod
    def _bucket(b: int) -> int:
        return int(max(64, 2 ** int(math.ceil(math.log2(max(b, 1))))))

    def __call__(self, U) -> tuple[np.ndarray, np.ndarray]:
        import tensorflow as tf
        U = np.atleast_2d(np.asarray(U, np.float64))
        step = len(U) if self.max_batch is None else self.max_batch
        lls, lps = [], []
        for s in range(0, len(U), max(step, 1)):
            ub = U[s:s + step]
            b = len(ub)
            if self.fixed_shape:
                bucket = self.max_batch
            else:
                bucket = self._bucket(b) if self.max_batch is None else min(self._bucket(b), self.max_batch)
            pad = bucket - b
            if pad > 0:
                ub = np.concatenate([ub, np.repeat(ub[:1], pad, axis=0)])
            ll, lp = self._fn(tf.constant(ub))
            if pad > 0:
                self.counter.n_points.assign_add(-pad)
            lls.append(ll.numpy()[:b])
            lps.append(lp.numpy()[:b])
        return np.concatenate(lls), np.concatenate(lps)

    def precompile(self, sizes) -> float:
        """Compile the XLA programs for the given batch sizes (evaluations removed from the counter)."""
        snap = self.counter.snapshot()
        t0 = time.perf_counter()
        for n in sorted(set(int(s) for s in sizes)):
            self(np.zeros((n, self.d)))
        self.counter.n_calls.assign(snap["n_calls"])
        self.counter.n_points.assign(snap["n_points"])
        return time.perf_counter() - t0


# ─────────────────────────────────────────────────────────────────────────────
# (a) tempered SMC
# ─────────────────────────────────────────────────────────────────────────────
def _tempered(beta, ll):
    """β·log L with β·(−inf) = −inf for β > 0 and 0 for β = 0."""
    beta = np.asarray(beta, float)
    return np.where(np.isfinite(ll), beta * np.where(np.isfinite(ll), ll, 0.0),
                    np.where(beta > 0, -np.inf, 0.0))


def next_beta(ll, beta: float, ess_target: float, tol: float = 1e-10) -> float:
    """
    Largest β' ∈ (β, 1] with ESS(L^{β'−β}) ≥ ess_target · N (bisection; uniform previous weights).
    """
    ll = np.asarray(ll, float)
    fin = np.isfinite(ll)
    if not fin.any():
        raise RuntimeError("all particles have log L = -inf")
    n = len(ll)

    def ess_frac(db):
        lw = np.where(fin, db * np.where(fin, ll, 0.0), -np.inf)
        return math.exp(2.0 * logsumexp(lw) - logsumexp(2.0 * lw)) / n

    if ess_frac(1.0 - beta) >= ess_target:
        return 1.0
    lo, hi = 0.0, 1.0 - beta
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if ess_frac(mid) >= ess_target:
            lo = mid
        else:
            hi = mid
        if hi - lo <= tol * hi:
            break
    # lo keeps ESS >= target; never return beta itself (no progress)
    return float(min(1.0, beta + (lo if lo > 0.0 else hi)))


def run_smc(log_post, n_particles: int = 4096, n_islands: int = 4, ess_target: float = 0.5,
            target_accept: float = 0.25, p_move: float = 0.95, min_mh: int = 2, max_mh: int = 30,
            p_small: float = 0.3, small_factor: float = 0.2, max_stages: int = 500,
            max_batch: int | None = None, history_n: int = 1000, seed: int = 0,
            evaluator: PopulationEvaluator | None = None, verbose: bool = True) -> WeightedSamplerResult:
    """
    Likelihood-tempered SMC in the unconstrained coordinates of the prior (see module docstring).

    Parameters
    ----------
    log_post : likelihood.LogPosterior (needs log_lik(theta) in TF, prior, counter)
    n_particles : particles per island
    n_islands : independent populations (R-hat across islands, MC error of log Z)
    ess_target : ESS fraction kept by every tempering step (0.5 = the usual choice)
    target_accept : acceptance rate the random-walk scale is adapted to
    p_move, min_mh, max_mh : number of MH moves per stage (see module docstring)
    p_small, small_factor : fraction of proposals with a step shrunk by small_factor
    max_batch : rows per surrogate call (None = whole population in one call)
    history_n : particles of island 0 stored after every stage (plots)
    evaluator : a PopulationEvaluator to reuse (compiled XLA programs)

    Returns
    -------
    WeightedSamplerResult (method 'smc'); theta/u (n_islands, n_particles, D)
    """
    from .benchmark import count_evals
    rng = np.random.default_rng(seed)
    ev = evaluator if evaluator is not None else PopulationEvaluator(log_post, max_batch=max_batch)
    prior = log_post.prior
    d, D = ev.d, ev.D
    I, N = int(n_islands), int(n_particles)
    s0 = 2.38 / math.sqrt(d)

    t_start = time.perf_counter()
    rows, history = [], []
    weighted = [None] * I
    with count_evals(log_post.counter) as cnt:
        # 1) prior draws
        th0 = prior.sample(I * N, rng=rng)
        U = ev.u_free_from_theta(th0).reshape(I, N, d)
        ll, lp = ev(U.reshape(-1, d))
        LL, LP = ll.reshape(I, N), lp.reshape(I, N)
        beta = np.zeros(I)
        logZ = np.zeros(I)
        scale = np.full(I, s0)
        active = np.ones(I, bool)
        stage = 0
        if history_n:
            history.append({"stage": 0, "beta": 0.0,
                            "theta": ev.theta(U[0, :history_n])})
        while active.any():
            stage += 1
            if stage > max_stages:
                raise RuntimeError(f"SMC did not reach beta = 1 within {max_stages} stages (beta = {beta})")
            t_stage = time.perf_counter()
            snap = log_post.counter.snapshot()
            act = np.nonzero(active)[0]
            Lc = np.zeros((I, d, d))
            st = {}
            # 2)–4) reweight, covariance from the weighted particles, resample
            for i in act:
                b_new = next_beta(LL[i], beta[i], ess_target)
                db = b_new - beta[i]
                lw = _tempered(db, LL[i])
                inc = float(logsumexp(lw) - math.log(N))
                logZ[i] += inc
                w = normalise_log_weights(lw)
                ess = 1.0 / np.sum(w ** 2)
                _, cov = weighted_cov(U[i], w)
                Lc[i] = _chol(cov)
                if b_new >= 1.0:          # weighted particles at beta = 1 (before the last resampling)
                    weighted[i] = {"u": U[i].copy(), "log_w": lw.copy(), "log_lik": LL[i].copy()}
                idx = systematic_resample(w, N, rng)
                U[i], LL[i], LP[i] = U[i][idx], LL[i][idx], LP[i][idx]
                st[i] = {"island": int(i), "stage": stage, "beta_prev": float(beta[i]), "beta": float(b_new),
                         "log_z_increment": inc, "log_z": float(logZ[i]), "ess": float(ess), "ess_frac": float(ess / N),
                         "n_unique_resampled": int(len(np.unique(idx)))}
                beta[i] = b_new
            # 5) random-walk MH moves at the new beta (one batched call per move for all moving islands)
            notmoved = np.ones(I)
            n_mh = np.zeros(I, int)
            acc_hist = {i: [] for i in act}
            moved_any = {i: np.zeros(N, bool) for i in act}
            while True:
                moving = [i for i in act
                          if n_mh[i] < max_mh and (n_mh[i] < min_mh or notmoved[i] > 1.0 - p_move)]
                if not moving:
                    break
                mv = np.asarray(moving)
                eps = rng.standard_normal((len(mv), N, d))
                s = scale[mv][:, None] * np.where(rng.random((len(mv), N)) < p_small, small_factor, 1.0)
                prop = U[mv] + s[..., None] * np.einsum("aij,anj->ani", Lc[mv], eps)
                ll_p, lp_p = ev(prop.reshape(-1, d))
                ll_p, lp_p = ll_p.reshape(len(mv), N), lp_p.reshape(len(mv), N)
                for a, i in enumerate(mv):
                    log_alpha = (_tempered(beta[i], ll_p[a]) + lp_p[a]) - (_tempered(beta[i], LL[i]) + LP[i])
                    log_alpha = np.where(np.isnan(log_alpha), -np.inf, log_alpha)
                    acc = np.log(rng.random(N)) < log_alpha
                    U[i][acc], LL[i][acc], LP[i][acc] = prop[a][acc], ll_p[a][acc], lp_p[a][acc]
                    moved_any[i] |= acc
                    r = float(acc.mean())
                    acc_hist[i].append(r)
                    notmoved[i] *= (1.0 - r)
                    n_mh[i] += 1
                    scale[i] = float(np.clip(scale[i] * math.exp(r - target_accept), 1e-3 * s0, 3.0 * s0))
            dt = time.perf_counter() - t_stage
            dc = log_post.counter.delta_since(snap)
            for i in act:
                row = st[i]
                row.update({"n_mh": int(n_mh[i]), "acceptance": float(np.mean(acc_hist[i])) if acc_hist[i] else np.nan,
                            "acceptance_last": acc_hist[i][-1] if acc_hist[i] else np.nan,
                            "scale": float(scale[i]), "frac_moved": float(moved_any[i].mean()),
                            "n_unique_after_move": int(len(np.unique(U[i], axis=0))),
                            "evals_stage_all_islands": int(dc["n_points"]), "time_stage_s": dt})
                rows.append(row)
                if beta[i] >= 1.0:
                    active[i] = False
            if history_n:
                history.append({"stage": stage, "beta": float(beta[0]),
                                "theta": ev.theta(U[0, :history_n])})
            if verbose:
                b_txt = ", ".join(f"{b:.3g}" for b in beta)
                print(f"  SMC stage {stage:3d}: beta = [{b_txt}], acc = "
                      f"{np.mean([rows[-k]['acceptance'] for k in range(1, len(act) + 1)]):.2f}, "
                      f"moves = {int(n_mh[act].max())}, {dt:.2f} s")
    t_total = time.perf_counter() - t_start

    # ---- outputs
    stages = pd.DataFrame(rows)
    theta = np.stack([ev.theta(U[i]) for i in range(I)])                        # (I, N, D)
    u_full = np.stack([ev.full_u(U[i]) for i in range(I)])
    lp_post = LL + LP
    th_w = np.concatenate([ev.theta(weighted[i]["u"]) for i in range(I)])
    lw_i = [weighted[i]["log_w"] for i in range(I)]
    w_all = np.concatenate([normalise_log_weights(l) / I for l in lw_i])        # islands with equal weight
    log_w_all = np.concatenate([l - logsumexp(l[np.isfinite(l)]) - math.log(I) for l in lw_i])
    ll_all = np.concatenate([weighted[i]["log_lik"] for i in range(I)])
    logz_comb = float(logsumexp(logZ) - math.log(I))
    logz_se = float(np.std(logZ, ddof=1) / math.sqrt(I)) if I > 1 else float("nan")
    from .validate import psis
    k_last = max(float(psis(l[np.isfinite(l)])[1]) for l in lw_i)       # worst island
    n_unique = [int(len(np.unique(U[i], axis=0))) for i in range(I)]
    info = {"n_particles": N, "n_islands": I, "dim_free": d, "ess_target": ess_target,
            "target_accept": target_accept, "p_move": p_move, "min_mh": min_mh, "max_mh": max_mh,
            "p_small": p_small, "small_factor": small_factor,
            "n_stages": [int((stages.island == i).sum()) for i in range(I)],
            "log_evidence_islands": logZ.tolist(), "log_evidence": logz_comb, "log_evidence_se": logz_se,
            "ess_weighted_kish": float(1.0 / np.sum(w_all ** 2)), "n_unique_final": n_unique,
            "k_hat_last_reweighting": float(k_last),
            "time_total_s": t_total,
            "evals_total_points": cnt.counts["n_points"], "evals_total_calls": cnt.counts["n_calls"],
            "max_batch": ev.max_batch}
    if verbose:
        print(f"  SMC done: {I} islands x {N} particles, stages {info['n_stages']}, "
              f"log Z = {logz_comb:.3f} ± {logz_se:.3f}, {cnt.counts['n_points']:,} evaluations in "
              f"{cnt.counts['n_calls']} calls, {t_total:.1f} s")
    return WeightedSamplerResult(
        "smc", theta=theta, u=u_full,
        sample_stats={"lp": lp_post, "log_lik": LL.copy()}, info=info,
        weighted={"theta": th_w, "w": w_all, "log_w": log_w_all, "log_lik": ll_all,
                  "group": np.repeat(np.arange(I), N)},
        log_evidence=logz_comb, log_evidence_se=logz_se, stages=stages, history=history)


def sigma_path(result: WeightedSamplerResult, sigma: float) -> pd.DataFrame:
    """
    log Z as a function of σ' = σ/√β from the tempering schedule of an SMC run (tr-PL MSE likelihood with ONE
    term of standard deviation σ; see module docstring).  One row per stage and island.
        log Z(σ') = log Z_β + β log(σ√2π) − log(σ'√2π)
    """
    st = result.stages
    c = math.log(sigma * math.sqrt(2.0 * math.pi))
    out = st[["island", "stage", "beta", "log_z"]].copy()
    out = out[out.beta > 0]
    out["sigma"] = sigma / np.sqrt(out.beta)
    out["log_z_sigma"] = out.log_z + out.beta * c - np.log(out.sigma * math.sqrt(2.0 * math.pi))
    return out.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# (b) AMIS: Student-t mixture, weighted EM, balance heuristic
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class TMixture:
    """Mixture of multivariate Student-t distributions (common ν)."""
    weights: np.ndarray        # (K,)
    means: np.ndarray          # (K, d)
    covs: np.ndarray           # (K, d, d)
    nu: float = 5.0

    def __post_init__(self):
        self.weights = np.asarray(self.weights, float) / np.sum(self.weights)
        self.means = np.atleast_2d(np.asarray(self.means, float))
        self.covs = np.asarray(self.covs, float).reshape(len(self.weights), self.means.shape[1], -1)
        self.chols = np.stack([_chol(c) for c in self.covs])
        self.logdets = 2.0 * np.sum(np.log(np.abs(np.einsum("kii->ki", self.chols))), axis=1)

    @property
    def K(self) -> int:
        return len(self.weights)

    @property
    def d(self) -> int:
        return self.means.shape[1]

    def _maha(self, x):
        """(M, K) squared Mahalanobis distances."""
        x = np.atleast_2d(x)
        out = np.empty((len(x), self.K))
        for k in range(self.K):
            z = solve_triangular(self.chols[k], (x - self.means[k]).T, lower=True)
            out[:, k] = np.sum(z * z, axis=0)
        return out

    def component_logpdf(self, x, maha=None):
        """(M, K) log t_k(x) (without the mixture weights)."""
        nu, d = self.nu, self.d
        maha = self._maha(x) if maha is None else maha
        c = gammaln(0.5 * (nu + d)) - gammaln(0.5 * nu) - 0.5 * d * math.log(nu * math.pi)
        return c - 0.5 * self.logdets[None, :] - 0.5 * (nu + d) * np.log1p(maha / nu)

    def logpdf(self, x) -> np.ndarray:
        return logsumexp(self.component_logpdf(x) + np.log(self.weights)[None, :], axis=1)

    def counts(self, n: int, rng) -> np.ndarray:
        """Deterministic (systematic) number of draws per component."""
        idx = systematic_resample(self.weights, n, rng)
        return np.bincount(idx, minlength=self.K)

    def sample(self, n: int, rng, counts=None):
        """n draws (stratified component counts); returns (x (n, d), component (n,))."""
        counts = self.counts(n, rng) if counts is None else counts
        xs, cs = [], []
        for k, m in enumerate(counts):
            if m == 0:
                continue
            z = rng.standard_normal((m, self.d))
            g = rng.chisquare(self.nu, size=m) / self.nu
            xs.append(self.means[k] + (z @ self.chols[k].T) / np.sqrt(g)[:, None])
            cs.append(np.full(m, k))
        return np.vstack(xs), np.concatenate(cs)


def _regularise(cov, floor: float):
    cov = 0.5 * (cov + cov.T)
    lam, V = np.linalg.eigh(cov)
    return (V * np.maximum(lam, floor ** 2)) @ V.T


def importance_subsample(w, n_max: int, rng):
    """
    Unbiased weighted subsample ('Russian roulette'): keep draw i with probability p_i = min(1, c w_i), weight
    w_i / p_i, with c such that about n_max draws are kept.  Returns (indices, new weights).
    """
    w = np.asarray(w, float)
    pos = np.nonzero(w > 0)[0]
    if len(pos) <= n_max:
        return pos, w[pos]
    wp = w[pos]
    lo, hi = 0.0, n_max / wp.min()
    for _ in range(100):
        c = 0.5 * (lo + hi)
        if np.minimum(1.0, c * wp).sum() > n_max:
            hi = c
        else:
            lo = c
    p = np.minimum(1.0, lo * wp)
    keep = rng.random(len(wp)) < p
    return pos[keep], wp[keep] / p[keep]


def fit_tmixture_em(x, w, init: TMixture, n_iter: int = 30, var_floor: float = 0.01,
                    min_weight: float = 2e-3, tol: float = 1e-6, n_max: int | None = 50_000,
                    rng=None) -> TMixture:
    """
    Weighted EM for a Student-t mixture with fixed ν (McLachlan & Peel 2000, ch. 7), warm-started from ``init``.

    x (M, d), w (M,) non-negative sample weights (normalised inside).  ``var_floor``: minimum standard
    deviation of every covariance eigen-direction.  Components whose weight drops below ``min_weight`` are
    removed.  More than ``n_max`` weighted points are first reduced by an unbiased importance subsample.
    """
    x = np.atleast_2d(x)
    w = np.asarray(w, float)
    if n_max is not None:
        keep, wk = importance_subsample(w, int(n_max), np.random.default_rng(rng))
        x, w = x[keep], wk / wk.sum()
    else:
        keep = w > 0
        x, w = x[keep], w[keep] / w[keep].sum()
    mix = init
    nu, d = init.nu, x.shape[1]
    prev = -np.inf
    for _ in range(int(n_iter)):
        maha = mix._maha(x)
        logp = mix.component_logpdf(x, maha) + np.log(mix.weights)[None, :]
        ll_i = logsumexp(logp, axis=1)
        obj = float(w @ ll_i)
        r = np.exp(logp - ll_i[:, None])                      # responsibilities (M, K)
        wr = w[:, None] * r                                   # (M, K)
        u = (nu + d) / (nu + maha)                            # latent precision scale
        pk = wr.sum(axis=0)
        alive = pk > min_weight
        if not alive.any():
            break
        means, covs = [], []
        for k in np.nonzero(alive)[0]:
            a = wr[:, k] * u[:, k]
            mu = (a @ x) / a.sum()
            xc = x - mu
            cov = (a[:, None] * xc).T @ xc / pk[k]
            means.append(mu)
            covs.append(_regularise(cov, var_floor))
        mix = TMixture(pk[alive], np.array(means), np.array(covs), nu)
        if abs(obj - prev) < tol * max(1.0, abs(obj)):
            break
        prev = obj
    return mix


def tmixture_from_labels(x, w, labels, nu: float = 5.0, var_floor: float = 0.01, mix_floor: float = 0.05,
                         min_points: int = 20) -> TMixture:
    """
    One t component per label: weighted mean/covariance of the points of that label.  The weighted
    covariance is blended with the unweighted one (``mix_floor``) so that components whose points carry
    almost no weight still get a sensible shape.  Component weights ∝ label weight + a small floor.
    """
    x = np.atleast_2d(x)
    w = np.asarray(w, float)
    labs = [l for l in np.unique(labels) if (labels == l).sum() >= min_points]
    W, M, C = [], [], []
    for l in labs:
        m = labels == l
        xs, ws = x[m], w[m]
        mu_u, cov_u = xs.mean(axis=0), np.cov(xs, rowvar=False)
        if ws.sum() > 0:
            wn = ws / ws.sum()
            mu_w, cov_w = weighted_cov(xs, wn)
            ess = 1.0 / np.sum(wn ** 2)
            if ess < d_min(x.shape[1]):
                mu_w, cov_w = mu_u, cov_u
        else:
            mu_w, cov_w = mu_u, cov_u
        cov = (1.0 - mix_floor) * cov_w + mix_floor * cov_u
        W.append(ws.sum())
        M.append(mu_w)
        C.append(_regularise(cov, var_floor))
    W = np.asarray(W)
    W = W / W.sum() if W.sum() > 0 else np.full(len(W), 1.0 / len(W))
    W = 0.9 * W + 0.1 / len(W)
    return TMixture(W, np.array(M), np.array(C), nu)


def fit_tmixture(x, w, n_components: int = 8, nu: float = 5.0, var_floor: float = 0.01, em_iter: int = 50,
                 seed: int = 0) -> TMixture:
    """
    t-mixture fitted to weighted samples (e.g. SMC particles in decades): weighted k-means initialisation,
    then weighted EM.  Used to start AMIS from SMC when no cylinder system exists.
    """
    from sklearn.cluster import KMeans
    x = np.atleast_2d(x)
    w = np.asarray(w, float) / np.sum(w)
    km = KMeans(n_clusters=int(n_components), n_init=4, random_state=seed).fit(x, sample_weight=w)
    init = tmixture_from_labels(x, w, km.labels_, nu=nu, var_floor=var_floor, mix_floor=0.0, min_points=2 * x.shape[1] + 2)
    return fit_tmixture_em(x, w, init, n_iter=em_iter, var_floor=var_floor)


def d_min(d: int) -> float:
    """Minimum Kish ESS for a weighted covariance in d dimensions."""
    return 2.0 * d + 2.0


class PriorX:
    """
    Uniform prior in the decade coordinates x of the free parameters (cylinder.DecadeMap): density and
    support test.  log π_x = −log V_free − Σ log|dx/dθ| on the support.
    """

    def __init__(self, prior, dmap, theta_ref):
        self.prior, self.dmap = prior, dmap
        self.theta_ref = np.asarray(theta_ref, float)
        fixed = np.asarray(prior.fixed_mask, bool)
        self.log_vol_free = prior.log_volume - float(np.sum(np.log(prior.hi[fixed] - prior.lo[fixed])))
        self.log_density = -self.log_vol_free - float(np.sum(np.log(np.abs(dmap.A))))

    def theta(self, x):
        return self.dmap.to_theta(x, self.theta_ref)

    def in_support(self, x):
        return self.prior.in_support(self.theta(x))

    def logpdf(self, x):
        return np.where(self.in_support(x), self.log_density, -np.inf)


# AMIS works in the free unconstrained coordinates u of the prior (the same as run_smc): every draw lies in the
# support, and a near-uniform posterior on the box is smooth in u.  Every proposal component c provides
# log r_c(u) = log q_c(u) − log p0(u), its density relative to the prior in u.  Such ratios do not depend on
# the coordinates, so the cylinder mixture (a density in decades x) enters as q_x(x(u)) / π_x(x(u)).
class _PriorComponent:
    """Defensive component: the prior itself (r = 1)."""

    def __init__(self, prior, theta_ref, ev):
        self.prior, self.theta_ref, self.ev = prior, np.asarray(theta_ref, float), ev

    def sample(self, n: int, rng):
        th = self.prior.sample(n, rng=rng)
        th[:, self.prior.fixed_mask] = self.theta_ref[self.prior.fixed_mask]
        return self.ev.u_free_from_theta(th)

    def log_ratio(self, U, lp0, theta):
        return np.zeros(len(U))


class _CylinderComponent:
    """The exact cylinder mixture density (cylinder.mixture_log_q), 0 outside box ∩ support."""

    def __init__(self, cyl, n_accepted, log_vol, px: PriorX):
        self.cyl, self.n_accepted, self.log_vol, self.px = cyl, np.asarray(n_accepted), np.asarray(log_vol), px
        self.log_jac = float(np.sum(np.log(cyl.es.sigma)))       # x = x* + V diag(σ) z

    def log_ratio(self, U, lp0, theta):
        from .cylinder import mixture_log_q, in_box
        x = self.px.dmap.to_x(theta)
        lq = mixture_log_q(self.cyl, self.cyl.es.to_z(x), self.n_accepted, self.log_vol) - self.log_jac
        ok = in_box(x, self.cyl.lo, self.cyl.hi) & self.px.prior.in_support(theta)
        return np.where(ok, lq - self.px.log_density, -np.inf)


class _TComponent:
    """t-mixture in u: r = T(u) / p0(u)."""

    def __init__(self, mix: TMixture):
        self.mix = mix

    def sample(self, n: int, rng):
        return self.mix.sample(n, rng)

    def log_ratio(self, U, lp0, theta):
        return self.mix.logpdf(U) - lp0


@dataclass
class _Proposal:
    """One AMIS iteration: deterministic mixture of (component, count) pairs."""
    parts: list                # [(component with log_ratio, n_drawn)]

    @property
    def n(self) -> int:
        return int(sum(c for _, c in self.parts))

    def log_ratio(self, U, lp0, theta):
        n = self.n
        terms = [p.log_ratio(U, lp0, theta) + math.log(c / n) for p, c in self.parts if c > 0]
        return logsumexp(np.stack(terms), axis=0)


def _birth(mix: TMixture, x_new) -> TMixture:
    """Add a component at x_new: covariance of the component with the highest density there, / 4; weight 5 %."""
    lp = mix.component_logpdf(np.atleast_2d(x_new))[0] + np.log(mix.weights)
    k = int(np.argmax(lp))
    w = np.r_[0.95 * mix.weights, 0.05]
    return TMixture(w, np.vstack([mix.means, x_new]), np.concatenate([mix.covs, 0.25 * mix.covs[k][None]]), mix.nu)


def cylinder_setup(prior, theta_star, cov_theta_curve_fit, sigma_new: float, sigma_cf: float,
                   radius: float = 2.0, sigma_min: float = 0.02):
    """
    Eigenvector cylinders as in cylinder.ipynb §2–3: covariance of the free parameters in decades, scaled by
    (σ_new/σ_cf)², eigen-system and cylinders inside prior box ∩ physical box.  Returns (dmap, es, cyl).
    """
    from . import cylinder as cy
    dmap = cy.DecadeMap(prior)
    cov = np.asarray(cov_theta_curve_fit, float)
    cov_free = cov[np.ix_(dmap.free, dmap.free)] if cov.shape[0] == prior.dim else cov
    cov_x = dmap.cov_to_x(cov_free) * (sigma_new / sigma_cf) ** 2
    lo, hi = dmap.box()
    x_star = np.clip(dmap.to_x(theta_star)[0], lo, hi)
    es = cy.eigen_system(x_star, cov_x, dmap.keys, sigma_min=sigma_min)
    return dmap, es, cy.Cylinders(es, lo, hi, radius=radius)


def run_amis(log_post, dmap, theta_ref, cyl=None, init_mixture: TMixture | None = None,
             n_per_iter: int = 16384, n_iter: int = 10, alpha_def: float = 0.075, nu: float = 5.0,
             cyl_m: int = 13, n_split: int = 3, em_iter: int = 30, var_floor: float = 0.01,
             cov_inflation: float = 1.5, birth_threshold: float | None = 20.0, max_components: int = 40,
             n_resample: int = 20000, max_batch: int | None = None, seed: int = 0,
             evaluator: PopulationEvaluator | None = None, verbose: bool = True) -> WeightedSamplerResult:
    """
    AMIS with a Student-t mixture proposal (see module docstring).  The t-mixture lives in the free
    unconstrained coordinates u (as run_smc); the cylinders of iteration 0 live in decades x.

    Parameters
    ----------
    log_post : likelihood.LogPosterior
    dmap : cylinder.DecadeMap of the prior
    theta_ref : (D,) values of the fixed parameters (any point of the support)
    cyl : cylinder.Cylinders — iteration 0 uses its Sobol points (2^cyl_m per cylinder) and the exact
        cylinder mixture density; the first t-mixture is initialised from them.  Alternatively
    init_mixture : TMixture in u — iteration 0 draws from (1 − α) init_mixture + α prior.
    n_per_iter : draws per iteration t ≥ 1 (incl. the defensive ones)
    alpha_def : fraction of defensive prior draws in every iteration (0.05–0.10)
    nu : degrees of freedom of the t components
    n_split : segments per cylinder along its axis for the first t-mixture
    em_iter : EM iterations per AMIS iteration
    var_floor : minimum standard deviation of a component in any direction (u units)
    cov_inflation : the proposal of iteration t is the EM fit with covariances × cov_inflation (heavier
        coverage of the tails; EM itself continues from the un-inflated fit)
    birth_threshold : if the largest normalised weight exceeds birth_threshold / M (M = number of draws), a
        new component is placed at that draw before the EM step (covariance of the nearest component / 4,
        weight 5 %): under-covered ridge tips get their own component.  None = no births.
    max_components : no births beyond this number of components
    n_resample : equally weighted draws in result.theta (systematic resample)
    max_batch, evaluator : as in run_smc (all draws of an iteration = one batched call)

    Returns
    -------
    WeightedSamplerResult (method 'amis'); result.info['proposals'] holds the fitted t-mixtures (in u).
    """
    from . import cylinder as cy
    from .benchmark import count_evals
    from .validate import psis
    if cyl is None and init_mixture is None:
        raise ValueError("run_amis needs cyl (cylinder initialisation) or init_mixture")
    rng = np.random.default_rng(seed)
    prior = log_post.prior
    ev = evaluator if evaluator is not None else PopulationEvaluator(log_post, max_batch=max_batch)
    if list(ev.free) != list(dmap.free):
        raise ValueError("evaluator and DecadeMap disagree on the free parameters")
    d = ev.d
    px = PriorX(prior, dmap, theta_ref)
    defensive = _PriorComponent(prior, theta_ref, ev)

    U, TH, LL, LP0, IT, LAB = [], [], [], [], [], []   # draws (u free), theta, log L, log p0(u), iteration, label
    proposals, mixtures, rows, births = [], [], [], [0]
    cache = []                                          # per proposal: log r_t at all draws (list of arrays)
    t_start = time.perf_counter()

    def add_draws(u, it, lab):
        ll, lp0 = ev(u)                                 # ONE batched call for the whole iteration
        U.append(u); TH.append(ev.theta(u)); LL.append(ll); LP0.append(lp0)
        IT.append(np.full(len(u), it)); LAB.append(lab)

    def all_(lst):
        return np.concatenate(lst)

    def update_cache():
        """log r of every proposal at every draw (cached; only new pairs are computed)."""
        Ua, lpa, tha = all_(U), all_(LP0), all_(TH)
        for t, prop in enumerate(proposals[:-1]):       # old proposals at the new draws
            have = sum(len(c) for c in cache[t])
            if have < len(Ua):
                cache[t].append(prop.log_ratio(Ua[have:], lpa[have:], tha[have:]))
        cache.append([proposals[-1].log_ratio(Ua, lpa, tha)])   # newest proposal at all draws

    def weights():
        """balance-heuristic log weights of all draws, log Z and its (relative) standard error."""
        lla = all_(LL)
        Ns = np.array([p.n for p in proposals], float)
        lr = np.stack([all_(c) for c in cache], axis=1)                    # (M, T)
        lrbar = logsumexp(lr + np.log(Ns / Ns.sum())[None, :], axis=1)
        lw = np.where(np.isfinite(lla), lla - lrbar, -np.inf)
        M = len(lw)
        lz = float(logsumexp(lw[np.isfinite(lw)]) - math.log(M))
        w = np.where(np.isfinite(lw), np.exp(lw - lz), 0.0)                # w / Z
        se = float(np.std(w, ddof=1) / math.sqrt(M))                       # SE of log Z (delta method)
        return lw, lz, se

    def record(it, dt, mix):
        lw, lz, se = weights()
        fin = np.isfinite(lw)
        k = float(psis(lw[fin])[1]) if fin.sum() > 10 else float("nan")
        new = all_(IT) == it
        lla = all_(LL)
        lw_own = np.where(np.isfinite(lla[new]), lla[new] - all_(cache[it])[new], -np.inf)   # q_t alone
        rows.append({"iteration": it, "n_new": int(new.sum()), "n_total": int(len(lw)),
                     "n_components": 0 if mix is None else mix.K, "births": births[-1],
                     "log_z": lz, "log_z_se": se,
                     "ess_total": ess_kish(lw), "ess_frac_total": ess_kish(lw) / len(lw),
                     "ess_new_own_proposal": ess_kish(lw_own),
                     "ess_frac_new_own_proposal": ess_kish(lw_own) / max(int(new.sum()), 1),
                     "k_hat": k, "max_weight": float(np.max(normalise_log_weights(lw))),
                     "evals_cumulative": int(log_post.counter.snapshot()["n_points"] - snap0["n_points"]),
                     "time_iter_s": dt})
        if verbose:
            r = rows[-1]
            print(f"  AMIS it {it:2d}: K = {r['n_components']:2d}, log Z = {lz:.3f} ± {se:.3f}, "
                  f"ESS = {r['ess_total']:.0f} ({100 * r['ess_frac_total']:.1f} %), "
                  f"ESS(new|q_t) = {100 * r['ess_frac_new_own_proposal']:.1f} %, k̂ = {k:.2f}, {dt:.1f} s")

    with count_evals(log_post.counter) as cnt:
        snap0 = cnt._start
        # ---- iteration 0: cylinders (or the given mixture) + defensive prior draws
        t0 = time.perf_counter()
        n_def0 = int(math.ceil(alpha_def * n_per_iter))
        if cyl is not None:
            z, lab, nd, na, lv = cy.draw(cyl, dmap, prior, theta_ref, m=cyl_m, seed=int(rng.integers(2**31)))
            u_cyl = ev.u_free_from_theta(dmap.to_theta(cyl.es.to_x(z), theta_ref))
            proposals.append(_Proposal([(_CylinderComponent(cyl, na, lv, px), len(u_cyl)), (defensive, n_def0)]))
            add_draws(np.vstack([u_cyl, defensive.sample(n_def0, rng)]), 0, np.r_[lab, np.full(n_def0, -1)])
        else:
            n_t = n_per_iter - n_def0
            u_t, comp = init_mixture.sample(n_t, rng)
            proposals.append(_Proposal([(_TComponent(init_mixture), n_t), (defensive, n_def0)]))
            add_draws(np.vstack([u_t, defensive.sample(n_def0, rng)]), 0, np.r_[comp, np.full(n_def0, -1)])
            mixtures.append(init_mixture)
        update_cache()
        record(0, time.perf_counter() - t0, init_mixture)

        # ---- iterations 1..n_iter: weighted EM on all draws, sample, reweight everything
        mix = init_mixture
        for it in range(1, int(n_iter) + 1):
            t0 = time.perf_counter()
            Ua = all_(U)
            lw, _, _ = weights()
            w = normalise_log_weights(lw)
            wf = np.minimum(w, math.sqrt(len(w)) / len(w))   # truncated weights for the fit (Ionides 2008)
            if mix is None:          # first fit from the cylinders: one component per cylinder segment
                labs = all_(LAB).astype(float)
                if n_split > 1:
                    z_all = cyl.es.to_z(dmap.to_x(all_(TH)))
                    seg = np.zeros(len(Ua), int)
                    for k in range(cyl.es.D):
                        m = labs == k
                        if m.sum() > 0:
                            edges = np.quantile(z_all[m, k], np.linspace(0, 1, n_split + 1)[1:-1])
                            seg[m] = np.searchsorted(edges, z_all[m, k])
                    labs = np.where(labs >= 0, labs * n_split + seg, -1)
                m = labs >= 0
                mix = tmixture_from_labels(Ua[m], wf[m], labs[m], nu=nu, var_floor=var_floor)
            n_birth = 0
            if birth_threshold is not None and mix.K < max_components and w.max() * len(w) > birth_threshold:
                mix, n_birth = _birth(mix, Ua[np.argmax(w)]), 1
            mix = fit_tmixture_em(Ua, wf, mix, n_iter=em_iter, var_floor=var_floor, rng=rng)
            prop = mix if cov_inflation == 1.0 else TMixture(mix.weights, mix.means, mix.covs * cov_inflation, nu)
            mixtures.append(prop)
            births.append(n_birth)
            n_def = int(math.ceil(alpha_def * n_per_iter))
            n_t = n_per_iter - n_def
            u_t, comp = prop.sample(n_t, rng)
            proposals.append(_Proposal([(_TComponent(prop), n_t), (defensive, n_def)]))
            add_draws(np.vstack([u_t, defensive.sample(n_def, rng)]), it, np.r_[comp, np.full(n_def, -1)])
            update_cache()
            record(it, time.perf_counter() - t0, mix)
    t_total = time.perf_counter() - t_start

    lw, lz, se = weights()
    fin = np.isfinite(lw)
    lw_s, k_hat = psis(lw[fin])
    w = normalise_log_weights(lw)
    th_all, Ua = all_(TH), all_(U)
    idx = systematic_resample(w, int(n_resample), rng)
    stages = pd.DataFrame(rows)
    info = {"n_per_iter": n_per_iter, "n_iter": n_iter, "alpha_def": alpha_def, "nu": nu, "cyl_m": cyl_m,
            "n_split": n_split, "em_iter": em_iter, "var_floor": var_floor, "dim_free": d,
            "cov_inflation": cov_inflation, "birth_threshold": birth_threshold, "n_births": int(sum(births)),
            "n_draws_total": int(len(lw)), "n_iteration0": int(np.sum(all_(IT) == 0)),
            "log_evidence": lz, "log_evidence_se": se, "ess_weighted_kish": ess_kish(lw),
            "k_hat": float(k_hat), "n_unique_resampled": int(len(np.unique(idx))),
            "time_total_s": t_total,
            "evals_total_points": cnt.counts["n_points"], "evals_total_calls": cnt.counts["n_calls"],
            "proposals": mixtures}
    if verbose:
        print(f"  AMIS done: {len(lw):,} draws, log Z = {lz:.3f} ± {se:.3f}, "
              f"ESS = {info['ess_weighted_kish']:.0f}, k̂ = {k_hat:.2f}, {t_total:.1f} s")
    return WeightedSamplerResult(
        "amis", theta=th_all[idx][None], u=ev.full_u(Ua[idx])[None],
        sample_stats={"log_lik": all_(LL)[idx][None]}, info=info,
        weighted={"theta": th_all[fin], "u_free": Ua[fin], "w": w[fin], "log_w": lw[fin],
                  "log_lik": all_(LL)[fin], "group": all_(IT)[fin], "log_w_psis": lw_s},
        log_evidence=lz, log_evidence_se=se, stages=stages, history=mixtures)


# ─────────────────────────────────────────────────────────────────────────────
# comparison helpers
# ─────────────────────────────────────────────────────────────────────────────
def mixture_hdr_fraction(mix: TMixture, x_ref, mass: float = 0.99, n: int = 200_000, seed: int = 0) -> float:
    """
    Fraction of reference points x_ref inside the ``mass`` highest-density region of the t-mixture — the
    analogue of 'NUTS samples inside the cylinders' for the AMIS proposal.  x_ref in the coordinates of the
    mixture (AMIS: free u coordinates, e.g. PopulationEvaluator.u_free_from_theta(theta_nuts)).
    """
    rng = np.random.default_rng(seed)
    xs, _ = mix.sample(n, rng)
    thr = np.quantile(mix.logpdf(xs), 1.0 - mass)
    return float(np.mean(mix.logpdf(np.atleast_2d(x_ref)) >= thr))


def weighted_quantiles(v, w, qs):
    o = np.argsort(v)
    cw = np.cumsum(np.asarray(w, float)[o])
    cw /= cw[-1]
    return np.interp(qs, cw, np.asarray(v, float)[o])


def marginal_table(sets: dict, prior, qs=(0.025, 0.16, 0.5, 0.84, 0.975)) -> pd.DataFrame:
    """
    Quantiles of every free parameter (display coordinates) for several posterior representations.
    sets: name -> (theta (M, D), weights (M,) or None)
    """
    rows = []
    for name, (th, w) in sets.items():
        disp = prior.to_display(th)
        w = np.full(len(th), 1.0 / len(th)) if w is None else np.asarray(w, float)
        for i in prior.free_idx:
            q = weighted_quantiles(disp[:, i], w, qs)
            mean = float(w @ disp[:, i])
            rows.append({"method": name, "parameter": prior.display_keys[i], "mean": mean,
                         "sd": float(np.sqrt(w @ (disp[:, i] - mean) ** 2)),
                         **{f"q{100 * q:g}": v for q, v in zip(qs, q)}})
    return pd.DataFrame(rows)


def c2st(x_a, x_b, w_b=None, n: int = 5000, seed: int = 0, folds: int = 5) -> float:
    """
    Classifier two-sample test: held-out accuracy of a gradient-boosted classifier separating the reference
    draws x_a (equally weighted, e.g. NUTS) from x_b (0.5 = indistinguishable, 1 = fully separable).

    Weighted x_b (importance weights w_b) are NOT resampled: duplicated points would be memorised across the
    folds and inflate the accuracy.  Instead an unbiased importance subsample of distinct points is used with
    its weights as sample weights (in the fit and in the accuracy); both classes carry total weight 1/2.
    Columns are standardised with the pooled mean/sd.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import StratifiedKFold
    rng = np.random.default_rng(seed)
    a = x_a[rng.choice(len(x_a), min(n, len(x_a)), replace=False)]
    if w_b is None:
        b = x_b[rng.choice(len(x_b), min(n, len(x_b)), replace=False)]
        wb = np.full(len(b), 1.0)
    else:
        idx, wb = importance_subsample(np.asarray(w_b, float), n, rng)
        b = x_b[idx]
    X = np.vstack([a, b])
    X = (X - X.mean(axis=0)) / np.maximum(X.std(axis=0), 1e-12)
    y = np.r_[np.zeros(len(a)), np.ones(len(b))]
    sw = np.r_[np.full(len(a), 0.5 / len(a)), 0.5 * wb / wb.sum()]
    acc = []
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, y):
        clf = HistGradientBoostingClassifier(max_iter=200, random_state=seed)
        clf.fit(X[tr], y[tr], sample_weight=sw[tr] / sw[tr].mean())
        ok = clf.predict(X[te]) == y[te]
        wt = sw[te].copy()
        for c in (0, 1):                                  # balance the classes inside the fold
            m = y[te] == c
            wt[m] *= 0.5 / wt[m].sum()
        acc.append(float(np.sum(wt * ok)))
    return float(np.mean(acc))


def resample_weighted(theta, w, n: int, seed: int = 0) -> np.ndarray:
    """n equally weighted draws (systematic resampling, random order)."""
    rng = np.random.default_rng(seed)
    idx = systematic_resample(w, n, rng)
    return theta[rng.permutation(idx)]


# ─────────────────────────────────────────────────────────────────────────────
# corner export (same HDF5 schema as cylinder.ipynb, /corner/physical)
# ─────────────────────────────────────────────────────────────────────────────
def corner_group(group, theta, w, prior, best_theta=None, truth_theta=None, start_theta=None,
                 bins_1d: int = 120, bins_2d: int = 60, smooth: float = 1.0, attrs: dict | None = None):
    """
    Write a physical-decade corner (n1 instead of E_t; corner_export schema) of weighted samples into an
    open h5py group, so that corner_plot.plot_corner and the Origin script read it like the cylinder corner.
    """
    from . import corner_export as ce
    sp = ce.physical_corner_spec(prior, best_theta, truth_theta, start_theta)
    x = sp["dmap"].to_x(theta)
    a = {"layout_kind": sp["layout_kind"]}
    a.update(attrs or {})
    return ce.write_corner(group, x[:, sp["perm"]], np.asarray(w, float) / np.sum(w), sp["keys"], sp["meta"],
                           sp["layout"], sp["ranges"], bins_1d, bins_2d, smooth, diag_mode="marginal",
                           coordinates="decades", attrs=a)


def write_h5(path, result: WeightedSamplerResult, prior, best_theta=None, truth_theta=None, start_theta=None,
             attrs: dict | None = None, **corner_kwargs):
    """One HDF5 file per run: /inference (weighted samples, stages, log Z) and /corner/physical."""
    import json
    import h5py
    from .report import _to_builtin
    with h5py.File(path, "w") as f:
        f.attrs.update({"schema_version": 1, "method": result.method, "n_traps": prior.n_traps,
                        "created": pd.Timestamp.now().isoformat(),
                        "info_json": json.dumps(_to_builtin({k: v for k, v in result.info.items()
                                                             if k != "proposals"}))})
        for k, v in (attrs or {}).items():
            f.attrs[k] = v
        g = f.create_group("inference")
        for k, v in result.weighted.items():
            g[k] = np.asarray(v)
        g["theta_equal_weight"] = result.flat_theta()
        g.attrs.update({"log_evidence": result.log_evidence, "log_evidence_se": result.log_evidence_se,
                        "ess": result.ess, "names": json.dumps(prior.names)})
        gs = g.create_group("stages")
        for c in result.stages.columns:
            gs[c] = result.stages[c].to_numpy()
        corner_group(f.create_group("corner/physical"), result.weighted["theta"], result.weighted["w"], prior,
                     best_theta, truth_theta, start_theta, **corner_kwargs)
    return path
