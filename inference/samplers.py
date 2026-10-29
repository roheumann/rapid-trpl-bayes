"""
samplers.py — full-dimensional posterior sampling
=================================================

Primary sampler: TensorFlow Probability NUTS (task T1, R2 comment 4)
--------------------------------------------------------------------
* Runs in the unconstrained coordinates u of ``prior.Prior`` on the density
  log p(u) = log L + log pi + log|J|  (``LogPosterior.log_prob_u``).
* All chains form one batch: every leapfrog step is ONE batched NN call.
* Warm-up in three stages (similar to Stan's windowed adaptation):
    stage 1  identity metric, dual-averaging step size
    stage 2  metric estimated from the second half of stage 1
    stage 3  metric re-estimated from the second half of stage 2,
             step size adapted again; sampling continues from here with the
             final averaged step size.
  The metric enters as an affine reparameterisation u = mu + L z (L the
  Cholesky factor of the regularised pooled covariance, or its diagonal),
  which is equivalent to a dense (diagonal) mass matrix.  mu and L are
  tf.Variables, so changing them does not trigger recompilation.
* The chain is advanced in compiled chunks of ``chunk`` steps
  (``tf.function(jit_compile=True)``).  The first chunk includes the XLA
  compilation; its extra time is reported as ``compile_s``.
* Diagnostics recorded per step and chain: divergences, leapfrog steps
  (= gradient evaluations), tree depth, acceptance, step size, energy.

Cross-check: emcee (affine-invariant ensemble sampler)
------------------------------------------------------
``run_emcee`` samples the SAME density log p(u) with
``EnsembleSampler(vectorize=True)`` — one batched call per step for all
walkers — and reports the integrated autocorrelation time.

Both return a :class:`SamplerResult` with samples in theta and in display
coordinates and an ArviZ DataTree (``to_arviz``).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class SamplerResult:
    """Posterior samples and diagnostics of one sampler run."""
    method: str
    theta: np.ndarray                 # (chains, draws, D)
    u: np.ndarray                     # (chains, draws, D)
    sample_stats: dict = field(default_factory=dict)   # name -> (chains, draws)
    info: dict = field(default_factory=dict)            # timings, counts, settings

    @property
    def n_chains(self) -> int:
        return self.theta.shape[0]

    @property
    def n_draws(self) -> int:
        return self.theta.shape[1]

    def flat_theta(self) -> np.ndarray:
        return self.theta.reshape(-1, self.theta.shape[-1])


# ─────────────────────────────────────────────────────────────────────────────
# initialisation helpers
# ─────────────────────────────────────────────────────────────────────────────

def init_dispersed(dim: int, n: int, rng=None, radius: float = 2.0) -> np.ndarray:
    """
    Over-dispersed starting points: u ~ U(-radius, radius)^D (Stan-style).
    With radius 2 each parameter starts in the central ~76 % of its prior
    range (the ordering is built into u, so all points are valid).
    """
    rng = np.random.default_rng(rng)
    return rng.uniform(-radius, radius, size=(n, dim))


def init_around(u_center, n: int, rng=None, scale: float = 0.05) -> np.ndarray:
    """Starting points in a small Gaussian ball around u_center (e.g. the best fit)."""
    rng = np.random.default_rng(rng)
    u_center = np.asarray(u_center, np.float64)
    return u_center[None, :] + scale * rng.standard_normal((n, len(u_center)))


# ─────────────────────────────────────────────────────────────────────────────
# NUTS
# ─────────────────────────────────────────────────────────────────────────────

class NutsSampler:
    """
    Chunked, XLA-compiled NUTS with staged step-size and metric adaptation.

    Parameters
    ----------
    log_post : object with ``log_prob_u(u)``, ``prior``, ``dim``, ``counter``
        (a likelihood.LogPosterior, or a bound multi-dataset posterior)
    n_chains : int
    groups : (n_chains,) int array or None
        Chains with the same group id sample the same posterior (e.g. the
        same synthetic data set in SBC).  The metric is estimated per group,
        stuck chains are reset onto chains of their own group.  Every chain
        has its own dual-averaging step size.  None = one group.
    chunk : int              NUTS steps per compiled call
    max_tree_depth : int     NUTS maximum tree depth (2**depth - 1 leapfrogs)
    target_accept : float    dual-averaging target acceptance probability (0.95: the
                             ridge-shaped posteriors diverge often with 0.8)
    """

    def __init__(self, log_post, n_chains: int, groups=None, chunk: int = 50,
                 max_tree_depth: int = 10, target_accept: float = 0.95):
        import tensorflow as tf
        import tensorflow_probability as tfp

        self.log_post = log_post
        self.dim = log_post.dim
        self.n_chains = int(n_chains)
        self.groups = (np.zeros(self.n_chains, int) if groups is None
                       else np.asarray(groups, int))
        if self.groups.shape != (self.n_chains,):
            raise ValueError("groups must have one entry per chain")
        self.chunk = int(chunk)
        self.max_tree_depth = int(max_tree_depth)
        self.target_accept = float(target_accept)
        C, D = self.n_chains, self.dim
        # per-chain affine preconditioner u = mu + L z (rows identical within a group)
        self._mu = tf.Variable(np.zeros((C, D)), dtype=tf.float64, trainable=False)
        self._L = tf.Variable(np.tile(np.eye(D), (C, 1, 1)), dtype=tf.float64, trainable=False)

        def target(z):
            u = self._mu + tf.einsum("cij,cj->ci", self._L, z)
            return log_post.log_prob_u(u)

        def make_kernel(step_size, num_adapt):
            nuts = tfp.mcmc.NoUTurnSampler(target, step_size=step_size,
                                           max_tree_depth=self.max_tree_depth)
            return tfp.mcmc.DualAveragingStepSizeAdaptation(
                nuts, num_adaptation_steps=num_adapt, target_accept_prob=self.target_accept)

        def trace_fn(_, pkr):
            r = pkr.inner_results
            return {"lp": r.target_log_prob,
                    "n_steps": r.leapfrogs_taken,
                    "diverging": r.has_divergence,
                    "reach_max_depth": r.reach_max_depth,
                    "log_accept_ratio": r.log_accept_ratio,
                    "energy": r.energy,
                    "step_size": pkr.new_step_size[:, 0]}

        @tf.function(jit_compile=True)
        def bootstrap(z, step_size, num_adapt):
            return make_kernel(step_size, num_adapt).bootstrap_results(z)

        @tf.function(jit_compile=True)
        def run_chunk(z, pkr, num_adapt, seed):
            kernel = make_kernel(pkr.new_step_size, num_adapt)
            res = tfp.mcmc.sample_chain(
                num_results=self.chunk, current_state=z, previous_kernel_results=pkr,
                kernel=kernel, trace_fn=trace_fn, return_final_kernel_results=True, seed=seed)
            return res.all_states, res.trace, res.final_kernel_results

        self._bootstrap = bootstrap
        self._run_chunk = run_chunk

    # ------------------------------------------------------------------
    def _set_metric(self, u_hist, good, kind: str):
        """
        Estimate mu, L per group from the u samples (steps, C, D) of the good
        chains of that group (second half of the previous stage).
        """
        C, D = self.n_chains, self.dim
        mu = np.zeros((C, D))
        L = np.tile(np.eye(D), (C, 1, 1))
        if kind != "identity":
            for g in np.unique(self.groups):
                ch = np.nonzero((self.groups == g) & good)[0]
                if len(ch) == 0:
                    ch = np.nonzero(self.groups == g)[0]
                x = u_hist[:, ch].reshape(-1, D)
                n = len(x)
                m = x.mean(axis=0)
                cov = np.cov(x, rowvar=False)
                # Stan's regularisation towards a small multiple of the identity
                cov = (n / (n + 5.0)) * cov + 1e-3 * (5.0 / (n + 5.0)) * np.eye(D)
                if kind == "diag":
                    Lg = np.diag(np.sqrt(np.diag(cov)))
                else:
                    try:
                        Lg = np.linalg.cholesky(cov)
                    except np.linalg.LinAlgError:
                        Lg = np.diag(np.sqrt(np.diag(cov)))
                sel = self.groups == g
                mu[sel], L[sel] = m, Lg
        self._mu.assign(mu)
        self._L.assign(L)

    def _u_from_z(self, z):
        """z (..., C, D) -> u with the per-chain metric."""
        return self._mu.numpy() + np.einsum("cij,...cj->...ci", self._L.numpy(), z)

    def _z_from_u(self, u):
        mu, L = self._mu.numpy(), self._L.numpy()
        return np.stack([np.linalg.solve(L[c], u[c] - mu[c]) for c in range(self.n_chains)])

    def _advance(self, z, pkr, n_steps, num_adapt, seed_gen, keep: bool, timings: list):
        """Run n_steps (multiple of chunk) and collect u samples and traces."""
        import tensorflow as tf
        n_chunks = int(math.ceil(n_steps / self.chunk))
        us, traces = [], []
        for _ in range(n_chunks):
            seed = tf.constant(seed_gen.integers(0, 2**31 - 1, size=2), tf.int32)
            t0 = time.perf_counter()
            states, trace, pkr = self._run_chunk(z, pkr, tf.constant(num_adapt, tf.int32), seed)
            z = states[-1]
            states_np = states.numpy()                       # (chunk, C, D), forces execution
            timings.append(time.perf_counter() - t0)
            us.append(self._u_from_z(states_np))
            # during warm-up only the log density is kept (for stuck-chain detection)
            traces.append({k: v.numpy() for k, v in trace.items()} if keep
                          else {"lp": trace["lp"].numpy()})
        u_all = np.concatenate(us)[:n_steps]
        tr = {k: np.concatenate([t[k] for t in traces])[:n_steps] for k in traces[0]}
        return z, pkr, u_all, tr

    def _stuck_chains(self, lp_stage: np.ndarray, threshold: float) -> np.ndarray:
        """
        Chains whose mean log density over the second half of a warm-up stage
        lies more than ``threshold`` below the best chain OF THEIR GROUP.
        Such chains sit in regions with negligible posterior mass (a plateau
        at the prior boundary or a poor local mode) and would otherwise also
        spoil the metric and the step size.
        """
        m = lp_stage[len(lp_stage) // 2:].mean(axis=0)
        m = np.where(np.isfinite(m), m, -np.inf)
        stuck = np.zeros(self.n_chains, bool)
        for g in np.unique(self.groups):
            sel = self.groups == g
            if sel.sum() > 1:
                stuck[sel] = m[sel] < np.max(m[sel]) - threshold
        return stuck

    def run(self, init_u, n_warmup: int = 1000, n_draws: int = 2000, metric: str = "dense",
            init_step_size: float = 0.1, reset_stuck: bool = True,
            stuck_threshold: float | None = None, seed: int = 0,
            verbose: bool = True) -> SamplerResult:
        """
        Run warm-up and sampling.

        Parameters
        ----------
        init_u : (n_chains, D) starting points in u coordinates
        n_warmup, n_draws : steps per chain (rounded up to multiples of chunk)
        metric : 'dense' | 'diag' | 'identity'
        reset_stuck : after warm-up stages 1 and 2, move chains whose mean log
            density is more than ``stuck_threshold`` (default max(10, 2 D))
            below the best chain of their group onto the position of a
            randomly chosen good chain of the same group.  Only warm-up is
            affected (no effect on detailed balance of the sampling phase);
            the resets are reported in ``info['chains_reset']``.  The metric
            is estimated from the good chains only.
        """
        import tensorflow as tf
        from .benchmark import count_evals

        C, D = self.n_chains, self.dim
        init_u = np.asarray(init_u, np.float64)
        if init_u.shape != (C, D):
            raise ValueError(f"init_u must have shape ({C}, {D})")
        ch = self.chunk
        n1 = max(ch, int(round(0.15 * n_warmup / ch)) * ch)
        n2 = max(ch, int(round(0.35 * n_warmup / ch)) * ch)
        n3 = max(2 * ch, int(math.ceil((n_warmup - n1 - n2) / ch)) * ch)
        n_draws = int(math.ceil(n_draws / ch)) * ch
        rng = np.random.default_rng(seed)
        thr = float(stuck_threshold) if stuck_threshold is not None else max(10.0, 2.0 * D)
        timings = []
        resets = []
        t_start = time.perf_counter()
        step0 = lambda s: tf.fill([C, 1], tf.constant(s, tf.float64))    # per-chain step sizes

        with count_evals(self.log_post.counter) as cnt:
            # stage 1: identity metric (u = z)
            self._set_metric(None, None, "identity")
            z = tf.constant(self._z_from_u(init_u))
            pkr = self._bootstrap(z, step0(init_step_size), tf.constant(n1, tf.int32))
            z, pkr, u_hist, tr = self._advance(z, pkr, n1, n1, rng, False, timings)
            if verbose:
                print(f"  NUTS warm-up stage 1 ({n1} steps) done, median step size "
                      f"{float(np.median(pkr.new_step_size.numpy())):.3g}")
            # stages 2 and 3: (reset stuck chains,) new metric from the second
            # half of the previous stage, restart the step-size adaptation
            for stage, n_s in ((2, n2), (3, n3)):
                u_last = self._u_from_z(z.numpy())
                good = np.ones(C, bool)
                if reset_stuck and C > 1:
                    stuck = self._stuck_chains(tr["lp"], thr)
                    good = ~stuck
                    if stuck.any():
                        spread = u_hist[len(u_hist) // 2:, good].reshape(-1, D).std(axis=0)
                        for c in np.nonzero(stuck)[0]:
                            donors = np.nonzero((self.groups == self.groups[c]) & good)[0]
                            d = rng.choice(donors)
                            u_last[c] = u_last[d] + 0.1 * spread * rng.standard_normal(D)
                        resets.append({"after_stage": stage - 1, "chains": np.nonzero(stuck)[0].tolist()})
                        if verbose:
                            print(f"  reset {int(stuck.sum())} stuck chain(s) after stage {stage - 1}"
                                  + (f": {np.nonzero(stuck)[0].tolist()}" if stuck.sum() <= 20 else ""))
                self._set_metric(u_hist[len(u_hist) // 2:], good, metric)
                z = tf.constant(self._z_from_u(u_last))
                pkr = self._bootstrap(z, step0(0.5), tf.constant(n_s, tf.int32))
                z, pkr, u_hist, tr = self._advance(z, pkr, n_s, n_s, rng, False, timings)
                if verbose:
                    print(f"  NUTS warm-up stage {stage} ({n_s} steps) done, median step size "
                          f"{float(np.median(pkr.new_step_size.numpy())):.3g}")
            t_warm = time.perf_counter() - t_start
            # sampling: step counter > n3, so the averaged step size is used
            z, pkr, u_draws, tr = self._advance(z, pkr, n_draws, n3, rng, True, timings)
        t_total = time.perf_counter() - t_start

        # (draws, C, D) -> (C, draws, D)
        u_draws = np.transpose(u_draws, (1, 0, 2))
        theta = self.log_post.prior.forward(u_draws.reshape(-1, D))[0].numpy().reshape(C, -1, D)
        stats = {k: np.transpose(v, (1, 0)) for k, v in tr.items()}
        stats["acceptance_rate"] = np.exp(np.minimum(stats.pop("log_accept_ratio"), 0.0))
        stats["tree_depth"] = np.log2(stats["n_steps"] + 1.0)
        chunk_t = np.asarray(timings)
        compile_s = float(max(0.0, chunk_t[0] - np.median(chunk_t[1:]))) if len(chunk_t) > 1 else 0.0
        info = {
            "n_chains": C, "n_groups": int(len(np.unique(self.groups))),
            "n_warmup": n1 + n2 + n3, "n_draws": n_draws, "metric": metric,
            "warmup_stages": [n1, n2, n3],
            "step_size": float(np.median(pkr.new_step_size.numpy())),
            "max_tree_depth": self.max_tree_depth, "target_accept": self.target_accept,
            "time_total_s": t_total, "time_warmup_s": t_warm, "time_sampling_s": t_total - t_warm,
            "compile_s": compile_s,
            "chains_reset": resets, "stuck_threshold": thr,
            "n_divergent": int(stats["diverging"].sum()),
            "frac_max_depth": float(stats["reach_max_depth"].mean()),
            "mean_leapfrogs": float(stats["n_steps"].mean()),
            # gradient evaluations actually used by the chains in the sampling phase
            "leapfrogs_sampling": int(stats["n_steps"].sum()),
            # all batched log-density(+grad) evaluations incl. warm-up and the
            # padding of chains that finished their tree early
            "evals_total_points": cnt.counts["n_points"], "evals_total_calls": cnt.counts["n_calls"],
        }
        if verbose:
            print(f"  NUTS done: {C} chains x {n_draws} draws, {info['n_divergent']} divergences, "
                  f"mean {info['mean_leapfrogs']:.1f} leapfrogs/step, {t_total:.1f} s "
                  f"(compile ≈ {compile_s:.1f} s)")
        return SamplerResult("nuts", theta=theta, u=u_draws, sample_stats=stats, info=info)


def run_nuts(log_post, init_u, n_warmup=1000, n_draws=2000, metric="dense", chunk=50,
             max_tree_depth=10, target_accept=0.95, seed=0, verbose=True, sampler=None, **run_kwargs):
    """
    Convenience wrapper around :class:`NutsSampler`.

    Pass the ``sampler`` returned by a previous call to reuse its compiled
    functions (same log_post and number of chains).  Further keyword
    arguments (reset_stuck, stuck_threshold, init_step_size) go to
    :meth:`NutsSampler.run`.  Returns (SamplerResult, sampler).
    """
    init_u = np.asarray(init_u, np.float64)
    groups = run_kwargs.pop("groups", None)
    same_groups = (sampler is not None and
                   np.array_equal(sampler.groups, np.zeros(len(init_u), int) if groups is None else np.asarray(groups)))
    if (sampler is None or sampler.n_chains != len(init_u) or sampler.log_post is not log_post
            or sampler.chunk != chunk or sampler.max_tree_depth != max_tree_depth
            or sampler.target_accept != target_accept or not same_groups):
        sampler = NutsSampler(log_post, n_chains=len(init_u), groups=groups, chunk=chunk,
                              max_tree_depth=max_tree_depth, target_accept=target_accept)
    return sampler.run(init_u, n_warmup=n_warmup, n_draws=n_draws, metric=metric,
                       seed=seed, verbose=verbose, **run_kwargs), sampler


# ─────────────────────────────────────────────────────────────────────────────
# emcee
# ─────────────────────────────────────────────────────────────────────────────

def run_emcee(log_post, init_u, n_steps: int = 20000, burn_frac: float = 0.5,
              thin: int | None = None, moves: str = "stretch", reset_stuck: bool = True,
              stuck_threshold: float | None = None, seed: int = 0,
              progress: bool = False) -> SamplerResult:
    """
    Affine-invariant ensemble sampling of log p(u) (vectorised, emcee).

    The run has two parts: burn-in (``burn_frac * n_steps``) and production
    (the rest).  Between them, walkers whose mean log density over the second
    half of the burn-in lies more than ``stuck_threshold`` (default
    max(10, 2 D)) below the best walker are moved onto randomly chosen good
    walkers (+ tiny jitter).  Such walkers sit on plateaus of negligible
    posterior mass; an ensemble sampler cannot pull them back efficiently and
    they would inflate the intervals and the autocorrelation time.  Only the
    production part is returned; ``info['walkers_reset']`` reports the resets.

    Parameters
    ----------
    init_u : (n_walkers, D) starting points (n_walkers >= 2 D; use >= 4 D)
    n_steps : total steps per walker (production should be >= 50 x tau_int;
              check info['steps_over_tau'])
    burn_frac : fraction of the steps used as burn-in
    thin : thinning of the production chain (default: half the smallest tau_int)
    moves : 'stretch' (emcee default, affine-invariant stretch move) or 'de'
            (differential-evolution move only).  NOTE: the mixture
            80 % DEMove + 20 % DESnookerMove was found to be biased in a test on
            the prior alone (all 68 % widths ~4 % too narrow), so the snooker
            move is not used.
    """
    import emcee
    from .benchmark import count_evals

    init_u = np.asarray(init_u, np.float64)
    W, D = init_u.shape
    if W < 2 * D:
        raise ValueError(f"emcee needs at least 2*D = {2 * D} walkers")
    mv = emcee.moves.DEMove() if moves == "de" else None
    np.random.seed(seed)        # emcee uses the global numpy RNG
    rng = np.random.default_rng(seed)
    thr = float(stuck_threshold) if stuck_threshold is not None else max(10.0, 2.0 * D)
    n_burn = int(burn_frac * n_steps)
    n_prod = n_steps - n_burn
    sampler = emcee.EnsembleSampler(W, D, log_post.np_log_prob_u, vectorize=True, moves=mv)
    t0 = time.perf_counter()
    reset = []
    with count_evals(log_post.counter) as cnt:
        # burn-in
        state = sampler.run_mcmc(init_u, n_burn, progress=progress)
        pos = np.array(state.coords)
        if reset_stuck:
            m = sampler.get_log_prob(discard=n_burn // 2).mean(axis=0)
            m = np.where(np.isfinite(m), m, -np.inf)
            stuck = m < np.max(m) - thr
            if stuck.any():
                good = np.nonzero(~stuck)[0]
                donors = rng.choice(good, size=int(stuck.sum()))
                spread = pos[good].std(axis=0)
                pos[stuck] = pos[donors] + 1e-3 * spread * rng.standard_normal((len(donors), D))
                reset = np.nonzero(stuck)[0].tolist()
        # production
        sampler.reset()
        # reset walkers are near-copies of good ones: emcee's linear-independence check of the start (a heuristic)
        # can then fail although the ensemble is fine; skip it only in that case (no reset: unchanged)
        sampler.run_mcmc(pos, n_prod, progress=progress, skip_initial_state_check=bool(reset))
    t_total = time.perf_counter() - t0
    chain = sampler.get_chain()                                   # (n_prod, W, D)
    tau = sampler.get_autocorr_time(tol=0)
    if thin is None:
        thin = max(1, int(0.5 * np.nanmin(tau))) if np.isfinite(tau).any() else 1    # NaN tau: too short a run
    u = np.transpose(chain[::thin], (1, 0, 2))                    # (W, draws, D)
    lp = np.transpose(sampler.get_log_prob()[::thin], (1, 0))
    theta = log_post.prior.forward(u.reshape(-1, D))[0].numpy().reshape(W, -1, D)
    info = {"n_walkers": W, "n_steps": n_steps, "burn": n_burn, "thin": thin, "moves": moves,
            "walkers_reset": reset, "stuck_threshold": thr,
            "acceptance_fraction": float(np.mean(sampler.acceptance_fraction)),
            "tau_int": [float(t) for t in tau], "steps_over_tau": float(n_prod / np.nanmax(tau)),
            "time_total_s": t_total,
            "evals_total_points": cnt.counts["n_points"], "evals_total_calls": cnt.counts["n_calls"]}
    return SamplerResult("emcee", theta=theta, u=u, sample_stats={"lp": lp}, info=info)


# ─────────────────────────────────────────────────────────────────────────────
# ArviZ conversion and diagnostics
# ─────────────────────────────────────────────────────────────────────────────

def to_arviz(result: SamplerResult, prior):
    """DataTree with the posterior in display coordinates (E_t in eV)."""
    import arviz as az
    disp = prior.to_display(result.theta.reshape(-1, prior.dim)).reshape(result.theta.shape)
    post = {k: disp[..., i] for i, k in enumerate(prior.display_keys)}
    stats = {k: v for k, v in result.sample_stats.items()}
    data = {"posterior": post}
    if stats:
        data["sample_stats"] = stats
    return az.from_dict(data)


def diagnostics(result: SamplerResult, prior) -> pd.DataFrame:
    """
    ArviZ summary per parameter: mean, sd, rank-normalised R-hat, bulk/tail
    ESS and MCSE.  (For emcee the walkers are not independent chains, so its
    R-hat is only indicative; use the autocorrelation time in result.info.)
    """
    import arviz as az
    dt = to_arviz(result, prior)
    try:
        s = az.summary(dt, round_to="none")
    except Exception:
        s = az.summary(dt)
    s = s.apply(pd.to_numeric, errors="coerce")
    s.index.name = "parameter"
    return s
