"""
likelihood.py — the single log-likelihood / log-posterior of the workflow
=========================================================================

Every step (best fit, NUTS, emcee, profile likelihood, sensitivity tests,
later SBC) evaluates :class:`LogPosterior`; there is no second objective
(REVISION_PLAN.md, finding 1.2).

Likelihood (kept in the MSE form of the original submission, Section 2 item 3)
-------------------------------------------------------------------------------
For every data set d (each tr-PL curve, the ss-PL data set):

    log L_d = - MSE_d / (2 sigma_d^2) - log(sigma_d sqrt(2 pi))

    MSE_d = mean_i (y_model,i - y_obs,i)^2

* tr-PL: y = log10 tau_diff at the measured QFLS points (NN surrogate).
* ss-PL: y = QFLS [eV] or log10 PLQY at the measured intensities/QFLS.
* The joint mode sums the terms.  The constant -log(sigma sqrt(2 pi)) is the
  legacy convention of 04_bayesian.py (it only matters for evidences).
* sigma is the error of the WHOLE curve (its RMS residual), not of a single
  point, and it is never rescaled: each curve counts as ONE effective data
  point (this is why the intervals are "conservative").  Written point by
  point, exp(-MSE / 2 sigma^2) = prod_i exp(-r_i^2 / 2 N sigma^2), i.e. the
  same likelihood as iid noise of standard deviation sigma*sqrt(N) per point.
  That per-point equivalent is used ONLY where noise must be drawn point by
  point (SBC data, PPC replicated data); it is not a change of sigma.

sigma_eff
---------
``set_sigma`` changes sigma without recompiling (sigmas are tf.Variables).
The workflow uses the legacy convention sigma_eff,d^2 = sigma_d^2 + MSE_min,d,
where MSE_min,d is the best-fit MSE of data set d fitted ALONE (tr-PL-only and
ss-PL-only fits).  This keeps sigma_trpl identical in the tr-PL-only and the
joint run, so that their posteriors can be compared directly (task T4).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .benchmark import EvalCounter

LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)


@dataclass
class TrplTerm:
    """One measured tr-PL curve: surrogate + observed log10 tau at the used points."""
    surrogate: object            # surrogate.TrplSurrogate
    y_obs: np.ndarray            # log10 tau_diff [s] at surrogate.qfls_used
    sigma: float                 # base sigma (log10 units)
    name: str = "trpl"


@dataclass
class SsplTerm:
    """The measured ss-PL data set."""
    observation: object          # sspl_tf.SsplObservation
    sigma: float                 # base sigma (eV or log10 PLQY)
    name: str = "sspl"
    # 24/10/2026 (Robin, 77:23 film): sigma_delta of a constant offset delta ~ N(0, sigma_delta^2) between the measured and
    # modelled ss-PL (calibration of Voc,rad / G(1 sun) vs the tr-PL axis), integrated out analytically:
    # -log L_ss = (MSE - w m^2)/(2 sigma^2) + const, m = mean residual, w = sigma_delta^2/(sigma_delta^2 + sigma^2);
    # the most probable offset is delta_hat = w m (model - data).  None = no offset (the old likelihood).
    offset_sigma: float | None = None


def make_trpl_term(surrogate, qfls_exp, tau_exp, sigma: float, name: str = "trpl") -> TrplTerm:
    """Build a tr-PL term; keeps only the points that lie on the NN QFLS axis."""
    tau_exp = np.asarray(tau_exp, np.float64)
    if np.any(tau_exp[surrogate.mask_used] <= 0):
        raise ValueError("tau_diff must be > 0 at all used points (clean the data first).")
    y_obs = np.log10(tau_exp[surrogate.mask_used])
    return TrplTerm(surrogate=surrogate, y_obs=y_obs, sigma=float(sigma), name=name)


class LogPosterior:
    """
    log-likelihood and log-posterior for one inference mode.

    Parameters
    ----------
    prior : prior.Prior
    trpl_terms : list of TrplTerm (may be empty for mode 'sspl')
    sspl_term : SsplTerm or None
    mode : 'trpl' | 'sspl' | 'joint'
    batch_size : int
        Maximum number of rows per call in the numpy wrappers (GPU memory).

    Coordinates
    -----------
    * theta : NN input coordinates (see prior.py); bounded.
    * u     : unconstrained coordinates, theta = prior.forward(u).
    The optimisers minimise -log L(theta(u)) (no Jacobian: the best fit is
    the maximum of the likelihood inside the prior support); the samplers
    use log p(u) = log L + log pi + log|J|.
    """

    def __init__(self, prior, trpl_terms=(), sspl_term: SsplTerm | None = None,
                 mode: str = "trpl", batch_size: int = 8192):
        import tensorflow as tf

        if mode not in ("trpl", "sspl", "joint"):
            raise ValueError("mode must be 'trpl', 'sspl' or 'joint'")
        self.prior = prior
        self.dim = prior.dim
        self.mode = mode
        self.batch_size = int(batch_size)
        self.trpl_terms = list(trpl_terms) if mode in ("trpl", "joint") else []
        self.sspl_term = sspl_term if mode in ("sspl", "joint") else None
        if mode in ("trpl", "joint") and not self.trpl_terms:
            raise ValueError(f"mode '{mode}' needs at least one tr-PL term")
        if mode in ("sspl", "joint") and self.sspl_term is None:
            raise ValueError(f"mode '{mode}' needs an ss-PL term")

        # observed data and sigmas as variables: compiled functions read their current values, so new data of the
        # same shape (set_data, e.g. SBC) and sigma_eff updates need no recompilation
        self._y_trpl = [tf.Variable(np.asarray(t.y_obs, np.float64), dtype=tf.float64, trainable=False)
                        for t in self.trpl_terms]
        self.sigma_trpl = [tf.Variable(t.sigma, dtype=tf.float64, trainable=False)
                           for t in self.trpl_terms]
        self.base_sigma_trpl = [t.sigma for t in self.trpl_terms]
        if self.sspl_term is not None:
            self._y_sspl = tf.Variable(np.asarray(self.sspl_term.observation.y_obs, np.float64), dtype=tf.float64,
                                       trainable=False)
            self.sigma_sspl = tf.Variable(self.sspl_term.sigma, dtype=tf.float64, trainable=False)
            self.base_sigma_sspl = self.sspl_term.sigma
        else:
            self.sigma_sspl, self.base_sigma_sspl = None, None

        self.counter = EvalCounter(f"log_lik_{mode}")
        self._compiled = {}

    # ------------------------------------------------------------------
    # data and sigma handling
    # ------------------------------------------------------------------
    def set_data(self, y_trpl=None, y_sspl=None) -> None:
        """
        Replace the observed data (same points, same shapes) without rebuilding or recompiling anything:
        y_trpl = log10 tau at the used tr-PL points (one array, or a list for several tr-PL terms), y_sspl = the
        ss-PL observable in the units of its comparison mode (log10 PLQY or QFLS).  Used by SBC (sbc.run_sbc_smc).
        """
        if y_trpl is not None:
            ys = [y_trpl] if len(self._y_trpl) == 1 and np.ndim(y_trpl) == 1 else list(y_trpl)
            for term, var, y in zip(self.trpl_terms, self._y_trpl, ys):
                y = np.asarray(y, np.float64)
                var.assign(y)
                term.y_obs = y
        if y_sspl is not None:
            if self.sspl_term is None:
                raise ValueError("this LogPosterior has no ss-PL term")
            y = np.asarray(y_sspl, np.float64)
            self._y_sspl.assign(y)
            self.sspl_term.observation.y_obs = y

    def set_sigma(self, trpl=None, sspl=None) -> None:
        """Set sigma of the tr-PL term(s) (scalar or list) and/or the ss-PL term."""
        if trpl is not None:
            vals = np.atleast_1d(trpl)
            if len(vals) == 1:
                vals = np.repeat(vals, len(self.sigma_trpl))
            for v, s in zip(self.sigma_trpl, vals):
                v.assign(float(s))
        if sspl is not None and self.sigma_sspl is not None:
            self.sigma_sspl.assign(float(sspl))

    def sigmas(self) -> dict:
        out = {"trpl": [float(v.numpy()) for v in self.sigma_trpl]}
        if self.sigma_sspl is not None:
            out["sspl"] = float(self.sigma_sspl.numpy())
        return out

    # ------------------------------------------------------------------
    # model predictions and MSEs (TF, batched)
    # ------------------------------------------------------------------
    def mse_terms(self, theta):
        """Per-data-set MSE, list of (B,) tensors (tr-PL terms first, then ss-PL)."""
        import tensorflow as tf
        out = []
        for term, y in zip(self.trpl_terms, self._y_trpl):
            r = term.surrogate.log10_tau(theta) - y
            out.append(tf.reduce_mean(r * r, axis=1))
        if self.sspl_term is not None:
            r = self.sspl_term.observation.predict(theta) - self._y_sspl
            out.append(tf.reduce_mean(r * r, axis=1))
        return out

    def _sspl_offset_weight(self):
        """w = sigma_delta^2 / (sigma_delta^2 + sigma_ss^2) of the marginalised ss-PL offset (0 without offset)."""
        import tensorflow as tf
        sd = self.sspl_term.offset_sigma if self.sspl_term is not None else None
        if not sd:
            return None
        sd2 = tf.constant(float(sd) ** 2, tf.float64)
        return sd2 / (sd2 + self.sigma_sspl * self.sigma_sspl)

    def sspl_offset(self, theta) -> np.ndarray:
        """Most probable ss-PL offset delta_hat = w * mean(model - data) (numpy, (B,)); zeros without offset."""
        import tensorflow as tf
        th = tf.constant(np.atleast_2d(theta), tf.float64)
        w = self._sspl_offset_weight()
        if w is None:
            return np.zeros(len(th))
        r = self.sspl_term.observation.predict(th) - self._y_sspl
        return (w * tf.reduce_mean(r, axis=1)).numpy()

    def log_lik(self, theta):
        """log L(theta), (B,).  Non-finite values are mapped to -inf."""
        import tensorflow as tf
        theta = tf.convert_to_tensor(theta, tf.float64)
        self.counter.record(tf.shape(theta)[0])
        mses = self.mse_terms(theta)
        w = self._sspl_offset_weight()
        if w is not None:                      # marginalised ss-PL offset: MSE - w m^2 (see SsplTerm.offset_sigma)
            m = tf.reduce_mean(self.sspl_term.observation.predict(theta) - self._y_sspl, axis=1)
            mses[-1] = mses[-1] - w * m * m
        sigmas = list(self.sigma_trpl) + ([self.sigma_sspl] if self.sspl_term is not None else [])
        ll = tf.zeros_like(mses[0])
        for mse, s in zip(mses, sigmas):
            ll += -mse / (2.0 * s * s) - tf.math.log(s) - LOG_SQRT_2PI
        return tf.where(tf.math.is_finite(ll), ll, tf.constant(-np.inf, tf.float64))

    def log_prob_theta(self, theta):
        """log posterior (unnormalised) in theta coordinates; -inf outside the support."""
        import tensorflow as tf
        theta = tf.convert_to_tensor(theta, tf.float64)
        lp = self.prior.log_prob(theta)
        # evaluate the likelihood only where the prior is finite (avoids
        # surrogate extrapolation entering e.g. emcee proposals)
        safe = tf.where(tf.math.is_finite(lp)[:, None], theta,
                        tf.broadcast_to(tf.constant(0.5 * (self.prior.lo + self.prior.hi)), tf.shape(theta)))
        return tf.where(tf.math.is_finite(lp), self.log_lik(safe) + lp,
                        tf.constant(-np.inf, tf.float64))

    def log_prob_u(self, u):
        """log p(u) = log L(theta(u)) + log pi(theta(u)) + log|J(u)|, (B,)."""
        theta, logdet = self.prior.forward(u)
        return self.log_lik(theta) - self.prior.log_volume + logdet

    def neg_log_lik_u(self, u, fixed=None):
        """Optimiser objective -log L(theta(u)); ``fixed`` = (j, value) for profiles."""
        theta, _ = self.prior.forward(u, fixed=fixed)
        return -self.log_lik(theta)

    # ------------------------------------------------------------------
    # compiled TF functions (built lazily, XLA)
    # ------------------------------------------------------------------
    def _fn(self, name, builder):
        if name not in self._compiled:
            self._compiled[name] = builder()
        return self._compiled[name]

    @property
    def tf_log_prob_u(self):
        import tensorflow as tf
        return self._fn("lp_u", lambda: tf.function(self.log_prob_u, jit_compile=True,
                                                     reduce_retracing=True))

    @property
    def tf_log_prob_and_grad_u(self):
        import tensorflow as tf
        import tensorflow_probability as tfp

        def f(u):
            return tfp.math.value_and_gradient(self.log_prob_u, u)
        return self._fn("lp_grad_u", lambda: tf.function(f, jit_compile=True, reduce_retracing=True))

    @property
    def tf_neg_log_lik_and_grad_u(self):
        import tensorflow as tf
        import tensorflow_probability as tfp

        def f(u):
            return tfp.math.value_and_gradient(self.neg_log_lik_u, u)
        return self._fn("nll_grad_u", lambda: tf.function(f, jit_compile=True, reduce_retracing=True))

    @property
    def tf_neg_log_lik_u(self):
        import tensorflow as tf
        return self._fn("nll_u", lambda: tf.function(self.neg_log_lik_u, jit_compile=True,
                                                      reduce_retracing=True))

    # ------------------------------------------------------------------
    # numpy wrappers (chunked to batch_size; used by CMA-ES, emcee, reports)
    # ------------------------------------------------------------------
    def _bucket(self, b: int) -> int:
        """Padded batch size: next power of two (>= 16, <= batch_size)."""
        return int(min(self.batch_size, max(16, 2 ** int(np.ceil(np.log2(max(b, 1)))))))

    def _chunked(self, fn, X, n_out=1):
        """
        Evaluate a compiled function on the rows of X in chunks.

        XLA compiles one program per input shape.  CMA-ES (lockstep
        multi-start) produces varying batch sizes, so every chunk is padded
        to a power-of-two bucket; at most log2(batch_size) programs are
        compiled.  Padded rows are removed from the output and from the
        evaluation counter.
        """
        import tensorflow as tf
        X = np.atleast_2d(np.asarray(X, np.float64))
        outs = []
        for s in range(0, len(X), self.batch_size):
            xb = X[s:s + self.batch_size]
            b = len(xb)
            pad = self._bucket(b) - b
            if pad:
                xb = np.concatenate([xb, np.repeat(xb[:1], pad, axis=0)])
            r = fn(tf.constant(xb))
            if pad:
                self.counter.n_points.assign_add(-pad)
            outs.append(r.numpy()[:b] if n_out == 1 else [x.numpy()[:b] for x in r])
        if n_out == 1:
            return np.concatenate(outs)
        return [np.concatenate([o[i] for o in outs]) for i in range(n_out)]

    def precompile(self, max_rows: int = 1024, which=("nll",)) -> float:
        """
        Trigger XLA compilation of the numpy wrappers for all padding buckets
        up to ``max_rows`` so that later timings exclude compilation.
        ``which`` ⊂ {'nll', 'lp', 'll'} ('ll': log L in theta coordinates, used
        by the grid).  Returns the compile time [s].
        The evaluations made here are removed from the counter again.
        """
        import time
        snap = self.counter.snapshot()
        t0 = time.perf_counter()
        theta_mid = 0.5 * (np.asarray(self.prior.lo) + np.asarray(self.prior.hi))
        b = 16
        while True:
            U = np.zeros((min(b, self.batch_size), self.dim))
            if "nll" in which:
                self.np_neg_log_lik_u(U)
            if "lp" in which:
                self.np_log_prob_u(U)
            if "ll" in which:
                self.np_log_lik_theta(np.tile(theta_mid, (len(U), 1)))
            if b >= min(max_rows, self.batch_size):
                break
            b *= 2
        dt = time.perf_counter() - t0
        self.counter.n_calls.assign(snap["n_calls"])
        self.counter.n_points.assign(snap["n_points"])
        return dt

    def np_neg_log_lik_u(self, U) -> np.ndarray:
        """-log L for rows of U; non-finite values replaced by 1e30 (for CMA-ES)."""
        f = self._chunked(self.tf_neg_log_lik_u, U)
        return np.where(np.isfinite(f), f, 1e30)

    def np_log_prob_u(self, U) -> np.ndarray:
        return self._chunked(self.tf_log_prob_u, U)

    def np_log_lik_theta(self, Theta) -> np.ndarray:
        import tensorflow as tf
        fn = self._fn("ll_theta", lambda: tf.function(self.log_lik, jit_compile=True,
                                                       reduce_retracing=True))
        return self._chunked(fn, Theta)

    def np_mse_terms(self, Theta) -> dict:
        """Per-data-set MSE for rows of Theta (numpy dict name -> (B,))."""
        import tensorflow as tf
        fn = self._fn("mse", lambda: tf.function(lambda th: tf.stack(self.mse_terms(th), axis=1),
                                                  jit_compile=True, reduce_retracing=True))
        M = self._chunked(fn, Theta)
        names = [t.name for t in self.trpl_terms] + ([self.sspl_term.name] if self.sspl_term else [])
        return {n: M[:, i] for i, n in enumerate(names)}

    # ------------------------------------------------------------------
    # predictions for plotting
    # ------------------------------------------------------------------
    def predict_trpl(self, theta, term: int = 0) -> dict:
        """
        NN prediction (numpy) on the NN QFLS axis and at the used points.

        ``qfls_axis`` / ``qfls_used`` are in the calibration of the data (nominal, high-injection formula).
        Axis-head networks (``surrogate.TrplSurrogateWindow``): ``log10_tau_axis`` is every row's curve on that
        fixed nominal axis (NaN beyond the end of the row's own curve, no filling); the physical picture is in
        ``qfls_axis_physical`` (B, 256) with ``log10_tau_own_axis`` and the data at ``qfls_used_physical`` (B, n),
        both on the model's q_max(theta).
        """
        import tensorflow as tf
        t = self.trpl_terms[term].surrogate
        th = tf.constant(np.atleast_2d(theta), tf.float64)
        out = {"qfls_axis": t.qfls_axis, "qfls_used": t.qfls_used, "log10_tau_used": t.log10_tau(th).numpy(),
               "log10_tau_obs": self.trpl_terms[term].y_obs}
        if hasattr(t, "nn_log10_tau_nominal"):
            out.update({"log10_tau_axis": t.nn_log10_tau_nominal(th), "qfls_axis_physical": t.qfls_axis_at(th),
                        "log10_tau_own_axis": t.nn_log10_tau(th).numpy(), "qfls_used_physical": t.data_qfls_at(th)})
        else:
            out["log10_tau_axis"] = t.nn_log10_tau(th).numpy()
        return out

    def predict_sspl(self, theta) -> dict:
        """ss-PL prediction (numpy) at the measured points and as full curves."""
        import tensorflow as tf
        obs = self.sspl_term.observation
        th = tf.constant(np.atleast_2d(theta), tf.float64)
        y, inside = obs.predict(th, return_inside=True)
        return {"x_exp": obs.x_exp, "y_obs": obs.y_obs, "y_model": y.numpy(),
                "inside": inside.numpy(), "curve": obs.model_curve(th)}

    def describe(self) -> dict:
        """Summary of the likelihood setup (exported with the results)."""
        d = {"mode": self.mode, "dim": self.dim, "sigmas": self.sigmas(),
             "trpl_points": [int(t.surrogate.n_points) for t in self.trpl_terms],
             "trpl_interp_space": [t.surrogate.interp_space for t in self.trpl_terms]}
        if self.sspl_term is not None:
            o = self.sspl_term.observation
            d.update({"sspl_mode": o.mode, "sspl_points": int(o.n_points),
                      "sspl_residual_unit": o.residual_unit, "sspl_G_per_sun": o.G_per_sun})
        return d
