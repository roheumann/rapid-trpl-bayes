"""
multi.py — tr-PL likelihood for MANY data sets in one batch (SBC, coverage)
==========================================================================

For the validation studies (tasks T5 and T6) hundreds of synthetic decays are
analysed at once.  Every row r of a batch belongs to data set ``idx[r]``:

    log L_r = -MSE(y_model(theta_r), y_obs[idx_r]) / (2 sigma[idx_r]^2) - log(sigma sqrt(2 pi))

* All data sets share the network inputs n_pulse and Eg (one NN QFLS axis).
* Each data set has its own measured QFLS points; the linear interpolation
  from the NN axis is stored as index/weight pairs (J, w) per data set and
  applied with a batched gather (memory O(rows x points), no dense matrices).
* Data sets with fewer points are padded and masked.
* sigma per data set is a tf.Variable (sigma_eff can be set per data set).

:class:`BoundPosterior` fixes the row -> data-set assignment so that the
object can be used by ``samplers.NutsSampler`` (with ``groups`` = the same
assignment) like an ordinary LogPosterior.
"""
from __future__ import annotations

import math

import numpy as np

from .benchmark import EvalCounter

LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)


class MultiTrplLikelihood:
    """
    Parameters
    ----------
    surrogate : surrogate.TrplSurrogate (any QFLS points; only its NN, scalers
        and QFLS axis are used)
    prior : prior.Prior
    datasets : list of (qfls_points, log10_tau_obs) — points outside the NN
        QFLS axis are dropped
    sigma : float or (n_datasets,) array
    """

    def __init__(self, surrogate, prior, datasets, sigma, batch_size: int = 8192):
        import tensorflow as tf
        self.sur = surrogate
        self.prior = prior
        self.dim = prior.dim
        self.batch_size = int(batch_size)
        axis = surrogate.qfls_axis
        n_sets = len(datasets)
        cleaned = []
        for q, y in datasets:
            q, y = np.asarray(q, np.float64), np.asarray(y, np.float64)
            ok = np.isfinite(q) & np.isfinite(y) & (q >= axis[0]) & (q <= axis[-1])
            cleaned.append((q[ok], y[ok]))
        self.n_points = np.array([len(q) for q, _ in cleaned])
        if np.any(self.n_points < 2):
            raise ValueError("every data set needs at least 2 points on the NN QFLS axis")
        n_max = int(self.n_points.max())
        J = np.zeros((n_sets, n_max), np.int32)
        W = np.zeros((n_sets, n_max))
        Y = np.zeros((n_sets, n_max))
        Mk = np.zeros((n_sets, n_max))
        # axis-head networks (surrogate.TrplSurrogateWindow): theta-dependent axis -> PL decades u of the points
        self.window = hasattr(surrogate, "values_at_u")
        U = np.zeros((n_sets, n_max))
        for d, (q, y) in enumerate(cleaned):
            j = np.clip(np.searchsorted(axis, q, side="right") - 1, 0, len(axis) - 2)
            J[d, :len(q)] = j
            W[d, :len(q)] = (q - axis[j]) / (axis[j + 1] - axis[j])
            Y[d, :len(q)] = y
            Mk[d, :len(q)] = 1.0
            if self.window:                                        # axis = nominal axis, axis[-1] = 2 Vt ln(n/n_i)
                U[d, :len(q)] = np.minimum((q - axis[-1]) / (surrogate.Vt * np.log(10.0)), 0.0)
        self._U = tf.constant(U)                                   # padding: u = 0 (always covered, masked)
        self._J = tf.constant(J)
        self._W = tf.constant(W)
        self._Y = tf.constant(Y)
        self._M = tf.constant(Mk)
        self._n = tf.constant(self.n_points.astype(np.float64))
        sig = np.broadcast_to(np.asarray(sigma, np.float64), (n_sets,)).copy()
        self.sigma = tf.Variable(sig, dtype=tf.float64, trainable=False)
        self.n_sets = n_sets
        self.datasets = cleaned
        self.counter = EvalCounter("multi_log_lik")
        self._compiled = {}

    # ------------------------------------------------------------------
    def set_sigma(self, sigma) -> None:
        self.sigma.assign(np.broadcast_to(np.asarray(sigma, np.float64), (self.n_sets,)))

    def mse(self, theta, idx):
        """MSE of every row against its data set, (R,)."""
        import tensorflow as tf
        if self.window:                                           # +inf where a curve ends above a point
            y_model = self.sur.values_at_u(theta, tf.gather(self._U, idx))[0]
        else:
            y_nn = self.sur.nn_log10_tau(theta)                   # (R, 256)
            J = tf.gather(self._J, idx)                           # (R, n_max)
            W = tf.gather(self._W, idx)
            y0 = tf.gather(y_nn, J, batch_dims=1)
            y1 = tf.gather(y_nn, J + 1, batch_dims=1)
            y_model = y0 + W * (y1 - y0)
        r = (y_model - tf.gather(self._Y, idx)) * tf.gather(self._M, idx)
        return tf.reduce_sum(r * r, axis=1) / tf.gather(self._n, idx)

    def log_lik(self, theta, idx):
        import tensorflow as tf
        theta = tf.convert_to_tensor(theta, tf.float64)
        self.counter.record(tf.shape(theta)[0])
        s = tf.gather(self.sigma, idx)
        ll = -self.mse(theta, idx) / (2.0 * s * s) - tf.math.log(s) - LOG_SQRT_2PI
        return tf.where(tf.math.is_finite(ll), ll, tf.constant(-np.inf, tf.float64))

    def log_prob_u(self, u, idx):
        theta, logdet = self.prior.forward(u)
        return self.log_lik(theta, idx) - self.prior.log_volume + logdet

    def neg_log_lik_u(self, u, idx):
        theta, _ = self.prior.forward(u)
        return -self.log_lik(theta, idx)

    # ------------------------------------------------------------------
    def _fn(self, name, f):
        import tensorflow as tf
        if name not in self._compiled:
            self._compiled[name] = tf.function(f, jit_compile=True, reduce_retracing=True)
        return self._compiled[name]

    def _chunked(self, fn, X, idx):
        """Evaluate fn(X, idx) in power-of-two padded chunks (see LogPosterior._chunked)."""
        import tensorflow as tf
        X = np.atleast_2d(np.asarray(X, np.float64))
        idx = np.asarray(idx, np.int32)
        out = []
        for s in range(0, len(X), self.batch_size):
            xb, ib = X[s:s + self.batch_size], idx[s:s + self.batch_size]
            b = len(xb)
            bucket = int(min(self.batch_size, max(16, 2 ** int(np.ceil(np.log2(max(b, 1)))))))
            pad = bucket - b
            if pad:
                xb = np.concatenate([xb, np.repeat(xb[:1], pad, axis=0)])
                ib = np.concatenate([ib, np.repeat(ib[:1], pad)])
            r = fn(tf.constant(xb), tf.constant(ib)).numpy()[:b]
            if pad:
                self.counter.n_points.assign_add(-pad)
            out.append(r)
        return np.concatenate(out)

    def np_neg_log_lik_u(self, U, idx) -> np.ndarray:
        f = self._chunked(self._fn("nll", self.neg_log_lik_u), U, idx)
        return np.where(np.isfinite(f), f, 1e30)

    def np_mse_theta(self, Theta, idx) -> np.ndarray:
        return self._chunked(self._fn("mse", self.mse), Theta, idx)

    def bind(self, idx) -> "BoundPosterior":
        return BoundPosterior(self, idx)


class BoundPosterior:
    """A MultiTrplLikelihood with a fixed row -> data-set assignment (for NUTS)."""

    def __init__(self, multi: MultiTrplLikelihood, idx):
        import tensorflow as tf
        self.multi = multi
        self.prior = multi.prior
        self.dim = multi.dim
        self.counter = multi.counter
        self.idx = np.asarray(idx, np.int32)
        self._idx = tf.constant(self.idx)

    def log_prob_u(self, u):
        return self.multi.log_prob_u(u, self._idx)


class MultiObjective:
    """
    Adapter for ``optimize.cmaes_multistart``: start s belongs to data set
    ``set_of_start[s]``; the CMA-ES populations are evaluated with the right
    data set per row.
    """

    def __init__(self, multi: MultiTrplLikelihood, set_of_start):
        self.multi = multi
        self.prior = multi.prior
        self.dim = multi.dim
        self.counter = multi.counter
        self.set_of_start = np.asarray(set_of_start, int)

    def np_neg_log_lik_u_rows(self, X, start_ids):
        return self.multi.np_neg_log_lik_u(X, self.set_of_start[np.asarray(start_ids)])
