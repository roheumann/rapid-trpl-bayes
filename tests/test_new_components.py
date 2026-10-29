"""
Tests for the components added in the second revision round:
grid posterior, PSIS, numpy prior map + fixed parameters, numpy ss-PL twin,
and the ODE refinement with curve_fit.
"""
import math

import numpy as np
import pytest

from conftest import exp_sspl_data, exp_trpl_data


def _prior(n_traps, fixed=None):
    from inference.prior import Prior, FIXED_HALF_WIDTH
    lo = np.array([-12] + [12, 0.5, -12, -12] * n_traps, float)
    hi = np.array([-10] + [20, 0.975, -4, -4] * n_traps, float)
    for i, v in (fixed or {}).items():
        lo[i], hi[i] = v - FIXED_HALF_WIDTH, v + FIXED_HALF_WIDTH
    return Prior(lo, hi, n_traps, Eg=1.625)


# ─────────────────────────────────────────────────────────────────────────────
def test_forward_np_and_fixed():
    import tensorflow as tf
    p = _prior(2, fixed={0: -11.0})
    assert p.fixed_mask[0] and not p.fixed_mask[1:].any() and p.free_idx == list(range(1, 9))
    u = np.random.default_rng(1).normal(scale=3, size=(500, p.dim))
    th_tf = p.forward(tf.constant(u))[0].numpy()
    assert np.abs(p.forward_np(u) - th_tf).max() < 1e-12
    assert np.allclose(th_tf[:, 0], -11.0, atol=2e-9)


# ─────────────────────────────────────────────────────────────────────────────
class _ToyPosterior:
    """Gaussian log-likelihood in theta (fast, exact reference for the grid)."""

    def __init__(self, prior, mu, sd):
        self.prior, self.mu, self.sd = prior, np.asarray(mu), np.asarray(sd)
        from inference.benchmark import PyCounter
        self.counter = PyCounter()

    def np_log_lik_theta(self, th):
        self.counter.record(len(th))
        return -0.5 * np.sum(((th - self.mu) / self.sd) ** 2, axis=1)


def test_grid_matches_brute_force():
    from inference import grid
    p = _prior(2)
    mu = 0.5 * (p.lo + p.hi)
    mu[2], mu[6] = 0.6, 0.8                                # ordered energies
    sd = 0.3 * (p.hi - p.lo)
    toy = _ToyPosterior(p, mu, sd)
    axes = grid.grid_axes(p, mu, n_points=4, mode="prior")
    res = grid.grid_posterior(toy, axes, chunk=1000, verbose=False)
    # brute force
    mesh = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, p.dim)
    ok = p.in_support(mesh)
    ll = np.full(len(mesh), -np.inf)
    ll[ok] = toy.np_log_lik_theta(mesh[ok])
    w = np.exp(ll - ll.max())
    w /= w.sum()
    idx = np.stack(np.unravel_index(np.arange(len(mesh)), [len(a) for a in axes]), 1)
    for d in range(p.dim):
        m = np.bincount(idx[:, d], weights=w, minlength=len(axes[d]))
        assert np.allclose(res.marg1[d], m, atol=1e-12)
    m2 = np.bincount(idx[:, 2] * 4 + idx[:, 6], weights=w, minlength=16).reshape(4, 4)
    assert np.allclose(res.marg2[(2, 6)], m2, atol=1e-12)
    assert res.n_evaluated == ok.sum()
    assert abs(res.log_z - (ll.max() + math.log(np.exp(ll[ok] - ll.max()).sum()))) < 1e-10


def test_grid_quantiles_uniform():
    from inference.grid import grid_quantiles
    v = np.linspace(0, 9, 10)
    q = grid_quantiles(v, np.full(10, 0.1), [0.05, 0.5, 0.95])
    assert np.allclose(q, [0.0, 4.5, 9.0], atol=1e-12)


# ─────────────────────────────────────────────────────────────────────────────
def test_psis_reduces_to_normalised_weights_for_light_tails():
    from inference.validate import psis
    rng = np.random.default_rng(3)
    lw = 0.1 * rng.standard_normal(4000)                  # nearly uniform weights
    lws, k = psis(lw)
    assert k < 0.5
    assert abs(np.exp(lws).sum() - 1) < 1e-10
    lw_heavy = 3.0 * rng.standard_cauchy(4000)            # very heavy tail
    assert psis(lw_heavy)[1] > 0.7


# ─────────────────────────────────────────────────────────────────────────────
def test_numpy_sspl_twin_matches_tf():
    import tensorflow as tf
    from inference import sspl_tf as SS
    for K, mode in ((1, "plqy_vs_qfls"), (2, "qfls_vs_suns")):
        p = _prior(K)
        th = p.sample(30, rng=4)
        x = np.linspace(1.0, 1.3, 7) if mode == "plqy_vs_qfls" else np.logspace(-2, 1, 7)
        obs = SS.SsplObservation(SS.SsplModel(1.625, K), mode, x, np.full(7, 0.5 if mode.startswith("plqy") else 1.2),
                                 G_per_sun=2.7e21)
        assert np.abs(obs.numpy_spec().predict(th) - obs.predict(tf.constant(th)).numpy()).max() < 1e-10


# ─────────────────────────────────────────────────────────────────────────────
def test_ode_curve_fit_recovers_known_parameters():
    """Noise-free ODE data from a known theta; curve_fit from a perturbed start returns close to it."""
    from inference import ode, refine
    p = _prior(1)
    constants = {"Nc": 2.21359e18, "Nv": 2.21359e18, "tmin": 1e-11, "tmax": 1e5, "PL_DECAY_MAGNITUDE": 12.0}
    theta_true = np.array([-11.0, 16.9, 0.923, -8.46, -7.36])
    q_all, lt_all = ode.simulate_trpl(theta_true, p, 1.6e17, constants)
    q = np.linspace(max(q_all.min(), 0.97), min(q_all.max(), 1.485), 40)
    y = ode.ode_log10_tau(theta_true, p, 1.6e17, constants, q)
    prob = refine.OdeFitProblem(prior=p, npulse=1.6e17, constants=constants, T=300.0, qfls=q, y_trpl=y,
                                sigma_trpl=0.03)
    start = theta_true + np.array([0.3, -0.3, 0.004, 0.15, -0.15])
    r = refine.ode_curve_fit(prob, start, max_nfev=60)
    assert r.neg_log_lik < r.start_neg_log_lik
    mse_end = 2 * (r.neg_log_lik - (math.log(0.03) + 0.5 * math.log(2 * math.pi))) * 0.03 ** 2
    assert mse_end < 1e-4
    assert r.history_best is not None and np.all(np.diff(r.history_best) <= 1e-12)
