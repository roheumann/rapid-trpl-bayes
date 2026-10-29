"""
Unit test 4 (REVISION_PLAN.md): prior and bijector.

* forward/inverse round trip
* log|det J| equals the log-determinant of the autodiff Jacobian
* Monte Carlo check of the ordering transform: the integral of |J(u)| over
  R^D equals the volume of the (ordered) support, i.e. the pushforward of the
  density |J| / volume is uniform on the support
* prior draws lie inside the training bounds and are ordered
* conditional transform used by the profile likelihood keeps theta_j fixed
"""
import math

import numpy as np
import pytest


def _prior(n_traps, photodoping=()):
    from inference.prior import Prior
    lo = np.array([-12] + [12, 0.5, -12, -12] * n_traps, float)
    hi = np.array([-10] + [20, 0.975, -4, -4] * n_traps, float)
    return Prior(lo, hi, n_traps, Eg=1.625, photodoping_traps=photodoping)


@pytest.mark.parametrize("n_traps,pd", [(1, ()), (2, ()), (3, ()), (2, (1,)), (2, (1, 2))])
def test_round_trip(n_traps, pd):
    import tensorflow as tf
    p = _prior(n_traps, pd)
    th = p.sample(500, rng=1)
    u = p.inverse(th)
    th2, _ = p.forward(tf.constant(u))
    assert np.allclose(th2.numpy(), th, atol=1e-8)
    u2 = np.random.default_rng(2).normal(scale=3, size=(500, p.dim))
    th3, _ = p.forward(tf.constant(u2))
    assert np.all(p.in_support(th3.numpy()))
    assert np.allclose(p.inverse(th3.numpy()), u2, atol=1e-6)


@pytest.mark.parametrize("n_traps,pd", [(1, ()), (2, ()), (2, (1, 2))])
def test_logdet_vs_autodiff(n_traps, pd):
    import tensorflow as tf
    p = _prior(n_traps, pd)
    u = tf.constant(np.random.default_rng(3).normal(scale=1.5, size=(64, p.dim)))
    with tf.GradientTape() as tape:
        tape.watch(u)
        th, logdet = p.forward(u)
    J = tape.batch_jacobian(th, u)
    _, ld_ad = np.linalg.slogdet(J.numpy())
    assert np.allclose(logdet.numpy(), ld_ad, atol=1e-8)


@pytest.mark.parametrize("n_traps,pd", [(2, ()), (3, ()), (2, (1,))])
def test_volume_monte_carlo(n_traps, pd):
    """E_q[|J(u)| / q(u)] = volume of the support (q: wide Student-t proposal)."""
    import tensorflow as tf
    from scipy import stats
    p = _prior(n_traps, pd)
    rng = np.random.default_rng(4)
    df, scale, N = 3.0, 3.0, 400_000
    u = stats.t.rvs(df, scale=scale, size=(N, p.dim), random_state=rng)
    log_q = stats.t.logpdf(u, df, scale=scale).sum(axis=1)
    _, logdet = p.forward(tf.constant(u))
    w = np.exp(logdet.numpy() - log_q)
    est = np.log(w.mean())
    se = w.std() / w.mean() / math.sqrt(N)
    assert abs(est - p.log_volume) < max(5 * se, 0.01)


def test_ordered_marginals_uniform():
    """Pushforward of p(u) ∝ |J| (sampled by SIR) is uniform on {a < e1 < e2 < b}."""
    import tensorflow as tf
    from scipy import stats
    p = _prior(2)
    rng = np.random.default_rng(5)
    u = stats.t.rvs(3, scale=3, size=(400_000, p.dim), random_state=rng)
    log_q = stats.t.logpdf(u, 3, scale=3).sum(axis=1)
    th, logdet = p.forward(tf.constant(u))
    w = np.exp(logdet.numpy() - log_q)
    idx = rng.choice(len(w), size=40_000, p=w / w.sum())
    e = th.numpy()[idx][:, p.idx_Et]
    a, b = p.lo[p.idx_Et[0]], p.hi[p.idx_Et[0]]
    # uniform on the triangle: e1 ~ 1 - ((b-e1)/(b-a))^2 CDF, e2 ~ ((e2-a)/(b-a))^2
    x1 = 1 - ((b - e[:, 0]) / (b - a)) ** 2
    x2 = ((e[:, 1] - a) / (b - a)) ** 2
    assert stats.kstest(x1, "uniform").pvalue > 1e-3
    assert stats.kstest(x2, "uniform").pvalue > 1e-3


def test_draws_inside_training_bounds(artifacts):
    from inference.prior import Prior
    p = Prior.from_artifacts(artifacts, Eg=1.625)
    th = p.sample(10_000, rng=6)
    assert np.all(th >= artifacts.lb[2:] - 1e-12) and np.all(th <= artifacts.ub[2:] + 1e-12)
    if p.n_traps > 1:
        assert np.all(np.diff(th[:, p.idx_Et], axis=1) > 0)
    s = p.sobol(256, seed=1)
    assert np.all(p.in_support(s))


def test_override_cannot_widen(artifacts):
    from inference.prior import Prior
    with pytest.raises(ValueError):
        Prior.from_artifacts(artifacts, Eg=1.625, bounds_override={"log10_krad": (-13, -10)})


@pytest.mark.parametrize("which", ["log10_Nt_2", "Et_rel_1", "Et_rel_2"])
def test_conditional_forward(which):
    import tensorflow as tf
    p = _prior(2)
    j = p.names.index(which)
    vals = np.linspace(p.lo[j] + 0.1 * (p.hi[j] - p.lo[j]), p.hi[j] - 0.1 * (p.hi[j] - p.lo[j]), 50)
    u = np.random.default_rng(7).normal(scale=2, size=(50, p.dim))
    th, _ = p.forward(tf.constant(u), fixed=(j, tf.constant(vals)))
    th = th.numpy()
    assert np.allclose(th[:, j], vals)
    assert np.all(p.in_support(th))
