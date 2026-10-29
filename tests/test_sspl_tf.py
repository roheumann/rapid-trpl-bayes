"""
Unit test 2 (REVISION_PLAN.md): the TensorFlow ss-PL model matches the numpy
solvers of the original grid pipeline (tests/legacy_reference.py) evaluated ROW BY ROW (which
also covers bug 1.1), and its gradients match finite differences.
"""
import numpy as np
import pytest

EG, NC, NV, T = 1.625, 2.21359e18, 2.21359e18, 300.0
KT = 8.617e-5 * T          # legacy value of kT


def _prior(n_traps):
    from inference.prior import Prior
    lo = np.array([-12] + [12, 0.5, -12, -12] * n_traps, float)
    hi = np.array([-10] + [20, 0.975, -4, -4] * n_traps, float)
    return Prior(lo, hi, n_traps, Eg=EG)


def _legacy_rowwise(m, theta, n_traps, n_sw):
    """Legacy numeric (fixed-point) solver, one parameter set at a time."""
    ni2 = m._ni2(NC, NV, EG, KT)
    out = []
    for th in theta:
        tp = [dict(Et=np.array([th[2 + 4 * k] * EG]), Nt=np.array([10 ** th[1 + 4 * k]]),
                   taun=np.array([10 ** th[3 + 4 * k]]), taup=np.array([10 ** th[4 + 4 * k]]))
              for k in range(n_traps)]
        q, plqy, G = m._sspl_numeric(tp, np.array([10 ** th[0]]), n_sw, NC, NV, EG, KT, ni2,
                                     max_iter=5000, rtol=1e-13)
        out.append((q[0], plqy[0], G[0]))
    return [np.array(x) for x in zip(*out)]


def _neutrality_residual(th, n_traps, n, p):
    """Relative residual |p - n - sum n_t(p)| / p of the charge-neutrality equation."""
    g = p - n
    for k in range(n_traps):
        Nt, Et = 10 ** th[1 + 4 * k], th[2 + 4 * k] * EG
        tn, tp = 10 ** th[3 + 4 * k], 10 ** th[4 + 4 * k]
        n1, p1 = NC * np.exp((Et - EG) / KT), NV * np.exp(-Et / KT)
        g = g - Nt * (tp * n + tn * p1) / (tp * (n + n1) + tn * (p + p1))
    return np.abs(g) / p


def _brentq_reference(th, n_traps, n_sw):
    """p(n) from scipy's Brent root finder (reference solution of the neutrality equation)."""
    from scipy.optimize import brentq
    P = np.zeros((len(th), len(n_sw)))
    for r, t in enumerate(th):
        Nt_sum = sum(10 ** t[1 + 4 * k] for k in range(n_traps))
        for c, n in enumerate(n_sw):
            # root of g(p) = p - n - sum n_t(p) in log p (g is strictly increasing)
            P[r, c] = np.exp(brentq(lambda lp: np.exp(lp) - n - _nt_sum(t, n_traps, n, np.exp(lp)),
                                    np.log(n), np.log(n + Nt_sum), xtol=1e-14, rtol=1e-15))
    return P


def _nt_sum(t, n_traps, n, p):
    s = 0.0
    for k in range(n_traps):
        Nt, Et = 10 ** t[1 + 4 * k], t[2 + 4 * k] * EG
        tn, tp = 10 ** t[3 + 4 * k], 10 ** t[4 + 4 * k]
        n1, p1 = NC * np.exp((Et - EG) / KT), NV * np.exp(-Et / KT)
        s += Nt * (tp * n + tn * p1) / (tp * (n + n1) + tn * (p + p1))
    return s


@pytest.mark.parametrize("n_traps,solver", [(1, "analytic"), (1, "root"), (2, "root"), (3, "root")])
def test_matches_legacy_rowwise(legacy_03, n_traps, solver):
    """
    TF model vs (a) a Brent root-finding reference everywhere and (b) the legacy
    03 solver evaluated row by row, wherever the legacy fixed-point iteration
    has actually converged.  (For deep traps with N_t >> n the legacy
    iteration p <- n + sum n_t(p) oscillates and does not converge; there the
    TF solution is the correct one.)
    """
    import tensorflow as tf
    from inference.sspl_tf import SsplModel
    p = _prior(n_traps)
    th = p.sample(40, rng=11)
    model = SsplModel(EG, n_traps, NC, NV, kT=KT, n_sweep=(1e8, NV, 80), solver=solver)
    s = model.solve(tf.constant(th))
    p_tf = s["p"].numpy()
    # (a) reference root
    p_ref = _brentq_reference(th, n_traps, model.n_sweep)
    assert np.max(np.abs(np.log(p_tf / p_ref))) < 1e-9
    # (b) legacy, where converged
    q_l, plqy_l, G_l = _legacy_rowwise(legacy_03, th, n_traps, model.n_sweep)
    p_l = np.exp(q_l / KT) * model.ni2 / model.n_sweep[None, :]
    conv = np.array([_neutrality_residual(th[r], n_traps, model.n_sweep, p_l[r])
                     for r in range(len(th))]) < 1e-9
    assert conv.mean() > 0.5
    assert np.allclose(s["qfls"].numpy()[conv], q_l[conv], rtol=0, atol=1e-7)
    assert np.allclose(np.log10(s["plqy"].numpy()[conv]), np.log10(plqy_l[conv]), atol=1e-6)
    assert np.allclose(np.log10(s["G"].numpy()[conv]), np.log10(G_l[conv]), atol=1e-6)


def test_analytic_vs_legacy_analytic(legacy_03):
    """1 trap: the stable closed form equals the legacy quadratic where the latter is accurate."""
    import tensorflow as tf
    from inference.sspl_tf import SsplModel
    p = _prior(1)
    th = p.sample(200, rng=12)
    model = SsplModel(EG, 1, NC, NV, kT=KT, n_sweep=(1e8, NV, 80), solver="analytic")
    s = model.solve(tf.constant(th))
    m = legacy_03
    q, plqy, _ = m._sspl_analytic(th[:, 2] * EG, 10 ** th[:, 1], 10 ** th[:, 3], 10 ** th[:, 4],
                                  10 ** th[:, 0], model.n_sweep, NC, NV, EG, KT, m._ni2(NC, NV, EG, KT))
    # the legacy (-b + sqrt(b^2 - 4ac)) / 2a form loses precision when n_t << n;
    # compare where both give a trapped density above 1e-6 of n
    assert np.median(np.abs(s["qfls"].numpy() - q)) < 1e-9
    assert np.percentile(np.abs(s["qfls"].numpy() - q), 99) < 1e-6


def test_observation_interp_rowwise():
    """Per-row interpolation in SsplObservation equals np.interp row by row."""
    import tensorflow as tf
    from inference.sspl_tf import SsplModel, SsplObservation
    p = _prior(2)
    th = p.sample(30, rng=13)
    model = SsplModel(EG, 2, NC, NV, n_sweep=(1e8, NV, 120))
    suns = np.logspace(-2, 0.5, 9)
    obs = SsplObservation(model, "qfls_vs_suns", suns, np.full(9, 1.2), G_per_sun=2.74e21)
    y, inside = obs.predict(tf.constant(th), return_inside=True)
    s = model.solve(tf.constant(th))
    lg = np.log10(s["G"].numpy())
    q = s["qfls"].numpy()
    xq = np.log10(suns * 2.74e21)
    for i in range(len(th)):
        ref = np.interp(xq, lg[i], q[i])
        ok = inside.numpy()[i]
        assert np.allclose(y.numpy()[i][ok], ref[ok], atol=1e-10)


@pytest.mark.parametrize("n_traps", [1, 2])
def test_gradient_finite_differences(n_traps):
    """Implicit-function gradient of the root solver vs central differences (float64)."""
    import tensorflow as tf
    from inference.sspl_tf import SsplModel, SsplObservation
    p = _prior(n_traps)
    th = p.sample(6, rng=14)
    model = SsplModel(EG, n_traps, NC, NV, n_sweep=(1e8, NV, 120), solver="root")
    suns = np.logspace(-1.5, 0.3, 8)
    obs = SsplObservation(model, "qfls_vs_suns", suns, np.full(8, 1.2), G_per_sun=2.74e21)

    def f(x):
        return tf.reduce_sum(obs.predict(x), axis=1)

    x = tf.constant(th)
    with tf.GradientTape() as tape:
        tape.watch(x)
        y = f(x)
    g = tape.gradient(y, x).numpy()
    h = 1e-5
    g_fd = np.zeros_like(th)
    for d in range(p.dim):
        e = np.zeros(p.dim)
        e[d] = h
        g_fd[:, d] = (f(tf.constant(th + e)).numpy() - f(tf.constant(th - e)).numpy()) / (2 * h)
    scale = np.abs(g_fd).max(axis=1, keepdims=True) + 1e-12
    assert np.max(np.abs(g - g_fd) / scale) < 1e-4
