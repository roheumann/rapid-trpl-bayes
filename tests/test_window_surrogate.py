"""Axis-head networks (surrogate.TrplSurrogateWindow, 16/10/2026): q_max + span of each curve's QFLS window.

A dummy two-output Keras model with known outputs is used:
  curve (scaled)  = ramp a + b j over the 256 points (exact under linear interpolation),
  axis  [d_top, span] in 10 meV units, optionally dependent on one input (gradient test).
"""
import json

import h5py
import numpy as np
import pytest
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from inference import surrogate as S
from inference.likelihood import LogPosterior, make_trpl_term
from inference.prior import Prior

VT = 0.02585199357397624
CONST = {"Nc": 2.21359e18, "Nv": 2.21359e18, "Vt": VT, "tmin": 1e-11, "tmax": 1e5, "N_TRAPS": 1.0,
         "QFLS_STEPS": 256.0, "PL_DECAY_MAGNITUDE": 12.0}
UNIT = 0.01
DEC = VT * np.log(10.0) / UNIT                       # one PL decade in axis units
NPULSE, EG = 1.6e17, 1.625
RAMP = np.linspace(0.2, 0.7, 256)                    # scaled curve -> log10 tau = -12 + 21 * RAMP


def _model(d_top_units, span_units, span_slope=0.0):
    import keras
    x = keras.Input((7,))
    curve = keras.layers.Dense(256, kernel_initializer="zeros",
                               bias_initializer=keras.initializers.Constant(RAMP), name="output")(x)
    k = np.zeros((7, 2), np.float32)
    k[3, 1] = span_slope                              # span depends on the scaled log10 N_t input
    axis = keras.layers.Dense(2, kernel_initializer=keras.initializers.Constant(k),
                              bias_initializer=keras.initializers.Constant([d_top_units, span_units]), name="axis")(x)
    return keras.Model(x, [keras.layers.Identity(name="curve")(curve), axis]), keras.Model(x, curve)


def _artifacts(model, meta=True):
    lb = np.array([12, 1.15, -12, 12, 0.02, -12, -12], float)
    ub = np.array([18, 2.4, -9, 18.3, 0.98, -4, -4], float)
    X = np.random.default_rng(0).uniform(lb, ub, size=(500, 7))
    si = StandardScaler().fit(X)
    so = MinMaxScaler().fit(np.array([[-12.0] * 256, [9.0] * 256]))
    m = {"variant": "axis_head", "axis": "predicted_window", "axis_unit_eV": UNIT} if meta else None
    names = ["log_n_pulse", "Eg_eV", "log_krad", "log_Ntrap_1", "DeltaEtrap_1", "log_taun_1", "log_taup_1"]
    return S.SurrogateArtifacts(model=model, input_scaler=si, output_scaler=so, constants=dict(CONST), lb=lb, ub=ub,
                                param_names=names, n_traps=1, meta=m)


def _prior(art):
    return Prior.from_artifacts(art, Eg=EG)


def _data_qfls(n=60, decades=9.9):
    q_an = S.nn_qfls_axis(NPULSE, EG, CONST)[-1]
    return q_an - np.linspace(0, decades, n)[::-1] * VT * np.log(10.0)


def test_equals_fixed_axis_surrogate_for_full_span():
    m2, m1 = _model(0.0, 12 * DEC)
    q = _data_qfls()
    win = S.make_trpl_surrogate(_artifacts(m2), NPULSE, EG, q)
    ref = S.TrplSurrogate(_artifacts(m1, meta=False), NPULSE, EG, q)
    assert isinstance(win, S.TrplSurrogateWindow)
    th = _prior(_artifacts(m1, meta=False)).sample(5, rng=np.random.default_rng(1))
    # float32 axis output: span 71.43 units to ~6e-8 relative -> ~1e-5 grid points -> < 1e-6 in log10 tau
    np.testing.assert_allclose(win.log10_tau(tf.constant(th)).numpy(), ref.log10_tau(tf.constant(th)).numpy(),
                               atol=1e-6)


def test_compressed_and_shifted_axis_read_at_the_right_pl_decade():
    span_dec, d_top = 9.0, 0.05
    m2, _ = _model(d_top / UNIT, span_dec * DEC)
    q = _data_qfls(decades=8.5)
    win = S.make_trpl_surrogate(_artifacts(m2), NPULSE, EG, q)
    th = _prior(_artifacts(m2)).sample(3, rng=np.random.default_rng(2))
    got = win.log10_tau(tf.constant(th)).numpy()
    u = (q - q[-1]) / (VT * np.log(10.0))
    j = 255 * (1 + u / span_dec)                       # q >> Vt: linear in u, independent of d_top
    expected = -12 + 21 * (RAMP[0] + (RAMP[1] - RAMP[0]) * j)
    np.testing.assert_allclose(got, np.broadcast_to(expected, got.shape), atol=1e-6)
    q_max, q_lo = win.window(th)
    np.testing.assert_allclose(q_max, q[-1] + d_top, atol=1e-6)
    np.testing.assert_allclose(q_max - q_lo, span_dec * VT * np.log(10.0), atol=1e-6)
    np.testing.assert_allclose(win.data_qfls_at(th)[0], q + d_top, atol=1e-6)    # data on the model's calibration


def test_uncovered_points_exclude_theta_and_gradient_is_finite():
    m2, _ = _model(0.0, 9.5 * DEC, span_slope=0.5 * DEC)   # span varies with N_t around 9.5 +- ~0.9 decades
    q = _data_qfls(decades=9.5)
    win = S.make_trpl_surrogate(_artifacts(m2), NPULSE, EG, q)
    art = _artifacts(m2)
    prior = _prior(art)
    lp = LogPosterior(prior, [make_trpl_term(win, q, 10 ** np.full(len(q), -7.0), sigma=0.05)], mode="trpl")
    th = prior.sample(400, rng=np.random.default_rng(3))
    cov = win.covered(th)
    ll = lp.log_lik(tf.constant(th)).numpy()
    assert cov.any() and (~cov).any()
    assert np.all(np.isneginf(ll[~cov])) and np.all(np.isfinite(ll[cov]))
    x = tf.Variable(th[cov][:20])
    with tf.GradientTape() as tape:
        f = tf.reduce_sum(lp.log_lik(x))
    g = tape.gradient(f, x).numpy()
    assert np.all(np.isfinite(g)) and np.abs(g[:, 1]).max() > 0                 # theta[1] = log10 N_t: via the span
    x = tf.Variable(th[~cov][:20])                                              # excluded theta: no NaN gradient
    with tf.GradientTape() as tape:
        f = tf.reduce_sum(lp.log_lik(x))                                        # = -inf
    assert not np.any(np.isnan(tape.gradient(f, x).numpy()))


def test_nominal_axis_has_nan_beyond_the_curve_end():
    m2, _ = _model(0.0, 9.0 * DEC)
    win = S.make_trpl_surrogate(_artifacts(m2), NPULSE, EG, _data_qfls(decades=8))
    v = win.nn_log10_tau_nominal(np.zeros((1, 5)) + _prior(_artifacts(m2)).sample(1, rng=np.random.default_rng(4)))
    u = (win.qfls_axis - win.qfls_axis[-1]) / (VT * np.log(10.0))
    assert np.all(np.isnan(v[0, u < -9.0 - 1e-6])) and np.all(np.isfinite(v[0, u > -9.0 + 1e-6]))


def test_save_load_and_dispatch(tmp_path):
    import joblib
    m2, _ = _model(0.3, 11.5 * DEC)
    art = _artifacts(m2)
    m2.save(tmp_path / "model.keras")
    joblib.dump(art.input_scaler, tmp_path / "x_param_scaler.joblib")
    joblib.dump(art.output_scaler, tmp_path / "x_output_scaler.joblib")
    with h5py.File(tmp_path / "x.hdf5", "w") as f:
        f.create_dataset("constants", data=np.array([tuple(CONST.values())],
                                                    dtype=[(k, "f8") for k in CONST])[0])
        f["lb"], f["ub"] = art.lb, art.ub
        f["param_names"] = np.array([n.encode() for n in art.param_names])
        X = np.random.default_rng(0).uniform(art.lb, art.ub, size=(50, 7))
        for s in ("train", "val", "test"):
            f[f"X_{s}_raw"] = X
    meta = {"variant": "axis_head", "axis": "predicted_window", "axis_unit_eV": UNIT,
            "param_scaler_path": str(tmp_path / "x_param_scaler.joblib"),
            "output_scaler_path": str(tmp_path / "x_output_scaler.joblib"), "training_hdf5": str(tmp_path / "x.hdf5")}
    (tmp_path / "surrogate_meta.json").write_text(json.dumps(meta))
    art2 = S.load_network(tmp_path)
    assert art2.axis == "predicted_window"
    q = _data_qfls()
    a = S.make_trpl_surrogate(art, NPULSE, EG, q)
    b = S.make_trpl_surrogate(art2, NPULSE, EG, q)
    assert isinstance(b, S.TrplSurrogateWindow)
    th = tf.constant(_prior(art).sample(4, rng=np.random.default_rng(5)))
    np.testing.assert_allclose(a.log10_tau(th).numpy(), b.log10_tau(th).numpy(), atol=1e-12)


def test_ode_nominal_axis_is_shifted_by_the_exact_top_offset():
    """Photodoping case (n_pulse 1e14, deep trap N_t 1e18): the ODE in the data calibration (nominal) is the physical
    curve shifted by d_top = kT ln((n0 + n)(p0 + n)/n_i^2) - 2 Vt ln(n/n_i) (exact for QFLS >> kT)."""
    from inference import ode
    m2, _ = _model(0.0, 12 * DEC)
    art = _artifacts(m2)
    prior = _prior(art)
    npulse = 1e14
    th = prior.from_physical(Et_eV=0.3, Nt=1e18, taun=1e-8, taup=1e-7, krad=1e-10)
    qn, ln_ = ode.simulate_trpl(th, prior, npulse, CONST, axis="nominal")
    qp, lp_ = ode.simulate_trpl(th, prior, npulse, CONST, axis="physical")
    mod = ode._ode_module()
    kT = 8.617e-5 * 300
    n0, p0, _ = mod.calcequil_general(EG, CONST["Nc"], CONST["Nv"], np.array([1e18]), np.array([0.3]), 300)
    ni = np.sqrt(CONST["Nc"] * CONST["Nv"] * np.exp(-EG / kT))
    d_top = kT * np.log((n0 + npulse) * (p0 + npulse) / ni ** 2) - 2 * VT * np.log(
        npulse / (np.sqrt(CONST["Nc"] * CONST["Nv"]) * np.exp(-EG / (2 * VT))))
    assert d_top > 0.05                                  # strongly low-injection case
    np.testing.assert_allclose(qp.max() - qn.max(), d_top, atol=2e-4)
    np.testing.assert_allclose(lp_[-1], ln_[-1], atol=1e-12)                    # same tau, other QFLS label
    hi = qp > qp.max() - 0.3                              # QFLS >> kT: constant shift
    np.testing.assert_allclose(qp[hi] - qn[-hi.sum():], d_top, atol=1e-3)
