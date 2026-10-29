"""LogPosterior.set_data, the Voc,rad-based 1-sun generation and SBC with tempered SMC (16/10/2026)."""
import math

import numpy as np
import pytest
import tensorflow as tf

from inference import data
from inference.likelihood import LogPosterior, make_trpl_term
from inference.prior import Prior
from inference.sbc import run_sbc_smc, uniformity_table


class _LinearSurrogate:
    """log10 tau = A theta + b at n points (a stand-in for TrplSurrogate with the same interface)."""

    def __init__(self, A, b):
        self.A = tf.constant(A, tf.float64)
        self.b = tf.constant(b, tf.float64)
        self.n_points = A.shape[0]
        self.mask_used = np.ones(self.n_points, bool)
        self.qfls_used = np.linspace(1.0, 1.4, self.n_points)
        self.interp_space = "log10"

    def log10_tau(self, theta):
        return tf.matmul(tf.convert_to_tensor(theta, tf.float64), self.A, transpose_b=True) + self.b


def _toy(n_points=12, sigma=0.3, seed=0):
    rng = np.random.default_rng(seed)
    prior = Prior(lo=np.array([-11.0, 12.0, 0.5, -10.0, -10.0]), hi=np.array([-9.0, 18.0, 0.95, -6.0, -6.0]),
                  n_traps=1, Eg=1.6)
    A = rng.normal(size=(n_points, 5)) * np.array([1.0, 0.5, 8.0, 1.0, 1.0])
    sur = _LinearSurrogate(A, rng.normal(size=n_points) - 7.0)
    th0 = prior.sample(1, rng=rng)
    y = sur.log10_tau(tf.constant(th0)).numpy()[0]
    lp = LogPosterior(prior, [make_trpl_term(sur, sur.qfls_used, 10 ** y, sigma=sigma)], mode="trpl")
    return prior, sur, lp


def test_set_data_equals_fresh_log_posterior():
    prior, sur, lp = _toy()
    rng = np.random.default_rng(1)
    th = prior.sample(16, rng=rng)
    y_new = sur.log10_tau(tf.constant(prior.sample(1, rng=rng))).numpy()[0] + 0.1
    f = tf.function(lp.log_lik, jit_compile=True)
    f(tf.constant(th))                                                     # compiled with the old data
    lp.set_data(y_trpl=y_new)
    fresh = LogPosterior(prior, [make_trpl_term(sur, sur.qfls_used, 10 ** y_new, sigma=0.3)], mode="trpl")
    np.testing.assert_allclose(f(tf.constant(th)).numpy(), fresh.log_lik(tf.constant(th)).numpy(), rtol=1e-12)
    np.testing.assert_allclose(lp.trpl_terms[0].y_obs, y_new)


def test_g_per_sun_from_voc_rad_round_trip():
    from inference.sspl_tf import KB_EV
    Nc = Nv = 2.21359e18
    Eg, krad, voc = 1.625, 4.7930e-11, 1.333
    G = data.g_per_sun_from_voc_rad(voc, krad, Eg, Nc, Nv, 300.0)
    kT = KB_EV * 300.0
    ni2 = Nc * Nv * math.exp(-Eg / kT)
    assert kT * math.log1p(G / (krad * ni2)) == pytest.approx(voc, abs=1e-12)
    assert G == pytest.approx(2.920e21, rel=2e-3)
    cfg = {"physics": {"Eg_eV": Eg, "T_K": 300.0, "thickness_nm": 450.0},
           "sspl": {"absorptance": 0.8, "voc_rad_V": voc}, "prior": {"fixed": {"log10_krad": math.log10(krad)}}}
    g2, info = data.sspl_g_per_sun(cfg, {"Nc": Nc, "Nv": Nv})
    assert g2 == pytest.approx(G) and info["voc_rad_model_V"] == pytest.approx(voc)
    assert info["voc_rad_model_absorptance_V"] == pytest.approx(1.3314, abs=5e-4)       # the 1.6 mV check
    cfg["sspl"].pop("voc_rad_V")
    g3, info3 = data.sspl_g_per_sun(cfg, {"Nc": Nc, "Nv": Nv})
    assert info3["source"] == "absorptance" and g3 == pytest.approx(info3["G_absorptance"])


def test_sbc_with_smc_gives_uniform_ranks_on_a_linear_model():
    prior, sur, lp = _toy()
    y_before = lp.trpl_terms[0].y_obs.copy()
    res = run_sbc_smc(lp, prior, n_datasets=40, n_keep=19,
                      smc_kwargs=dict(n_particles=512, n_islands=2, ess_target=0.5, target_accept=0.25,
                                      p_move=0.95, min_mh=2, max_mh=20),
                      seed=3, max_batch=1024, verbose=False)
    np.testing.assert_allclose(lp.trpl_terms[0].y_obs, y_before)          # data restored
    uni = uniformity_table(res, n_bins=5)
    free = [prior.display_keys[i] for i in prior.free_idx]
    assert uni.loc[free, "p_value"].min() > 1e-3
    assert np.all(np.isfinite(res.ranks[:, prior.free_idx]))
    assert len(res.diagnostics) == 40 and res.info["sampler"] == "tempered SMC"
