"""
Marginalised ss-PL offset (likelihood.SsplTerm.offset_sigma, refine.OdeFitProblem.offset_sigma; 24/10/2026).

-log L_ss = (MSE - w m^2)/(2 sigma^2) + const with w = sigma_d^2/(sigma_d^2 + sigma^2) must equal the integral over the
offset delta ~ N(0, sigma_d^2) of the per-point likelihood (noise s = sigma sqrt(N)) up to a theta-independent constant,
and the least-squares form of the ODE refinement must give the same value.
"""
import math
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.integrate import quad

tf = pytest.importorskip("tensorflow")

from inference import likelihood, refine


class _Obs:
    """Stand-in for sspl_tf.SsplObservation: the model prediction is the first N columns of theta."""
    residual_unit = "eV"

    def __init__(self, y):
        self.y_obs = np.asarray(y, np.float64)

    def predict(self, theta):
        return tf.convert_to_tensor(theta, tf.float64)[:, :len(self.y_obs)]


def _log_post(y, sigma, offset_sigma):
    term = likelihood.SsplTerm(_Obs(y), sigma=sigma, offset_sigma=offset_sigma)
    return likelihood.LogPosterior(SimpleNamespace(dim=len(y)), [], term, "sspl")


def _brute_log_marginal(r, sigma, sd):
    """log of int prod_i N(r_i - d; 0, s) N(d; 0, sd) dd with s = sigma sqrt(N) (shifted for stability)."""
    n = len(r)
    s = sigma * math.sqrt(n)
    f = lambda d: np.sum(-0.5 * ((r - d) / s) ** 2) - 0.5 * (d / sd) ** 2
    d0 = np.sum(r) / (n + s ** 2 / sd ** 2)
    val, _ = quad(lambda d: math.exp(f(d) - f(d0)), d0 - 12 * sd, d0 + 12 * sd, points=[d0], limit=400)
    return math.log(val) + f(d0)


@pytest.mark.parametrize("sd", [0.005, 0.02, 0.1])
def test_marginal_matches_integration(sd):
    rng = np.random.default_rng(0)
    y = rng.normal(1.2, 0.05, 15)
    sigma = 0.0029
    lp = _log_post(y, sigma, sd)
    vals_tf, vals_brute = [], []
    for k in range(4):                                  # different residual vectors: compare DIFFERENCES (constant drops)
        pred = y + rng.normal(0.0, 0.01, 15) + rng.normal(0.0, 0.03)
        vals_tf.append(float(lp.np_log_lik_theta(pred[None])[0]))
        vals_brute.append(_brute_log_marginal(pred - y, sigma, sd))
    d_tf, d_brute = np.diff(vals_tf), np.diff(vals_brute)
    np.testing.assert_allclose(d_tf, d_brute, rtol=1e-6, atol=1e-6)
    # delta_hat = w m
    w = sd ** 2 / (sd ** 2 + sigma ** 2)
    np.testing.assert_allclose(lp.sspl_offset(pred[None])[0], w * np.mean(pred - y), rtol=1e-10)


def test_no_offset_is_the_old_likelihood():
    rng = np.random.default_rng(1)
    y = rng.normal(1.2, 0.05, 10)
    pred = y + rng.normal(0.0, 0.01, 10)
    sigma = 0.002
    ll = float(_log_post(y, sigma, None).np_log_lik_theta(pred[None])[0])
    old = -np.mean((pred - y) ** 2) / (2 * sigma ** 2) - math.log(sigma) - 0.5 * math.log(2 * math.pi)
    assert abs(ll - old) < 1e-10
    assert _log_post(y, sigma, None).sspl_offset(pred[None])[0] == 0.0


def test_least_squares_form_equals_tf_marginal():
    """refine.OdeFitProblem (ss-PL residuals shifted by delta_hat + pseudo-observation) == LogPosterior marginal."""
    rng = np.random.default_rng(2)
    y = rng.normal(1.2, 0.05, 15)
    sigma, sd = 0.002, 0.02
    pred = y + rng.normal(0.0, 0.005, 15) - 0.04
    prob = refine.OdeFitProblem(prior=None, npulse=1.0, constants={}, T=300.0, qfls=np.zeros(0), y_trpl=np.zeros(0),
                                sigma_trpl=1.0, sspl=SimpleNamespace(predict=lambda th: np.atleast_2d(pred)),
                                y_sspl=y, sigma_sspl=sigma, offset_sigma=sd)
    d = prob.sspl_offset(pred)
    vec = np.concatenate([pred - d, [d]])
    nll_ls = prob._nll_from_pred(vec) - (math.log(1.0) + 0.5 * math.log(2 * math.pi))   # drop the (empty) tr-PL constant
    nll_tf = -float(_log_post(y, sigma, sd).np_log_lik_theta(pred[None])[0])
    assert abs(nll_ls - nll_tf) < 1e-8
