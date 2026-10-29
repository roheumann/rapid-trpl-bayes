"""Tests of inference/cylinder.py on toy problems with known answers (no network, no TF)."""
import math

import numpy as np
import pytest

from inference import cylinder as cy


class BoxPrior:
    """Minimal stand-in for prior.Prior: uniform box, everything free."""

    def __init__(self, lo, hi, names, Eg=1.6, kT=0.025852, Nc=2.2e18):
        self.lo, self.hi, self.names = np.asarray(lo, float), np.asarray(hi, float), names
        self.free_idx = list(range(len(lo)))
        self.Eg, self.kT, self.Nc = Eg, kT, Nc

    def in_support(self, theta, tol=0.0):
        return np.all((theta >= self.lo - tol) & (theta <= self.hi + tol), axis=1)


def _toy(D=4, seed=0, half=5.0):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(D, D))
    cov = A @ A.T / D * 0.05 + np.diag(np.full(D, 0.01))
    names = ["log10_a_1", "log10_b_1", "log10_c_1", "log10_d_1"][:D]      # not in PHYS_BOX
    prior = BoxPrior([-half] * D, [half] * D, names)
    dmap = cy.DecadeMap(prior)
    x_star = np.zeros(D)
    return prior, dmap, x_star, cov


def test_decade_map_et_round_trip():
    prior = BoxPrior([12, 0.5, -12, -12], [18, 0.98, -4, -4],
                     ["log10_Nt_1", "Et_rel_1", "log10_taun_1", "log10_taup_1"])
    dmap = cy.DecadeMap(prior)
    th = np.array([[15.0, 0.9, -8.0, -7.0]])
    x = dmap.to_x(th)
    n1 = prior.Nc * math.exp((0.9 * prior.Eg - prior.Eg) / prior.kT)
    assert x[0, 1] == pytest.approx(math.log10(n1), rel=1e-12)
    assert np.allclose(dmap.to_theta(x, th[0]), th)
    lo, hi = dmap.box()
    assert hi[1] == pytest.approx(dmap.offset + dmap.slope * 0.98)   # min(prior 0.98, phys 1.0)


def test_chord_and_membership():
    prior, dmap, x_star, cov = _toy()
    es = cy.eigen_system(x_star, cov, dmap.keys, sigma_min=0.0)
    lo, hi = dmap.box()
    cyl = cy.Cylinders(es, lo, hi, radius=2.0)
    for k in range(es.D):
        a, b = cyl.chords_t[k]
        for t in (a, b):
            x = x_star + t * es.V[:, k]
            assert np.isclose(np.max(np.maximum(lo - x, x - hi)), 0.0, atol=1e-9)   # on a face
    z, _ = cyl.sample(0, 10, seed=1)
    assert cyl.member(z)[:, 0].all()


def test_mixture_density_integrates_to_one():
    prior, dmap, x_star, cov = _toy(D=3, half=0.6)
    es = cy.eigen_system(x_star, cov, dmap.keys, sigma_min=0.0)
    lo, hi = dmap.box()
    cyl = cy.Cylinders(es, lo, hi, radius=2.0)
    z, cid, nd, na, lv = cy.draw(cyl, dmap, prior, x_star, m=12, seed=3)
    # MC estimate of ∫ q dz over a box in z that encloses all cylinders: E_uniform[q] * vol
    zl, zh = z.min(axis=0) - 0.1, z.max(axis=0) + 0.1
    u = np.random.default_rng(0).uniform(zl, zh, size=(2_000_000, 3))
    xu = es.to_x(u)
    keep = cy.in_box(xu, lo, hi)
    q = np.zeros(len(u))
    q[keep] = np.exp(cy.mixture_log_q(cyl, u[keep], na, lv))
    assert np.mean(q) * np.prod(zh - zl) == pytest.approx(1.0, rel=0.03)


def test_weights_recover_gaussian_posterior():
    """L = Laplace Gaussian -> weighted mean/cov of the cylinder samples = (x*, cov)."""
    prior, dmap, x_star, cov = _toy(D=4)
    es = cy.eigen_system(x_star, cov, dmap.keys, sigma_min=0.0)
    lo, hi = dmap.box()
    cyl = cy.Cylinders(es, lo, hi, radius=4.0)        # wide enough to hold the Gaussian mass
    P = np.linalg.inv(cov)
    ll = lambda th: -0.5 * np.einsum("bi,ij,bj->b", th - x_star, P, th - x_star)
    res = cy.run(cyl, dmap, prior, x_star, ll, m=14, seed=0)
    mean = res.w @ res.x
    c = (res.x - mean).T @ ((res.x - mean) * res.w[:, None])
    assert np.allclose(mean, x_star, atol=0.02)
    assert np.allclose(np.sqrt(np.diag(c)), np.sqrt(np.diag(cov)), rtol=0.05)
    assert res.ess > 200          # long chords put most points in the tails (expected)


def test_conditional_cov_schur():
    rng = np.random.default_rng(2)
    A = rng.normal(size=(4, 4)); cov = A @ A.T + np.eye(4)
    keep = [0, 1, 3]
    c = cy.conditional_cov(cov, keep)
    # conditional covariance = S_kk - S_kr S_rr^-1 S_rk
    r = [2]
    ref = cov[np.ix_(keep, keep)] - cov[np.ix_(keep, r)] @ np.linalg.inv(cov[np.ix_(r, r)]) @ cov[np.ix_(r, keep)]
    assert np.allclose(c, ref)


def test_sloppy_direction_matches_uniform_box_reference():
    """sigma_1 >> box (as for N_t at the prior bound): cylinder weights vs brute-force uniform-box weights."""
    D = 3
    names = ["log10_a_1", "log10_b_1", "log10_c_1"]
    prior = BoxPrior([-1] * D, [1] * D, names)
    dmap = cy.DecadeMap(prior)
    R = np.linalg.qr(np.random.default_rng(5).normal(size=(D, D)))[0]
    cov = R @ np.diag([50.0 ** 2, 0.3 ** 2, 0.08 ** 2]) @ R.T          # one direction far wider than the box
    x_star = np.array([0.8, -0.5, 0.1])                                 # near a corner, as in the real runs
    P = np.linalg.inv(cov)
    ll = lambda th: -0.5 * np.einsum("bi,ij,bj->b", th - x_star, P, th - x_star)
    es = cy.eigen_system(x_star, cov, dmap.keys, sigma_min=0.0)
    lo, hi = dmap.box()
    cyl = cy.Cylinders(es, lo, hi, radius=2.0)
    res = cy.run(cyl, dmap, prior, x_star, ll, m=14, seed=1)
    assert (res.n_accepted > 0.05 * res.n_drawn).all()                 # proposals no longer mostly outside
    u = np.random.default_rng(1).uniform(lo, hi, size=(2_000_000, D))
    wu = np.exp(ll(u) - ll(u).max()); wu /= wu.sum()
    in_cyl = cyl.member(es.to_z(u)).any(axis=1)                         # the cylinders' support
    wu_c = wu * in_cyl; wu_c /= wu_c.sum()
    assert np.allclose(res.w @ res.x, wu_c @ u, atol=0.03)
    sd = lambda x, w: np.sqrt(w @ (x - w @ x) ** 2)
    assert np.allclose(sd(res.x, res.w), sd(u, wu_c), rtol=0.08)
