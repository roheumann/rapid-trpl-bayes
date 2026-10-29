"""
Tests of inference/smc.py (tempered SMC and AMIS) on targets with a known answer.

* prior only (log L = 0): one stage, log Z = 0, the particles reproduce the prior;
* Gaussian likelihood well inside the box: Z = 1/V exactly (log Z = -log V), known moments;
* two Gaussian modes 30:70: the mode weights;
* curved ridge ("banana") into a corner of the box, the case the linearised cylinders miss:
  brute-force prior Monte Carlo as the reference;
* ordered 2-trap prior and fixed parameters (support, bijection on the free coordinates);
* every likelihood evaluation is ONE batched call for the whole population;
* the result works with samplers.to_arviz / diagnostics, report.export_samples and the HDF5 corner
  export read by corner_plot (same schema as cylinder.ipynb).
The analytic targets need no network; the last test checks the population evaluator against
LogPosterior.log_prob_u with the (real or random-weight) surrogate of conftest.py.
"""
import math

import numpy as np
import pytest
from scipy.special import logsumexp


class _Analytic:
    """LogPosterior stand-in: log L(theta) in TF, the prior and an evaluation counter."""

    def __init__(self, prior, kind, mu=None, sd=None):
        from inference.benchmark import EvalCounter
        self.prior, self.dim, self.kind = prior, prior.dim, kind
        self.counter = EvalCounter(f"analytic_{kind}")
        self.mu = np.asarray(mu if mu is not None else [-10.0, 16.0, 0.85, -8.0, -7.0], float)
        self.mu2 = self.mu.copy()
        self.mu2[1] = 13.5
        self.sd = np.asarray(sd if sd is not None else [0.3, 0.3, 0.02, 0.4, 0.3], float)

    def log_lik(self, theta):
        import tensorflow as tf
        theta = tf.convert_to_tensor(theta, tf.float64)
        self.counter.record(tf.shape(theta)[0])
        D = len(self.mu)

        def lg(mu):
            z = (theta - mu) / self.sd
            return -0.5 * tf.reduce_sum(z * z, 1) - float(np.sum(np.log(self.sd))) - 0.5 * D * math.log(2 * math.pi)

        if self.kind == "zero":
            return tf.zeros(tf.shape(theta)[:1], tf.float64)
        if self.kind == "gauss":
            return lg(self.mu)
        if self.kind == "bimodal":
            return tf.reduce_logsumexp(tf.stack([lg(self.mu) + math.log(0.3), lg(self.mu2) + math.log(0.7)]), 0)
        if self.kind == "banana":   # E_t/E_g = 0.80 + 0.02 (log10 N_t - 15)^2, width 0.01: runs into the corner
            r = (theta[:, 2] - (0.80 + 0.02 * (theta[:, 1] - 15.0) ** 2)) / 0.01
            g = theta[:, 1] - 15.0
            z = (tf.gather(theta, [0, 3, 4], axis=1) - np.array([-10.0, -8.0, -7.0])) / np.array([0.3, 0.4, 0.3])
            return -0.5 * (r * r + g * g + tf.reduce_sum(z * z, 1))
        raise KeyError(self.kind)

    def np_log_lik(self, theta):
        import tensorflow as tf
        return self.log_lik(tf.constant(np.asarray(theta, np.float64))).numpy()


def _prior1(fixed_krad=False):
    from inference.prior import Prior, FIXED_HALF_WIDTH
    lo = np.array([-12, 12, 0.5, -12, -12.0])
    hi = np.array([-9, 18.30103, 0.98, -4, -4.0])
    if fixed_krad:
        lo[0], hi[0] = -10.0 - FIXED_HALF_WIDTH, -10.0 + FIXED_HALF_WIDTH
    return Prior(lo, hi, 1, Eg=1.625)


def _w68(x):
    return np.subtract(*np.percentile(x, [84, 16], axis=0))


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────
def test_systematic_resample_counts():
    from inference.smc import systematic_resample
    rng = np.random.default_rng(0)
    w = rng.dirichlet(np.ones(50))
    for _ in range(20):
        idx = systematic_resample(w, 1000, rng)
        c = np.bincount(idx, minlength=50)
        assert np.all(np.abs(c - 1000 * w) < 1.0 + 1e-9)
        assert np.all(np.diff(idx) >= 0)          # sorted: descendants stay neighbours


def test_next_beta_hits_ess_target():
    from inference.smc import next_beta
    ll = np.random.default_rng(1).normal(-50, 10, 4000)
    b = next_beta(ll, 0.0, 0.5)
    lw = b * ll
    ess = math.exp(2 * logsumexp(lw) - logsumexp(2 * lw)) / len(ll)
    assert 0 < b < 1 and abs(ess - 0.5) < 1e-3
    assert next_beta(np.zeros(100), 0.3, 0.5) == 1.0


def test_tmixture_logpdf_and_em():
    from scipy.stats import multivariate_t
    from inference.smc import TMixture, fit_tmixture_em
    rng = np.random.default_rng(2)
    A = rng.normal(size=(3, 3))
    cov = A @ A.T + 0.5 * np.eye(3)
    mix = TMixture([0.3, 0.7], [[0, 0, 0], [4, 1, -2]], [cov, 0.5 * cov], nu=5.0)
    x, comp = mix.sample(40000, rng)
    ref = np.logaddexp(math.log(0.3) + multivariate_t(np.zeros(3), cov, df=5).logpdf(x),
                       math.log(0.7) + multivariate_t([4, 1, -2], 0.5 * cov, df=5).logpdf(x))
    assert np.allclose(mix.logpdf(x), ref, atol=1e-10)
    assert abs(np.mean(comp == 0) - 0.3) < 1e-3                          # stratified counts
    init = TMixture([0.5, 0.5], [[1, 1, 1], [3, 0, -1]], [np.eye(3), np.eye(3)], nu=5.0)
    fit = fit_tmixture_em(x, np.ones(len(x)), init, n_iter=200, var_floor=1e-3)
    o = np.argsort(fit.means[:, 0])
    assert np.allclose(fit.weights[o], [0.3, 0.7], atol=0.02)
    assert np.allclose(fit.means[o], mix.means, atol=0.1)


# ─────────────────────────────────────────────────────────────────────────────
# tempered SMC
# ─────────────────────────────────────────────────────────────────────────────
def test_smc_prior_only():
    from inference import smc
    pr = _prior1()
    res = smc.run_smc(_Analytic(pr, "zero"), n_particles=2048, n_islands=2, seed=0, verbose=False)
    assert res.info["n_stages"] == [1, 1]
    assert abs(res.log_evidence) < 1e-12
    exact = pr.sample(200_000, rng=0)
    ratio = _w68(res.flat_theta()) / _w68(exact)
    assert np.all(np.abs(ratio - 1) < 0.06), ratio


@pytest.mark.parametrize("kind", ["gauss", "bimodal"])
def test_smc_evidence_and_moments(kind):
    from inference import smc
    pr = _prior1()
    lp = _Analytic(pr, kind)
    res = smc.run_smc(lp, n_particles=2048, n_islands=4, seed=3, verbose=False)
    # both targets are normalised densities well inside the box: Z = 1/V
    assert abs(res.log_evidence + pr.log_volume) < 4 * res.log_evidence_se + 0.05, \
        (res.log_evidence, -pr.log_volume, res.log_evidence_se)
    th, w = res.flat_weighted()
    if kind == "gauss":
        assert np.all(np.abs(w @ th - lp.mu) < 0.05 * lp.sd)
        sd = np.sqrt(w @ (th - w @ th) ** 2)
        assert np.all(np.abs(sd / lp.sd - 1) < 0.06), sd / lp.sd
    else:
        assert abs(w @ (th[:, 1] < 14.75) - 0.7) < 0.03
        assert abs(np.mean(res.flat_theta()[:, 1] < 14.75) - 0.7) < 0.03
    assert np.all(res.stages.groupby("island").beta.max() == 1.0)
    assert res.stages.ess_frac.min() >= 0.5 - 1e-6


def test_smc_one_batched_call_per_move():
    """Initial draw + one call per MH step; every call carries the whole (active) population."""
    from inference import smc
    pr = _prior1()
    lp = _Analytic(pr, "gauss")
    res = smc.run_smc(lp, n_particles=1000, n_islands=3, seed=4, verbose=False)
    calls_expected = 1 + int(res.stages.groupby("stage").n_mh.max().sum())
    assert res.info["evals_total_calls"] == calls_expected
    # points: islands still active in a stage x particles per MH step (padding removed from the counter)
    pts = 3000 + sum(int(g.n_mh.sum()) * 1000 for _, g in res.stages.groupby("stage"))
    assert res.info["evals_total_points"] == pts


def test_smc_banana_matches_brute_force():
    from inference import smc
    pr = _prior1()
    lp = _Analytic(pr, "banana")
    th_ref = pr.sample(2_000_000, rng=5)
    ll = lp.np_log_lik(th_ref)
    lz_ref = logsumexp(ll) - math.log(len(ll))
    wr = np.exp(ll - ll.max())
    wr /= wr.sum()
    res = smc.run_smc(lp, n_particles=2048, n_islands=4, seed=6, verbose=False)
    assert abs(res.log_evidence - lz_ref) < 4 * res.log_evidence_se + 0.08
    th, w = res.flat_weighted()
    m_ref, m = wr @ th_ref, w @ th
    s_ref = np.sqrt(wr @ (th_ref - m_ref) ** 2)
    assert np.all(np.abs(m - m_ref) < 0.1 * s_ref), (m, m_ref)


def test_smc_ordered_two_traps_and_fixed_parameter():
    from inference import smc
    from inference.prior import Prior, FIXED_HALF_WIDTH
    lo = np.array([-10 - FIXED_HALF_WIDTH] + [12, 0.5, -12, -12] * 2, float)
    hi = np.array([-10 + FIXED_HALF_WIDTH] + [18.3, 0.98, -4, -4] * 2, float)
    pr = Prior(lo, hi, 2, Eg=1.625)
    mu = np.array([-10, 15, 0.70, -8, -7, 16, 0.72, -7, -8.0])
    sd = np.array([1.0, 0.5, 0.03, 0.5, 0.5, 0.5, 0.03, 0.5, 0.5])
    lp = _Analytic(pr, "gauss", mu=mu, sd=sd)
    res = smc.run_smc(lp, n_particles=2048, n_islands=2, seed=7, verbose=False)
    th = res.flat_theta()
    assert np.all(pr.in_support(th))
    assert np.allclose(th[:, 0], -10.0, atol=1e-8)                       # fixed k_rad stays at its value
    assert res.info["dim_free"] == 8
    # the energies overlap (0.70 vs 0.72, sd 0.03): the ordering cuts the Gaussian -> E[E_t1] < 0.70
    assert np.mean(th[:, 2]) < 0.70 < np.mean(th[:, 6])


def test_sigma_path_endpoint():
    from inference import smc
    pr = _prior1()
    res = smc.run_smc(_Analytic(pr, "gauss"), n_particles=1024, n_islands=2, seed=8, verbose=False)
    sp = smc.sigma_path(res, sigma=0.1)
    last = sp[sp.beta == 1.0]
    assert np.allclose(last.sigma, 0.1) and np.allclose(last.log_z_sigma, last.log_z)
    assert np.all(np.diff(sp[sp.island == 0].sigma.values) < 0)           # sigma' decreases along the schedule


# ─────────────────────────────────────────────────────────────────────────────
# AMIS
# ─────────────────────────────────────────────────────────────────────────────
def test_amis_gaussian_from_cylinders():
    from inference import smc
    pr = _prior1()
    lp = _Analytic(pr, "gauss")
    # the cylinder covariance is deliberately 2x too wide and rotated a bit; AMIS has to adapt
    cov = np.diag((2.0 * lp.sd) ** 2)
    cov[1, 2] = cov[2, 1] = 0.3 * np.sqrt(cov[1, 1] * cov[2, 2])
    dmap, es, cyl = smc.cylinder_setup(pr, lp.mu, cov, 1.0, 1.0)
    res = smc.run_amis(lp, dmap, lp.mu, cyl=cyl, n_per_iter=4096, n_iter=5, cyl_m=11, seed=9, verbose=False)
    assert abs(res.log_evidence + pr.log_volume) < 0.03
    assert res.info["k_hat"] < 0.7
    assert res.stages.ess_frac_total.iloc[-1] > 0.3
    th, w = res.flat_weighted()
    assert np.all(np.abs(w @ th - lp.mu) < 0.05 * lp.sd)
    # defensive component: every iteration has prior draws, so q > 0 on the whole support
    assert np.all(np.isfinite(res.weighted["log_w"]))


def test_amis_banana_beats_fixed_cylinders():
    """The linearised covariance misses the curved ridge; the adapted t-mixture covers it."""
    from inference import cylinder as cy
    from inference import smc
    pr = _prior1()
    lp = _Analytic(pr, "banana")
    theta_star = np.array([-10.0, 15.0, 0.80, -8.0, -7.0])
    cov = np.diag(np.array([0.3, 1.0, 0.01, 0.4, 0.3]) ** 2)              # Laplace covariance at the mode
    dmap, es, cyl = smc.cylinder_setup(pr, theta_star, cov, 1.0, 1.0)
    th_ref = pr.sample(2_000_000, rng=5)
    ll = lp.np_log_lik(th_ref)
    lz_ref = logsumexp(ll) - math.log(len(ll))
    wr = np.exp(ll - ll.max())
    wr /= wr.sum()
    th_post = th_ref[np.random.default_rng(0).choice(len(th_ref), 4000, p=wr)]
    res = smc.run_amis(lp, dmap, theta_star, cyl=cyl, n_per_iter=8192, n_iter=6, cyl_m=12, seed=3, verbose=False)
    assert abs(res.log_evidence - lz_ref) < 0.05
    frac_cyl = cy.fraction_inside(cyl, dmap.to_x(th_post))
    frac_amis = smc.mixture_hdr_fraction(res.info["proposals"][-1], pr.inverse(th_post))   # t-mixture lives in u
    assert frac_cyl < 0.85 and frac_amis > 0.97, (frac_cyl, frac_amis)


def test_amis_from_smc_particles():
    from inference import cylinder as cy
    from inference import smc
    pr = _prior1()
    lp = _Analytic(pr, "bimodal")
    s = smc.run_smc(lp, n_particles=2048, n_islands=2, seed=10, verbose=False)
    dmap = cy.DecadeMap(pr)
    th, w = s.flat_weighted()
    mix = smc.fit_tmixture(pr.inverse(th), w, n_components=4)                   # in u (no fixed parameters)
    res = smc.run_amis(lp, dmap, th[0], init_mixture=mix, n_per_iter=4096, n_iter=4, seed=11, verbose=False)
    assert abs(res.log_evidence + pr.log_volume) < 0.03
    tw, ww = res.flat_weighted()
    assert abs(ww @ (tw[:, 1] < 14.75) - 0.7) < 0.02


# ─────────────────────────────────────────────────────────────────────────────
# compatibility with the rest of the package
# ─────────────────────────────────────────────────────────────────────────────
def test_result_compatibility(tmp_path):
    import h5py
    import matplotlib
    matplotlib.use("Agg")
    import pandas as pd
    from inference import corner_plot, report, samplers, smc
    pr = _prior1()
    lp = _Analytic(pr, "gauss")
    res = smc.run_smc(lp, n_particles=1024, n_islands=4, seed=12, verbose=False)
    assert isinstance(res, samplers.SamplerResult)
    assert res.theta.shape == (4, 1024, 5)
    diag = samplers.diagnostics(res, pr)
    assert diag.r_hat.max() < 1.05 and diag.ess_bulk.min() > 200
    report.export_samples(res, pr, tmp_path / "s.csv")
    assert len(pd.read_csv(tmp_path / "s.csv")) == 4096
    h5 = smc.write_h5(tmp_path / "smc.h5", res, pr, best_theta=lp.mu, truth_theta=lp.mu)
    with h5py.File(h5, "r") as f:
        assert abs(f["inference"].attrs["log_evidence"] - res.log_evidence) < 1e-12
        g = f["corner/physical"]
        assert "log10_n1_1" in g["panels"] and len(g["panels"]) == 15          # 5 diagonal + 10 pairs
        fig = corner_plot.plot_corner(g)
        assert fig is not None


def test_population_evaluator_matches_log_prob_u(artifacts):
    """ll + log p0 of the evaluator = LogPosterior.log_prob_u (no fixed parameters), real or random network."""
    from conftest import exp_trpl_data
    from inference import likelihood as L
    from inference import prior as P
    from inference import smc
    from inference import surrogate as S
    q, tau = exp_trpl_data()
    pr = P.Prior.from_artifacts(artifacts, Eg=1.625)
    sur = S.TrplSurrogate(artifacts, 1.6e17, 1.625, q)
    lp = L.LogPosterior(pr, [L.make_trpl_term(sur, q, tau, sigma=0.3)], None, "trpl")
    ev = smc.PopulationEvaluator(lp)
    U = np.random.default_rng(13).uniform(-2, 2, size=(100, pr.dim))
    snap = lp.counter.snapshot()
    ll, l0 = ev(U)
    assert lp.counter.delta_since(snap) == {"n_calls": 1, "n_points": 100}   # padded to 128, counted 100
    ref = lp.np_log_prob_u(U)
    assert np.allclose(ll + l0, ref, rtol=1e-6, atol=1e-6)
    # fixed shape: every chunk padded to max_batch (one XLA program), padding not counted
    ev2 = smc.PopulationEvaluator(lp, max_batch=64, fixed_shape=True)
    snap = lp.counter.snapshot()
    ll2, l02 = ev2(U)
    assert lp.counter.delta_since(snap) == {"n_calls": 2, "n_points": 100}
    assert np.allclose(ll2 + l02, ref, rtol=1e-6, atol=1e-6)
