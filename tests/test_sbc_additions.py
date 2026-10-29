"""ECDF bands, coverage / contraction tables and the pairing of SBC runs with another data model (19/10/2026)."""
import numpy as np
import tensorflow as tf

from inference import sbc
from test_smc_sbc import _toy

SMC = dict(n_particles=512, n_islands=2, ess_target=0.5, target_accept=0.25, p_move=0.95, min_mh=2, max_mh=20)


def test_simultaneous_band_has_nominal_coverage():
    n, L = 200, 100
    b = sbc.ecdf_band(n, L)
    rng = np.random.default_rng(7)
    inside = []
    for _ in range(2000):
        r = rng.integers(0, L + 1, size=n)
        z, e = sbc.ecdf_difference(r, L)
        inside.append(np.all((e[1:-1] >= b["lo"] - 1e-12) & (e[1:-1] <= b["hi"] + 1e-12)))
    assert 0.93 < np.mean(inside) < 0.97
    pw = sbc.ecdf_band(n, L, method="pointwise")
    mid = (b["z"] > 0.1) & (b["z"] < 0.9)                       # (ends: binomial discreteness)
    assert np.all(pw["hi"][mid] <= b["hi"][mid] + 1e-12)       # pointwise band is narrower
    p = [sbc.ecdf_test_p(rng.integers(0, L + 1, size=n), L) for _ in range(400)]
    assert 0.03 < np.mean(np.array(p) < 0.05) < 0.08          # test p-value ~ uniform under H0


def test_sbc_tables_and_pairing_with_another_data_model():
    prior, sur, lp = _toy()
    kw = dict(n_datasets=24, n_keep=19, smc_kwargs=SMC, seed=5, max_batch=1024, verbose=False)
    a = sbc.run_sbc_smc(lp, prior, **kw)
    # "other model": the same linear model, NaN (= not covering) for half of the parameter space -> redraws
    def other(th):
        y = sur.log10_tau(tf.constant(th)).numpy() + 0.01
        y[th[:, 1] > 15.0] = np.nan
        return [y]
    b = sbc.run_sbc_smc(lp, prior, trpl_model=other, data_model="toy", **kw)
    assert b.info["n_redrawn"] > 0 and b.info["data_model"] == "toy"
    same = np.all(a.theta_true == b.theta_true, axis=1)
    assert np.array_equal(same, ~b.diagnostics.redrawn.to_numpy())   # only redrawn data sets differ
    assert np.all(b.theta_true[:, 1] <= 15.0)
    # same noise and SMC seeds on the paired data sets: B's ranks equal A's up to the small model offset
    cal = sbc.calibration_table(a)
    assert {"cov50", "cov68_se", "cov95", "contraction_median", "z_mean"} <= set(cal.columns)
    assert np.all((cal.contraction_median > -0.5) & (cal.contraction_median <= 1.0))
    pc = sbc.paired_comparison(a, b)
    assert np.all(pc.n_paired == same.sum())
    uni = sbc.uniformity_table(a)
    assert "p_ecdf_sim" in uni.columns
    s = sbc.subset(a, a.diagnostics.max_r_hat <= 10)
    assert len(s.ranks) == len(a.ranks)


def test_default_path_reproduces_the_former_random_stream():
    """The NN-data path draws theta, redraws and noise from one generator, exactly as before 19/10/2026."""
    prior, sur, lp = _toy()
    rng = np.random.default_rng(11)
    th, yt, ys = sbc.simulate_mode_data(lp, prior, 6, rng)
    rng2 = np.random.default_rng(11)
    th2 = prior.sample(6, rng=rng2)
    y2 = sur.log10_tau(tf.constant(th2)).numpy()
    y2 = y2 + float(lp.sigma_trpl[0].numpy()) * np.sqrt(y2.shape[1]) * rng2.standard_normal(y2.shape)
    np.testing.assert_array_equal(th, th2)
    np.testing.assert_allclose(yt[0], y2, rtol=0, atol=0)
