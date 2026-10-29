"""
Multi-data-set likelihood (SBC / coverage): every row must give exactly the
log-likelihood of an ordinary LogPosterior built for its own data set.
"""
import numpy as np

from conftest import exp_trpl_data


def test_multi_equals_single(artifacts):
    import tensorflow as tf
    from inference import likelihood as L, multi, prior as P, surrogate as S
    pr = P.Prior.from_artifacts(artifacts, Eg=1.625)
    q, tau = exp_trpl_data()
    rng = np.random.default_rng(31)
    sets = []
    for k in range(3):                     # three data sets on different QFLS subsets
        sel = np.sort(rng.choice(len(q), size=60 + 20 * k, replace=False))
        sets.append((q[sel], np.log10(tau[sel]) + 0.05 * rng.standard_normal(len(sel))))
    sigmas = np.array([0.1, 0.3, 0.05])
    sur = S.TrplSurrogate(artifacts, 1.6e17, 1.625, q)
    m = multi.MultiTrplLikelihood(sur, pr, sets, sigmas)
    th = pr.sample(12, rng=32)
    idx = np.arange(12) % 3
    ll_multi = m.log_lik(tf.constant(th), tf.constant(idx.astype(np.int32))).numpy()
    for d in range(3):
        sur_d = S.TrplSurrogate(artifacts, 1.6e17, 1.625, sets[d][0])
        lp = L.LogPosterior(pr, [L.make_trpl_term(sur_d, sets[d][0], 10 ** sets[d][1], sigmas[d])], mode="trpl")
        ll_single = lp.np_log_lik_theta(th[idx == d])
        # float32 network: eager (multi) vs XLA-compiled (single) calls differ by ~1e-6 relative
        assert np.allclose(ll_multi[idx == d], ll_single, rtol=1e-5, atol=1e-6)
