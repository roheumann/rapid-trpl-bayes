"""
Sampler check with a known answer: sampling the PRIOR ALONE (log L = 0) must
reproduce the uniform distribution on the ordered training box.

This tests NUTS (warm-up, metric, transforms, stuck-chain logic) and emcee
end to end.  It caught a bias of emcee's DEMove + DESnookerMove mixture
(68 % widths ~4 % too narrow), which is why the default move is 'stretch'.
"""
import numpy as np
import pytest


class _PriorOnly:
    """Minimal LogPosterior stand-in whose density is the prior in u."""

    def __init__(self, prior):
        from inference.benchmark import EvalCounter
        self.prior, self.dim, self.counter = prior, prior.dim, EvalCounter("prior_only")

    def log_prob_u(self, u):
        import tensorflow as tf
        _, logdet = self.prior.forward(u)
        self.counter.record(tf.shape(u)[0])
        return logdet - self.prior.log_volume

    def np_log_prob_u(self, U):
        import tensorflow as tf
        return self.log_prob_u(tf.constant(np.asarray(U, np.float64))).numpy()


def _prior():
    from inference.prior import Prior
    lo = np.array([-12] + [12, 0.5, -12, -12] * 2, float)
    hi = np.array([-10] + [20, 0.975, -4, -4] * 2, float)
    return Prior(lo, hi, 2, Eg=1.625)


def _w68(x):
    return np.subtract(*np.percentile(x, [84, 16], axis=0))


@pytest.fixture(scope="module")
def setup():
    pr = _prior()
    return pr, _PriorOnly(pr), pr.sample(200_000, rng=0)


def test_nuts_recovers_prior(setup):
    from inference import samplers as SM
    pr, po, exact = setup
    res, _ = SM.run_nuts(po, SM.init_dispersed(pr.dim, 8, 0), n_warmup=400, n_draws=1000, verbose=False)
    ratio = _w68(res.flat_theta()) / _w68(exact)
    assert np.all(np.abs(ratio - 1) < 0.05), ratio
    assert res.info["n_divergent"] == 0
    assert np.all(pr.in_support(res.flat_theta()))


def test_emcee_stretch_recovers_prior(setup):
    from inference import samplers as SM
    pr, po, exact = setup
    res = SM.run_emcee(po, SM.init_dispersed(pr.dim, 40, 1), n_steps=12000, moves="stretch", seed=1)
    ratio = _w68(res.flat_theta()) / _w68(exact)
    assert np.all(np.abs(ratio - 1) < 0.05), ratio
