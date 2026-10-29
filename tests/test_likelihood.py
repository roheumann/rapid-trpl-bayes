"""
Unit tests 1, 3 and 5 (REVISION_PLAN.md).

1. log_lik_trpl reproduces error_trpl_mse of the original grid pipeline (tests/legacy_reference.py)
   row by row, using the same linear interpolation of tau (interp_space='linear').
3. Autodiff gradients of the log density in u match central finite differences.
5. Timing of likelihood + gradient per batch size runs (feeds task T8), and the
   evaluation counter works inside XLA-compiled code.
"""
import numpy as np
import pandas as pd
import pytest

from conftest import exp_sspl_data, exp_trpl_data


def _setup(art, mode="trpl", interp_space="log10", sigma=0.1):
    from inference import likelihood as L
    from inference import prior as P
    from inference import sspl_tf as SS
    from inference import surrogate as S
    q, tau = exp_trpl_data()
    pr = P.Prior.from_artifacts(art, Eg=1.625)
    sur = S.TrplSurrogate(art, 1.6e17, 1.625, q, interp_space=interp_space)
    terms = [L.make_trpl_term(sur, q, tau, sigma=sigma)]
    sspl = None
    if mode in ("joint", "sspl"):
        x, y = exp_sspl_data()
        obs = SS.SsplObservation(SS.SsplModel(1.625, art.n_traps), "qfls_vs_suns", x, y, G_per_sun=2.743e21)
        sspl = L.SsplTerm(obs, 0.034)
    return pr, L.LogPosterior(pr, terms, sspl, mode=mode)


def test_trpl_mse_matches_legacy_02(artifacts, legacy_02):
    import tensorflow as tf
    pr, lp = _setup(artifacts, interp_space="linear")
    q, tau = exp_trpl_data()
    th = pr.sample(300, rng=21)
    # legacy pipeline: grid DataFrame -> NN inputs -> model -> tau -> MSE
    phys = pr.to_physical(th)
    grid = pd.DataFrame({"krad_cm3s": phys["krad_cm3s"]})
    for k in range(1, pr.n_traps + 1):
        grid[f"Nt_{k}_cm-3"] = phys[f"Nt_{k}_cm-3"]
        grid[f"Et_{k}_eV"] = phys[f"Et_{k}_eV"]
        grid[f"tau_n_{k}_s"] = phys[f"tau_n_{k}_s"]
        grid[f"tau_p_{k}_s"] = phys[f"tau_p_{k}_s"]
    x_nn = legacy_02._build_nn_inputs(grid, pr.n_traps, 1.625, 1.6e17)
    tau_b, valid = legacy_02._predict_gpu(x_nn, artifacts.model, artifacts.input_scaler,
                                          artifacts.output_scaler, batch_size=1024)
    qfls_nn = lp.trpl_terms[0].surrogate.qfls_axis
    mse_legacy, _ = legacy_02._compute_errors(tau_b, valid, qfls_nn, q, tau)
    mse_new = lp.np_mse_terms(th)["trpl"]
    # float32 network: XLA-compiled (new) and eager (legacy) calls differ by
    # float32 rounding, i.e. ~1e-6 relative in the MSE
    rel = np.abs(mse_new - mse_legacy) / np.abs(mse_legacy)
    assert np.median(rel) < 1e-6
    assert np.max(rel) < 1e-5


@pytest.mark.parametrize("mode", ["trpl", "joint"])
def test_gradient_finite_differences(artifacts_2trap, mode):
    """Autodiff gradient of log p(u) vs central differences in u (float32 NN -> loose tol)."""
    import tensorflow as tf
    pr, lp = _setup(artifacts_2trap, mode=mode, sigma=0.5)
    th = pr.sample(8, rng=22)
    u = pr.inverse(th)
    v, g = lp.tf_log_prob_and_grad_u(tf.constant(u))
    g = g.numpy()
    h = 1e-3
    g_fd = np.zeros_like(u)
    for d in range(pr.dim):
        e = np.zeros(pr.dim)
        e[d] = h
        g_fd[:, d] = (lp.np_log_prob_u(u + e) - lp.np_log_prob_u(u - e)) / (2 * h)
    err = np.abs(g - g_fd) / (np.abs(g_fd).max(axis=1, keepdims=True) + 1e-8)
    assert np.median(err) < 1e-3
    assert np.max(err) < 3e-2


def test_counter_and_benchmark(artifacts_2trap):
    from inference.benchmark import benchmark_log_post, count_evals
    pr, lp = _setup(artifacts_2trap, mode="joint")
    with count_evals(lp.counter) as c:
        lp.np_log_prob_u(np.zeros((37, pr.dim)))
        lp.np_log_prob_u(np.zeros((5, pr.dim)))
    assert c.counts == {"n_calls": 2, "n_points": 42}
    df = benchmark_log_post(lp, batch_sizes=(1, 64), repeats=2)
    assert list(df.batch_size) == [1, 64] and np.all(df.time_per_call_ms > 0)


def test_log_prob_theta_outside_support(artifacts_2trap):
    """Outside the prior support the log posterior is -inf (no NN extrapolation)."""
    import tensorflow as tf
    pr, lp = _setup(artifacts_2trap)
    th = pr.sample(4, rng=23)
    th[0, 0] = pr.hi[0] + 0.5                          # k_rad above the training range
    if pr.n_traps > 1:
        th[1, pr.idx_Et] = th[1, pr.idx_Et][::-1]       # E_t1 > E_t2
    lpv = lp.log_prob_theta(tf.constant(th)).numpy()
    assert np.isneginf(lpv[0]) and np.isfinite(lpv[2:]).all()
    if pr.n_traps > 1:
        assert np.isneginf(lpv[1])
