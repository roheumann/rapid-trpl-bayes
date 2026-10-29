"""The legacy ODE solver: settings fixed by Robin (16/10/2026) and the two active copies kept identical.

* atol = atol_vec (per-component: n0/10 for n, max(ni/10, 1) for the traps); a scalar atol sometimes fails;
* integration from t = 0.0 (the right-hand side has no explicit t, so this only shifts the time axis);
* QFLS = kT ln(n p / ni^2), PL = n p - ni^2 (unchanged physics);
* network-training/trpl_module.py (used by inference.ode) == ../nn-training/trpl_module.py (generates training data).
"""
import hashlib
import inspect

import numpy as np
import pytest

from inference import NETWORK_TRAINING_DIR, REPO_ROOT, ode

REPO_COPY = NETWORK_TRAINING_DIR / "trpl_module.py"
TRAINING_COPY = REPO_ROOT.parent / "nn-training" / "trpl_module.py"


def _md5(p):
    return hashlib.md5(p.read_bytes()).hexdigest()


@pytest.mark.skipif(not TRAINING_COPY.exists(), reason="nn-training/ not next to the repository")
def test_repo_copy_identical_to_training_copy():
    assert _md5(REPO_COPY) == _md5(TRAINING_COPY)


def test_solver_settings():
    src = inspect.getsource(ode._ode_module().solve_transient)
    assert "atol_vec = np.concatenate([[n0 * 1e-1], np.maximum(ni * 1e-1 * np.ones(M), 1.0)])" in src
    assert "solve_ivp(_ode, [0.0, tarr[-1]], x0," in src and "atol=atol_vec" in src
    assert "QFLS_tr = kT * np.log(n_tr * p_tr / ni**2)" in src and "PL_tr   = n_tr * p_tr - ni**2" in src


def test_solution_starts_at_t0():
    out = ode._ode_module().solve_transient([1e-11, 1e5], 1.6e17, 1.625, 2.21359e18, 2.21359e18, 1e-10,
                                            np.array([3.76e16]), np.array([1.4964]), np.array([1.633e-9]),
                                            np.array([6.6e-8]))
    t, qfls = out[0], out[4]
    assert t[0] == 0.0 and np.all(np.diff(t) >= 0)
    assert qfls[0] == pytest.approx(1.4892, abs=2e-4)          # = 2 Vt ln(n_pulse/n_i) at high injection


GEN_REPO = NETWORK_TRAINING_DIR / "generate_trpl_dataset.py"
GEN_TRAINING = REPO_ROOT.parent / "nn-training" / "generate_trpl_dataset.py"


@pytest.mark.skipif(not GEN_TRAINING.exists(), reason="nn-training/ not next to the repository")
def test_generator_copy_identical_to_training_copy():
    assert _md5(GEN_REPO) == _md5(GEN_TRAINING)


def test_stored_time_starts_at_tmin():
    """generate_trpl_dataset stores log10((t + tmin)/ns): first point = tmin = 1e-11 s (-2), finite, non-decreasing,
    although the solver starts at t = 0 and takes steps far below 1e-11 s (fast traps)."""
    import importlib.util
    import sys
    prev = sys.modules.get("trpl_module")
    sys.modules["trpl_module"] = ode._ode_module()                    # the generator imports trpl_module by name
    try:
        spec = importlib.util.spec_from_file_location("rtb_generate_trpl_dataset", GEN_REPO)
        gen = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gen)
    finally:                                                          # do not leak the name into other tests
        if prev is None:
            del sys.modules["trpl_module"]
        else:
            sys.modules["trpl_module"] = prev
    for par in ([np.log10(1.6e17), 1.625, -10.0, 16.57, 0.918, -8.8, -7.2],
                [14.0, 1.4, -10.5, 15.0, 0.6, -12.0, -12.0]):       # tau_n = tau_p = 1e-12 s
        ok, res, err, _ = gen._simulate_one(np.array(par), 1, 1e-11, 1e5)
        assert ok, err
        t = res["time_log"]
        assert t[0] == pytest.approx(-2.0, abs=1e-12)
        assert np.all(np.isfinite(t)) and np.all(np.diff(t) >= 0)
