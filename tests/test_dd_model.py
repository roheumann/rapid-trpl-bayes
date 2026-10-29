"""
Fast checks of the drift-diffusion port model-error-analysis/dd_model.py (< ~1 min):

* krad = 0 and no traps: total n and p are conserved under zero-flux BC (while the profile diffuses);
* flat initial profile: tau_diff(QFLS) agrees with the legacy rate equations (trpl_module.solve_transient)
  for a short 1-trap (paper case) and 2-trap transient;
* matlab_gradient reproduces MATLAB gradient(F, h) with a coordinate vector.
"""
from __future__ import annotations

import importlib.util
import warnings
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("dd_model", REPO / "model-error-analysis" / "dd_model.py")
dd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dd)

PAPER = dict(Et=[1.4932], Nt=[4.1975e16], taun=[3.0115e-9], taup=[6.822e-8], krad=1e-10, n_pulse=1.6e17)
TWO_TRAP = dict(Et=[1.4932, 0.85], Nt=[4.1975e16, 1e16], taun=[3.0115e-9, 1e-7], taup=[6.822e-8, 1e-6],
                krad=1e-10, n_pulse=1.6e17)
EG = 1.625


def test_matlab_gradient():
    F = np.array([0.0, 1.0, 4.0, 9.0])
    h = np.array([0.0, 0.5, 2.0, 2.5])
    g = dd.matlab_gradient(F, h)
    expected = [(1 - 0) / 0.5, (4 - 0) / 2.0, (9 - 1) / 2.0, (9 - 4) / 0.5]
    np.testing.assert_allclose(g, expected)


def test_conservation_without_recombination():
    t = np.logspace(-11, -6, 60)
    r = dd.simulate_dd([], [], [], [], 0.0, 1e17, 1e6, nx=60, t=t, return_profiles=True)
    for key in ("n_avg", "p_avg"):
        np.testing.assert_allclose(r[key], r[key][0], rtol=1e-9)
    # the steep exp(-alpha x) profile has flattened (otherwise the test would be trivial)
    assert r["n"][0, 0] / r["n"][0, -1] > 1e10
    assert r["n"][-1, 0] / r["n"][-1, -1] < 1.01


def _legacy_curve(P, tmax):
    leg = dd.legacy_module()
    kT = dd.KB_LEGACY * 300.0
    ni = np.sqrt(dd.INFERENCE_CONSTANTS["Nc"] * dd.INFERENCE_CONSTANTS["Nv"] * np.exp(-EG / kT))
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        out = leg.solve_transient([1e-11, tmax], P["n_pulse"], EG, dd.INFERENCE_CONSTANTS["Nc"],
                                  dd.INFERENCE_CONSTANTS["Nv"], P["krad"], P["Nt"], P["Et"], P["taun"], P["taup"],
                                  300.0)
        tau, pl = out[5], out[8]
        q = 2 * kT * np.log(P["n_pulse"] / ni) + kT * np.log(pl / pl[0])     # nominal axis, as the DD QFLS
    return q, tau, pl


@pytest.mark.parametrize("P", [PAPER, TWO_TRAP], ids=["1trap", "2trap"])
def test_flat_profile_matches_legacy_ode(P):
    tmax = 3e-6
    t = np.logspace(-11, np.log10(tmax), 300)
    r = dd.simulate_dd(P["Et"], P["Nt"], P["taun"], P["taup"], P["krad"], P["n_pulse"], 1e5, nx=8, t=t,
                       profile="flat")
    q_leg, tau_leg, pl_leg = _legacy_curve(P, tmax)
    lo = max(np.nanmin(r["QFLS"]), np.nanmin(q_leg)) + 0.005
    grid = np.linspace(lo, r["QFLS"][0] - 1e-3, 100)
    d = dd.log10_tau_on_qfls(r, grid) - dd.curve_log10_tau(q_leg, tau_leg, pl_leg, grid)
    d = d[np.isfinite(d)]
    assert len(d) > 80
    assert np.sqrt(np.mean(d ** 2)) < 0.01
