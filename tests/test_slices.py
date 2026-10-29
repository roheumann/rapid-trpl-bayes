"""Tests of the paper's sliced grid (inference/slices.py)."""
import math

import numpy as np
import pytest

from inference import slices


def test_anchored_axis_contains_anchor_with_linspace_step():
    ax, i = slices.anchored_axis(-12.0, -4.0, 100, -7.123)
    assert ax[i] == pytest.approx(-7.123) and len(ax) == 100
    assert np.allclose(np.diff(ax), 8.0 / 99)


def test_layout_slices_no_diagonal_and_curve_panels():
    keys = ["log10_n1_1", "log10_taup_1", "log10_taun_1", "log10_Nt_1"]
    lay = slices.layout_slices(keys, ["tau_qfls", "pl_t"])
    k2 = lay[lay["kind"] == b"2d"]
    assert len(k2) == 6 and not any(r["kx"] == r["ky"] for r in k2)          # no diagonal
    assert {(r["kx"].decode(), r["ky"].decode()) for r in k2 if r["row"] == 2} == {
        ("log10_n1_1", "log10_Nt_1"), ("log10_taup_1", "log10_Nt_1"), ("log10_taun_1", "log10_Nt_1")}
    cur = lay[lay["kind"] == b"curve"]
    assert [c["kx"].decode() for c in cur] == ["tau_qfls", "pl_t"] and cur[0]["extra"] == b"span:1x2"
    assert lay[(lay["row"] == 0) & (lay["col"] == 0)][0]["extra"] == b"top:Et-Eg"
    keys2 = keys + ["log10_n1_2", "log10_taup_2"]                         # 2 traps: also on top of the n1,2 column
    lay2 = slices.layout_slices(keys2, ["tau_qfls", "pl_t"])
    et = {(int(r["row"]), int(r["col"])) for r in lay2 if r["extra"] == b"top:Et-Eg"}
    assert et == {(0, 0), (4, 4)}


def test_nn_time_axis_reproduces_exponential_decay():
    # tau_diff = const (mono-exponential decay with lifetime tau_e: PL ~ exp(-t/tau_e) -> tau_diff = 2 tau_e)
    kT, tau_e = 0.025852, 1e-7
    q = np.linspace(1.4, 1.0, 400)
    t_true = -(q - 1.4) / kT * tau_e                                       # ln PL = (q - q0)/kT = -t/tau_e
    qq, tau, t, pl = slices.nn_time_axis(q, np.full_like(q, 2 * tau_e), lambda x: 0.0, 1.4, kT)
    assert np.allclose(t, t_true[np.argsort(-q)], rtol=1e-9, atol=1e-15)
    assert np.allclose(pl, np.exp(-t / tau_e))


class _Prior:                       # minimal prior: 3 free parameters, no Et (decade map = identity)
    names = ["log10_Nt_1", "log10_taun_1", "log10_taup_1"]
    free_idx = [0, 1, 2]
    lo, hi = np.array([12.0, -12.0, -12.0]), np.array([18.0, -4.0, -4.0])
    Eg, kT, Nc, n_traps = 1.6, 0.025852, 2e18, 1

    def in_support(self, th):
        th = np.atleast_2d(th)
        return np.all((th >= self.lo) & (th <= self.hi), axis=1)


class _LP:                          # Gaussian "MSE" around a centre
    trpl_terms = [type("T", (), {"name": "trpl"})()]
    sspl_term = None

    def np_mse_terms(self, th):
        return {"trpl": 1e-3 + 0.01 * np.sum((th - np.array([16.0, -8.0, -7.0])) ** 2, axis=1)}


def test_run_slices_rank_and_sigma_eff():
    S = slices.run_slices(_LP(), _Prior(), np.array([16.0, -8.0, -7.0]), n=21, sigma_base={"trpl": 0.03})
    assert S["sigma"]["trpl"]["mse_min"] == pytest.approx(1e-3)
    assert S["sigma"]["trpl"]["eff"] == pytest.approx(math.sqrt(0.03 ** 2 + 1e-3))
    assert len(S["panels"]) == 3
    for p in S["panels"].values():
        assert p["prob"].sum() == pytest.approx(1.0)
        assert np.nanmax(p["log10_rel"]) == pytest.approx(0.0, abs=1e-9)       # best fit = best node
        assert p["rank"].min() == pytest.approx(3 / S["n_ranked"])             # 3 panels share the best node
        assert np.all((p["rank"] > 0) & (p["rank"] <= 1))


def test_rel_drop_is_zero_at_the_best_fit_and_in_unit_interval():
    S = slices.run_slices(_LP(), _Prior(), np.array([16.0, -8.0, -7.0]), n=21, sigma_base={"trpl": 0.03})
    for p in S["panels"].values():
        d = p["rel_drop"]
        assert np.nanmin(d) == pytest.approx(0.0, abs=1e-12)                  # best-fit node of every panel
        assert np.all((d >= 0) & (d <= 1))
        fin = np.isfinite(p["log10_rel"])
        np.testing.assert_allclose(d[fin], 1 - 10 ** p["log10_rel"][fin], atol=1e-12)
