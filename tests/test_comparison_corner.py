"""Tests of the comparison corner (corner_export.write_comparison_corner), the Gauss-Newton covariance and the
Kish-ESS efficiency rows (15/10/2026)."""
import json

import h5py
import numpy as np
import pytest

from inference import corner_export as ce
from inference.benchmark import efficiency_text, importance_efficiency
from inference.cylinder import gauss_newton_cov


def _toy(n, seed, shift=0.0):
    rng = np.random.default_rng(seed)
    return rng.multivariate_normal([shift, 0.0, 1.0], [[1, 0.6, 0], [0.6, 1, 0], [0, 0, 0.5]], size=n)


def test_comparison_corner_schema(tmp_path):
    keys = ["a", "b", "c"]
    meta = {k: {"label_origin": k, "label_mpl": k, "unit": "", "scale": "linear", "best_fit": 0.0, "truth": 0.0,
                "start": None} for k in keys}
    ranges = {k: (-5.0, 5.0) for k in keys}
    lay = ce.layout_triangle(keys)
    methods = {"NUTS": (_toy(20000, 0), None), "SMC": (_toy(8000, 1), np.full(8000, 2.0))}
    scatter = _toy(3000, 2)
    with h5py.File(tmp_path / "c.h5", "w") as f:
        ce.write_comparison_corner(f.create_group("corner/physical"), "SMC", methods, keys, meta, lay, ranges,
                                   scatter=scatter, method_info={"SMC": {"label": "SMC (1 s)", "log_z": -3.0}},
                                   bins_2d=40, bins_1d=96, smooth=0.5)   # small smoothing: HPD mass unbiased
    with h5py.File(tmp_path / "c.h5", "r") as f:
        g = f["corner/physical"]
        assert g.attrs["fill_method"] == "SMC" and json.loads(g.attrs["methods"]) == ["NUTS", "SMC"]
        assert json.loads(g.attrs["method_dash_json"]) == {"NUTS": "solid", "SMC": "dot"}
        assert len(json.loads(g.attrs["band_colors_json"])) == len(ce.HPD_LEVELS) + 1
        assert g["methods/SMC"].attrs["label"] == "SMC (1 s)" and g["methods/SMC"].attrs["log_z"] == -3.0
        for m in ("NUTS", "SMC"):
            assert len(g[f"methods/{m}/a__b/sigma1"][()]) > 10 and len(g[f"methods/{m}/a__b/sigma2"][()]) > 10
            iv = g[f"methods/{m}/a/intervals68"][()]
            assert iv.shape == (1, 2) and iv[0, 0] == pytest.approx(-1.0, abs=0.25) and iv[0, 1] == pytest.approx(1.0, abs=0.25)
        sc = g["panels/a__b/scatter"][()]
        assert sc.shape == (3000, 3) and np.all(np.diff(sc[:, 2]) <= 1e-6)      # outermost first, densest last
        assert 0.0 <= sc[:, 2].min() and sc[:, 2].max() <= 1.0
        # ~68 % of the equal-weight samples lie inside the 1σ level of the fill map
        assert np.mean(sc[:, 2] <= ce.SIGMA_LEVELS[0]) == pytest.approx(0.68, abs=0.05)


def test_level_at_outside_is_one():
    xc, yc = np.linspace(0, 1, 5), np.linspace(0, 2, 3)
    lev = np.arange(15, dtype=float).reshape(3, 5) / 20
    out = ce.level_at(np.array([0.0, 1.0, 5.0]), np.array([0.0, 2.0, 0.0]), xc, yc, lev)
    assert out[0] == lev[0, 0] and out[1] == lev[2, 4] and out[2] == 1.0


def test_gauss_newton_cov_linear_model():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(30, 3))
    s = np.full(30, 0.2)
    cov, n = gauss_newton_cov(lambda TH: TH @ A.T, np.array([0.3, -1.0, 2.0]), [0, 1, 2], s)
    assert n == 30
    np.testing.assert_allclose(cov, np.linalg.inv(A.T @ A / 0.04), rtol=1e-6)


def test_importance_efficiency_text():
    e = importance_efficiency("cylinder IS", 3275.2, 153619, 14.2, sigma_trpl=0.11, note="x")
    assert e["min_ess_bulk"] is None and e["ess_per_s"] == pytest.approx(3275.2 / 14.2)
    assert efficiency_text(e) == "Kish ESS 3,275 (231 ESS/s)"
