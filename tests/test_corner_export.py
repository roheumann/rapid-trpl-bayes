"""Tests of inference/corner_export.py: HPD maps, contours, layouts, HDF5 round trip."""
import json

import h5py
import numpy as np
import pytest

from inference import corner_export as ce


def test_hpd_level_map_gaussian_sigma1_mass():
    rng = np.random.default_rng(0)
    s = rng.normal(size=(400_000, 2))
    x, y, d = ce.density_2d(s[:, 0], s[:, 1], np.full(len(s), 1 / len(s)), (-5, 5), (-5, 5), bins=100, smooth=0)
    lev = ce.hpd_level_map(d)
    inside = lev <= ce.SIGMA_LEVELS[0]
    mass = d[inside].sum() / d.sum()
    assert mass == pytest.approx(0.6827, abs=0.01)
    # 2D Gaussian: the 68.27 % HPD region is a circle of radius sqrt(-2 ln(1-0.6827)) = 1.515
    poly = ce.contour_polyline(x, y, lev, ce.SIGMA_LEVELS[0])
    r = np.hypot(poly[:, 0], poly[:, 1])
    assert np.nanmedian(r) == pytest.approx(np.sqrt(-2 * np.log(1 - 0.6827)), rel=0.05)


def test_hpd_intervals_1d_bimodal_are_disjoint():
    x = np.linspace(-6, 6, 601)
    pdf = np.exp(-0.5 * ((x + 3) / 0.4) ** 2) + np.exp(-0.5 * ((x - 3) / 0.4) ** 2)
    iv = ce.hpd_intervals_1d(x, pdf / np.trapezoid(pdf, x), 0.6827)
    assert len(iv) == 2


def test_layouts():
    keys = ["log10_n1_1", "log10_taup_1", "log10_taun_1", "log10_Nt_1"]
    L = ce.layout_triangle(keys, et_cols=["log10_n1_1"])
    assert len(L) == 10 and (L["kind"] == b"1d").sum() == 4
    assert L[(L["row"] == 0) & (L["col"] == 0)]["extra"][0] == b"top:Et-Eg"
    t1 = keys
    t2 = ["log10_Nt_2", "log10_taun_2", "log10_taup_2", "log10_n1_2"]
    M = ce.layout_2trap_mirrored(t1, t2)
    assert int(M["row"].max()) == 4 and int(M["col"].max()) == 3 and len(M) == 20
    # trap-1 diagonal at row = col + 1, trap-2 diagonal at row = col (as in corner-2traps.png)
    d = M[M["kind"] == b"1d"]
    assert {(int(r), int(c)) for r, c in zip(d["row"], d["col"])} == \
        {(1, 0), (2, 1), (3, 2), (4, 3), (0, 0), (1, 1), (2, 2), (3, 3)}
    assert ce.physical_order(["log10_krad", "log10_Nt_1", "log10_n1_1", "log10_taun_1", "log10_taup_1"]) == \
        ["log10_n1_1", "log10_taup_1", "log10_taun_1", "log10_Nt_1", "log10_krad"]


def test_write_corner_round_trip(tmp_path):
    rng = np.random.default_rng(1)
    keys = ["log10_n1_1", "log10_taup_1"]
    s = rng.normal([15, -7], [1, 0.5], size=(20_000, 2))
    w = np.full(len(s), 1 / len(s))
    meta = {k: {"label_origin": ce.labels(k, 1)[0], "label_mpl": ce.labels(k, 1)[1], "unit": ce.labels(k, 1)[2],
                "scale": "log10", "best_fit": 15.0 if "n1" in k else -7.0, "truth": None, "start": None} for k in keys}
    with h5py.File(tmp_path / "c.h5", "w") as f:
        ce.write_corner(f.create_group("corner/physical"), s, w, keys, meta, ce.layout_triangle(keys),
                        {"log10_n1_1": (10, 20), "log10_taup_1": (-10, -4)})
    with h5py.File(tmp_path / "c.h5", "r") as f:
        g = f["corner/physical"]
        assert json.loads(g.attrs["param_order"]) == keys
        p = g["panels/log10_n1_1__log10_taup_1"]
        assert p["hpd"].shape == (60, 60) and len(p["sigma1"]) > 0
        assert np.isnan(g["params/log10_n1_1"].attrs["truth"])
        assert g["panels/log10_n1_1"]["intervals68"].shape[1] == 2
