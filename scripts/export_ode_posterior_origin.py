"""
export_ode_posterior_origin.py — the posterior of the SMC run with the rate equations (ODE) as Origin worksheets
==============================================================================================================
19/10/2026 (Robin).  Input: a run folder with ode_vs_nn/ (ode_inference.ipynb) and the comparison corner written by

    python scripts/combined_corner.py results/<run>_<mode> --smc smc_B2nf \
        --extra "ODE SMC=results/<run>_<mode>/ode_vs_nn/ode_smc_weighted_samples.csv" --out corner_combined_with_ode
    python scripts/export_ode_posterior_origin.py results/<run>_<mode> [--corner corner_combined_with_ode]

The HDF5 file <run>/<corner>/<corner>.h5 is the combined corner (NUTS solid, emcee dashed, NN SMC dotted) plus the
ODE SMC posterior (dash-dot) in the same bins and smoothing; build_corner.py draws it as it is.  This script writes the
ODE posterior once more as plain CSV worksheets into <run>/<corner>/origin_csv/ (all numbers read from the HDF5 file,
so they are identical to the contours in the corner):

  ode_smc_marginals_1d.csv  per parameter: <key>_x (plot coordinate), <key>_pdf_<method> for ODE SMC and every other
                            posterior of the file (same bins); n1 panels also <key>_Et_minus_Eg_eV
  ode_smc_contours_2d.csv   per 2D panel <kx>__<ky>: _sigma1_x, _sigma1_y, _sigma2_x, _sigma2_y of the ODE SMC
                            posterior (68.27 / 95.45 % HPD), NaN rows = gaps between closed lines
  ode_smc_intervals.csv     per parameter and posterior: median, 68 / 95 % HPD intervals (plot coordinates and
                            physical units), shift of the ODE median vs the NN SMC median in NN posterior sd
  ode_smc_samples.csv       equal-weight particles of the ODE SMC (physical units + plot coordinates)
  README.txt                column description
Plot coordinates: log10 of the physical value (cm^-3, s) for N_t, tau_n, tau_p and n1 (n1 replaces E_t, as in every
corner): E_t - E_g = (log10 n1 - log10 N_C) kT ln 10.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
ODE = "ODE SMC"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("run")
    ap.add_argument("--corner", default="corner_combined_with_ode")
    ap.add_argument("--ode-dir", default="ode_vs_nn")
    args = ap.parse_args()
    import h5py
    run = Path(args.run)
    h5p = run / args.corner / f"{args.corner}.h5"
    out = run / args.corner / "origin_csv"
    out.mkdir(parents=True, exist_ok=True)
    f = h5py.File(h5p, "r")
    g = f["corner/physical"]
    keys = json.loads(g.attrs["param_order"])
    methods = json.loads(g.attrs["methods"])
    assert ODE in methods, f"{h5p} has no '{ODE}' posterior (run combined_corner.py with --extra)"
    gm = g["methods"]
    meta = {k: dict(g["params"][k].attrs) for k in keys}
    unit = {k: meta[k].get("unit", "") for k in keys}
    if isinstance(next(iter(unit.values())), bytes):
        unit = {k: v.decode() for k, v in unit.items()}
    order = [ODE] + [m for m in methods if m != ODE]

    # ── 1D marginals (same x for every posterior) ──────────────────────────────
    cols = {}
    for k in keys:
        x = gm[ODE][k]["x"][()]
        cols[f"{k}_x"] = x
        if "log10_Nc" in meta[k]:                       # n1 panels: the E_t - E_g axis of the corner
            cols[f"{k}_Et_minus_Eg_eV"] = (x - meta[k]["log10_Nc"]) * meta[k]["kT_ln10"]
        for m in order:
            assert np.allclose(gm[m][k]["x"][()], x)
            cols[f"{k}_pdf_{m.replace(' ', '_')}"] = gm[m][k]["pdf"][()]
    pd.DataFrame(cols).to_csv(out / "ode_smc_marginals_1d.csv", index=False, float_format="%.8g")

    # ── 2D 1σ / 2σ contours of the ODE posterior (one column block per panel) ───
    blocks = {}
    for name in gm[ODE]:
        if "__" not in name:
            continue
        for s in ("sigma1", "sigma2"):
            a = gm[ODE][name][s][()]
            blocks[f"{name}_{s}_x"], blocks[f"{name}_{s}_y"] = a[:, 0], a[:, 1]
    n = max(len(v) for v in blocks.values())
    pd.DataFrame({c: np.r_[v, np.full(n - len(v), np.nan)] for c, v in blocks.items()}).to_csv(
        out / "ode_smc_contours_2d.csv", index=False, float_format="%.8g")

    # ── intervals ──────────────────────────────────────────────────────────────
    w_nn = pd.read_csv(run / "smc_B2nf" / "posterior_samples_smc.csv") if (run / "smc_B2nf").exists() else None
    rows = []
    for k in keys:
        for m in order:
            a = gm[m][k]
            iv = lambda d: "; ".join(f"[{lo:.4f}, {hi:.4f}]" for lo, hi in a[d][()])
            ivp = lambda d: "; ".join(f"[{10 ** lo:.4g}, {10 ** hi:.4g}]" for lo, hi in a[d][()])
            med = float(a.attrs["median"])
            rows.append({"parameter": k, "posterior": m, "median_plot": med, "median_physical": 10 ** med,
                         "unit": unit[k], "HPD68_plot": iv("intervals68"), "HPD95_plot": iv("intervals95"),
                         "HPD68_physical": ivp("intervals68"), "HPD95_physical": ivp("intervals95"),
                         "best_fit_plot": meta[k].get("best_fit"), "truth_plot": meta[k].get("truth")})
    iv_tab = pd.DataFrame(rows)
    # shift of the ODE median relative to the NN SMC posterior (in NN posterior standard deviations)
    if "SMC" in methods:
        sds = {}
        for k in keys:
            x, p_ = gm["SMC"][k]["x"][()], gm["SMC"][k]["pdf"][()]
            p_ = p_ / p_.sum()
            mu = float(np.sum(x * p_))
            sds[k] = float(np.sqrt(np.sum(p_ * (x - mu) ** 2)))
        med = iv_tab.set_index(["parameter", "posterior"]).median_plot
        iv_tab["shift_vs_NN_SMC_in_NN_sd"] = [(med[(r.parameter, r.posterior)] - med[(r.parameter, "SMC")]) / sds[r.parameter]
                                              for r in iv_tab.itertuples()]
    iv_tab.to_csv(out / "ode_smc_intervals.csv", index=False, float_format="%.6g")

    # ── equal-weight particles of the ODE SMC ──────────────────────────────────
    smp = pd.read_csv(run / args.ode_dir / "posterior_samples_ode_smc.csv")
    phys = [c for c in smp.columns if c.endswith(("_eV", "_cm-3", "_s", "_cm3s")) and smp[c].std() > 0]
    s_out = smp[phys].copy()
    for k in keys:              # plot coordinates of the corner
        if k.startswith("log10_n1"):
            t = k.rsplit("_", 1)[1]
            s_out[k] = meta[k]["log10_Nc"] + (smp[f"Et_{t}_eV"] - float(g["params"][k].attrs["Eg"])) / meta[k]["kT_ln10"]
        else:
            s_out[k] = smp[k]
    s_out.to_csv(out / "ode_smc_samples.csv", index=False, float_format="%.8g")

    info = dict(f.attrs)
    (out / "README.txt").write_text(
        f"ODE SMC posterior of {run.name} (likelihood-tempered SMC with the legacy rate equations instead of the network,"
        f" ode_inference.ipynb; sigma of the run), exported {pd.Timestamp.now():%Y-%m-%d %H:%M} from {h5p.name}.\n"
        f"Posteriors in the file: {', '.join(methods)} (contour dash: {g.attrs['method_dash_json']}).\n"
        f"Plot coordinates (decades): {', '.join(keys)}; n1 replaces E_t: E_t - E_g = (log10 n1 - log10 N_C) kT ln 10.\n"
        f"Bins: 1D {g.attrs['bins_1d']}, 2D {g.attrs['bins_2d']} per axis over the prior box, Gaussian smoothing "
        f"{g.attrs['smooth_bins']} bins (1D: x 1.5) — the same for every posterior.\n\n"
        "ode_smc_marginals_1d.csv  <key>_x, <key>_pdf_<posterior> (normalised per unit of x); n1: <key>_Et_minus_Eg_eV\n"
        "ode_smc_contours_2d.csv   <kx>__<ky>_sigma1_x/_y, _sigma2_x/_y: 68.27 / 95.45 % HPD contour of the ODE SMC "
        "posterior in panel x = kx, y = ky; empty rows separate closed lines\n"
        "ode_smc_intervals.csv     median and HPD intervals (68 / 95 %), plot coordinates and physical units; "
        "shift_vs_NN_SMC_in_NN_sd = (median - median NN SMC) / sd NN SMC\n"
        "ode_smc_samples.csv       16384 equal-weight particles: physical units and plot coordinates\n"
        f"Run info: mode {info.get('mode')}, fill (colour map) of the h5 corner: {info.get('fill_method')}.\n")
    print(iv_tab[["parameter", "posterior", "median_plot", "HPD68_plot", "HPD95_plot"]
                 + (["shift_vs_NN_SMC_in_NN_sd"] if "SMC" in methods else [])].to_string(index=False))
    print("written:", out)


if __name__ == "__main__":
    main()
