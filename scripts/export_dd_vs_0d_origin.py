"""
export_dd_vs_0d_origin.py — drift-diffusion vs 0D tau_diff(QFLS) at ONE alpha, as CSV for Origin
==================================================================================================
27/10/2026 (Robin).  Input: a sigma_model folder of model-error-analysis/sigma_model.py, e.g. the chosen
85:15 run (reflection-corrected alpha 5.07e4 /cm, results/2trap_85-15_film/sigma_model/chosen.json):

    python scripts/export_dd_vs_0d_origin.py \
        results/2trap_85-15_film/sigma_model/stageA_v3_best_fit/reflection_corrected [--label best_fit]

The figure is figures/dd_vs_0d.png of that folder with only the chosen alpha: top = data, 0D rate equations and
drift-diffusion at alpha (tau_diff in s, for a log axis); bottom = residuals DD - 0D (log10 tau_diff) at the spline
points, RMSE = sigma_model.  The alpha_half / alpha_double sensitivity runs are left out.
--raw (28/10/2026, Robin): the data shown are the RAW measured tau_diff vs QFLS points (default: the 90 points of
the raw-tr-PL run, results/2trap_85-15_film/2trap_85-15_film_joint/trpl_used.csv) instead of the
256 spline points; the residuals stay at the spline points (that is how sigma_model is defined).
Other films: --raw <run>/trpl_raw_points.csv from scripts/raw_trpl_points.py <run>.

Output in <folder>/origin_dd_vs_0d/:
  data_points.csv  qfls_eV, log10_tau_data, log10_tau_0d, log10_tau_dd, delta_dd_minus_0d   (spline points)
  raw_points.csv   qfls_eV, tau_diff_s       (raw measured points, shown in the top panel)
  curve_0d.csv     qfls_eV, log10_tau_0d, tau_diff_0d_s   (0D curve, cut to the measured QFLS range)
  curve_dd.csv     qfls_eV, log10_tau_dd, tau_diff_dd_s   (drift-diffusion curve, same cut)
  axes.json        alpha, sigma_model, axis ranges and tick steps (read by the Origin builder)
  preview.png      matplotlib version of the Origin figure
Build in Origin: corner-plot/origin/build_dd_vs_0d.py.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


RAW_85_15 = Path(__file__).resolve().parent.parent / \
    "results/2trap_85-15_film/2trap_85-15_film_joint/trpl_used.csv"


def strictly_rising(c, dq_min=1e-5):
    """Drop curve points less than dq_min eV above the last kept QFLS.  At the start of the transient (top QFLS end)
    PL ~ PL_max, so the QFLS stops moving while tau_diff still changes: a vertical line (77:23 F6, 25 points within
    1e-7 eV of 1.5035 eV, tau_diff 4.5e-9 -> 7e-10 s)."""
    q, keep, last = c.qfls_eV.to_numpy(float), [], -np.inf
    for v in q:
        keep.append(v - last >= dq_min)
        if keep[-1]:
            last = v
    return c[np.array(keep, bool)]


def nice_range(lo, hi, step):
    return float(np.floor(lo / step) * step), float(np.ceil(hi / step) * step)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("folder", help="sigma_model output folder (dd_curve_*.csv, ode0d_curve_*.csv, summary.json)")
    ap.add_argument("--label", default="best_fit", help="which run of the folder (best_fit = the chosen alpha)")
    ap.add_argument("--raw", default=str(RAW_85_15), help="raw tau_diff vs QFLS points (columns qfls, tau_diff)")
    ap.add_argument("--no-clip-to-data", dest="clip_to_data", action="store_false",
                    help="draw the model curves over the whole x range, not only the measured QFLS range")
    args = ap.parse_args()
    d = Path(args.folder)
    out = d / "origin_dd_vs_0d"
    out.mkdir(exist_ok=True)

    pts = pd.read_csv(d / f"dd_vs_0d_{args.label}_data_points.csv")
    c0 = pd.read_csv(d / f"ode0d_curve_{args.label}.csv")
    cd = pd.read_csv(d / f"dd_curve_{args.label}.csv")
    summ = json.loads((d / "summary.json").read_text())
    run = next(r for r in summ["dd_vs_0d"] if r["label"] == args.label)
    alpha, rmse = float(run["alpha_cm"]), float(run["rmse_data_points"])

    raw = pd.read_csv(args.raw)
    raw = pd.DataFrame({"qfls_eV": raw.qfls, "tau_diff_s": raw.tau_diff})
    raw = raw[np.isfinite(raw.tau_diff_s) & (raw.tau_diff_s > 0)].sort_values("qfls_eV")
    q_all = np.r_[pts.qfls_eV, raw.qfls_eV]
    x = (float(q_all.min()) - 0.02, float(q_all.max()) + 0.02)                   # as the matplotlib figure
    # model curves only over the measured QFLS range: below it the simulated PL reaches its decay cap and the 0D
    # tau_diff jumps (77:23 F6 inference, 0.965 eV)
    qc = (float(q_all.min()), float(q_all.max())) if args.clip_to_data else x
    c0 = c0[(c0.qfls_eV >= qc[0]) & (c0.qfls_eV <= qc[1])].assign(tau_diff_0d_s=lambda f: 10 ** f.log10_tau_0d)
    cd = cd[(cd.qfls_eV >= qc[0]) & (cd.qfls_eV <= qc[1])].assign(tau_diff_dd_s=lambda f: 10 ** f.log10_tau_dd)
    c0, cd = strictly_rising(c0), strictly_rising(cd)
    ys = np.concatenate([np.log10(raw.tau_diff_s), c0.log10_tau_0d, cd.log10_tau_dd])
    y = nice_range(np.nanmin(ys), np.nanmax(ys), 1.0)                             # whole decades (log axis)
    dl = pts.delta_dd_minus_0d.dropna()
    dstep = 0.05 if dl.abs().max() < 0.25 else 0.1
    dy = nice_range(min(dl.min(), 0.0) - 0.2 * dstep, max(dl.max(), 0.0) + 0.2 * dstep, dstep)

    pts[["qfls_eV", "log10_tau_data", "log10_tau_0d", "log10_tau_dd", "delta_dd_minus_0d"]].to_csv(
        out / "data_points.csv", index=False, float_format="%.10g")
    raw.to_csv(out / "raw_points.csv", index=False, float_format="%.10g")
    c0.to_csv(out / "curve_0d.csv", index=False, float_format="%.10g")
    cd.to_csv(out / "curve_dd.csv", index=False, float_format="%.10g")
    axes = {"source": str(d.resolve()), "label": args.label, "alpha_cm": alpha, "sigma_model_dec": rmse,
            "n_points_compared": int(run["n_data_points_compared"]), "n_points": int(run["n_data_points"]),
            "best_fit": summ.get("best_fit"), "npulse_cm3": summ.get("npulse_cm3"),
            "raw_points": str(Path(args.raw).resolve()), "n_raw_points": int(len(raw)),
            "x": [round(v, 4) for v in x], "x_inc": 0.1, "y_log10": list(y),
            "delta_y": list(dy), "delta_inc": 2 * dstep}
    (out / "axes.json").write_text(json.dumps(axes, indent=2))
    print(json.dumps(axes, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1, figsize=(6, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1.3]})
    ax[0].semilogy(raw.qfls_eV, raw.tau_diff_s, "o", ms=3.5, mfc="none", color="0.4", label="data")
    ax[0].semilogy(c0.qfls_eV, c0.tau_diff_0d_s, "-", color="#00CED1", label="0D rate equations")
    ax[0].semilogy(cd.qfls_eV, cd.tau_diff_dd_s, "--", color="#D62728",
                   label=f"drift-diffusion, $\\alpha$ = {alpha:.3g} cm$^{{-1}}$")
    ax[0].set(xlim=x, ylim=(10 ** y[0], 10 ** y[1]), ylabel=r"$\tau_{diff}$ (s)")
    ax[0].legend(fontsize=8, frameon=False)
    ax[1].plot(pts.qfls_eV, pts.delta_dd_minus_0d, "-", color="#D62728", lw=1)
    ax[1].axhline(0, color="k", lw=0.5, ls=":")
    ax[1].set(ylim=dy, xlabel=r"Fermi-level splitting $\Delta E_F$ (eV)", ylabel="Residuals (dec)")
    ax[1].text(0.98, 0.08, f"RMSE = {rmse:.3f} dec", transform=ax[1].transAxes, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "preview.png", dpi=150)


if __name__ == "__main__":
    main()
