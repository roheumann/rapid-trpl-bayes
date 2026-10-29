"""
export_convergence_origin.py — convergence curves of the start-point study with ABSOLUTE -log L, as Origin worksheets
=====================================================================================================================
21/10/2026 (Robin).  Input: <run>/start_point_study/ of scripts/start_point_study.py.

    python scripts/export_convergence_origin.py results/1trap_synthetic/1trap_85-15_film_fit_joint

convergence_physics_seeds.png shows -log L - min + 1e-3, so every converged start sits on the same 1e-3 floor and
starts that end better than the physics-informed start cannot be told apart.  Here every curve is -log L itself
(NN likelihood for CMA-ES, ODE likelihood for curve_fit), with the run's best fit as a horizontal reference line.
The values are rebuilt from the study's CSVs (histories hold -log L minus a reference; the reference is recovered
from starts.csv), nothing is recomputed.

-log L is negative near the optimum, so a log y axis is impossible; the figure uses a split y axis:
  lower row  linear y in [Y_LIN_LO, Y_SPLIT]  (the converged region, best fit, NN maximum, the plateaus)
  upper row  log y    in [Y_SPLIT, Y_LOG_HI]  (the starts and the early CMA-ES generations)
plus a third column: the final ODE -log L of every start, sorted, zoomed around the best fit.

Output in <run>/start_point_study/origin_convergence/:
  cmaes_histories.csv       per start: <label>_evals, <label>_nll_nn  (best NN -log L so far vs NN evaluations)
  curve_fit_histories.csv   per start: <label>_solves, <label>_nll_ode (first row = curve_fit start = CMA-ES result)
  starts_endpoints.csv      one row per start: colour, CMA-ES end (evals, NN -log L), curve_fit start/end (ODE),
                            delta to the best fit, rank of the final value, better_than_physics (vs the first seed)
  final_sorted.csv          final ODE -log L sorted (x = rank), split into physics seeds / Sobol columns
  reference_lines.csv       best fit (ODE) and NN maximum: -log L, for the dotted horizontal lines
  axes.json                 the y ranges of the split axis (read by build_convergence_origin.py)
  preview.png / .pdf        matplotlib version of the Origin figure
Build in Origin: corner-plot/origin/build_convergence_origin.py.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

Y_LIN_LO = -2.3        # lower row (linear): from just below the NN maximum ...
Y_SPLIT = 1.0          # ... to here; upper row (log) from here ...
Y_LOG_HI = 2000.0      # ... to here
ZOOM_PAD = 0.0005      # zoom column: from best fit - pad to the worst start that reached the best-fit basin + pad
BASIN_TOL = 0.1        # "reached the best fit": final delta <= this


def colours(labels):
    """viridis over the Sobol starts, red for every physics-informed seed (as report._start_colors)."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_hex
    cmap, n, k, out = plt.get_cmap("viridis"), max(1, len(labels) - 1), 0, []
    for lb in labels:
        if "physics" in lb:
            out.append("#D62728")
        else:
            out.append(to_hex(cmap(k / n)))
            k += 1
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("run")
    ap.add_argument("--study", default="start_point_study")
    ap.add_argument("--out", default="origin_convergence")
    ap.add_argument("--y-lin-lo", type=float, default=Y_LIN_LO, help="lower end of the linear row (other sigma -> other -log L)")
    ap.add_argument("--y-split", type=float, default=Y_SPLIT)
    ap.add_argument("--y-log-hi", type=float, default=Y_LOG_HI)
    args = ap.parse_args()
    d = Path(args.run) / args.study
    out = d / args.out
    out.mkdir(parents=True, exist_ok=True)

    st = pd.read_csv(d / "starts.csv")
    hc = pd.read_csv(d / "histories_cmaes.csv")
    ho = pd.read_csv(d / "histories_curve_fit.csv")
    summ = json.loads((d / "summary.json").read_text())

    # references used by start_point_study.py: delta columns = -log L - ref
    ref_nn = st.cmaes_neg_log_lik_nn - st.cmaes_delta_nn
    ref_ode = st.ode_end_neg_log_lik - st.delta_neg_log_lik
    assert ref_nn.std() < 1e-5 and ref_ode.std() < 1e-5, "inconsistent references in starts.csv"
    ref_nn, ref_ode = float(ref_nn.mean()), float(ref_ode.mean())
    print(f"NN maximum: -log L = {ref_nn:.6f};  best fit (ODE): -log L = {ref_ode:.6f}  "
          f"(summary.json: {summ['best_fit_of_run']['neg_log_lik_final_model']:.6f})")

    labels = list(st.label)
    cols = colours(labels)

    # ── wide histories with absolute -log L ───────────────────────────────────────
    cma = pd.DataFrame({c: (hc[c] + ref_nn if c.endswith("_delta") else hc[c]) for c in hc.columns})
    cma.columns = [c.replace("_x", "_evals").replace("_delta", "_nll_nn") if c.endswith(("_x", "_delta")) else c
                   for c in cma.columns]
    ode = pd.DataFrame({c: (ho[c] + ref_ode if c.endswith("_delta") else ho[c]) for c in ho.columns})
    ode.columns = [c[:-2] + "_solves" if c.endswith("_x") else c[:-6] + "_nll_ode" for c in ode.columns]
    cma.to_csv(out / "cmaes_histories.csv", index=False, float_format="%.10g")
    ode.to_csv(out / "curve_fit_histories.csv", index=False, float_format="%.10g")

    # ── end points ────────────────────────────────────────────────────────────────
    nll_phys = float(st.loc[st.label == "physics", "ode_end_neg_log_lik"].iloc[0])
    ep = pd.DataFrame({
        "label": labels, "is_physics_seed": st.label.str.startswith("physics"), "color": cols,
        "start_nll_nn": st.start_neg_log_lik_nn,
        "cmaes_end_evals": [cma[f"{lb}_evals"].dropna().iloc[-1] for lb in labels],
        "cmaes_end_nll_nn": st.cmaes_neg_log_lik_nn,
        "curve_fit_start_nll_ode": st.ode_start_neg_log_lik,
        "curve_fit_end_solves": [ode[f"{lb}_solves"].dropna().iloc[-1] for lb in labels],
        "curve_fit_end_nll_ode": st.ode_end_neg_log_lik,
        "delta_to_best_fit": st.delta_neg_log_lik})
    ep["rank_final"] = ep.curve_fit_end_nll_ode.rank(method="min").astype(int)
    ep["better_than_physics_seed1"] = ep.curve_fit_end_nll_ode < nll_phys
    ep.to_csv(out / "starts_endpoints.csv", index=False, float_format="%.10g")

    srt = ep.sort_values("curve_fit_end_nll_ode").reset_index(drop=True)
    srt.insert(0, "rank", np.arange(1, len(srt) + 1))
    fs = pd.DataFrame({"rank": srt["rank"], "label": srt.label, "color": srt.color,
                       "nll_ode_physics": srt.curve_fit_end_nll_ode.where(srt.is_physics_seed),
                       "nll_ode_sobol": srt.curve_fit_end_nll_ode.where(~srt.is_physics_seed),
                       "nll_ode_physics_seed1": srt.curve_fit_end_nll_ode.where(srt.label == "physics")})
    fs.to_csv(out / "final_sorted.csv", index=False, float_format="%.10g")
    pd.DataFrame({"name": ["best fit (ODE)", "NN maximum"], "neg_log_lik": [ref_ode, ref_nn]}).to_csv(
        out / "reference_lines.csv", index=False, float_format="%.10g")

    basin = ep.curve_fit_end_nll_ode[ep.delta_to_best_fit <= BASIN_TOL]
    n_basin = int(len(basin))
    axes = {"y_lin": [args.y_lin_lo, args.y_split], "y_log": [args.y_split, args.y_log_hi],
            "y_zoom": [ref_ode - ZOOM_PAD, float(basin.max()) + ZOOM_PAD], "zoom_n": n_basin,
            "x_cmaes": [6, 1e4], "x_curve_fit": [-10, float(np.ceil(ep.curve_fit_end_solves.max() / 50) * 50) + 10],
            "best_fit_nll_ode": ref_ode, "nn_max_nll_nn": ref_nn, "physics_seed1_nll_ode": nll_phys,
            "n_starts": len(ep), "n_physics_seeds": int(ep.is_physics_seed.sum()),
            "n_better_than_physics_seed1": int(ep.better_than_physics_seed1.sum()),
            "n_sobol_better_than_physics_seed1": int((ep.better_than_physics_seed1 & ~ep.is_physics_seed).sum())}
    (out / "axes.json").write_text(json.dumps(axes, indent=1))
    print(json.dumps({k: v for k, v in axes.items() if k.startswith(("n_", "physics", "zoom"))}, indent=1))

    preview(cma, ode, ep, fs, axes, labels, out)
    (out / "README.txt").write_text(__doc__)


def preview(cma, ode, ep, fs, ax_, labels, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    fig, A = plt.subplots(2, 3, figsize=(13, 5.6), sharey="row",
                          gridspec_kw={"width_ratios": [2, 1, 1.1], "height_ratios": [1, 1.6], "hspace": 0.0,
                                       "wspace": 0.05})
    for c in range(3):
        if c == 2:
            continue
        A[0, c].set_yscale("log"); A[0, c].set_ylim(ax_["y_log"][0] * 1.001, ax_["y_log"][1])
        A[1, c].set_ylim(*ax_["y_lin"])
    order = sorted(range(len(labels)), key=lambda i: "physics" in labels[i])     # red on top
    for i in order:
        lb, col = labels[i], ep.color[i]
        hl = "physics" in lb
        lw, ms, z = (2.0, 7, 5) if hl else (0.8, 4, 2)
        x, y = cma[f"{lb}_evals"].dropna(), cma[f"{lb}_nll_nn"].dropna()
        xo, yo = ode[f"{lb}_solves"].dropna(), ode[f"{lb}_nll_ode"].dropna()
        for r in range(2):
            A[r, 0].plot(x, y, "-", color=col, lw=lw, zorder=z)
            A[r, 0].plot(x.iloc[-1:], y.iloc[-1:], "o", mfc=col, mec="k", ms=ms, zorder=z + 1)
            A[r, 1].plot(xo, yo, "--", color=col, lw=lw, zorder=z)
            A[r, 1].plot(xo.iloc[:1], yo.iloc[:1], "o", mfc=col, mec="k", ms=ms, zorder=z + 1)
            A[r, 1].plot(xo.iloc[-1:], yo.iloc[-1:], "*", color=col, mec="k", mew=0.4, ms=ms + 4, zorder=z + 2)
    for r in range(2):
        A[r, 0].set_xscale("log"); A[r, 0].set_xlim(*ax_["x_cmaes"])
        A[r, 1].set_xlim(*ax_["x_curve_fit"]); A[r, 1].axvline(0, color="k", lw=1)
        for c in range(2):
            A[r, c].grid(alpha=0.3, which="both")
    for c in range(2):
        A[1, c].axhline(ax_["best_fit_nll_ode"], color="k", ls=":", lw=1.4, zorder=10)
        A[0, c].tick_params(labelbottom=False)
        A[0, c].spines["bottom"].set_linestyle((0, (4, 3))); A[1, c].spines["top"].set_linestyle((0, (4, 3)))
    A[1, 0].axhline(ax_["nn_max_nll_nn"], color="0.45", ls=":", lw=1.2, zorder=10)
    A[1, 0].set_xlabel("NN evaluations (CMA-ES)"); A[1, 1].set_xlabel("ODE solves (curve_fit)")
    A[1, 0].set_ylabel(r"$-\log L$  (linear)"); A[0, 0].set_ylabel(r"$-\log L$  (log)")

    # zoom column: final ODE -log L of every start, sorted
    A[0, 2].axis("off")
    z = A[1, 2]
    zz = z.inset_axes([0, 0, 1, 1]); z.axis("off")
    n = ax_["zoom_n"]
    f = fs.iloc[:n]
    zz.scatter(f["rank"], f.nll_ode_sobol, c=f.color, ec="k", lw=0.4, s=30, zorder=3)
    zz.scatter(f["rank"], f.nll_ode_physics, c="#D62728", ec="k", lw=0.4, s=45, zorder=4)
    zz.scatter(f["rank"], f.nll_ode_physics_seed1, facecolors="none", ec="k", lw=1.4, s=110, zorder=5)
    zz.axhline(ax_["best_fit_nll_ode"], color="k", ls=":", lw=1.4)
    zz.axhline(ax_["physics_seed1_nll_ode"], color="#D62728", ls=":", lw=1.0)
    zz.set_ylim(*ax_["y_zoom"]); zz.set_xlim(0, n + 1)
    zz.yaxis.tick_right(); zz.yaxis.set_label_position("right")
    zz.set_ylabel(r"final $-\log L$ (ODE), zoom"); zz.set_xlabel(f"rank (the {n} starts within {BASIN_TOL} of the best fit)")
    zz.grid(alpha=0.3)
    zz.ticklabel_format(axis="y", useOffset=False)

    h = [Line2D([0], [0], color="0.4", lw=1.5, label="CMA-ES (NN likelihood)"),
         Line2D([0], [0], marker="o", color="w", mfc="0.6", mec="k", label="switch to curve_fit"),
         Line2D([0], [0], color="0.4", lw=1.5, ls="--", label="curve_fit (ODE likelihood)"),
         Line2D([0], [0], marker="*", color="w", mfc="0.6", mec="k", ms=10, label="end"),
         Line2D([0], [0], color="#D62728", lw=2, label="physics-informed start (16 CMA-ES seeds)"),
         Line2D([0], [0], marker="o", color="w", mfc="none", mec="k", mew=1.4, ms=9, label="physics start, seed 1"),
         Line2D([0], [0], color="k", ls=":", lw=1.4, label=f"best fit (ODE), {ax_['best_fit_nll_ode']:.4f}"),
         Line2D([0], [0], color="0.45", ls=":", lw=1.2, label=f"NN maximum, {ax_['nn_max_nll_nn']:.4f}")]
    A[0, 2].legend(handles=h, fontsize=8, loc="center")
    fig.suptitle(f"{ax_['n_physics_seeds']} CMA-ES seeds of the physics-informed start (red) + "
                 f"{ax_['n_starts'] - ax_['n_physics_seeds']} Sobol starts: absolute $-\\log L$")
    fig.savefig(out / "preview.png", dpi=150, bbox_inches="tight"); fig.savefig(out / "preview.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
