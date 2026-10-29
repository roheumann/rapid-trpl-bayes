"""
export_error_histogram_origin.py — histogram of -log L after N CMA-ES evaluations, physics-informed vs random starts
====================================================================================================================
21/10/2026 (Robin).  Input: <run>/start_point_study/origin_convergence/ (scripts/export_convergence_origin.py).

    python scripts/export_error_histogram_origin.py results/1trap_synthetic/1trap_85-15_film_fit_joint \
        [--evals 1000] [--edges -2.24 -2.23 -2.0 -1.6 -1.5 20 30 60]

For every start: the best NN -log L that CMA-ES has found after `--evals` NN evaluations (absolute, not relative to
the best fit; the last generation with evals <= N).  Groups: the 16 CMA-ES seeds of the physics-informed start (red)
and the 64 Sobol starts (orange).  The bins are not equal-width: the values form separate plateaus (NN maximum
-2.237, the low-N_t plateaus -1.64 ... -1.54, the far plateaus 21 ... 24 and 50), so the edges lie between them.
Heights: percent of each group (16 vs 64 starts), counts in the CSV as well.

Output in <run>/start_point_study/origin_convergence/:
  error_histogram_<N>evals.csv         one row per bin: label, edges, counts, percent per group
  error_values_<N>evals.csv            one row per start: label, group, -log L (NN) after N evaluations, bin
  error_histogram_<N>evals_preview.png matplotlib version of the Origin figure
Build in Origin: corner-plot/origin/build_error_histogram.py.

--log-delta (23/10/2026, Robin): x = Delta = -log L - NN maximum (reference_lines.csv) on a logarithmic axis instead.
First bin [0, DELTA_MIN) = converged (0.02 = the narrowest bin of the linear histogram), then --per-decade log-equal
bins up to the largest Delta.  Files with the suffix _logdelta (same columns + geo_center); the Origin builder puts
the bin index on x, which for log-equal bins is a log axis.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

EDGES = [-2.24, -2.23, -2.0, -1.6, -1.5, 20, 30, 60]
COLORS = {"physics": "#D62728", "random": "#FF7F0E"}


def fmt(v):
    return f"{v:g}".replace("-", "−")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("run")
    ap.add_argument("--study", default="start_point_study/origin_convergence")
    ap.add_argument("--evals", type=int, default=1000)
    ap.add_argument("--edges", type=float, nargs="+", default=EDGES)
    ap.add_argument("--log-delta", action="store_true", help="Delta to the NN maximum on log-equal bins")
    ap.add_argument("--delta-min", type=float, default=0.02, help="upper edge of the converged bin = lower end of the log axis")
    ap.add_argument("--per-decade", type=int, default=2)
    ap.add_argument("--delta-ref", type=float, default=None, help="reference -log L (default: NN maximum, reference_lines.csv)")
    args = ap.parse_args()
    d = Path(args.run) / args.study
    cma = pd.read_csv(d / "cmaes_histories.csv")
    ep = pd.read_csv(d / "starts_endpoints.csv")
    N, edges = args.evals, np.asarray(args.edges, float)

    rows = []
    for lb in ep.label:
        x, y = cma[f"{lb}_evals"].dropna().values, cma[f"{lb}_nll_nn"].dropna().values
        k = np.searchsorted(x, N, side="right") - 1
        if k < 0:
            raise SystemExit(f"{lb}: no generation with <= {N} evaluations")
        if x[-1] < N:
            print(f"note: {lb} stopped at {x[-1]:.0f} evaluations; its final value is used")
        rows.append({"label": lb, "group": "physics" if lb.startswith("physics") else "random",
                     "evals_used": x[k], "neg_log_lik_nn": y[k]})
    v = pd.DataFrame(rows)
    suf, xcol = "", "neg_log_lik_nn"
    if args.log_delta:
        ref = args.delta_ref
        if ref is None:
            rl = pd.read_csv(d / "reference_lines.csv")
            ref = float(rl.loc[rl.name == "NN maximum", "neg_log_lik"].iloc[0])
        v["delta"] = v.neg_log_lik_nn - ref
        n_neg = int((v.delta < 0).sum())
        print(f"reference (NN maximum) {ref:.6f}; {n_neg} values below it (min {v.delta.min():.2e}) clipped to 0")
        v["delta"] = v.delta.clip(lower=0.0)
        k_max = int(np.ceil(args.per_decade * np.log10(max(v.delta.max(), args.delta_min) / args.delta_min) + 1e-9))
        edges = np.r_[0.0, args.delta_min * 10 ** (np.arange(k_max + 1) / args.per_decade)]
        if edges[-1] <= v.delta.max():
            edges = np.r_[edges, args.delta_min * 10 ** ((k_max + 1) / args.per_decade)]
        suf, xcol = "_logdelta", "delta"
    out_of = (v[xcol] < edges[0]) | (v[xcol] >= edges[-1])
    if out_of.any():
        raise SystemExit(f"values outside the edges: {v[out_of].to_string()}")
    v["bin"] = np.searchsorted(edges, v[xcol], side="right") - 1
    v.to_csv(d / f"error_values_{N}evals{suf}.csv", index=False, float_format="%.10g")

    n_g = v.group.value_counts()
    h = pd.DataFrame({"bin": np.arange(len(edges) - 1), "lo": edges[:-1], "hi": edges[1:]})
    h["label"] = [f"[{fmt(a)}, {fmt(b)})" for a, b in zip(h.lo, h.hi)]
    if args.log_delta:
        h["label"] = [f"< {fmt(args.delta_min)} (converged)"] + [f"[{a:.3g}, {b:.3g})" for a, b in zip(h.lo[1:], h.hi[1:])]
        h["geo_center"] = np.r_[np.nan, np.sqrt(h.lo[1:] * h.hi[1:])]
    for g in ("physics", "random"):
        c = v[v.group == g].bin.value_counts().reindex(h.bin, fill_value=0).values
        h[f"count_{g}"] = c
        h[f"percent_{g}"] = 100 * c / n_g[g]
    h = h[["bin", "label", "lo", "hi", "count_physics", "count_random", "percent_physics", "percent_random"]
          + (["geo_center"] if args.log_delta else [])]
    h.to_csv(d / f"error_histogram_{N}evals{suf}.csv", index=False, float_format="%.6g", encoding="utf-8")
    print(f"{n_g['physics']} physics-informed, {n_g['random']} random starts; -log L (NN) after {N} evaluations")
    print(h.to_string(index=False))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if args.log_delta:
        fig, ax = plt.subplots(figsize=(7, 3.8))
        lo = np.r_[args.delta_min / 10, h.lo[1:]]                       # converged bin drawn as [delta_min/10, delta_min)
        hi = h.hi.to_numpy(float)
        mid = np.sqrt(lo * hi)
        for (a, b), g, name in (((lo, mid), "physics", f"physics-informed start ({n_g['physics']} CMA-ES seeds)"),
                                ((mid, hi), "random", f"random (Sobol) starts ({n_g['random']})")):
            bars = ax.bar(a, h[f"percent_{g}"], width=b - a, align="edge", color=COLORS[g], ec="k", lw=0.6, label=name)
            ax.bar_label(bars, labels=[str(c) if c else "" for c in h[f"count_{g}"]], fontsize=7, padding=1)
        ax.set_xscale("log")
        ax.axvline(args.delta_min, color="0.4", ls=":", lw=0.8)
        ax.set_xlim(lo[0], hi[-1])
        dec = 10.0 ** np.arange(np.ceil(np.log10(args.delta_min)), np.floor(np.log10(hi[-1])) + 1)
        ax.set_xticks(np.r_[np.sqrt(lo[0] * hi[0]), dec],
                      [f"< {fmt(args.delta_min)}\n(converged)"] + [f"$10^{{{int(np.log10(x))}}}$" for x in dec])
        ax.set_xticks([x for x in ax.get_xticks(minor=True) if x >= args.delta_min], minor=True)
        ax.set_xlim(lo[0], hi[-1])
        ax.set_xlabel(f"$\\Delta(-\\log L)$ to the NN maximum after {N} CMA-ES evaluations")
        ax.set_ylabel("starts of the group (%)")
        ax.legend(fontsize=8, frameon=False)
        ax.set_title("numbers on the bars: counts", fontsize=8, loc="right")
        fig.tight_layout()
        fig.savefig(d / f"error_histogram_{N}evals{suf}_preview.png", dpi=150)
        return
    fig, ax = plt.subplots(figsize=(7, 3.8))
    i, w = np.arange(len(h)), 0.38
    for off, g, name in ((-w / 2, "physics", f"physics-informed start ({n_g['physics']} CMA-ES seeds)"),
                         (w / 2, "random", f"random (Sobol) starts ({n_g['random']})")):
        b = ax.bar(i + off, h[f"percent_{g}"], w, color=COLORS[g], ec="k", lw=0.6, label=name)
        ax.bar_label(b, labels=[str(c) if c else "" for c in h[f"count_{g}"]], fontsize=7, padding=1)
    ax.set_xticks(i, h.label, rotation=30, ha="right")
    ax.set_xlabel(f"$-\\log L$ (NN) after {N} CMA-ES evaluations")
    ax.set_ylabel("starts of the group (%)")
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("numbers on the bars: counts", fontsize=8, loc="right")
    fig.tight_layout()
    fig.savefig(d / f"error_histogram_{N}evals_preview.png", dpi=150)


if __name__ == "__main__":
    main()
