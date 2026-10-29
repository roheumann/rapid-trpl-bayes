"""
sbc_compare.py — SBC variants A (NN data) and B (ODE data) side by side, all modes, summary table + figures
==========================================================================================================
19/10/2026 (revision).  Reads the outputs of ``scripts/sbc_smc.py``:

    <base>_<mode>/sbc_smc/          variant A: NN data  -> NN-SMC (computational correctness of sampler + likelihood + prior)
    <base>_<mode>/sbc_smc_odedata/  variant B: ODE data -> NN-SMC (same theta_true, noise and SMC seeds; can the NN
                                    stand in for the legacy ODE?)

and writes, for every mode × quantity (free parameters and derived quantities):

* the summary CSV (``--csv``): chi^2 p and simultaneous-ECDF p of the ranks (A, B), mean relative rank, coverage of
  the central 50/68/95 % intervals with binomial SE, median contraction 1 - var_post/var_prior, z-score mean/sd,
  fraction of data sets with max R-hat > 1.01, median Kish ESS, and the paired A-vs-B comparison (mean rank
  difference, Wilcoxon signed-rank p, chi^2 homogeneity p, posterior-mean shift in posterior sd);
  rows ``subset = all`` always; rows ``subset = max_r_hat<=1.01`` in addition when some data sets exceed 1.01
  (nothing is dropped silently);
* figures (``--out``/figures): ECDF-difference panels with simultaneous 95 % bands (and pointwise), rank histograms,
  coverage, z-score vs contraction — A and B overlaid in the same per-quantity panels — and paired rank scatter;
  an overview figure (rows = modes, columns = parameters);
* the curves behind the ECDF figures as CSV (for Origin), a summary.json, and a markdown block for STATUS.md
  (``--status``): min p per mode, quantities whose ranks differ between A and B, convergence-failure fractions.

Usage: python scripts/sbc_compare.py --base results/1trap_synthetic_paper_B2nf_kradfixed --out results/sbc_1trap_B2nf
                                     --csv results/sbc_1trap_B2nf_summary.csv [--status results/logs/.../STATUS.md]
                                     [--run trpl=<run folder> ...]   (a mode whose run is not <base>_<mode>)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MODES = ("trpl", "joint", "sspl")
VARIANTS = {"A": ("sbc_smc", "A: NN data"), "B": ("sbc_smc_odedata", "B: ODE data")}
COLORS = {"A: NN data": "C0", "B: ODE data": "C3"}
RHAT = 1.01


def load_result(folder: Path):
    """SBCResult from the files written by scripts/sbc_smc.py (theta_true in display units)."""
    from inference.sbc import SBCResult
    s = json.loads((folder / "summary.json").read_text())
    rk = pd.read_csv(folder / "ranks.csv")
    names = [c[len("rank_"):] for c in rk.columns if c.startswith("rank_")]
    post = pd.read_csv(folder / "posterior.csv") if (folder / "posterior.csv").exists() else pd.DataFrame()
    info = dict(s["info"])
    info["summary"] = s
    return SBCResult(theta_true=rk[[f"true_{n}" for n in names]].to_numpy(float),
                     ranks=rk[[f"rank_{n}" for n in names]].to_numpy(float),
                     ranks_derived=pd.read_csv(folder / "ranks_derived.csv"), n_keep=int(info["n_keep"]), names=names,
                     diagnostics=pd.read_csv(folder / "diagnostics.csv"), info=info, posterior=post)


def variant_table(res, prefix: str) -> pd.DataFrame:
    """uniformity + calibration of one SBC run, columns suffixed with _A / _B."""
    from inference import sbc
    u = sbc.uniformity_table(res).rename(columns={"p_value": "chi2_p", "mean_rank_rel": "mean_rel_rank"})
    c = sbc.calibration_table(res)
    t = u[["n", "chi2_p", "p_ecdf_sim", "mean_rel_rank"]].join(c.drop(columns=["n"]), how="left")
    dg = res.diagnostics
    t["frac_rhat_gt_1.01"] = float((dg.max_r_hat > RHAT).mean())
    t["frac_rhat_gt_1.05"] = float((dg.max_r_hat > 1.05).mean())
    t["median_ess_kish"] = float(dg.ess_kish.median())
    if "redrawn" in dg:
        t["n_redrawn"] = int(dg.redrawn.sum())
    if "rms_nn_minus_data_model" in dg:
        t["nn_minus_ode_rms_median"] = float(dg.rms_nn_minus_data_model.median())
    return t.add_suffix(f"_{prefix}")


def main(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from inference import report, sbc

    out = Path(a.out)
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    res = {}                                          # (mode, variant) -> SBCResult
    run_dirs = dict(r.split("=", 1) for r in a.run or [])  # --run MODE=PATH: a mode's run folder outside <base>_<mode>
    for mode in MODES:
        for v, (folder, _) in VARIANTS.items():
            p = Path(run_dirs.get(mode, f"{a.base}_{mode}")) / folder
            if (p / "summary.json").exists():
                res[(mode, v)] = load_result(p)
    if not res:
        raise SystemExit(f"no SBC results found for {a.base}_<mode>/")
    modes = [m for m in MODES if (m, "A") in res or (m, "B") in res]

    rows, paired_all, curves, bands, status = [], [], [], [], []
    for mode in modes:
        runs = {VARIANTS[v][1]: res[(mode, v)] for v in VARIANTS if (mode, v) in res}
        first = next(iter(runs.values()))
        free = [n for i, n in enumerate(first.names) if np.isfinite(first.ranks[:, i]).any()]
        derived = list(first.ranks_derived.columns)
        # ---- subsets: all data sets, and (if some exceed it) only data sets with max R-hat <= 1.01
        subsets = {"all": {lab: r for lab, r in runs.items()}}
        if any((r.diagnostics.max_r_hat > RHAT).any() for r in runs.values()):
            subsets[f"max_r_hat<={RHAT}"] = {lab: sbc.subset(r, r.diagnostics.max_r_hat <= RHAT) for lab, r in runs.items()}
        for sub, rr in subsets.items():
            tabs = [variant_table(r, v) for v in VARIANTS for lab, r in rr.items() if lab == VARIANTS[v][1]]
            t = pd.concat(tabs, axis=1)
            if (mode, "A") in res and (mode, "B") in res and sub == "all":
                pc = sbc.paired_comparison(res[(mode, "A")], res[(mode, "B")])
                n_q = len(pc)
                pc["ranks_differ_bonferroni"] = (pc.wilcoxon_p < 0.05 / n_q) | (pc.homogeneity_chi2_p < 0.05 / n_q)
                pc["ranks_differ_p05"] = (pc.wilcoxon_p < 0.05) | (pc.homogeneity_chi2_p < 0.05)
                t = t.join(pc, how="left")
                paired_all.append(pc.assign(mode=mode))
            t.insert(0, "kind", ["parameter" if q in free else "derived" for q in t.index])
            t.insert(0, "subset", sub)
            t.insert(0, "mode", mode)
            rows.append(t.reset_index().rename(columns={"index": "quantity"}))

        # ---- figures: A and B overlaid in the same per-quantity panels
        tag = f"{mode}"
        for names, nm in ((free, "parameters"), (derived, "derived")):
            for band in ("simultaneous", "pointwise"):
                fig = sbc.plot_ecdf_panels(runs, names, band=band, colors=COLORS,
                                           title=f"{mode}: SBC ECDF difference, grey = 95 % {band} band (n = 200, L = 100)")
                sfx = "" if band == "simultaneous" else "_pointwise"
                report.save_figure(fig, fig_dir / f"{tag}_ecdf_{nm}{sfx}.png")
                if band == "simultaneous":
                    fig.savefig(fig_dir / f"{tag}_ecdf_{nm}.pdf")
                plt.close(fig)
            fig = sbc.plot_rank_histogram_panels(runs, names, colors=COLORS, title=f"{mode}: SBC rank histograms (20 bins)")
            report.save_figure(fig, fig_dir / f"{tag}_rank_histograms_{nm}.png"); plt.close(fig)
            fig = sbc.plot_coverage(runs, names, colors=COLORS, title=f"{mode}: coverage of central intervals (+-2 SE)")
            report.save_figure(fig, fig_dir / f"{tag}_coverage_{nm}.png"); plt.close(fig)
            fig = sbc.plot_zscore_contraction(runs, names, colors=COLORS, title=f"{mode}: posterior z-score vs contraction")
            report.save_figure(fig, fig_dir / f"{tag}_zscore_contraction_{nm}.png"); plt.close(fig)
        if (mode, "A") in res and (mode, "B") in res:
            ra, rb = sbc.rank_columns(res[(mode, "A")]), sbc.rank_columns(res[(mode, "B")])
            fig, axes = sbc._panel_grid(len(free) + len(derived))
            for ax, q in zip(axes, free + derived):
                ax.plot([0, 100], [0, 100], color="0.6", lw=0.8)
                ax.scatter(ra[q], rb[q], s=6, alpha=0.6, lw=0)
                ax.set_title(q, fontsize=9); ax.tick_params(labelsize=7)
                ax.set_xlabel("rank A (NN data)", fontsize=8); ax.set_ylabel("rank B (ODE data)", fontsize=8)
            fig.suptitle(f"{mode}: paired ranks, same theta_true, noise and SMC seeds", fontsize=10)
            fig.tight_layout(); report.save_figure(fig, fig_dir / f"{tag}_paired_ranks.png"); plt.close(fig)
            pa, pb = res[(mode, "A")].posterior, res[(mode, "B")].posterior
            fig, axes = sbc._panel_grid(len(free) + len(derived))
            for ax, q in zip(axes, free + derived):
                sh = (pb[f"mean_{q}"] - pa[f"mean_{q}"]) / pa[f"sd_{q}"]
                ax.hist(sh, bins=30, color="C3")
                ax.axvline(0, color="k", lw=0.7)
                ax.set_title(q, fontsize=9); ax.tick_params(labelsize=7)
                ax.set_xlabel("(mean_B - mean_A) / sd_A", fontsize=8)
            fig.suptitle(f"{mode}: shift of the posterior mean, ODE data vs NN data, in posterior sd", fontsize=10)
            fig.tight_layout(); report.save_figure(fig, fig_dir / f"{tag}_paired_mean_shift.png"); plt.close(fig)
        # ---- curves for Origin
        for lab, r in runs.items():
            for q, rk in sbc.rank_columns(r).items():
                z, e = sbc.ecdf_difference(rk, r.n_keep)
                curves.append(pd.DataFrame({"mode": mode, "variant": lab, "quantity": q, "z": z, "ecdf_minus_uniform": e}))
        n0, L0 = len(first.ranks), first.n_keep
        for band in ("simultaneous", "pointwise"):
            b = sbc.ecdf_band(n0, L0, method=band)
            bands.append(pd.DataFrame({"mode": mode, "n": n0, "band": band, "z": b["z"], "lo": b["lo"], "hi": b["hi"],
                                       "gamma": b["gamma"]}))

    tab = pd.concat(rows, ignore_index=True)
    tab.to_csv(a.csv, index=False)
    pd.concat(curves).to_csv(out / "ecdf_curves.csv", index=False)
    pd.concat(bands).drop_duplicates().to_csv(out / "ecdf_bands.csv", index=False)
    if paired_all:
        pd.concat(paired_all).to_csv(out / "paired_A_vs_B.csv")

    # ---- overview figure: rows = modes, columns = free parameters, A and B overlaid, simultaneous band
    free0 = [n for i, n in enumerate(next(iter(res.values())).names) if np.isfinite(next(iter(res.values())).ranks[:, i]).any()]
    fig, axes = plt.subplots(len(modes), len(free0), figsize=(3.0 * len(free0), 2.5 * len(modes)), squeeze=False,
                             sharex=True, sharey=True)
    for i, mode in enumerate(modes):
        for j, q in enumerate(free0):
            ax = axes[i, j]
            for v in VARIANTS:
                if (mode, v) not in res:
                    continue
                r = res[(mode, v)]
                rk = sbc.rank_columns(r)[q]
                if v == "A" or (mode, "A") not in res:
                    b = sbc.ecdf_band(int(np.isfinite(rk).sum()), r.n_keep)
                    ax.fill_between(b["z"], b["lo"], b["hi"], color="0.85", step="mid", lw=0)
                z, e = sbc.ecdf_difference(rk, r.n_keep)
                ax.plot(z, e, color=COLORS[VARIANTS[v][1]], lw=1.3, label=VARIANTS[v][1])
            ax.axhline(0, color="k", lw=0.6)
            ax.grid(alpha=0.3); ax.tick_params(labelsize=7)
            if i == 0:
                ax.set_title(q, fontsize=9)
            if j == 0:
                ax.set_ylabel(f"{mode}\nECDF - uniform", fontsize=8)
            if i == len(modes) - 1:
                ax.set_xlabel("normalised rank", fontsize=8)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("SBC, synthetic 1-trap case (B2nf, k_rad fixed): grey = simultaneous 95 % band", fontsize=10)
    fig.tight_layout()
    report.save_figure(fig, fig_dir / "overview_ecdf_parameters.png"); fig.savefig(fig_dir / "overview_ecdf_parameters.pdf")
    plt.close(fig)

    # ---- STATUS block
    allrows = tab[tab.subset == "all"]
    L = ["", "## SBC summary (scripts/sbc_compare.py)", f"- table: `{a.csv}`; figures: `{fig_dir}`", ""]
    L.append("**Min p-values per mode** (free parameters; in brackets incl. derived quantities):")
    L.append("")
    L.append("| mode | variant | min χ² p | min sim.-ECDF p | R-hat > 1.01 | R-hat > 1.05 | median Kish ESS | redrawn |")
    L.append("|---|---|---|---|---|---|---|---|")
    for mode in modes:
        t = allrows[allrows["mode"] == mode]
        for v in VARIANTS:
            if f"chi2_p_{v}" not in t or t[f"chi2_p_{v}"].isna().all():
                continue
            par = t[t.kind == "parameter"]
            red = t[f"n_redrawn_{v}"].iloc[0] if f"n_redrawn_{v}" in t else 0
            L.append(f"| {mode} | {VARIANTS[v][1]} | {par[f'chi2_p_{v}'].min():.3f} ({t[f'chi2_p_{v}'].min():.3f}) | "
                     f"{par[f'p_ecdf_sim_{v}'].min():.3f} ({t[f'p_ecdf_sim_{v}'].min():.3f}) | "
                     f"{100 * t[f'frac_rhat_gt_1.01_{v}'].iloc[0]:.1f} % | {100 * t[f'frac_rhat_gt_1.05_{v}'].iloc[0]:.1f} % | "
                     f"{t[f'median_ess_kish_{v}'].iloc[0]:.0f} | {int(red)} |")
    L.append("")
    L.append("**Coverage (68 / 95 %) and median contraction, free parameters** (nominal 0.68 / 0.95; ± binomial SE):")
    L.append("")
    L.append("| mode | quantity | cov68 A | cov68 B | cov95 A | cov95 B | contraction A | contraction B |")
    L.append("|---|---|---|---|---|---|---|---|")
    f = lambda t, c: "–" if c not in t or pd.isna(t[c]) else f"{t[c]:.3f}"
    fse = lambda t, c, v: "–" if f"{c}_{v}" not in t or pd.isna(t[f"{c}_{v}"]) else f"{t[f'{c}_{v}']:.3f} ± {t[f'{c}_se_{v}']:.3f}"
    for _, t in allrows[allrows.kind == "parameter"].iterrows():
        L.append(f"| {t['mode']} | {t.quantity} | {fse(t, 'cov68', 'A')} | {fse(t, 'cov68', 'B')} | {fse(t, 'cov95', 'A')} | "
                 f"{fse(t, 'cov95', 'B')} | {f(t, 'contraction_median_A')} | {f(t, 'contraction_median_B')} |")
    L.append("")
    if paired_all:
        pc = pd.concat(paired_all)
        L.append("**Ranks A vs B (paired: same theta_true, noise, SMC seeds)** — Wilcoxon signed-rank p of the paired "
                 "differences, χ² homogeneity p of the two histograms, posterior-mean shift in posterior sd:")
        L.append("")
        L.append("| mode | quantity | n paired | mean Δrank/L | Wilcoxon p | homog. p | median abs shift [sd] | max abs shift [sd] | differ (Bonferroni) |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for q, t in pc.iterrows():
            L.append(f"| {t['mode']} | {q} | {t.n_paired} | {t.mean_rank_diff_rel:+.4f} | {t.wilcoxon_p:.3g} | "
                     f"{t.homogeneity_chi2_p:.3f} | {t.mean_shift_sd_median_abs:.3f} | {t.mean_shift_sd_max_abs:.3f} | "
                     f"{'YES' if t.ranks_differ_bonferroni else 'no'} |")
        diff = pc[pc.ranks_differ_bonferroni]
        diff05 = pc[pc.ranks_differ_p05]
        L.append("")
        L.append(f"- quantities whose ranks differ between A and B (Bonferroni, 0.05 / {len(pc) // max(1, len(paired_all))} per mode): "
                 + (", ".join(f"{m}:{q}" for q, m in zip(diff.index, diff["mode"])) or "none"))
        L.append(f"- at uncorrected p < 0.05: " + (", ".join(f"{m}:{q}" for q, m in zip(diff05.index, diff05["mode"])) or "none"))
    conv = tab[tab.subset != "all"]
    L.append(f"- data sets with max R-hat > {RHAT}: " + ("none in any run — no data set excluded, no second table needed"
                                                      if conv.empty else "present — rows `subset = max_r_hat<=1.01` in the CSV "
                                                      "give the SBC without them (the main rows keep ALL data sets)"))
    if not conv.empty:
        for mode in conv["mode"].unique():
            t = conv[(conv["mode"] == mode) & (conv.kind == "parameter")]
            for v in VARIANTS:
                if f"chi2_p_{v}" in t and not t[f"chi2_p_{v}"].isna().all():
                    L.append(f"  - {mode} {VARIANTS[v][1]} without them (n = {int(t[f'n_{v}'].iloc[0])}): min χ² p "
                             f"{t[f'chi2_p_{v}'].min():.3f}, min sim.-ECDF p {t[f'p_ecdf_sim_{v}'].min():.3f}")
    txt = "\n".join(L) + "\n"
    print(txt)
    if a.status:
        with open(a.status, "a") as fh:
            fh.write(txt)
    report.save_json({"base": a.base, "run_overrides": run_dirs, "csv": a.csv, "modes": modes,
                      "runs": {f"{m}/{v}": {"n": len(r.ranks), "n_redrawn": int(r.info.get("n_redrawn", 0)),
                                            "time_s": r.info["summary"]["time"]["total_s"],
                                            "data_model_info": r.info.get("data_model_info")} for (m, v), r in res.items()}},
                     out / "summary.json")
    print("written:", a.csv, out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="results/1trap_synthetic_paper_B2nf_kradfixed")
    p.add_argument("--out", default="results/sbc_1trap_B2nf")
    p.add_argument("--csv", default="results/sbc_1trap_B2nf_summary.csv")
    p.add_argument("--status", default=None)
    p.add_argument("--run", action="append", metavar="MODE=PATH",
                   help="run folder of MODE instead of <base>_<mode> (repeatable), e.g. trpl=results/.../2trap_85-15_trpl")
    main(p.parse_args())
