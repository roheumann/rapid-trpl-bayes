"""
combined_corner.py — ONE corner plot of a run with the posteriors of NUTS, emcee and tempered SMC (Origin + preview)
==================================================================================================================
15/10/2026 (revision).  Input: a finished main.ipynb run (any mode: trpl, sspl, joint) and its smc.ipynb output.

    python scripts/combined_corner.py results/<run>_<mode> [--smc smc_B2nf] [--fill fastest|SMC|NUTS|emcee]

* contours: the 1σ / 2σ HPD regions (and 1D marginals) of NUTS, emcee and SMC — same colours as every corner
  (1σ sky blue, 2σ light grey), told apart by the dash pattern (NUTS solid, emcee dashed, SMC dotted);
* colour map: the posterior of the FASTEST sampler whose diagnostics pass (max R-hat <= 1.01, min ESS >= 400) —
  normally SMC, which also gives the evidence log Z — as the posterior-mass bands, and its samples drawn in the
  same colours (each sample coloured by the posterior-mass level at its position);
* output: <run>/corner_combined/corner_combined.h5 (corner_export schema + methods/ + panels/*/scatter, read by
  corner-plot/origin/build_corner.py with FIGURE = "combined"), previews (samples / bands) and samplers.csv.
* --extra NAME=CSV (19/10/2026, repeatable): more posteriors drawn as contours / 1D marginals, e.g. the SMC run with
  the rate equations: --extra "ODE SMC=<run>/ode_vs_nn/ode_smc_weighted_samples.csv" --out corner_combined_with_ode
  (column 'weight' = importance weights; <name>_info.json next to <name>_weighted_samples.csv gives time and log Z).
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

R_HAT_MAX, ESS_MIN = 1.01, 400


def sampler_table(run: Path, smc_dir: Path | None) -> pd.DataFrame:
    """time, evaluations and diagnostics of NUTS / emcee (main.ipynb) and SMC (smc.ipynb)."""
    tm = pd.read_csv(run / "11_timing.csv")
    eff = pd.read_csv(run / "11_sampler_efficiency.csv").set_index("sampler")
    rows = []
    for name, key, prefix, csv in (("NUTS", "nuts", "NUTS (warm-up", "posterior_samples_nuts.csv"),
                                   ("emcee", "emcee", "emcee", "posterior_samples_emcee.csv")):
        t = tm[tm.label.astype(str).str.startswith(prefix)]
        if t.empty or not (run / csv).exists() or key not in eff.index:
            continue
        rows.append({"method": name, "time_s": float(t.mean_s.iloc[0]), "time_sd_s": float(t.std_s.iloc[0]),
                     "repeats": int(t.repeats.iloc[0]), "n_evals": float(eff.loc[key, "n_evals"]),
                     "min_ess_bulk": float(eff.loc[key, "min_ess_bulk"]), "max_r_hat": float(eff.loc[key, "max_r_hat"]),
                     "log_z": np.nan, "log_z_se": np.nan, "samples": str(run / csv)})
    if smc_dir is not None and (smc_dir / "summary.json").exists():
        sm = json.loads((smc_dir / "summary.json").read_text())["smc"]
        rows.append({"method": "SMC", "time_s": sm["time_mean_s"], "time_sd_s": sm.get("time_sd_s", 0.0),
                     "repeats": int(pd.read_csv(smc_dir / "8_timing.csv").query("label == 'tempered SMC'").repeats.iloc[0]),
                     "n_evals": sm["evals"], "min_ess_bulk": sm["min_ess_bulk"], "max_r_hat": sm["max_r_hat"],
                     "log_z": sm["log_z"], "log_z_se": sm["log_z_se"], "samples": str(smc_dir / "2_smc_weighted_samples.csv")})
    t = pd.DataFrame(rows)
    t["diagnostics_ok"] = (t.max_r_hat <= R_HAT_MAX) & (t.min_ess_bulk >= ESS_MIN)
    return t


def extra_row(name: str, csv: Path) -> dict:
    """Sampler row of an extra posterior (--extra NAME=CSV); time / evaluations / log Z from <stem>_info.json."""
    info_p = csv.with_name(csv.name.replace("_weighted_samples.csv", "_info.json"))
    info = json.loads(info_p.read_text()) if info_p != csv and info_p.exists() else {}
    return {"method": name, "time_s": info.get("time_total_s", np.nan), "time_sd_s": 0.0, "repeats": 1,
            "n_evals": info.get("evals_total_points", np.nan), "min_ess_bulk": np.nan, "max_r_hat": np.nan,
            "log_z": info.get("log_evidence", np.nan), "log_z_se": info.get("log_evidence_se", np.nan),
            "samples": str(csv), "diagnostics_ok": False}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("run")
    ap.add_argument("--smc", default=None, help="SMC folder inside the run (default: the newest smc_*)")
    ap.add_argument("--fill", default="fastest", help="fastest | SMC | NUTS | emcee")
    ap.add_argument("--out", default="corner_combined")
    ap.add_argument("--max-scatter", type=int, default=20000, help="samples drawn in the scatter (thinned above)")
    ap.add_argument("--bins", type=int, default=50, help="2D bins per axis (1D: 2.4x)")
    ap.add_argument("--smooth", type=float, default=1.6, help="Gaussian smoothing in bins (flat ridges need > 1)")
    ap.add_argument("--extra", action="append", default=[], metavar="NAME=CSV",
                    help="extra posterior drawn as contours (weighted samples, column 'weight'); repeatable")
    args = ap.parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")            # the network is loaded only for the prior
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import h5py
    import matplotlib
    matplotlib.use("Agg")
    from inference import corner_export as ce, corner_plot as cp, report, surrogate
    from inference.prior import Prior

    run = Path(args.run)
    st = json.loads((run / "settings.json").read_text())
    cfg, mode = st["config"], st["mode"]
    smc_dir = run / args.smc if args.smc else max(run.glob("smc_*"), key=lambda p: p.stat().st_mtime, default=None)
    tab = sampler_table(run, smc_dir)
    for spec in args.extra:                     # extra posteriors (e.g. SMC with the rate equations): contours only
        name, path = spec.split("=", 1)
        tab = pd.concat([tab, pd.DataFrame([extra_row(name.strip(), Path(path.strip()))])], ignore_index=True)
    if args.fill == "fastest":
        ok = tab[tab.diagnostics_ok]
        fill = (ok if len(ok) else tab).sort_values("time_s").method.iloc[0]
    else:
        fill = args.fill
    tab["fill"] = tab.method == fill
    print(tab.drop(columns="samples").round(4).to_string(index=False))
    print(f"colour map and samples: {fill}")

    art = surrogate.load_network(cfg["nn"]["dir"], cfg["nn"].get("stem"))
    prior = Prior.from_artifacts(art, Eg=cfg["physics"]["Eg_eV"], order_traps=cfg["prior"]["order_traps"],
                                 photodoping_traps=cfg["prior"]["photodoping_traps"],
                                 bounds_override=cfg["prior"]["bounds_override"], fixed=cfg["prior"].get("fixed", {}),
                                 T=cfg["physics"]["T_K"])
    best = np.array(json.loads((run / "6_best_fit.json").read_text())["theta_best"], float)
    start = np.array(json.loads((run / "start_point.json").read_text())["theta"], float) \
        if (run / "start_point.json").exists() else None
    truth = None
    if cfg.get("true_params"):
        tp = cfg["true_params"]
        truth = np.atleast_2d(prior.from_physical(tp["Et_eV"], tp["Nt_cm3"], tp["tau_n_s"], tp["tau_p_s"],
                                                  tp["krad_cm3s"]))[0]
    sp = ce.physical_corner_spec(prior, best, truth, start)
    to_plot = lambda th: sp["dmap"].to_x(th)[:, sp["perm"]]

    methods, info, scatter = {}, {}, None
    for r in tab.itertuples():
        d = pd.read_csv(r.samples)
        w = d["weight"].values if "weight" in d else None
        methods[r.method] = (to_plot(d[prior.names].values), w)
        lab = f"{r.method} ({report_time(r.time_s)}" if np.isfinite(r.time_s) else f"{r.method} ("
        lab += f", log Z = {r.log_z:.2f} ± {r.log_z_se:.2f})" if np.isfinite(r.log_z) else ")"
        info[r.method] = {"label": lab, "time_s": r.time_s, "time_sd_s": r.time_sd_s, "repeats": r.repeats,
                          "n_evals": r.n_evals, "min_ess_bulk": r.min_ess_bulk, "max_r_hat": r.max_r_hat,
                          "log_z": r.log_z, "log_z_se": r.log_z_se, "samples_file": r.samples}
        if r.method == fill:
            if r.method == "SMC":            # equal-weight final particles of the SMC run
                scatter = pd.read_csv(Path(r.samples).parent / "posterior_samples_smc.csv")[prior.names].values
            else:
                scatter = d[prior.names].values
            if len(scatter) > args.max_scatter:
                scatter = scatter[np.random.default_rng(0).choice(len(scatter), args.max_scatter, replace=False)]
            scatter = to_plot(scatter)
    # draw order: the fill sampler last (its dotted / solid lines on top)
    methods = dict(sorted(methods.items(), key=lambda kv: kv[0] == fill))

    out = run / args.out
    (out / "figures").mkdir(parents=True, exist_ok=True)
    h5 = out / f"{args.out}.h5"
    with h5py.File(h5, "w") as f:
        f.attrs.update({"schema_version": 1, "run_in": str(run), "mode": mode, "fill_method": fill,
                        "n_traps": prior.n_traps, "created": pd.Timestamp.now().isoformat(),
                        "config_json": json.dumps(cfg), "selection": f"fastest sampler with max R-hat <= {R_HAT_MAX} "
                                                                     f"and min ESS >= {ESS_MIN}" if args.fill == "fastest"
                        else "chosen by hand"})
        ce.write_comparison_corner(f.create_group("corner/physical"), fill, methods, sp["keys"], sp["meta"],
                                   sp["layout"], sp["ranges"], scatter=scatter, method_info=info,
                                   bins_1d=int(2.4 * args.bins), bins_2d=args.bins, smooth=args.smooth,
                                   attrs={"layout_kind": sp["layout_kind"]})
        gs = f.create_group("samplers")
        for c in ("method", "time_s", "time_sd_s", "n_evals", "min_ess_bulk", "max_r_hat", "log_z", "log_z_se"):
            gs[c] = tab[c].astype(str).values.astype("S") if c == "method" else tab[c].values.astype(float)
    tab.to_csv(out / "samplers.csv", index=False)

    # 68 / 95 % HPD intervals of every sampler (decades; n1 instead of E_t)
    rows = []
    with h5py.File(h5, "r") as f:
        g = f["corner/physical"]
        for m in g["methods"]:
            for k in sp["keys"]:
                gm = g["methods"][m][k]
                rows.append({"method": m, "parameter": k, "median": gm.attrs["median"],
                             "HPD68": "; ".join(f"[{a:.3f}, {b:.3f}]" for a, b in gm["intervals68"][()]),
                             "HPD95": "; ".join(f"[{a:.3f}, {b:.3f}]" for a, b in gm["intervals95"][()]),
                             "truth": sp["meta"][k]["truth"], "best_fit": sp["meta"][k]["best_fit"]})
        pd.DataFrame(rows).to_csv(out / "marginals_by_sampler.csv", index=False)
        sm = tab[tab.method == fill].iloc[0]
        dash_txt = {"solid": "solid", "dash": "dashed", "dot": "dotted", "dashdot": "dash-dot"}
        ttl = (f"{run.name}\n" + ", ".join(f"{m} ({dash_txt[ce.METHOD_DASH.get(m, 'dashdot')]})" for m in tab.method)
               + f"; colour = {fill} "
               f"({report_time(sm.time_s)})" + (f", log Z = {sm.log_z:.2f} ± {sm.log_z_se:.2f}" if np.isfinite(sm.log_z) else ""))
        for bg in ("scatter", "bands"):
            fig = cp.plot_corner(g, title=ttl, background=bg)
            name = "samples" if bg == "scatter" else "bands"
            report.save_figure(fig, out / "figures" / f"corner_combined_{name}.png")
            fig.savefig(out / "figures" / f"corner_combined_{name}.pdf")
    print("written:", h5, "and", out / "figures")


def report_time(s):
    return f"{s:.1f} s" if s < 120 else f"{s / 60:.1f} min"


if __name__ == "__main__":
    main()
