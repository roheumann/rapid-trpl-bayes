"""
paper_slices.py — the paper's sliced grid at the best fit + best-fit / truth simulations, ONE HDF5 file for Origin
================================================================================================================
15/10/2026 (revision).  Input: a finished main.ipynb run (mode trpl, sspl or joint).

    python scripts/paper_slices.py results/<run>_<mode> [--n 100] [--gpu 0]

1. Sliced grid of the original submission (inference/slices.py): one 2-D slice n x n through the best fit for every
   pair of free parameters, sigma_eff = sqrt(sigma_config^2 + MSE_min over the slices), probability normalised per
   slice.  Timed (timeit, compile excluded); a row "sliced grid at the best fit" is added to the run's T8 table.
2. Simulations at the best fit (6_best_fit.json) and at the true parameters (config true_params):
   NN tau_diff(QFLS) and the full ODE solution (PL(t), tau_diff(QFLS), n, p, n_t, trapping / detrapping rates);
   t and PL of the NN curves from dt = -tau_diff/(2kT) dQFLS, anchored on the ODE of the same parameter set.
3. Output in <run>/slices_paper/:
     slices_paper.h5   /corner/physical  slice corner WITHOUT diagonal (corner_export schema, layout 'slices':
                                         curve panels tau(QFLS), PL(t) [, ss-PL] in the empty upper right)
                       /curves           data, best_fit/{nn, ode}, truth/{nn, ode} [, sspl]
     figures/          slices_corner.png/pdf, t_pl_qfls_tau_diff.png/pdf, concentration_rates.png/pdf
     csv/              trpl_sim_<set>.csv with the column names of the paper export (fitting/trpl_fitting.py),
                       tau_fit_<set>.csv (NN at the data), slices_<kx>_vs_<ky>.csv, sigma.json,
                       tau0_lines_<n1>_vs_<taup>.csv
   iso-tau_0 lines (19/10/2026, Robin) in the n1_k - tau_p,k panels: tau_0 = tau_p a / (1 + k_rad tau_p n1),
   a = sqrt(n1 N_t)/n_i for N_t >= 10 n1 and n1/n_i for N_t <= 0.1 n1, each line only in its regime; N_t, k_rad at the
   best fit; tau_0 from the shallow regime of the data (or --tau0, one value or one per trap; --no-tau0: none).
   Origin: corner-plot/origin/build_corner.py (pick slices_paper.h5): the slice corner page, plus graphs from the
   templates t_pl_qfls_tau_diff_paper and concentration-rates (User Files folder) — see the README there.
--mode / --sigma-trpl (22/10/2026, Robin): slices of another likelihood mode than the run's, through that mode's best fit
(6_best_fit.json best_per_mode, e.g. the trpl/sspl fits of a joint run), and a fixed tr-PL sigma.  Any sigma override
fixes the sigmas as given (no MSE_min term); with --mode the run's T8 table is left alone.
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

COL = {"best_fit": "#E41A1C", "truth": "#00CED1"}
NAME = {"best_fit": "best fit", "truth": "true parameters"}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("run")
    ap.add_argument("--n", type=int, default=100, help="points per axis (paper: 100)")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--colour", choices=["rel_drop", "rank"], default="rel_drop",
                    help="fill: (p(best fit) - p)/p(best fit) (default) or the paper's pooled best-N % rank")
    ap.add_argument("--tau0", type=float, nargs="+", default=None,
                    help="tau_0 (s) of the iso-tau_0 lines, one value or one per trap (default: shallow regime of the data)")
    ap.add_argument("--no-tau0", action="store_true", help="no iso-tau_0 lines in the n1 - tau_p panels")
    ap.add_argument("--sigma-sspl", type=float, default=None,
                    help="override the ss-PL sigma of the run (e.g. 0.0020214 eV = 0.033958 log10 PLQY x kT ln10; 20/10/2026)")
    ap.add_argument("--sigma-trpl", type=float, default=None,
                    help="override the tr-PL sigma of the run (fixed, no MSE_min term; 22/10/2026)")
    ap.add_argument("--mode", choices=["trpl", "sspl", "joint"], default=None,
                    help="likelihood mode of the slices (default: the run's); centre = best_per_mode[mode] of 6_best_fit.json")
    ap.add_argument("--out-tag", default="slices_paper", help="output folder inside the run")
    ap.add_argument("--center", choices=["best_fit", "truth"], default="best_fit",
                    help="parameter set the 2D slices go through (truth: the true parameters of a synthetic run; 20/10/2026)")
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import h5py
    import matplotlib
    matplotlib.use("Agg")
    from inference import corner_export as ce, corner_plot as cp, ode, report, runs, slices, surrogate
    from inference import benchmark as bm
    from inference.benchmark import timeit_call
    from inference.prior import Prior
    surrogate.configure_tensorflow()

    run = Path(args.run)
    st = json.loads((run / "settings.json").read_text())
    cfg, mode = st["config"], st["mode"]
    other_mode = args.mode is not None and args.mode != mode
    mode = args.mode or mode
    Eg, npulse, T = cfg["physics"]["Eg_eV"], cfg["physics"]["npulse_cm3"], cfg["physics"]["T_K"]
    art = surrogate.load_network(cfg["nn"]["dir"], cfg["nn"].get("stem"))
    prior = Prior.from_artifacts(art, Eg=Eg, order_traps=cfg["prior"]["order_traps"],
                                 photodoping_traps=cfg["prior"]["photodoping_traps"],
                                 bounds_override=cfg["prior"]["bounds_override"], fixed=cfg["prior"].get("fixed", {}), T=T)
    best = json.loads((run / "6_best_fit.json").read_text())
    theta_best = np.array(best["best_per_mode"][mode]["theta"] if other_mode else best["theta_best"], float)
    truth = None
    if cfg.get("true_params"):
        tp = cfg["true_params"]
        truth = np.atleast_2d(prior.from_physical(tp["Et_eV"], tp["Nt_cm3"], tp["tau_n_s"], tp["tau_p_s"], tp["krad_cm3s"]))[0]
    start = np.array(json.loads((run / "start_point.json").read_text())["theta"], float) \
        if (run / "start_point.json").exists() else None
    has_ss = mode in ("sspl", "joint")
    sig_base = {"trpl": float(cfg["trpl"]["sigma"]), "sspl": float(cfg["sspl"]["sigma"])}
    add_mse_min = cfg.get("sigma_rule") != "model_plus_nn" and args.sigma_trpl is None and args.sigma_sspl is None
    if not add_mse_min:     # sigma = sqrt(sigma_model^2 + sigma_nn^2) of the run (no MSE_min term), as for its fits and SMC
        rs = runs.run_sigmas(run)
        sig_base = {k: float(v) if v is not None else sig_base[k] for k, v in rs.items()}
    if args.sigma_sspl is not None:
        print(f"sigma_ssPL overridden: {sig_base['sspl']} -> {args.sigma_sspl}")
        sig_base["sspl"] = float(args.sigma_sspl)
    if args.sigma_trpl is not None:
        print(f"sigma_trPL overridden: {sig_base['trpl']} -> {args.sigma_trpl}")
        sig_base["trpl"] = float(args.sigma_trpl)
    lp, _ = runs.make_log_post(cfg, run, art, prior, mode, sig_base["trpl"], sig_base["sspl"] if has_ss else None)
    lp_tr, sur = runs.make_log_post(cfg, run, art, prior, "trpl", sig_base["trpl"], None)          # NN curves
    sur_tr = lp_tr.trpl_terms[0].surrogate
    window = isinstance(sur_tr, surrogate.TrplSurrogateWindow)            # axis-head network: physical QFLS axis per theta
    lp_ss = runs.make_log_post(cfg, run, art, prior, "sspl", None, sig_base["sspl"])[0] if cfg["sspl"].get("use") else None

    out = run / args.out_tag
    for d in ("figures", "csv"):
        (out / d).mkdir(parents=True, exist_ok=True)

    # ── 1. sliced grid (timed; the first call compiles) ─────────────────────────
    if args.center == "truth" and truth is None:
        raise SystemExit("--center truth needs true_params in the config of the run")
    theta_center = truth if args.center == "truth" else theta_best
    print(f"slices through the {args.center.replace('_', ' ')}")
    S, t_sl = timeit_call(slices.run_slices, lp, prior, theta_center, n=args.n, sigma_base=sig_base, add_mse_min=add_mse_min,
                          repeats=args.repeats, warmup=1, label=f"sliced grid ({args.n}^2 per pair)")
    print(t_sl, f"| {S['n_evals']:,} evaluations | sigma {json.dumps(S['sigma'])}")
    sp = ce.physical_corner_spec(prior, theta_best, truth, start)
    keys = sp["keys"]
    # corner: tau(QFLS) wide over the top row + PL(t) (trpl) or the fitted ss-PL data (sspl, joint); PL(t) is always
    # in figures/t_pl_qfls_tau_diff and in the template graph t_pl_qfls_tau_diff_paper
    curve_panels = ["tau_qfls", "sspl"] if (has_ss and lp_ss is not None) else ["tau_qfls", "pl_t"]
    layout = slices.layout_slices(keys, curve_panels)

    # ── 2. simulations at the best fit and the true parameters ──────────────────
    kT = prior.kT
    trpl = pd.read_csv(run / "trpl_used.csv")
    sets = {"best_fit": theta_best} | ({"truth": truth} if truth is not None else {})
    sims = {}
    for name, th in sets.items():
        sim = ode.simulate_trpl_full(th, prior, npulse, art.constants, T)
        sim = sim.drop_duplicates(subset="qfls_eV").reset_index(drop=True)
        qs = np.argsort(sim.qfls_eV.values)
        t_at = (lambda q, s=sim, o=qs: np.interp(q, s.qfls_eV.values[o], s.time_s.values[o]))
        q0, qmax = float(sim.qfls_eV.iloc[0]), float(sim.qfls_eV.max())
        pr = lp_tr.predict_trpl(th)
        if window:   # axis-head network: NN curve and data on the physical axis of theta (q_max node), like the ODE
            qn, tn, tt, pln = slices.nn_time_axis(pr["qfls_axis_physical"][0], 10 ** pr["log10_tau_own_axis"][0],
                                                  t_at, q0, kT, q_max=qmax)
            q_data = pr["qfls_used_physical"][0]
        else:        # fixed analytic axis (high-injection calibration of data and network)
            qn, tn, tt, pln = slices.nn_time_axis(pr["qfls_axis"], 10 ** pr["log10_tau_axis"][0], t_at, q0, kT,
                                                  q_max=qmax)
            q_data = pr["qfls_used"]
        y_ode = ode.ode_log10_tau(th, prior, npulse, art.constants, pr["qfls_used"], T)
        ok = np.isfinite(y_ode)
        sims[name] = {"theta": th, "ode": sim, "t_at": t_at, "q0": q0,
                      "nn": {"qfls": qn, "tau_diff": tn, "t": tt, "pl": pln},
                      "q_max_nn": float(sur_tr.window(th)[0][0]) if window else float(sur_tr.qfls_axis[-1]),
                      "at_data": {"qfls": q_data, "tau_obs": 10 ** pr["log10_tau_obs"],
                                  "tau_nn": 10 ** pr["log10_tau_used"][0], "tau_ode": 10 ** y_ode},
                      "eps_rms_nn": float(np.sqrt(np.mean((pr["log10_tau_used"][0] - pr["log10_tau_obs"]) ** 2))),
                      "eps_rms_ode": float(np.sqrt(np.mean((y_ode[ok] - pr["log10_tau_obs"][ok]) ** 2))),
                      "ode_coverage": float(ok.mean())}
        if lp_ss is not None:
            ps = lp_ss.predict_sspl(th)
            sims[name]["sspl"] = {"x_exp": ps["x_exp"], "y_obs": ps["y_obs"], "y_model": ps["y_model"][0],
                                  "curve_x": ps["curve"]["x"][0], "curve_y": ps["curve"]["y"][0]}
        print(f"{name}: eps_rms NN {sims[name]['eps_rms_nn']:.3g}, ODE {sims[name]['eps_rms_ode']:.3g} (log10 tau)")
    # data: time from the ODE of the true parameters (synthetic) or of the best fit, PL from QFLS
    ref = sims["truth"] if "truth" in sims else sims["best_fit"]
    q_trpl = trpl.qfls.values
    if window:       # data on the calibration (q_max node) of the reference set, consistent with its ODE time axis
        q_trpl = surrogate.calibrate_qfls(q_trpl, sur_tr.qfls_max_analytic, ref["q_max_nn"], sur_tr.Vt)
    qd, td, t_d, pl_d = slices.nn_time_axis(q_trpl, trpl.tau_diff.values, ref["t_at"], ref["q0"], kT)
    t_src = ("anchored on the ODE solution of the " + ("true parameters" if "truth" in sims else "best fit")
             + "; dt = -tau_diff/(2kT) dQFLS (the data file has no time axis)")

    # iso-tau_0 lines in the n1_k - tau_p,k panels (Robin, 19/10/2026); never stops the slices
    tau0, t0_lines = None, {}
    if not args.no_tau0:
        try:
            tau0 = tau0_settings(args, cfg, trpl, q_trpl, prior, art.constants, theta_center)
            for row in layout:
                if row["kind"] != b"2d":
                    continue
                kx, ky = row["kx"].decode(), row["ky"].decode()
                for k, t in tau0.items():
                    pair = (f"log10_n1_{t['trap']}", f"log10_taup_{t['trap']}")
                    if {kx, ky} != set(pair):
                        continue
                    ax_n1, ax_tp = S["axes"][pair[0]], S["axes"][pair[1]]
                    L = slices.tau0_lines((ax_n1.min(), ax_n1.max()), (ax_tp.min(), ax_tp.max()), t["tau0_s"], t["Nt_cm3"],
                                          t["krad_cm3s"], t["ni_cm3"])
                    t0_lines[(kx, ky)] = (k, {r: (v if kx == pair[0] else v[:, ::-1]) for r, v in L.items()})
            for k, t in tau0.items():
                print(f"tau_0 {k}: {t['tau0_s']:.3e} s ({t['source']}); best fit: N_t {t['Nt_cm3']:.3g}, n1 {t['n1_cm3']:.3g} "
                      f"({t['regime_best_fit']}), tau_0 of the best fit {t['tau0_best_fit_s']:.3e} s")
        except Exception as e:                                               # noqa: BLE001
            print(f"WARNING: no iso-tau_0 lines ({type(e).__name__}: {e})")
            tau0, t0_lines = None, {}

    # ── 3. HDF5 ──────────────────────────────────────────────────────────────────
    h5 = out / f"{args.out_tag}.h5"
    with h5py.File(h5, "w") as f:
        f.attrs.update({"schema_version": 1, "run_in": str(run), "mode": mode, "n_traps": prior.n_traps,
                        "created": pd.Timestamp.now().isoformat(), "config_json": json.dumps(cfg),
                        "method": "sliced grid of the original submission (2-D slices through the best fit)",
                        "sigma_json": json.dumps(S["sigma"]), "n_per_axis": args.n,
                        "qfls_calibration": ("physical: data and NN on the network's q_max(theta) (axis head), ODE "
                                             "kT ln(np/ni^2)" if window else "nominal: 2 Vt ln(n_pulse/n_i) + Vt ln(PL/PL0)")})
        g = f.create_group("corner/physical")
        g.attrs.update({"layout_kind": "slices", "param_order": json.dumps(keys), "hpd_levels": np.array(ce.HPD_LEVELS),
                        "sigma_levels": np.array(ce.SIGMA_LEVELS), "colormap": ce.COLORMAP,
                        "colors_json": json.dumps(ce.COLORS), "band_colors_json": json.dumps(ce.band_colors_hex()),
                        "diag_mode": "none", "coordinates": "decades", "logL_ref": S["logL_ref"],
                        "fill_quantity": args.colour,
                        "colour_quantity": ("(p(best fit) - p) / p(best fit) = 1 - exp(log L - log L(best fit)); 0 = best fit"
                                            if args.colour == "rel_drop" else
                                            "paper: rank of the likelihood ratio to the best fit among all slice points "
                                            "(pooled; 'best N %')") + "; mass_level = probability mass inside each slice",
                        "colour_bar_title": COLOUR_TITLE[args.colour],
                        "n_ranked": S["n_ranked"]})
        if tau0:
            g.attrs.update({"tau0_json": json.dumps(tau0), "tau0_dash_json": json.dumps(ce.TAU0_DASH),
                            "tau0_formula": "tau_0 = tau_p a / (1 + k_rad tau_p n1), a = sqrt(n1 N_t)/n_i (N_t >= 10 n1) or "
                                            "n1/n_i (N_t <= 0.1 n1); N_t, k_rad at the best fit"})
        gp = g.create_group("params")
        for i, k in enumerate(keys):
            a = gp.create_group(k)
            a.attrs["index"] = i
            a.attrs["lo"], a.attrs["hi"] = float(S["axes"][k].min()), float(S["axes"][k].max())
            for kk, v in sp["meta"][k].items():
                a.attrs[kk] = np.nan if v is None else v
        g.create_dataset("layout", data=layout)
        pn = g.create_group("panels")
        for row in layout:
            if row["kind"] != b"2d":
                continue
            kx, ky = row["kx"].decode(), row["ky"].decode()
            P = S["panels"].get((kx, ky))
            tr = False
            if P is None:
                P, tr = S["panels"][(ky, kx)], True
            prob = P["prob"].T if tr else P["prob"]
            rel = P["log10_rel"].T if tr else P["log10_rel"]
            rank = P["rank"].T if tr else P["rank"]
            drop = P["rel_drop"].T if tr else P["rel_drop"]
            x, y = S["axes"][kx], S["axes"][ky]
            lev = ce.hpd_level_map(prob)
            q = pn.create_group(f"{kx}__{ky}")
            q["x"], q["y"] = x, y
            q["prob"] = prob.astype("f8")
            q["density"] = (prob / ((x[1] - x[0]) * (y[1] - y[0]))).astype("f4")
            q["hpd"] = (drop if args.colour == "rel_drop" else rank).astype("f4")     # colour levels (fill)
            q["rank"] = rank.astype("f4")                    # paper's 'best N %' rank
            q["rel_drop"] = drop.astype("f4")                # (p(best fit) - p) / p(best fit)
            q["mass_level"] = lev.astype("f4")               # HPD mass inside the slice (1σ / 2σ contours)
            q["log10_rel"] = rel.astype("f4")
            q["sigma1"] = ce.contour_polyline(x, y, lev, ce.SIGMA_LEVELS[0])
            q["sigma2"] = ce.contour_polyline(x, y, lev, ce.SIGMA_LEVELS[1])
            if (kx, ky) in t0_lines:                         # iso-tau_0 lines [x, y], NaN rows = gaps
                k_, L = t0_lines[(kx, ky)]
                q.attrs["tau0_trap"] = k_
                rows_ = []
                for reg, v in L.items():
                    q[f"tau0_{reg}"] = v
                    rows_.append(pd.DataFrame({"regime": reg, "condition": slices.TAU0_REGIMES[reg], kx: v[:, 0], ky: v[:, 1]}))
                d_ = pd.concat(rows_, ignore_index=True)
                for c_ in (kx, ky):
                    d_[c_.replace("log10_", "") + ("_s" if "tau" in c_ else "_cm3")] = 10 ** d_[c_]
                d_.to_csv(out / "csv" / f"tau0_lines_{kx}_vs_{ky}.csv", index=False, float_format="%.6e")
            pd.DataFrame({kx: np.tile(x, len(y)), ky: np.repeat(y, len(x)), "probability": prob.ravel(),
                          "log10_relative_prob": rel.ravel(), "rel_drop_vs_best_fit": drop.ravel(),
                          "rank_best_fraction": rank.ravel(),
                          "hpd_mass_level": lev.ravel()}).to_csv(
                out / "csv" / f"slices_{kx}_vs_{ky}.csv", index=False, float_format="%.6e")
        gc = f.create_group("curves")
        gd = gc.create_group("data")
        gd["qfls"], gd["tau_diff"], gd["t"], gd["pl"] = qd, td, t_d, pl_d
        gd.attrs["t_source"] = t_src
        for name, sm in sims.items():
            gs = gc.create_group(name)
            gs.attrs.update({"label": NAME[name], "color": COL[name], "eps_rms_nn": sm["eps_rms_nn"],
                             "eps_rms_ode": sm["eps_rms_ode"], "ode_coverage_of_data": sm["ode_coverage"],
                             "theta": sm["theta"], "names": json.dumps(prior.names),
                             "physical_json": json.dumps(prior.to_physical(sm["theta"][None]).iloc[0].to_dict())})
            for k_, v in sm["nn"].items():
                gs[f"nn/{k_}"] = v
            for k_, v in sm["at_data"].items():
                gs[f"at_data/{k_}"] = v
            for c in sm["ode"].columns:
                gs[f"ode/{c}"] = sm["ode"][c].values
            if "sspl" in sm:
                for k_, v in sm["sspl"].items():
                    gs[f"sspl/{k_}"] = np.asarray(v, float)
        if lp_ss is not None:
            gc.attrs["sspl_mode"] = lp_ss.sspl_term.observation.mode

    # CSVs with the column names of the paper export (fitting/trpl_fitting.py: results['trpl_sim'] / ['trpl_fit'])
    K = prior.n_traps
    for name, sm in sims.items():
        o = sm["ode"]
        d = {"time_tr_s": o.time_s, "pl_tr": o.pl_norm, "qfls_tr_eV": o.qfls_eV, "tau_tr_s": o.tau_diff_s,
             "n_tr_1/cm3": o.n_cm3, "p_tr_1/cm3": o.p_cm3}
        for k in range(1, K + 1):
            d[f"nt_tr_{k}_1/cm3"] = o[f"nt_{k}_cm3"]
        for k in range(1, K + 1):
            d[f"e_trapping_tr_trap{k}_1/cm3/s"] = o[f"e_capture_{k}"]
            d[f"e_detrapping_tr_trap{k}_1/cm3/s"] = o[f"e_emission_{k}"]
            d[f"h_trapping_tr_trap{k}_1/cm3/s"] = o[f"h_capture_{k}"]
            d[f"h_detrapping_tr_trap{k}_1/cm3/s"] = o[f"h_emission_{k}"]
        d["R_rad_1/cm3/s"] = o.R_rad_cm3s
        pd.DataFrame(d).to_csv(out / "csv" / f"trpl_sim_{name}.csv", index=False, float_format="%.8g")
        nn = sm["nn"]
        pd.DataFrame({"time_nn_s": nn["t"], "pl_nn": nn["pl"], "qfls_nn_eV": nn["qfls"], "tau_nn_s": nn["tau_diff"]}).to_csv(
            out / "csv" / f"nn_sim_{name}.csv", index=False, float_format="%.8g")
        ad = sm["at_data"]
        pd.DataFrame({"qfls": ad["qfls"], "tau_data": ad["tau_obs"], "tau_fit_nn": ad["tau_nn"],
                      "tau_fit_ode": ad["tau_ode"]}).to_csv(out / "csv" / f"tau_fit_{name}.csv", index=False,
                                                            float_format="%.8g")
    pd.DataFrame({"time_s": t_d, "pl": pl_d, "qfls_eV": qd, "tau_diff_s": td}).to_csv(out / "csv" / "data.csv", index=False,
                                                                                        float_format="%.8g")
    summary = {"run": str(run), "mode": mode, "center": args.center, "n_per_axis": args.n, "sigma": S["sigma"], "n_evals": S["n_evals"],
               "time": t_sl.as_dict(), "eps_rms": {n: {"nn": s["eps_rms_nn"], "ode": s["eps_rms_ode"]} for n, s in sims.items()},
               "data_time_axis": t_src, "h5": str(h5), "tau0": tau0}
    report.save_json(summary, out / "summary.json")
    pd.DataFrame([t_sl.as_dict() | {"n_evals": S["n_evals"], "note": f"{len(S['panels'])} slices of {args.n}^2 points"}]).to_csv(
        out / "timing.csv", index=False)
    if (run / "11_T8_timing_table.csv").exists() and not other_mode:
        t8 = bm.update_t8(run, slices_dir=out)
        print("T8:", t8.step.iloc[-1], "|", t8.nn_workflow.iloc[-1])

    # ── 4. previews ─────────────────────────────────────────────────────────────
    import matplotlib.pyplot as plt
    with h5py.File(h5, "r") as f:
        fig = cp.plot_corner(f["corner/physical"], curves=f["curves"],
                             title=f"{run.name}\nsliced grid at the best fit (paper method, {args.n}² per slice); "
                                   f"contours: 1σ / 2σ mass inside each slice",
                             cbar_label=COLOUR_LABEL[args.colour])
        report.save_figure(fig, out / "figures" / "slices_corner.png"); fig.savefig(out / "figures" / "slices_corner.pdf")
        plt.close(fig)
        fig = plot_t_pl_qfls_tau(f["curves"], run.name)
        report.save_figure(fig, out / "figures" / "t_pl_qfls_tau_diff.png"); fig.savefig(out / "figures" / "t_pl_qfls_tau_diff.pdf")
        plt.close(fig)
        fig = plot_concentration_rates(f["curves"], K, run.name)
        report.save_figure(fig, out / "figures" / "concentration_rates.png"); fig.savefig(out / "figures" / "concentration_rates.pdf")
        plt.close(fig)
    print("written:", out)


def tau0_settings(args, cfg, trpl, q_cal, prior, constants, theta_best):
    """
    tau_0 per trap for the iso-tau_0 lines + N_t, k_rad, n1 at the best fit (the values of the n1 - tau_p slice).
    tau_0: --tau0, else physics.tau0_s of the config, else the shallow regime of the data, tau_diff = tau_0
    exp(-QFLS/(2kT)) (legacy start-point estimator), on the QFLS calibration of the plotted data (q_cal) and in the
    QFLS window of the trap's start_point setting (automatic shallow regime without one).
    """
    from inference import add_legacy_paths, slices
    kT = prior.kT
    ni = math.sqrt(constants["Nc"] * constants["Nv"]) * math.exp(-prior.Eg / (2.0 * kT))
    ph = prior.to_physical(np.asarray(theta_best, float)[None]).iloc[0]
    krad = float(ph["krad_cm3s"])
    traps = (cfg.get("start_point") or {}).get("traps") or []
    auto, out = None, {}
    for k in range(1, prior.n_traps + 1):
        Nt, Et, taup = float(ph[f"Nt_{k}_cm-3"]), float(ph[f"Et_{k}_eV"]), float(ph[f"tau_p_{k}_s"])
        n1 = prior.Nc * math.exp((Et - prior.Eg) / kT)
        win, n_pts = None, 0
        if args.tau0:
            tau0, src = float(args.tau0[min(k, len(args.tau0)) - 1]), "--tau0"
        elif cfg["physics"].get("tau0_s") is not None:
            tau0, src = float(cfg["physics"]["tau0_s"]), "config physics.tau0_s"
        else:
            s = traps[k - 1] if k <= len(traps) else {}
            win = s.get("qfls_window")
            if win is not None:
                mask = ((trpl.qfls.values >= win[0]) & (trpl.qfls.values <= win[1]))
                src = f"data, shallow regime: QFLS window [{win[0]}, {win[1]}] eV of start_point trap {k}"
            else:
                if auto is None:
                    add_legacy_paths()
                    import start_point_estimation_module as spe
                    auto = np.asarray(spe.determine_trap_regime(trpl)[1], bool)
                mask = auto
                src = "data, automatic shallow regime (determine_trap_regime)"
            tau0, n_pts = slices.tau0_shallow(q_cal, trpl.tau_diff.values, kT, mask), int(mask.sum())
        regime = "photodoping" if Nt >= 10 * n1 else "non_photodoping" if Nt <= 0.1 * n1 else "transition"
        t0_bf = {r: float(slices.tau0_of_taup(n1, taup, Nt, krad, ni, r == "photodoping")) for r in slices.TAU0_REGIMES}
        out[f"trap {k}"] = {"trap": k, "tau0_s": tau0, "source": src, "qfls_window": win, "n_points": n_pts,
                            "Nt_cm3": Nt, "krad_cm3s": krad, "ni_cm3": ni, "n1_cm3": n1, "taup_s": taup,
                            "regime_best_fit": regime,
                            "tau0_best_fit_s": t0_bf.get(regime, float("nan")),
                            "tau0_best_fit_photodoping_formula_s": t0_bf["photodoping"],
                            "tau0_best_fit_non_photodoping_formula_s": t0_bf["non_photodoping"]}
    return out


def plot_t_pl_qfls_tau(c, title):
    """PL(t) and tau_diff(QFLS): data, NN and ODE at the best fit and at the true parameters."""
    import matplotlib.pyplot as plt
    from inference.corner_plot import draw_curve_panel
    fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.8))
    draw_curve_panel(axs[0], c, "pl_t", legend=False)
    draw_curve_panel(axs[1], c, "tau_qfls", legend=True)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    return fig


def plot_concentration_rates(c, n_traps, title):
    """n, p, n_t and the trapping / detrapping rates vs QFLS (ODE; best fit solid, true parameters dashed)."""
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    styles = {"best_fit": "-", "truth": "--"}
    conc = [("n_cm3", r"$n$", "C0"), ("p_cm3", r"$p$", "C3")] + \
           [(f"nt_{k}_cm3", rf"$n_{{t{k if n_traps > 1 else ''}}}$", f"C{1 + k}") for k in range(1, n_traps + 1)]
    rates = []
    for k in range(1, n_traps + 1):
        s = str(k) if n_traps > 1 else ""
        rates += [(f"e_capture_{k}", rf"e$^-$ trapping {s}", "C0"), (f"e_emission_{k}", rf"e$^-$ detrapping {s}", "C9"),
                  (f"h_capture_{k}", rf"h$^+$ trapping {s}", "C3"), (f"h_emission_{k}", rf"h$^+$ detrapping {s}", "C1")]
    for name in ("best_fit", "truth"):
        if name not in c:
            continue
        o = c[name]["ode"]
        q = o["qfls_eV"][()]
        for col, lab, colr in conc:
            axs[0].semilogy(q, o[col][()], styles[name], color=colr, lw=1.3, label=lab if name == "best_fit" else None)
        for col, lab, colr in rates + [("R_rad_cm3s", r"radiative $k_\mathrm{rad}np$", "k")]:
            v = np.where(o[col][()] > 0, o[col][()], np.nan)
            axs[1].semilogy(q, v, styles[name], color=colr, lw=1.2, label=lab if name == "best_fit" else None)
    qd = c["data"]["qfls"][()]
    for a in axs:
        a.axvspan(qd.min(), qd.max(), color="0.93", zorder=0)
        a.set_xlabel(r"QFLS $\Delta E_\mathrm{F}$ (eV)")
        a.grid(alpha=0.3, which="both")
    axs[0].set_ylabel(r"concentration (cm$^{-3}$)")
    axs[1].set_ylabel(r"rate (cm$^{-3}$ s$^{-1}$)")
    h = [plt.Line2D([], [], color="k", ls=styles[n], label=NAME[n]) for n in ("best_fit", "truth") if n in c]
    axs[0].legend(fontsize=7, loc="lower right")
    axs[1].legend(handles=axs[1].get_legend_handles_labels()[0] + h, fontsize=6.5, loc="lower right", ncol=2)
    lo = np.nanmin([np.nanmin(np.where(c["best_fit"]["ode"][k][()] > 0, c["best_fit"]["ode"][k][()], np.nan))
                    for k, _, _ in rates])
    axs[1].set_ylim(bottom=max(1e-3, 0.3 * lo))
    fig.suptitle(f"{title}: carrier concentrations and trapping / detrapping rates vs QFLS (ODE; shaded: QFLS range of the data)",
                 fontsize=9)
    fig.tight_layout()
    return fig


COLOUR_LABEL = {"rel_drop": r"$(p_\mathrm{best\,fit} - p)\,/\,p_\mathrm{best\,fit}$",
                "rank": r"relative $p(\Theta\,|\,\mathrm{data})$: best N % of the slice points"}
COLOUR_TITLE = {"rel_drop": "(p(best fit) - p) / p(best fit)", "rank": "relative p: best N % of the slice points"}


if __name__ == "__main__":
    main()
