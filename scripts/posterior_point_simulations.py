"""
posterior_point_simulations.py — single parameter sets picked from SMC posteriors, simulated with the ODE (tr-PL) and
the exact ss-PL model, exported for Origin
===================================================================================================================
19/10/2026 (Robin).  Two picks (1-trap synthetic case):

  marginal_map  (--joint run): every parameter at the maximum of its own 1D marginal posterior of the joint SMC
                (the diagonal of the SMC corner: smc_B2nf/smc.h5, corner coordinates n1 / tau_p / tau_n / N_t in
                decades, 120 bins over the prior box, Gaussian smoothing 1.5 bins; the maximum is refined by a parabola
                through the highest bin and its neighbours).  The combination of the marginal maxima is NOT the joint
                maximum; its likelihood is reported next to that of the best fit.
  Nt_1e13       (--trpl run): a good fit with N_t ≈ 1e13 cm^-3 from the tr-PL SMC posterior: SMC particles with
                |log10 N_t - 13| <= --nt-halfwidth, the --n-ode best of them by NN likelihood re-evaluated with the ODE,
                the one with the best ODE likelihood is taken (a posterior particle, not a new fit).

For each pick, in <run>/posterior_points/<pick>/:
  parameters.dat              human-readable parameter set with units (+ likelihoods, RMSE, how it was chosen)
  parameters.json             the same, machine-readable (theta, physical values, plot coordinates)
  trpl_ode_simulation.csv     legacy ODE solution: time_s, pl_norm, qfls_eV, tau_diff_s (+ n, p, n_t, rates)
  trpl_at_data_points.csv     QFLS of the data, tau_diff data / ODE / NN
  sspl_simulation.csv         exact ss-PL model: QFLS, PLQY, n, p, G, suns
  sspl_at_data_points.csv     ss-PL data and model at the measured points
  figures/simulation.png      tau_diff(QFLS), PL(t) and ss-PL with the data and the run's best fit
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def load(run: Path):
    from inference import runs, surrogate
    from inference.prior import Prior
    st = json.loads((run / "settings.json").read_text())
    cfg = st["config"]
    art = surrogate.load_network(cfg["nn"]["dir"], cfg["nn"].get("stem"))
    prior = Prior.from_artifacts(art, Eg=cfg["physics"]["Eg_eV"], order_traps=cfg["prior"]["order_traps"],
                                 photodoping_traps=cfg["prior"]["photodoping_traps"],
                                 bounds_override=cfg["prior"]["bounds_override"], fixed=cfg["prior"].get("fixed", {}),
                                 T=cfg["physics"]["T_K"])
    sig = runs.run_sigmas(run)
    return cfg, st["mode"], art, prior, sig


def marginal_modes(h5_path, keys):
    """Maximum of every 1D marginal of an SMC corner file (parabola through the highest bin and its neighbours)."""
    import h5py
    out = {}
    with h5py.File(h5_path, "r") as f:
        g = f["corner/physical"]
        info = {a: g.attrs[a] for a in ("bins_1d", "smooth_bins") if a in g.attrs}
        for k in keys:
            x, p = g["panels"][k]["x"][()], g["panels"][k]["pdf"][()]
            i = int(np.argmax(p))
            xm = x[i]
            if 0 < i < len(x) - 1:
                y0, y1, y2 = p[i - 1], p[i], p[i + 1]
                den = y0 - 2 * y1 + y2
                if den < 0:
                    xm = x[i] + 0.5 * (y0 - y2) / den * (x[1] - x[0])
            out[k] = {"mode": float(xm), "bin_centre": float(x[i]), "pdf_max": float(p[i]),
                      "median": float(g["panels"][k].attrs["median"])}
    info = {k: (int(v) if k == "bins_1d" else float(v)) for k, v in info.items()}
    info["smooth_1d_bins"] = 1.5 * info.get("smooth_bins", 1.0)        # corner_export.write_corner: 1D smoothing = 1.5 x
    return out, info


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--joint", default="results/1trap_synthetic_paper_B2nf_kradfixed_joint")
    ap.add_argument("--trpl", default="results/1trap_synthetic_paper_B2nf_kradfixed_trpl")
    ap.add_argument("--smc", default="smc_B2nf")
    ap.add_argument("--nt", type=float, default=13.0, help="log10 N_t of the second pick")
    ap.add_argument("--nt-halfwidth", type=float, default=0.1)
    ap.add_argument("--n-ode", type=int, default=40, help="candidates re-evaluated with the ODE")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--picks", default="marginal_map,Nt_1e13",
                    help="comma-separated picks to compute: marginal_map (joint run), Nt_1e13 (tr-PL run)")
    args = ap.parse_args()
    picks = {p.strip() for p in args.picks.split(",") if p.strip()}
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from inference import corner_export as ce, data as dmod, ode, refine, report, runs, surrogate
    from inference.cylinder import DecadeMap
    surrogate.configure_tensorflow()

    rj, rt = Path(args.joint), Path(args.trpl)
    cfg, _, art, prior, sig_j = load(rj)
    cfg_t, _, _, prior_t, sig_t = load(rt)
    assert np.allclose(prior.lo, prior_t.lo) and np.allclose(prior.hi, prior_t.hi)
    npulse, T = cfg["physics"]["npulse_cm3"], cfg["physics"]["T_K"]
    dmap = DecadeMap(prior)
    sp = ce.physical_corner_spec(prior)
    keys = sp["keys"]
    g_sun = dmod.sspl_g_per_sun(cfg, art.constants, prior)[0]

    # likelihoods: tr-PL (NN and ODE) with the sigma of each run, ss-PL (exact) and joint with the joint run's sigmas
    LP = {"joint": runs.make_log_post(cfg, rj, art, prior, "joint", sig_j["trpl"], sig_j["sspl"])[0],
          "sspl": runs.make_log_post(cfg, rj, art, prior, "sspl", None, sig_j["sspl"])[0],
          "trpl_joint_sigma": runs.make_log_post(cfg, rj, art, prior, "trpl", sig_j["trpl"], None)[0],
          "trpl_trpl_run": runs.make_log_post(cfg_t, rt, art, prior, "trpl", sig_t["trpl"], None)[0]}
    PROB = {"joint": refine.make_problem(LP["joint"], art, npulse, T),
            "trpl_trpl_run": refine.make_problem(LP["trpl_trpl_run"], art, npulse, T)}
    best = {"joint": np.array(json.loads((rj / "6_best_fit.json").read_text())["theta_best"], float),
            "trpl": np.array(json.loads((rt / "6_best_fit.json").read_text())["theta_best"], float)}
    tp = cfg["true_params"]
    truth = np.atleast_2d(prior.from_physical(tp["Et_eV"], tp["Nt_cm3"], tp["tau_n_s"], tp["tau_p_s"], tp["krad_cm3s"]))[0]

    def scores(th):
        th = np.asarray(th, float)
        lj, ls, lt = LP["joint"], LP["sspl"], LP["trpl_trpl_run"]
        mse = lj.np_mse_terms(th[None])
        y_ode = ode.ode_log10_tau(th, prior, npulse, art.constants, lj.trpl_terms[0].surrogate.qfls_used, T)
        ok = np.isfinite(y_ode)
        y_obs = np.asarray(lj.trpl_terms[0].y_obs)
        return {"-log L joint (NN tr-PL + ss-PL), joint-run sigmas": float(-lj.np_log_lik_theta(th[None])[0]),
                "-log L joint (ODE tr-PL + ss-PL), joint-run sigmas": float(PROB["joint"].neg_log_lik(th)),
                "-log L tr-PL (NN), tr-PL-run sigma": float(-lt.np_log_lik_theta(th[None])[0]),
                "-log L tr-PL (ODE), tr-PL-run sigma": float(PROB["trpl_trpl_run"].neg_log_lik(th)),
                "-log L ss-PL (exact model), joint-run sigma": float(-ls.np_log_lik_theta(th[None])[0]),
                "RMSE tr-PL NN (decades of tau_diff)": float(np.sqrt(mse["trpl"][0])),
                "RMSE tr-PL ODE (decades of tau_diff)": float(np.sqrt(np.mean((y_ode[ok] - y_obs[ok]) ** 2))),
                "ODE covers tr-PL data points (fraction)": float(ok.mean()),
                f"RMSE ss-PL (exact model, {LP['sspl'].sspl_term.observation.residual_unit})": float(np.sqrt(mse["sspl"][0]))}

    def export(th, out: Path, title: str, how: dict, ref_label: str, ref_theta):
        out.mkdir(parents=True, exist_ok=True)
        (out / "figures").mkdir(exist_ok=True)
        sc, sc_ref = scores(th), scores(ref_theta)
        x = dmap.to_x(th[None])[0]
        plot_xy = {k: float(x[dmap.keys.index(k)]) for k in keys}
        extra = {**{k: f"{v:.6g}" for k, v in sc.items()},
                 **{f"{k} at the {ref_label}": f"{v:.6g}" for k, v in sc_ref.items()},
                 "corner coordinates (decades; n1 replaces E_t)": json.dumps({k: round(v, 5) for k, v in plot_xy.items()}),
                 "n1_cm-3": f"{10 ** plot_xy['log10_n1_1']:.6e}",
                 **{k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in how.items()}}
        report.write_parameters_dat(out / "parameters.dat", prior, th, title=title, extra=extra)
        report.save_json({"title": title, "names": prior.names, "theta": th, "physical": prior.to_physical(th).iloc[0].to_dict(),
                          "corner_coordinates": plot_xy, "scores": sc, f"scores_{ref_label.replace(' ', '_')}": sc_ref,
                          "how_chosen": how}, out / "parameters.json")
        # tr-PL: full legacy ODE solution
        sim = ode.simulate_trpl_full(th, prior, npulse, art.constants, T)
        sim.to_csv(out / "trpl_ode_simulation.csv", index=False, float_format="%.8g")
        lt = LP["trpl_trpl_run"]
        p_nn = lt.predict_trpl(th)
        q_used = p_nn["qfls_used"]
        y_ode = ode.ode_log10_tau(th, prior, npulse, art.constants, q_used, T)
        pd.DataFrame({"qfls_eV": q_used, "tau_diff_data_s": 10 ** p_nn["log10_tau_obs"],
                      "tau_diff_ode_s": 10 ** y_ode, "tau_diff_nn_s": 10 ** p_nn["log10_tau_used"][0]}).to_csv(
            out / "trpl_at_data_points.csv", index=False, float_format="%.8g")
        # ss-PL: exact steady-state model (same model and data as the joint run)
        ps = LP["sspl"].predict_sspl(th)
        c = ps["curve"]
        mode_ss = LP["sspl"].sspl_term.observation.mode
        xs, ys = mode_ss.split("_vs_")[1], mode_ss.split("_vs_")[0]
        pd.DataFrame({f"{xs}_eV" if xs == "qfls" else xs: c["x"][0], ys: c["y"][0], "n_cm3": c["n"][0], "p_cm3": c["p"][0],
                      "G_cm3s": c["G"][0], "suns": c["G"][0] / g_sun}).to_csv(out / "sspl_simulation.csv", index=False,
                                                                             float_format="%.8g")
        lg = LP["sspl"].sspl_term.observation.residual_unit.startswith("log10")   # observations stored as log10 PLQY
        lin = (lambda v: 10 ** np.asarray(v)) if lg else (lambda v: np.asarray(v))
        at = {xs: ps["x_exp"], f"{ys}_data": lin(ps["y_obs"]), f"{ys}_model": lin(ps["y_model"][0])}
        if lg:
            at.update({f"log10_{ys}_data": ps["y_obs"], f"log10_{ys}_model": ps["y_model"][0]})
        pd.DataFrame(at).to_csv(out / "sspl_at_data_points.csv", index=False, float_format="%.8g")
        # preview
        sim_ref = ode.simulate_trpl_full(ref_theta, prior, npulse, art.constants, T)
        c_ref = LP["sspl"].predict_sspl(ref_theta)["curve"]
        fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
        a = axs[0]
        a.semilogy(q_used, 10 ** p_nn["log10_tau_obs"], "o", ms=3, color="k", label="data")
        a.semilogy(sim.qfls_eV, sim.tau_diff_s, "-", color="C3", lw=1.6, label=f"ODE, {out.name}")
        a.semilogy(sim_ref.qfls_eV, sim_ref.tau_diff_s, ":", color="0.3", lw=1.2, label=f"ODE, {ref_label}")
        a.set_xlim(q_used.min() - 0.05, q_used.max() + 0.03)
        a.set_xlabel("QFLS (eV)"); a.set_ylabel(r"$\tau_\mathrm{diff}$ (s)"); a.legend(fontsize=7)
        a = axs[1]
        a.loglog(sim.time_s, sim.pl_norm, "-", color="C3", lw=1.6, label=out.name)
        a.loglog(sim_ref.time_s, sim_ref.pl_norm, ":", color="0.3", lw=1.2, label=ref_label)
        a.set_xlabel("t (s)"); a.set_ylabel(r"PL/PL$_0$"); a.set_ylim(1e-12, 2); a.set_xlim(1e-11, None)
        a.legend(fontsize=7)
        a = axs[2]
        # 20/10/2026 (Robin): model curves only inside the data range, logarithmic x axis
        xe = np.asarray(ps["x_exp"], float)
        x_lo, x_hi = xe.min(), xe.max()
        for cc, sty, col, lw, lab in ((c, "-", "C3", 1.6, out.name), (c_ref, ":", "0.3", 1.2, ref_label)):
            xx, yy = np.asarray(cc["x"][0], float), np.asarray(cc["y"][0], float)
            m = (xx >= x_lo) & (xx <= x_hi)
            a.plot(xx[m], yy[m], sty, color=col, lw=lw, label=lab)
        a.plot(xe, lin(ps["y_obs"]), "o", ms=4, mfc="none", color="k", label="data")
        a.set_xscale("log")
        if "PLQY" in ys:
            a.set_yscale("log")
        a.set_xlabel(xs); a.set_ylabel(ys); a.legend(fontsize=7)
        k_ss = next(k for k in sc if k.startswith("RMSE ss-PL"))
        fig.suptitle(f"{title}\nRMSE tr-PL (ODE) {sc['RMSE tr-PL ODE (decades of tau_diff)']:.3f} dec "
                     f"({ref_label}: {sc_ref['RMSE tr-PL ODE (decades of tau_diff)']:.3f}); {k_ss[:10]} "
                     f"{sc[k_ss]:.4f} ({ref_label}: {sc_ref[k_ss]:.4f}) {k_ss.split(', ')[-1].rstrip(')')}",
                     fontsize=9)
        fig.tight_layout(); report.save_figure(fig, out / "figures" / "simulation.png")
        fig.savefig(out / "figures" / "simulation.pdf"); plt.close(fig)
        print(f"\n== {title}\n" + prior.to_physical(th).T.to_string() + "\n" + json.dumps(sc, indent=1))
        print("written:", out)

    # ── 1. joint SMC: every parameter at the maximum of its marginal ──────────────────
    if "marginal_map" in picks:
        pick_marginal_map(args, rj, keys, dmap, best, export)
    if "Nt_1e13" not in picks:
        return
    pick_nt(args, rt, prior, LP, PROB, best, export)


def pick_marginal_map(args, rj, keys, dmap, best, export):
    modes, binning = marginal_modes(rj / args.smc / "smc.h5", keys)
    x = np.array([modes[k]["mode"] for k in dmap.keys])
    th_map = dmap.to_theta(x[None], best["joint"])[0]
    export(th_map, rj / "posterior_points" / "marginal_map",
           title="joint SMC posterior: every parameter at the maximum of its 1D marginal (1-trap synthetic, joint)",
           how={"method": "maximum of each 1D marginal of the joint NN SMC posterior (corner diagonal), combined",
                "source": str(rj / args.smc / "smc.h5"), "binning": binning,
                "marginal_maxima_corner_coordinates": {k: round(v["mode"], 5) for k, v in modes.items()},
                "marginal_medians_corner_coordinates": {k: round(v["median"], 5) for k, v in modes.items()},
                "note": "the combination of the marginal maxima is not the joint maximum of the posterior"},
           ref_label="best fit", ref_theta=best["joint"])


def pick_nt(args, rt, prior, LP, PROB, best, export):
    # ── 2. tr-PL SMC: a good fit with N_t ≈ 1e13 ─────────────────────────────────────
    ws = pd.read_csv(rt / args.smc / "2_smc_weighted_samples.csv")
    th_all = ws[prior.names].values
    sel = np.abs(ws["log10_Nt_1"].values - args.nt) <= args.nt_halfwidth
    cand = np.unique(th_all[sel], axis=0)
    if not len(cand):
        msg = (f"no tr-PL SMC particle with |log10 N_t - {args.nt:g}| <= {args.nt_halfwidth}; log10 N_t of the particles: "
               f"min {ws['log10_Nt_1'].min():.2f}, weighted quantiles 5/50/95 % "
               f"{np.round(np.interp([0.05, 0.5, 0.95], np.cumsum(ws.sort_values('log10_Nt_1')['weight']) / ws['weight'].sum(), ws.sort_values('log10_Nt_1')['log10_Nt_1']), 2).tolist()}")
        (rt / "posterior_points").mkdir(parents=True, exist_ok=True)
        (rt / "posterior_points" / "Nt_1e13_NOT_FOUND.txt").write_text(msg + "\n")
        print(msg)
        return
    ll_nn = LP["trpl_trpl_run"].np_log_lik_theta(cand)
    top = cand[np.argsort(-ll_nn)[:args.n_ode]]
    nll_ode = np.array([PROB["trpl_trpl_run"].neg_log_lik(t) for t in top])
    i = int(np.argmin(nll_ode))
    w_sel = ws["weight"].values[sel].sum() / ws["weight"].values.sum()
    export(top[i], rt / "posterior_points" / "Nt_1e13",
           title=f"tr-PL SMC posterior particle with N_t ≈ 1e{args.nt:g} cm^-3 (1-trap synthetic, tr-PL only)",
           how={"method": f"tr-PL NN SMC particles with |log10 N_t - {args.nt:g}| <= {args.nt_halfwidth}; the "
                          f"{len(top)} best by NN likelihood re-evaluated with the ODE; best ODE likelihood taken",
                "source": str(rt / args.smc / "2_smc_weighted_samples.csv"),
                "distinct particles in the window": int(len(cand)),
                "posterior mass in the window": round(float(w_sel), 5),
                "-log L tr-PL (ODE) of the candidates: min / median / max": [round(float(nll_ode.min()), 4),
                                                                             round(float(np.median(nll_ode)), 4),
                                                                             round(float(nll_ode.max()), 4)]},
           ref_label="tr-PL best fit", ref_theta=best["trpl"])


if __name__ == "__main__":
    main()
