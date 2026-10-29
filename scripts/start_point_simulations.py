"""
start_point_simulations.py — the physics start point of a fit run, simulated with the ODE (tr-PL) and the exact ss-PL
model, exported for Origin
===================================================================================================================
21/10/2026 (Robin).  main.ipynb writes only the start parameters (start_point.json); this script simulates them.
Same files and models as posterior_point_simulations.py (legacy ODE, exact steady-state ss-PL model, run's sigmas).

In <run>/start_point_simulations/:
  parameters.dat              start point with units (+ likelihoods, RMSE; the same at the run's best fit)
  parameters.json             the same, machine-readable
  trpl_ode_simulation.csv     legacy ODE solution: time_s, pl_norm, qfls_eV, tau_diff_s (+ n, p, n_t, rates)
  trpl_at_data_points.csv     QFLS of the data, tau_diff data / ODE / NN
  sspl_simulation.csv         exact ss-PL model: QFLS, n, p, G, suns
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


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--run", default="results/1trap_synthetic/1trap_85-15_film_fit_joint")
    ap.add_argument("--gpu", default="0")
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from inference import data as dmod, ode, refine, report, runs, surrogate
    from inference.prior import Prior
    surrogate.configure_tensorflow()

    run = Path(args.run)
    st = json.loads((run / "settings.json").read_text())
    cfg = st["config"]
    art = surrogate.load_network(cfg["nn"]["dir"], cfg["nn"].get("stem"))
    prior = Prior.from_artifacts(art, Eg=cfg["physics"]["Eg_eV"], order_traps=cfg["prior"]["order_traps"],
                                 photodoping_traps=cfg["prior"]["photodoping_traps"],
                                 bounds_override=cfg["prior"]["bounds_override"], fixed=cfg["prior"].get("fixed", {}),
                                 T=cfg["physics"]["T_K"])
    sig = runs.run_sigmas(run)
    npulse, T = cfg["physics"]["npulse_cm3"], cfg["physics"]["T_K"]
    g_sun = dmod.sspl_g_per_sun(cfg, art.constants, prior)[0]
    LP = {"joint": runs.make_log_post(cfg, run, art, prior, "joint", sig["trpl"], sig["sspl"])[0],
          "trpl": runs.make_log_post(cfg, run, art, prior, "trpl", sig["trpl"], None)[0],
          "sspl": runs.make_log_post(cfg, run, art, prior, "sspl", None, sig["sspl"])[0]}
    PROB = {m: refine.make_problem(LP[m], art, npulse, T) for m in ("joint", "trpl")}
    sp = json.loads((run / "start_point.json").read_text())
    assert sp["names"] == prior.names
    th = np.array(sp["theta"], float)
    best = np.array(json.loads((run / "6_best_fit.json").read_text())["theta_best"], float)

    def scores(t):
        lj = LP["joint"]
        mse = lj.np_mse_terms(t[None])
        y_ode = ode.ode_log10_tau(t, prior, npulse, art.constants, lj.trpl_terms[0].surrogate.qfls_used, T)
        ok = np.isfinite(y_ode)
        y_obs = np.asarray(lj.trpl_terms[0].y_obs)
        return {"-log L joint (ODE tr-PL + ss-PL)": float(PROB["joint"].neg_log_lik(t)),
                "-log L joint (NN tr-PL + ss-PL)": float(-lj.np_log_lik_theta(t[None])[0]),
                "-log L tr-PL (ODE)": float(PROB["trpl"].neg_log_lik(t)),
                "-log L ss-PL (exact model)": float(-LP["sspl"].np_log_lik_theta(t[None])[0]),
                "RMSE tr-PL ODE (decades of tau_diff)": float(np.sqrt(np.mean((y_ode[ok] - y_obs[ok]) ** 2))),
                "RMSE tr-PL NN (decades of tau_diff)": float(np.sqrt(mse["trpl"][0])),
                "ODE covers tr-PL data points (fraction)": float(ok.mean()),
                f"RMSE ss-PL (exact model, {LP['sspl'].sspl_term.observation.residual_unit})": float(np.sqrt(mse["sspl"][0]))}

    out = run / "start_point_simulations"
    (out / "figures").mkdir(parents=True, exist_ok=True)
    sc, sc_best = scores(th), scores(best)
    title = f"physics start point — {cfg['run_name']} ({st['mode']})"
    extra = {**{k: f"{v:.6g}" for k, v in sc.items()},
             **{f"{k} at the best fit": f"{v:.6g}" for k, v in sc_best.items()},
             "sigma_trpl": sig["trpl"], "sigma_sspl": sig["sspl"],
             "start-point settings": json.dumps(sp["settings"]), "source": str(run / "start_point.json")}
    report.write_parameters_dat(out / "parameters.dat", prior, th, title=title, extra=extra)
    report.save_json({"title": title, "names": prior.names, "theta": th, "physical": prior.to_physical(th).iloc[0].to_dict(),
                      "scores": sc, "scores_best_fit": sc_best, "start_point_settings": sp["settings"]},
                     out / "parameters.json")
    # tr-PL: full legacy ODE solution
    sim = ode.simulate_trpl_full(th, prior, npulse, art.constants, T)
    sim.to_csv(out / "trpl_ode_simulation.csv", index=False, float_format="%.8g")
    p_nn = LP["trpl"].predict_trpl(th)
    q_used = p_nn["qfls_used"]
    y_ode = ode.ode_log10_tau(th, prior, npulse, art.constants, q_used, T)
    pd.DataFrame({"qfls_eV": q_used, "tau_diff_data_s": 10 ** p_nn["log10_tau_obs"],
                  "tau_diff_ode_s": 10 ** y_ode, "tau_diff_nn_s": 10 ** p_nn["log10_tau_used"][0]}).to_csv(
        out / "trpl_at_data_points.csv", index=False, float_format="%.8g")
    # ss-PL: exact steady-state model (same model and data as the run)
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
    sim_ref = ode.simulate_trpl_full(best, prior, npulse, art.constants, T)
    c_ref = LP["sspl"].predict_sspl(best)["curve"]
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
    a = axs[0]
    a.semilogy(q_used, 10 ** p_nn["log10_tau_obs"], "o", ms=3, color="k", label="data")
    a.semilogy(sim.qfls_eV, sim.tau_diff_s, "-", color="C3", lw=1.6, label="ODE, start point")
    a.semilogy(sim_ref.qfls_eV, sim_ref.tau_diff_s, ":", color="0.3", lw=1.2, label="ODE, best fit")
    a.set_xlim(q_used.min() - 0.05, q_used.max() + 0.03)
    a.set_xlabel("QFLS (eV)"); a.set_ylabel(r"$\tau_\mathrm{diff}$ (s)"); a.legend(fontsize=7)
    a = axs[1]
    a.loglog(sim.time_s, sim.pl_norm, "-", color="C3", lw=1.6, label="start point")
    a.loglog(sim_ref.time_s, sim_ref.pl_norm, ":", color="0.3", lw=1.2, label="best fit")
    a.set_xlabel("t (s)"); a.set_ylabel(r"PL/PL$_0$"); a.set_ylim(1e-12, 2); a.set_xlim(1e-11, None)
    a.legend(fontsize=7)
    a = axs[2]
    xe = np.asarray(ps["x_exp"], float)
    x_lo, x_hi = xe.min(), xe.max()
    for cc, sty, col, lw, lab in ((c, "-", "C3", 1.6, "start point"), (c_ref, ":", "0.3", 1.2, "best fit")):
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
                 f"(best fit: {sc_best['RMSE tr-PL ODE (decades of tau_diff)']:.3f}); {k_ss[:10]} "
                 f"{sc[k_ss]:.4f} (best fit: {sc_best[k_ss]:.4f}) {k_ss.split(', ')[-1].rstrip(')')}", fontsize=9)
    fig.tight_layout(); report.save_figure(fig, out / "figures" / "simulation.png")
    fig.savefig(out / "figures" / "simulation.pdf"); plt.close(fig)
    print(f"\n== {title}\n" + prior.to_physical(th).T.to_string() + "\n" + json.dumps(sc, indent=1)
          + "\nbest fit:\n" + json.dumps(sc_best, indent=1))
    print("written:", out)


if __name__ == "__main__":
    main()
