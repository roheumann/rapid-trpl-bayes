"""
start_point_study.py — how good is the physics-informed start?  CMA-ES (NN) -> curve_fit (ODE) from many starts
==============================================================================================================
19/10/2026 (Robin).  Input: a finished main.ipynb run (mode trpl, sspl or joint).

    python scripts/start_point_study.py results/<run>_<mode> [--n-sobol 64] [--gpu 0]

Same procedure as main.ipynb section 8a (T2), on its own so that it can be rerun without NUTS:
  1. starts: the physics-informed start of the run (start_point.json) + `--n-sobol` scrambled Sobol points over the
     prior (seed `--sobol-seed`, as in 8a); in addition the physics-informed start `--physics-seeds` - 1 more times
     with other CMA-ES seeds (labels physics_s<k>): its success rate, not only one CMA-ES run, is compared with the
     random starts (CMA-ES instance s has seed cma_seed + s);
  2. CMA-ES on the NN likelihood of the run (sigma_eff of the run, fit settings of the config) from every start, all
     instances in lockstep (one batched NN call per generation);
  3. curve_fit with the rate equations (legacy ODE) from every CMA-ES result (modes trpl / joint; the ss-PL model is
     exact, so mode sspl stops after CMA-ES);
  4. comparison with the run's best fit (6_best_fit.json, ODE-refined): Delta(-log L) of every start after CMA-ES and
     after curve_fit, rank of the physics-informed start, fraction of the random starts that reach the best fit,
     evaluations / ODE solves needed, distance of the final parameters from the best fit.
Output in <run>/start_point_study/:
  starts.csv                one row per start (start / CMA-ES / curve_fit values, evaluations, parameters, distances);
                            ranks among physics + Sobol starts (the seed repeats physics_s<k> are not ranked)
  summary.json              fractions within tolerance, rank and evaluations of the physics start vs random starts
  histories_cmaes.csv       convergence curves for Origin: per start the columns <label>_evals, <label>_delta
  histories_curve_fit.csv   same for curve_fit (x = ODE solves; first row = start of curve_fit = CMA-ES result)
  figures/convergence.png   as 6_fit_convergence.png of main.ipynb (red = physics-informed start, first CMA-ES seed)
  figures/convergence_physics_seeds.png   the same with all CMA-ES seeds of the physics-informed start (red)
  figures/start_quality.png start value vs result, sorted results, evaluations;  figures/trajectories_corner.png
y of the convergence curves: -log L - (overall minimum) + 1e-3 (as in main.ipynb); 'delta' columns: -log L minus the
-log L (ODE) of the run's best fit; 'cmaes_delta_nn' and the CMA-ES histories: NN -log L minus the NN maximum.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
TOLS = (0.01, 0.1, 0.5, 1.0)          # -log L units (the run's sigma_eff); 0.5 = sensitivity.tol_neg_log_lik


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("run")
    ap.add_argument("--n-sobol", type=int, default=64)
    ap.add_argument("--sobol-seed", type=int, default=11)
    ap.add_argument("--cma-seed", type=int, default=1)
    ap.add_argument("--physics-seeds", type=int, default=16, help="CMA-ES runs from the physics-informed start")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--n-jobs", type=int, default=None, help="ODE worker processes (default: fit.n_jobs of the config)")
    ap.add_argument("--out", default="start_point_study")
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from inference import optimize, refine, report, runs, surrogate
    from inference.benchmark import count_evals
    from inference.prior import Prior
    surrogate.configure_tensorflow()

    run = Path(args.run)
    st = json.loads((run / "settings.json").read_text())
    cfg, mode = st["config"], st["mode"]
    fc = cfg["fit"]
    n_jobs = args.n_jobs or int(fc.get("n_jobs", 8))
    Eg, npulse, T = cfg["physics"]["Eg_eV"], cfg["physics"]["npulse_cm3"], cfg["physics"]["T_K"]
    art = surrogate.load_network(cfg["nn"]["dir"], cfg["nn"].get("stem"))
    prior = Prior.from_artifacts(art, Eg=Eg, order_traps=cfg["prior"]["order_traps"],
                                 photodoping_traps=cfg["prior"]["photodoping_traps"],
                                 bounds_override=cfg["prior"]["bounds_override"], fixed=cfg["prior"].get("fixed", {}), T=T)
    sig = runs.run_sigmas(run)
    lp, _ = runs.make_log_post(cfg, run, art, prior, mode, sig["trpl"], sig["sspl"])
    best = json.loads((run / "6_best_fit.json").read_text())
    theta_best = np.array(best["theta_best"], float)
    theta_phys = np.array(json.loads((run / "start_point.json").read_text())["theta"], float)
    truth = None
    if cfg.get("true_params"):
        tp = cfg["true_params"]
        truth = np.atleast_2d(prior.from_physical(tp["Et_eV"], tp["Nt_cm3"], tp["tau_n_s"], tp["tau_p_s"], tp["krad_cm3s"]))[0]
    out = run / args.out
    (out / "figures").mkdir(parents=True, exist_ok=True)

    n_rep = max(0, args.physics_seeds - 1)
    starts = np.vstack([theta_phys, prior.sobol(args.n_sobol, seed=args.sobol_seed), np.repeat(theta_phys[None], n_rep, 0)])
    labels = ["physics"] + [f"sobol_{i}" for i in range(args.n_sobol)] + [f"physics_s{k}" for k in range(2, n_rep + 2)]
    print(f"{run.name} | mode {mode} | sigma {sig} | {len(starts)} starts | ODE workers {n_jobs}")

    # ── CMA-ES (NN likelihood), all starts in lockstep ─────────────────────────────
    t_comp = lp.precompile(max_rows=len(starts) * 16)
    with count_evals(lp.counter) as c:
        t0 = time.perf_counter()
        cm = optimize.cmaes_multistart(lp, starts, sigma0=fc["sigma0"], max_evals=fc["max_evals"], tolfun=fc["tolfun"],
                                       tolx=fc["tolx"], seed=args.cma_seed, labels=labels, verbose=False)
        t_cma = time.perf_counter() - t0
    print(f"CMA-ES: {t_cma:.1f} s, {c.counts['n_points']:,} evaluations (XLA compile {t_comp:.1f} s excluded)")

    # ── curve_fit with the rate equations from every CMA-ES result ────────────────
    od, prob, t_ode = None, None, 0.0
    if mode != "sspl":
        prob = refine.make_problem(lp, art, npulse, T)
        t0 = time.perf_counter()
        od = refine.refine_many(prob, np.stack([r.theta for r in cm]), labels=labels, max_nfev=fc["ode_max_nfev"],
                                n_jobs=n_jobs)
        t_ode = time.perf_counter() - t0
        print(f"curve_fit (ODE): {t_ode:.1f} s, {sum(r.n_ode_solves for r in od):,} ODE solves")

    # ── reference: the run's best fit (ODE-refined), with the same likelihood ──────
    nll_nn = lambda th: float(-lp.np_log_lik_theta(np.atleast_2d(th))[0])
    # NN reference: the maximum of the NN likelihood (run's NN maximum or better point found here), as in main 8a
    ref_nn = min(nll_nn(np.array(best["theta_best_nn"], float)), min(r.neg_log_lik for r in cm))
    ref = prob.neg_log_lik(theta_best) if prob is not None else nll_nn(theta_best)   # ODE (exact ss-PL model in mode sspl)
    disp = lambda th: prior.to_display(np.atleast_2d(th))
    keys = [prior.display_keys[i] for i in prior.free_idx]
    d_best = disp(theta_best)[0][prior.free_idx]

    rows = []
    for i, r in enumerate(cm):
        row = {"label": r.label, "is_physics": r.label == "physics", "start_neg_log_lik_nn": nll_nn(starts[i]),
               "cmaes_neg_log_lik_nn": r.neg_log_lik, "cmaes_evals": r.n_evals, "cmaes_iterations": r.n_iter,
               "cmaes_stop": r.stop_reason, "cmaes_delta_nn": r.neg_log_lik - ref_nn}
        for tol in TOLS:
            row[f"cmaes_evals_to_reach_nn_max_plus_{tol:g}"] = r.evals_to_reach(ref_nn + tol)
        final = r.theta
        if od is not None:
            o = od[i]
            row.update({"ode_start_neg_log_lik": o.start_neg_log_lik, "ode_end_neg_log_lik": o.neg_log_lik,
                        "ode_solves": o.n_ode_solves, "delta_neg_log_lik": o.neg_log_lik - ref})
            final = o.theta
        else:
            row["delta_neg_log_lik"] = r.neg_log_lik - ref
        dd = disp(final)[0][prior.free_idx] - d_best
        row.update({f"start_{k}": v for k, v in zip(keys, disp(starts[i])[0][prior.free_idx])})
        row.update({f"final_{k}": v for k, v in zip(keys, disp(final)[0][prior.free_idx])})
        row.update({f"final_minus_best_{k}": v for k, v in zip(keys, dd)})
        row["start_distance_to_best_fit"] = float(np.linalg.norm(disp(starts[i])[0][prior.free_idx] - d_best))
        rows.append(row)
    tab_all = pd.DataFrame(rows)
    tab_all["physics_seed_repeat"] = tab_all.label.str.startswith("physics_s")
    tab_all["cma_seed"] = args.cma_seed + np.arange(len(tab_all))
    tab = tab_all[~tab_all.physics_seed_repeat].copy()                               # physics (first seed) + Sobol
    tab["rank_final"] = tab.delta_neg_log_lik.rank(method="min").astype(int)          # 1 = best of these starts
    tab["rank_start_value"] = tab.start_neg_log_lik_nn.rank(method="min").astype(int)
    tab_all = tab_all.join(tab[["rank_final", "rank_start_value"]])
    tab_all.to_csv(out / "starts.csv", index=False, float_format="%.8g")
    pr = tab_all[tab_all.label.str.startswith("physics")]                              # all CMA-ES seeds of the physics start

    ph, rd = tab.iloc[0], tab.iloc[1:]
    summ = {"run": str(run), "mode": mode, "n_random_starts": len(rd), "sobol_seed": args.sobol_seed,
            "cma_seed": args.cma_seed, "sigma": sig, "fit_settings": {k: fc[k] for k in ("sigma0", "max_evals", "tolfun", "tolx", "ode_max_nfev")},
            "best_fit_of_run": {"neg_log_lik_final_model": ref, "neg_log_lik_nn_at_best_fit": nll_nn(theta_best),
                                "neg_log_lik_nn_maximum": ref_nn,
                                "final_model": "ODE" if prob is not None else "exact ss-PL model",
                                "physical": prior.to_physical(theta_best).iloc[0].to_dict()},
            "best_of_this_study_delta": float(tab.delta_neg_log_lik.min()),
            "time_cmaes_s": t_cma, "evals_cmaes_total": int(c.counts["n_points"]), "time_curve_fit_s": t_ode,
            "ode_solves_total": int(tab.ode_solves.sum()) if od is not None else 0,
            "physics_start": {"start_neg_log_lik_nn": ph.start_neg_log_lik_nn,
                              "start_value_rank": int(ph.rank_start_value), "final_delta": ph.delta_neg_log_lik,
                              "final_rank": int(ph.rank_final), "of": len(tab), "cmaes_evals": int(ph.cmaes_evals),
                              "cmaes_delta_nn": ph.cmaes_delta_nn,
                              "ode_solves": int(ph.ode_solves) if od is not None else None,
                              "start_distance_to_best_fit": ph.start_distance_to_best_fit},
            "random_starts": {"start_neg_log_lik_nn_median": float(rd.start_neg_log_lik_nn.median()),
                              "final_delta_median": float(rd.delta_neg_log_lik.median()),
                              "cmaes_evals_median": float(rd.cmaes_evals.median()),
                              "cmaes_evals_q25_q75": [float(rd.cmaes_evals.quantile(.25)), float(rd.cmaes_evals.quantile(.75))],
                              "ode_solves_median": float(rd.ode_solves.median()) if od is not None else None,
                              "start_distance_to_best_fit_median": float(rd.start_distance_to_best_fit.median())},
            "physics_start_all_seeds": {"n_seeds": len(pr), "final_delta_median": float(pr.delta_neg_log_lik.median()),
                                        "final_delta_min": float(pr.delta_neg_log_lik.min()),
                                        "cmaes_evals_median": float(pr.cmaes_evals.median()),
                                        "ode_solves_median": float(pr.ode_solves.median()) if od is not None else None},
            "within_tolerance": []}
    for tol in TOLS:
        e_col = f"cmaes_evals_to_reach_nn_max_plus_{tol:g}"
        summ["within_tolerance"].append({
            "tol_neg_log_lik": tol,
            "fraction_random_final": float((rd.delta_neg_log_lik <= tol).mean()),
            "physics_final": bool(ph.delta_neg_log_lik <= tol),
            "fraction_physics_seeds_final": float((pr.delta_neg_log_lik <= tol).mean()),
            "fraction_physics_seeds_after_cmaes_nn": float((pr.cmaes_delta_nn <= tol).mean()),
            "fraction_random_after_cmaes_nn": float((rd.cmaes_delta_nn <= tol).mean()),
            "physics_after_cmaes_nn": bool(ph.cmaes_delta_nn <= tol),
            "cmaes_evals_to_reach_physics": None if np.isnan(ph[e_col]) else float(ph[e_col]),
            "cmaes_evals_to_reach_random_median_of_those_reaching": float(np.nanmedian(rd[e_col])) if rd[e_col].notna().any() else None,
            "random_reaching_with_cmaes": int(rd[e_col].notna().sum())})
    report.save_json(summ, out / "summary.json")
    print(json.dumps(summ["physics_start"], indent=1, default=float))
    print(pd.DataFrame(summ["within_tolerance"]).round(3).to_string(index=False))

    # ── convergence curves for Origin (wide: one column pair per start, NaN-padded) ─
    def wide(curves):
        n = max(len(x) for x, _ in curves.values())
        cols = {}
        for lb, (x, y) in curves.items():
            cols[f"{lb}_x"] = np.r_[x, np.full(n - len(x), np.nan)]
            cols[f"{lb}_delta"] = np.r_[y, np.full(n - len(y), np.nan)]
        return pd.DataFrame(cols)
    wide({r.label: (r.history_evals, r.history_best - ref_nn) for r in cm}).to_csv(
        out / "histories_cmaes.csv", index=False, float_format="%.8g")
    if od is not None:
        wide({o.label: (np.r_[0, o.history_evals if o.history_evals is not None else []],
                        np.r_[o.start_neg_log_lik, o.history_best if o.history_best is not None else []] - ref)
              for o in od}).to_csv(out / "histories_curve_fit.csv", index=False, float_format="%.8g")

    # ── figures ───────────────────────────────────────────────────────────────────
    sel = [i for i, lb in enumerate(labels) if not lb.startswith("physics_s")]
    what = "CMA-ES (NN) then curve_fit (ODE)" if od is not None else "CMA-ES (exact ss-PL model)"
    fig, _ = report.plot_fit_histories([cm[i] for i in sel], None if od is None else [od[i] for i in sel],
                                       title=f"physics-informed + {args.n_sobol} Sobol starts ({mode}): {what}")
    report.save_figure(fig, out / "figures" / "convergence.png"); fig.savefig(out / "figures" / "convergence.pdf")
    plt.close(fig)
    if n_rep:
        fig, _ = report.plot_fit_histories(cm, od, title=f"{len(pr)} CMA-ES seeds of the physics-informed start (red) + "
                                                         f"{args.n_sobol} Sobol starts ({mode}): {what}")
        report.save_figure(fig, out / "figures" / "convergence_physics_seeds.png"); plt.close(fig)

    fig, axs = plt.subplots(1, 3, figsize=(13, 3.9))
    y = tab.delta_neg_log_lik.clip(lower=0) + 1e-3
    a = axs[0]
    a.scatter(rd.start_neg_log_lik_nn - ref_nn, y.iloc[1:], s=14, c="0.4", label=f"{len(rd)} Sobol starts")
    a.scatter([ph.start_neg_log_lik_nn - ref_nn], [y.iloc[0]], s=60, c="C3", ec="k", zorder=5, label="physics-informed start")
    a.set_xscale("symlog", linthresh=1); a.set_yscale("log")
    a.set_xlabel(r"$-\log L$ at the start (NN) $-$ best fit"); a.set_ylabel(r"final $-\log L$ $-$ best fit $+10^{-3}$")
    a.legend(fontsize=7)
    a = axs[1]
    o_ = np.argsort(tab.delta_neg_log_lik.values)
    a.semilogy(np.arange(1, len(tab) + 1), tab.cmaes_delta_nn.clip(lower=0).values[o_] + 1e-3, ".", color="0.6",
               label="after CMA-ES (NN, minus the NN maximum)")
    a.semilogy(np.arange(1, len(tab) + 1), y.values[o_], "o", ms=3.5, color="0.2",
               label="after curve_fit (ODE)" if od is not None else "after CMA-ES")
    i_ph = int(np.nonzero(o_ == 0)[0][0])
    a.semilogy([i_ph + 1], [y.iloc[0]], "o", ms=8, mfc="C3", mec="k", label="physics-informed start")
    for tol in (0.1, 0.5):
        a.axhline(tol, color="C0", ls=":", lw=0.8)
    a.set_xlabel("start, sorted by the final value"); a.set_ylabel(r"$-\log L$ $-$ best fit $+10^{-3}$")
    a.legend(fontsize=7)
    a = axs[2]
    yy = lambda d: d.delta_neg_log_lik.clip(lower=0) + 1e-3
    a.scatter(rd.cmaes_evals, yy(rd), s=14, c="0.4", label="Sobol starts")
    if len(pr) > 1:
        a.scatter(pr.cmaes_evals.iloc[1:], yy(pr).iloc[1:], s=22, marker="D", c="none", ec="C3",
                  label=f"physics-informed start, {len(pr) - 1} other CMA-ES seeds")
    a.scatter([ph.cmaes_evals], [y.iloc[0]], s=60, c="C3", ec="k", zorder=5, label="physics-informed start")
    a.set_yscale("log"); a.set_xlabel("NN evaluations until CMA-ES stops")
    a.set_ylabel(r"final $-\log L$ $-$ best fit $+10^{-3}$"); a.legend(fontsize=7)
    w05 = summ["within_tolerance"][TOLS.index(0.5)]
    fig.suptitle(f"{run.name}: physics-informed start rank {int(ph.rank_final)} of {len(tab)} "
                 f"(Δ = {ph.delta_neg_log_lik:.3g}); within 0.5 of the best fit: Sobol starts "
                 f"{100 * w05['fraction_random_final']:.0f} %, physics start {100 * w05['fraction_physics_seeds_final']:.0f} % "
                 f"of {len(pr)} CMA-ES seeds", fontsize=9)
    fig.tight_layout(); report.save_figure(fig, out / "figures" / "start_quality.png")
    fig.savefig(out / "figures" / "start_quality.pdf"); plt.close(fig)

    fig = report.plot_trajectory_corner(prior, cm, od, truth=None if truth is None else disp(truth)[0],
                                        title=f"optimiser trajectories of {len(cm)} starts (red: physics-informed)")
    report.save_figure(fig, out / "figures" / "trajectories_corner.png"); plt.close(fig)
    print("written:", out)


if __name__ == "__main__":
    main()
