"""
timing_repeats.py — mean ± SD of wall time AND number of evaluations over independent runs (different seeds)
===========================================================================================================
19/10/2026 (Robin).  Input: a finished main.ipynb run (mode trpl, sspl or joint) and its SMC folder.

    python scripts/timing_repeats.py results/<run>_<mode> [--repeats 5] [--smc smc_B2nf] [--gpu 0] [--quick]

The T8 table of main.ipynb times IDENTICAL repeats (same seeds), so the number of evaluations has no spread and NUTS /
emcee were run once (n = 1).  Here every method of the NN workflow is run --repeats times with the settings of the run
but a different random seed per repeat (repeat 0 = the seeds of the original run):

  CMA-ES     every fit the mode needs, as in main.ipynb 6a/6b: tr-PL (physics-informed start + n_extra_sobol_starts
             Sobol starts, CMA-ES + L-BFGS polish, sigma of the config), for sspl / joint also the ss-PL-only fit and
             for joint the joint fit (starts: ODE-refined tr-PL best fit, physics, Sobol; sigma_eff of the run).
             Seeds: Sobol scramble seed 1 + r, CMA-ES seed 1 + 1000 r.
  curve_fit  ODE refinement of every CMA-ES result of the same repeat: tr-PL fit (+ joint fit in mode joint);
             evaluations = ODE solves (parallel worker processes, fit.n_jobs).
  NUTS       config nuts block (chains, warm-up, draws, dense metric, ...), dispersed start; seed nuts.seed + r.
  emcee      config emcee block; seed emcee.seed + r.
  SMC        settings of <run>/<smc>/settings.json (particles, islands, ...); seed SEED + r.
  slices     paper's sliced grid (100^2 per pair) at the run's best fit — deterministic: the evaluations have SD 0,
             only the time varies.
One untimed warm-up before the timed repeats of every method (XLA compilation, worker pools, numba caches; NUTS: a short
untimed run).  ODE workflow: evaluations x (one ODE solve at the best fit, T8 row of the run), per repeat
(NUTS: finite-difference gradients, (D_free + 1) solves per leapfrog step incl. warm-up; mode sspl: the same as NN).
Output <run>/timing_repeats/: raw.csv (one row per method and repeat), summary.csv (mean ± SD), settings.json.
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
METHODS = ("cmaes", "curve_fit", "nuts", "emcee", "smc", "slices")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("run")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--smc", default="smc_B2nf")
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--out", default="timing_repeats")
    ap.add_argument("--quick", action="store_true", help="smoke test: tiny sizes (NOT for results)")
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf
    from inference import optimize, refine, report, runs, samplers, slices, smc, surrogate
    from inference.prior import Prior
    surrogate.configure_tensorflow()

    run = Path(args.run)
    st = json.loads((run / "settings.json").read_text())
    cfg, mode = st["config"], st["mode"]
    methods = [m for m in args.methods.split(",") if m]
    R = int(args.repeats)
    fc, nc, ec = dict(cfg["fit"]), dict(cfg["nuts"]), dict(cfg["emcee"])
    smc_set = json.loads((run / args.smc / "settings.json").read_text())
    SMC, SMC_SEED, MAX_BATCH = dict(smc_set["smc"]), int(smc_set["seed"]), int(smc_set["max_batch"])
    n_sl = 100
    if args.quick:
        fc.update(max_evals=2000, n_extra_sobol_starts=2, ode_max_nfev=8)
        nc.update(n_chains=4, n_warmup=60, n_draws=60)
        ec.update(n_steps=400)
        SMC.update(n_particles=256, max_mh=4)
        n_sl = 12
    Eg, npulse, T = cfg["physics"]["Eg_eV"], cfg["physics"]["npulse_cm3"], cfg["physics"]["T_K"]
    art = surrogate.load_network(cfg["nn"]["dir"], cfg["nn"].get("stem"))
    prior = Prior.from_artifacts(art, Eg=Eg, order_traps=cfg["prior"]["order_traps"],
                                 photodoping_traps=cfg["prior"]["photodoping_traps"],
                                 bounds_override=cfg["prior"]["bounds_override"], fixed=cfg["prior"].get("fixed", {}), T=T)
    D, D_free = prior.dim, len(prior.free_idx)
    free_keys = [prior.display_keys[i] for i in prior.free_idx]
    sig_run = runs.run_sigmas(run)                                     # sigma_eff of the run (samplers, joint fit)
    sig_base = {"trpl": float(cfg["trpl"]["sigma"]), "sspl": float(cfg["sspl"]["sigma"]) if cfg["sspl"].get("use") else None}
    best = json.loads((run / "6_best_fit.json").read_text())
    if best["sigmas"].get("trpl_fit") is not None:      # main.ipynb >= 20/10/2026: the sigmas its section-6a fits used
        sig_base = {"trpl": float(best["sigmas"]["trpl_fit"]),
                    "sspl": float(best["sigmas"]["sspl_fit"]) if best["sigmas"].get("sspl_fit") is not None else None}
    theta_best = np.array(best["theta_best"], float)
    theta_phys = np.array(json.loads((run / "start_point.json").read_text())["theta"], float)
    t8 = pd.read_csv(run / "11_T8_timing_table.csv")
    r_ode = t8[t8.step.str.startswith("one ODE solve")]
    t_ode = float(r_ode.ode_mean_s.iloc[0]) if len(r_ode) else np.nan
    bs = cfg["hardware"]["batch_size"]
    has_ss = mode in ("sspl", "joint")
    # likelihoods: fits as in main.ipynb (tr-PL and ss-PL fit with the config sigmas, joint fit with sigma_eff),
    # samplers with sigma_eff of the run, slices with the config sigmas (paper_slices.py)
    LPF = {"trpl": runs.make_log_post(cfg, run, art, prior, "trpl", sig_base["trpl"], None, bs)[0]}
    if has_ss:
        LPF["sspl"] = runs.make_log_post(cfg, run, art, prior, "sspl", None, sig_base["sspl"], bs)[0]
    if mode == "joint":
        LPF["joint"] = runs.make_log_post(cfg, run, art, prior, "joint", sig_run["trpl"], sig_run["sspl"], bs)[0]
    lp = runs.make_log_post(cfg, run, art, prior, mode, sig_run["trpl"] if mode != "sspl" else None,
                            sig_run["sspl"] if has_ss else None, bs)[0]
    out = run / (args.out + ("_QUICK" if args.quick else ""))
    out.mkdir(parents=True, exist_ok=True)
    report.save_json({"run": str(run), "mode": mode, "repeats": R, "methods": methods, "sigma_run": sig_run,
                      "sigma_base": sig_base, "fit": fc, "nuts": nc, "emcee": ec, "smc": SMC, "smc_seed": SMC_SEED,
                      "max_batch": MAX_BATCH, "slices_n": n_sl, "ode_seconds_per_solve": t_ode, "quick": args.quick,
                      "device": tf.config.list_physical_devices("GPU")[0].name if tf.config.list_physical_devices("GPU") else "CPU",
                      "cpu_count": os.cpu_count()}, out / "settings.json")
    RAW = []

    def rec(method, r, seed, t, n_evals, **kw):
        row = {"method": method, "repeat": r, "seed": seed, "time_s": t, "n_evals": n_evals, **kw}
        RAW.append(row)
        pd.DataFrame(RAW).to_csv(out / "raw.csv", index=False)
        print(f"[{time.strftime('%H:%M:%S')}] {method:9s} r={r} seed={seed}: {t:9.2f} s, {n_evals:,} evaluations "
              + " ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in kw.items()), flush=True)

    # ── CMA-ES (+ L-BFGS) and curve_fit, as in main.ipynb 6a / 6b ────────────────────────
    if "cmaes" in methods or "curve_fit" in methods:
        n_sob = int(fc["n_extra_sobol_starts"])
        S = 1 + n_sob + (1 if has_ss else 0)
        for l_ in LPF.values():
            l_.precompile(max_rows=S * 16)
            l_.tf_neg_log_lik_and_grad_u(tf.zeros([max(1, fc["n_polish"]), D], tf.float64))
        probs = {m: refine.make_problem(LPF[m], art, npulse, T) for m in ("trpl", "joint") if m in LPF}

        def one_fit(m, starts, labels, seed):
            t0 = time.perf_counter()
            res = optimize.fit(LPF[m], starts, labels=labels, n_polish=fc["n_polish"], verbose=False,
                               cmaes_kwargs=dict(sigma0=fc["sigma0"], max_evals=fc["max_evals"], tolfun=fc["tolfun"],
                                                 tolx=fc["tolx"], seed=seed, verbose=False))
            return res, time.perf_counter() - t0, res.n_evals_cmaes + res.n_evals_lbfgs

        def one_refine(m, res):
            t0 = time.perf_counter()
            od = refine.refine_many(probs[m], np.stack([x.theta for x in res.cmaes]), labels=[x.label for x in res.cmaes],
                                    max_nfev=fc["ode_max_nfev"], n_jobs=int(fc["n_jobs"]))
            return od, time.perf_counter() - t0, int(sum(o.n_ode_solves for o in od))

        for r in range(-1, R):                       # r = -1: untimed warm-up
            rr = max(r, 0)
            sob_seed, cma_seed = 1 + rr, 1 + 1000 * rr
            sobol = prior.sobol(n_sob, seed=sob_seed) if n_sob else np.zeros((0, D))
            starts = np.vstack([theta_phys, sobol])
            labels = ["physics"] + [f"sobol_{i}" for i in range(len(sobol))]
            t_cm = n_cm = t_cf = n_cf = 0.0
            res, t, n = one_fit("trpl", starts, labels, cma_seed)                         # 6a tr-PL
            t_cm, n_cm = t_cm + t, n_cm + n
            parts = {"trpl_cmaes_s": t, "trpl_cmaes_evals": n}
            od, t, n = one_refine("trpl", res)
            t_cf, n_cf = t_cf + t, n_cf + n
            parts.update(trpl_curve_fit_s=t, trpl_curve_fit_evals=n)
            best_tr = min(od, key=lambda o: o.neg_log_lik).theta
            if has_ss:                                                                     # 6b ss-PL only
                res_s, t, n = one_fit("sspl", np.vstack([best_tr, starts]), ["trpl_best"] + labels, cma_seed)
                t_cm, n_cm = t_cm + t, n_cm + n
                parts.update(sspl_cmaes_s=t, sspl_cmaes_evals=n)
            nll_final = min(o.neg_log_lik for o in od)
            if mode == "joint":                                                            # 6b joint
                res_j, t, n = one_fit("joint", np.vstack([best_tr, starts]), ["trpl_best"] + labels, cma_seed)
                t_cm, n_cm = t_cm + t, n_cm + n
                parts.update(joint_cmaes_s=t, joint_cmaes_evals=n)
                od_j, t, n = one_refine("joint", res_j)
                t_cf, n_cf = t_cf + t, n_cf + n
                parts.update(joint_curve_fit_s=t, joint_curve_fit_evals=n)
                nll_final = min(o.neg_log_lik for o in od_j)
            elif mode == "sspl":
                nll_final = res_s.best.neg_log_lik
            if r < 0:
                print(f"warm-up fit done ({t_cm:.1f} s CMA-ES, {t_cf:.1f} s curve_fit)", flush=True)
                continue
            if "cmaes" in methods:
                rec("cmaes", r, cma_seed, t_cm, int(n_cm), sobol_seed=sob_seed, best_neg_log_lik_final=float(nll_final),
                    **{k: v for k, v in parts.items() if "cmaes" in k})
            if "curve_fit" in methods:
                rec("curve_fit", r, cma_seed, t_cf, int(n_cf), sobol_seed=sob_seed,
                    **{k: v for k, v in parts.items() if "curve_fit" in k})

    # ── NUTS ──────────────────────────────────────────────────────────────────────────────
    if "nuts" in methods:
        kw = dict(metric=nc["metric"], target_accept=nc["target_accept"], max_tree_depth=nc["max_tree_depth"], verbose=False)
        init0 = samplers.init_dispersed(D, nc["n_chains"], np.random.default_rng(nc["seed"]))
        t0 = time.perf_counter()
        _, sampler = samplers.run_nuts(lp, init0, n_warmup=1, n_draws=1, seed=nc["seed"], **kw)
        print(f"NUTS compile (short untimed run): {time.perf_counter() - t0:.0f} s", flush=True)
        for r in range(R):
            seed = int(nc["seed"]) + r
            init = samplers.init_dispersed(D, nc["n_chains"], np.random.default_rng(seed)) if nc["init"] == "dispersed" \
                else samplers.init_around(prior.inverse(np.array(best["theta_best_nn"])[None])[0], nc["n_chains"],
                                          np.random.default_rng(seed))
            t0 = time.perf_counter()
            res, sampler = samplers.run_nuts(lp, init, n_warmup=nc["n_warmup"], n_draws=nc["n_draws"], seed=seed,
                                             sampler=sampler, **kw)
            t = time.perf_counter() - t0
            info = res.info
            dg = samplers.diagnostics(res, prior).loc[free_keys]
            lf = info["leapfrogs_sampling"] * (1 + info["n_warmup"] / info["n_draws"])
            rec("nuts", r, seed, t, int(info["evals_total_points"]), min_ess_bulk=float(dg.ess_bulk.min()),
                max_r_hat=float(dg.r_hat.max()), n_divergent=int(info["n_divergent"]), step_size=float(info["step_size"]),
                leapfrogs_incl_warmup=float(lf), compile_s=float(info.get("compile_s") or 0.0),
                ode_solves_fd=float(lf * (D_free + 1)))

    # ── emcee ─────────────────────────────────────────────────────────────────────────────
    if "emcee" in methods:
        lp.precompile(max_rows=ec["n_walkers"], which=("lp",))
        w0 = samplers.init_dispersed(D, ec["n_walkers"], np.random.default_rng(ec["seed"]))
        samplers.run_emcee(lp, w0, n_steps=min(200, ec["n_steps"]), burn_frac=ec["burn_frac"], moves=ec["moves"],
                           seed=ec["seed"])                                                       # warm-up
        for r in range(R):
            seed = int(ec["seed"]) + r
            init = samplers.init_dispersed(D, ec["n_walkers"], np.random.default_rng(seed))
            t0 = time.perf_counter()
            res = samplers.run_emcee(lp, init, n_steps=ec["n_steps"], burn_frac=ec["burn_frac"], moves=ec["moves"], seed=seed)
            t = time.perf_counter() - t0
            dg = samplers.diagnostics(res, prior).loc[free_keys]
            rec("emcee", r, seed, t, int(res.info["evals_total_points"]), min_ess_bulk=float(dg.ess_bulk.min()),
                max_r_hat=float(dg.r_hat.max()), acceptance_fraction=float(res.info["acceptance_fraction"]),
                max_tau_int=float(np.max(res.info["tau_int"])))

    # ── tempered SMC ──────────────────────────────────────────────────────────────────────
    if "smc" in methods:
        ev = smc.PopulationEvaluator(lp, max_batch=MAX_BATCH, fixed_shape=True)
        ev.precompile([MAX_BATCH])
        smc.run_smc(lp, **SMC, seed=SMC_SEED, evaluator=ev, verbose=False)                          # warm-up
        for r in range(R):
            seed = SMC_SEED + r
            t0 = time.perf_counter()
            res = smc.run_smc(lp, **SMC, seed=seed, evaluator=ev, verbose=False)
            t = time.perf_counter() - t0
            dg = samplers.diagnostics(res, prior).loc[free_keys]
            rec("smc", r, seed, t, int(res.info["evals_total_points"]), min_ess_bulk=float(dg.ess_bulk.min()),
                max_r_hat=float(dg.r_hat.max()), ess_kish=float(res.info["ess_weighted_kish"]),
                log_z=float(res.log_evidence), log_z_se=float(res.log_evidence_se),
                n_stages=json.dumps([int(x) for x in np.atleast_1d(res.info["n_stages"])]))

    # ── sliced grid (paper method) ──────────────────────────────────────────────────────
    if "slices" in methods:
        add_mse_min = cfg.get("sigma_rule") != "model_plus_nn"
        sb = dict(sig_base) if add_mse_min else {k: v for k, v in sig_run.items()}
        lp_sl = runs.make_log_post(cfg, run, art, prior, mode, sb["trpl"] if mode != "sspl" else None,
                                   sb["sspl"] if has_ss else None, bs)[0]
        slices.run_slices(lp_sl, prior, theta_best, n=n_sl, sigma_base=sb, add_mse_min=add_mse_min)  # warm-up
        for r in range(R):
            t0 = time.perf_counter()
            S_ = slices.run_slices(lp_sl, prior, theta_best, n=n_sl, sigma_base=sb, add_mse_min=add_mse_min)
            rec("slices", r, 0, time.perf_counter() - t0, int(S_["n_evals"]))

    # ── summary: mean ± SD over the repeats ─────────────────────────────────────────────
    raw = pd.DataFrame(RAW)
    rows = []
    for m, g in raw.groupby("method", sort=False):
        n_ev = g.n_evals.astype(float)
        if m == "curve_fit":
            ode = g.time_s.values                                    # ODE in both workflows
        elif m == "cmaes":                                           # ss-PL-only fit: exact model in both workflows
            n_ss = g.sspl_cmaes_evals.values if "sspl_cmaes_evals" in g else 0.0
            t_ss = g.sspl_cmaes_s.values if "sspl_cmaes_s" in g else 0.0
            ode = (n_ev.values - n_ss) * t_ode + t_ss
        elif mode == "sspl":
            ode = g.time_s.values                                    # exact ss-PL model: same computation
        elif m == "nuts":
            ode = g.ode_solves_fd.values * t_ode                     # finite-difference gradients
        else:
            ode = n_ev.values * t_ode
        row = {"method": m, "n_repeats": len(g), "time_mean_s": g.time_s.mean(), "time_sd_s": g.time_s.std(ddof=1),
               "evals_mean": n_ev.mean(), "evals_sd": n_ev.std(ddof=1), "evals_min": n_ev.min(), "evals_max": n_ev.max()}
        ode = np.asarray(ode, float)
        row.update(ode_est_mean_s=float(np.mean(ode)), ode_est_sd_s=float(np.std(ode, ddof=1)) if len(ode) > 1 else 0.0)
        for c in ("min_ess_bulk", "max_r_hat", "log_z", "ess_kish"):
            if c in g and g[c].notna().any():
                row[f"{c}_mean"], row[f"{c}_sd"] = float(g[c].mean()), float(g[c].std(ddof=1))
        rows.append(row)
    summ = pd.DataFrame(rows)
    summ.to_csv(out / "summary.csv", index=False)
    print(summ.round(4).to_string(index=False))
    print("written:", out)


if __name__ == "__main__":
    main()
