"""
sbc_smc.py — computational simulation-based calibration with tempered SMC for a finished main.ipynb run (any mode)
================================================================================================================
16/10/2026 (revision).  Talts et al. 2018: theta_true ~ prior, synthetic data from the model of the run with the noise
of its curve-level likelihood (sigma = the whole-curve error of the run, fixed and never rescaled; drawn point by point
as iid N(0, N sigma^2), the per-point equivalent of -MSE/(2 sigma^2)), posterior with the
production tempered SMC (inference.smc.run_smc), rank of theta_true among L draws.  Uniform ranks for every parameter
(and derived quantity) = the sampler + likelihood + prior are computationally correct.

The data points are those of the run (tr-PL: the used QFLS points; ss-PL: the measured intensities); fixed parameters
(e.g. k_rad) stay fixed.  One compiled population evaluator serves all data sets (LogPosterior.set_data).

--data-model ode (19/10/2026, variant B): the tr-PL data come from the legacy rate equations (inference.sbc.OdeTrplData:
trpl_module.solve_transient unchanged, nominal QFLS axis as in the ODE-vs-NN comparison; solves in --jobs worker
processes), the inference stays NN-SMC.  Same seed = same theta_true, noise and SMC seeds as the NN-data run, except
for the data sets whose ODE curve does not cover every point (redrawn from the prior on a separate stream; counted).
Modes trpl and joint only (ss-PL has no surrogate).  Default output folder sbc_smc_odedata.

Usage:  python scripts/sbc_smc.py RUN [--n 200] [--n-keep 100] [--particles 2048] [--islands 4] [--gpu 0] [--seed 1]
                                      [--data-model nn|ode] [--jobs 8] [--out DIR]
Output: RUN/sbc_smc/ (RUN/sbc_smc_odedata/) — ranks.csv, ranks_derived.csv, diagnostics.csv, posterior.csv,
        uniformity.csv, calibration.csv, summary.json, figures/
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(a):
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(a.gpu))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from inference import report, runs, sbc, smc, surrogate
    from inference.benchmark import timeit_call
    from inference.prior import Prior
    surrogate.configure_tensorflow()

    run = Path(a.run)
    st = json.loads((run / "settings.json").read_text())
    cfg, mode = st["config"], st["mode"]
    if a.data_model == "ode" and mode == "sspl":
        raise SystemExit("--data-model ode needs a tr-PL term (modes trpl, joint); ss-PL has no surrogate")
    Eg, T = cfg["physics"]["Eg_eV"], cfg["physics"]["T_K"]
    art = surrogate.load_network(cfg["nn"]["dir"], cfg["nn"].get("stem"))
    prior = Prior.from_artifacts(art, Eg=Eg, order_traps=cfg["prior"]["order_traps"],
                                 photodoping_traps=cfg["prior"]["photodoping_traps"],
                                 bounds_override=cfg["prior"]["bounds_override"], fixed=cfg["prior"].get("fixed", {}), T=T)
    sig = runs.run_sigmas(run)
    lp, _ = runs.make_log_post(cfg, run, art, prior, mode, sig["trpl"] if mode != "sspl" else None,
                               sig["sspl"] if mode != "trpl" else None)
    out = run / (a.out or ("sbc_smc" if a.data_model == "nn" else "sbc_smc_odedata"))
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    smc_kw = dict(n_particles=a.particles, n_islands=a.islands, ess_target=0.5, target_accept=0.25, p_move=0.95,
                  min_mh=2, max_mh=30)
    max_batch = a.particles * a.islands
    ev = smc.PopulationEvaluator(lp, max_batch=max_batch, fixed_shape=True)
    _, t_comp = timeit_call(ev.precompile, [max_batch], repeats=1, label="XLA compile of the population evaluator")
    trpl_model, t_warm = None, None
    if a.data_model == "ode":
        trpl_model = sbc.OdeTrplData(lp, prior, art.constants, T=T, n_jobs=a.jobs)
        _, t_warm = timeit_call(trpl_model.warm_up, repeats=1, label=f"ODE worker start-up + numba compile ({a.jobs} processes)")
        print(t_warm, flush=True)
    print(f"run {run.name} | mode {mode} | data model {a.data_model} | sigmas {sig} | {a.n} data sets, L = {a.n_keep}, "
          f"SMC {smc_kw}", flush=True)
    res, t_sbc = timeit_call(sbc.run_sbc_smc, lp, prior, n_datasets=a.n, n_keep=a.n_keep, smc_kwargs=smc_kw,
                             seed=a.seed, evaluator=ev, verbose=True, trpl_model=trpl_model, data_model=a.data_model,
                             repeats=1, label=f"SBC with SMC ({a.n} data sets, {a.data_model} data)")
    print(t_sbc, flush=True)
    if trpl_model is not None:
        oi = res.info["data_model_info"]
        print(f"ODE data: {oi['n_solves']} solves ({res.info['n_redrawn']} of {a.n} first draws redrawn, "
              f"{res.info['n_replacement_draws']} replacement draws) in {oi['time_ode_s']:.1f} s wall on {oi['n_jobs']} "
              f"workers; NN - ODE at theta_true: median RMS {res.diagnostics.rms_nn_minus_data_model.median():.4f} dec",
              flush=True)

    names = list(prior.display_keys)
    pd.concat([pd.DataFrame(prior.to_display(res.theta_true), columns=[f"true_{n}" for n in names]),
               pd.DataFrame(res.ranks, columns=[f"rank_{n}" for n in names])], axis=1).to_csv(out / "ranks.csv", index=False)
    res.ranks_derived.to_csv(out / "ranks_derived.csv", index=False)
    res.diagnostics.to_csv(out / "diagnostics.csv", index=False)
    res.posterior.to_csv(out / "posterior.csv", index=False)
    uni = sbc.uniformity_table(res)
    uni.to_csv(out / "uniformity.csv")
    cal = sbc.calibration_table(res)
    cal.to_csv(out / "calibration.csv")
    lab = {"nn": "NN data", "ode": "ODE data"}[a.data_model]
    fig = sbc.plot_rank_histograms(res, prior)
    report.save_figure(fig, fig_dir / "sbc_rank_histograms.png"); fig.savefig(fig_dir / "sbc_rank_histograms.pdf")
    fig = sbc.plot_rank_histograms(res, prior, derived=True)
    report.save_figure(fig, fig_dir / "sbc_rank_histograms_derived.png")
    for band in ("simultaneous", "pointwise"):
        fig = sbc.plot_ecdf_difference(res, prior, band=band)
        sfx = "" if band == "simultaneous" else "_pointwise"
        report.save_figure(fig, fig_dir / f"sbc_ecdf_difference{sfx}.png"); fig.savefig(fig_dir / f"sbc_ecdf_difference{sfx}.pdf")
    fig = sbc.plot_ecdf_panels({lab: res})
    report.save_figure(fig, fig_dir / "sbc_ecdf_panels.png"); fig.savefig(fig_dir / "sbc_ecdf_panels.pdf")
    fig = sbc.plot_coverage({lab: res})
    report.save_figure(fig, fig_dir / "sbc_coverage.png")
    fig = sbc.plot_zscore_contraction({lab: res})
    report.save_figure(fig, fig_dir / "sbc_zscore_contraction.png")
    if trpl_model is not None:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.hist(res.diagnostics.rms_nn_minus_data_model, bins=30, color="C1")
        ax.set_xlabel("RMS(NN - ODE) at theta_true over the used points [dec]")
        ax.set_ylabel("data sets")
        ax.set_title(f"surrogate error at the {a.n} SBC parameter sets", fontsize=9)
        fig.tight_layout()
        report.save_figure(fig, fig_dir / "sbc_nn_minus_ode_at_truth.png")
    plt.close("all")

    dg = res.diagnostics
    free = [names[i] for i in prior.free_idx]
    fp = [n for n in free if n in uni.index]
    summ = {"run": str(run), "mode": mode, "data_model": a.data_model, "sigmas": sig, "info": res.info,
            "time": {"compile_s": t_comp.mean_s, "total_s": t_sbc.mean_s, "per_dataset_mean_s": float(dg.time_s.mean()),
                     "per_dataset_sd_s": float(dg.time_s.std(ddof=1)),
                     "ode_warm_up_s": None if t_warm is None else t_warm.mean_s},
            "uniformity_p_values": uni["p_value"].to_dict(),
            "min_p_value_parameters": float(uni.loc[fp, "p_value"].min()),
            "ecdf_simultaneous_p_values": uni["p_ecdf_sim"].to_dict(),
            "min_p_ecdf_sim_parameters": float(uni.loc[fp, "p_ecdf_sim"].min()),
            "coverage": cal[[c for c in cal.columns if c.startswith("cov")]].to_dict(orient="index"),
            "contraction_median": cal["contraction_median"].to_dict(),
            "n_redrawn": res.info["n_redrawn"],
            "datasets_max_r_hat_gt_1.01": float((dg.max_r_hat > 1.01).mean()),
            "datasets_max_r_hat_gt_1.05": float((dg.max_r_hat > 1.05).mean()),
            "median_ess_kish": float(dg.ess_kish.median())}
    if trpl_model is not None:
        summ["nn_minus_ode_at_truth"] = {"rms_median": float(dg.rms_nn_minus_data_model.median()),
                                         "rms_p95": float(dg.rms_nn_minus_data_model.quantile(0.95)),
                                         "rms_max": float(dg.rms_nn_minus_data_model.max())}
    (out / "summary.json").write_text(json.dumps(summ, indent=1, default=float))
    print(json.dumps({k: v for k, v in summ.items() if k not in ("info", "coverage")}, indent=1, default=float))
    print("written:", out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("run")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--n-keep", type=int, default=100)
    p.add_argument("--particles", type=int, default=2048)
    p.add_argument("--islands", type=int, default=4)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--gpu", default="0")
    p.add_argument("--data-model", choices=("nn", "ode"), default="nn",
                   help="model of the tr-PL data: the network (default) or the legacy ODE (variant B)")
    p.add_argument("--jobs", type=int, default=8, help="ODE worker processes (--data-model ode)")
    p.add_argument("--out", default=None, help="output folder inside RUN (default sbc_smc / sbc_smc_odedata)")
    main(p.parse_args())
