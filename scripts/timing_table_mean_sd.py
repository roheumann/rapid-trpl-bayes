"""
timing_table_mean_sd.py — ONE timing table: earlier T8 results + mean ± SD of time and evaluations over independent runs
=====================================================================================================================
19/10/2026 (Robin).  Input: run folders with 11_T8_timing_table.csv (main.ipynb) and timing_repeats/summary.csv
(scripts/timing_repeats.py).

    python scripts/timing_table_mean_sd.py results/<run>_trpl results/<run>_sspl results/<run>_joint --out results/<name>

Rows: every row of the earlier T8 table of each run (offline costs, start values, one ODE solve, grid, cylinder IS, ...)
plus, where missing there, the tempered-SMC row of smc.ipynb and the measured SMC with the rate equations
(ode_vs_nn).  For CMA-ES, curve_fit, NUTS, emcee, SMC and the sliced grid the new columns give mean ± SD over
independent runs with different seeds (n in the table):
  time_new            total wall time of one run (s / min / h), mean ± SD
  evaluations_new     likelihood evaluations of one run (curve_fit: ODE solves), mean ± SD
  ode_workflow_new    the same with the rate equations: evaluations x one ODE solve per run (NUTS: finite-difference
                      gradients), mean ± SD over the runs; curve_fit and mode sspl: same as the NN workflow
The earlier text columns (nn_workflow_previous, ode_workflow_previous) are kept unchanged.
Output: <out>.csv, <out>.md and per run <run>/11_T8_timing_table_mean_sd.csv.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

STEP_OF = {"cmaes": "optimisation (CMA-ES", "curve_fit": "local refinement curve_fit", "nuts": "posterior sampling: NUTS",
           "emcee": "posterior sampling: emcee", "smc": "posterior sampling: tempered SMC", "slices": "sliced grid"}
NAME_OF = {"cmaes": "optimisation (CMA-ES + L-BFGS, all starts and fits of the mode)",
           "curve_fit": "local refinement curve_fit (ODE in both workflows)", "nuts": "posterior sampling: NUTS",
           "emcee": "posterior sampling: emcee", "smc": "posterior sampling: tempered SMC",
           "slices": "sliced grid at the best fit (paper method, 100^2 per pair)"}


def fmt_t(m, s, n=None, prefix=""):
    if m is None or not np.isfinite(m):
        return "—"
    unit, f = ("ms", 1e3) if m < 0.1 else ("s", 1.0) if m < 120 else ("min", 1 / 60) if m < 7200 else ("h", 1 / 3600)
    mm, ss = m * f, (s or 0.0) * f
    dec = int(min(2, max(0, 2 - np.floor(np.log10(abs(mm)))))) if mm else 0
    txt = f"{prefix}{mm:.{dec}f}" + (f" ± {ss:.{dec}f}" if (n is None and ss > 0) or (n and n > 1) else "") + f" {unit}"
    return txt + (f" (n = {n})" if n else "")


def fmt_n(m, s):
    return f"{m:,.0f} ± {s:,.0f}"


def mode_table(run: Path) -> pd.DataFrame:
    st = json.loads((run / "settings.json").read_text())
    mode = st["mode"]
    prev = pd.read_csv(run / "11_T8_timing_table.csv")
    new = pd.read_csv(run / "timing_repeats" / "summary.csv").set_index("method")
    rows = []
    for r in prev.itertuples():
        rows.append({"mode": mode, "stage": r.stage, "step": r.step, "nn_workflow_previous": r.nn_workflow,
                     "ode_workflow_previous": r.ode_workflow, "_key": None})
    have = " | ".join(prev.step)
    smc_dir = next((d for d in (run / "smc_B2nf", run / "smc") if (d / "summary.json").exists()), None)
    if "tempered SMC" not in have and smc_dir is not None:          # earlier SMC run of smc.ipynb (not in this T8 table)
        s = json.loads((smc_dir / "summary.json").read_text())["smc"]
        n_rep = int(pd.read_csv(smc_dir / "8_timing.csv").query("label == 'tempered SMC'").repeats.iloc[0])
        rows.append({"mode": mode, "stage": "per sample", "step": f"posterior sampling: tempered SMC (smc.ipynb, {smc_dir.name})",
                     "nn_workflow_previous": f"{fmt_t(s['time_mean_s'], s['time_sd_s'], n_rep)}; {s['evals']:,} evaluations; "
                                             f"min ESS {s['min_ess_bulk']:,.0f}; R-hat <= {s['max_r_hat']:.3f}; "
                                             f"log Z = {s['log_z']:.3f} ± {s['log_z_se']:.3f} (identical repeats, same seed)",
                     "ode_workflow_previous": "—", "_key": None})
    ode_info = run / "ode_vs_nn" / "ode_smc_info.json"
    if ode_info.exists():                                            # measured SMC with the rate equations
        o = json.loads(ode_info.read_text())
        ns = json.loads((run / "ode_vs_nn" / "settings.json").read_text())
        rows.append({"mode": mode, "stage": "per sample",
                     "step": "posterior sampling: tempered SMC with the rate equations (ode_inference.ipynb, measured)",
                     "nn_workflow_previous": "—",
                     "ode_workflow_previous": f"{fmt_t(o['time_total_s'], 0, 1)}; {o['evals_total_points']:,} ODE solves on "
                                              f"{ns.get('n_jobs')} worker processes; log Z = {o['log_evidence']:.3f} ± "
                                              f"{o['log_evidence_se']:.3f}", "_key": None})
    tab = pd.DataFrame(rows)
    for m, row in new.iterrows():                                    # attach the new mean ± SD columns
        hit = tab.index[tab.step.str.startswith(STEP_OF[m])]
        if len(hit):
            i = hit[0]
        else:
            tab.loc[len(tab)] = {"mode": mode, "stage": "per sample", "step": NAME_OF[m], "nn_workflow_previous": "—",
                                 "ode_workflow_previous": "—", "_key": None}
            i = tab.index[-1]
        n = int(row.n_repeats)
        tab.loc[i, "_key"] = m
        tab.loc[i, "time_new"] = fmt_t(row.time_mean_s, row.time_sd_s, n)
        tab.loc[i, "evaluations_new"] = fmt_n(row.evals_mean, row.evals_sd) + (" ODE solves" if m == "curve_fit" else "")
        if m == "curve_fit":
            ode_txt = "same as NN workflow (ODE in both)"
        elif mode == "sspl" and m != "cmaes":
            ode_txt = "same as NN workflow (exact ss-PL model, no surrogate)"
        else:
            note = {"nuts": "estimated (FD gradients, serial)",
                    "cmaes": "estimated (serial" + ("; ss-PL-only fit as measured)" if mode != "trpl" else ")")}.get(m, "estimated (serial)")
            ode_txt = f"{fmt_t(row.ode_est_mean_s, row.ode_est_sd_s, prefix='≈ ')} {note}"
        tab.loc[i, "ode_workflow_new"] = ode_txt
        for c in ("time_mean_s", "time_sd_s", "evals_mean", "evals_sd", "evals_min", "evals_max", "ode_est_mean_s",
                  "ode_est_sd_s"):
            tab.loc[i, c] = row[c]
        tab.loc[i, "n_repeats_new"] = n
        diag = []
        if "min_ess_bulk_mean" in row and np.isfinite(row.get("min_ess_bulk_mean", np.nan)):
            diag.append(f"min ESS {row.min_ess_bulk_mean:,.0f} ± {row.min_ess_bulk_sd:,.0f}")
            diag.append(f"max R-hat {row.max_r_hat_mean:.3f} ± {row.max_r_hat_sd:.3f}")
        if "log_z_mean" in row and np.isfinite(row.get("log_z_mean", np.nan)):
            diag.append(f"log Z {row.log_z_mean:.3f} ± {row.log_z_sd:.3f}")
        tab.loc[i, "diagnostics_new"] = "; ".join(diag)
    return tab.drop(columns="_key")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    tabs = []
    for r in args.runs:
        run = Path(r)
        if not (run / "timing_repeats" / "summary.csv").exists():
            print("no timing_repeats/summary.csv in", run)
            continue
        t = mode_table(run)
        t.to_csv(run / "11_T8_timing_table_mean_sd.csv", index=False)
        tabs.append(t)
    tab = pd.concat(tabs, ignore_index=True)
    out = Path(args.out)
    tab.to_csv(out.with_suffix(".csv"), index=False)
    cols = ["mode", "step", "time_new", "evaluations_new", "ode_workflow_new", "diagnostics_new", "nn_workflow_previous",
            "ode_workflow_previous"]
    md = ["# Timing, 1-trap synthetic case: mean ± SD over independent runs (new) and the earlier T8 entries", "",
          "time_new / evaluations_new: mean ± SD over n runs with different seeds (repeat 0 = seeds of the original run); "
          "emcee and the sliced grid evaluate a fixed number of points (SD 0). ode_workflow_new: evaluations × one ODE "
          "solve at the best fit, per run (NUTS: finite-difference gradients). *_previous: the earlier T8 table "
          "(identical repeats, same seed; NUTS/emcee n = 1).", "",
          "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in tab[cols].fillna("").itertuples(index=False):
        md.append("| " + " | ".join(str(v).replace("|", "/") for v in r) + " |")
    out.with_suffix(".md").write_text("\n".join(md) + "\n")
    print(tab[["mode", "step", "time_new", "evaluations_new", "ode_workflow_new"]].fillna("").to_string(index=False))
    print("written:", out.with_suffix(".csv"), out.with_suffix(".md"))


if __name__ == "__main__":
    main()
