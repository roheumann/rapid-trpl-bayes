"""
collect_log_evidence.py — the log evidence log Z of every SMC run of one or more run folders, in one table
=========================================================================================================
19/10/2026 (revision).  log Z = log ∫ L(θ) π(θ) dθ with the prior π normalised on its support (inference/smc.py), from

  <run>/smc*/summary.json         tempered SMC of smc.ipynb (network, or exact ss-PL model in mode sspl)
  <run>/ode_vs_nn*/summary.json   ode_inference.ipynb: network SMC and rate-equation (ODE) SMC at the same sigma

and the sigmas / prior of the run.  log Z values are only comparable between models on the SAME data with the SAME
sigma (Bayes factor = difference of log Z).

Usage:  python scripts/collect_log_evidence.py RUN [RUN ...] [--out summary.csv]
Output: <run>/log_evidence.csv for every run (+ --out: all runs in one file)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def rows_of(run: Path) -> list:
    from inference import runs
    st = json.loads((run / "settings.json").read_text())
    cfg, mode = st["config"], st["mode"]
    sig = runs.run_sigmas(run)
    pb = pd.read_csv(run / "prior_bounds.csv")
    free = pb[pb.theta_lo < pb.theta_hi]
    et = "; ".join(f"{r.parameter} {r.theta_lo:.2f}-{r.theta_hi:.2f}" for r in pb.itertuples() if r.parameter.startswith("Et_rel"))
    base = {"run": run.name, "mode": mode, "network": Path(cfg["nn"]["dir"]).name, "n_free": int(len(free)),
            "prior_Et_rel": et, "fixed": json.dumps(cfg["prior"].get("fixed", {}))}
    out = []
    for d in sorted(run.glob("smc*/summary.json")):
        s = json.loads(d.read_text())
        m = s["smc"]
        out.append({**base, "model": "exact ss-PL model" if mode == "sspl" else "network", "sampler": "tempered SMC",
                    "sigma_trpl": s.get("sigma_trpl") if mode != "sspl" else None, "sigma_sspl": s.get("sigma_sspl"),
                    "log_z": m["log_z"], "log_z_se": m["log_z_se"], "time_s": m.get("time_mean_s"),
                    "evals": m.get("evals"), "source": str(d.relative_to(run))})
    for d in sorted(run.glob("ode_vs_nn*/summary.json")):
        s = json.loads(d.read_text())
        sg = s.get("sigmas", {})
        for key, model in (("nn_smc", "network"), ("ode_smc", "rate equations (ODE)")):
            if key in s.get("log_z", {}):
                lz, se = s["log_z"][key]
                out.append({**base, "mode": s.get("mode", mode), "model": model, "sampler": "tempered SMC",
                            "sigma_trpl": sg.get("trpl"), "sigma_sspl": sg.get("sspl") if s.get("mode", mode) != "trpl" else None,
                            "log_z": lz, "log_z_se": se, "time_s": None, "evals": None, "source": str(d.relative_to(run))})
    if not out:
        print(f"{run}: no SMC summary found")
    return out


def main(a):
    allrows = []
    for r in map(Path, a.runs):
        rows = rows_of(r)
        if rows:
            pd.DataFrame(rows).to_csv(r / "log_evidence.csv", index=False, float_format="%.6g")
            allrows += rows
            print(f"written {r / 'log_evidence.csv'} ({len(rows)} rows)")
    if a.out and allrows:
        t = pd.DataFrame(allrows)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        t.to_csv(a.out, index=False, float_format="%.6g")
        print(f"written {a.out}")
        print(t[["run", "mode", "model", "sigma_trpl", "sigma_sspl", "log_z", "log_z_se"]].to_string(index=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    p.add_argument("--out", default=None)
    main(p.parse_args())
