"""
end_to_end_timing.py — end-to-end time table of a synthetic case (Robin, 2026-10-20)
====================================================================================

Rows (offline cost of the network, then the per-sample analysis):
  training data: ODE simulations          T8 offline row (CPU-h, wall-h, ms per simulation)
  data preparation                        offline_provenance.json (data_preparation_h)
  network training                        T8 offline row + offline_provenance.json (epochs, fit hours)
  CMA-ES fit (NN)                         T8 'optimisation (CMA-ES, all starts)' of every mode
  curve_fit (ODE)                         T8 'local refinement curve_fit' (tr-PL and joint modes)
  sliced grid                             T8 'sliced grid ...' of every mode
  SMC with the network: tr-PL, ss-PL, joint   T8 'posterior sampling: tempered SMC' of every mode (ss-PL: exact model)
  SMC tr-PL with the ODE (measured)       <base>_trpl/ode_vs_nn/summary.json, same SMC settings -> measured speed-up

Times are mean ± SD over the timeit repeats of the notebooks (n given); the 'ODE estimate' column is the T8 estimate of
the same step with the rate equations (serial).

    python scripts/end_to_end_timing.py results/1trap_synthetic/1trap_synthetic \
        --out results/1trap_synthetic/end_to_end_timing
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


def evals(text):
    m = re.search(r"([\d,]+) (evaluations|ODE solves)", str(text))
    return int(m.group(1).replace(",", "")) if m else None


def fmt_s(mean, sd, n):
    if mean is None or not np.isfinite(mean):
        return "—"
    if mean >= 3600:
        return f"{mean / 3600:.2f} ± {sd / 3600:.2f} h (n = {int(n)})" if sd == sd else f"{mean / 3600:.2f} h"
    if mean >= 60:
        return f"{mean / 60:.2f} ± {sd / 60:.2f} min (n = {int(n)})" if sd == sd else f"{mean / 60:.2f} min"
    return f"{mean:.3g} ± {sd:.2g} s (n = {int(n)})" if sd == sd else f"{mean:.3g} s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("--out", required=True)
    ap.add_argument("--suffix", default="", help="run-folder suffix, e.g. _QUICK")
    a = ap.parse_args()
    runs = {m: Path(f"{a.base}_{m}{a.suffix}") for m in ("trpl", "sspl", "joint")}
    T8 = {m: pd.read_csv(r / "11_T8_timing_table.csv") for m, r in runs.items() if (r / "11_T8_timing_table.csv").exists()}
    cfg = json.loads((runs["trpl"] / "settings.json").read_text())["config"]
    nn_dir = Path(cfg["nn"]["dir"])
    prov = json.loads((nn_dir / "offline_provenance.json").read_text()) if (nn_dir / "offline_provenance.json").exists() else {}
    rows = []

    def add(group, step, mode, nn_text, mean=None, sd=None, n=None, ev=None, ode_text="—", speed=None, source=""):
        rows.append({"group": group, "step": step, "mode": mode, "network workflow": nn_text, "time_mean_s": mean,
                     "time_sd_s": sd, "n_repeats": n, "evaluations": ev, "ODE workflow": ode_text,
                     "speed-up (ODE/NN)": speed, "source": source})

    t = T8["trpl"]
    off = t[t.stage == "offline"]
    sim = off[off.step.str.startswith("training simulations")].iloc[0]
    add("offline (once per network)", sim.step, "—", sim.nn_workflow, source="T8 offline row; offline_provenance.json")
    dp = prov.get("data_preparation_h", {})
    dph = sum(v for v in dp.values() if isinstance(v, (int, float)))
    if dph:
        add("offline (once per network)", "data preparation (training HDF5)", "—", f"{dph:.2f} h",
            mean=dph * 3600, source="offline_provenance.json data_preparation_h")
    tr = off[off.step.str.startswith("network training")].iloc[0]
    txt = tr.nn_workflow + (f"; fit {prov['training_fit_h']:.2f} h, {prov.get('epochs_trained', '?')} epochs"
                            if "training_fit_h" in prov else "")
    add("offline (once per network)", "network training", "—", txt,
        mean=prov.get("training_fit_h", np.nan) * 3600 if "training_fit_h" in prov else None, source="T8 offline row")

    def t8_row(mode, prefix):
        if mode not in T8:
            return None
        d = T8[mode]
        s = d[d.step.str.startswith(prefix)]
        return s.iloc[0] if len(s) else None

    for prefix, label, modes in (("optimisation (CMA-ES", "CMA-ES fit (network)", ("trpl", "sspl", "joint")),
                                 ("local refinement curve_fit", "curve_fit (ODE)", ("trpl", "joint")),
                                 ("sliced grid", "sliced grid (paper method)", ("trpl", "sspl", "joint")),
                                 ("posterior sampling: tempered SMC", "SMC with the network", ("trpl", "sspl", "joint"))):
        for m in modes:
            r = t8_row(m, prefix)
            if r is None:
                continue
            lab = label if not (label.startswith("SMC") and m == "sspl") else "SMC (exact ss-PL model, no network)"
            add("per sample", lab, m, r.nn_workflow, mean=r.nn_mean_s, sd=r.nn_sd_s, n=r.n_repeats, ev=evals(r.nn_workflow),
                ode_text=r.ode_workflow, source=f"{runs[m].name}/11_T8_timing_table.csv: {r.step}")
    od = runs["trpl"] / "ode_vs_nn" / "summary.json"
    if od.exists():
        s = json.loads(od.read_text())
        for rr in s.get("speed_up_same_settings", []):
            if str(rr.get("sampler", "")).lower().startswith("smc"):
                add("per sample", "SMC with the ODE (measured, same settings)", "trpl",
                    f"network SMC {rr['time_nn_s']:.2f} s", mean=rr["time_ode_s"], ev=rr.get("evals_ode"),
                    ode_text=f"{rr['time_ode_s'] / 60:.1f} min for {rr.get('evals_ode', 0):,.0f} ODE solves",
                    speed=rr.get("speed_up_wall"), source=str(od))
    df = pd.DataFrame(rows)
    out = Path(a.out)
    df.to_csv(out.with_suffix(".csv"), index=False)
    md = ["# End-to-end time table — " + Path(a.base).name, "",
          "| group | step | mode | network workflow | evaluations | ODE workflow | speed-up |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        ev = f"{int(r['evaluations']):,}" if r["evaluations"] else ""
        sp = f"×{r['speed-up (ODE/NN)']:.0f}" if r["speed-up (ODE/NN)"] else ""
        md.append(f"| {r['group']} | {r['step']} | {r['mode']} | {r['network workflow']} | {ev} | {r['ODE workflow']} | {sp} |")
    md += ["", "Times: mean ± SD over the timeit repeats of the notebooks (n). ODE workflow: the same step with the rate "
               "equations — measured for curve_fit (ODE in both workflows) and for the tr-PL SMC (ode_inference.ipynb), "
               "otherwise the T8 estimate (serial ODE time per evaluation × evaluations)."]
    out.with_suffix(".md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
