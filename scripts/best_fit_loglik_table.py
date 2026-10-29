"""
best_fit_loglik_table.py — log L of the physics start point and of the best fits per mode (NN and ODE tr-PL, ss-PL)
===================================================================================================================
23/10/2026 (Robin).  Same evaluation as best_fit_mse_table.py (tr-PL MSE of log10 tau_diff with the NN surrogate and
with the legacy ODE, ss-PL MSE with the exact model), plus the physics-informed start point (start_point.json).
log L per curve = -MSE/(2 sigma^2) - ln sigma - 0.5 ln 2pi (whole-curve sigma, as the run's likelihood);
joint = tr-PL + ss-PL.  sigma: the run's sigmas, plus optional extra tr-PL sigmas (--extra-sigma-trpl) for reference.

    python scripts/best_fit_loglik_table.py <run> [--extra-sigma-trpl 0.11235]

Output in <run>: best_fit_logL.csv and best_fit_logL.md
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def loglik(mse, sigma):
    return -mse / (2 * sigma ** 2) - np.log(sigma) - 0.5 * np.log(2 * np.pi)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("run")
    ap.add_argument("--extra-sigma-trpl", type=float, nargs="*", default=[])
    ap.add_argument("--gpu", default="0")
    args = ap.parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.gpu)
    from inference import refine, runs, surrogate
    from inference.prior import Prior
    surrogate.configure_tensorflow()

    run = Path(args.run)
    cfg = json.loads((run / "settings.json").read_text())["config"]
    art = surrogate.load_network(cfg["nn"]["dir"], cfg["nn"].get("stem"))
    prior = Prior.from_artifacts(art, Eg=cfg["physics"]["Eg_eV"], order_traps=cfg["prior"]["order_traps"],
                                 photodoping_traps=cfg["prior"]["photodoping_traps"],
                                 bounds_override=cfg["prior"]["bounds_override"], fixed=cfg["prior"].get("fixed", {}),
                                 T=cfg["physics"]["T_K"])
    sig = runs.run_sigmas(run)
    lp, _ = runs.make_log_post(cfg, run, art, prior, "joint", sig["trpl"], sig["sspl"])
    prob = refine.make_problem(lp, art, cfg["physics"]["npulse_cm3"], cfg["physics"]["T_K"])
    n_tr = len(prob.y_trpl)

    points = {"physics start": json.loads((run / "start_point.json").read_text())["theta"]}
    for mode, b in json.loads((run / "6_best_fit.json").read_text())["best_per_mode"].items():
        points[f"best fit {mode}"] = b["theta"]
    rows = []
    for label, th in points.items():
        th = np.array(th, float)
        m = lp.np_mse_terms(th[None])
        y_ode = prob.predict(th)[0][:n_tr]
        ok = np.isfinite(y_ode)
        r = {"point": label, **prior.to_physical(th).iloc[0].to_dict(),
             "mse_trpl_nn_dec2": float(m["trpl"][0]),
             "mse_trpl_ode_dec2": float(np.mean((y_ode[ok] - prob.y_trpl[ok]) ** 2)),
             "ode_points_covered": f"{ok.sum()}/{n_tr}", "mse_sspl_eV2": float(m["sspl"][0])}
        for tag, s_tr in [("", sig["trpl"])] + [(f"_sigtr{s:g}", s) for s in args.extra_sigma_trpl]:
            r[f"logL_trpl_nn{tag}"] = loglik(r["mse_trpl_nn_dec2"], s_tr)
            r[f"logL_trpl_ode{tag}"] = loglik(r["mse_trpl_ode_dec2"], s_tr)
            r[f"logL_sspl{tag}"] = loglik(r["mse_sspl_eV2"], sig["sspl"])
            r[f"logL_joint_nn{tag}"] = r[f"logL_trpl_nn{tag}"] + r[f"logL_sspl{tag}"]
            r[f"logL_joint_ode{tag}"] = r[f"logL_trpl_ode{tag}"] + r[f"logL_sspl{tag}"]
        rows.append(r)
    t = pd.DataFrame(rows)
    t.to_csv(run / "best_fit_logL.csv", index=False, float_format="%.6g")

    md = [f"# log L of the physics start and the best fits — `{run}`", "",
          f"log L = -MSE/(2 sigma^2) - ln sigma - 0.5 ln 2pi per curve; joint = tr-PL + ss-PL; ss-PL: exact model; "
          f"{n_tr} tr-PL points.", ""]
    for tag, s_tr in [("", sig["trpl"])] + [(f"_sigtr{s:g}", s) for s in args.extra_sigma_trpl]:
        md += [f"## sigma_trPL = {s_tr:.6g} dec, sigma_ssPL = {sig['sspl']:.6g} eV"
               + (" (sigmas of the fit)" if not tag else " (reference)"),
               f"Maximum possible: tr-PL {loglik(0, s_tr):.4f}, ss-PL {loglik(0, sig['sspl']):.4f}.", "",
               "| point | log L tr-PL NN | log L tr-PL ODE | log L ss-PL | log L joint NN | log L joint ODE | "
               "RMSE tr-PL NN / ODE (dec) | RMSE ss-PL (meV) |", "|---|---|---|---|---|---|---|---|"]
        md += [f"| {r['point']} | {r[f'logL_trpl_nn{tag}']:.3f} | {r[f'logL_trpl_ode{tag}']:.3f} | {r[f'logL_sspl{tag}']:.3f} | "
               f"{r[f'logL_joint_nn{tag}']:.3f} | {r[f'logL_joint_ode{tag}']:.3f} | "
               f"{np.sqrt(r['mse_trpl_nn_dec2']):.4f} / {np.sqrt(r['mse_trpl_ode_dec2']):.4f} | "
               f"{1e3 * np.sqrt(r['mse_sspl_eV2']):.2f} |" for r in rows]
        md.append("")
    cols = list(prior.to_physical(np.array(points["physics start"], float)).columns)
    md += ["## parameters", "", "| point | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    md += [f"| {r['point']} | " + " | ".join(f"{r[c]:.5g}" for c in cols) + " |" for r in rows]
    (run / "best_fit_logL.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
