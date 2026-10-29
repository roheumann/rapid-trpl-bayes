"""
best_fit_mse_table.py — MSE of the best fits per mode: tr-PL (NN and ODE) and ss-PL (exact model)
==================================================================================================
22/10/2026 (Robin).  For every mode in <run>/6_best_fit.json best_per_mode (trpl, sspl, joint) the theta of the best
fit is evaluated with the run's likelihood: tr-PL MSE of log10 tau_diff with the NN surrogate and with the rate
equations (legacy ODE), ss-PL MSE of QFLS (eV^2) with the exact ss-PL model.  No sigma enters.

    python scripts/best_fit_mse_table.py results/1trap_synthetic/1trap_85-15_film_fit_joint

Output in <run>: best_fit_mse.csv and best_fit_mse.md
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("run")
    ap.add_argument("--gpu", default="0")
    args = ap.parse_args()
    import os
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.gpu)
    from inference import ode, refine, runs, surrogate
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
    lp, _ = runs.make_log_post(cfg, run, art, prior, "joint", sig["trpl"], sig["sspl"])
    prob = refine.make_problem(lp, art, cfg["physics"]["npulse_cm3"], cfg["physics"]["T_K"])
    n_tr = len(prob.y_trpl)

    best = json.loads((run / "6_best_fit.json").read_text())["best_per_mode"]
    rows = []
    for mode, b in best.items():
        th = np.array(b["theta"], float)
        m = lp.np_mse_terms(th[None])
        # raw ODE curve (NaN outside the simulated QFLS range); prob.predict would put penalty values there
        y_ode = ode.ode_log10_tau(th, prob.prior, prob.npulse, prob.constants, prob.qfls, prob.T)
        ok = np.isfinite(y_ode)
        rows.append({"mode": mode, **{k: v for k, v in b["physical"].items()},
                     "mse_trpl_nn_dec2": float(m["trpl"][0]),
                     "mse_trpl_ode_dec2": float(np.mean((y_ode[ok] - prob.y_trpl[ok]) ** 2)),
                     "ode_points_covered": f"{ok.sum()}/{n_tr}",
                     "mse_sspl_eV2": float(m["sspl"][0]),
                     # marginalised ss-PL offset (sspl.offset_sigma_eV, 24/10/2026): delta_hat and the RMSE around it
                     "sspl_offset_meV": 1e3 * float(lp.sspl_offset(th[None])[0]),
                     "rmse_sspl_after_offset_meV": 1e3 * float(np.sqrt(np.mean(
                         (lp.predict_sspl(th)["y_model"][0] - lp.predict_sspl(th)["y_obs"] - lp.sspl_offset(th[None])[0]) ** 2)))})
    t = pd.DataFrame(rows)
    t["rmse_trpl_nn_dec"], t["rmse_trpl_ode_dec"] = np.sqrt(t.mse_trpl_nn_dec2), np.sqrt(t.mse_trpl_ode_dec2)
    t["rmse_sspl_meV"] = 1e3 * np.sqrt(t.mse_sspl_eV2)
    t.to_csv(run / "best_fit_mse.csv", index=False, float_format="%.6g")
    md = [f"# MSE of the best fits — `{run}`", "",
          f"sigmas of the run (not used here): {sig}. tr-PL: log10 tau_diff at {n_tr} points; ss-PL: QFLS in eV.", "",
          "| mode | tr-PL MSE NN (dec²) | tr-PL MSE ODE (dec²) | ss-PL MSE (eV²) | tr-PL RMSE NN / ODE (dec) | ss-PL RMSE (meV) "
          "| ss-PL offset δ̂ (meV) | ss-PL RMSE after δ̂ (meV) |",
          "|---|---|---|---|---|---|---|---|"]
    md += [f"| {r.mode} | {r.mse_trpl_nn_dec2:.4g} | {r.mse_trpl_ode_dec2:.4g} | {r.mse_sspl_eV2:.4g} | "
           f"{r.rmse_trpl_nn_dec:.4f} / {r.rmse_trpl_ode_dec:.4f} | {r.rmse_sspl_meV:.2f} | {r.sspl_offset_meV:+.1f} | "
           f"{r.rmse_sspl_after_offset_meV:.2f} |" for r in t.itertuples()]
    (run / "best_fit_mse.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
