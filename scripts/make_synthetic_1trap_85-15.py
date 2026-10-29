"""
make_synthetic_1trap_85-15.py — new synthetic 1-trap case from the 1-trap fit of the measured 85:15 film (Robin, 2026-10-20)
==========================================================================================================================

Ground truth = the ODE-refined joint best fit of results/1trap_synthetic/1trap_85-15_film_fit_joint
(E_g 1.63 eV, absorbed n_pulse 1.130e17 cm^-3, k_rad,ext fixed, Voc,rad 1.333 V).  Noise-free data on the SAME x axes
as the measured data:

  ode_full_simulation.csv        legacy ODE solution at the truth: time_s, PL_norm, QFLS_eV (data calibration
                                 2 Vt ln(n_pulse/n_i) + Vt ln(PL/PL_max), as the measured QFLS), tau_diff_s, then the
                                 physical QFLS (kT ln(np/n_i^2)), n, p, n_t and the trap rates
  synthetic_trpl_qfls_tau.csv    tau_diff of the ODE at the 90 measured QFLS points (trpl_used.csv of the fit run)
  synthetic_sspl_suns_qfls.csv   exact ss-PL model at the 10 measured intensities (QFLS and PLQY), 1-sun generation
                                 from Voc,rad (the G of the fit run)
  true_parameters.json / .dat    the ground truth
  figures/synthetic_data.png     measured vs synthetic data

and the config configs/final_1trap_synthetic_85-15.json (same physics, prior, sigmas and fit settings; true_params).

usage (from rapid-trpl-bayes/):  python scripts/make_synthetic_1trap_85-15.py [--fit-run <dir>] [--out <dir>]
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-run", default="results/1trap_synthetic/1trap_85-15_film_fit_joint")
    ap.add_argument("--out", default="data/synthetic_1trap")
    ap.add_argument("--config-out", default="configs/final_1trap_synthetic_85-15.json")
    ap.add_argument("--run-output-dir", default="results/1trap_synthetic")
    ap.add_argument("--base-config", default="configs/final_1trap_85-15_fit.json",
                    help="config the synthetic config is copied from (fit config of the ground truth)")
    ap.add_argument("--comment", default=None, help="_comment of the synthetic config (default: the 2026-10-20 text)")
    a = ap.parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.chdir(ROOT)
    from inference import ode, surrogate, synthetic
    from inference.prior import Prior
    from inference.report import write_parameters_dat

    run, out = Path(a.fit_run), Path(a.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    cfg = json.loads((run / "settings.json").read_text())["config"]
    best = json.loads((run / "6_best_fit.json").read_text())
    art = surrogate.load_network(Path(cfg["nn"]["dir"]), cfg["nn"].get("stem"))
    Eg, npulse, T = cfg["physics"]["Eg_eV"], cfg["physics"]["npulse_cm3"], cfg["physics"]["T_K"]
    prior = Prior.from_artifacts(art, Eg=Eg, order_traps=cfg["prior"]["order_traps"],
                                 photodoping_traps=cfg["prior"]["photodoping_traps"],
                                 bounds_override=cfg["prior"]["bounds_override"], fixed=cfg["prior"].get("fixed", {}), T=T)
    C = art.constants
    theta = np.asarray(best["theta_best"], float)
    phys = prior.to_physical(theta[None]).iloc[0]

    # 1. full ODE solution at the truth (t, PL, QFLS, tau_diff first)
    full = ode.simulate_trpl_full(theta, prior, npulse, C, T)
    first = pd.DataFrame({"time_s": full["time_s"], "PL_norm": full["pl_norm"], "QFLS_eV": full["qfls_nominal_eV"],
                          "tau_diff_s": full["tau_diff_s"], "QFLS_physical_eV": full["qfls_eV"]})
    rest = full.drop(columns=["time_s", "pl_norm", "qfls_nominal_eV", "tau_diff_s", "qfls_eV"])
    pd.concat([first, rest], axis=1).to_csv(out / "ode_full_simulation.csv", index=False, float_format="%.10g")

    # 2. tr-PL at the 90 measured QFLS points (data calibration of the QFLS axis, as the measured points)
    meas_t = pd.read_csv(run / "trpl_used.csv")
    q = meas_t["qfls"].to_numpy(float)
    lt = ode.ode_log10_tau(theta, prior, npulse, C, q, T, axis="nominal")
    n_out = int((~np.isfinite(lt)).sum())
    if n_out:
        print(f"WARNING: {n_out} of {len(q)} measured QFLS points lie outside the simulated ODE curve (dropped)")
    ok = np.isfinite(lt)
    trpl = pd.DataFrame({"qfls_eV": q[ok], "tau_diff_s": 10 ** lt[ok]})
    trpl.to_csv(out / "synthetic_trpl_qfls_tau.csv", index=False, float_format="%.10g")

    # 3. ss-PL at the 10 measured intensities, G(1 sun) of the fit run (from Voc,rad)
    meas_s = pd.read_csv(run / "sspl_used_qfls_vs_suns.csv")
    suns = meas_s["x"].to_numpy(float)
    G = float(best["voc_rad_check"]["G_per_sun"])
    lo_n, hi_n, nn = cfg["sspl"]["n_sweep"]
    case = synthetic.make_case(theta, prior, npulse, C, qfls_window=(q.min(), q.max()), n_points=10, suns=suns,
                               G_per_sun=G, n_sweep=(lo_n, hi_n, nn), T=T)
    sspl = case["sspl"]
    sspl.to_csv(out / "synthetic_sspl_suns_qfls.csv", index=False, float_format="%.10g")

    # 4. ground truth
    truth = {"Et_eV": [float(phys["Et_1_eV"])], "Nt_cm3": [float(phys["Nt_1_cm-3"])],
             "tau_n_s": [float(phys["tau_n_1_s"])], "tau_p_s": [float(phys["tau_p_1_s"])],
             "krad_cm3s": float(phys["krad_cm3s"]), "theta": [float(v) for v in theta],
             "source": f"ODE-refined joint 1-trap best fit of {run}", "Eg_eV": Eg, "npulse_cm3": npulse,
             "G_per_sun": G, "n_trpl_points": int(ok.sum()), "n_sspl_points": int(len(sspl))}
    (out / "true_parameters.json").write_text(json.dumps(truth, indent=2))
    write_parameters_dat(out / "true_parameters.dat", prior, theta,
                         title="ground truth of the synthetic 1-trap case (1-trap fit of the measured 85:15 film)")

    # 5. figure: measured vs synthetic
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    ax[0].semilogy(q, meas_t["tau_diff"], "o", ms=3, mfc="none", color="0.4", label="measured (raw points)")
    ax[0].semilogy(trpl.qfls_eV, trpl.tau_diff_s, "-", color="C3", label="synthetic (ODE at the truth)")
    ax[0].set_xlabel("QFLS (eV)"); ax[0].set_ylabel(r"$\tau_\mathrm{diff}$ (s)"); ax[0].legend(fontsize=8)
    ax[1].loglog(first.time_s, first.PL_norm, "-", color="C3"); ax[1].set_xlabel("t (s)"); ax[1].set_ylabel("PL (norm.)")
    ax[1].set_ylim(1e-13, 2)
    ax[2].semilogx(suns, meas_s["y"], "o", mfc="none", color="0.4", label="measured")
    ax[2].semilogx(sspl.intensity_suns, sspl.qfls_eV, "s-", ms=4, color="C3", label="synthetic")
    ax[2].set_xlabel("illumination (suns)"); ax[2].set_ylabel("QFLS (eV)"); ax[2].legend(fontsize=8)
    fig.suptitle("synthetic 1-trap case: " + ", ".join(f"{k} {v:.4g}" for k, v in phys.items()), fontsize=9)
    fig.tight_layout(); fig.savefig(out / "figures" / "synthetic_data.png", dpi=150); plt.close(fig)

    # 6. config of the synthetic case
    c = json.loads(Path(a.base_config).read_text(), object_pairs_hook=collections.OrderedDict)
    c["_comment"] = ("NEW synthetic 1-trap case (Robin, 2026-10-20): ground truth = ODE-refined joint 1-trap fit of the measured "
                     "85:15 film (configs/final_1trap_85-15_fit.json); noise-free data at the 90 measured QFLS points and the "
                     "10 measured ss-PL intensities (scripts/make_synthetic_1trap_85-15.py). E_g 1.63 eV, n_pulse 1.130e17, "
                     "k_rad fixed at k_rad,ext = 4.793e-11 (= truth), Voc,rad 1.333 V. sigma_trPL = sigma_nn = 0.1124 (network "
                     "test error), sigma_ssPL = 0.033958 eV. SMC only (no NUTS / emcee / grid).")
    if a.comment:
        c["_comment"] = a.comment
    c["sigma_rule"] = "model_plus_nn"                   # sigma_trPL = sigma_nn (trpl.sigma = 0 below)
    c["run_name"] = "1trap_synthetic"
    c["output_dir"] = a.run_output_dir
    c["trpl"] = collections.OrderedDict(path=str(out / "synthetic_trpl_qfls_tau.csv"), format="processed", preprocess=False,
                                        qfls_window=None, sigma=0.0, interp_space="log10", skiprows="auto")
    c["sspl"]["path"] = str(out / "synthetic_sspl_suns_qfls.csv")
    c["sspl"]["format"] = "auto"
    c["true_params"] = {k: truth[k] for k in ("Et_eV", "Nt_cm3", "tau_n_s", "tau_p_s", "krad_cm3s")}
    Path(a.config_out).write_text(json.dumps(c, indent=2))
    print(json.dumps({k: truth[k] for k in ("Et_eV", "Nt_cm3", "tau_n_s", "tau_p_s", "krad_cm3s", "n_trpl_points",
                                            "n_sspl_points", "G_per_sun")}, indent=1))
    print("written:", out, "and", a.config_out)


if __name__ == "__main__":
    main()
