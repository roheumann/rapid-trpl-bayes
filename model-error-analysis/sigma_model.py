"""
sigma_model.py — model error sigma_model of the 85:15 film from the drift-diffusion model (Robin, 2026-10-19/20)
=====================================================================================================================

sigma_model = RMSE of log10 tau_diff between the drift-diffusion (DD) model (Python port of the MATLAB mobility
model pdex1_sproul_Chris_frequency.m, dd_model.py) and the 0D rate equations (legacy trpl_module.solve_transient,
inference/ode.py), both at the 85:15 best fit (2 traps) and the absorption coefficient alpha at the tr-PL excitation
wavelength (630 nm), on the measured tr-PL QFLS points.  It enters the likelihood as the error of the WHOLE curve:
sigma_trPL = sqrt(sigma_model^2 + sigma_nn^2), log L = -MSE/(2 sigma^2).

1. alpha(E) from the measured absorptance: A = 1 - exp(-alpha d)  ->  alpha = -ln(1 - A)/d  (d = film thickness).
   alpha_630 at E = hc/630 nm = 1.968 eV.  If A has saturated there (A >= 0.98 max A above 1.8 eV, i.e. limited by
   reflection), alpha_630 is only a lower bound -> flagged (Robin decides).
2. DD vs 0D at the best fit with alpha_630: both on the data calibration of the QFLS axis (2 Vt ln(n_pulse/n_i) +
   Vt ln(PL/PL_max)), same constants (N_C, N_V, Vt of the network's training data), interpolated onto the measured QFLS
   points (trpl_used.csv) -> sigma_model = sqrt(mean(Delta^2)).  Robustness (reported only): alpha_630/2, 2 alpha_630,
   the tr-PL-only best fit, the RMSE on the 256-point grid [0.9, 2 Vt ln(n_pulse/n_i)] of the old MATLAB analysis.
3. For comparison: the old 5-scenario method (homoskedastic_sigma_derivative_alpha.m) evaluated at alpha_630:
   sigma = RMS over QFLS of |mean over scenarios of d log10 tau/d log10 alpha| x dlog10 alpha (= 1); the scenario
   derivative columns (alpha 1e4 ... 1e6) are interpolated linearly in log10 alpha.  Sanity check: the all-alpha
   average of the .m reproduces 0.493608.

usage (from rapid-trpl-bayes/):
    python model-error-analysis/sigma_model.py                       # everything, default paths
    python model-error-analysis/sigma_model.py --best-fit <run>/6_best_fit.json --out <dir> --dd-only
        (DD vs 0D only, alpha_630 from the summary.json of the main analysis; e.g. at a new best fit)
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

HC_EV_NM = 1239.84198
FINAL = REPO / "results" / "2trap_85-15_film" / "sigma_model"


# ─────────────────────────────────────────────────────────────────────────────
# 1. absorptance -> alpha(E)
# ─────────────────────────────────────────────────────────────────────────────
def read_absorptance(path) -> pd.DataFrame:
    """Two numeric columns (energy [eV] or wavelength [nm], absorptance [0-1 or %]); header / text lines skipped."""
    rows = []
    for line in Path(path).read_text(errors="replace").splitlines():
        tok = [t for t in re.split(r"[,;\s]+", line.strip()) if t]
        if len(tok) < 2:
            continue
        try:
            rows.append((float(tok[0]), float(tok[1])))
        except ValueError:
            continue
    x, a = np.array(rows).T
    if np.nanmedian(x) > 50:                     # wavelength in nm
        lam, E = x, HC_EV_NM / x
    else:
        E, lam = x, HC_EV_NM / x
    if np.nanmax(a) > 1.5:                       # percent
        a = a / 100.0
    df = pd.DataFrame({"E_eV": E, "wavelength_nm": lam, "A": a}).sort_values("E_eV").reset_index(drop=True)
    return df


def alpha_from_absorptance(df: pd.DataFrame, d_cm: float) -> pd.DataFrame:
    df = df.copy()
    with np.errstate(all="ignore"):
        df["alpha_cm"] = -np.log1p(-np.clip(df["A"].values, None, 1 - 1e-12)) / d_cm
    return df


def alpha_at(df: pd.DataFrame, E: float) -> dict:
    ok = np.isfinite(df["alpha_cm"]) & (df["alpha_cm"] > 0)
    e, la = df["E_eV"][ok].values, np.log10(df["alpha_cm"][ok].values)
    if not (e.min() <= E <= e.max()):
        raise SystemExit(f"E = {E:.4f} eV is outside the absorptance data ({e.min():.3f}-{e.max():.3f} eV)")
    a_E = float(np.interp(E, df["E_eV"].values, df["A"].values))
    hi = df["E_eV"] >= 1.8
    a_max = float(df.loc[hi, "A"].max()) if hi.any() else float(df["A"].max())
    return {"E_eV": float(E), "alpha_cm": float(10 ** np.interp(E, e, la)), "A": a_E, "A_max_above_1.8eV": a_max,
            "saturated": bool(a_E >= 0.98 * a_max),
            "alpha_max_cm_in_data": float(df["alpha_cm"][ok].max())}


# ─────────────────────────────────────────────────────────────────────────────
# 3. old 5-scenario derivative method at alpha_630
# ─────────────────────────────────────────────────────────────────────────────
SCEN_ALPHAS = np.array([1e4, 5e4, 1e5, 5e5, 1e6])


def scenario_derivatives():
    fs = sorted(glob.glob(str(HERE / "derivative_analysis_output" / "deriv_per_file_*.csv")))
    names = [Path(f).stem.replace("deriv_per_file_simulation_results-", "") for f in fs]
    D, q = [], None
    for f in fs:
        d = pd.read_csv(f)
        q = d["QFLS_eV"].values
        D.append(d.iloc[:, 1:].to_numpy(float))            # columns ordered alpha 1e4 ... 1e6
    return q, np.stack(D), names                            # (256,), (scen, 256, 5)


def scenario_sigma(alpha: float | None, D) -> dict:
    """RMS over QFLS of |mean over scenarios| x 1; alpha None = the .m's all-alpha average (sanity check)."""
    with np.errstate(all="ignore"):
        import warnings
        warnings.simplefilter("ignore")
        if alpha is None:
            per_folder = np.nanmean(D, axis=2)              # mean over alpha files (omitnan), as in the .m
        else:
            x = np.log10(SCEN_ALPHAS)
            xa = np.clip(np.log10(alpha), x[0], x[-1])
            j = int(np.clip(np.searchsorted(x, xa) - 1, 0, len(x) - 2))
            w = (xa - x[j]) / (x[j + 1] - x[j])
            per_folder = (1 - w) * D[:, :, j] + w * D[:, :, j + 1]
        m = np.nanmean(per_folder, axis=0)
        s = np.abs(m)
        ok = np.isfinite(s)
        per_scen = [float(np.sqrt(np.nanmean(per_folder[k] ** 2))) for k in range(len(per_folder))]
    return {"sigma": float(np.sqrt(np.mean(s[ok] ** 2))), "per_scenario_rms": per_scen, "mean_derivative": m}


# ─────────────────────────────────────────────────────────────────────────────
# 2. DD vs 0D at the best fit
# ─────────────────────────────────────────────────────────────────────────────
def load_run(run_dir: Path):
    """config snapshot, network artifacts and prior of a finished main.ipynb run."""
    from inference import surrogate
    from inference.prior import Prior
    cfg = json.loads((run_dir / "settings.json").read_text())["config"]
    nn_dir = Path(cfg["nn"]["dir"])
    if not nn_dir.is_absolute():
        nn_dir = (REPO / nn_dir).resolve()
    art = surrogate.load_network(nn_dir, cfg["nn"].get("stem"))
    prior = Prior.from_artifacts(art, Eg=cfg["physics"]["Eg_eV"], order_traps=cfg["prior"]["order_traps"],
                                 photodoping_traps=cfg["prior"]["photodoping_traps"],
                                 bounds_override=cfg["prior"]["bounds_override"],
                                 fixed=cfg["prior"].get("fixed", {}), T=cfg["physics"]["T_K"])
    return cfg, art, prior


def physical_traps(theta, prior) -> dict:
    phys = prior.to_physical(np.asarray(theta, float)[None]).iloc[0]
    K = prior.n_traps
    return {"krad": float(phys["krad_cm3s"]),
            "Et": [float(phys[f"Et_{k}_eV"]) for k in range(1, K + 1)],
            "Nt": [float(phys[f"Nt_{k}_cm-3"]) for k in range(1, K + 1)],
            "taun": [float(phys[f"tau_n_{k}_s"]) for k in range(1, K + 1)],
            "taup": [float(phys[f"tau_p_{k}_s"]) for k in range(1, K + 1)]}


def dd_log10_tau(par: dict, alpha: float, cfg: dict, constants: dict, qfls_points, mu: float = 1.0) -> tuple:
    """
    DD log10 tau_diff at the QFLS points + the raw DD curve, on the nominal (data) calibration of the QFLS axis,
    2 Vt ln(n_pulse/n_i) + Vt ln(PL/PL_max), post-processed like inference/ode.py (PL capped PL_DECAY_MAGNITUDE
    decades below its maximum, Pchip).  The DD PL is normalised to its maximum exactly as the measured PL is, so the
    QFLS-axis offset of the exp(-alpha x) start profile (dd_model.initial_pl_enhancement) is part of the model error.
    Constants: N_C, N_V, Vt of the network's training data; kT of the DD = legacy 8.617e-5 T (as the 0D solver).
    """
    import dd_model
    res = dd_model.simulate_dd(par["Et"], par["Nt"], par["taun"], par["taup"], par["krad"],
                               cfg["physics"]["npulse_cm3"], alpha, d=cfg["physics"]["thickness_nm"] * 1e-7, mu=mu,
                               Eg=cfg["physics"]["Eg_eV"], Nc=constants["Nc"], Nv=constants["Nv"],
                               T=cfg["physics"]["T_K"])
    cap = float(constants.get("PL_DECAY_MAGNITUDE", 12.0))
    y = dd_model.log10_tau_on_qfls(res, qfls_points, pl_decay_magnitude=cap, vt=constants["Vt"])
    q_full = dd_model.nominal_qfls(res, constants["Vt"])
    q_dense = np.linspace(np.nanmin(q_full[np.isfinite(q_full)]), np.nanmax(q_full[np.isfinite(q_full)]), 600)
    lt_dense = dd_model.curve_log10_tau(q_full, res["tau_diff"], res["PL"], q_dense, cap)
    ok = np.isfinite(lt_dense)
    return y, q_dense[ok], lt_dense[ok], res["info"]


def ode_log10_tau(theta, prior, cfg, constants, qfls_points):
    from inference import ode
    q, lt = ode.simulate_trpl(theta, prior, cfg["physics"]["npulse_cm3"], constants, cfg["physics"]["T_K"],
                              axis="nominal")
    from scipy.interpolate import PchipInterpolator
    return PchipInterpolator(q, lt, extrapolate=False)(np.asarray(qfls_points, float)), q, lt


def rmse(a, b):
    d = np.asarray(a) - np.asarray(b)
    ok = np.isfinite(d)
    return float(np.sqrt(np.mean(d[ok] ** 2))) if ok.any() else float("nan"), int(ok.sum())


def dd_vs_0d(theta, prior, cfg, constants, alpha, q_data, q_grid, label, out_dir, y_data=None):
    t0 = time.perf_counter()
    y_dd, qd, ltd, dd_info = dd_log10_tau(physical_traps(theta, prior), alpha, cfg, constants, q_data)
    t_dd = time.perf_counter() - t0
    y_0d, q0, lt0 = ode_log10_tau(theta, prior, cfg, constants, q_data)
    r_data, n_data = rmse(y_dd, y_0d)
    from scipy.interpolate import PchipInterpolator
    g_dd = PchipInterpolator(qd, ltd, extrapolate=False)(q_grid)
    g_0d = PchipInterpolator(q0, lt0, extrapolate=False)(q_grid)
    r_grid, n_grid = rmse(g_dd, g_0d)
    pd.DataFrame({"qfls_eV": q_data, "log10_tau_dd": y_dd, "log10_tau_0d": y_0d,
                  "log10_tau_data": y_data if y_data is not None else np.nan,
                  "delta_dd_minus_0d": y_dd - y_0d}).to_csv(out_dir / f"dd_vs_0d_{label}_data_points.csv", index=False)
    pd.DataFrame({"qfls_eV": qd, "log10_tau_dd": ltd}).to_csv(out_dir / f"dd_curve_{label}.csv", index=False)
    pd.DataFrame({"qfls_eV": q0, "log10_tau_0d": lt0}).to_csv(out_dir / f"ode0d_curve_{label}.csv", index=False)
    return {"label": label, "alpha_cm": float(alpha), "rmse_data_points": r_data, "n_data_points_compared": n_data,
            "n_data_points": int(len(q_data)), "rmse_grid_0.9_to_qmax": r_grid, "n_grid_points_compared": n_grid,
            "time_dd_s": t_dd,
            "dd_solver": {k: v for k, v in dd_info.items() if isinstance(v, (int, float, str, bool))},
            "curves": {"q_data": q_data, "y_dd": y_dd, "y_0d": y_0d, "qd": qd, "ltd": ltd, "q0": q0, "lt0": lt0}}


# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--absorptance", default=str(REPO / "data/85-15_film/absorptance.dat"))
    ap.add_argument("--wavelength-nm", type=float, default=630.0)
    ap.add_argument("--best-fit", default=str(REPO / "results/2trap_85-15_film/2trap_85-15_film_fit_joint/6_best_fit.json"))
    ap.add_argument("--best-fit-trpl", default=str(REPO / "results/2trap_85-15_experimental_v3_trpl/6_best_fit.json"),
                    help="robustness: tr-PL-only best fit ('' = skip)")
    ap.add_argument("--alpha", type=float, default=None, help="override alpha at the laser wavelength [1/cm]")
    ap.add_argument("--out", default=str(FINAL))
    ap.add_argument("--dd-only", action="store_true", help="only DD vs 0D at --best-fit; alpha_630 from the main summary")
    ap.add_argument("--no-robustness", action="store_true")
    ap.add_argument("--npulse", type=float, default=None,
                    help="override n_pulse [cm^-3] of the run (e.g. the absorbed density); the data QFLS points of the run "
                         "are shifted by 2 Vt ln(n_new/n_run), exactly as the preprocessing would place them")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    fig_dir = out / "figures"; fig_dir.mkdir(exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_dir = Path(a.best_fit).resolve().parent
    cfg, art, prior = load_run(run_dir)
    C = art.constants
    npulse_run = float(cfg["physics"]["npulse_cm3"])
    if a.npulse:
        cfg["physics"]["npulse_cm3"] = float(a.npulse)
    d_cm = cfg["physics"]["thickness_nm"] * 1e-7
    E_las = HC_EV_NM / a.wavelength_nm
    summary = {"date": time.strftime("%Y-%m-%d %H:%M"), "best_fit": str(a.best_fit), "wavelength_nm": a.wavelength_nm,
               "E_laser_eV": E_las, "thickness_cm": d_cm, "mu_cm2_Vs": 1.0,
               "constants": {k: float(C[k]) for k in ("Nc", "Nv", "Vt", "tmin", "tmax") if k in C},
               "note": "sigma_model = RMSE(log10 tau_diff DD - 0D) at the measured QFLS points; whole-curve error"}

    # ── 1. alpha(E) ──────────────────────────────────────────────────────────
    if a.dd_only:
        alpha = float(a.alpha) if a.alpha else float(json.loads((FINAL / "summary.json").read_text())["alpha_630"]["alpha_cm"])
        summary["alpha_630"] = {"alpha_cm": alpha, "source": str(FINAL / "summary.json") if not a.alpha else "--alpha"}
    else:
        dfa = alpha_from_absorptance(read_absorptance(a.absorptance), d_cm)
        dfa.to_csv(out / "alpha_vs_E.csv", index=False)
        info = alpha_at(dfa, E_las)
        alpha = float(a.alpha or info["alpha_cm"])
        info["alpha_used_cm"] = alpha
        summary["alpha_630"] = info
        summary["absorptance_file"] = str(a.absorptance)
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
        ax[0].plot(dfa.E_eV, dfa.A, "k-"); ax[0].axvline(E_las, color="C3", ls="--", label=f"{a.wavelength_nm:.0f} nm")
        ax[0].set_xlabel("E (eV)"); ax[0].set_ylabel("absorptance A"); ax[0].legend()
        ax[1].semilogy(dfa.E_eV, dfa.alpha_cm, "k-"); ax[1].axvline(E_las, color="C3", ls="--")
        ax[1].axhline(info["alpha_cm"], color="C3", lw=0.8)
        ax[1].set_xlabel("E (eV)"); ax[1].set_ylabel(r"$\alpha = -\ln(1-A)/d$ (cm$^{-1}$)")
        ax[1].set_title(f"alpha(630 nm) = {info['alpha_cm']:.3g} /cm" + ("  (A SATURATED: lower bound)" if info["saturated"] else ""),
                        fontsize=9)
        fig.tight_layout(); fig.savefig(fig_dir / "alpha_vs_E.png", dpi=150); plt.close(fig)

        # ── 3. old scenario method ────────────────────────────────────────────
        qs, D, names = scenario_derivatives()
        old = scenario_sigma(None, D)
        at = scenario_sigma(alpha, D)
        grid = np.logspace(4, 6, 41)
        sv = [scenario_sigma(x, D) for x in grid]
        pd.DataFrame({"alpha_cm": grid, "sigma_scenario_mean": [s["sigma"] for s in sv],
                      **{f"rms_{n}": [s["per_scenario_rms"][k] for s in sv] for k, n in enumerate(names)}}
                     ).to_csv(out / "scenario_sigma_vs_alpha.csv", index=False)
        summary["scenario_method"] = {"sanity_all_alpha_average": old["sigma"], "expected": 0.493608,
                                      "sigma_at_alpha_630": at["sigma"],
                                      "per_scenario_rms_at_alpha_630": dict(zip(names, at["per_scenario_rms"])),
                                      "dlog10_alpha": 1.0}
        fig, axs = plt.subplots(figsize=(5.5, 3.8))
        axs.semilogx(grid, [s["sigma"] for s in sv], "k-", lw=2, label="mean over scenarios")
        for k, n in enumerate(names):
            axs.semilogx(grid, [s["per_scenario_rms"][k] for s in sv], lw=0.8, label=n[:28])
        axs.axvline(alpha, color="C3", ls="--", label=r"$\alpha$(630 nm)")
        axs.set_xlabel(r"$\alpha$ (cm$^{-1}$)"); axs.set_ylabel(r"$\sigma$ = RMS |d log$_{10}\tau$/d log$_{10}\alpha$|")
        axs.legend(fontsize=6); fig.tight_layout(); fig.savefig(fig_dir / "scenario_sigma_vs_alpha.png", dpi=150); plt.close(fig)

    # ── 2. DD vs 0D ──────────────────────────────────────────────────────────
    trpl = pd.read_csv(run_dir / "trpl_used.csv")
    q_shift = 2.0 * C["Vt"] * np.log(cfg["physics"]["npulse_cm3"] / npulse_run)     # data_utils.t_pl_to_qfls_tau
    q_data = trpl["qfls"].values + q_shift
    summary["npulse_cm3"] = cfg["physics"]["npulse_cm3"]
    summary["npulse_run_cm3"] = npulse_run
    summary["data_qfls_shift_eV"] = float(q_shift)
    with np.errstate(all="ignore"):
        y_data = np.log10(trpl["tau_diff"].values)
    ni = np.sqrt(C["Nc"] * C["Nv"]) * np.exp(-cfg["physics"]["Eg_eV"] / (2 * C["Vt"]))
    q_grid = np.linspace(0.9, 2 * C["Vt"] * np.log(cfg["physics"]["npulse_cm3"] / ni), 256)
    theta = np.array(json.loads(Path(a.best_fit).read_text())["theta_best"], float)
    summary["best_fit_physical"] = physical_traps(theta, prior)
    res = [dd_vs_0d(theta, prior, cfg, C, alpha, q_data, q_grid, "best_fit", out, y_data)]
    summary["sigma_model"] = res[0]["rmse_data_points"]
    if not a.no_robustness:
        for f, lab in ((0.5, "alpha_half"), (2.0, "alpha_double")):
            res.append(dd_vs_0d(theta, prior, cfg, C, alpha * f, q_data, q_grid, lab, out, y_data))
        if a.best_fit_trpl and Path(a.best_fit_trpl).exists() and not a.dd_only:
            th_t = np.array(json.loads(Path(a.best_fit_trpl).read_text())["theta_best"], float)
            res.append(dd_vs_0d(th_t, prior, cfg, C, alpha, q_data, q_grid, "trpl_best_fit", out, y_data))
    summary["dd_vs_0d"] = [{k: v for k, v in r.items() if k != "curves"} for r in res]

    fig, axs = plt.subplots(2, 1, figsize=(6, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1.3]})
    c = res[0]["curves"]
    axs[0].plot(q_data, y_data, "o", ms=2.5, mfc="none", color="0.4", label="data")
    axs[0].plot(c["q0"], c["lt0"], "-", color="#00CED1", label="0D rate equations")
    axs[0].plot(c["qd"], c["ltd"], "--", color="C3", label=f"drift-diffusion, $\\alpha$ = {alpha:.3g} /cm")
    axs[0].set_xlim(q_data.min() - 0.02, q_data.max() + 0.02)
    axs[0].set_ylabel(r"log$_{10}$ $\tau_{diff}$ (s)"); axs[0].legend(fontsize=8)
    axs[0].set_title(f"sigma_model = RMSE(DD - 0D) at the data points = {summary['sigma_model']:.4f} dec", fontsize=9)
    for r, col in zip(res, ("C3", "C1", "C4", "C2")):
        d = r["curves"]["y_dd"] - r["curves"]["y_0d"]
        axs[1].plot(q_data, d, color=col, lw=1, label=f"{r['label']}: {r['rmse_data_points']:.3f}")
    axs[1].axhline(0, color="k", lw=0.5); axs[1].set_xlabel("QFLS (eV)"); axs[1].set_ylabel("DD - 0D (dec)")
    axs[1].legend(fontsize=6)
    fig.tight_layout(); fig.savefig(fig_dir / "dd_vs_0d.png", dpi=150); plt.close(fig)

    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps({k: summary[k] for k in summary if k in ("alpha_630", "sigma_model", "scenario_method")},
                     indent=1, default=float))
    for r in summary["dd_vs_0d"]:
        print(f"  {r['label']:14s} alpha {r['alpha_cm']:.3g}: RMSE data {r['rmse_data_points']:.4f} "
              f"({r['n_data_points_compared']}/{r['n_data_points']} points), grid {r['rmse_grid_0.9_to_qmax']:.4f}, "
              f"DD {r['time_dd_s']:.1f} s")


if __name__ == "__main__":
    main()
