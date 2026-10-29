"""
validate_dd_1trap.py — validation of the Python drift-diffusion port (dd_model.py)
==================================================================================

1. Scenario check: is simulation_results-shallow-plateau_alphas-mu-1/0D_shallow-plateau.csv the legacy ODE
   (trpl_module.solve_transient) at the paper's 1-trap synthetic case?  If yes, the shallow-plateau MATLAB
   runs are the paper case; if not, the trap-high-Nt scenario (parameters as set in the .m file) is used
   for the MATLAB comparison.
2. 0D limit: DD with a flat initial profile vs the legacy ODE (via inference/ode.py: atol_vec, t from 0.0,
   training post-processing), paper case, inference constants.  Target <= 0.01 dec RMS.
3. DD vs MATLAB (pdepe) CSVs for alpha = 1e4 ... 1e6, MATLAB constants: RMS of Delta log10 tau_diff on
   QFLS in [0.9, 2 kT ln(n_pulse/n_i)] (target <= 0.02 dec), and the derivative curves
   d log10 tau / d log10 alpha of homoskedastic_sigma_derivative_alpha.m.
4. Tolerance / mesh / quadrature study (rtol, nx, midpoint vs nodal source, pdepe-like tolerances).
5. Paper case DD (alpha sweep) vs legacy 0D, inference constants (informative: size of the DD-0D gap).
6. 2-trap smoke run (two active traps): flat DD vs legacy 0D, and the exp-profile DD.
7. Runtime per DD solve (mean +- SD over repeats).

Outputs (CSVs, figures, summary.json) go to
rapid-trpl-bayes/results/2trap_85-15_film/sigma_model/dd_validation/.

Run (from anywhere):  python model-error-analysis/validate_dd_1trap.py [--workers 4] [--repeats 5]
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import inspect
import json
import platform
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))
import dd_model as dd  # noqa: E402

OUT = REPO / "results" / "2trap_85-15_film" / "sigma_model" / "dd_validation"

EG = 1.625
#: paper's 1-trap synthetic test case
PAPER = dict(Et=[1.4932], Nt=[4.1975e16], taun=[3.0115e-9], taup=[6.822e-8], krad=1e-10, n_pulse=1.6e17)
#: trap-high-Nt scenario as currently set in pdex1_sproul_Chris_frequency.m (traps 2, 3 off)
HIGH_NT = dict(Et=[EG / 2], Nt=[1e17], taun=[1e-7], taup=[1e-6], krad=1e-11, n_pulse=1.6e17)
#: 2-trap smoke case: paper trap + a deep trap
TWO_TRAP = dict(Et=[1.4932, 0.85], Nt=[4.1975e16, 1e16], taun=[3.0115e-9, 1e-7], taup=[6.822e-8, 1e-6],
                krad=1e-10, n_pulse=1.6e17)
ALPHAS = [1e4, 5e4, 1e5, 5e5, 1e6]
D_FILM = 450e-7
#: thermal voltage of the training HDF5 / inference nominal QFLS axis (sobol generator: K T / Q)
VT_INFERENCE = 1.38064852e-23 * 300 / 1.602176462e-19
#: constants dict as inference/ode.py gets it from the training HDF5
INF_CONSTANTS = dict(Nc=dd.INFERENCE_CONSTANTS["Nc"], Nv=dd.INFERENCE_CONSTANTS["Nv"], Vt=VT_INFERENCE,
                     tmin=1e-11, tmax=1e5, PL_DECAY_MAGNITUDE=12)
#: MATLAB constants with Eg removed (Eg is passed explicitly)
MATLAB_C = {k: v for k, v in dd.MATLAB_CONSTANTS.items() if k != "Eg"}
INF_C = dict(dd.INFERENCE_CONSTANTS)
TARGET_0D = 0.01
TARGET_MATLAB = 0.02
N_GRID = 256
QFLS_MIN = 0.9
MATLAB_DIR = HERE

# categorical slots 1-5 (fixed order) of the dataviz reference palette, light mode
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


# ----------------------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------------------
def rms(a) -> float:
    a = np.asarray(a, np.float64)
    a = a[np.isfinite(a)]
    return float(np.sqrt(np.mean(a ** 2))) if len(a) else float("nan")


def maxabs(a) -> float:
    a = np.asarray(a, np.float64)
    a = a[np.isfinite(a)]
    return float(np.max(np.abs(a))) if len(a) else float("nan")


def qfls_max(kT: float, Nc: float, Nv: float, n_pulse: float, Eg: float = EG) -> float:
    """2 kT ln(n_pulse / n_i)."""
    return float(2 * kT * np.log(n_pulse / np.sqrt(Nc * Nv * np.exp(-Eg / kT))))


def style(ax):
    ax.grid(True, color=GRID, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelcolor=INK)


class FixedPrior:
    """Minimal stand-in for inference.prior (to_physical / n_traps / Eg) for one fixed parameter set."""

    def __init__(self, Eg, P):
        self.Eg, self.P, self.n_traps = Eg, P, len(P["Nt"])

    def to_physical(self, _theta):
        row = {"krad_cm3s": self.P["krad"]}
        for k in range(self.n_traps):
            row.update({f"Nt_{k + 1}_cm-3": self.P["Nt"][k], f"Et_{k + 1}_eV": self.P["Et"][k],
                        f"tau_n_{k + 1}_s": self.P["taun"][k], f"tau_p_{k + 1}_s": self.P["taup"][k]})
        return pd.DataFrame([row])


def legacy_log10_tau(P, qfls_points, axis="nominal"):
    """Legacy ODE log10 tau_diff at QFLS points exactly as the inference computes it (inference/ode.py)."""
    from inference import ode
    th = np.zeros(1 + 4 * len(P["Nt"]))
    return ode.ode_log10_tau(th, FixedPrior(EG, P), P["n_pulse"], INF_CONSTANTS, qfls_points, 300.0, axis=axis)


def legacy_curve(P, T, Nc, Nv):
    """Raw legacy solve_transient curve (physical QFLS, tau, PL) for arbitrary constants."""
    leg = dd.legacy_module()
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        out = leg.solve_transient([1e-11, 1e5], P["n_pulse"], EG, Nc, Nv, P["krad"], P["Nt"], P["Et"],
                                  P["taun"], P["taup"], T)
    return out[4], out[5], out[8]


def dd_job(kw: dict) -> dict:
    """One simulate_dd run in a worker (no profiles; numba of calcequil compiles once per worker)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return dd.simulate_dd(**kw)


def run_parallel(jobs: dict, workers: int) -> dict:
    from joblib import Parallel, delayed
    keys = list(jobs)
    res = Parallel(n_jobs=min(workers, len(keys)))(delayed(dd_job)(jobs[k]) for k in keys)
    return dict(zip(keys, res))


def dd_kwargs(P, alpha, consts, **extra) -> dict:
    kw = dict(Et=P["Et"], Nt=P["Nt"], taun=P["taun"], taup=P["taup"], krad=P["krad"], n_pulse=P["n_pulse"],
              alpha=alpha, d=D_FILM, mu=1.0, Eg=EG, **consts)
    kw.update(extra)
    return kw


def matlab_csv(scenario_dir: str, alpha: float) -> pd.DataFrame:
    return pd.read_csv(MATLAB_DIR / scenario_dir / f"alpha-{alpha:g}-mu-1.csv")


def deriv_pairs_like_matlab(curves: list[tuple[np.ndarray, np.ndarray]], alphas, qfls_common, qmin, qmax):
    """
    homoskedastic_sigma_derivative_alpha.m: per alpha file, filter tau > 0 and QFLS in [qmin, qmax], unique,
    linear interp1 (NaN outside) onto qfls_common, centred d log10 tau / d log10 alpha (one-sided at ends).
    """
    lt = []
    for q, tau in curves:
        q, tau = np.asarray(q, np.float64), np.asarray(tau, np.float64)
        ok = np.isfinite(q) & np.isfinite(tau) & (tau > 0) & (q >= qmin) & (q <= qmax)
        qu, ia = np.unique(q[ok], return_index=True)
        tu = tau[ok][ia]
        lt.append(np.interp(qfls_common, qu, np.log10(tu), left=np.nan, right=np.nan) if len(qu) >= 2
                  else np.full(len(qfls_common), np.nan))
    la = np.log10(np.asarray(alphas, np.float64))
    nF = len(alphas)
    out = np.full((len(qfls_common), nF), np.nan)
    for k in range(nF):
        lo, hi = (0, 1) if k == 0 else ((nF - 2, nF - 1) if k == nF - 1 else (k - 1, k + 1))
        out[:, k] = (lt[hi] - lt[lo]) / (la[hi] - la[lo])
    return out


# ----------------------------------------------------------------------------------------------------------
# sections
# ----------------------------------------------------------------------------------------------------------
def section_scenario() -> dict:
    """Is 0D_shallow-plateau.csv the legacy ODE at the paper parameters?"""
    res = {}
    for scen, fname in [("shallow-plateau", "simulation_results-shallow-plateau_alphas-mu-1/0D_shallow-plateau.csv"),
                        ("shallow_photodoping",
                         "simulation_results-shallow_photodoping_alphas-mu-1/0D_shallow-trap-photodoping.csv")]:
        df = pd.read_csv(MATLAB_DIR / fname)
        q0, tau0, pl0 = df.iloc[:, 2].values, df.iloc[:, 3].values, df.iloc[:, 1].values
        r = dict(file=fname, qfls_t0_csv=float(q0[0]))
        for lab, (T, Nc) in {"matlab_constants": (MATLAB_C["kT"] / dd.KB_LEGACY, MATLAB_C["Nc"]),
                             "inference_constants": (INF_C["T"], INF_C["Nc"])}.items():
            q, tau, pl = legacy_curve(PAPER, T, Nc, Nc)
            qhi = min(np.nanmax(q0), np.nanmax(q))
            grid = np.linspace(QFLS_MIN, qhi, N_GRID)
            dlt = dd.curve_log10_tau(q, tau, pl, grid) - dd.curve_log10_tau(q0, tau0, pl0, grid)
            r[lab] = dict(qfls_t0_legacy=float(q[0]), rms_dec=rms(dlt), max_dec=maxabs(dlt),
                          n_points=int(np.isfinite(dlt).sum()))
        r["equals_paper_case"] = bool(min(r["matlab_constants"]["rms_dec"],
                                          r["inference_constants"]["rms_dec"]) <= TARGET_MATLAB)
        res[scen] = r
    paper = res["shallow-plateau"]["equals_paper_case"]
    res["decision"] = dict(
        shallow_plateau_is_paper_case=paper,
        matlab_scenario="shallow-plateau" if paper else "trap-high-Nt",
        note=("0D_shallow-plateau.csv starts at the QFLS of the inference constants (T = 300 K, Nc = 2.21359e18) "
              "but its tau(QFLS) differs from the legacy ODE at the paper parameters by ~0.9 dec RMS: it is a "
              "different parameter set, so the trap-high-Nt scenario (parameters known from the .m file) is used."
              if not paper else "shallow-plateau 0D CSV reproduces the legacy ODE at the paper parameters."))
    return res


def section_0d_limit(workers: int) -> dict:
    jobs = {"flat_mu1": dd_kwargs(PAPER, 1e5, INF_C, profile="flat"),
            "flat_mu1_volumePL": dd_kwargs(PAPER, 1e5, INF_C, profile="flat", pl_mode="volume"),
            "flat_mu1_nx20": dd_kwargs(PAPER, 1e5, INF_C, profile="flat", nx=20)}
    runs = run_parallel(jobs, workers)
    qmax = qfls_max(VT_INFERENCE, INF_C["Nc"], INF_C["Nv"], PAPER["n_pulse"])
    out = dict(constants=dict(INF_C, Vt_axis=VT_INFERENCE), qfls_max_eV=qmax, target_dec=TARGET_0D, runs={})
    tab = None
    for lo in (QFLS_MIN, 0.8):
        grid = np.linspace(lo, qmax, N_GRID)
        y_leg = legacy_log10_tau(PAPER, grid)
        for k, r in runs.items():
            y = dd.log10_tau_on_qfls(r, grid, vt=VT_INFERENCE)
            d = y - y_leg
            out["runs"].setdefault(k, {})[f"qfls_{lo:g}_to_max"] = dict(rms_dec=rms(d), max_dec=maxabs(d),
                                                                          n_points=int(np.isfinite(d).sum()))
            out["runs"][k]["runtime_s"] = r["info"]["runtime_s"]
            if lo == QFLS_MIN and k == "flat_mu1":
                tab = pd.DataFrame(dict(QFLS_eV=grid, log10_tau_dd_flat=y, log10_tau_legacy_0d=y_leg, diff_dec=d))
                dd.to_matlab_table(r).to_csv(OUT / "dd_paper_flat_profile.csv", index=False)
    tab.to_csv(OUT / "0d_limit_comparison.csv", index=False)
    out["rms_dec"] = out["runs"]["flat_mu1"][f"qfls_{QFLS_MIN:g}_to_max"]["rms_dec"]
    out["pass"] = bool(out["rms_dec"] <= TARGET_0D)

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.4, 5.6), sharex=True, gridspec_kw=dict(height_ratios=[3, 1.3]))
    a1.plot(tab.QFLS_eV, tab.log10_tau_legacy_0d, color=COLORS[1], lw=2, label="legacy 0D (trpl_module)")
    a1.plot(tab.QFLS_eV, tab.log10_tau_dd_flat, color=COLORS[0], lw=2, ls="--", label="DD, flat profile")
    a1.set_ylabel(r"log$_{10}$ $\tau_{diff}$ (s)", color=INK)
    a1.legend(frameon=False)
    a1.set_title(f"0D limit, paper 1-trap case: RMS {out['rms_dec']:.1e} dec", color=INK, fontsize=10)
    a2.plot(tab.QFLS_eV, tab.diff_dec, color=COLORS[0], lw=2)
    a2.axhline(0, color=INK2, lw=0.8)
    a2.set_ylabel(r"$\Delta$ (dec)", color=INK)
    a2.set_xlabel("QFLS (eV)", color=INK)
    for a in (a1, a2):
        style(a)
    fig.tight_layout()
    fig.savefig(OUT / "0d_limit.png", dpi=150)
    plt.close(fig)
    return out


def compare_with_matlab(res: dict, ml: pd.DataFrame, grid: np.ndarray, qmin: float, qmax: float) -> dict:
    y = dd.log10_tau_on_qfls(res, grid)
    ym = dd.curve_log10_tau(ml.QFLS_eV.values, ml.tau_diff_s.values, ml.PL_norm.values, grid)
    yl = dd.log10_tau_on_qfls(res, grid, kind="linear")
    yml = dd.curve_log10_tau(ml.QFLS_eV.values, ml.tau_diff_s.values, ml.PL_norm.values, grid, kind="linear")
    # interpolation-free: same output times
    with np.errstate(all="ignore"):
        dt = np.log10(res["tau_diff"]) - np.log10(ml.tau_diff_s.values)
        dpl = np.log10(res["PL_norm"]) - np.log10(ml.PL_norm.values)
    sel = (ml.QFLS_eV.values >= qmin) & (ml.QFLS_eV.values <= qmax)
    return dict(y=y, ym=ym, stats=dict(
        rms_dec=rms(y - ym), max_dec=maxabs(y - ym), rms_dec_linear_interp=rms(yl - yml),
        rms_dec_same_times=rms(dt[sel]), max_dec_same_times=maxabs(dt[sel]),
        rms_log10_PLnorm_same_times=rms(dpl[sel]), n_points=int(np.isfinite(y - ym).sum()),
        runtime_s=res["info"]["runtime_s"], nfev=res["info"]["nfev"]))


def section_matlab(scenario: str, workers: int) -> tuple[dict, dict]:
    P = PAPER if scenario == "shallow-plateau" else HIGH_NT
    sdir = f"simulation_results-{scenario}_alphas-mu-1"
    runs = run_parallel({a: dd_kwargs(P, a, MATLAB_C) for a in ALPHAS}, workers)
    kT = MATLAB_C["kT"]
    qmax = qfls_max(kT, MATLAB_C["Nc"], MATLAB_C["Nv"], P["n_pulse"])
    grid = np.linspace(QFLS_MIN, qmax, N_GRID)
    out = dict(scenario=scenario, matlab_dir=sdir, parameters=P, constants=dict(MATLAB_C, Eg=EG),
               qfls_range_eV=[QFLS_MIN, qmax], target_dec=TARGET_MATLAB, per_alpha={})
    tab = dict(QFLS_eV=grid)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.0, 6.2), sharex=True, gridspec_kw=dict(height_ratios=[3, 1.4]))
    for i, a in enumerate(ALPHAS):
        r = runs[a]
        dd.to_matlab_table(r).to_csv(OUT / f"dd_{scenario}_alpha-{a:g}-mu-1.csv", index=False)
        ml = matlab_csv(sdir, a)
        c = compare_with_matlab(r, ml, grid, QFLS_MIN, qmax)
        out["per_alpha"][f"{a:g}"] = c["stats"]
        tab[f"log10_tau_dd_alpha_{a:g}"] = c["y"]
        tab[f"log10_tau_matlab_alpha_{a:g}"] = c["ym"]
        tab[f"diff_dec_alpha_{a:g}"] = c["y"] - c["ym"]
        lab = rf"$\alpha$ = {a:.0e} cm$^{{-1}}$"
        a1.plot(grid, c["y"], color=COLORS[i], lw=2, label=lab + " DD (Python)")
        a1.plot(grid[::8], c["ym"][::8], ls="none", marker="o", ms=4.5, mfc="none", mec=COLORS[i], mew=1.2)
        a2.plot(grid, c["y"] - c["ym"], color=COLORS[i], lw=2)
    a1.plot([], [], ls="none", marker="o", ms=4.5, mfc="none", mec=INK2, label="MATLAB pdepe (markers)")
    pd.DataFrame(tab).to_csv(OUT / f"matlab_comparison_{scenario}.csv", index=False)
    worst = max(v["rms_dec"] for v in out["per_alpha"].values())
    out["max_rms_dec_over_alpha"] = worst
    out["pass"] = bool(worst <= TARGET_MATLAB)
    a1.set_ylabel(r"log$_{10}$ $\tau_{diff}$ (s)", color=INK)
    a1.legend(frameon=False, fontsize=7.5)
    a1.set_title(f"DD vs MATLAB, {scenario} (MATLAB constants): max RMS {worst:.1e} dec", color=INK, fontsize=10)
    a2.axhline(0, color=INK2, lw=0.8)
    a2.set_ylabel(r"$\Delta$ (dec)", color=INK)
    a2.set_xlabel("QFLS (eV)", color=INK)
    for ax in (a1, a2):
        style(ax)
    fig.tight_layout()
    fig.savefig(OUT / f"matlab_comparison_{scenario}.png", dpi=150)
    plt.close(fig)

    # derivative curves (homoskedastic_sigma_derivative_alpha.m)
    kTd = 8.617333262e-5 * 300
    qmax_d = 2 * kTd * np.log(P["n_pulse"] / np.sqrt(2.2e18 * 2.2e18 * np.exp(-EG / kTd)))
    qc = np.linspace(QFLS_MIN, qmax_d, N_GRID)
    ref = pd.read_csv(MATLAB_DIR / "derivative_analysis_output" / f"deriv_per_file_{sdir}.csv")
    ref_d = ref.iloc[:, 1:].values
    d_ml = deriv_pairs_like_matlab([(matlab_csv(sdir, a).QFLS_eV.values, matlab_csv(sdir, a).tau_diff_s.values)
                                    for a in ALPHAS], ALPHAS, qc, QFLS_MIN, qmax_d)
    d_py = deriv_pairs_like_matlab([(runs[a]["QFLS"], runs[a]["tau_diff"]) for a in ALPHAS], ALPHAS, qc,
                                   QFLS_MIN, qmax_d)
    deriv = dict(reference_csv=str(Path("derivative_analysis_output") / f"deriv_per_file_{sdir}.csv"),
                 python_reimplementation_vs_matlab_csv_max_abs=maxabs(d_ml - ref_d),
                 dd_vs_matlab_per_alpha={f"{a:g}": dict(rms=rms(d_py[:, k] - ref_d[:, k]),
                                                       max_abs=maxabs(d_py[:, k] - ref_d[:, k]),
                                                       rms_of_matlab_derivative=rms(ref_d[:, k]))
                                         for k, a in enumerate(ALPHAS)})
    dtab = dict(QFLS_eV=qc)
    fig, (b1, b2) = plt.subplots(2, 1, figsize=(7.0, 6.0), sharex=True, gridspec_kw=dict(height_ratios=[3, 1.4]))
    for k, a in enumerate(ALPHAS):
        dtab[f"deriv_dd_alpha_{a:g}"] = d_py[:, k]
        dtab[f"deriv_matlab_alpha_{a:g}"] = ref_d[:, k]
        b1.plot(qc, d_py[:, k], color=COLORS[k], lw=2, label=rf"$\alpha$ = {a:.0e} DD")
        b1.plot(qc[::8], ref_d[::8, k], ls="none", marker="o", ms=4.5, mfc="none", mec=COLORS[k], mew=1.2)
        b2.plot(qc, d_py[:, k] - ref_d[:, k], color=COLORS[k], lw=2)
    b1.plot([], [], ls="none", marker="o", ms=4.5, mfc="none", mec=INK2, label="MATLAB deriv_per_file (markers)")
    pd.DataFrame(dtab).to_csv(OUT / f"derivative_comparison_{scenario}.csv", index=False)
    b1.set_ylabel(r"d log$_{10}\tau$ / d log$_{10}\alpha$", color=INK)
    b1.legend(frameon=False, fontsize=7.5)
    b1.set_title(f"Derivative curves, {scenario}", color=INK, fontsize=10)
    b2.axhline(0, color=INK2, lw=0.8)
    b2.set_ylabel(r"$\Delta$", color=INK)
    b2.set_xlabel("QFLS (eV)", color=INK)
    for ax in (b1, b2):
        style(ax)
    fig.tight_layout()
    fig.savefig(OUT / f"derivative_comparison_{scenario}.png", dpi=150)
    plt.close(fig)
    return out, deriv


def section_convergence(scenario: str, workers: int) -> dict:
    """rtol / mesh / source-quadrature / pdepe-like tolerance study against MATLAB and a fine reference."""
    P = PAPER if scenario == "shallow-plateau" else HIGH_NT
    sdir = f"simulation_results-{scenario}_alphas-mu-1"
    variants = {"base_nx150_rtol1e-8_midpoint": {},
                "rtol1e-6": dict(rtol=1e-6), "rtol1e-10": dict(rtol=1e-10),
                "nx300": dict(nx=300), "nx600": dict(nx=600),
                "nodal_quadrature": dict(quadrature="nodal"),
                "pdepe_tolerances_rtol1e-3_atol1e-6": dict(rtol=1e-3, atol_np=1e-6, atol_nt=1e-6 / P["Nt"][0]),
                "reference_nx600_rtol1e-10": dict(nx=600, rtol=1e-10)}
    alphas = [1e4, 1e6]
    jobs = {(a, v): dd_kwargs(P, a, MATLAB_C, **kw) for a in alphas for v, kw in variants.items()}
    runs = run_parallel(jobs, workers)
    qmax = qfls_max(MATLAB_C["kT"], MATLAB_C["Nc"], MATLAB_C["Nv"], P["n_pulse"])
    grid = np.linspace(QFLS_MIN, qmax, N_GRID)
    out = {}
    for a in alphas:
        ml = matlab_csv(sdir, a)
        ym = dd.curve_log10_tau(ml.QFLS_eV.values, ml.tau_diff_s.values, ml.PL_norm.values, grid)
        yref = dd.log10_tau_on_qfls(runs[(a, "reference_nx600_rtol1e-10")], grid)
        out[f"{a:g}"] = {v: dict(rms_vs_matlab_dec=rms(dd.log10_tau_on_qfls(runs[(a, v)], grid) - ym),
                                 rms_vs_reference_dec=rms(dd.log10_tau_on_qfls(runs[(a, v)], grid) - yref),
                                 runtime_s=runs[(a, v)]["info"]["runtime_s"], nfev=runs[(a, v)]["info"]["nfev"])
                         for v in variants}
    rows = [dict(case=scenario, alpha=a, variant=v, **s) for a, d in out.items() for v, s in d.items()]

    # paper case, inference constants: mesh / rtol convergence against nx = 600, rtol = 1e-10
    pv = {"nx150": {}, "nx300": dict(nx=300), "nx600": dict(nx=600), "rtol1e-6": dict(rtol=1e-6),
          "reference_nx600_rtol1e-10": dict(nx=600, rtol=1e-10)}
    pjobs = {(a, v): dd_kwargs(PAPER, a, INF_C, **kw) for a in (1e5, 1e6) for v, kw in pv.items()}
    pruns = run_parallel(pjobs, workers)
    qmax_p = qfls_max(VT_INFERENCE, INF_C["Nc"], INF_C["Nv"], PAPER["n_pulse"])
    grid_p = np.linspace(QFLS_MIN, qmax_p, N_GRID)
    paper = {}
    for a in (1e5, 1e6):
        yref = dd.log10_tau_on_qfls(pruns[(a, "reference_nx600_rtol1e-10")], grid_p, vt=VT_INFERENCE)
        paper[f"{a:g}"] = {v: dict(rms_vs_reference_dec=rms(dd.log10_tau_on_qfls(pruns[(a, v)], grid_p,
                                                                                 vt=VT_INFERENCE) - yref),
                                   runtime_s=pruns[(a, v)]["info"]["runtime_s"], nfev=pruns[(a, v)]["info"]["nfev"])
                           for v in pv}
    rows += [dict(case="paper_inference_constants", alpha=a, variant=v, **s)
             for a, d in paper.items() for v, s in d.items()]
    pd.DataFrame(rows).to_csv(OUT / "convergence_study.csv", index=False)
    return {scenario: out, "paper_inference_constants": paper,
            "note": ("MATLAB and the Python DD share the nx = 150 mesh, so they agree to < 1e-3 dec even where "
                     "the mesh is not converged (alpha = 1e6: 1/alpha = 10 nm vs h = 3 nm); "
                     "rms_vs_reference measures that discretisation error.")}


def section_paper_alpha(workers: int) -> dict:
    runs = run_parallel({a: dd_kwargs(PAPER, a, INF_C) for a in ALPHAS}, workers)
    qmax = qfls_max(VT_INFERENCE, INF_C["Nc"], INF_C["Nv"], PAPER["n_pulse"])
    grid = np.linspace(QFLS_MIN, qmax, N_GRID)
    y_leg = legacy_log10_tau(PAPER, grid)
    tab = dict(QFLS_eV=grid, log10_tau_legacy_0d=y_leg)
    out = dict(constants=dict(INF_C, Vt_axis=VT_INFERENCE), qfls_range_eV=[QFLS_MIN, qmax], per_alpha={},
               note=("Below ~1.3 eV the DD - 0D gap is a pure shift of the QFLS axis: the DD PL is normalised to "
                     "PL(t0) n_pulse^2, and PL(t0) of the exp(-alpha x) profile (with escape weighting) exceeds "
                     "that of a flat profile by pl_t0_enhancement, so the DD axis lies kT ln(enhancement) low. "
                     "rms_after_axis_shift removes that offset; the rest is the early-time diffusion regime."))
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.0, 6.2), sharex=True, gridspec_kw=dict(height_ratios=[3, 1.4]))
    a1.plot(grid, y_leg, color=INK, lw=2.4, label="legacy 0D")
    for i, a in enumerate(ALPHAS):
        r = runs[a]
        dd.to_matlab_table(r).to_csv(OUT / f"dd_paper_alpha-{a:g}-mu-1.csv", index=False)
        y = dd.log10_tau_on_qfls(r, grid, vt=VT_INFERENCE)
        tab[f"log10_tau_dd_alpha_{a:g}"] = y
        # QFLS-axis offset of the PL/max(PL) n_pulse^2 calibration (steep initial profile, escape weighting)
        ratio = dd.initial_pl_enhancement(a, D_FILM, r["info"]["nx"], r["info"]["kT"])
        dq = r["info"]["kT"] * np.log(ratio)
        y_shift = dd.curve_log10_tau(dd.nominal_qfls(r, VT_INFERENCE) + dq, r["tau_diff"], r["PL"], grid)
        lowq = grid < 1.3
        out["per_alpha"][f"{a:g}"] = dict(
            rms_dd_minus_0d_dec=rms(y - y_leg), max_dec=maxabs(y - y_leg),
            mean_dd_minus_0d_below_1p3eV_dec=float(np.nanmean((y - y_leg)[lowq])),
            pl_t0_enhancement=ratio, qfls_axis_offset_meV=1e3 * dq,
            rms_after_axis_shift_dec=rms(y_shift - y_leg),
            rms_after_axis_shift_below_1p3eV_dec=rms((y_shift - y_leg)[lowq]),
            runtime_s=r["info"]["runtime_s"])
        tab[f"log10_tau_dd_axis_shifted_alpha_{a:g}"] = y_shift
        a1.plot(grid, y, color=COLORS[i], lw=2, label=rf"DD $\alpha$ = {a:.0e} cm$^{{-1}}$")
        a2.plot(grid, y - y_leg, color=COLORS[i], lw=2)
    pd.DataFrame(tab).to_csv(OUT / "paper_case_dd_vs_0d.csv", index=False)
    a1.set_ylabel(r"log$_{10}$ $\tau_{diff}$ (s)", color=INK)
    a1.legend(frameon=False, fontsize=7.5)
    a1.set_title(r"Paper 1-trap case: DD ($\mu$ = 1 cm$^2$/Vs) vs legacy 0D (inference constants)",
                 color=INK, fontsize=10)
    a2.axhline(0, color=INK2, lw=0.8)
    a2.set_ylabel(r"DD $-$ 0D (dec)", color=INK)
    a2.set_xlabel("QFLS (eV)", color=INK)
    for ax in (a1, a2):
        style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "paper_case_dd_vs_0d.png", dpi=150)
    plt.close(fig)
    return out


def section_two_trap(workers: int) -> dict:
    jobs = {"flat": dd_kwargs(TWO_TRAP, 1e5, INF_C, profile="flat"),
            "exp_alpha1e5": dd_kwargs(TWO_TRAP, 1e5, INF_C)}
    runs = run_parallel(jobs, workers)
    qmax = qfls_max(VT_INFERENCE, INF_C["Nc"], INF_C["Nv"], TWO_TRAP["n_pulse"])
    grid = np.linspace(QFLS_MIN, qmax, N_GRID)
    y_leg = legacy_log10_tau(TWO_TRAP, grid)
    y_flat = dd.log10_tau_on_qfls(runs["flat"], grid, vt=VT_INFERENCE)
    y_exp = dd.log10_tau_on_qfls(runs["exp_alpha1e5"], grid, vt=VT_INFERENCE)
    nt0 = np.asarray(runs["exp_alpha1e5"]["info"]["nt0"])
    Nt = np.asarray(TWO_TRAP["Nt"])
    occ = runs["exp_alpha1e5"]["nt_avg"] / Nt[:, None]
    out = dict(parameters=TWO_TRAP, constants=dict(INF_C, Vt_axis=VT_INFERENCE),
               flat_vs_legacy_0d=dict(rms_dec=rms(y_flat - y_leg), max_dec=maxabs(y_flat - y_leg)),
               exp_alpha1e5_vs_legacy_0d=dict(rms_dec=rms(y_exp - y_leg), max_dec=maxabs(y_exp - y_leg)),
               occupancy_t0=(nt0 / Nt).tolist(), occupancy_max=occ.max(axis=1).tolist(),
               occupancy_min=occ.min(axis=1).tolist(),
               both_traps_active=bool(np.all(occ.max(axis=1) - occ.min(axis=1) > 0.01)),
               runtime_s={k: r["info"]["runtime_s"] for k, r in runs.items()},
               nfev={k: r["info"]["nfev"] for k, r in runs.items()})
    for k, r in runs.items():
        dd.to_matlab_table(r).to_csv(OUT / f"dd_2trap_smoke_{k}.csv", index=False)
    pd.DataFrame(dict(QFLS_eV=grid, log10_tau_legacy_0d=y_leg, log10_tau_dd_flat=y_flat,
                      log10_tau_dd_exp_alpha1e5=y_exp)).to_csv(OUT / "2trap_smoke.csv", index=False)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.0, 4.2))
    a1.plot(grid, y_leg, color=COLORS[1], lw=2, label="legacy 0D")
    a1.plot(grid, y_flat, color=COLORS[0], lw=2, ls="--", label="DD flat profile")
    a1.plot(grid, y_exp, color=COLORS[2], lw=2, label=r"DD $\alpha$ = 1e5 cm$^{-1}$")
    a1.set_xlabel("QFLS (eV)", color=INK)
    a1.set_ylabel(r"log$_{10}$ $\tau_{diff}$ (s)", color=INK)
    a1.legend(frameon=False, fontsize=8)
    a1.set_title("2-trap smoke run", color=INK, fontsize=10)
    t = runs["exp_alpha1e5"]["t"]
    for k in range(2):
        a2.semilogx(t, occ[k], color=COLORS[k], lw=2,
                    label=f"trap {k + 1}: E$_t$ = {TWO_TRAP['Et'][k]} eV, N$_t$ = {TWO_TRAP['Nt'][k]:.1e}")
    a2.set_xlabel("time (s)", color=INK)
    a2.set_ylabel(r"$\langle n_t\rangle / N_t$", color=INK)
    a2.legend(frameon=False, fontsize=8)
    a2.set_title(r"trap occupancy (DD, $\alpha$ = 1e5)", color=INK, fontsize=10)
    for ax in (a1, a2):
        style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "2trap_smoke.png", dpi=150)
    plt.close(fig)
    return out


def section_timing(repeats: int) -> dict:
    """Wall time of one DD solve in this process (after a warm-up), paper case, default settings."""
    out = {}
    for lab, kw in {"paper_1trap_alpha1e5_nx150": dd_kwargs(PAPER, 1e5, INF_C),
                    "paper_1trap_alpha1e6_nx150": dd_kwargs(PAPER, 1e6, INF_C),
                    "2trap_alpha1e5_nx150": dd_kwargs(TWO_TRAP, 1e5, INF_C)}.items():
        dd_job(kw)                                                 # warm-up (numba compile of calcequil)
        ts = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            dd_job(kw)
            ts.append(time.perf_counter() - t0)
        out[lab] = dict(mean_s=float(np.mean(ts)), sd_s=float(np.std(ts, ddof=1)) if repeats > 1 else 0.0,
                        repeats=repeats, times_s=ts)
    try:
        out["load_average_1_5_15"] = list(os.getloadavg())
    except OSError:
        pass
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=4, help="worker processes (<= 4 on rekenbeest)")
    ap.add_argument("--repeats", type=int, default=5, help="repeats for the timing")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t_all = time.perf_counter()
    summary = dict(script=str(Path(__file__).relative_to(REPO)), model=str(Path(dd.__file__).relative_to(REPO)),
                   date=time.strftime("%Y-%m-%d %H:%M:%S"), host=platform.node(), workers=args.workers,
                   simulate_dd_signature="simulate_dd" + str(inspect.signature(dd.simulate_dd)),
                   paper_case=PAPER, d_cm=D_FILM, mu_cm2Vs=1.0, Eg_eV=EG,
                   matlab_constants=dd.MATLAB_CONSTANTS, inference_constants=dict(INF_C, Vt_axis=VT_INFERENCE),
                   qfls_metric_range=f"[{QFLS_MIN}, 2 kT ln(n_pulse/n_i)] eV, {N_GRID} points, PCHIP")

    print("[1] scenario check", flush=True)
    summary["scenario_check"] = section_scenario()
    scenario = summary["scenario_check"]["decision"]["matlab_scenario"]
    print("    ->", scenario, {k: v.get("inference_constants", {}).get("rms_dec")
                               for k, v in summary["scenario_check"].items() if k != "decision"}, flush=True)

    print("[2] 0D limit", flush=True)
    summary["0d_limit"] = section_0d_limit(args.workers)
    print(f"    RMS {summary['0d_limit']['rms_dec']:.2e} dec", flush=True)

    print(f"[3] MATLAB comparison ({scenario})", flush=True)
    summary["matlab_comparison"], summary["derivative_curves"] = section_matlab(scenario, args.workers)
    for a, s in summary["matlab_comparison"]["per_alpha"].items():
        print(f"    alpha {a}: RMS {s['rms_dec']:.2e} dec (same times {s['rms_dec_same_times']:.2e})", flush=True)

    print("[4] tolerance / mesh study", flush=True)
    summary["convergence_study"] = section_convergence(scenario, args.workers)

    print("[5] paper case DD vs 0D (alpha sweep)", flush=True)
    summary["paper_case_dd_vs_0d"] = section_paper_alpha(args.workers)

    print("[6] 2-trap smoke run", flush=True)
    summary["two_trap_smoke"] = section_two_trap(args.workers)
    print(f"    flat vs 0D RMS {summary['two_trap_smoke']['flat_vs_legacy_0d']['rms_dec']:.2e} dec, "
          f"both traps active: {summary['two_trap_smoke']['both_traps_active']}", flush=True)

    print("[7] timing", flush=True)
    summary["timing"] = section_timing(args.repeats)
    for k, v in summary["timing"].items():
        if isinstance(v, dict):
            print(f"    {k}: {v['mean_s']:.2f} +- {v['sd_s']:.2f} s", flush=True)

    summary["total_runtime_s"] = time.perf_counter() - t_all
    summary["pass"] = dict(zero_d_limit=summary["0d_limit"]["pass"], matlab=summary["matlab_comparison"]["pass"])
    with open(OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"written {OUT / 'summary.json'} ({summary['total_runtime_s']:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
