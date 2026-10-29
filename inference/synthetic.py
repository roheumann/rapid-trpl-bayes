"""
synthetic.py — noise-free synthetic use case from a parameter set
=================================================================

Builds the synthetic test case of Section 5.1 (Figure 5 / Table 2) from a
"true" parameter vector, in the same format as the original synthetic files:

* tr-PL: ODE (training-data solver and post-processing, ode.py) tau_diff vs
  QFLS, Pchip-interpolated onto ``n_points`` uniform QFLS values inside a
  window (default: the QFLS range of the 85:15 measurement) — columns
  qfls_eV, tau_diff_s;
* ss-PL: exact steady-state model (sspl_tf numpy twin) at log-spaced
  intensities — columns intensity_suns, qfls_eV, plqy.

``write_case`` also writes the true parameters (JSON and human-readable .dat).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def make_case(theta, prior, npulse: float, constants: dict, qfls_window=(0.968, 1.489),
              n_points: int = 100, suns=None, G_per_sun: float = 2.743e21, n_sweep=(1e8, None, 400),
              T: float = 300.0) -> dict:
    """tr-PL and ss-PL tables for one parameter set (noise free)."""
    from scipy.interpolate import PchipInterpolator
    from .ode import simulate_trpl
    from .sspl_tf import SsplSpecNumpy, KB_EV
    q, lt = simulate_trpl(theta, prior, npulse, constants, T)
    lo, hi = max(qfls_window[0], q.min()), min(qfls_window[1], q.max())
    qg = np.linspace(lo, hi, n_points)
    trpl = pd.DataFrame({"qfls_eV": qg, "tau_diff_s": 10 ** PchipInterpolator(q, lt)(qg)})
    suns = np.logspace(-2, 0.5, 16) if suns is None else np.asarray(suns, float)
    lo_n, hi_n, nn = n_sweep
    spec = SsplSpecNumpy(Eg=prior.Eg, n_traps=prior.n_traps, Nc=constants["Nc"], Nv=constants["Nv"],
                         kT=KB_EV * T, n_sweep=np.logspace(np.log10(lo_n), np.log10(hi_n or constants["Nv"]), nn),
                         solver="auto", mode="qfls_vs_suns", x_model=np.log10(suns * G_per_sun))
    s = spec.solve(np.atleast_2d(theta))
    lg = np.log10(s["G"][0])
    o = np.argsort(lg)
    x = np.log10(suns * G_per_sun)
    sspl = pd.DataFrame({"intensity_suns": suns,
                         "qfls_eV": np.interp(x, lg[o], s["qfls"][0][o]),
                         "plqy": 10 ** np.interp(x, lg[o], np.log10(s["plqy"][0][o]))})
    return {"trpl": trpl, "sspl": sspl, "qfls_range": (lo, hi), "G_per_sun": G_per_sun}


def write_case(case: dict, theta, prior, out_dir, source: str = "") -> dict:
    """Write trpl/sspl CSVs, true_parameters.json and true_parameters.dat."""
    from .report import write_parameters_dat
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    p_trpl = out / "synthetic_trpl_qfls_tau.csv"
    p_sspl = out / "synthetic_sspl_suns_qfls_plqy.csv"
    case["trpl"].to_csv(p_trpl, index=False, float_format="%.10g")
    case["sspl"].to_csv(p_sspl, index=False, float_format="%.10g")
    phys = prior.to_physical(np.atleast_2d(theta)).iloc[0]
    truth = {"Et_eV": [float(phys[f"Et_{k}_eV"]) for k in range(1, prior.n_traps + 1)],
             "Nt_cm3": [float(phys[f"Nt_{k}_cm-3"]) for k in range(1, prior.n_traps + 1)],
             "tau_n_s": [float(phys[f"tau_n_{k}_s"]) for k in range(1, prior.n_traps + 1)],
             "tau_p_s": [float(phys[f"tau_p_{k}_s"]) for k in range(1, prior.n_traps + 1)],
             "krad_cm3s": float(phys["krad_cm3s"]), "theta": [float(v) for v in np.asarray(theta)],
             "source": source, "G_per_sun": case["G_per_sun"], "qfls_range": list(case["qfls_range"])}
    (out / "true_parameters.json").write_text(json.dumps(truth, indent=2))
    write_parameters_dat(out / "true_parameters.dat", prior, theta,
                         title=f"true parameters of the synthetic 1-trap case ({source})")
    return {"trpl": str(p_trpl), "sspl": str(p_sspl), "truth": truth}
