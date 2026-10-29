"""
start_points.py — physics-informed start values
===============================================

Wraps the estimator ``preprocessing/start_point_estimation_module.py``
(curvature analysis of tau_diff vs QFLS, Section 5.2.1 of the paper) and
converts its output into a start vector theta for the optimiser.

For each trap one call of ``start_point_estimator`` is made with its own
settings (trap type, QFLS window, dEt, ...), exactly as in the Figure
notebooks; k_rad is taken from the first trap (the estimator always returns
1e-11 cm^3/s).  The resulting vector is projected into the prior support (and
the traps relabelled so that E_t1 < E_t2).

Note on the legacy estimator (not changed here): for ``photodoping=False``
the default ``Nt_n1_factor=10`` still gives N_t = 10 n_1, i.e. a photodoping
density, while the tau formula assumes N_t < n_1.  Pass e.g.
``Nt_n1_factor=0.1`` together with ``photodoping=False``.

Assumed values (15/10/2026): a trap setting may fix E_t and/or tau_n by
assumption instead of taking the estimator's choices (E_t = E_g - dEt,
tau_n = tau_p):

    "Et_eV":  assumed trap energy; passed to the estimator as dEt = E_g - E_t
              (shallow trap: n_1, N_t = Nt_n1_factor n_1 and tau_p follow from
              it and from tau_0 of the data via the analytic shallow-regime
              formula tau_p = tau_0 / (a - b tau_0)); deep trap: replaces E_g/2.
    "taun_s": assumed tau_n; replaces tau_n = tau_p.  tau_p is NOT changed.
              "first_tau_diff" (23/10/2026, Robin; paper procedure): tau_n = tau_diff of the first
              time point, i.e. the point with the highest QFLS of the cleaned tr-PL.

The estimator evaluates b = k_rad n_1 with its own k_rad = 1e-11 cm^3/s; with
k_rad fixed at 1e-10 this changes tau_p by < 0.03 decades (b tau_0 << a).
"""
from __future__ import annotations

import contextlib
import io

import numpy as np
import pandas as pd

from . import add_legacy_paths


@contextlib.contextmanager
def _no_show(show: bool):
    """Suppress plt.show() inside the legacy estimator (it always plots)."""
    import matplotlib.pyplot as plt
    if show:
        yield
        return
    orig = plt.show
    plt.show = lambda *a, **k: None
    try:
        yield
    finally:
        plt.show = orig
        plt.close("all")


def estimate_start_point(df_trpl: pd.DataFrame, Eg: float, prior, trap_settings: list,
                         show_plots: bool = False) -> dict:
    """
    Physics-informed start vector.

    Parameters
    ----------
    df_trpl : DataFrame with columns qfls, tau_diff (cleaned)
    Eg : float
    prior : prior.Prior (defines the support the start is projected into)
    trap_settings : list of dicts, one per trap, with keys
        trap_type    'shallow' | 'deep'
        qfls_window  [qfls_min, qfls_max] or None (automatic regime detection
                     via ``determine_trap_regime``)
        dEt, photodoping, Nt_n1_factor, Ntrap   passed to the estimator
        Et_eV, taun_s  optional assumed E_t / tau_n (see the module docstring)
    show_plots : show the legacy diagnostic plot of each estimate

    Returns
    -------
    dict with theta (D,), theta_raw (before projection), table (DataFrame of
    the physical start values per trap) and the regime masks.
    """
    add_legacy_paths()
    import start_point_estimation_module as spe

    if len(trap_settings) != prior.n_traps:
        raise ValueError(f"Need {prior.n_traps} trap settings, got {len(trap_settings)}")

    estimates, masks = [], []
    mask_deep, mask_shallow = spe.determine_trap_regime(df_trpl)
    for s in trap_settings:
        ttype = s.get("trap_type", "shallow")
        if s.get("qfls_window") is not None:
            lo, hi = s["qfls_window"]
            mask = ((df_trpl["qfls"] >= lo) & (df_trpl["qfls"] <= hi)).to_numpy()
        else:
            mask = mask_shallow if ttype == "shallow" else mask_deep
        if mask.sum() < 2:
            raise ValueError(f"Trap regime '{ttype}' has < 2 points; set 'qfls_window'.")
        kwargs = {k: s[k] for k in ("photodoping", "dEt", "Nt_n1_factor", "Ntrap") if k in s}
        assumed = []
        if s.get("Et_eV") is not None and ttype == "shallow":
            kwargs["dEt"] = Eg - float(s["Et_eV"])          # n1, N_t and tau_p follow from the assumed E_t
            assumed.append("Et")
        sink = io.StringIO()
        with _no_show(show_plots), (contextlib.nullcontext() if show_plots
                                    else contextlib.redirect_stdout(sink)):
            krad, Et, Nt, taun, taup = spe.start_point_estimator(
                df_trpl, mask, Eg, trap_type=ttype, **kwargs)
        if s.get("Et_eV") is not None and ttype == "deep":
            Et = float(s["Et_eV"])
            assumed.append("Et")
        if s.get("krad_cm3s") is not None and ttype == "shallow":
            # 20/10/2026 (Robin): tau_p from tau_0 with the given k_rad (e.g. "prior" = the fixed k_rad,ext of the prior)
            # instead of the estimator's built-in 1e-11.  tau_0 = tau_p a / (1 + k_rad tau_p n1), a = sqrt(n1 N_t)/n_i
            # (photodoping) or n1/n_i: tau_0 is recovered from the estimator's tau_p with its own k_rad and constants,
            # then tau_p = tau_0 / (a - k_rad,new n1 tau_0)
            kr = s["krad_cm3s"]
            if kr == "prior":
                j = list(prior.names).index("log10_krad")
                kr = 10.0 ** (0.5 * (prior.lo[j] + prior.hi[j]))
            kr = float(kr)
            ni = np.sqrt(spe.NC * spe.NV) * np.exp(-Eg / (2.0 * spe.VT))
            n1 = spe.NC * np.exp((float(Et) - Eg) / spe.VT)
            a = np.sqrt(n1 * float(Nt)) / ni if kwargs.get("photodoping", True) else n1 / ni
            tau0 = a * float(taup) / (1.0 + float(krad) * n1 * float(taup))
            taup = tau0 / (a - kr * n1 * tau0)
            taun = taup
            krad = kr
            assumed.append(f"krad={kr:.4g}")
        if s.get("taun_s") == "first_tau_diff":
            taun = float(df_trpl["tau_diff"].to_numpy(float)[np.argmax(df_trpl["qfls"].to_numpy(float))])
            assumed.append("taun=first tau_diff")
        elif s.get("taun_s") is not None:
            taun = float(s["taun_s"])
            assumed.append("taun")
        estimates.append(dict(krad=float(krad), Et_eV=float(Et), Nt_cm3=float(Nt),
                              taun_s=float(taun), taup_s=float(taup), trap_type=ttype,
                              n_points_regime=int(mask.sum()), assumed=",".join(assumed) or "none"))
        masks.append(mask)

    table = pd.DataFrame(estimates)
    table.index = [f"trap {k + 1}" for k in range(len(table))]
    theta_raw = prior.from_physical(table.Et_eV, table.Nt_cm3, table.taun_s, table.taup_s,
                                    krad=table.krad.iloc[0])
    if not np.all(np.isfinite(theta_raw)):
        raise ValueError(f"Start-point estimator returned non-finite values:\n{table}")
    theta = prior.project(theta_raw)[0]
    moved = np.abs(theta - theta_raw) > 1e-9
    if moved.any():
        print("Start point projected into the prior support for: "
              + ", ".join(np.array(prior.names)[moved]))
    return {"theta": theta, "theta_raw": theta_raw, "table": table, "masks": masks,
            "mask_deep_auto": mask_deep, "mask_shallow_auto": mask_shallow}


def explicit_start_point(prior, spec: dict) -> dict:
    """
    Start point given explicitly (e.g. the physics-informed start values published in the paper), same return
    structure as ``estimate_start_point``.

    spec: {"Et_eV": [..], "Nt_cm3": [..], "tau_n_s": [..], "tau_p_s": [..], "krad_cm3s": float, "source": str}
    (one value per trap).  Fixed parameters of the prior (e.g. k_rad) are set to their fixed value by the projection
    into the support; n_1 = N_C exp((E_t - E_g)/kT) is listed for checking against the source.
    """
    from .sspl_tf import KB_EV
    Et = np.atleast_1d(np.asarray(spec["Et_eV"], float))
    Nt = np.atleast_1d(np.asarray(spec["Nt_cm3"], float))
    tn = np.atleast_1d(np.asarray(spec["tau_n_s"], float))
    tp = np.atleast_1d(np.asarray(spec["tau_p_s"], float))
    krad = float(spec.get("krad_cm3s", 1e-11))
    if not (len(Et) == len(Nt) == len(tn) == len(tp) == prior.n_traps):
        raise ValueError(f"explicit start point: one value per trap needed ({prior.n_traps} traps)")
    kT = KB_EV * prior.T
    table = pd.DataFrame({"krad": krad, "Et_eV": Et, "Nt_cm3": Nt, "taun_s": tn, "taup_s": tp,
                          "n1_cm3": prior.Nc * np.exp((Et - prior.Eg) / kT), "trap_type": "explicit",
                          "assumed": spec.get("source", "explicit")})
    table.index = [f"trap {k + 1}" for k in range(len(table))]
    theta_raw = prior.from_physical(Et, Nt, tn, tp, krad=krad)
    theta = prior.project(theta_raw)[0]
    moved = np.abs(theta - theta_raw) > 1e-9
    if moved.any():
        print("Start point projected into the prior support for: " + ", ".join(np.array(prior.names)[moved]))
    return {"theta": theta, "theta_raw": theta_raw, "table": table, "masks": [], "mask_deep_auto": None,
            "mask_shallow_auto": None}
