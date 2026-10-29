"""
dynamics.py — trap dynamics across the posterior (task T9, Figure 8)
====================================================================

Figure 8 of the paper discusses, for ONE best fit, that the net electron
capture of defect 2 changes sign during the decay (it starts to emit
electrons) and that these electrons are captured by defect 1.  Here the same
quantities are computed with the ODE for many posterior samples:

* net electron capture of trap k:  R_n,k = b_n n (N_t - n_t) - e_n n_t
  (= e_capture_k - e_emission_k of ``ode.simulate_trpl_full``);
* "sign change of defect 2": R_n,2 > +tol at some QFLS and < -tol at a LOWER
  QFLS (later in the decay) inside the analysed QFLS window;
* "transfer 2 -> 1": at some QFLS inside the window trap 2 net-emits
  electrons (R_n,2 < -tol) while trap 1 net-captures (R_n,1 > +tol).

``tol`` is a fraction of the total recombination rate at that point
(default 1e-3), so numerical noise does not count as a sign change.
Labels: trap 1 is the deeper trap (E_t1 < E_t2 by the prior ordering), as in
the original Table 4.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _analyse_one(theta, prior, npulse, constants, qfls_window, q_grid, rel_tol, T):
    import warnings
    warnings.filterwarnings("ignore")
    from .ode import simulate_trpl_full
    try:
        df = simulate_trpl_full(theta, prior, npulse, constants, T)
    except Exception as e:
        return {"ok": False, "reason": str(e)}
    q = df["qfls_eV"].to_numpy()
    sel = (q >= qfls_window[0]) & (q <= qfls_window[1])
    if sel.sum() < 3:
        return {"ok": False, "reason": "no ODE points in the QFLS window"}
    d = df[sel]
    rtot = d["R_rad_cm3s"].to_numpy() + sum(d[f"net_e_capture_{k}"].abs().to_numpy()
                                            for k in range(1, prior.n_traps + 1))
    r1 = d["net_e_capture_1"].to_numpy() / rtot
    r2 = d["net_e_capture_2"].to_numpy() / rtot
    qq = d["qfls_eV"].to_numpy()
    # time runs from high to low QFLS: sign change = positive earlier (higher QFLS), negative later
    pos = np.nonzero(r2 > rel_tol)[0]
    neg = np.nonzero(r2 < -rel_tol)[0]
    sign_change = bool(len(pos) and len(neg) and qq[pos].max() > qq[neg].min())
    transfer = bool(np.any((r2 < -rel_tol) & (r1 > rel_tol)))
    o = np.argsort(qq)
    return {"ok": True, "sign_change_2": sign_change, "transfer_2_to_1": transfer,
            "min_r2": float(r2.min()), "max_r2": float(r2.max()),
            "r1_grid": np.interp(q_grid, qq[o], r1[o], left=np.nan, right=np.nan),
            "r2_grid": np.interp(q_grid, qq[o], r2[o], left=np.nan, right=np.nan)}


def trap_dynamics(theta_samples, prior, npulse: float, constants: dict, qfls_window,
                  n_samples: int = 500, rel_tol: float = 1e-3, n_jobs: int = 16, seed: int = 0,
                  T: float = 300.0, theta_best=None) -> dict:
    """
    Run the analysis for ``n_samples`` posterior samples (and the best fit).
    Returns dict with the per-sample table, the fractions with sign change /
    transfer, and normalised net-capture curves on a QFLS grid for plotting.
    """
    from joblib import Parallel, delayed
    if prior.n_traps < 2:
        raise ValueError("trap dynamics (Figure 8) needs the 2-trap model")
    rng = np.random.default_rng(seed)
    th = np.asarray(theta_samples)
    th = th[rng.choice(len(th), size=min(n_samples, len(th)), replace=False)]
    q_grid = np.linspace(qfls_window[0], qfls_window[1], 200)
    res = Parallel(n_jobs=n_jobs)(delayed(_analyse_one)(t, prior, npulse, constants, qfls_window,
                                                        q_grid, rel_tol, T) for t in th)
    best = _analyse_one(theta_best, prior, npulse, constants, qfls_window, q_grid, rel_tol, T) \
        if theta_best is not None else None
    ok = [r for r in res if r["ok"]]
    tab = pd.DataFrame([{k: v for k, v in r.items() if not k.endswith("_grid")} for r in res])
    out = {"table": tab, "n_used": len(ok), "n_failed": len(res) - len(ok), "q_grid": q_grid,
           "frac_sign_change_2": float(np.mean([r["sign_change_2"] for r in ok])) if ok else np.nan,
           "frac_transfer_2_to_1": float(np.mean([r["transfer_2_to_1"] for r in ok])) if ok else np.nan,
           "frac_both": float(np.mean([r["sign_change_2"] and r["transfer_2_to_1"] for r in ok])) if ok else np.nan,
           "r1": np.array([r["r1_grid"] for r in ok]), "r2": np.array([r["r2_grid"] for r in ok]),
           "best": best, "rel_tol": rel_tol}
    return out


def plot_trap_dynamics(dyn: dict, max_curves: int = 200):
    """Net electron capture of both traps (normalised to the total rate) vs QFLS."""
    import matplotlib.pyplot as plt
    q = dyn["q_grid"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, key, name in ((axes[0], "r1", "trap 1 (deeper)"), (axes[1], "r2", "trap 2 (shallower)")):
        for c in dyn[key][:max_curves]:
            ax.plot(q, c, color="C0", alpha=0.08, lw=0.8)
        if dyn["best"] is not None and dyn["best"]["ok"]:
            ax.plot(q, dyn["best"][f"{key}_grid"], color="C3", lw=2, label="best fit")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xlabel(r"$\Delta E_\mathrm{F}$ [eV]")
        ax.set_title(name)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(r"net electron capture / total rate")
    axes[0].set_yscale("symlog", linthresh=1e-3)
    axes[1].legend(fontsize=8)
    fig.suptitle(f"posterior samples: sign change of trap 2 in {100 * dyn['frac_sign_change_2']:.0f} %, "
                 f"transfer 2→1 in {100 * dyn['frac_transfer_2_to_1']:.0f} % (n = {dyn['n_used']})")
    fig.tight_layout()
    return fig
