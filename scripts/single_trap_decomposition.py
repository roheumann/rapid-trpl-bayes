"""
single_trap_decomposition.py — ODE of every trap of a best fit on its own (1 trap) and their parallel connection
================================================================================================================
22/10/2026 (Robin).  Input: a finished main.ipynb run (6_best_fit.json, physical best fit with K traps).

    python scripts/single_trap_decomposition.py results/2trap_85-15_film/2trap_85-15_film_fit_joint

Sets, all with the legacy ODE (inference/ode.py, training post-processing, nominal QFLS axis as for the data) and the
same k_rad, E_g, n_pulse, T:
  all_traps   the best fit as fitted (K traps together)
  trap_k      trap k alone (1-trap ODE with N_t, E_t, tau_n, tau_p of trap k)
  parallel    1/tau_par(QFLS) = sum_k 1/tau_trap_k(QFLS), on the QFLS range covered by every single-trap curve
              (no extrapolation: NaN outside).  Each single-trap tau_diff contains the radiative recombination, so
              it enters the sum K times.
Output in <run>/single_trap_decomposition/:
  parameters.csv                    physical parameters of every set
  ode_full_<set>.csv                full ODE output (time, PL, QFLS physical + nominal, tau_diff, n, p, n_t, rates)
  tau_diff_vs_qfls.csv              common QFLS grid: log10 tau_diff and tau_diff of every set + parallel
  tau_diff_at_data.csv              the same at the data QFLS points, with the data
  summary.json / summary.md         RMSE (log10 tau_diff) of every set against the data
  figures/tau_diff_vs_qfls.png/pdf
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class FixedParams:
    """Minimal prior stand-in for inference.ode: to_physical always returns the given physical parameter set."""

    def __init__(self, phys: dict, Eg: float):
        self.phys, self.Eg = phys, Eg
        self.n_traps = sum(1 for k in phys if k.startswith("Nt_"))

    def to_physical(self, theta):
        return pd.DataFrame([self.phys])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("run")
    ap.add_argument("--n-grid", type=int, default=400)
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.interpolate import PchipInterpolator
    from inference import ode, surrogate

    run = Path(args.run)
    cfg = json.loads((run / "settings.json").read_text())["config"]
    Eg, npulse, T = cfg["physics"]["Eg_eV"], cfg["physics"]["npulse_cm3"], cfg["physics"]["T_K"]
    constants = surrogate.load_network(cfg["nn"]["dir"], cfg["nn"].get("stem")).constants
    best = json.loads((run / "6_best_fit.json").read_text())["physical"]
    K = sum(1 for k in best if k.startswith("Nt_"))
    sets = {"all_traps": dict(best)}
    for k in range(1, K + 1):
        sets[f"trap_{k}"] = {"krad_cm3s": best["krad_cm3s"], "Et_1_eV": best[f"Et_{k}_eV"], "Nt_1_cm-3": best[f"Nt_{k}_cm-3"],
                             "tau_n_1_s": best[f"tau_n_{k}_s"], "tau_p_1_s": best[f"tau_p_{k}_s"]}
    out = run / "single_trap_decomposition"
    (out / "figures").mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"set": s, **p} for s, p in sets.items()]).to_csv(out / "parameters.csv", index=False, float_format="%.8g")

    data = pd.read_csv(run / "trpl_used.csv")
    curves = {}
    for name, phys in sets.items():
        pr = FixedParams(phys, Eg)
        ode.simulate_trpl_full(None, pr, npulse, constants, T).to_csv(out / f"ode_full_{name}.csv", index=False,
                                                                       float_format="%.8g")
        curves[name] = ode.simulate_trpl(None, pr, npulse, constants, T)          # nominal QFLS axis, log10 tau_diff
        print(f"{name}: {len(curves[name][0])} points, QFLS {curves[name][0].min():.4f} ... {curves[name][0].max():.4f} eV")

    singles = [f"trap_{k}" for k in range(1, K + 1)]
    q_lo = max(curves[s][0].min() for s in singles)
    q_hi = min(curves[s][0].max() for s in singles)
    lo_all = min(c[0].min() for c in curves.values())
    hi_all = max(c[0].max() for c in curves.values())
    grid = np.linspace(lo_all, hi_all, args.n_grid)

    def at(q):
        cols = {s: PchipInterpolator(*curves[s], extrapolate=False)(q) for s in curves}
        inside = (q >= q_lo) & (q <= q_hi)
        inv = sum(10.0 ** -cols[s] for s in singles)
        cols["parallel"] = np.where(inside, -np.log10(inv), np.nan)
        return cols

    g = at(grid)
    tab = pd.DataFrame({"qfls_eV": grid} | {f"log10_tau_{s}": v for s, v in g.items()}
                       | {f"tau_{s}_s": 10.0 ** v for s, v in g.items()})
    tab.to_csv(out / "tau_diff_vs_qfls.csv", index=False, float_format="%.8g")
    d = at(data.qfls.values)
    y = np.log10(data.tau_diff.values)
    td = pd.DataFrame({"qfls_eV": data.qfls.values, "tau_data_s": data.tau_diff.values}
                      | {f"tau_{s}_s": 10.0 ** v for s, v in d.items()})
    td.to_csv(out / "tau_diff_at_data.csv", index=False, float_format="%.8g")

    rmse = {}
    for s, v in d.items():
        ok = np.isfinite(v)
        rmse[s] = {"rmse_log10_tau": float(np.sqrt(np.mean((v[ok] - y[ok]) ** 2))) if ok.any() else None,
                   "points_covered": f"{int(ok.sum())}/{len(y)}"}
    summ = {"run": str(run), "parameters": sets, "parallel_qfls_range_eV": [float(q_lo), float(q_hi)],
            "rmse_vs_data": rmse,
            "note": "parallel: 1/tau = sum_k 1/tau_trap_k; each single-trap tau_diff includes radiative recombination"}
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    md = [f"# Single-trap ODE decomposition — `{run}`", "",
          "Legacy ODE, nominal QFLS axis (as the data); same k_rad, E_g, n_pulse. parallel: 1/tau_par = "
          + " + ".join(f"1/tau_{s}" for s in singles) + f", only on {q_lo:.4f} … {q_hi:.4f} eV (covered by every trap).", "",
          "| set | E_t (eV) | N_t (cm⁻³) | τ_n (s) | τ_p (s) | RMSE vs data (dec) | points |", "|---|---|---|---|---|---|---|"]
    for s in list(curves) + ["parallel"]:
        r = rmse[s]
        if s in singles:
            p = sets[s]
            par = f"{p['Et_1_eV']:.4f} | {p['Nt_1_cm-3']:.3e} | {p['tau_n_1_s']:.3e} | {p['tau_p_1_s']:.3e}"
        else:
            par = "– | – | – | –"
        md.append(f"| {s} | {par} | {r['rmse_log10_tau']:.4f} | {r['points_covered']} |")
    (out / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.semilogy(data.qfls, data.tau_diff, "ko", mfc="none", ms=4, label="data")
    style = {"all_traps": ("k", "-", f"{K} traps (best fit)"), "parallel": ("C3", "--", "parallel: 1/τ = Σ 1/τ_k")}
    for s in curves:
        c, ls, lab = style.get(s, (f"C{singles.index(s)}" if s in singles else "k", "-", s.replace("_", " ") + " alone"))
        ax.semilogy(*[curves[s][0], 10.0 ** curves[s][1]], color=c, ls=ls, lw=1.5, label=lab)
    ax.semilogy(grid, 10.0 ** g["parallel"], color="C3", ls="--", lw=2, label=style["parallel"][2])
    ax.set_xlabel("QFLS (eV)"); ax.set_ylabel("τ_diff (s)"); ax.legend(fontsize=8, frameon=False)
    ax.set_title(run.name + ": single traps (ODE) and their parallel connection", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "figures" / "tau_diff_vs_qfls.png", dpi=150); fig.savefig(out / "figures" / "tau_diff_vs_qfls.pdf")


if __name__ == "__main__":
    main()
