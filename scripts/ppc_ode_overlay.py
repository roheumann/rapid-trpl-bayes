"""
ppc_ode_overlay.py — NN inference vs inference with the rate equations in ONE posterior-predictive figure (+ Origin CSV)
=====================================================================================================================
19/10/2026 (Robin).  Input: a ppc.ipynb output folder that was run with EXTRA_SAMPLES={"ODE SMC": ...} and N_ODE > 0.

    python scripts/ppc_ode_overlay.py results/<run>_<mode>/ppc

Bands of the model curve f (parameter uncertainty only, 68 / 95 % pointwise) at the data points:
  NN SMC, NN curves            posterior of the network inference, predicted by the network
  NN SMC, ODE curves           the SAME draws re-evaluated with the rate equations (legacy ODE)
  ODE SMC, ODE curves          posterior of the inference with the rate equations ("ODE inference bounds")
ss-PL (modes sspl / joint): exact model for both posteriors (no surrogate), NN SMC vs ODE SMC.
Output in the ppc folder: 5_trpl_bands_nn_vs_ode.csv, 5_sspl_bands_nn_vs_ode.csv (linear units: tau_diff in s,
PLQY), figures/5_ppc_nn_vs_ode_inference.png/pdf.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

Q = ("q2.5", "q16", "q50", "q84", "q97.5")
SETS = (("NN SMC, NN", "2_bands_nn_SMC_per_point.csv", "C0"),
        ("NN SMC, ODE re-evaluated", "3_bands_ode_SMC.csv", "C1"),
        ("ODE SMC, ODE", "3_bands_ode_ODE SMC.csv", "k"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("ppc_dir")
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = Path(args.ppc_dir)
    s = json.loads((d / "summary.json").read_text())
    B = {lab: pd.read_csv(d / f) for lab, f, _ in SETS if (d / f).exists()}
    first = next(iter(B.values()))
    q, y = first.qfls.values, first.y_obs.values
    tab = {"qfls_eV": q, "tau_diff_data_s": 10 ** y}
    for lab, b in B.items():
        assert np.allclose(b.qfls.values, q)
        key = lab.replace(", ", "_").replace(" ", "_")
        for c in Q:
            tab[f"{key}_f_{c}_s"] = 10 ** b[f"f_{c}"].values
    pd.DataFrame(tab).to_csv(d / "5_trpl_bands_nn_vs_ode.csv", index=False, float_format="%.8g")

    SS = {}
    for lab, f in (("NN SMC", "2b_bands_sspl_SMC_per_point.csv"), ("ODE SMC", "2b_bands_sspl_ODE SMC_per_point.csv")):
        if (d / f).exists():
            SS[lab] = pd.read_csv(d / f)
    if SS:
        b0 = next(iter(SS.values()))
        t2 = {"qfls_eV": b0.x.values, "plqy_data": 10 ** b0.y_obs.values}
        for lab, b in SS.items():
            for c in Q:
                t2[f"{lab.replace(' ', '_')}_f_{c}"] = 10 ** b[f"f_{c}"].values
        pd.DataFrame(t2).to_csv(d / "5_sspl_bands_nn_vs_ode.csv", index=False, float_format="%.8g")

    ncol = 2 if SS else 1
    fig, axs = plt.subplots(2, ncol, figsize=(6.2 * ncol, 6.2), sharex="col", gridspec_kw={"height_ratios": [2, 1]},
                            squeeze=False)
    ref = B.get("ODE SMC, ODE", first)["f_q50"].values
    for lab, _, col in SETS:
        if lab not in B:
            continue
        b = B[lab]
        if col == "k":      # ODE inference: lines
            for c, ls in (("q2.5", "-"), ("q97.5", "-"), ("q16", "--"), ("q84", "--")):
                axs[0, 0].semilogy(q, 10 ** b[f"f_{c}"], ls, color="k", lw=0.9, label=f"{lab} 95 % / 68 %" if c == "q2.5" else None)
                axs[1, 0].plot(q, b[f"f_{c}"] - ref, ls, color="k", lw=0.9)
        else:
            axs[0, 0].fill_between(q, 10 ** b["f_q2.5"], 10 ** b["f_q97.5"], color=col, alpha=0.25, lw=0, label=f"{lab} 95 %")
            axs[0, 0].fill_between(q, 10 ** b["f_q16"], 10 ** b["f_q84"], color=col, alpha=0.35, lw=0)
            axs[1, 0].fill_between(q, b["f_q2.5"] - ref, b["f_q97.5"] - ref, color=col, alpha=0.25, lw=0)
            axs[1, 0].fill_between(q, b["f_q16"] - ref, b["f_q84"] - ref, color=col, alpha=0.35, lw=0)
    axs[0, 0].semilogy(q, 10 ** y, "o", ms=2.5, color="C3", label="data")
    axs[1, 0].plot(q, y - ref, "o", ms=2.5, color="C3"); axs[1, 0].axhline(0, color="0.4", lw=0.6)
    axs[0, 0].set_ylabel(r"$\tau_\mathrm{diff}$ (s)"); axs[1, 0].set_ylabel("− median ODE SMC (dec.)")
    axs[1, 0].set_xlabel("QFLS (eV)"); axs[0, 0].legend(fontsize=7); axs[0, 0].set_title("tr-PL", fontsize=9)
    if SS:
        b0 = next(iter(SS.values()))
        x, yo = b0.x.values, b0.y_obs.values
        rs = SS.get("ODE SMC", b0)["f_q50"].values
        for (lab, b), col in zip(SS.items(), ("C0", "k")):
            if col == "k":
                for c, ls in (("q2.5", "-"), ("q97.5", "-"), ("q16", "--"), ("q84", "--")):
                    axs[0, 1].semilogy(x, 10 ** b[f"f_{c}"], ls, color="k", lw=0.9, label=f"{lab} 95 % / 68 %" if c == "q2.5" else None)
                    axs[1, 1].plot(x, b[f"f_{c}"] - rs, ls, color="k", lw=0.9)
            else:
                axs[0, 1].fill_between(x, 10 ** b["f_q2.5"], 10 ** b["f_q97.5"], color=col, alpha=0.25, lw=0, label=f"{lab} 95 %")
                axs[0, 1].fill_between(x, 10 ** b["f_q16"], 10 ** b["f_q84"], color=col, alpha=0.35, lw=0)
                axs[1, 1].fill_between(x, b["f_q2.5"] - rs, b["f_q97.5"] - rs, color=col, alpha=0.25, lw=0)
                axs[1, 1].fill_between(x, b["f_q16"] - rs, b["f_q84"] - rs, color=col, alpha=0.35, lw=0)
        axs[0, 1].semilogy(x, 10 ** yo, "o", ms=2.5, color="C3", label="data")
        axs[1, 1].plot(x, yo - rs, "o", ms=2.5, color="C3"); axs[1, 1].axhline(0, color="0.4", lw=0.6)
        axs[0, 1].set_ylabel("PLQY"); axs[1, 1].set_ylabel("− median ODE SMC (dec.)"); axs[1, 1].set_xlabel("QFLS (eV)")
        axs[0, 1].legend(fontsize=7); axs[0, 1].set_title("ss-PL (exact model for both posteriors)", fontsize=9)
    ov = {r["sampler"]: r for r in s.get("ode_vs_nn") or []}
    txt = "; ".join(f"{k}: RMS(NN − ODE) {v['rms_nn_minus_ode']:.3f} dec, p_mse ODE {v['p_mse_ode']:.2f}"
                    for k, v in ov.items() if k in ("SMC", "ODE SMC"))
    fig.suptitle(f"{Path(s['run_in']).name}: posterior predictive, network vs rate equations (bands of f, σ per point)\n{txt}",
                 fontsize=9)
    fig.tight_layout()
    (d / "figures").mkdir(exist_ok=True)
    fig.savefig(d / "figures" / "5_ppc_nn_vs_ode_inference.png", dpi=200)
    fig.savefig(d / "figures" / "5_ppc_nn_vs_ode_inference.pdf")
    print("written:", d / "5_trpl_bands_nn_vs_ode.csv", "and figures/5_ppc_nn_vs_ode_inference.png")


if __name__ == "__main__":
    main()
