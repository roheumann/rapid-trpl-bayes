"""
alpha_spectrum.py — absorption coefficient alpha(E) from the measured absorptance (Robin, 2026-10-27)
=====================================================================================================

Reflection-corrected alpha, as chosen by Robin for the 85:15 film on 2026-10-19 (results/2trap_85-15/
sigma_model/chosen.json):  alpha = -ln(1 - A/A_max)/d,  A_max = max A above 1.8 eV (the plateau = 1 - R).
The plain alpha = -ln(1 - A)/d is written next to it for comparison.

usage (from rapid-trpl-bayes/):
    python model-error-analysis/alpha_spectrum.py --d-nm 450 --wavelength-nm 514 --out <dir>
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
from importlib import import_module

sm = import_module("sigma_model")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--absorptance", default=str(REPO / "data/85-15_film/absorptance.dat"))
    ap.add_argument("--d-nm", type=float, required=True, help="thickness of the film the absorptance was measured on")
    ap.add_argument("--wavelength-nm", type=float, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    d_cm = a.d_nm * 1e-7

    df = sm.read_absorptance(a.absorptance)
    a_max = float(df.loc[df.E_eV >= 1.8, "A"].max())
    with np.errstate(all="ignore"):
        df["alpha_plain_cm"] = -np.log1p(-np.clip(df.A.values, None, 1 - 1e-12)) / d_cm
        df["alpha_cm"] = -np.log1p(-np.clip(df.A.values / a_max, None, 1 - 1e-12)) / d_cm
    df.to_csv(out / "alpha_vs_E.csv", index=False)

    E = sm.HC_EV_NM / a.wavelength_nm
    look = {}
    for col in ("alpha_cm", "alpha_plain_cm"):
        ok = np.isfinite(df[col]) & (df[col] > 0)
        look[col] = float(10 ** np.interp(E, df.E_eV[ok], np.log10(df[col][ok])))
    info = {"date": time.strftime("%Y-%m-%d %H:%M"), "absorptance_file": a.absorptance, "d_nm": a.d_nm,
            "formula": "alpha = -ln(1 - A/A_max)/d (reflection-corrected); plain = -ln(1 - A)/d",
            "A_max_above_1.8eV": a_max, "wavelength_nm": a.wavelength_nm, "E_eV": E,
            "A_at_wavelength": float(np.interp(E, df.E_eV, df.A)),
            "alpha_cm": look["alpha_cm"], "alpha_plain_cm": look["alpha_plain_cm"]}
    (out / "alpha_lookup.json").write_text(json.dumps(info, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].plot(df.E_eV, df.A, "k-"); ax[0].axhline(a_max, color="0.5", ls=":", label=f"A_max {a_max:.3f}")
    ax[0].axvline(E, color="C3", ls="--", label=f"{a.wavelength_nm:.0f} nm")
    ax[0].set_xlabel("E (eV)"); ax[0].set_ylabel("absorptance A"); ax[0].legend(fontsize=8)
    ax[1].semilogy(df.E_eV, df.alpha_cm, "k-", label=r"$-\ln(1-A/A_{max})/d$")
    ax[1].semilogy(df.E_eV, df.alpha_plain_cm, "-", color="0.6", lw=0.8, label=r"$-\ln(1-A)/d$")
    ax[1].axvline(E, color="C3", ls="--"); ax[1].axhline(look["alpha_cm"], color="C3", lw=0.8)
    ax[1].set_xlabel("E (eV)"); ax[1].set_ylabel(r"$\alpha$ (cm$^{-1}$)"); ax[1].legend(fontsize=8)
    ax[1].set_title(f"alpha({a.wavelength_nm:.0f} nm) = {look['alpha_cm']:.4g} /cm (d = {a.d_nm:.0f} nm)", fontsize=9)
    fig.tight_layout(); fig.savefig(out / "alpha_vs_E.png", dpi=150); plt.close(fig)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
