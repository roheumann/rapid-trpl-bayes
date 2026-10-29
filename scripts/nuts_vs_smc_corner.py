"""
nuts_vs_smc_corner.py — NUTS and SMC posteriors of one run in one corner plot, for Origin
==========================================================================================
26/10/2026.  The SMC-vs-NUTS check of the 1-trap synthetic case (B2nf, k_rad fixed, run of 15/10/2026):
NUTS reference = <run>/posterior_samples_nuts.csv (64,000 draws), SMC = <run>/<smc>/posterior_samples_smc.csv
(16,384 equally weighted draws).  Truth from the run config, best fit from 6_best_fit.json.

    python scripts/nuts_vs_smc_corner.py results/1trap_synthetic_paper_B2nf_kradfixed_joint [--smc smc_B2nf]

Writes <run>/<smc>/corner_nuts_vs_smc/corner_nuts_vs_smc.h5 (+ _preview.png), the same layout as
corner_nn_vs_ode.h5 (nn_vs_ode_corner.write_corner): NUTS filled in blue, SMC as wine-red lines.
Origin: corner-plot/origin/build_corner_nuts_vs_smc.py.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from nn_vs_ode_corner import COLORS, run_prior, write_corner   # noqa: E402

METHODS = ("nuts", "smc")
METHOD_LABEL = {"nuts": "NUTS", "smc": "SMC"}
NUTS_SMC_COLORS = {"nuts": COLORS["nn"], "smc": COLORS["ode"]}   # same look: reference filled blue, other wine red


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="run folder with posterior_samples_nuts.csv (e.g. results/1trap_..._joint)")
    ap.add_argument("--smc", default="smc_B2nf", help="subfolder of the SMC run (posterior_samples_smc.csv)")
    ap.add_argument("--grid", type=int, default=240, help="resolution of the filled regions")
    ap.add_argument("--gpu", default="")
    a = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import pandas as pd

    run = Path(a.run)
    cfg, prior = run_prior(run)
    names = list(prior.names)
    src = {"nuts": run / "posterior_samples_nuts.csv", "smc": run / a.smc / "posterior_samples_smc.csv"}
    theta = {m: pd.read_csv(p)[names].to_numpy(float) for m, p in src.items()}
    weights = {m: np.full(len(t), 1 / len(t)) for m, t in theta.items()}
    for m in METHODS:
        print(f"{METHOD_LABEL[m]}: {len(theta[m])} draws from {src[m]}")
    write_corner(run, cfg, prior, theta, weights, run / a.smc / "corner_nuts_vs_smc" / "corner_nuts_vs_smc.h5", a.grid,
                 methods=METHODS, labels=METHOD_LABEL, colors=NUTS_SMC_COLORS, fill_mode="nuts",
                 extra_attrs={"smc_dir": a.smc})


if __name__ == "__main__":
    main()
