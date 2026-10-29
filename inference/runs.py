"""
runs.py — likelihood of a finished main.ipynb run for the follow-up notebooks (smc.ipynb, cylinder.ipynb)
========================================================================================================
15/10/2026 (revision).

The follow-up notebooks rebuild the likelihood of a run folder ``results/<run>_<mode>/`` from the data the run
actually used (``trpl_used.csv``, ``sspl_used_<ss-PL mode>.csv``) and its config snapshot (``settings.json``), for
every mode of main.ipynb:

    'trpl'   tr-PL term (NN surrogate)
    'sspl'   ss-PL term only (exact ss-PL model, TF; no surrogate)
    'joint'  both
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def sspl_term(cfg: dict, run_dir, n_traps: int, constants: dict, sigma: float):
    """likelihood.SsplTerm with the ss-PL points of the run and the ss-PL model of the config."""
    from . import data, likelihood, sspl_tf
    ss = cfg["sspl"]
    mode = sspl_tf.canonical_mode(ss["mode"])
    xy = pd.read_csv(Path(run_dir) / f"sspl_used_{mode}.csv")
    Eg, T = cfg["physics"]["Eg_eV"], cfg["physics"]["T_K"]
    g_sun = data.sspl_g_per_sun(cfg, constants)[0] if mode.endswith("_suns") else None
    lo, hi, n = ss["n_sweep"]
    model = sspl_tf.SsplModel(Eg, n_traps, constants["Nc"], constants["Nv"], T=T, n_sweep=(lo, hi, n))
    obs = sspl_tf.SsplObservation(model, mode, xy["x"].values, xy["y"].values, G_per_sun=g_sun)
    return likelihood.SsplTerm(obs, sigma=float(sigma), offset_sigma=ss.get("offset_sigma_eV"))


def make_log_post(cfg: dict, run_dir, art, prior, mode: str, sigma_trpl: float | None,
                  sigma_sspl: float | None, batch_size: int | None = None):
    """
    LogPosterior of `mode` for the data of a main.ipynb run.  Returns (log_post, tr-PL surrogate or None).
    sigma_trpl / sigma_sspl: sigmas of the terms (None only for a term the mode does not use).
    """
    from . import likelihood, surrogate
    run_dir = Path(run_dir)
    bs = batch_size or cfg["hardware"]["batch_size"]
    terms, sur, ss = [], None, None
    if mode in ("trpl", "joint"):
        trpl = pd.read_csv(run_dir / "trpl_used.csv")
        sur = surrogate.make_trpl_surrogate(art, cfg["physics"]["npulse_cm3"], cfg["physics"]["Eg_eV"],
                                            trpl.qfls.values, interp_space=cfg["trpl"]["interp_space"])
        terms = [likelihood.make_trpl_term(sur, trpl.qfls.values, trpl.tau_diff.values, sigma=sigma_trpl)]
    if mode in ("sspl", "joint"):
        ss = sspl_term(cfg, run_dir, prior.n_traps, art.constants, sigma_sspl)
    return likelihood.LogPosterior(prior, terms, ss, mode, batch_size=bs), sur


def run_sigmas(run_dir) -> dict:
    """sigma_eff of a run (the sigmas of its NUTS / emcee / grid): {'trpl': float | None, 'sspl': float | None}."""
    import json
    p = Path(run_dir) / "sampling_info.json"           # written with the NUTS results (main.ipynb 7e)
    if not p.exists():                                  # RUN_NUTS = False: the same sigmas are in 6_best_fit.json
        p = Path(run_dir) / "6_best_fit.json"
    s = json.loads(p.read_text())["sigmas"]
    return {"trpl": s.get("trpl"), "sspl": s.get("sspl")}
