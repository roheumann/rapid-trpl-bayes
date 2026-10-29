"""
inference — full-dimensional Bayesian inference for the SRH trap model
======================================================================

This package replaces the pairwise-sliced grid posterior of the original
submission with full-dimensional posterior sampling (see
REVISION_PLAN.md, Sections 2–3).  Every analysis — best fit, posterior
sampling, profile likelihood, sensitivity tests — evaluates ONE log-posterior
(``likelihood.LogPosterior``), so the best fit is by construction the maximum
of the same likelihood that the samplers explore.

Modules
-------
benchmark     timeit-based timing helpers and evaluation counters
surrogate     NN surrogate: theta (B, D) -> log10 tau_diff at the measured QFLS
sspl_tf       steady-state PL forward model in TensorFlow (float64, batched)
prior         uniform prior on the NN training box (+ E_t ordering) and the
              bijection to unconstrained coordinates used by optimiser/samplers
likelihood    tr-PL / ss-PL / joint log-likelihood and log-posterior
data          loading and preprocessing of tr-PL and ss-PL data
start_points  physics-informed start values (wraps the legacy estimator)
optimize      multi-start CMA-ES (gradient-free) and batched L-BFGS
samplers      TFP NUTS (primary) and emcee (cross-check) -> ArviZ
smc           tempered SMC and AMIS (gradient-free, batched, weighted samples + log evidence)
profile       profile likelihood in one parameter (all others re-optimised)
ode           rate-equation (ODE) evaluation of parameter sets for validation
report        credible intervals, derived quantities, plots and export

Parameter vector ``theta`` (the NN input coordinates without the two fixed
experimental inputs log10(n_pulse) and Eg)::

    theta = [log10 k_rad,
             log10 N_t1, E_t1/E_g, log10 tau_n1, log10 tau_p1,
             log10 N_t2, E_t2/E_g, log10 tau_n2, log10 tau_p2,  ...]

D = 1 + 4 * n_traps.  Units: k_rad [cm^3/s], N_t [cm^-3], tau [s].
"""
from __future__ import annotations

import sys
from pathlib import Path

#: Repository root (the folder that contains inference/, preprocessing/, ...)
REPO_ROOT = Path(__file__).resolve().parent.parent
#: Preprocessing and start-point code kept from the original submission
#: (data_utils, start_point_estimation_module, constants)
PREPROCESSING_DIR = REPO_ROOT / "preprocessing"
#: Data folder (experimental and synthetic data, AM1.5G spectrum)
DATA_DIR = REPO_ROOT / "data"
#: Network-training code (contains the ODE that generated the training data)
NETWORK_TRAINING_DIR = REPO_ROOT / "network-training"


def add_legacy_paths() -> None:
    """
    Make the modules in ``preprocessing/`` importable.

    ``preprocessing/data_utils.py`` and
    ``preprocessing/start_point_estimation_module.py`` use
    ``from constants import *``, so ``preprocessing/`` itself has to be on
    ``sys.path``.  ``network-training/`` is NOT added: it contains a module
    ``trpl_module`` that is loaded under a private name by ``inference.ode``.
    """
    p = str(PREPROCESSING_DIR)
    if p not in sys.path:
        sys.path.insert(0, p)


__all__ = ["REPO_ROOT", "PREPROCESSING_DIR", "DATA_DIR", "NETWORK_TRAINING_DIR", "add_legacy_paths"]
