# TRPL/SSPL Multitrap Analysis — Overview

This repository contains two complementary toolsets for extracting trap parameters
from time-resolved (TRPL) and steady-state photoluminescence (SSPL) measurements
of semiconductor thin films using Shockley–Read–Hall (SRH) recombination models.

---

## Where to find this work

| | Location |
|---|---|
| **Source code — current and all future versions** | <https://github.com/roheumann/rapid-trpl-bayes> |
| **Archived snapshot for the publication (frozen)** | [`10.26165/JUELICH-DATA/2IEHBF`](https://doi.org/10.26165/JUELICH-DATA/2IEHBF) — Jülich DATA |

> **All future updates to the code are published on GitHub.**
> Bug fixes, new features and later versions appear there and are *not*
> back-ported to the Jülich DATA deposit, which stays frozen as the exact code
> used to produce the published results. Before using this software, check the
> GitHub repository for the current version.

Cite the DOI when referring to this work; clone the GitHub repository when you
want to run or build on the code.

The measurement data used in the publication are deposited in Jülich DATA under
the DOI above. Download them from there and place them in
`scripts/fitting/data/` before running the workflow described below.

---

## Repository structure

```
scripts/
├── fitting/                 ← Step 1 — optimiser-based parameter fitting
│   ├── README.md            ← detailed fitting documentation (start here)
│   ├── sstrpl_fitting.py    ← main fitting engine (fit_multitrap, plot_fit, plot_corner)
│   ├── trpl_fitting.py      ← TRPL-only fitting + NN artefact loader (load_nn_artifacts)
│   ├── sspl_module.py       ← steady-state SRH physics solver
│   ├── trpl_module.py       ← transient SRH ODE solver
│   ├── data_utils.py        ← preprocessing helpers and export utilities
│   ├── start_point_estimation_module.py  ← heuristic initial-guess estimator
│   ├── constants.py         ← shared physical constants (NC, NV, VT, …)
│   ├── network/             ← pre-trained neural-network surrogate models
│   └── data/                ← place your experimental data files here
│
└── uncertainity_analysis/   ← Step 2 — Bayesian uncertainty quantification
    ├── README.md            ← detailed pipeline documentation (start here)
    ├── 01_create_grid.py    ← build parameter grid (Cartesian or slice)
    ├── 02_trpl_errors.py    ← evaluate TRPL errors via NN (GPU)
    ├── 03_sspl_errors.py    ← evaluate SSPL errors via physics solver (CPU)
    ├── 04_bayesian.py       ← compute posterior + corner plots
    ├── trpl_module.py       ← transient SRH ODE solver (shared with fitting/)
    ├── constants.py         ← physical constants (shared with fitting/)
    ├── trpl_fitting.py      ← NN artefact loader used by 02_trpl_errors.py
    └── config-paper*.json   ← example configuration files
```

---

## Recommended workflow

### Step 1 — Fit parameters (`fitting/`)

Use the fitting toolset to find the best-fit SRH parameters for your sample.
The fitting engine uses CMA-ES global optimisation to minimise the error between
measured and simulated TRPL (τ_diff vs QFLS) and/or SSPL (PLQY vs QFLS) data.

A pre-trained neural-network surrogate can replace the ODE solver during fitting
for a ≈ 100× speed-up (see `scripts/fitting/README.md`, Section "Neural-network
surrogate fitting").

See **[scripts/fitting/README.md](https://github.com/roheumann/rapid-trpl-bayes/blob/main/scripts/fitting/README.md)**
for the full API reference, data format specifications, and worked examples.

### Step 2 — Quantify uncertainty (`uncertainity_analysis/`)

Once a best-fit point is available, use the four-script pipeline to map the
Bayesian posterior over the full parameter space:

```
01_create_grid.py  →  02_trpl_errors.py  →  03_sspl_errors.py  →  04_bayesian.py
```

This produces posterior corner plots, 1-σ credible intervals, and contour files
suitable for publication figures.

See **[scripts/uncertainity_analysis/README.md](https://github.com/roheumann/rapid-trpl-bayes/blob/main/scripts/uncertainity_analysis/README.md)**
for the full config reference, workflow examples, and output file descriptions.

---

## How the two toolsets relate

| | `fitting/` | `uncertainity_analysis/` |
|---|---|---|
| **Purpose** | Find the best-fit parameters | Map the posterior probability over all parameters |
| **Method** | CMA-ES optimisation | Bayesian grid evaluation |
| **TRPL forward model** | ODE solver **or** NN surrogate | NN surrogate only |
| **SSPL forward model** | ODE solver | Analytic or numeric solver |
| **Output** | Best-fit parameters + fit plots | Posterior distributions + credible intervals |
| **Typical runtime** | Minutes to hours | Hours (full grid) or minutes (slice grid) |

The `best_fit.json_path` config field in the uncertainty pipeline accepts the
`*_parameters.json` file exported by `fitting/data_utils.export_fit_results`,
so the two tools connect directly.

---

## Dependencies

Install all required packages with pip:

```bash
pip install -r requirements.txt
```

Some packages are only required for specific features:

| Package | Required for |
|---|---|
| `numpy`, `pandas`, `scipy`, `matplotlib` | everything |
| `pymoo` | CMA-ES fitting (`fitting/`) |
| `joblib` | loading NN scalers |
| `h5py` | reading training HDF5 files |
| `tensorflow` | NN surrogate fitting and grid evaluation |
| `scikit-learn` | NN scaler objects (loaded with joblib) |
| `torch`, `splinetorch` | spline preprocessing (`data_utils.spline_torch`) |
| `seaborn` | corner plots in the fitting module |
| `corner` | legacy corner plot in `trpl_fitting.plot_corner` |
| `ipywidgets`, `ipympl` | interactive notebook widgets |

---

## Citation

If you use this software or the accompanying data, please cite this paper:

> Heumann, R., & Dreessen, C. (2026). * Rapid Parameter Estimation from Photoluminescence Decays of Halide Perovskite Thin Films*.
> Jülich DATA. https://doi.org/10.26165/JUELICH-DATA/2IEHBF

A machine-readable `CITATION.cff` is included in the root of the Git repository,
so GitHub's "Cite this repository" button will generate BibTeX and other formats
automatically.

---

## License

LICENCE-NAME (see `LICENSE` in the repository root).

---

## Authors

Robin Heumann, Toby Rudolph, Huang Gaosheng, Thomas Kirchartz and Chris Dreessen
Forschungszentrum Jülich GmbH — 2026
