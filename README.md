# rapid-trpl-bayes — trap parameters from photoluminescence with a neural-network surrogate

Code, data and trained networks of the publication

> R. Heumann, T. Rudolph, G. Huang, T. Kirchartz, C. Dreessen,
> *Rapid Parameter Estimation from Photoluminescence Decays of Halide Perovskite Thin Films* (2026),
> [arXiv:2609.18438](https://doi.org/10.48550/arXiv.2609.18438).

The pipeline extracts Shockley–Read–Hall trap parameters (N_t, E_t, τ_n, τ_p for one or two traps) from a
time-resolved photoluminescence (tr-PL) decay and, optionally, steady-state PL (ss-PL) data. A neural network replaces
the rate-equation (ODE) solver for the tr-PL curve τ_diff(QFLS). With it, the best fit and the full Bayesian posterior
take seconds instead of hours.

| | Location |
|---|---|
| **Publication** | <https://doi.org/10.48550/arXiv.2609.18438> |
| **Source code (current version)** | <https://github.com/roheumann/rapid-trpl-bayes> |
| **Archived snapshot, full training data** | [`10.26165/JUELICH-DATA/2IEHBF`](https://doi.org/10.26165/JUELICH-DATA/2IEHBF) (Jülich DATA) |

---

## Repository structure

```
main.ipynb            1. data -> preprocessing -> physics-informed start point -> fit (CMA-ES on the network,
                         refined with curve_fit on the rate equations)
smc.ipynb             2. posterior: tempered sequential Monte Carlo (credible intervals, log Z, corner HDF5)
ppc.ipynb             3. posterior predictive checks (network and rate equations)
ode_inference.ipynb   4. the same inference with the rate equations instead of the network (speed-up, NN vs ODE)

configs/              one JSON file per analysis of the publication (table below)
data/                 measured tr-PL and ss-PL data of the two films, synthetic 1-trap data, AM1.5G spectrum
networks/             the two trained networks (1 trap, 2 traps)
inference/            the inference package (likelihood, prior, surrogate, samplers, ...) — see inference/README.md
preprocessing/        tr-PL preprocessing (trim, spline, τ_diff vs QFLS) and the physics-informed start-point estimator
scripts/              command-line helpers: notebook runner, SBC, sliced corner, corner figures, tables, timing
model-error-analysis/ drift-diffusion model and σ_model (model error of the 0D rate equations)
network-training/     simulation of the training data and training of the networks (incl. the rate-equation solver)
corner-plot/          Origin scripts and templates for the corner figures
tests/                unit tests (python -m pytest tests -q)
results/              output of the notebooks and scripts (one folder per run)
```

All paths in the configs are relative to the repository root. Run the notebooks and scripts from there.

---

## Installation

```bash
pip install -r requirements.txt
```

Tested with Python 3.13, TensorFlow 2.21 (Keras 3.13), TensorFlow Probability 0.25 (+ tf_keras), emcee 3.1.6, ArviZ 1.3,
pycma 4.4, numba (rate equations), torch + splinetorch (spline preprocessing). A GPU is optional. Without one,
TensorFlow runs on the CPU.

---

## The analyses of the publication

| config | case |
|---|---|
| `configs/1trap_85-15_film_fit.json` | 85:15 film, 1-trap joint fit. Its best fit is the ground truth of the synthetic case |
| `configs/1trap_synthetic_inference.json` | synthetic 1-trap case (noise-free data in `data/synthetic_1trap/`): fit + posterior in the modes `trpl`, `sspl`, `joint` |
| `configs/2trap_85-15_film_fit.json` | 85:15 film, 2 traps, joint fit (reported best fit; σ_trPL = instrument noise of a GaAs reference) |
| `configs/2trap_85-15_film_inference.json` | 85:15 film, 2 traps, joint posterior (σ_trPL = √(σ_model² + σ_nn²) = 0.191 dec) |
| `configs/2trap_77-23_film_fit.json` | 77:23 film (PI-treated), 2 traps, joint fit (reported best fit) |
| `configs/2trap_77-23_film_inference.json` | 77:23 film, 2 traps, joint posterior (σ_trPL = 0.374 dec) |

Each config holds the film parameters (E_g, thickness, absorbed excitation density n_pulse, fixed k_rad, Voc,rad),
the network, the prior bounds, the data files, the σ of both data sets and the settings of every step. The
`_comment` field of each config says how its values were obtained.

### Likelihood

log L = Σ_d [−MSE_d / (2σ_d²) − log(σ_d √(2π))], where d is the tr-PL and/or the ss-PL curve. σ is the error of the
**whole curve** (each curve counts as one observation). For the inference runs, `sigma_rule: model_plus_nn` gives
σ_trPL = √(σ_model² + σ_nn²). σ_nn is the test error of the network, and σ_model is the RMSE between the drift-diffusion
model and the 0D rate equations at the best fit (`model-error-analysis/`). The prior is uniform on the training box of
the network, with E_t,1 < E_t,2 and the bounds of `prior.bounds_override`.

---

## How to run

Interactively: open `main.ipynb`, set `CONFIG_PATH` and `MODE` in Section 0, and run all cells. Then open `smc.ipynb`
(and `ppc.ipynb`) with `RUN_IN` = the run folder that `main.ipynb` wrote.

Headless, `scripts/run_notebook.py NOTEBOOK OUT [KEY=VALUE ...]` executes a notebook with the given settings. These are
the commands of the publication's runs (`QUICK=True` gives a fast smoke test; drop `CUDA_DEVICE` without a GPU):

**1. Fit and posterior of a measured film** (example: 77:23 film; the 85:15 film works the same with
`2trap_85-15_film_*` and `results/2trap_85-15_film/2trap_85-15_film_*`)

```bash
# reported best fit (σ_trPL = GaAs instrument noise)
python scripts/run_notebook.py main.ipynb results/2trap_77-23_film/main_fit.ipynb \
    'CONFIG_PATH="configs/2trap_77-23_film_fit.json"' 'MODE="joint"' 'NO_TIMING=True'

# σ_model at that best fit (drift-diffusion vs rate equations); the value is already in the inference config
python model-error-analysis/sigma_model.py --best-fit results/2trap_77-23_film/2trap_77-23_film_fit_joint/6_best_fit.json \
    --dd-only --alpha 61624.1675 --wavelength-nm 514 --no-robustness --out results/2trap_77-23_film/sigma_model

# fit at the inference σ, then the posterior (tempered SMC), posterior predictive checks and log Z
python scripts/run_notebook.py main.ipynb results/2trap_77-23_film/main.ipynb \
    'CONFIG_PATH="configs/2trap_77-23_film_inference.json"' 'MODE="joint"' 'NO_TIMING=True'
RUN=results/2trap_77-23_film/2trap_77-23_film_joint
python scripts/run_notebook.py smc.ipynb $RUN/smc.ipynb "RUN_IN=\"$RUN\"" 'TIMING_CONFIG=None' 'SMC_REPEATS=1' \
    'NUTS_REFERENCE=None' 'RUN_AMIS=False' 'RUN_CYL_IS=False' 'OUT_TAG="smc"'
python scripts/run_notebook.py ppc.ipynb $RUN/ppc.ipynb "RUN_IN=\"$RUN\"" 'SMC_DIR="smc"' 'N_DRAWS=2000' 'N_ODE=300'
python scripts/collect_log_evidence.py $RUN --out results/2trap_77-23_film/log_evidence_summary.csv

# corner figure: posterior (SMC) + slices through the reported best fit
FIT=results/2trap_77-23_film/2trap_77-23_film_fit_joint
python scripts/paper_slices.py $FIT --mode joint --n 100 --repeats 1 --sigma-trpl 0.3743 --sigma-sspl 0.0020214 \
    --out-tag slices_joint
python scripts/modes_corner.py results/2trap_77-23_film/2trap_77-23_film --modes joint --smc smc --draws 2000 \
    --best-fit-run $FIT --slice-file $FIT/slices_joint/slices_joint.h5 \
    --slice-panel log10_n1_1__log10_taup_1,log10_n1_2__log10_taup_2 --tau0-full
python scripts/render_corner_modes.py results/2trap_77-23_film/2trap_77-23_film_joint/corner_modes/corner_modes.h5
```

**2. Synthetic 1-trap case**

```bash
# ground truth: 1-trap fit of the measured 85:15 film
python scripts/run_notebook.py main.ipynb results/1trap_synthetic/main_fit.ipynb \
    'CONFIG_PATH="configs/1trap_85-15_film_fit.json"' 'MODE="joint"'
python scripts/start_point_study.py results/1trap_synthetic/1trap_85-15_film_fit_joint --n-sobol 64 --physics-seeds 16

# noise-free synthetic data from that fit (already in data/synthetic_1trap/ and configs/1trap_synthetic_inference.json;
# this command regenerates both)
python scripts/make_synthetic_1trap_85-15.py --fit-run results/1trap_synthetic/1trap_85-15_film_fit_joint \
    --out data/synthetic_1trap --config-out configs/1trap_synthetic_inference.json \
    --run-output-dir results/1trap_synthetic --base-config configs/1trap_85-15_film_fit.json

# per mode (trpl, sspl, joint): fit, sliced corner around the truth, SMC posterior, PPC, SBC
for MODE in trpl sspl joint; do
    RUN=results/1trap_synthetic/1trap_synthetic_$MODE
    python scripts/run_notebook.py main.ipynb results/1trap_synthetic/main_$MODE.ipynb \
        'CONFIG_PATH="configs/1trap_synthetic_inference.json"' "MODE=\"$MODE\""
    python scripts/paper_slices.py $RUN --center truth --n 100
    python scripts/run_notebook.py smc.ipynb $RUN/smc.ipynb "RUN_IN=\"$RUN\"" 'TIMING_CONFIG=None' \
        'NUTS_REFERENCE=None' 'RUN_AMIS=False' 'RUN_CYL_IS=False' 'OUT_TAG="smc"'
    python scripts/run_notebook.py ppc.ipynb $RUN/ppc.ipynb "RUN_IN=\"$RUN\"" 'SMC_DIR="smc"' 'N_ODE=0'
    python scripts/sbc_smc.py $RUN --n 200 --n-keep 100 --particles 2048 --islands 4 --seed 1
done

# tr-PL inference with the rate equations instead of the network (hours on 8 CPU cores), tables
python scripts/run_notebook.py ode_inference.ipynb results/1trap_synthetic/ode_inference_trpl.ipynb \
    'RUN_IN="results/1trap_synthetic/1trap_synthetic_trpl"' 'MODE="trpl"' 'RUN_ODE_EMCEE=False' 'RUN_NN_EMCEE=False'
python scripts/credible_intervals_table.py results/1trap_synthetic/1trap_synthetic --smc smc \
    --out results/1trap_synthetic/credible_intervals_1trap_synthetic
python scripts/end_to_end_timing.py results/1trap_synthetic/1trap_synthetic --out results/1trap_synthetic/end_to_end_timing
```

`scripts/README.md` lists every helper script. `corner-plot/origin/README.md` describes the Origin figures.

---

## Trained networks

`networks/1trap/` (1 trap) and `networks/2trap/` (2 traps, with an axis head that predicts the QFLS window of every
curve) hold the Keras models, the input/output scalers and a small HDF5 file with the metadata of their training set.
Training data, architecture and accuracy are described in `networks/README.md`. The full training data (5–7 GB per
network) are in Jülich DATA. `network-training/` has the code that simulated and trained them.

---

## Citation

```bibtex
@misc{heumann2026rapidparameterestimationphotoluminescence,
  title         = {Rapid Parameter Estimation from Photoluminescence Decays of Halide Perovskite Thin Films},
  author        = {Robin Heumann and Toby Rudolph and Gaosheng Huang and Thomas Kirchartz and Chris Dreessen},
  year          = {2026},
  eprint        = {2609.18438},
  archivePrefix = {arXiv},
  primaryClass  = {cond-mat.mtrl-sci},
  url           = {https://arxiv.org/abs/2609.18438},
}
```

A machine-readable `CITATION.cff` is included, so GitHub's "Cite this repository" button gives BibTeX and other formats.

## License

See `LICENSE`.

## Authors

Robin Heumann, Toby Rudolph, Gaosheng Huang, Thomas Kirchartz and Chris Dreessen — Forschungszentrum Jülich GmbH, 2026
