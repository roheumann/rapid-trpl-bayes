"""
Shared fixtures for the unit tests of the ``inference`` package.

The trained network weights are not part of the repository (Jülich DATA).
If they are found (default location ``../nn-training/<run>`` relative to the
repository, or the environment variable RTB_NN_DIR pointing to a run folder),
the tests use them.  Otherwise a randomly initialised model with the same
architecture (``network-training/train_nn.build_model``) and dummy scalers of
the right shape are used — all tests check consistency (legacy vs new code,
autodiff vs finite differences, bijector properties), not physics, so they
are meaningful with random weights as well.

Run from the repository root:   python -m pytest tests -q
"""
from __future__ import annotations

import glob
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

DEFAULT_RUNS = {
    1: REPO.parent / "nn-training" / "20260402-112530-3579-1Trap-65536sims",
    2: REPO.parent / "nn-training" / "20260407-231442-3594-2Trap-262128sims",
}


def load_module(path: Path, name: str):
    """Import a script by path under a private module name."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _real_artifacts(n_traps: int):
    from inference import surrogate as S
    run = Path(os.environ.get(f"RTB_NN_DIR_{n_traps}TRAP", DEFAULT_RUNS[n_traps]))
    h5 = sorted(glob.glob(str(run / "*_training.hdf5")))
    if not (run / "model.keras").exists() or not h5:
        return None
    stem = h5[0][:-len(".hdf5")]
    return S.load_artifacts(run / "model.keras", stem + "_param_scaler.joblib",
                            stem + "_output_scaler.joblib", h5[0])


def _random_artifacts(n_traps: int):
    """Random-weight network of the production architecture + dummy scalers."""
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    from inference import surrogate as S
    S.configure_tensorflow()
    train_nn = load_module(REPO / "network-training" / "train_nn.py", "rtb_train_nn")
    D = 3 + 4 * n_traps
    model = train_nn.build_model(input_dim=D, output_dim=256, num_conv_layers=3,
                                 kernel_size=16, max_filter=256)
    lb = np.array([np.log10(1.6e17), 1.625, -12] + [12, 0.5, -12, -12] * n_traps, float)
    ub = np.array([np.log10(1.6e17), 1.625, -10] + [20, 0.975, -4, -4] * n_traps, float)
    rng = np.random.default_rng(0)
    X = rng.uniform(lb, ub, size=(2000, D))
    si = StandardScaler().fit(X)
    so = MinMaxScaler().fit(np.array([[-12.0] * 256, [9.0] * 256]))
    constants = {"Nc": 2.21359e18, "Nv": 2.21359e18, "Vt": 0.02585199357397624,
                 "tmin": 1e-11, "tmax": 1e5, "N_TRAPS": float(n_traps),
                 "QFLS_STEPS": 256.0, "PL_DECAY_MAGNITUDE": 12.0}
    names = ["log_n_pulse", "Eg_eV", "log_krad"] + [
        f"{p}_{k}" for k in range(1, n_traps + 1)
        for p in ("log_Ntrap", "DeltaEtrap", "log_taun", "log_taup")]
    return S.SurrogateArtifacts(model=model, input_scaler=si, output_scaler=so,
                                constants=constants, lb=lb, ub=ub, param_names=names,
                                n_traps=n_traps, split_sizes={}, paths={"model": "random"})


@pytest.fixture(scope="session", params=[1, 2], ids=["1trap", "2trap"])
def artifacts(request):
    art = _real_artifacts(request.param)
    if art is None:
        art = _random_artifacts(request.param)
    return art


@pytest.fixture(scope="session")
def artifacts_2trap():
    return _real_artifacts(2) or _random_artifacts(2)


@pytest.fixture(scope="session")
def legacy_02():
    """Frozen tr-PL error code of the original grid pipeline (tests/legacy_reference.py)."""
    return load_module(REPO / "tests" / "legacy_reference.py", "legacy_reference")


@pytest.fixture(scope="session")
def legacy_03():
    """Frozen ss-PL solvers of the original grid pipeline (tests/legacy_reference.py)."""
    return load_module(REPO / "tests" / "legacy_reference.py", "legacy_reference")


def exp_trpl_data():
    """The experimental 85:15 tr-PL curve (preprocessed from the raw file in the repo)."""
    import pandas as pd
    paper = REPO.parent / "NN_Paper/fit_results/Ye-Genghua-Shallow-Trap/2trap-fit-exp/trpl_preprocessed_ye-genghua.csv"
    if paper.exists():
        df = pd.read_csv(paper)
    else:   # synthetic stand-in on the NN QFLS range
        q = np.linspace(0.97, 1.488, 120)
        df = pd.DataFrame({"qfls": q, "tau_diff": 1e-6 * np.exp(-(q - 0.97) / 0.06)})
    df = df[np.isfinite(df.tau_diff) & (df.tau_diff > 0)].sort_values("qfls")
    return df["qfls"].to_numpy(), df["tau_diff"].to_numpy()


def exp_sspl_data():
    import pandas as pd
    df = pd.read_csv(REPO / "data/85-15_film/sspl_suns_qfls.dat", sep="\t", skiprows=[1])
    return df["intensity"].to_numpy(float), df["qfls"].to_numpy(float)
