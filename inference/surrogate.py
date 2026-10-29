"""
surrogate.py — neural-network surrogate for tr-PL decays
========================================================

theta (B, D)  ->  log10 tau_diff at the measured QFLS points (B, n_exp)

Batched, differentiable (TensorFlow) and usable inside ``tf.function`` with
XLA.  This replaces the per-individual ``nn_model.predict`` calls of the legacy
optimiser (REVISION_PLAN.md, finding 1.4).

Facts about the trained networks that this module relies on
-----------------------------------------------------------
* Input vector, in this order:
  ``[log10 n_pulse, Eg_eV, log10 k_rad, (log10 N_t, E_t/Eg, log10 tau_n,
  log10 tau_p) per trap]``.  Inputs are scaled with a StandardScaler, outputs
  with a MinMaxScaler; both are affine and are applied here as TF constants.
* Output: 256 values of log10 tau_diff.  In training every curve was interpolated
  onto ITS OWN axis linspace(q_lo, q_hi, 256) of the physical ODE QFLS
  (prepare_training_data.py).  :class:`TrplSurrogate` assumes the analytic
  high-injection axis ``linspace(qfls_max - DECAY * Vt * ln10, qfls_max, 256)``,
  ``qfls_max = 2 Vt ln(n_pulse / n_i)``, which does not depend on the trap
  parameters (fixed interpolation matrix ``W`` (n_exp, 256)).  That holds for
  the fixed-n_pulse/Eg networks (3579/3594: upper end exact, lower end within
  10 meV for 99.5 % of the curves) but not for the generalised ones (3609, B2nf,
  3611): at low injection q_hi = kT ln((n0 + n_pulse)(p0 + n_pulse)/n_i^2) >
  qfls_max (19.6 % of the B2nf curves > 1 meV, up to ~200 meV), and 12 % of the
  curves reach equilibrium before 12 PL decades (shorter span).
  :class:`TrplSurrogateWindow` (axis-head networks from 16/10/2026) predicts
  both ends per theta and maps the data on that axis.
* The 2-trap training data contain ONLY E_t1 < E_t2 (Sobol design with an
  ordering filter).  Evaluating the network with E_t1 > E_t2 is
  extrapolation; ``prior.Prior`` therefore enforces the ordering.

Interpolation
-------------
``interp_space='log10'`` (default) interpolates log10 tau linearly in QFLS.
``interp_space='linear'`` reproduces the legacy grid pipeline
(``02_trpl_errors.py`` of the original submission), which interpolates tau linearly
and takes log10 afterwards; it is used by the unit test against the legacy
code.  With 256 points over ~0.7 eV (2.8 meV spacing) both agree to well
below the noise level.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


def _pow10(x):
    """10**x for float64 tensors (tf.pow(10.0, x) would promote the base to float32)."""
    import tensorflow as tf
    return tf.exp(tf.convert_to_tensor(x, tf.float64) * math.log(10.0))


# ─────────────────────────────────────────────────────────────────────────────
# Artefacts
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SurrogateArtifacts:
    """Everything that belongs to one trained network."""
    model: object                 # keras.Model
    input_scaler: object          # sklearn StandardScaler
    output_scaler: object         # sklearn MinMaxScaler
    constants: dict               # Nc, Nv, Vt, QFLS_STEPS, PL_DECAY_MAGNITUDE, ...
    lb: np.ndarray                # training lower bounds of the NN input vector
    ub: np.ndarray                # training upper bounds of the NN input vector
    param_names: list             # names of the NN input columns
    n_traps: int
    split_sizes: dict = field(default_factory=dict)   # train / val / test counts
    paths: dict = field(default_factory=dict)
    # PL-decade networks (from 10/2026): surrogate_meta.json next to the model,
    # and for the hybrid variant the branch/trunk sub-models
    meta: dict | None = None
    branch: object = None
    trunk: object = None
    heads: object = None          # DeepONet MLP: b -> (offset, window fraction)

    @property
    def axis(self) -> str:
        return self.meta["axis"] if self.meta else "qfls_window"

    @property
    def n_inputs(self) -> int:
        return len(self.lb)

    @property
    def dim(self) -> int:
        """Dimension of theta (NN inputs without log10 n_pulse and Eg)."""
        return self.n_inputs - 2


def configure_tensorflow(allow_tf32: bool = False, memory_growth: bool = True) -> None:
    """
    Global TensorFlow settings used by the whole workflow.

    * TensorFloat-32 is disabled by default: on Ampere GPUs (RTX A6000) TF32
      matmuls change the NN output by ~1e-3 in log10 tau and make XLA and
      non-XLA results differ; full float32 keeps optimiser, samplers and unit
      tests consistent.
    * GPU memory growth avoids grabbing the whole GPU.
    """
    import tensorflow as tf
    tf.config.experimental.enable_tensor_float_32_execution(bool(allow_tf32))
    if memory_growth:
        for gpu in tf.config.list_physical_devices("GPU"):
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                pass   # GPUs already initialised — setting cannot be changed any more


def load_artifacts(model_path, param_scaler_path, output_scaler_path,
                   training_hdf5_path, allow_tf32: bool = False) -> SurrogateArtifacts:
    """
    Load the Keras model, both scalers and the metadata of the training HDF5.

    The training HDF5 (written by ``network-training/prepare_training_data.py``)
    provides the physical constants used for the QFLS axis, the parameter
    bounds of the Sobol design (``lb``/``ub``, used as the prior support) and
    the sizes of the train/val/test splits (needed for R2 comment 3).
    """
    import h5py
    import joblib
    import keras

    configure_tensorflow(allow_tf32=allow_tf32)

    with warnings.catch_warnings():
        # sklearn warns when the scaler was pickled with another version; the
        # affine parameters (mean_, scale_, min_) are read directly below.
        warnings.simplefilter("ignore")
        input_scaler = joblib.load(param_scaler_path)
        output_scaler = joblib.load(output_scaler_path)
    from . import nn_layers  # noqa: F401  (registers custom layers, e.g. StepHead of the jump-aware networks)
    model = keras.models.load_model(model_path, compile=False)
    import json
    meta_path = Path(model_path).parent / "surrogate_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else None
    branch = trunk = heads = None
    if meta is not None and meta["variant"] in ("hybrid", "deeponet_mlp"):
        branch = keras.models.load_model(Path(model_path).parent / meta["files"]["branch"], compile=False)
        trunk = keras.models.load_model(Path(model_path).parent / meta["files"]["trunk"], compile=False)
    if meta is not None and meta["variant"] == "deeponet_mlp":
        heads = keras.models.load_model(Path(model_path).parent / meta["files"]["heads"], compile=False)

    with h5py.File(training_hdf5_path, "r") as hf:
        c = hf["constants"][()]
        constants = {name: float(c[name]) for name in c.dtype.names}
        lb = np.asarray(hf["lb"][:], dtype=np.float64)
        ub = np.asarray(hf["ub"][:], dtype=np.float64)
        names = [n.decode() if isinstance(n, bytes) else str(n)
                 for n in hf["param_names"][:]]
        split_sizes = {s: int(hf[f"X_{s}_raw"].shape[0])
                       for s in ("train", "val", "test") if f"X_{s}_raw" in hf}
        # Consistency check: the design bounds must contain the training data
        x_min = np.asarray(hf["X_train_raw"][:].min(axis=0), np.float64)
        x_max = np.asarray(hf["X_train_raw"][:].max(axis=0), np.float64)
    if np.any(x_min < lb - 1e-3) or np.any(x_max > ub + 1e-3):
        warnings.warn("Training data exceed the lb/ub bounds stored in the HDF5 file.")

    n_traps = (len(lb) - 3) // 4
    if n_traps < 1 or 3 + 4 * n_traps != len(lb):
        raise ValueError(f"Unexpected number of NN inputs: {len(lb)}")
    n_in = (branch if branch is not None else model).input_shape[-1]
    if n_in != len(lb):
        raise ValueError(f"Model expects {n_in} inputs, HDF5 lists {len(lb)} parameters.")

    return SurrogateArtifacts(
        model=model, input_scaler=input_scaler, output_scaler=output_scaler,
        constants=constants, lb=lb, ub=ub, param_names=names, n_traps=n_traps,
        split_sizes=split_sizes,
        paths={"model": str(model_path), "param_scaler": str(param_scaler_path),
               "output_scaler": str(output_scaler_path),
               "training_hdf5": str(training_hdf5_path)},
        meta=meta, branch=branch, trunk=trunk, heads=heads)


# ─────────────────────────────────────────────────────────────────────────────
# QFLS axis and interpolation matrix
# ─────────────────────────────────────────────────────────────────────────────

def nn_qfls_axis(npulse: float, Eg: float, constants: dict) -> np.ndarray:
    """
    QFLS axis [eV] of the network output for given n_pulse and Eg.

    Identical to ``compute_qfls_network`` of the original fitting code but uses the
    constants stored in the training HDF5 (Nc, Nv, Vt, QFLS_STEPS,
    PL_DECAY_MAGNITUDE), so it always matches the training data.
    """
    Nc, Nv, Vt = constants["Nc"], constants["Nv"], constants["Vt"]
    steps = int(constants["QFLS_STEPS"])
    decay = float(constants["PL_DECAY_MAGNITUDE"])
    ni = np.sqrt(Nc * Nv) * np.exp(-Eg / (2.0 * Vt))
    qfls_max = 2.0 * Vt * np.log(npulse / ni)
    qfls_min = qfls_max - decay * Vt * np.log(10.0)
    return np.linspace(qfls_min, qfls_max, steps)


def qfls_axis_consistency(artifacts: "SurrogateArtifacts", npulse: float | None = None,
                          Eg: float | None = None, near: tuple = (0.2, 0.06),
                          tol_hi: float = 1e-3, tol_lo: float = 0.01) -> dict:
    """
    How well does the fixed analytic QFLS axis describe the training targets?

    During training every curve was interpolated onto ITS OWN axis
    linspace(qfls_min_i, qfls_max_i, 256) (prepare_training_data.py), while
    the surrogate assumes the analytic axis of :func:`nn_qfls_axis`.  This
    function compares both ends for all training samples and, if n_pulse and
    Eg are given, for the samples near the experiment
    (|log10 n_pulse - log10 n| < near[0], |Eg - Eg_i| < near[1]).

    Returns fractions of samples whose upper end deviates by more than
    ``tol_hi`` [eV] and whose lower end is truncated by more than ``tol_lo``.
    (Fixed-n_pulse/Eg networks: ~0 / ~0.5 %; generalised networks: up to
    ~1/3 of the samples, mostly at low n_pulse — check before using them.)

    PL-decade networks (from 10/2026) have no per-curve axis: every training
    curve spans the full 12 decades (curves that did not were removed in
    prepare_training_data.py).  For them the removed counts are returned.
    """
    import h5py
    import json
    if artifacts.axis == "pl_decades":
        with h5py.File(artifacts.paths["training_hdf5"], "r") as hf:
            removed = json.loads(hf["removed_counts"][()].decode()) if "removed_counts" in hf else {}
            n = int(hf["X_train_raw"].shape[0] + hf["X_val_raw"].shape[0] + hf["X_test_raw"].shape[0])
            masked = float(np.mean(~np.isfinite(hf["y_train_raw"][:20000])))
        return {"axis": "pl_decades", "n_samples": n, "removed_in_preparation": removed,
                "frac_removed": sum(removed.values()) / max(1, n + sum(removed.values())),
                "frac_masked_grid_points_train": masked}
    with h5py.File(artifacts.paths["training_hdf5"], "r") as hf:
        X = hf["X_train_raw"][:, :2].astype(np.float64)
        q_lo = hf["qfls_train"][:, 0].astype(np.float64)
        q_hi = hf["qfls_train"][:, -1].astype(np.float64)
    c = artifacts.constants
    Vt, Nc, Nv = c["Vt"], c["Nc"], c["Nv"]
    ni = np.sqrt(Nc * Nv) * np.exp(-X[:, 1] / (2 * Vt))
    qmax = 2 * Vt * np.log(10 ** X[:, 0] / ni)
    qmin = qmax - c["PL_DECAY_MAGNITUDE"] * Vt * np.log(10)
    off_hi = np.abs(q_hi - qmax) > tol_hi
    trunc = (q_lo - qmin) > tol_lo
    out = {"n_samples": int(len(X)), "frac_upper_end_off": float(off_hi.mean()),
           "frac_lower_end_truncated": float(trunc.mean())}
    if npulse is not None and Eg is not None:
        sel = (np.abs(X[:, 0] - np.log10(npulse)) <= near[0]) & (np.abs(X[:, 1] - Eg) <= near[1])
        out.update({"n_near": int(sel.sum()),
                    "frac_upper_end_off_near": float(off_hi[sel].mean()) if sel.any() else np.nan,
                    "frac_lower_end_truncated_near": float(trunc[sel].mean()) if sel.any() else np.nan})
    return out


def linear_interp_matrix(x_grid: np.ndarray, x_new: np.ndarray) -> np.ndarray:
    """
    Matrix W with  y(x_new) = W @ y(x_grid)  for piecewise-linear interpolation.

    x_grid must be strictly increasing and every x_new must lie inside
    [x_grid[0], x_grid[-1]].  Each row of W has at most two non-zero entries
    (1 - w, w) that sum to one.
    """
    x_grid = np.asarray(x_grid, np.float64)
    x_new = np.asarray(x_new, np.float64)
    if np.any(x_new < x_grid[0] - 1e-12) or np.any(x_new > x_grid[-1] + 1e-12):
        raise ValueError("x_new outside the interpolation grid")
    j = np.clip(np.searchsorted(x_grid, x_new, side="right") - 1, 0, len(x_grid) - 2)
    w = (x_new - x_grid[j]) / (x_grid[j + 1] - x_grid[j])
    W = np.zeros((len(x_new), len(x_grid)))
    rows = np.arange(len(x_new))
    W[rows, j] = 1.0 - w
    W[rows, j + 1] = w
    return W


# ─────────────────────────────────────────────────────────────────────────────
# Surrogate for one measured tr-PL curve
# ─────────────────────────────────────────────────────────────────────────────

class TrplSurrogate:
    """
    NN prediction of ONE measured tr-PL curve.

    Parameters
    ----------
    artifacts : SurrogateArtifacts
    npulse, Eg : float
        Experimental excitation density [cm^-3] and bandgap [eV] — the two
        fixed NN inputs.  They must lie inside the training range (fixed
        networks 3579/3594: n_pulse = 1.6e17 cm^-3, Eg = 1.625 eV only;
        generalised networks: 1e12-1e18 cm^-3, 1.15-2.4 eV).
    qfls_exp : array (n,)
        Measured QFLS points [eV].  Points outside the NN QFLS axis cannot be
        predicted and are excluded; ``mask_used`` tells which ones are kept.
    interp_space : 'log10' | 'linear'
        Interpolate log10(tau) (default) or tau (legacy grid pipeline).
    """

    def __init__(self, artifacts: SurrogateArtifacts, npulse: float, Eg: float,
                 qfls_exp, interp_space: str = "log10"):
        import tensorflow as tf

        if interp_space not in ("log10", "linear"):
            raise ValueError("interp_space must be 'log10' or 'linear'")
        self.artifacts = artifacts
        self.npulse = float(npulse)
        self.Eg = float(Eg)
        self.interp_space = interp_space
        self.dim = artifacts.dim
        self.n_traps = artifacts.n_traps

        # ── the experimental inputs must lie inside the training range ───────
        lb, ub = artifacts.lb, artifacts.ub
        x_fixed = np.array([np.log10(self.npulse), self.Eg])
        tol = np.array([1e-3, 1e-4])
        if np.any(x_fixed < lb[:2] - tol) or np.any(x_fixed > ub[:2] + tol):
            raise ValueError(
                f"(log10 n_pulse, Eg) = {x_fixed} outside the training range "
                f"[{lb[:2]}, {ub[:2]}] of this network — that would be extrapolation.")

        # ── fixed QFLS axis and interpolation matrix ─────────────────────────
        self.qfls_axis = nn_qfls_axis(self.npulse, self.Eg, artifacts.constants)
        qfls_exp = np.asarray(qfls_exp, np.float64)
        self.mask_used = (qfls_exp >= self.qfls_axis[0]) & (qfls_exp <= self.qfls_axis[-1])
        self.qfls_used = qfls_exp[self.mask_used]
        if len(self.qfls_used) == 0:
            raise ValueError("No measured QFLS point lies on the NN QFLS axis.")
        self.n_points = len(self.qfls_used)
        W = linear_interp_matrix(self.qfls_axis, self.qfls_used)       # (n, 256)
        self._W_T = tf.constant(W.T, tf.float64)                        # (256, n)

        # ── affine scalers as constants ───────────────────────────────────────
        si, so = artifacts.input_scaler, artifacts.output_scaler
        self._x_fixed = tf.constant(x_fixed, tf.float64)
        self._in_mean = tf.constant(np.asarray(si.mean_, np.float64))
        self._in_scale = tf.constant(np.asarray(si.scale_, np.float64))
        # MinMaxScaler:  y_scaled = y * scale_ + min_   ->  y = (y_scaled - min_) / scale_
        self._out_min = tf.constant(np.asarray(so.min_, np.float64))
        self._out_scale = tf.constant(np.asarray(so.scale_, np.float64))
        self._model = artifacts.model

    # ------------------------------------------------------------------
    def nn_inputs(self, theta):
        """Full NN input vector [log10 n_pulse, Eg, theta] (unscaled), (B, 2+D)."""
        import tensorflow as tf
        theta = tf.convert_to_tensor(theta, tf.float64)
        fixed = tf.broadcast_to(self._x_fixed, [tf.shape(theta)[0], 2])
        return tf.concat([fixed, theta], axis=1)

    def nn_log10_tau(self, theta):
        """
        log10 tau_diff [s] on the NN QFLS axis, (B, 256), float64.

        The input scaling is done in float64 and the result cast to float32
        for the network (as in the legacy code); the output is unscaled in
        float64.
        """
        import tensorflow as tf
        x = (self.nn_inputs(theta) - self._in_mean) / self._in_scale
        y = self._model(tf.cast(x, tf.float32), training=False)
        return (tf.cast(y, tf.float64) - self._out_min) / self._out_scale

    def log10_tau(self, theta):
        """log10 tau_diff [s] at the measured (used) QFLS points, (B, n_points)."""
        import tensorflow as tf
        y = self.nn_log10_tau(theta)
        if self.interp_space == "log10":
            return tf.matmul(y, self._W_T)
        # legacy: interpolate tau linearly, then log10
        return tf.math.log(tf.matmul(_pow10(y), self._W_T)) / np.log(10.0)

    # ------------------------------------------------------------------
    def predict_numpy(self, theta, batch_size: int = 8192) -> np.ndarray:
        """Convenience: log10 tau on the NN axis for a numpy array of theta."""
        import tensorflow as tf
        theta = np.atleast_2d(np.asarray(theta, np.float64))
        out = [self.nn_log10_tau(tf.constant(theta[s:s + batch_size])).numpy()
               for s in range(0, len(theta), batch_size)]
        return np.concatenate(out, axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# PL-decade networks (from 10/2026)
# ─────────────────────────────────────────────────────────────────────────────

def u_features(u, decades: float, n_freq: int) -> np.ndarray:
    """Trunk input phi(u) of the hybrid networks — must match network-training/train_nn.u_features."""
    u = np.asarray(u, np.float64)
    un = (u + decades / 2.0) / (decades / 2.0)
    feats = [un]
    for k in range(n_freq):
        w = (2.0 ** k) * np.pi
        feats += [np.sin(w * un), np.cos(w * un)]
    return np.stack(feats, axis=-1).astype(np.float32)


class TrplSurrogateU(TrplSurrogate):
    """
    Surrogate of a network trained on the PL-decade axis u = log10(PL/PL0).

    Same interface as :class:`TrplSurrogate`.  The measured QFLS points come
    from ``data_utils.t_pl_to_qfls_tau`` (calibration with p0 = 0:
    QFLS = qfls_max + Vt ln(PL/PL_max), qfls_max = 2 Vt ln(n_pulse/n_i)), so
    u = (QFLS - qfls_max)/(Vt ln10) recovers the calibration-free PL decades
    exactly; the true calibration of the sample (with p0 from trap
    neutrality) never enters the likelihood.  ``qfls_axis`` is the 256-point
    evaluation grid of the network in that same nominal QFLS
    (= the analytic axis of the older networks).

    minimal : network output on the 256-point u grid, linear interpolation
              to the measured points (as TrplSurrogate)
    hybrid  : branch(theta) . trunk(phi(u)) evaluated directly at the measured
              u; the trunk part is computed once here, so per theta only the
              branch runs.
    """

    def __init__(self, artifacts: SurrogateArtifacts, npulse: float, Eg: float,
                 qfls_exp, interp_space: str = "log10"):
        import tensorflow as tf

        if interp_space != "log10":
            raise ValueError("PL-decade networks support interp_space='log10' only")
        meta = artifacts.meta
        self.artifacts = artifacts
        self.npulse, self.Eg = float(npulse), float(Eg)
        self.interp_space = interp_space
        self.dim = artifacts.dim
        self.n_traps = artifacts.n_traps
        self.variant = meta["variant"]

        lb, ub = artifacts.lb, artifacts.ub
        x_fixed = np.array([np.log10(self.npulse), self.Eg])
        tol = np.array([1e-3, 1e-4])
        if np.any(x_fixed < lb[:2] - tol) or np.any(x_fixed > ub[:2] + tol):
            raise ValueError(
                f"(log10 n_pulse, Eg) = {x_fixed} outside the training range "
                f"[{lb[:2]}, {ub[:2]}] of this network — that would be extrapolation.")

        c = artifacts.constants
        self.u_axis = np.asarray(meta["u_eval"], np.float64)                   # (256,) increasing, -12 ... 0
        Vt = c["Vt"]
        ni = np.sqrt(c["Nc"] * c["Nv"]) * np.exp(-self.Eg / (2.0 * Vt))
        self.qfls_max = 2.0 * Vt * np.log(self.npulse / ni)
        self._to_u = 1.0 / (Vt * np.log(10.0))
        self.qfls_axis = self.qfls_max + self.u_axis / self._to_u               # nominal QFLS of the grid

        qfls_exp = np.asarray(qfls_exp, np.float64)
        u_exp = (qfls_exp - self.qfls_max) * self._to_u
        self.mask_used = (u_exp >= self.u_axis[0] - 1e-9) & (u_exp <= self.u_axis[-1] + 1e-9)
        self.qfls_used = qfls_exp[self.mask_used]
        self.u_used = np.clip(u_exp[self.mask_used], self.u_axis[0], self.u_axis[-1])
        if len(self.qfls_used) == 0:
            raise ValueError("No measured point lies within the 12 PL decades of the network.")
        self.n_points = len(self.qfls_used)

        si = artifacts.input_scaler
        self._x_fixed = tf.constant(x_fixed, tf.float64)
        self._in_mean = tf.constant(np.asarray(si.mean_, np.float64))
        self._in_scale = tf.constant(np.asarray(si.scale_, np.float64))
        y_lo, y_hi = meta["y_scaled_to_log10_tau"]["min"], meta["y_scaled_to_log10_tau"]["max"]
        self._y_lo, self._y_span = float(y_lo), float(y_hi - y_lo)

        if self.variant == "hybrid":
            f = lambda u: u_features(u, meta["decades"], meta["fourier_freqs"])[None]
            self._branch = artifacts.branch
            T_data = artifacts.trunk(tf.constant(f(self.u_used)), training=False)[0]      # (n, p)
            T_axis = artifacts.trunk(tf.constant(f(self.u_axis)), training=False)[0]      # (256, p)
            self._T_data_T = tf.transpose(tf.cast(T_data, tf.float32))
            self._T_axis_T = tf.transpose(tf.cast(T_axis, tf.float32))
        else:
            self._model = artifacts.model
            W = linear_interp_matrix(self.u_axis, self.u_used)
            self._W_T = tf.constant(W.T, tf.float64)

    # ------------------------------------------------------------------
    def _unscale(self, y_scaled):
        import tensorflow as tf
        return tf.cast(y_scaled, tf.float64) * self._y_span + self._y_lo

    def _x32(self, theta):
        import tensorflow as tf
        return tf.cast((self.nn_inputs(theta) - self._in_mean) / self._in_scale, tf.float32)

    def nn_log10_tau(self, theta):
        """log10 tau_diff [s] on the 256-point u grid (= qfls_axis), (B, 256), float64."""
        import tensorflow as tf
        if self.variant == "hybrid":
            b = self._branch(self._x32(theta), training=False)
            return self._unscale(tf.sigmoid(tf.matmul(b, self._T_axis_T)))
        return self._unscale(self._model(self._x32(theta), training=False))

    def log10_tau(self, theta):
        """log10 tau_diff [s] at the measured (used) points, (B, n_points)."""
        import tensorflow as tf
        if self.variant == "hybrid":
            b = self._branch(self._x32(theta), training=False)
            return self._unscale(tf.sigmoid(tf.matmul(b, self._T_data_T)))
        return tf.matmul(self.nn_log10_tau(theta), self._W_T)


def qfls_features(q, lo: float, hi: float, n_freq: int = 0) -> np.ndarray:
    """Trunk input of the DeepONet MLP — must match network-training/train_deeponet.qfls_features."""
    qn = 2.0 * (np.asarray(q, np.float64) - lo) / (hi - lo) - 1.0
    feats = [qn]
    for k in range(n_freq):
        w = (2.0 ** k) * np.pi
        feats += [np.sin(w * qn), np.cos(w * qn)]
    return np.stack(feats, axis=-1).astype(np.float32)


class TrplSurrogateDeepONet(TrplSurrogate):
    """
    Simple DeepONet (MLP branch + MLP trunk, network-training/train_deeponet.py).

    The trunk is a function of the absolute QFLS, so the network is evaluated
    directly AT the measured QFLS points (high-injection calibration of the
    preprocessing, QFLS(0) = 2 Vt ln(n_pulse/n_i)); the trunk part is computed
    once here and per theta only the branch runs.  ``qfls_axis`` is the
    analytic 12-decade axis (for SBC / plots, as for the older networks).
    ``window_start(theta)`` returns the start of the predicted window
    (span head).
    """

    def __init__(self, artifacts: SurrogateArtifacts, npulse: float, Eg: float,
                 qfls_exp, interp_space: str = "log10"):
        import tensorflow as tf
        if interp_space != "log10":
            raise ValueError("the DeepONet surrogate supports interp_space='log10' only")
        meta = artifacts.meta
        self.artifacts, self.variant = artifacts, meta["variant"]
        self.npulse, self.Eg = float(npulse), float(Eg)
        self.interp_space = interp_space
        self.dim, self.n_traps = artifacts.dim, artifacts.n_traps
        lb, ub = artifacts.lb, artifacts.ub
        x_fixed = np.array([np.log10(self.npulse), self.Eg])
        tol = np.array([1e-3, 1e-4])
        if np.any(x_fixed < lb[:2] - tol) or np.any(x_fixed > ub[:2] + tol):
            raise ValueError(f"(log10 n_pulse, Eg) = {x_fixed} outside the training range "
                             f"[{lb[:2]}, {ub[:2]}] of this network — that would be extrapolation.")
        self.qfls_axis = nn_qfls_axis(self.npulse, self.Eg, artifacts.constants)
        qfls_exp = np.asarray(qfls_exp, np.float64)
        self.mask_used = (qfls_exp >= self.qfls_axis[0]) & (qfls_exp <= self.qfls_axis[-1])
        self.qfls_used = qfls_exp[self.mask_used]
        if len(self.qfls_used) == 0:
            raise ValueError("No measured QFLS point lies within the 12 PL decades of the network.")
        self.n_points = len(self.qfls_used)

        si = artifacts.input_scaler
        self._x_fixed = tf.constant(x_fixed, tf.float64)
        self._in_mean = tf.constant(np.asarray(si.mean_, np.float64))
        self._in_scale = tf.constant(np.asarray(si.scale_, np.float64))
        y = meta["y_scaled_to_log10_tau"]
        self._y_lo, self._y_span = float(y["min"]), float(y["max"] - y["min"])
        qn = meta["qfls_norm"]
        feat = lambda q: tf.constant(qfls_features(q, qn["lo"], qn["hi"], meta.get("fourier_freqs", 0))[None])
        self._branch, self._heads = artifacts.branch, artifacts.heads
        self._T_data_T = tf.transpose(tf.cast(artifacts.trunk(feat(self.qfls_used), training=False)[0], tf.float32))
        self._T_axis_T = tf.transpose(tf.cast(artifacts.trunk(feat(self.qfls_axis), training=False)[0], tf.float32))
        self._floor = self.qfls_axis[-1] - meta["decades"] * artifacts.constants["Vt"] * np.log(10.0)

    def _x32(self, theta):
        import tensorflow as tf
        return tf.cast((self.nn_inputs(theta) - self._in_mean) / self._in_scale, tf.float32)

    def _eval(self, theta, T_T):
        import tensorflow as tf
        b = self._branch(self._x32(theta), training=False)
        offset, _ = self._heads(b, training=False)
        return tf.cast(tf.matmul(b, T_T) + offset, tf.float64) * self._y_span + self._y_lo

    def nn_log10_tau(self, theta):
        """log10 tau_diff [s] on the analytic 12-decade QFLS axis, (B, 256)."""
        return self._eval(theta, self._T_axis_T)

    def log10_tau(self, theta):
        """log10 tau_diff [s] at the measured (used) QFLS points, (B, n_points)."""
        return self._eval(theta, self._T_data_T)

    def window_start(self, theta) -> np.ndarray:
        """Predicted lower end of the curve's QFLS window [eV] (span head), numpy (B,)."""
        import tensorflow as tf
        _, frac = self._heads(self._branch(self._x32(tf.constant(np.atleast_2d(theta))), training=False))
        return self._floor + (self.qfls_axis[-1] - self._floor) * frac.numpy()[:, 0]


def calibrate_qfls(qfls_nominal, qfls_max_analytic: float, q_max, Vt: float) -> np.ndarray:
    """
    Measured QFLS (calibrated with the high-injection formula, top = ``qfls_max_analytic``) on the calibration of a
    model with QFLS(t=0) = ``q_max``: same PL decade u = log10(PL/PL_0), q = Vt ln(1 + 10^u (exp(q_max/Vt) - 1)).
    q_max scalar or (B,) -> (n,) or (B, n).
    """
    q_max = np.asarray(q_max, np.float64)
    u = (np.asarray(qfls_nominal, np.float64) - qfls_max_analytic) / (Vt * np.log(10.0))
    pu = 10.0 ** np.minimum(u, 0.0)
    qm = q_max[..., None] if q_max.ndim else q_max
    return qm + Vt * np.log(pu + (1.0 - pu) * np.exp(-qm / Vt))


class TrplSurrogateWindow(TrplSurrogate):
    """
    Network with an axis head (``nn-training/train_axis_head.py``, 16/10/2026): outputs
    [curve (256), axis (2)] with axis = [d_top, span] / axis_unit_eV, where

        q_max(theta) = 2 Vt ln(n_pulse/n_i) + d_top(theta)     (QFLS at t = 0, also at low injection)
        axis(theta)  = linspace(q_max - span, q_max, 256)      (the curve's own training axis)

    The data enter as PL decades u_k = log10(PL_k/PL_0) = (QFLS_exp,k - qfls_max)/(Vt ln10): the measured QFLS of
    ``data_utils.t_pl_to_qfls_tau`` is calibrated with the high-injection formula, so u_k is exact whatever the
    calibration.  Their position on the model's axis follows from PL = n p - n_i^2 = n_i^2 (exp(q/Vt) - 1):

        q_k(theta) = Vt ln(1 + 10^u_k (exp(q_max/Vt) - 1))     (= q_max + u_k Vt ln10 for q >> Vt)

    and the curve is read by linear interpolation at the fractional index 255 (q_k - q_lo)/(q_max - q_lo).
    Exclusion rule (Robin, 16/10/2026): if the curve ends above a measured point (q_k < q_lo, the decay reaches
    equilibrium before it has dropped as far as measured), that theta cannot produce the data: ``log10_tau`` returns
    +inf there, which the likelihood maps to log L = -inf.  No padding or filling.

    Points more than 12 decades below the PL maximum are dropped for every theta (as before); ``qfls_axis`` stays
    the nominal analytic axis for code that needs a fixed grid.  For plots use ``qfls_axis_at(theta)`` and
    ``data_qfls_at(theta)`` (the data on the model's calibration).
    """

    def __init__(self, artifacts: SurrogateArtifacts, npulse: float, Eg: float,
                 qfls_exp, interp_space: str = "log10"):
        import tensorflow as tf
        if interp_space != "log10":
            raise ValueError("axis-head networks support interp_space='log10' only")
        meta = artifacts.meta
        self.artifacts = artifacts
        self.npulse, self.Eg = float(npulse), float(Eg)
        self.interp_space = interp_space
        self.dim, self.n_traps = artifacts.dim, artifacts.n_traps
        lb, ub = artifacts.lb, artifacts.ub
        x_fixed = np.array([np.log10(self.npulse), self.Eg])
        tol = np.array([1e-3, 1e-4])
        if np.any(x_fixed < lb[:2] - tol) or np.any(x_fixed > ub[:2] + tol):
            raise ValueError(f"(log10 n_pulse, Eg) = {x_fixed} outside the training range "
                             f"[{lb[:2]}, {ub[:2]}] of this network — that would be extrapolation.")
        c = artifacts.constants
        self.Vt = float(c["Vt"])
        self.decades = float(c["PL_DECAY_MAGNITUDE"])
        self.qfls_axis = nn_qfls_axis(self.npulse, self.Eg, c)                  # nominal (analytic) axis
        self.qfls_max_analytic = float(self.qfls_axis[-1])
        self._unit = float(meta["axis_unit_eV"])
        qfls_exp = np.asarray(qfls_exp, np.float64)
        u_exp = (qfls_exp - self.qfls_max_analytic) / (self.Vt * np.log(10.0))
        self.mask_used = (u_exp >= -self.decades - 1e-9) & (u_exp <= 1e-9)
        self.qfls_used = qfls_exp[self.mask_used]
        self.u_used = np.minimum(u_exp[self.mask_used], 0.0)
        if len(self.qfls_used) == 0:
            raise ValueError("No measured point lies within the 12 PL decades of the network.")
        self.n_points = len(self.qfls_used)
        self._u_T = tf.constant(self.u_used[None], tf.float64)                   # (1, n)
        si, so = artifacts.input_scaler, artifacts.output_scaler
        self._x_fixed = tf.constant(x_fixed, tf.float64)
        self._in_mean = tf.constant(np.asarray(si.mean_, np.float64))
        self._in_scale = tf.constant(np.asarray(si.scale_, np.float64))
        self._out_min = tf.constant(np.asarray(so.min_, np.float64))
        self._out_scale = tf.constant(np.asarray(so.scale_, np.float64))
        self._model = artifacts.model
        self._steps = int(c["QFLS_STEPS"])

    # ------------------------------------------------------------------
    def _forward(self, theta):
        """(log10 tau (B, 256), q_max (B,), q_lo (B,)) in float64."""
        import tensorflow as tf
        x = (self.nn_inputs(theta) - self._in_mean) / self._in_scale
        curve, axis = self._model(tf.cast(x, tf.float32), training=False)
        y = (tf.cast(curve, tf.float64) - self._out_min) / self._out_scale
        a = tf.cast(axis, tf.float64) * self._unit
        q_max = self.qfls_max_analytic + a[:, 0]
        return y, q_max, q_max - a[:, 1]

    def nn_log10_tau(self, theta):
        """log10 tau_diff [s] on the curve's own axis (``qfls_axis_at(theta)``), (B, 256), float64."""
        return self._forward(theta)[0]

    def values_at_u(self, theta, u):
        """
        log10 tau at PL decades u (B, n) or (1, n) for every row of theta, and the coverage mask (B, n).
        Uncovered points (below the end of the curve) are +inf; the gradient stays finite on covered points.
        """
        import tensorflow as tf
        y, q_max, q_lo = self._forward(theta)
        u = tf.convert_to_tensor(u, tf.float64)
        x = q_max[:, None] / self.Vt
        # q_k = Vt ln(1 + 10^u (exp(q_max/Vt) - 1)), evaluated as q_max + Vt ln(10^u + (1 - 10^u) exp(-q_max/Vt))
        pu = _pow10(u)
        q = q_max[:, None] + self.Vt * tf.math.log(pu + (1.0 - pu) * tf.exp(-x))
        span = tf.maximum(q_max - q_lo, 1e-9)[:, None]
        j = (self._steps - 1) * (q - q_lo[:, None]) / span
        covered = j >= -1e-9
        j = tf.clip_by_value(j, 0.0, float(self._steps - 1))
        i0 = tf.minimum(tf.cast(tf.floor(j), tf.int32), self._steps - 2)
        w = j - tf.cast(i0, tf.float64)
        y0 = tf.gather(y, i0, batch_dims=1)
        y1 = tf.gather(y, i0 + 1, batch_dims=1)
        val = y0 + w * (y1 - y0)
        return tf.where(covered, val, tf.constant(np.inf, tf.float64)), covered

    def log10_tau(self, theta):
        """log10 tau_diff [s] at the measured (used) points, (B, n_points); +inf where the curve ends above them."""
        import tensorflow as tf
        theta = tf.convert_to_tensor(theta, tf.float64)
        u = tf.broadcast_to(self._u_T, [tf.shape(theta)[0], self.n_points])
        return self.values_at_u(theta, u)[0]

    # ------------------------------------------------------------------ numpy helpers (plots, exports)
    def nn_log10_tau_nominal(self, theta) -> np.ndarray:
        """Every row's curve on the fixed nominal axis ``qfls_axis`` (data calibration), NaN beyond its end, (B, 256)."""
        import tensorflow as tf
        th = tf.constant(np.atleast_2d(np.asarray(theta, np.float64)))
        u = (self.qfls_axis - self.qfls_max_analytic) / (self.Vt * np.log(10.0))
        v = self.values_at_u(th, tf.constant(np.minimum(u, 0.0)[None]))[0].numpy()
        return np.where(np.isfinite(v), v, np.nan)

    def window(self, theta):
        """(q_max, q_lo) [eV] of the predicted axis, numpy (B,) each."""
        import tensorflow as tf
        _, q_max, q_lo = self._forward(tf.constant(np.atleast_2d(np.asarray(theta, np.float64))))
        return q_max.numpy(), q_lo.numpy()

    def qfls_axis_at(self, theta) -> np.ndarray:
        """Physical QFLS axis of the network output for every row of theta, (B, 256)."""
        q_max, q_lo = self.window(theta)
        return q_lo[:, None] + (q_max - q_lo)[:, None] * np.linspace(0.0, 1.0, self._steps)[None]

    def data_qfls_at(self, theta) -> np.ndarray:
        """QFLS of the used data points on the model's calibration (q_max of theta), (B, n_points)."""
        q_max, _ = self.window(theta)
        return calibrate_qfls(self.qfls_used, self.qfls_max_analytic, q_max, self.Vt)

    def covered(self, theta) -> np.ndarray:
        """True where the predicted curve reaches every used data point, numpy (B,)."""
        import tensorflow as tf
        th = tf.constant(np.atleast_2d(np.asarray(theta, np.float64)))
        u = tf.broadcast_to(self._u_T, [th.shape[0], self.n_points])
        return np.all(self.values_at_u(th, u)[1].numpy(), axis=1)


def make_trpl_surrogate(artifacts: SurrogateArtifacts, npulse: float, Eg: float,
                        qfls_exp, interp_space: str = "log10") -> TrplSurrogate:
    """Pick the surrogate class from the network metadata (surrogate_meta.json)."""
    if artifacts.meta is not None and artifacts.meta.get("axis") == "predicted_window":
        cls = TrplSurrogateWindow
    elif artifacts.meta is not None and artifacts.meta.get("variant") == "deeponet_mlp":
        cls = TrplSurrogateDeepONet
    elif artifacts.axis == "pl_decades":
        cls = TrplSurrogateU
    else:
        cls = TrplSurrogate
    return cls(artifacts, npulse, Eg, qfls_exp, interp_space=interp_space)


def load_network(nn_dir, stem: str | None = None, **kwargs) -> SurrogateArtifacts:
    """
    Network folder -> artifacts.  PL-decade networks (from 10/2026) carry
    ``surrogate_meta.json`` with the paths of their scalers and training HDF5
    (shared by the minimal and the hybrid run); older networks use
    ``<stem>_param_scaler.joblib``, ``<stem>_output_scaler.joblib`` and
    ``<stem>.hdf5`` inside the folder.
    """
    import json
    nn_dir = Path(nn_dir)
    meta = nn_dir / "surrogate_meta.json"
    if meta.exists():
        m = json.loads(meta.read_text())
        return load_artifacts(nn_dir / "model.keras", m["param_scaler_path"], m["output_scaler_path"],
                              m["training_hdf5"], **kwargs)
    if not stem:
        raise ValueError(f"{nn_dir}: no surrogate_meta.json, so cfg['nn']['stem'] is needed")
    return load_artifacts(nn_dir / "model.keras", nn_dir / f"{stem}_param_scaler.joblib",
                          nn_dir / f"{stem}_output_scaler.joblib", nn_dir / f"{stem}.hdf5", **kwargs)
