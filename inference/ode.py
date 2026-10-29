"""
ode.py — rate-equation (ODE) evaluation of parameter sets
=========================================================

Used to check the surrogate where it matters (best fit, profile endpoints,
posterior samples; tasks T3, T7, T8).

The ODE solver is the one that generated the TRAINING data
(``network-training/trpl_module.py``, numba-accelerated) and the curve is
post-processed exactly as in ``network-training/prepare_training_data.py``:
drop non-finite tau, cap the PL at PL_DECAY_MAGNITUDE decades below its
maximum, reverse to increasing QFLS, de-duplicate, Pchip-interpolate log10
tau in QFLS.  Hence NN and ODE predictions are directly comparable.

``network-training/trpl_module.py`` is loaded under a private module name
(it would otherwise clash with other modules called ``trpl_module``).
"""
from __future__ import annotations

import importlib.util
import time

import numpy as np

from . import NETWORK_TRAINING_DIR

_ODE_MODULE = None


def _ode_module():
    """Load network-training/trpl_module.py once (as 'nt_trpl_module')."""
    global _ODE_MODULE
    if _ODE_MODULE is None:
        spec = importlib.util.spec_from_file_location(
            "nt_trpl_module", NETWORK_TRAINING_DIR / "trpl_module.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _ODE_MODULE = mod
    return _ODE_MODULE


def solve(constants: dict, npulse, Eg, krad, Nt, Et, taun, taup, T: float = 300.0):
    """
    ``trpl_module.solve_transient`` on the time span of the network's training
    data (tmin, tmax from the constants of its training HDF5).
    """
    mod = _ode_module()
    return mod.solve_transient([constants["tmin"], constants["tmax"]], npulse, Eg, constants["Nc"],
                               constants["Nv"], krad, Nt, Et, taun, taup, T)


def nominal_qfls(pl, npulse: float, Eg: float, constants: dict) -> np.ndarray:
    """
    QFLS on the axis of the measured data and of the PL-decade networks:
    2 Vt ln(n_pulse/n_i) + Vt ln(PL/PL(t=0)) — the calibration of
    ``data_utils.t_pl_to_qfls_tau`` (p0 = 0).  The physical QFLS of a sample
    with p0 > 0 is shifted by kT ln(1 + p0/n_pulse); using the same nominal
    axis for data, network and ODE keeps all comparisons on log10(PL/PL0).
    """
    Vt = constants["Vt"]
    ni = np.sqrt(constants["Nc"] * constants["Nv"]) * np.exp(-Eg / (2.0 * Vt))
    pl = np.asarray(pl, np.float64)
    with np.errstate(all="ignore"):
        return 2.0 * Vt * np.log(npulse / ni) + Vt * np.log(pl / pl[0])


def simulate_trpl(theta, prior, npulse: float, constants: dict, T: float = 300.0, axis: str = "nominal"):
    """
    ODE tr-PL curve for one theta (training-data post-processing).

    axis = "nominal" (default, for every comparison with data): QFLS in the calibration of the measured data,
           2 Vt ln(n_pulse/n_i) + Vt ln(PL/PL_max) (``nominal_qfls``), i.e. data and ODE are compared at the same
           PL decade u = log10(PL/PL_max) whatever p0 is (16/10/2026; before, the physical QFLS was compared with
           the nominally calibrated data, off by kT ln(1 + p0/n_pulse) at low injection);
    axis = "physical": the ODE's own QFLS = kT ln(n p / n_i^2) (plots on the absolute QFLS axis).
    Returns (qfls_ascending [eV], log10_tau_diff [s]); empty arrays on failure.
    """
    if axis not in ("nominal", "physical"):
        raise ValueError("axis must be 'nominal' or 'physical'")
    mod = _ode_module()
    th = np.asarray(theta, np.float64)
    phys = prior.to_physical(th[None]).iloc[0]
    K = prior.n_traps
    Nt = np.array([phys[f"Nt_{k}_cm-3"] for k in range(1, K + 1)])
    Et = np.array([phys[f"Et_{k}_eV"] for k in range(1, K + 1)])
    taun = np.array([phys[f"tau_n_{k}_s"] for k in range(1, K + 1)])
    taup = np.array([phys[f"tau_p_{k}_s"] for k in range(1, K + 1)])
    import warnings
    with np.errstate(all="ignore"), warnings.catch_warnings():
        # np.gradient on the adaptive ODE time grid warns for repeated time
        # points; the affected values are removed below (non-finite)
        warnings.simplefilter("ignore")
        out = solve(constants, npulse, prior.Eg, phys["krad_cm3s"], Nt, Et, taun, taup, T)
    qfls, tau, pl = out[4], out[5], out[8]
    with np.errstate(all="ignore"):
        log_tau = np.log10(tau)
        log_pl = np.log10(pl)
    ok = np.isfinite(log_tau) & np.isfinite(log_pl)
    qfls, log_tau, log_pl = qfls[ok], log_tau[ok], log_pl[ok]
    if len(qfls) < 2:
        return np.array([]), np.array([])
    log_pl = log_pl - np.max(log_pl)
    cap = np.nonzero(log_pl < -float(constants["PL_DECAY_MAGNITUDE"]))[0]
    if len(cap):
        qfls, log_tau, log_pl = qfls[:cap[0]], log_tau[:cap[0]], log_pl[:cap[0]]
    if axis == "nominal":
        Vt = constants.get("Vt", 1.380649e-23 * T / 1.602176634e-19)      # training HDF5: k_B T / q
        ni = np.sqrt(constants["Nc"] * constants["Nv"]) * np.exp(-prior.Eg / (2.0 * Vt))
        qfls = 2.0 * Vt * np.log(npulse / ni) + Vt * np.log(10.0) * log_pl
    q, idx = np.unique(qfls[::-1], return_index=True)
    return q, log_tau[::-1][idx]


def ode_log10_tau(theta, prior, npulse, constants, qfls_points, T: float = 300.0,
                  axis: str = "nominal") -> np.ndarray:
    """ODE log10 tau_diff at given QFLS points (NaN outside the simulated range); see ``simulate_trpl`` for axis."""
    from scipy.interpolate import PchipInterpolator
    q, lt = simulate_trpl(theta, prior, npulse, constants, T, axis=axis)
    if len(q) < 2:
        return np.full(len(qfls_points), np.nan)
    return PchipInterpolator(q, lt, extrapolate=False)(np.asarray(qfls_points, np.float64))


def ode_mse(thetas, prior, npulse, constants, qfls_points, log10_tau_obs, n_jobs: int = 8,
            T: float = 300.0) -> dict:
    """
    MSE of the ODE prediction vs observed log10 tau for many parameter sets
    (parallel over CPU cores with joblib).  Points outside the simulated QFLS
    range are ignored (``frac_covered`` reports how many were used).

    Returns dict with mse (N,), frac_covered (N,), time_s (wall) and
    time_per_solve_s (wall time x n_jobs / N, i.e. CPU time per ODE solve).
    """
    from joblib import Parallel, delayed
    thetas = np.atleast_2d(np.asarray(thetas, np.float64))
    obs = np.asarray(log10_tau_obs, np.float64)

    def one(th):
        y = ode_log10_tau(th, prior, npulse, constants, qfls_points, T)
        ok = np.isfinite(y)
        return (float(np.mean((y[ok] - obs[ok]) ** 2)) if ok.any() else np.nan, float(ok.mean()))

    _ode_module()          # compile numba functions once in the parent process
    t0 = time.perf_counter()
    res = Parallel(n_jobs=n_jobs)(delayed(one)(th) for th in thetas)
    t = time.perf_counter() - t0
    mse = np.array([r[0] for r in res])
    cov = np.array([r[1] for r in res])
    return {"mse": mse, "frac_covered": cov, "time_s": t,
            "time_per_solve_s": t * min(n_jobs, len(thetas)) / len(thetas)}


def _predict_block(problem, thetas) -> tuple[np.ndarray, np.ndarray]:
    """ODE log10 tau at problem.qfls for a block of thetas (one worker task), with the time of each solve."""
    import warnings
    warnings.filterwarnings("ignore")
    y = np.empty((len(thetas), len(problem.qfls)))
    t = np.empty(len(thetas))
    for i, th in enumerate(thetas):
        t0 = time.perf_counter()
        y[i] = ode_log10_tau(th, problem.prior, problem.npulse, problem.constants, problem.qfls, problem.T)
        t[i] = time.perf_counter() - t0
    return y, t


def ode_predictions(problem, thetas, n_jobs: int = 8, block: int = 64, verbose: bool = True) -> dict:
    """
    ODE tr-PL predictions for many parameter sets (e.g. every point of a grid),
    parallel over CPU cores with joblib.

    problem : refine.OdeFitProblem (data points, npulse, constants, T)
    thetas  : (N, D) parameter sets
    block   : parameter sets per worker task (amortises the task overhead)

    Returns dict with
      y        (N, n_points) log10 tau_diff at problem.qfls, NaN where the ODE
               curve does not reach the point (OdeFitProblem penalises those)
      t_solve  (N,) wall time of each ODE solve inside its worker [s]
      time_s   total wall time [s]
    The likelihood is NOT computed here: with the predictions stored, -log L
    for any sigma is a cheap array operation (OdeFitProblem.neg_log_lik_many).
    """
    from joblib import Parallel, delayed
    thetas = np.atleast_2d(np.asarray(thetas, np.float64))
    N = len(thetas)
    starts = list(range(0, N, block))
    _ode_module()          # compile numba functions once in the parent process
    y, t_solve = np.empty((N, len(problem.qfls))), np.empty(N)
    report_every = max(1, len(starts) // 20)
    t0 = time.perf_counter()
    gen = Parallel(n_jobs=n_jobs, return_as="generator")(
        delayed(_predict_block)(problem, thetas[s:s + block]) for s in starts)
    for k, (s, (yb, tb)) in enumerate(zip(starts, gen)):
        y[s:s + len(yb)], t_solve[s:s + len(tb)] = yb, tb
        if verbose and (k % report_every == 0 or k == len(starts) - 1):
            done, el = min(N, s + block), time.perf_counter() - t0
            print(f"  ODE: {done:,}/{N:,} parameter sets ({100 * done / N:.0f} %), {el:.0f} s, "
                  f"ETA {el / done * (N - done):.0f} s", flush=True)
    return {"y": y, "t_solve": t_solve, "time_s": time.perf_counter() - t0}


class OdeTrplObjective:
    """
    tr-PL log-likelihood evaluated with the ODE instead of the NN.

    Exposes the attributes that ``optimize.cmaes_multistart`` / ``optimize.fit``
    need (``prior``, ``dim``, ``counter``, ``np_neg_log_lik_u``), so the SAME
    CMA-ES code, start point and tolerances can be run with the rate
    equations (timing table, task T8).  Each CMA-ES population is solved in
    parallel on ``n_jobs`` CPU cores.  Points outside the simulated QFLS range
    are ignored; a curve that covers less than half of the data gets -log L = 1e30.
    """

    def __init__(self, prior, npulse, constants, qfls_points, log10_tau_obs, sigma: float,
                 n_jobs: int = 8, T: float = 300.0):
        from .benchmark import PyCounter
        self.prior = prior
        self.dim = prior.dim
        self.npulse = float(npulse)
        self.constants = constants
        self.qfls = np.asarray(qfls_points, np.float64)
        self.y_obs = np.asarray(log10_tau_obs, np.float64)
        self.sigma = float(sigma)
        self.n_jobs = int(n_jobs)
        self.T = float(T)
        self.counter = PyCounter("ode")

    def np_neg_log_lik_u(self, U) -> np.ndarray:
        import tensorflow as tf
        U = np.atleast_2d(np.asarray(U, np.float64))
        theta = self.prior.forward(tf.constant(U))[0].numpy()
        r = ode_mse(theta, self.prior, self.npulse, self.constants, self.qfls, self.y_obs,
                    n_jobs=self.n_jobs, T=self.T)
        self.counter.record(len(U))
        nll = r["mse"] / (2 * self.sigma ** 2) + np.log(self.sigma) + 0.5 * np.log(2 * np.pi)
        return np.where(np.isfinite(nll) & (r["frac_covered"] >= 0.5), nll, 1e30)


def simulate_trpl_full(theta, prior, npulse: float, constants: dict, T: float = 300.0):
    """
    Full ODE output for one parameter set as a DataFrame (for export and for
    the trap-dynamics analysis, task T9).

    Columns: time_s, pl_norm (PL / PL_max), qfls_eV (physical, kT ln(n p / n_i^2)), qfls_nominal_eV (data
    calibration, 2 Vt ln(n_pulse/n_i) + Vt ln(PL/PL_0)), tau_diff_s, n_cm3, p_cm3,
    and per trap k: nt_k_cm3, e_capture_k, e_emission_k, h_capture_k,
    h_emission_k [cm^-3 s^-1], net_e_capture_k (= e_capture - e_emission),
    net_h_capture_k, plus R_rad (k_rad n p).  Rows with non-finite values of the
    differential lifetime are removed (as for the training data).
    """
    import warnings
    import pandas as pd
    th = np.asarray(theta, np.float64)
    phys = prior.to_physical(th[None]).iloc[0]
    K = prior.n_traps
    Nt = np.array([phys[f"Nt_{k}_cm-3"] for k in range(1, K + 1)])
    Et = np.array([phys[f"Et_{k}_eV"] for k in range(1, K + 1)])
    taun = np.array([phys[f"tau_n_{k}_s"] for k in range(1, K + 1)])
    taup = np.array([phys[f"tau_p_{k}_s"] for k in range(1, K + 1)])
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        (t, n, p, nt, qfls, tau, _tn, _tp, pl, e_tr, e_de, h_tr, h_de, rrad) = solve(
            constants, npulse, prior.Eg, phys["krad_cm3s"], Nt, Et, taun, taup, T)
    df = pd.DataFrame({"time_s": t, "pl_norm": pl / np.nanmax(pl), "qfls_eV": qfls,
                       "qfls_nominal_eV": nominal_qfls(pl, npulse, prior.Eg, constants),   # data calibration
                       "tau_diff_s": tau, "n_cm3": n, "p_cm3": p, "R_rad_cm3s": rrad})
    for k in range(K):
        df[f"nt_{k + 1}_cm3"] = nt[:, k]
        df[f"e_capture_{k + 1}"] = e_tr[:, k]
        df[f"e_emission_{k + 1}"] = e_de[:, k]
        df[f"h_capture_{k + 1}"] = h_tr[:, k]
        df[f"h_emission_{k + 1}"] = h_de[:, k]
        df[f"net_e_capture_{k + 1}"] = e_tr[:, k] - e_de[:, k]
        df[f"net_h_capture_{k + 1}"] = h_tr[:, k] - h_de[:, k]
    ok = np.isfinite(df["tau_diff_s"]) & (df["tau_diff_s"] > 0)
    return df[ok].reset_index(drop=True)
