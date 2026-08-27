"Author Robin Heumann 22/01/2026"
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import time
from pathlib import Path
import seaborn as sns
from scipy.interpolate import PchipInterpolator
from scipy.integrate import trapezoid
from pymoo.algorithms.soo.nonconvex.cmaes import CMAES
from pymoo.core.problem import Problem
from pymoo.core.termination import Termination
from pymoo.optimize import minimize

import joblib
from sspl_module import solve_steady_state
from trpl_module import solve_transient
from constants import NC, NV, VT, DECAY_MAGNITUDE, QFLS_STEPS


def _compute_qfls_network(npulse: float, Eg: float) -> np.ndarray:
    """
    Build the QFLS axis used as the output grid of the neural-network TRPL surrogate.

    The grid spans from qfls_min (carrier density decayed by DECAY_MAGNITUDE decades
    below npulse) to qfls_max (carrier density equal to npulse), computed from the
    quasi-Fermi level splitting of an ideal semiconductor.

    Parameters
    ----------
    npulse : float
        Initial photo-excited carrier density at t=0 [cm⁻³].
    Eg : float
        Bandgap energy [eV].

    Returns
    -------
    np.ndarray
        Uniformly-spaced QFLS array of length QFLS_STEPS [eV], sorted ascending
        (qfls_min … qfls_max).
    """
    ni       = np.sqrt(NC * NV) * np.exp(-Eg / (2.0 * VT))
    qfls_max = 2.0 * VT * np.log(npulse / ni)
    qfls_min = qfls_max + VT * np.log(1.0 / 10**DECAY_MAGNITUDE)
    return np.linspace(qfls_min, qfls_max, QFLS_STEPS)


def _compute_G_per_sun(
    Eg: float,
    absorb_eff: float = 1.0,
    am15g_path=None,
    d = None,
) -> float:
    """
    Integrate the AM 1.5G photon flux above the bandgap Eg.

    Parameters
    ----------
    Eg : float
        Bandgap [eV]. Only photons with E ≥ Eg are counted.
    absorb_eff : float
        Absorptance pre-factor in [0, 1].  Multiplied with the integrated
        flux so that partial absorption is accounted for.  Default 1.0
        (100 % absorptance).
    am15g_path : str or Path or None
        Path to the AM 1.5G data file (two whitespace-separated columns:
        photon energy [eV] and spectral photon flux [photons cm⁻² s⁻¹ eV⁻¹]).
        Defaults to ``<package_dir>/data/AM15G.dat``.

    Returns
    -------
    float
        Above-gap absorbed photon flux at 1 sun [photons cm⁻² s⁻¹].
        Use this as the ``G_per_sun`` argument when the simulation generates
        carriers per unit *area*.  For volumetric generation (cm⁻³ s⁻¹)
        divide by the film thickness in cm.
    """
    if am15g_path is None:
        am15g_path = Path(__file__).parent / "data" / "AM15G.dat"
    if d is None:
        raise ValueError("Film thickness 'd' [cm] must be provided to compute G_per_sun from the AM 1.5G spectrum.")
    am15g = np.loadtxt(am15g_path)
    E_eV  = am15g[:, 0]        # photon energy [eV]
    phi   = am15g[:, 1]        # spectral photon flux [photons/cm²/s/eV]
    mask  = E_eV >= Eg
    if not np.any(mask):
        raise ValueError(f"No AM1.5G data points found above Eg = {Eg} eV.")
    return float(absorb_eff * trapezoid(phi[mask], E_eV[mask]) / d)


def fit_multitrap(
    df_sspl: pd.DataFrame | None,
    df_trpl: "pd.DataFrame | list[pd.DataFrame] | None",
    Eg: float,
    npulse: "float | list[float]",
    num_traps: int = 1,
    **kwargs,
) -> dict:
    """
    CMA-ES fitting for multitrap model.

    Parameters
    ----------
    df_sspl : pd.DataFrame or None
        Steady-state data with columns ['qfls', 'plqy']. Can be None if
        fit_mode='trpl'.
    df_trpl : pd.DataFrame or list of pd.DataFrame or None
        Transient data with columns ['qfls', 'tau_diff']. Can be None if
        fit_mode='sspl'. Pass a **list** of DataFrames to fit multiple TRPL
        curves simultaneously (all sharing the same trap parameters).
    Eg : float
        Bandgap [eV]
    npulse : float or list of float
        Laser-excited carrier density [photons/cm³] at t=0.  When *df_trpl*
        is a list, supply one value per curve; a scalar is broadcast to all.
    num_traps : int
        Number of trap levels
    **kwargs : optional parameters
        fit_mode : str
            Which dataset(s) to fit. Options:
                'both'  - fit SSPL and TRPL simultaneously (default)
                'sspl'  - fit steady-state PLQY only
                'trpl'  - fit time-resolved differential lifetime only
        error_type : str
            Error metric used in the objective function. Options:
                'rmse'          - root mean squared error in log space (default)
                'mae'           - mean absolute error in log space
                'integral_mae'  - integral of |error| normalised by x-range
                'integral_rmse' - sqrt of integral of error^2 normalised by x-range
            For SSPL the integration variable depends on sspl_residual_type.
            For TRPL the integration variable is QFLS [eV].
        sspl_residual_type : str
            Which axes to use when computing SSPL residuals. Only active when
            fit_mode is 'sspl' or 'both'. Each type requires specific columns
            in df_sspl. Options:
                'qfls_plqy'           - PLQY(y) vs QFLS(x) — default;
                                        requires columns: 'qfls', 'plqy'
                'intensity_suns_plqy' - PLQY(y) vs intensity in suns(x);
                                        requires columns: 'intensity_suns', 'plqy'
                'intensity_suns_qfls' - QFLS(y) vs intensity in suns(x);
                                        requires columns: 'intensity_suns', 'qfls'
                'g_plqy'              - PLQY(y) vs generation rate(x) [cm⁻³s⁻¹];
                                        requires columns: 'generation_rate', 'plqy'
                'g_qfls'              - QFLS(y) vs generation rate(x) [cm⁻³s⁻¹];
                                        requires columns: 'generation_rate', 'qfls'
        n_interp_sspl : int or None
            Number of uniformly-spaced QFLS points on which both the
            experimental PLQY and the simulated PLQY are interpolated before
            computing the error. None (default) evaluates the simulation at the
            original experimental QFLS points.
        n_interp_trpl : int or None
            Same as n_interp_sspl but for the TRPL differential lifetime.
        qfls_range_sspl : tuple (float, float) or None
            (qfls_min, qfls_max) [eV] — only SSPL data within this QFLS
            window contribute to the error. None (default) uses all data.
            When used together with n_interp_sspl the interpolation grid is
            restricted to this window.
        qfls_range_trpl : tuple (float, float) or None
            Same as qfls_range_sspl but for the TRPL experiment.
        absorb_eff : float
            Absorptance pre-factor in [0, 1] used to compute ``G_per_sun``
            from the AM 1.5G spectrum (∫Φ_AM15G dE × absorb_eff).  Default
            1.0 (100 % absorptance).  Ignored when ``G_per_sun`` is given
            explicitly.
        G_per_sun : float or None
            Override the generation rate at 1 sun [photons cm⁻² s⁻¹].
            When ``None`` (default), computed automatically from the AM 1.5G
            spectrum using ``absorb_eff`` and the bandgap ``Eg``.
        Nc, Nv, T, Ngrid, sigma, maxfevals,
        Et_bounds, Nt_bounds, tau_n_bounds, tau_p_bounds, krad_bounds,
        w_sspl, w_trpl, trials, x0_multitrap, tspan

    Returns
    -------
    results : dict
        Fitting results and fitted parameters. Keys that are not relevant
        to the chosen fit_mode are set to None.
        When multiple TRPL curves are fitted, ``trpl_sim``, ``trpl_fit``,
        ``trpl_npulse_list``, and ``error_trpl_per_curve`` are lists with one
        entry per curve.
    """

    # Default parameters
    Nc = kwargs.get('Nc', 2.2e18)
    Nv = kwargs.get('Nv', 2.2e18)
    T = kwargs.get('T', 300)
    x0_multitrap = kwargs.get('x0_multitrap', None)
    absorb_eff = kwargs.get('absorb_eff', 1.0)
    Ngrid = kwargs.get('Ngrid', 1000)
    sigma = kwargs.get('sigma', 0.25)
    maxfevals = kwargs.get('maxfevals', 50000)
    _G_per_sun_explicit = kwargs.get('G_per_sun', None)
    d_cm = kwargs.get('d_cm', None)   # film thickness [cm] — required for intensity_suns_* types
    if _G_per_sun_explicit is not None:
        G_per_sun = float(_G_per_sun_explicit)
    else:
        G_per_sun = _compute_G_per_sun(Eg, absorb_eff, d=d_cm)
    fit_mode = kwargs.get('fit_mode', 'both')   # 'sspl', 'trpl', or 'both'
    error_type = kwargs.get('error_type', 'rmse')  # 'rmse', 'mae', 'integral_mae', 'integral_rmse'
    sspl_residual_type = kwargs.get('sspl_residual_type', 'qfls_plqy')
    
    ftarget = kwargs.get('ftarget', 1e-2) # target error for termination
    period = kwargs.get('termination_period', 50) # generations between termination checks
    # List of 0-based trap indices for which Nt > n1 = Nc*exp(-Et/kT) is enforced
    # as a pymoo inequality constraint (G[j] = n1[j] - Nt[j] <= 0).
    # Pass a list, e.g. photodoping=[0] or photodoping=[0, 1].  [] disables.
    _pd_raw = kwargs.get('photodoping', [])
    if isinstance(_pd_raw, (int, np.integer)):
        _pd_raw = [int(_pd_raw)]
    photodoping_traps = [int(t) for t in _pd_raw]
    for _ti in photodoping_traps:
        if not (0 <= _ti < num_traps):
            raise ValueError(
                f"photodoping trap index {_ti} out of range [0, {num_traps - 1}]"
            )
    if fit_mode not in ('sspl', 'trpl', 'both'):
        raise ValueError(f"fit_mode must be 'sspl', 'trpl', or 'both', got '{fit_mode}'")
    if error_type not in ('rmse', 'mae', 'integral_mae', 'integral_rmse'):
        raise ValueError(f"error_type must be 'rmse', 'mae', 'integral_mae', or 'integral_rmse', got '{error_type}'")
    _VALID_SSPL_RESIDUAL_TYPES = (
        'qfls_plqy', 'intensity_suns_plqy', 'intensity_suns_qfls', 'g_plqy', 'g_qfls')
    if sspl_residual_type not in _VALID_SSPL_RESIDUAL_TYPES:
        raise ValueError(
            f"sspl_residual_type must be one of {_VALID_SSPL_RESIDUAL_TYPES}, "
            f"got '{sspl_residual_type}'")

    # G_per_sun from _compute_G_per_sun is an *areal* photon flux [photons cm⁻² s⁻¹].
    # solve_steady_state works with *volumetric* quantities [cm⁻³ s⁻¹].
    # For intensity_suns_* types the conversion Rtot → suns therefore requires
    # dividing by the volumetric generation rate at 1 sun: G_vol = G_per_sun / d_cm.
    _intensity_suns_types = ('intensity_suns_plqy', 'intensity_suns_qfls')
    if sspl_residual_type in _intensity_suns_types and fit_mode in ('sspl', 'both'):
        if d_cm is None:
            raise ValueError(
                f"sspl_residual_type='{sspl_residual_type}' requires the film thickness "
                f"'d_cm' [cm] to convert volumetric recombination rates to intensity in "
                f"suns.  Pass it as a keyword argument, e.g. d_cm=240e-7.")
    

    # Bounds — accept either a single (lo, hi) tuple (broadcast to all traps)
    # or a list of (lo, hi) tuples of length num_traps (per-trap bounds).
    def _to_per_trap(raw, K):
        """
        Normalise a bounds argument so that each trap has its own (lo, hi) tuple.

        Accepts either a single tuple that is broadcast to all traps, or a list of
        K tuples providing per-trap bounds individually.

        Parameters
        ----------
        raw : tuple or list of tuples
            Either a single (lo, hi) bounds pair applied to every trap, or a list of
            K (lo, hi) pairs where K == num_traps.
        K : int
            Number of trap levels (num_traps).

        Returns
        -------
        list of tuple
            List of K (lo, hi) bounds tuples, one per trap.

        Raises
        ------
        ValueError
            If *raw* is a list whose length differs from K.
        """
        if isinstance(raw, list):
            if len(raw) != K:
                raise ValueError(
                    f"Expected {K} bound tuples, got {len(raw)}")
            return list(raw)
        return [raw] * K

    Et_bounds_list    = _to_per_trap(kwargs.get('Et_bounds',    (0.5*Eg, Eg)),  num_traps)
    Nt_bounds_list    = _to_per_trap(kwargs.get('Nt_bounds',    (1e12, 1e20)),  num_traps)
    tau_n_bounds_list = _to_per_trap(kwargs.get('tau_n_bounds', (1e-11, 1e-4)), num_traps)
    tau_p_bounds_list = _to_per_trap(kwargs.get('tau_p_bounds', (1e-11, 1e-4)), num_traps)
    krad_bounds       = kwargs.get('krad_bounds', (1e-12, 1e-9))

    # For backward-compat references in the rest of the file (single-trap case)
    Et_bounds    = Et_bounds_list[0]
    Nt_bounds    = Nt_bounds_list[0]
    tau_n_bounds = tau_n_bounds_list[0]
    tau_p_bounds = tau_p_bounds_list[0]

    # Fixed parameters
    # Pass as fixed_params={trap_idx: {param: value, ...}, 'krad': value}
    # trap_idx is 0-based; param names are 'Et', 'Nt', 'tau_n', 'tau_p'.
    # Fixed parameters are removed from the optimisation search space entirely.
    # Example: {0: {'Et': 0.65, 'tau_n': 1e-8}, 'krad': 5e-11}
    fixed_params_raw = kwargs.get('fixed_params', {})

    _TRAP_PARAM_NAMES = ['Et', 'Nt', 'tau_n', 'tau_p']
    _fixed_phys: dict[int, float] = {}   # full_index -> fixed physical value

    for _key, _val in fixed_params_raw.items():
        if _key == 'krad':
            _fixed_phys[4 * num_traps] = float(_val)
        else:
            _ti = int(_key)
            if not (0 <= _ti < num_traps):
                raise ValueError(
                    f"fixed_params key {_key!r} out of range [0, {num_traps - 1}]")
            for _pname, _pval in _val.items():
                if _pname not in _TRAP_PARAM_NAMES:
                    raise ValueError(
                        f"Unknown parameter {_pname!r}, must be one of {_TRAP_PARAM_NAMES}")
                _fixed_phys[4 * _ti + _TRAP_PARAM_NAMES.index(_pname)] = float(_pval)

    _all_idx  = list(range(4 * num_traps + 1))
    _free_idx = [i for i in _all_idx if i not in _fixed_phys]
    _n_free   = len(_free_idx)

    # Weights (only relevant for fit_mode='both')
    w_sspl = kwargs.get('w_sspl', 1.0)
    w_trpl = kwargs.get('w_trpl', 1.0)

    n_fits = kwargs.get('trials', 1)

    # Optional neural-network surrogate for TRPL (replaces solve_transient)
    nn_model         = kwargs.get('nn_model', None)
    nn_input_scaler  = kwargs.get('nn_input_scaler', None)
    nn_output_scaler = kwargs.get('nn_output_scaler', None)
    # Canonical QFLS axis loaded from the training HDF5 via load_nn_artifacts.
    # When provided it is used directly instead of recomputing from constants.py.
    nn_qfls_axis     = kwargs.get('nn_qfls_axis', None)
    if nn_model is not None and fit_mode == 'sspl':
        raise ValueError("nn_model has no effect when fit_mode='sspl'. "
                         "Set fit_mode='trpl' or 'both'.")

    # --- Error metric helper ---
    def compute_error(
        log_residuals: np.ndarray,
        x_vals: np.ndarray,
        etype: str,
    ) -> float:
        """
        Compute a scalar error from log-space residuals.

        All errors operate in log10 space so that decades of PLQY or lifetime
        are weighted equally regardless of absolute magnitude.

        Parameters
        ----------
        log_residuals : np.ndarray
            Element-wise log10(sim) - log10(data).  Must be the same length
            as x_vals and already sorted in ascending x order.
        x_vals : np.ndarray
            x-axis values corresponding to each residual, sorted ascending.
            Only used for the integral error types.
        etype : str
            One of:
            - 'rmse'          : sqrt( mean( residuals^2 ) )
            - 'mae'           : mean( |residuals| )
            - 'integral_mae'  : ∫|residuals| dx / Δx  (area-normalised)
            - 'integral_rmse' : sqrt( ∫residuals^2 dx / Δx )  (area-normalised)

        Returns
        -------
        float
            Scalar error value.
        """
        abs_res = np.abs(log_residuals)
        if etype == 'rmse':
            return np.sqrt(np.mean(log_residuals**2))
        elif etype == 'mae':
            return np.mean(abs_res)
        elif etype == 'integral_mae':
            # Fall back to pointwise MAE if the range is degenerate
            if len(x_vals) < 2 or (x_vals[-1] - x_vals[0]) == 0:
                return np.mean(abs_res)
            return trapezoid(abs_res, x_vals) / (x_vals[-1] - x_vals[0])
        else:  # 'integral_rmse'
            if len(x_vals) < 2 or (x_vals[-1] - x_vals[0]) == 0:
                return np.sqrt(np.mean(log_residuals**2))
            return np.sqrt(trapezoid(log_residuals**2, x_vals) / (x_vals[-1] - x_vals[0]))

    # Number of points for the common comparison grid.
    # When set, both experiment and simulation are interpolated onto a uniform
    # QFLS grid of this size before computing the error.
    # None (default) keeps the original behaviour: evaluate sim at data points.
    n_interp_sspl = kwargs.get('n_interp_sspl', None)
    n_interp_trpl = kwargs.get('n_interp_trpl', None)

    # QFLS range over which the error is computed, e.g. (0.9, 1.15) [eV].
    # None (default) uses the full data range.
    # Applied after cleaning and sorting; the experimental interpolator (for
    # n_interp mode) is always built from the full cleaned data so that
    # interpolation accuracy is not degraded near the range boundaries.
    qfls_range_sspl = kwargs.get('qfls_range_sspl', None)
    qfls_range_trpl = kwargs.get('qfls_range_trpl', None)

    # --- Extract, clean, sort, and optionally range-filter data ---
    # SSPL — each residual type uses a distinct pair of columns
    if fit_mode in ('sspl', 'both'):
        if df_sspl is None:
            raise ValueError("df_sspl must be provided when fit_mode is 'sspl' or 'both'")

        if sspl_residual_type == 'qfls_plqy':
            # Required columns: qfls, plqy
            qfls_raw  = df_sspl['qfls'].values
            plqy_raw  = df_sspl['plqy'].values
            mask_sspl = np.isfinite(qfls_raw) & np.isfinite(plqy_raw) & (plqy_raw > 0)
            x_raw = qfls_raw[mask_sspl]
            y_raw = plqy_raw[mask_sspl]
            _ss_order    = np.argsort(x_raw)
            sspl_x_data  = x_raw[_ss_order]
            sspl_y_data  = y_raw[_ss_order]
            intensity_suns_data  = None
            generation_rate_data = None

        elif sspl_residual_type in ('intensity_suns_plqy', 'intensity_suns_qfls'):
            # Required columns: intensity_suns + plqy or qfls
            if 'intensity_suns' not in df_sspl.columns:
                raise ValueError(
                    f"df_sspl must contain an 'intensity_suns' column when "
                    f"sspl_residual_type='{sspl_residual_type}'")
            intensity_suns_raw = df_sspl['intensity_suns'].values
            mask_sspl = np.isfinite(intensity_suns_raw) & (intensity_suns_raw > 0)
            if sspl_residual_type == 'intensity_suns_plqy':
                plqy_raw  = df_sspl['plqy'].values
                mask_sspl = mask_sspl & np.isfinite(plqy_raw) & (plqy_raw > 0)
                x_raw = intensity_suns_raw[mask_sspl]
                y_raw = plqy_raw[mask_sspl]
            else:  # intensity_suns_qfls
                qfls_raw  = df_sspl['qfls'].values
                mask_sspl = mask_sspl & np.isfinite(qfls_raw)
                x_raw = intensity_suns_raw[mask_sspl]
                y_raw = qfls_raw[mask_sspl]
            _ss_order           = np.argsort(x_raw)
            sspl_x_data         = x_raw[_ss_order]
            sspl_y_data         = y_raw[_ss_order]
            intensity_suns_data = sspl_x_data
            generation_rate_data = None

        else:  # g_plqy or g_qfls — generation rate already in cm⁻³s⁻¹, no conversion needed
            if 'generation_rate' not in df_sspl.columns:
                raise ValueError(
                    f"df_sspl must contain a 'generation_rate' column when "
                    f"sspl_residual_type='{sspl_residual_type}'")
            generation_rate_raw = df_sspl['generation_rate'].values
            mask_sspl = np.isfinite(generation_rate_raw) & (generation_rate_raw > 0)
            if sspl_residual_type == 'g_plqy':
                plqy_raw  = df_sspl['plqy'].values
                mask_sspl = mask_sspl & np.isfinite(plqy_raw) & (plqy_raw > 0)
                x_raw = generation_rate_raw[mask_sspl]
                y_raw = plqy_raw[mask_sspl]
            else:  # g_qfls
                qfls_raw  = df_sspl['qfls'].values
                mask_sspl = mask_sspl & np.isfinite(qfls_raw)
                x_raw = generation_rate_raw[mask_sspl]
                y_raw = qfls_raw[mask_sspl]
            _ss_order            = np.argsort(x_raw)
            sspl_x_data          = x_raw[_ss_order]
            sspl_y_data          = y_raw[_ss_order]
            generation_rate_data = sspl_x_data
            intensity_suns_data  = None

        # Build interpolator from the FULL cleaned+sorted data (better boundary accuracy).
        # x-axis: log10 for intensity/generation types (log-distributed).
        # y-axis: log10 for PLQY types (PLQY spans orders of magnitude).
        _sspl_log_x = sspl_residual_type != 'qfls_plqy'
        _sspl_log_y = sspl_residual_type not in ('intensity_suns_qfls', 'g_qfls')
        if n_interp_sspl is not None and len(sspl_x_data) > 1:
            _xi_exp = np.log10(sspl_x_data) if _sspl_log_x else sspl_x_data
            _yi_exp = np.log10(sspl_y_data) if _sspl_log_y else sspl_y_data
            _sspl_exp_interp = PchipInterpolator(_xi_exp, _yi_exp, extrapolate=False)
        else:
            _sspl_exp_interp = None

        # Apply user-specified QFLS range (only meaningful for qfls_plqy)
        if qfls_range_sspl is not None and sspl_residual_type == 'qfls_plqy':
            _rng_mask   = ((sspl_x_data >= qfls_range_sspl[0]) &
                           (sspl_x_data <= qfls_range_sspl[1]))
            sspl_x_data = sspl_x_data[_rng_mask]
            sspl_y_data = sspl_y_data[_rng_mask]

        n_sspl = len(sspl_x_data)
    else:
        intensity_suns_data  = None
        generation_rate_data = None
        sspl_x_data          = np.array([])
        sspl_y_data          = np.array([])
        n_sspl = 0
        _sspl_exp_interp = None

    # ── Normalise df_trpl / npulse to lists ────────────────────────────────
    if fit_mode in ('trpl', 'both'):
        if df_trpl is None:
            raise ValueError("df_trpl must be provided when fit_mode is 'trpl' or 'both'")
        if isinstance(df_trpl, pd.DataFrame):
            df_trpl_list = [df_trpl]
        else:
            df_trpl_list = list(df_trpl)
        if np.isscalar(npulse):
            npulse_list = [float(npulse)] * len(df_trpl_list)
        else:
            npulse_list = [float(v) for v in npulse]
        if len(npulse_list) != len(df_trpl_list):
            raise ValueError(
                f"npulse must be a scalar or a list with the same length as "
                f"df_trpl (got {len(npulse_list)} npulse values for "
                f"{len(df_trpl_list)} TRPL curve(s))."
            )
    else:
        df_trpl_list = []
        npulse_list  = []

    # TRPL — process each curve independently
    if fit_mode in ('trpl', 'both'):
        trpl_curves = []
        for df_tr_i, np_i in zip(df_trpl_list, npulse_list):
            qfls_raw_tr = df_tr_i['qfls'].values
            tau_raw     = df_tr_i['tau_diff'].values
            mask_i = np.isfinite(qfls_raw_tr) & np.isfinite(tau_raw) & (tau_raw > 0)
            q_tr = qfls_raw_tr[mask_i]
            t_tr = tau_raw[mask_i]
            _order = np.argsort(q_tr)
            q_tr = q_tr[_order];  t_tr = t_tr[_order]
            # Interpolator built from FULL cleaned data for n_interp mode
            _interp_i = (PchipInterpolator(q_tr, t_tr,extrapolate=False)
                         if n_interp_trpl is not None and len(q_tr) > 1
                         else None)
            # Apply user-specified QFLS range
            if qfls_range_trpl is not None:
                _rng = ((q_tr >= qfls_range_trpl[0]) &
                        (q_tr <= qfls_range_trpl[1]))
                q_tr = q_tr[_rng];  t_tr = t_tr[_rng]
            curve_entry = {
                'qfls': q_tr, 'tau': t_tr,
                'npulse': np_i, 'interp': _interp_i,
            }
            if nn_model is not None:
                # Use the axis loaded from the training HDF5 when available;
                # fall back to recomputing from constants.py otherwise.
                if nn_qfls_axis is not None:
                    curve_entry['qfls_nn'] = nn_qfls_axis
                else:
                    curve_entry['qfls_nn'] = _compute_qfls_network(np_i, Eg)
            trpl_curves.append(curve_entry)
        n_trpl_total = sum(len(c['qfls']) for c in trpl_curves)
    else:
        trpl_curves  = []
        n_trpl_total = 0

    n_total = n_sspl + n_trpl_total

    print("="*70)
    print("MULTITRAP MODEL FITTING - CMA-ES OPTIMIZATION")
    if nn_model is not None:
        print("TRPL surrogate  : NEURAL NETWORK")
    else:
        print("TRPL surrogate  : ODE SOLVER")
    print("="*70)
    print(f"Number of traps : {num_traps}")
    print(f"Free parameters : {_n_free}  (of {4 * num_traps + 1} total)")
    if _fixed_phys:
        _PNAME_SHORT = {0: 'Et', 1: 'Nt', 2: 'τn', 3: 'τp'}
        for _fi, _fv in sorted(_fixed_phys.items()):
            if _fi == 4 * num_traps:
                print(f"  Fixed: krad = {_fv:.3e} cm³/s")
            else:
                _ti, _si = divmod(_fi, 4)
                print(f"  Fixed: trap {_ti}, {_PNAME_SHORT[_si]} = {_fv:.4g}")
    print(f"Fit mode        : {fit_mode}")
    print(f"Error type      : {error_type}")
    if fit_mode in ('sspl', 'both'):
        print(f"SSPL residuals  : {sspl_residual_type}")
        if _G_per_sun_explicit is not None:
            print(f"G_per_sun       : {G_per_sun:.4e} photons/cm²/s  (user-supplied)")
        else:
            print(f"G_per_sun       : {G_per_sun:.4e} photons/cm²/s  "
                  f"(AM1.5G × absorb_eff={absorb_eff:.3f}, Eg={Eg:.3f} eV)")
        if sspl_residual_type in _intensity_suns_types:
            print(f"d_cm            : {d_cm:.4e} cm")
            print(f"G_vol_per_sun   : {G_per_sun:.4e} cm⁻³/s  (= G_per_sun / d_cm)")
    if n_interp_sspl is not None:
        print(f"SSPL interp pts : {n_interp_sspl}")
    if n_interp_trpl is not None:
        print(f"TRPL interp pts : {n_interp_trpl}")
    if qfls_range_sspl is not None:
        print(f"SSPL QFLS range : {qfls_range_sspl[0]:.3f} – {qfls_range_sspl[1]:.3f} eV")
    if qfls_range_trpl is not None:
        print(f"TRPL QFLS range : {qfls_range_trpl[0]:.3f} – {qfls_range_trpl[1]:.3f} eV")
    print(f"SSPL data points: {n_sspl}")
    if fit_mode in ('trpl', 'both'):
        print(f"TRPL curves     : {len(trpl_curves)}")
        print(f"TRPL data points: {n_trpl_total} total")
        for i_c, c in enumerate(trpl_curves):
            print(f"  Curve {i_c+1}: {len(c['qfls'])} pts, "
                  f"npulse = {c['npulse']:.3e} cm⁻³")
    print(f"Total data points: {n_total}")
    if photodoping_traps:
        _kT_info = 8.617333e-5 * T
        print(f"Photodoping constraint: Nt > n1 = Nc·exp(-Et/kT) for trap indices {photodoping_traps}")
        for _ti in photodoping_traps:
            _Et_mid = 0.5 * (Et_bounds_list[_ti][0] + Et_bounds_list[_ti][1])
            print(f"  Trap {_ti+1}: n1 at Et_mid={_Et_mid:.3f} eV → {Nc * np.exp(-(Eg - _Et_mid) / _kT_info):.2e} cm⁻³")
    print("="*70)
    
    # --- Parameter normalisation / denormalisation ---
    # CMA-ES works best in a bounded, unit-hypercube search space.
    # Et is mapped linearly; Nt, tau_n, tau_p, and krad are mapped in
    # log10 space so that CMA-ES explores multiplicative changes equally.
    # The parameter vector layout is:
    #   [Et_0, Nt_0, tau_n_0, tau_p_0,  Et_1, ...,  krad]
    #   (4 values per trap, krad always last)

    def normalize_params(
        Et: np.ndarray,
        Nt: np.ndarray,
        tau_n: np.ndarray,
        tau_p: np.ndarray,
        krad_val: float,
    ) -> np.ndarray:
        """
        Map physical parameters to the unit hypercube, returning only free dims.

        Fixed parameters (defined via ``fixed_params`` kwarg) are excluded from
        the returned vector so the optimiser never sees them.  Per-trap bounds
        are used when available.

        Returns
        -------
        np.ndarray
            Normalised free-parameter vector, shape (_n_free,), clipped to [0, 1].
        """
        K = len(Et)
        full_norm = np.zeros(4 * K + 1)
        for i in range(K):
            idx = i * 4
            lo, hi = Et_bounds_list[i]
            full_norm[idx]   = (Et[i] - lo) / (hi - lo)
            lo, hi = Nt_bounds_list[i]
            full_norm[idx+1] = (np.log10(Nt[i])    - np.log10(lo)) / (np.log10(hi) - np.log10(lo))
            lo, hi = tau_n_bounds_list[i]
            full_norm[idx+2] = (np.log10(tau_n[i]) - np.log10(lo)) / (np.log10(hi) - np.log10(lo))
            lo, hi = tau_p_bounds_list[i]
            full_norm[idx+3] = (np.log10(tau_p[i]) - np.log10(lo)) / (np.log10(hi) - np.log10(lo))
        lo, hi = krad_bounds
        full_norm[-1] = (np.log10(krad_val) - np.log10(lo)) / (np.log10(hi) - np.log10(lo))
        return np.clip(full_norm[_free_idx], 0, 1)

    def denormalize_params(
        x: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Map a reduced normalised vector back to physical units.

        ``x`` contains only the *free* parameters (_n_free values).  Fixed
        parameters are inserted from ``_fixed_phys`` so the returned arrays
        always have shape (num_traps,).

        Parameters
        ----------
        x : np.ndarray
            Normalised free-parameter vector in [0, 1]^_n_free.

        Returns
        -------
        Et      : trap energies [eV], shape (num_traps,)
        Nt      : trap densities [cm^-3], shape (num_traps,)
        tau_n   : electron SRH lifetimes [s], shape (num_traps,)
        tau_p   : hole SRH lifetimes [s], shape (num_traps,)
        krad    : radiative recombination coefficient [cm^3/s]
        """
        K = num_traps
        # Expand reduced vector into full-length normalised vector
        full_norm = np.zeros(4 * K + 1)
        for k, full_i in enumerate(_free_idx):
            full_norm[full_i] = x[k]

        Et    = np.zeros(K)
        Nt    = np.zeros(K)
        tau_n = np.zeros(K)
        tau_p = np.zeros(K)
        for i in range(K):
            idx = i * 4
            if idx in _fixed_phys:
                Et[i] = _fixed_phys[idx]
            else:
                lo, hi = Et_bounds_list[i]
                Et[i] = lo + np.clip(full_norm[idx], 0, 1) * (hi - lo)

            if idx+1 in _fixed_phys:
                Nt[i] = _fixed_phys[idx+1]
            else:
                lo, hi = Nt_bounds_list[i]
                Nt[i] = 10**(np.log10(lo) + np.clip(full_norm[idx+1], 0, 1) * (np.log10(hi) - np.log10(lo)))

            if idx+2 in _fixed_phys:
                tau_n[i] = _fixed_phys[idx+2]
            else:
                lo, hi = tau_n_bounds_list[i]
                tau_n[i] = 10**(np.log10(lo) + np.clip(full_norm[idx+2], 0, 1) * (np.log10(hi) - np.log10(lo)))

            if idx+3 in _fixed_phys:
                tau_p[i] = _fixed_phys[idx+3]
            else:
                lo, hi = tau_p_bounds_list[i]
                tau_p[i] = 10**(np.log10(lo) + np.clip(full_norm[idx+3], 0, 1) * (np.log10(hi) - np.log10(lo)))

        krad_full_i = 4 * K
        if krad_full_i in _fixed_phys:
            krad_val = _fixed_phys[krad_full_i]
        else:
            lo, hi = krad_bounds
            krad_val = 10**(np.log10(lo) + np.clip(full_norm[krad_full_i], 0, 1) * (np.log10(hi) - np.log10(lo)))

        return Et, Nt, tau_n, tau_p, krad_val

    # --- NN surrogate helper (only used when nn_model is provided) ---
    def _predict_nn(Et, Nt, tau_n, tau_p, krad_val, npulse_val):
        """
        Predict the differential TRPL lifetime curve using the neural-network surrogate.

        Assembles the feature vector in the order expected by the trained model, applies
        the input scaler, runs inference, and inverts the output scaler to recover
        physical differential lifetimes.

        Parameters
        ----------
        Et : np.ndarray
            Trap energy levels [eV], shape (num_traps,).  Each value is the energy
            measured from the valence band (0 … Eg).
        Nt : np.ndarray
            Trap densities [cm⁻³], shape (num_traps,).
        tau_n : np.ndarray
            Electron SRH capture lifetimes [s], shape (num_traps,).
        tau_p : np.ndarray
            Hole SRH capture lifetimes [s], shape (num_traps,).
        krad_val : float
            Radiative bimolecular recombination coefficient [cm³/s].
        npulse_val : float
            Initial photo-excited carrier density for this TRPL curve [cm⁻³].

        Returns
        -------
        np.ndarray
            Differential lifetime τ_diff [s] on the fixed NN output QFLS grid
            (QFLS_STEPS points).  Must be combined with the corresponding QFLS axis
            (stored in each curve_entry['qfls_nn']) for interpolation.
        """
        # Feature order must match training: log_n_pulse, Eg_eV, log_krad,
        # then per-trap: log_Ntrap, DeltaEtrap (= Et/Eg), log_taun, log_taup.
        # Training stored Eg as a linear value (Eg_eV), not log10(Eg).
        row = [np.log10(npulse_val), Eg, np.log10(krad_val)]
        for _i in range(num_traps):
            row.extend([np.log10(Nt[_i]), Et[_i] / Eg,
                        np.log10(tau_n[_i]), np.log10(tau_p[_i])])
        x_nn     = np.array([row])
        x_sc     = nn_input_scaler.transform(x_nn)
        y_sc     = nn_model.predict(x_sc, verbose=0)
        log_tau  = nn_output_scaler.inverse_transform(y_sc)[0]
        return 10**log_tau   # tau_diff [s], shape (QFLS_STEPS,)

    # --- Objective function ---
    def objective(x: np.ndarray) -> float:
        """
        Evaluate the combined fitting error for a single parameter vector.

        The function is intentionally broad-catching: any exception during
        the physics simulation returns a large penalty (1e6) so that CMA-ES
        can continue without crashing.

        Parameters
        ----------
        x : np.ndarray
            Normalised parameter vector in [0, 1]^n_var.

        Returns
        -------
        float
            Scalar error. Lower is better. Returns a large penalty value on
            failure so CMA-ES can continue.
        """
        try:
            Et, Nt, tau_n, tau_p, krad_val = denormalize_params(x)

            if photodoping_traps:
                _kT_eV = 8.617333e-5 * T
                constraint_penalty = 0.0
                for _ti in photodoping_traps:
                    n1_i = Nc * np.exp(-(Eg - Et[_ti]) / _kT_eV)
                    if Nt[_ti] < n1_i:
                        constraint_penalty += np.log10(n1_i / Nt[_ti]) ** 2
                if constraint_penalty > 0.0:
                    return 1e4 * constraint_penalty

            if nn_model is not None and num_traps > 1:
                ordering_penalty = 0.0
                for _i in range(num_traps - 1):
                    if Et[_i] >= Et[_i + 1]:
                        # constant base ensures any violation dominates the fit error
                        ordering_penalty += 1.0 + ((Et[_i] - Et[_i + 1]) / Eg) ** 2
                if ordering_penalty > 0.0:
                    return 1e4 * ordering_penalty

            error_sspl = 0.0
            error_trpl = 0.0

            # ---- SSPL ----
            if fit_mode in ('sspl', 'both'):
                result_ss = solve_steady_state(Ngrid, Eg, Nc, Nv, krad_val, Nt, Et, tau_n, tau_p, T)
                if not isinstance(result_ss, tuple) or len(result_ss) != 16:
                    return 1e10
                _, _, _, Rtot_ss_obj, _, _, QFLS_ss, PLQY_sim_obj = result_ss[:8]
                Qfls_ss_arr_obj = QFLS_ss.flatten()
                PLQY_sim_obj    = PLQY_sim_obj.flatten()

                # Build sim interpolator for the active residual type
                _qfls_y_types = ('intensity_suns_qfls', 'g_qfls')
                _g_types      = ('g_plqy', 'g_qfls')
                if sspl_residual_type == 'qfls_plqy':
                    sim_x_arr = Qfls_ss_arr_obj
                    sim_y_arr = PLQY_sim_obj
                elif sspl_residual_type in ('intensity_suns_plqy', 'intensity_suns_qfls'):
                    _G_sim = Rtot_ss_obj.flatten() / G_per_sun
                    _sort  = np.argsort(_G_sim)
                    sim_x_arr = _G_sim[_sort]
                    sim_y_arr = (PLQY_sim_obj[_sort] if sspl_residual_type == 'intensity_suns_plqy'
                                 else Qfls_ss_arr_obj[_sort])
                else:  # g_plqy or g_qfls — Rtot is already in cm⁻³s⁻¹
                    _G_sim = Rtot_ss_obj.flatten()
                    _sort  = np.argsort(_G_sim)
                    sim_x_arr = _G_sim[_sort]
                    sim_y_arr = (PLQY_sim_obj[_sort] if sspl_residual_type == 'g_plqy'
                                 else Qfls_ss_arr_obj[_sort])

                # True when QFLS is on y-axis (linear residual); False → PLQY (log10 residual)
                _is_qfls_y = sspl_residual_type in _qfls_y_types
                # Log10 x-axis for intensity/generation types; log10 y-axis for PLQY types
                _log_x = sspl_residual_type != 'qfls_plqy'
                _log_y = not _is_qfls_y
                # Build sim interpolator in log10 space where data is log-distributed.
                # Residuals become y_sim - y_exp in the transformed space throughout.
                _sim_xi = np.log10(sim_x_arr) if _log_x else sim_x_arr
                _sim_yi = np.log10(sim_y_arr) if _log_y else sim_y_arr
                sim_ss_interp = PchipInterpolator(_sim_xi, _sim_yi, extrapolate=False)
                # Clip bounds in interpolator output space (log10 for PLQY, linear for QFLS)
                _y_lo_clip = -12.0 if _log_y else 1e-12
                _y_hi_clip = np.log10(1 - 1e-12) if _log_y else None

                if n_interp_sspl is not None and _sspl_exp_interp is not None:
                    x_lo = max(sspl_x_data[0], sim_x_arr.min())
                    x_hi = min(sspl_x_data[-1], sim_x_arr.max())
                    if x_lo >= x_hi:
                        return 1e10
                    if _log_x and x_lo > 0:
                        x_grid = np.logspace(np.log10(x_lo), np.log10(x_hi), n_interp_sspl)
                    else:
                        x_grid = np.linspace(x_lo, x_hi, n_interp_sspl)
                    _x_eval = np.log10(x_grid) if _log_x else x_grid
                    y_exp  = np.clip(_sspl_exp_interp(_x_eval), _y_lo_clip, _y_hi_clip)
                    y_sim  = np.clip(sim_ss_interp(_x_eval),    _y_lo_clip, _y_hi_clip)
                    res_sspl = y_sim - y_exp  # log10(sim)-log10(exp) for PLQY; eV diff for QFLS
                    _x_err = np.log10(x_grid) if _log_x else x_grid
                    error_sspl = compute_error(res_sspl, _x_err, error_type)
                else:
                    # Transform x to interpolator input space, clamp to sim range
                    _x_q = np.log10(sspl_x_data) if _log_x else sspl_x_data
                    x_eval_data = np.clip(_x_q, _sim_xi.min(), _sim_xi.max())
                    y_fit = np.clip(sim_ss_interp(x_eval_data), _y_lo_clip, _y_hi_clip)
                    # Reference y in same space as interpolator output
                    y_ref = (np.log10(np.clip(sspl_y_data, 1e-12, 1 - 1e-12)) if _log_y
                             else sspl_y_data)
                    res_sspl = y_fit - y_ref
                    error_sspl = compute_error(res_sspl, x_eval_data, error_type)

            # ---- TRPL ----
            if fit_mode in ('trpl', 'both'):
                error_trpl_parts = []
                if nn_model is None:
                    tspan = kwargs.get('tspan', [1e-10, 1e5])
                    t_arr = np.logspace(np.log10(max(tspan[0], 1e-12)),
                                        np.log10(tspan[-1]), 256)

                for curve in trpl_curves:
                    q_data = curve['qfls']
                    t_data = curve['tau']

                    if nn_model is not None:
                        # NN path: fixed axis always covers the data QFLS range.
                        # Interpolate NN output directly onto all data QFLS values.
                        tau_nn = _predict_nn(Et, Nt, tau_n, tau_p, krad_val, curve['npulse'])
                        if not np.all(np.isfinite(tau_nn)) or np.any(tau_nn <= 0):
                            return 1e6
                        interp_nn = PchipInterpolator(curve['qfls_nn'], tau_nn,
                                                      extrapolate=False)
                        if n_interp_trpl is not None and curve['interp'] is not None:
                            q_lo = max(q_data[0],  curve['qfls_nn'].min())
                            q_hi = min(q_data[-1], curve['qfls_nn'].max())
                            if q_lo >= q_hi:
                                return 1e8
                            qfls_grid = np.linspace(q_lo, q_hi, n_interp_trpl)
                            tau_exp   = curve['interp'](qfls_grid)
                            tau_sim   = interp_nn(qfls_grid)
                            if not np.all(np.isfinite(tau_sim)) or np.any(tau_sim <= 0):
                                return 1e2
                            if not np.all(np.isfinite(tau_exp)) or np.any(tau_exp <= 0):
                                return 2e2
                            log_res = np.log10(tau_sim) - np.log10(tau_exp)
                            error_trpl_parts.append(
                                compute_error(log_res, qfls_grid, error_type))
                        else:
                            tau_fit = interp_nn(q_data)
                            if not np.all(np.isfinite(tau_fit)) or np.any(tau_fit <= 0):
                                return 1e6
                            log_res = np.log10(tau_fit) - np.log10(t_data)
                            error_trpl_parts.append(
                                compute_error(log_res, q_data, error_type))
                    else:
                        # ODE path: simulation QFLS range depends on parameters.
                        result_tr = solve_transient(
                            t_arr, curve['npulse'], Eg, Nc, Nv, krad_val,
                            Nt, Et, tau_n, tau_p, T, 0)
                        if not isinstance(result_tr, tuple) or len(result_tr) != 14:
                            return 1e8
                        _, _, _, _, QFLS_tr, tau_tr = result_tr[:6]
                        mask = np.isfinite(QFLS_tr) & np.isfinite(tau_tr) & (tau_tr >= 0)
                        if not np.any(mask):
                            return 1e6
                        Q_sort   = np.sort(QFLS_tr[mask])
                        tau_sort = tau_tr[mask][np.argsort(QFLS_tr[mask])]
                        Q_unique, idx_unique = np.unique(Q_sort, return_index=True)
                        tau_unique = tau_sort[idx_unique]
                        if len(Q_unique) < 2:
                            return 1e4
                        sim_tr_interp = PchipInterpolator(Q_unique, tau_unique,
                                                          extrapolate=False)
                        q_sim_min = Q_unique.min()
                        q_sim_max = Q_unique.max()
                        if n_interp_trpl is not None and curve['interp'] is not None:
                            q_lo = max(q_data[0],  q_sim_min)
                            q_hi = min(q_data[-1], q_sim_max)
                            if q_lo >= q_hi:
                                return 1e8
                            qfls_grid = np.linspace(q_lo, q_hi, n_interp_trpl)
                            tau_exp   = curve['interp'](qfls_grid)
                            tau_sim   = sim_tr_interp(qfls_grid)
                            if not np.all(np.isfinite(tau_sim)) or np.any(tau_sim <= 0):
                                return 1e2
                            if not np.all(np.isfinite(tau_exp)) or np.any(tau_exp <= 0):
                                return 2e2
                            log_res = np.log10(tau_sim) - np.log10(tau_exp)
                            error_trpl_parts.append(
                                compute_error(log_res, qfls_grid, error_type))
                        else:
                            in_range = (q_data >= q_sim_min) & (q_data <= q_sim_max)
                            if not np.any(in_range):
                                return 3e2
                            tau_fit = sim_tr_interp(q_data[in_range])
                            if not np.all(np.isfinite(tau_fit)) or np.any(tau_fit < 0):
                                return 4e2
                            log_res = np.log10(tau_fit) - np.log10(t_data[in_range])
                            error_trpl_parts.append(
                                compute_error(log_res, q_data[in_range], error_type))

                error_trpl = float(np.mean(error_trpl_parts))

            # ---- Combine ----
            if fit_mode == 'sspl':
                return error_sspl
            elif fit_mode == 'trpl':
                return error_trpl
            else:
                return w_sspl * error_sspl + w_trpl * error_trpl

        except Exception:
            return 1e6
    
    # Define optimization problem
    class FitProblem(Problem):
        def __init__(self, n_var):
            """
            Initialise the pymoo single-objective problem for CMA-ES fitting.

            Defines a box-constrained problem in the unit hypercube [0, 1]^n_var,
            where each dimension corresponds to one free (un-fixed) physical parameter
            in normalised space.

            Parameters
            ----------
            n_var : int
                Number of free parameters to optimise (_n_free), i.e. the dimensionality
                of the search space after removing any fixed parameters.
            """
            super().__init__(
                n_var=n_var,
                n_obj=1,
                xl=np.zeros(n_var),
                xu=np.ones(n_var),
                elementwise_evaluation=False,
            )

        def _evaluate(self, X, out, *args, **kwargs):
            """
            Evaluate the objective function for a population of candidate solutions.

            Called by pymoo on each generation; maps each row of *X* through the
            ``objective`` closure and collects the scalar errors for the optimiser.

            Parameters
            ----------
            X : np.ndarray
                Population matrix of shape (pop_size, n_var).  Each row is a
                normalised parameter vector in [0, 1]^n_var.
            out : dict
                pymoo output dictionary.  The key ``"F"`` is set to a column vector
                of shape (pop_size, 1) containing the scalar error for each individual.
            """
            out["F"] = np.array([objective(x) for x in X]).reshape(-1, 1)

    from pymoo.termination.default import DefaultSingleObjectiveTermination

    termination = DefaultSingleObjectiveTermination(
        #xtol=1e-5,#1e-6
        #cvtol=1e-6,
        #ftol=1e-2, #1e-6
        period=period,
        #n_max_gen=1000,
        #n_max_evals=100000
    )


    
    # Initial guess
    #Et0 = np.linspace(0.6*Eg, 0.9*Eg, num_traps)
    #Nt0 = 1e15 * np.ones(num_traps)
    #tau_n0 = 1e-7 * np.ones(num_traps)
    #tau_p0 = 1e-7 * np.ones(num_traps)
    #krad0 = krad_init
    if x0_multitrap is not None:    #if guess is provided
        if len(x0_multitrap) != num_traps * 4 + 1:
            raise ValueError(f"Initial guess x0_multitrap must have length {num_traps * 4 + 1}")
        # x0_multitrap layout: [Et_0, Nt_0, tau_n_0, tau_p_0,  Et_1, ...,  krad]
        x0_multitrap_norm = normalize_params(
            np.array([x0_multitrap[i * 4]     for i in range(num_traps)]),  # Et
            np.array([x0_multitrap[i * 4 + 1] for i in range(num_traps)]),  # Nt
            np.array([x0_multitrap[i * 4 + 2] for i in range(num_traps)]),  # tau_n
            np.array([x0_multitrap[i * 4 + 3] for i in range(num_traps)]),  # tau_p
            x0_multitrap[num_traps * 4],                                     # krad
        )

    def random_start(problem: Problem) -> np.ndarray:
        """
        Draw a uniformly random starting point for CMA-ES inside the search bounds.

        Parameters
        ----------
        problem : pymoo Problem
            The FitProblem instance; its ``xl`` and ``xu`` attributes define the
            lower and upper bounds of the search space (both are 0 and 1 in normalised
            space).

        Returns
        -------
        np.ndarray
            Random starting vector of shape (n_var,) with each element drawn
            uniformly from [xl_i, xu_i] = [0, 1].
        """
        return np.random.uniform(problem.xl, problem.xu)
    
    # Run optimization (only free parameters enter the search space)
    n_var = _n_free
    problem = FitProblem(n_var)


    all_trial_results = []

    def _reevaluate(x_vec):
        """
        Run the full physics simulation for a given parameter vector and return
        detailed outputs for result storage and plotting.

        Unlike the lightweight ``objective`` closure used during optimisation,
        this function runs ``solve_steady_state`` and/or ``solve_transient`` at
        full resolution and packs every output quantity into tidy DataFrames.
        It is called once per trial on the best-found parameter vector.

        Parameters
        ----------
        x_vec : np.ndarray
            Normalised free-parameter vector in [0, 1]^_n_free, as returned by
            the CMA-ES optimiser (``res.X``).

        Returns
        -------
        dict with keys:
            params : pd.DataFrame
                Fitted trap parameters — columns 'trap', 'Et_eV', 'Nt_1/cm3',
                'tau_n_s', 'tau_p_s' — one row per trap.
            krad_cm3s : float
                Fitted radiative recombination coefficient [cm³/s].
            error_total : float or None
                Weighted combined error (w_sspl·err_ss + w_trpl·err_tr) or
                the individual error when fit_mode is 'sspl' or 'trpl'.
            error_sspl : float or None
                SSPL error for the chosen error_type; None if not fitting SSPL.
            error_trpl : float or None
                Mean TRPL error across all curves; None if not fitting TRPL.
            error_trpl_per_curve : list of float or None
                Per-curve TRPL errors; None if not fitting TRPL.
            sspl_sim : pd.DataFrame or None
                Full steady-state simulation output (n, p, nt, Rtot, PLQY, tau, …)
                on the solver QFLS grid.
            sspl_fit : pd.DataFrame or None
                Comparison table of data vs. fit on the error-evaluation grid,
                including residuals.  Column names depend on sspl_residual_type.
            trpl_sim : pd.DataFrame or list of pd.DataFrame or None
                Full transient simulation output (time, n, p, QFLS, tau_diff, …)
                for each curve.  Single DataFrame for single-curve fits.
            trpl_fit : pd.DataFrame or list of pd.DataFrame or None
                Comparison table of data vs. fit for each TRPL curve, including
                log-space residuals.  Single DataFrame for single-curve fits.
            trpl_sim_list : list of pd.DataFrame or None
                Always-list version of trpl_sim (convenient for iteration).
            trpl_fit_list : list of pd.DataFrame or None
                Always-list version of trpl_fit (convenient for iteration).
        """
        Et_r, Nt_r, tau_n_r, tau_p_r, krad_r = denormalize_params(x_vec)

        # ---- SSPL ----
        if fit_mode in ('sspl', 'both'):
            result_ss_r = solve_steady_state(Ngrid, Eg, Nc, Nv, krad_r, Nt_r, Et_r, tau_n_r, tau_p_r, T)
            (n_arr_r, p_arr_r, nt_mat_r, Rtot_arr_raw_r, Rrad_arr_r, Rsrh_mat_r,
             qfls_arr_r, PLQY_arr_ss_r, tau_ss_r, tau_ss_n_r, tau_ss_p_r, nid_r,
             e_trap_r, e_detrap_r, h_trap_r, h_detrap_r) = result_ss_r

            Qfls_ss_arr_r   = qfls_arr_r.flatten()
            Rtot_arr_sims_r = Rtot_arr_raw_r.flatten()
            G_ss_suns_r     = Rtot_arr_sims_r / G_per_sun
            PLQY_arr_ss_r   = PLQY_arr_ss_r.flatten()

            _qfls_y_types_r = ('intensity_suns_qfls', 'g_qfls')
            if sspl_residual_type == 'qfls_plqy':
                _sim_x_r = Qfls_ss_arr_r
                _sim_y_r = PLQY_arr_ss_r
            elif sspl_residual_type in ('intensity_suns_plqy', 'intensity_suns_qfls'):
                _gs = np.argsort(G_ss_suns_r)
                _sim_x_r = G_ss_suns_r[_gs]
                _sim_y_r = (PLQY_arr_ss_r[_gs] if sspl_residual_type == 'intensity_suns_plqy'
                            else Qfls_ss_arr_r[_gs])
            else:
                _gs = np.argsort(Rtot_arr_sims_r)
                _sim_x_r = Rtot_arr_sims_r[_gs]
                _sim_y_r = (PLQY_arr_ss_r[_gs] if sspl_residual_type == 'g_plqy'
                            else Qfls_ss_arr_r[_gs])

            _is_qy_r = sspl_residual_type in _qfls_y_types_r
            _log_x_r = sspl_residual_type != 'qfls_plqy'
            _log_y_r = not _is_qy_r
            # Build sim interpolator in log10 space where data is log-distributed
            _sim_xi_r = np.log10(_sim_x_r) if _log_x_r else _sim_x_r
            _sim_yi_r = np.log10(_sim_y_r) if _log_y_r else _sim_y_r
            sim_ss_r = PchipInterpolator(_sim_xi_r, _sim_yi_r, extrapolate=False)
            # Clip bounds in interpolator output space (log10 for PLQY, linear for QFLS)
            _y_lo_clip_r = -12.0 if _log_y_r else 1e-12
            _y_hi_clip_r = np.log10(1 - 1e-12) if _log_y_r else None

            if n_interp_sspl is not None and _sspl_exp_interp is not None:
                xlo_r = max(sspl_x_data[0],  _sim_x_r.min())
                xhi_r = min(sspl_x_data[-1], _sim_x_r.max())
                xg_r  = (np.logspace(np.log10(xlo_r), np.log10(xhi_r), n_interp_sspl)
                         if _log_x_r and xlo_r > 0
                         else np.linspace(xlo_r, xhi_r, n_interp_sspl))
                _xg_eval_r = np.log10(xg_r) if _log_x_r else xg_r
                ye_r  = np.clip(_sspl_exp_interp(_xg_eval_r), _y_lo_clip_r, _y_hi_clip_r)
                ys_r  = np.clip(sim_ss_r(_xg_eval_r),         _y_lo_clip_r, _y_hi_clip_r)
                rs_r  = ys_r - ye_r  # log10(sim)-log10(exp) for PLQY; eV diff for QFLS
                xe_r  = np.log10(xg_r) if _log_x_r else xg_r
                err_ss_r = compute_error(rs_r, xe_r, error_type)
                # Convert back to linear for DataFrame storage
                fyd_r = 10**ye_r if _log_y_r else ye_r
                fyf_r = 10**ys_r if _log_y_r else ys_r
                fx_r  = xg_r
            else:
                _xq_r = np.log10(sspl_x_data) if _log_x_r else sspl_x_data
                xc_r  = np.clip(_xq_r, _sim_xi_r.min(), _sim_xi_r.max())
                yf_r  = np.clip(sim_ss_r(xc_r), _y_lo_clip_r, _y_hi_clip_r)
                y_ref_r = (np.log10(np.clip(sspl_y_data, 1e-12, 1 - 1e-12)) if _log_y_r
                           else sspl_y_data)
                rs_r  = yf_r - y_ref_r
                err_ss_r = compute_error(rs_r, xc_r, error_type)
                fx_r  = sspl_x_data
                fyd_r = sspl_y_data
                fyf_r = 10**yf_r if _log_y_r else yf_r

            if sspl_residual_type == 'qfls_plqy':
                _xc_r, _yd_r, _yf_r, _rc_r = 'qfls_ss_eV', 'plqy_data', 'plqy_fit', 'log_residual'
            elif sspl_residual_type == 'intensity_suns_plqy':
                _xc_r, _yd_r, _yf_r, _rc_r = 'intensity_suns', 'plqy_data', 'plqy_fit', 'log_residual'
            elif sspl_residual_type == 'intensity_suns_qfls':
                _xc_r, _yd_r, _yf_r, _rc_r = 'intensity_suns', 'qfls_data_eV', 'qfls_fit_eV', 'residual_eV'
            elif sspl_residual_type == 'g_plqy':
                _xc_r, _yd_r, _yf_r, _rc_r = 'generation_rate_cm3s', 'plqy_data', 'plqy_fit', 'log_residual'
            else:
                _xc_r, _yd_r, _yf_r, _rc_r = 'generation_rate_cm3s', 'qfls_data_eV', 'qfls_fit_eV', 'residual_eV'

            sspl_sim_r = pd.DataFrame({
                'n_ss_1/cm3':       n_arr_r.flatten(),
                'p_ss_1/cm3':       p_arr_r.flatten(),
                **{f'nt_ss_{k+1}_1/cm3': nt_mat_r[k, :] for k in range(num_traps)},
                'Rtot_ss_1/cm3s':   Rtot_arr_sims_r,
                'G_ss_suns':        G_ss_suns_r,
                'Rrad_ss_1/cm3s':   Rrad_arr_r.flatten(),
                **{f'Rsrh_ss_trap{k+1}_1/cm3s': Rsrh_mat_r[k, :] for k in range(num_traps)},
                'qfls_ss_eV':       Qfls_ss_arr_r,
                'PLQY_ss':          PLQY_arr_ss_r,
                'tau_ss_s':         tau_ss_r.flatten(),
                'tau_n_ss_s':       tau_ss_n_r.flatten(),
                'tau_p_ss_s':       tau_ss_p_r.flatten(),
                'nid':              nid_r.flatten(),
                **{f'e_trapping_ss_trap{k+1}_1/cm3/s':   e_trap_r[k, :]   for k in range(num_traps)},
                **{f'e_detrapping_ss_trap{k+1}_1/cm3/s': e_detrap_r[k, :] for k in range(num_traps)},
                **{f'h_trapping_ss_trap{k+1}_1/cm3/s':   h_trap_r[k, :]   for k in range(num_traps)},
                **{f'h_detrapping_ss_trap{k+1}_1/cm3/s': h_detrap_r[k, :] for k in range(num_traps)},
            })
            sspl_fit_r = pd.DataFrame({_xc_r: fx_r, _yd_r: fyd_r, _yf_r: fyf_r, _rc_r: rs_r})
        else:
            err_ss_r   = None
            sspl_sim_r = None
            sspl_fit_r = None

        # ---- TRPL ----
        if fit_mode in ('trpl', 'both'):
            trpl_sims_r = []
            trpl_fits_r = []
            errs_trpl_r = []

            if nn_model is None:
                tsp_r = kwargs.get('tspan', [1e-10, 1e5])
                t_r   = np.logspace(np.log10(max(tsp_r[0], 1e-12)), np.log10(tsp_r[-1]), 256)

            for curve_r in trpl_curves:
                qd_r = curve_r['qfls']
                td_r = curve_r['tau']

                if nn_model is not None:
                    tau_nn_r = _predict_nn(Et_r, Nt_r, tau_n_r, tau_p_r, krad_r, curve_r['npulse'])
                    interp_r = PchipInterpolator(curve_r['qfls_nn'], tau_nn_r, extrapolate=False)
                    qmn_r, qmx_r = curve_r['qfls_nn'].min(), curve_r['qfls_nn'].max()
                    trpl_sims_r.append(pd.DataFrame({
                        'qfls_nn_eV': curve_r['qfls_nn'],
                        'tau_nn_s':   tau_nn_r,
                    }))
                else:
                    res_tr_r = solve_transient(
                        t_r, curve_r['npulse'], Eg, Nc, Nv, krad_r,
                        Nt_r, Et_r, tau_n_r, tau_p_r, T, 0)
                    (t_out_r, n_tr_r, p_tr_r, nt_tr_r, QFLS_tr_r, tau_tr_r,
                     tau_n_tr_r, tau_p_tr_r, PL_tr_r, e_trap_tr_r,
                     e_detrap_tr_r, h_trap_tr_r, h_detrap_tr_r, kr_r) = res_tr_r
                    mk_r = np.isfinite(QFLS_tr_r) & np.isfinite(tau_tr_r) & (tau_tr_r > 0)
                    Qs_r = np.sort(QFLS_tr_r[mk_r])
                    ts_r = tau_tr_r[mk_r][np.argsort(QFLS_tr_r[mk_r])]
                    Qu_r, iu_r = np.unique(Qs_r, return_index=True)
                    interp_r = PchipInterpolator(Qu_r, ts_r[iu_r], extrapolate=False)
                    qmn_r, qmx_r = Qu_r.min(), Qu_r.max()
                    trpl_sims_r.append(pd.DataFrame({
                        'time_tr_s':   t_out_r.flatten(),
                        'pl_tr':       PL_tr_r.flatten(),
                        'qfls_tr_eV':  QFLS_tr_r.flatten(),
                        'tau_tr_s':    tau_tr_r.flatten(),
                        'tau_n_tr_s':  tau_n_tr_r.flatten(),
                        'tau_p_tr_s':  tau_p_tr_r.flatten(),
                        'n_tr_1/cm3':  n_tr_r.flatten(),
                        'p_tr_1/cm3':  p_tr_r.flatten(),
                        **{f'nt_tr_{j+1}_1/cm3':                nt_tr_r[:, j]        for j in range(num_traps)},
                        **{f'e_trapping_tr_trap{j+1}_1/cm3/s':  e_trap_tr_r[:, j]    for j in range(num_traps)},
                        **{f'e_detrapping_tr_trap{j+1}_1/cm3/s':e_detrap_tr_r[:, j]  for j in range(num_traps)},
                        **{f'h_trapping_tr_trap{j+1}_1/cm3/s':  h_trap_tr_r[:, j]    for j in range(num_traps)},
                        **{f'h_detrapping_tr_trap{j+1}_1/cm3/s':h_detrap_tr_r[:, j]  for j in range(num_traps)},
                    }))

                if n_interp_trpl is not None and curve_r['interp'] is not None:
                    ql_r = max(qd_r[0], qmn_r);  qh_r = min(qd_r[-1], qmx_r)
                    qg_r = np.linspace(ql_r, qh_r, n_interp_trpl)
                    te_r = curve_r['interp'](qg_r)
                    ts_r = interp_r(qg_r)
                    lr_r = np.log10(ts_r) - np.log10(te_r)
                    ec_r = compute_error(lr_r, qg_r, error_type)
                    fq_r, ftd_r, fts_r = qg_r, te_r, ts_r
                else:
                    inr_r  = (qd_r >= qmn_r) & (qd_r <= qmx_r)
                    tft_r  = np.full(len(qd_r), np.nan)
                    lrf_r  = np.full(len(qd_r), np.nan)
                    tft_r[inr_r]  = interp_r(qd_r[inr_r])
                    lrf_r[inr_r]  = np.log10(tft_r[inr_r]) - np.log10(td_r[inr_r])
                    lr_r = lrf_r[inr_r]
                    ec_r = compute_error(lr_r, qd_r[inr_r], error_type)
                    fq_r, ftd_r, fts_r = qd_r, td_r, tft_r
                    lr_r = lrf_r

                errs_trpl_r.append(ec_r)
                trpl_fits_r.append(pd.DataFrame({
                    'qfls_eV':      fq_r,
                    'tau_data_s':   ftd_r,
                    'tau_fit_s':    fts_r,
                    'log_residual': lr_r,
                }))

            err_tr_r   = float(np.mean(errs_trpl_r))
            trpl_sim_r = trpl_sims_r[0] if len(trpl_sims_r) == 1 else trpl_sims_r
            trpl_fit_r = trpl_fits_r[0] if len(trpl_fits_r) == 1 else trpl_fits_r
        else:
            err_tr_r    = None
            errs_trpl_r = None
            trpl_sims_r = None
            trpl_fits_r = None
            trpl_sim_r  = None
            trpl_fit_r  = None

        err_tot_r = (err_ss_r      if fit_mode == 'sspl'
                     else err_tr_r if fit_mode == 'trpl'
                     else w_sspl * err_ss_r + w_trpl * err_tr_r)

        return {
            'params': pd.DataFrame({
                'trap':     np.arange(1, num_traps + 1),
                'Et_eV':    Et_r,
                'Nt_1/cm3': Nt_r,
                'tau_n_s':  tau_n_r,
                'tau_p_s':  tau_p_r,
            }),
            'krad_cm3s':            krad_r,
            'error_total':          err_tot_r,
            'error_sspl':           err_ss_r,
            'error_trpl':           err_tr_r,
            'error_trpl_per_curve': errs_trpl_r,
            'sspl_sim':             sspl_sim_r,
            'trpl_sim':             trpl_sim_r,
            'sspl_fit':             sspl_fit_r,
            'trpl_fit':             trpl_fit_r,
            'trpl_sim_list':        trpl_sims_r,
            'trpl_fit_list':        trpl_fits_r,
        }

    res = None
    x0 = None

    for i in range(n_fits):
        print(f"\n===== Fit {i+1}/{n_fits} =====")
        if i == 0 and x0_multitrap is not None: #use user provided guess for first fit
            x0_rand = x0_multitrap_norm #user provided initial guess
        else:
            x0_rand = random_start(problem)
    
        print("\nStarting values:")
        Et_start, Nt_start, tau_n_start, tau_p_start, krad_start = denormalize_params(x0_rand)
        for i in range(num_traps):
            print(f"  Trap {i+1}: Et={Et_start[i]:.3f} eV, Nt={Nt_start[i]:.2e} cm⁻³, "
                f"τn={tau_n_start[i]:.2e} s, τp={tau_p_start[i]:.2e} s")
        print(f"  krad={krad_start:.2e} cm³/s")
        print(f"\nStarting RMSE: {objective(x0_rand):.6f}")
        print("\nOptimizing...")
        print("-"*70)
    
        algorithm = CMAES(
            x0=x0_rand,
            sigma=sigma, 
            maxfevals=maxfevals, 
            parallelize=True,
            restarts=0,
            restart_from_best=False,
            bipop=False,
            ftarget=ftarget,
        )

        start_time = time.time()
        trial_res = minimize(
            problem, 
            algorithm, 
            seed=43, 
            verbose=True, 
            save_history=True,
            termination=termination,)
        solve_time = time.time() - start_time
        print(f"Final Error: {trial_res.F[0]:.6f}")

        all_trial_results.append({
            'x_opt':      trial_res.X.copy(),
            'x0':         x0_rand.copy(),
            'error':      float(trial_res.F[0]),
            'solve_time': solve_time,
            'n_evals':    trial_res.algorithm.evaluator.n_eval,
            'n_gens':     len(trial_res.history) if hasattr(trial_res, 'history') else 0,
        })
        if res is None or trial_res.F < res.F:
            res = trial_res #save best result
            x0 = x0_rand

    # Extract results
    x_opt = res.X
    rmse_final = res.F[0] if isinstance(res.F, np.ndarray) else res.F
    n_evals = res.algorithm.evaluator.n_eval
    n_gens = len(res.history) if hasattr(res, 'history') else 0
    
    Et_opt, Nt_opt, tau_n_opt, tau_p_opt, krad_opt = denormalize_params(x_opt)

    # ---- Re-evaluate best fit at full resolution ----
    best_eval       = _reevaluate(x_opt)
    sspl_sim_df     = best_eval['sspl_sim']
    sspl_fit_df     = best_eval['sspl_fit']
    trpl_sim_df     = best_eval['trpl_sim']
    trpl_fit_df     = best_eval['trpl_fit']
    trpl_sim_dfs    = best_eval['trpl_sim_list']
    trpl_fit_dfs    = best_eval['trpl_fit_list']
    error_sspl      = best_eval['error_sspl']
    error_trpl      = best_eval['error_trpl']
    error_trpl_list = best_eval['error_trpl_per_curve']
    error_total     = best_eval['error_total']

    # ---- Re-evaluate all trials at full resolution ----
    print("\nEvaluating all trials at full resolution...")
    all_trials = []
    for _trial in all_trial_results:
        _teval = _reevaluate(_trial['x_opt'])
        _teval['x_opt']           = _trial['x_opt']
        _teval['x0']              = _trial['x0']
        _teval['x0_unnormalized'] = denormalize_params(_trial['x0'])
        _teval['solve_time']      = _trial['solve_time']
        _teval['n_evals']         = _trial['n_evals']
        _teval['n_gens']          = _trial['n_gens']
        all_trials.append(_teval)

    # Print results
    print("\n" + "="*70)
    print("OPTIMIZATION RESULTS")
    print("="*70)
    print(f"Function evaluations: {n_evals}")
    print(f"Generations: {n_gens}")
    print(f"Solve time: {solve_time:.2f} s")
    print(f"Fit mode  : {fit_mode}   |   Error type: {error_type}")
    print(f"\nTotal error: {error_total:.6f}")
    if error_sspl is not None:
        print(f"SSPL error : {error_sspl:.6f}")
    if error_trpl is not None:
        print(f"TRPL error : {error_trpl:.6f}  (mean over {len(error_trpl_list)} curve(s))")
        if error_trpl_list and len(error_trpl_list) > 1:
            for i_c, ec in enumerate(error_trpl_list):
                print(f"  Curve {i_c+1} : {ec:.6f}  (npulse = {npulse_list[i_c]:.3e})")
    print("\nFitted parameters:")
    for i in range(num_traps):
        print(f"  Trap {i+1}:")
        print(f"    Et = {Et_opt[i]:.4f} eV")
        print(f"    Nt = {Nt_opt[i]:.3e} cm⁻³")
        print(f"    τn = {tau_n_opt[i]:.3e} s")
        print(f"    τp = {tau_p_opt[i]:.3e} s")
    print(f"  krad = {krad_opt:.3e} cm³/s")
    print("="*70)

    results = {
        'params': pd.DataFrame({
            'trap':      np.arange(1, num_traps + 1),
            'Et_eV':     Et_opt,
            'Nt_1/cm3':  Nt_opt,
            'tau_n_s':   tau_n_opt,
            'tau_p_s':   tau_p_opt,
        }),
        'krad_cm3s':        krad_opt,
        'fit_mode':            fit_mode,
        'error_type':          error_type,
        'sspl_residual_type':  sspl_residual_type,
        'n_interp_sspl':    n_interp_sspl,
        'n_interp_trpl':    n_interp_trpl,
        'qfls_range_sspl':  qfls_range_sspl,
        'qfls_range_trpl':  qfls_range_trpl,
        'error_total':  error_total,
        'error_sspl':   error_sspl,
        'error_trpl':   error_trpl,
        # Per-curve TRPL errors (list, or None if not fitting TRPL)
        'error_trpl_per_curve': error_trpl_list,
        # npulse values used for each TRPL curve
        'trpl_npulse_list': npulse_list if fit_mode in ('trpl', 'both') else None,
        'n_evals':      n_evals,
        'n_gens':       n_gens,
        'solve_time':   solve_time,
        'history':      res.history if hasattr(res, 'history') else None,
        'x_opt':        res.X,
        'x0':           x0,
        'x0_unnormalized': denormalize_params(x0),
        'bounds': {
            'Et_bounds':    Et_bounds_list,
            'Nt_bounds':    Nt_bounds_list,
            'tau_n_bounds': tau_n_bounds_list,
            'tau_p_bounds': tau_p_bounds_list,
            'krad_bounds':  krad_bounds,
        },
        # Fixed-parameter info needed by plot_corner
        'free_indices':        _free_idx,
        'fixed_params_physical': _fixed_phys,
        'sspl_sim':  sspl_sim_df,
        # For single-curve fits: plain DataFrame (backward-compatible)
        # For multi-curve fits: list of DataFrames
        'trpl_sim':  trpl_sim_df,
        'sspl_fit':  sspl_fit_df,
        'trpl_fit':  trpl_fit_df,
        # Always-list versions (convenient for multi-curve loops)
        'trpl_sim_list': trpl_sim_dfs,
        'trpl_fit_list': trpl_fit_dfs,
        'all_trials':    all_trials,
    }

    return results


def plot_fit(
    results: dict,
    figsize: tuple | None = None,
    save_path: str | None = None,
) -> tuple[Figure, tuple[Axes, ...]]:
    """
    Plot SSPL and/or TRPL fits depending on the fit_mode used.

    Produces one panel per active dataset (SSPL left, TRPL right).
    The error metric label shown in the corner of each panel matches the
    ``error_type`` stored in *results*.

    Parameters
    ----------
    results   : dict
        Output dictionary from :func:`fit_multitrap`.
    figsize   : tuple or None
        Figure size ``(width, height)`` in inches.  Defaults to
        ``(7.5, 3.75)`` when both panels are shown, ``(4.5, 4.0)`` for one.
    save_path : str or None
        If given, the figure is saved to this path at 300 dpi.

    Returns
    -------
    fig   : matplotlib Figure
    axes  : tuple of Axes  (one per panel)
    """
    fit_mode           = results.get('fit_mode', 'both')
    error_type         = results.get('error_type', 'rmse')
    sspl_residual_type = results.get('sspl_residual_type', 'plqy_vs_qfls')

    has_sspl = fit_mode in ('sspl', 'both') and results.get('sspl_fit') is not None
    has_trpl = fit_mode in ('trpl', 'both') and results.get('trpl_fit') is not None

    n_panels = int(has_sspl) + int(has_trpl)
    if n_panels == 0:
        raise ValueError("No fit data available to plot.")

    if figsize is None:
        figsize = (7.5, 3.75) if n_panels == 2 else (4.5, 4.0)

    fig, axes = plt.subplots(1, n_panels, figsize=figsize)
    if n_panels == 1:
        axes = [axes]

    ax_idx = 0

    # ---- SSPL panel ----
    if has_sspl:
        ax1 = axes[ax_idx]; ax_idx += 1
        df_sspl = results['sspl_fit']

        if sspl_residual_type == 'qfls_plqy':
            ax1.semilogy(df_sspl['qfls_ss_eV'], df_sspl['plqy_data'], 'o',
                         color='#2E86AB', markersize=8, label='Data',
                         markeredgecolor='white', markeredgewidth=1.5)
            ax1.semilogy(df_sspl['qfls_ss_eV'], df_sspl['plqy_fit'], '-',
                         color='#A23B72', linewidth=2.5, label='Fit')
            ax1.set_xlabel('QFLS [eV]', fontsize=12, fontweight='bold')
            ax1.set_ylabel('PLQY', fontsize=12, fontweight='bold')
        elif sspl_residual_type == 'intensity_suns_plqy':
            ax1.loglog(df_sspl['intensity_suns'], df_sspl['plqy_data'], 'o',
                       color='#2E86AB', markersize=8, label='Data',
                       markeredgecolor='white', markeredgewidth=1.5)
            ax1.loglog(df_sspl['intensity_suns'], df_sspl['plqy_fit'], '-',
                       color='#A23B72', linewidth=2.5, label='Fit')
            ax1.set_xlabel('Intensity [suns]', fontsize=12, fontweight='bold')
            ax1.set_ylabel('PLQY', fontsize=12, fontweight='bold')
        elif sspl_residual_type == 'intensity_suns_qfls':
            ax1.semilogx(df_sspl['intensity_suns'], df_sspl['qfls_data_eV'], 'o',
                         color='#2E86AB', markersize=8, label='Data',
                         markeredgecolor='white', markeredgewidth=1.5)
            ax1.semilogx(df_sspl['intensity_suns'], df_sspl['qfls_fit_eV'], '-',
                         color='#A23B72', linewidth=2.5, label='Fit')
            ax1.set_xlabel('Intensity [suns]', fontsize=12, fontweight='bold')
            ax1.set_ylabel('QFLS [eV]', fontsize=12, fontweight='bold')
        elif sspl_residual_type == 'g_plqy':
            ax1.loglog(df_sspl['generation_rate_cm3s'], df_sspl['plqy_data'], 'o',
                       color='#2E86AB', markersize=8, label='Data',
                       markeredgecolor='white', markeredgewidth=1.5)
            ax1.loglog(df_sspl['generation_rate_cm3s'], df_sspl['plqy_fit'], '-',
                       color='#A23B72', linewidth=2.5, label='Fit')
            ax1.set_xlabel('Generation rate [cm⁻³s⁻¹]', fontsize=12, fontweight='bold')
            ax1.set_ylabel('PLQY', fontsize=12, fontweight='bold')
        else:  # g_qfls
            ax1.semilogx(df_sspl['generation_rate_cm3s'], df_sspl['qfls_data_eV'], 'o',
                         color='#2E86AB', markersize=8, label='Data',
                         markeredgecolor='white', markeredgewidth=1.5)
            ax1.semilogx(df_sspl['generation_rate_cm3s'], df_sspl['qfls_fit_eV'], '-',
                         color='#A23B72', linewidth=2.5, label='Fit')
            ax1.set_xlabel('Generation rate [cm⁻³s⁻¹]', fontsize=12, fontweight='bold')
            ax1.set_ylabel('QFLS [eV]', fontsize=12, fontweight='bold')

        ax1.set_title('Steady-State PL', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        err_sspl = results.get('error_sspl')
        if err_sspl is not None:
            ax1.text(0.05, 0.95, f"{error_type}: {err_sspl:.4f}",
                     transform=ax1.transAxes, fontsize=10, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # ---- TRPL panel ----
    if has_trpl:
        ax2 = axes[ax_idx]; ax_idx += 1

        # Support single DataFrame (backward-compat) or list of DataFrames
        trpl_fit = results['trpl_fit']
        if isinstance(trpl_fit, pd.DataFrame):
            trpl_fit_list = [trpl_fit]
        else:
            trpl_fit_list = list(trpl_fit)

        npulse_list = results.get('trpl_npulse_list') or [None] * len(trpl_fit_list)
        err_per_curve = results.get('error_trpl_per_curve') or [None] * len(trpl_fit_list)

        # Colour cycle for multiple curves
        _cmap = plt.cm.tab10
        for i_c, df_tr in enumerate(trpl_fit_list):
            col = _cmap(i_c % 10)
            label_suffix = (f"  (n={npulse_list[i_c]:.2e})"
                            if npulse_list[i_c] is not None and len(trpl_fit_list) > 1
                            else "")
            ax2.semilogy(df_tr['qfls_eV'], df_tr['tau_data_s'], 'o',
                         color=col, markersize=7,
                         label=f'Data{label_suffix}' if len(trpl_fit_list) == 1 else f'Data {i_c+1}{label_suffix}',
                         markeredgecolor='white', markeredgewidth=1.0,
                         alpha=0.8)
            ax2.semilogy(df_tr['qfls_eV'], df_tr['tau_fit_s'], '-',
                         color=col, linewidth=2.5,
                         label=f'Fit{label_suffix}' if len(trpl_fit_list) == 1 else f'Fit {i_c+1}')

        ax2.set_xlabel('QFLS [eV]', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Differential Lifetime [s]', fontsize=12, fontweight='bold')
        ax2.set_title('Time-Resolved PL', fontsize=13, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        err_trpl = results.get('error_trpl')
        if err_trpl is not None:
            ax2.text(0.05, 0.95, f"{error_type}: {err_trpl:.4f}",
                     transform=ax2.transAxes, fontsize=10, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    return fig, tuple(axes)


def plot_corner(
    results: dict,
    max_points: int = 5000,
    figsize: tuple = (12, 12),
    save_path: str | None = None,
    error_threshold: float = 100,
    n_bins: int = 10,
) -> Figure:
    """
    Seaborn PairGrid corner plot of the CMA-ES optimisation history.

    Off-diagonal panels show scatter coloured by MSE quantile bin (viridis_r).
    Points exceeding *error_threshold* are rendered in light grey.  The best-fit
    point (cyan star) and starting point (lime square) are overlaid on every panel.

    Parameters
    ----------
    results : dict
        Output dictionary from :func:`fit_multitrap`.
    max_points : int
        Maximum evaluations to render; larger histories are sub-sampled.
    figsize : tuple
        Figure size ``(width, height)`` in inches.
    save_path : str or None
        If given, the figure is saved to this path at 300 dpi.
    error_threshold : float
        Evaluations with error above this value are shown in grey.
    n_bins : int
        Number of MSE quantile bins for the viridis_r colour scale.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    if results['history'] is None:
        raise ValueError("No optimization history available")

    # --- Extract history ---
    history = results['history']
    X_all = np.vstack([e.pop.get("X") for e in history])
    F_all = np.vstack([e.pop.get("F") for e in history]).flatten()

    n_points = len(X_all)
    if n_points > max_points:
        idx = np.random.choice(n_points, max_points, replace=False)
        X_all = X_all[idx]; F_all = F_all[idx]
        print(f"Subsampled {max_points} from {n_points} evaluations")

    # --- Parameter info ---
    num_traps = len(results['params'])
    bounds    = results['bounds']
    x_opt     = results['x_opt']
    x0        = results['x0']
    free_idx  = results.get('free_indices', list(range(4 * num_traps + 1)))

    def _bounds_for(full_i):
        """
        Return the physical bounds and scale type for a parameter by its full index.

        The full parameter vector has layout
        [Et_0, Nt_0, tau_n_0, tau_p_0, …, Et_{K-1}, …, tau_p_{K-1}, krad],
        i.e. 4·K + 1 entries total.

        Parameters
        ----------
        full_i : int
            0-based index into the full (un-reduced) parameter vector.
            Indices 0…4K-1 address trap parameters; index 4K addresses krad.

        Returns
        -------
        lo : float
            Lower bound in physical units.
        hi : float
            Upper bound in physical units.
        use_log : bool
            True when the parameter is best displayed / spaced on a log10 axis
            (Nt, tau_n, tau_p, krad).  False for Et (linear).
        """
        K = num_traps
        if full_i == 4 * K:
            lo, hi = bounds['krad_bounds']
            return lo, hi, True
        trap_i, sub_i = divmod(full_i, 4)
        key = ['Et_bounds', 'Nt_bounds', 'tau_n_bounds', 'tau_p_bounds'][sub_i]
        raw = bounds[key]
        lo, hi = (raw[trap_i] if isinstance(raw, list) else raw)
        return lo, hi, (sub_i != 0)   # Et linear; others log

    def _full_label(full_i):
        """
        Return the LaTeX axis label (with units) for a parameter by its full index.

        Parameters
        ----------
        full_i : int
            0-based index into the full parameter vector (same convention as
            ``_bounds_for``).

        Returns
        -------
        str
            Matplotlib-compatible LaTeX string, e.g. ``r'$E_{t1}$ [eV]'``.
            Trap subscripts are omitted for single-trap models to keep labels clean.
        """
        K = num_traps
        if full_i == 4 * K:
            return r'$k_\mathrm{rad}$ [cm$^3$/s]'
        trap_i, sub_i = divmod(full_i, 4)
        tl = f"{trap_i + 1}" if K > 1 else ""
        return [
            rf'$E_{{t{tl}}}$ [eV]',
            rf'$N_{{t{tl}}}$ [cm$^{{-3}}$]',
            rf'$\tau_{{n{tl}}}$ [s]',
            rf'$\tau_{{p{tl}}}$ [s]',
        ][sub_i]

    def _to_physical(x_norm: np.ndarray) -> np.ndarray:
        """
        Convert a matrix of normalised parameter vectors back to physical units.

        Applies the inverse of the normalisation used during optimisation: linear
        de-scaling for Et and log10 de-scaling for Nt, tau_n, tau_p, and krad.
        Only free parameter columns are returned (fixed parameters are excluded).

        Parameters
        ----------
        x_norm : np.ndarray
            Matrix of shape (N, _n_free) where each row is a normalised parameter
            vector in [0, 1]^_n_free from the CMA-ES history.

        Returns
        -------
        np.ndarray
            Matrix of shape (N, _n_free) with each column in its natural physical
            units (eV, cm⁻³, s, or cm³/s depending on the parameter).
        """
        X_p = np.zeros((len(x_norm), len(free_idx)))
        for col, fi in enumerate(free_idx):
            lo, hi, use_log = _bounds_for(fi)
            xc = np.clip(x_norm[:, col], 0, 1)
            X_p[:, col] = (10**(np.log10(lo) + xc * (np.log10(hi) - np.log10(lo)))
                           if use_log else lo + xc * (hi - lo))
        return X_p

    X_phys     = _to_physical(X_all)
    x_opt_phys = _to_physical(x_opt.reshape(1, -1))[0]
    x0_phys    = _to_physical(x0.reshape(1, -1))[0]

    param_names   = [_full_label(fi) for fi in free_idx]
    log_scale     = [_bounds_for(fi)[2] for fi in free_idx]
    col_idx       = {name: i for i, name in enumerate(param_names)}
    n_params      = len(param_names)

    # --- Quantile-bin colours ---
    good_mask = F_all <= error_threshold
    F_good    = F_all[good_mask]
    n_good    = int(good_mask.sum())
    n_bad     = int((~good_mask).sum())
    print(f"Good points (≤{error_threshold}): {n_good}  |  Bad: {n_bad}")

    if n_good > 0:
        q_edges   = np.unique(np.nanpercentile(F_good, np.linspace(0, 100, n_bins + 1)))
        n_q       = max(len(q_edges) - 1, 1)
        F_min, F_max = float(F_good.min()), float(F_good.max())
    else:
        q_edges = np.array([0.0, 1.0]); n_q = 1
        F_min = F_max = 0.0

    bin_color = np.zeros(len(F_all))
    for k in range(len(F_all)):
        if good_mask[k]:
            b = int(np.searchsorted(q_edges[1:], F_all[k]))
            bin_color[k] = b / max(n_q - 1, 1)

    # --- Axis limits (bounds ± 10 % padding) ---
    def _axis_limits(col: int):
        """
        Compute display axis limits for a parameter column with 10 % padding.

        Padding is applied in log10 space for log-scaled parameters and in linear
        space for Et, so that the search bounds are always visible on the plot.

        Parameters
        ----------
        col : int
            Column index into the free-parameter array (0 … _n_free - 1),
            corresponding to ``free_idx[col]`` in the full parameter vector.

        Returns
        -------
        tuple of float
            (lower_limit, upper_limit) in physical units suitable for passing to
            ``ax.set_xlim`` or ``ax.set_ylim``.
        """
        lo, hi, use_log = _bounds_for(free_idx[col])
        if use_log:
            pad = 0.1 * (np.log10(hi) - np.log10(lo))
            return 10**(np.log10(lo) - pad), 10**(np.log10(hi) + pad)
        pad = 0.1 * (hi - lo)
        return lo - pad, hi + pad

    # --- Build DataFrame for PairGrid ---
    df = pd.DataFrame(X_phys, columns=param_names)
    df['__good']  = good_mask
    df['__color'] = bin_color

    # --- PairGrid ---
    cell = figsize[0] / n_params
    g = sns.PairGrid(df[param_names], corner=True, height=cell, aspect=1.0)

    def _scatter(x_s, y_s, **kw):
        """
        PairGrid off-diagonal plot function: scatter coloured by error quantile bin.

        Evaluations that exceed *error_threshold* are drawn in light grey;
        all others are coloured by their viridis_r quantile bin (dark = low error,
        bright = high error within the 'good' subset).  Passed directly to
        ``sns.PairGrid.map_lower``.

        Parameters
        ----------
        x_s : pd.Series
            x-axis data for the current panel (one free parameter in physical units).
        y_s : pd.Series
            y-axis data for the current panel (another free parameter in physical units).
        **kw
            Ignored extra keyword arguments passed by seaborn.
        """
        ax  = plt.gca()
        cx, cy = x_s.name, y_s.name
        bad = df[~df['__good']]
        gd  = df[df['__good']]
        if len(bad):
            ax.scatter(bad[cx], bad[cy], c='lightgray', s=4,
                       alpha=0.3, rasterized=True, zorder=1)
        if len(gd):
            ax.scatter(gd[cx], gd[cy],
                       c=gd['__color'].values, cmap='viridis_r',
                       vmin=0, vmax=1, s=4, alpha=0.5, rasterized=True, zorder=2)

    def _hist(x_s, **kw):
        """
        PairGrid diagonal plot function: histogram of sampled parameter values.

        Draws two overlaid histograms — one for evaluations above *error_threshold*
        (light grey) and one for 'good' evaluations (viridis_r mid-colour).
        Passed directly to ``sns.PairGrid.map_diag``.

        Parameters
        ----------
        x_s : pd.Series
            Data for the diagonal parameter (one free parameter in physical units).
        **kw
            Ignored extra keyword arguments passed by seaborn.
        """
        ax  = plt.gca()
        cx  = x_s.name
        bad = df[~df['__good']]
        gd  = df[df['__good']]
        if len(bad):
            ax.hist(bad[cx], bins=30, color='lightgray', alpha=0.5,
                    edgecolor='black', linewidth=0.5)
        if len(gd):
            ax.hist(gd[cx], bins=30, color=plt.cm.viridis_r(0.5),
                    alpha=0.7, edgecolor='black', linewidth=0.5)

    g.map_lower(_scatter)
    g.map_diag(_hist)

    fig = g.figure
    fig.set_size_inches(figsize)

    # --- Second pass: scales, limits, markers, grid ---
    for i in range(n_params):
        for j in range(n_params):
            ax = g.axes[i][j]
            if ax is None:
                continue
            if i == j:
                if log_scale[i]:
                    ax.set_xscale('log')
                ax.set_xlim(_axis_limits(i))
                ax.axvline(x0_phys[i],    color='lime', ls='--', lw=2,   zorder=10)
                ax.axvline(x_opt_phys[i], color='cyan', ls='--', lw=2,   zorder=10)
                ax.grid(True, alpha=0.3, ls='--', lw=0.5)
            elif j < i:
                if log_scale[j]: ax.set_xscale('log')
                if log_scale[i]: ax.set_yscale('log')
                ax.set_xlim(_axis_limits(j))
                ax.set_ylim(_axis_limits(i))
                ax.plot(x0_phys[j],    x0_phys[i],    's', color='lime',
                        ms=10, mec='white', mew=1.5, zorder=10)
                ax.plot(x_opt_phys[j], x_opt_phys[i], '*', color='cyan',
                        ms=14, mec='white', mew=1.5, zorder=10)
                ax.grid(True, alpha=0.3, ls='--', lw=0.5)

    # --- Colorbar ---
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.87, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(
        norm=plt.Normalize(vmin=F_min, vmax=F_max), cmap='viridis_r')
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cbar_ax)
    _err_lbl = results.get('error_type', 'rmse').upper()
    cbar.set_label(_err_lbl, fontsize=12, fontweight='bold')
    cbar.ax.tick_params(labelsize=10)

    # --- Legend ---
    legend_elements = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor='lime',
               markeredgecolor='white', markeredgewidth=2, markersize=10, label='Start'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='cyan',
               markeredgecolor='white', markeredgewidth=2, markersize=14, label='Best Fit'),
    ]
    if n_bad > 0:
        legend_elements.append(
            Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgray',
                   markersize=8, alpha=0.5, label=f'Outliers (>{error_threshold})'))
    fig.legend(handles=legend_elements, loc='upper right', fontsize=10,
               framealpha=0.9, edgecolor='gray', bbox_to_anchor=(0.85, 0.98))

    title = f'CMA-ES Optimization History  |  {len(X_all)} evaluations'
    if n_good > 0:
        title += f'  |  Best {_err_lbl}: {F_min:.4f}'
    fig.suptitle(title, fontsize=13, fontweight='bold')

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Corner plot saved to {save_path}")

    return fig