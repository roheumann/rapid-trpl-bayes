from __future__ import annotations

import numpy as np
from constants import *
import pandas as pd
import torch
from splinetorch.b_spline import BSpline
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
from pathlib import Path
import json








def voc_(g: np.ndarray, plqy: np.ndarray, Eg: float, T: float = 300.0, a0:float = 1.0, Eu:float = None, am15g_path=None) -> float:
    """
    Compute the radiative open-circuit voltage V_oc,rad at 1-sun in the
    Shockley-Queisser limit (step-function absorption above Eg).

    Uses the measured AM 1.5G photon flux for J_sc and the Planck blackbody
    spectrum for J_0,rad.

    Formula
    -------
    V_oc,rad = kT · ln(J_sc / J_0,rad + 1)

    where

        J_sc    = q  ∫_{Eg}^∞ Phi_AM1.5G(E) dE          [A/cm²]
        J_0,rad = q  ∫_{Eg}^∞ Phi_bb(E) dE              [A/cm²]
        Phi_bb(E) = 2π / h³c² · E² / (exp(E/kT) − 1)   [photons cm⁻² s⁻¹ eV⁻¹]

    Parameters
    ----------
    Eg : float
        Bandgap [eV].
    T : float
        Device temperature [K].  Default 300 K.
    am15g_path : str or Path or None
        Path to the AM 1.5G data file (two whitespace-separated columns:
        energy [eV] and photon flux [photons cm⁻² s⁻¹ eV⁻¹]).
        Defaults to ``<package_dir>/data/AM15G.dat``.

    Returns
    -------
    float
        V_oc,rad [eV].
    """
    from scipy.integrate import trapezoid

    q  = 1.602e-19        # elementary charge [As]
    c  = 2.998e10         # speed of light [cm/s]
    h  = 6.626e-34 / q   # Planck’s constant [eVs]
    kT = 8.617333e-5 * T  # thermal energy [eV]

    # ── Load AM 1.5G spectrum ────────────────────────────────────────────────
    if am15g_path is None:
        am15g_path = Path(__file__).parent / "data" / "AM15G.dat"
    am15g = pd.read_csv(
        am15g_path, sep=r'\s+', header=None,
        names=['E', 'sunspectrum'], engine='python',
    )

    def phi_bb(E):
        """Planck blackbody photon flux [photons cm⁻² s⁻¹ eV⁻¹], overflow-safe."""
        exponent = np.clip(E / kT, 0.0, 700.0)
        return 2 * np.pi / h**3 / c**2 * E**2 / (np.exp(exponent) - 1) #factor 4 instead of 2: considers recombination from whole sphere

    E=np.linspace(0.4, 4,1000) #am15g.E.max()
    am15g_inter = PchipInterpolator(am15g['E'], am15g['sunspectrum'], extrapolate=False)(E)
    sq_mask = E >= Eg
    E_arr   = E[sq_mask]
    phi_sun_arr = am15g_inter[sq_mask]

    Jsc_AM_sq   = q * trapezoid(phi_sun_arr, E_arr)  # A/cm²
    J0_sq    = q * trapezoid(phi_bb(E_arr),  E_arr)  # A/cm²

    V_oc_rad_sq = kT * np.log(Jsc_AM_sq / J0_sq + 1)


    def urbach_absorbtance(E, Eg, Eu, a0=1.0):
        """Urbach tail absorbtance (arbitrary units)."""
        absorbtance = np.zeros_like(E)
        sq_mask = E >= Eg
        absorbtance[sq_mask] =  a0 
        absorbtance[~sq_mask] = a0 * np.exp((E[sq_mask] - Eg) / Eu)
        return absorbtance



    if Eu != None:
        J0_rad = q * trapezoid(phi_bb(E_arr) * urbach_absorbtance(E_arr, Eg, Eu, a0), E_arr)

        V_oc = V_oc_rad_sq + kT * np.log(J0_sq/J0_rad) + kT*np.log(g) + kT * np.log(plqy) #check this!
    else:
        V_oc = V_oc_rad_sq + kT + kT*np.log(intensity) + kT * np.log(plqy)
    return V_oc


def preprocess_sspl(
    df: pd.DataFrame,
    Eg: float,
    krad: float | None = None,
    Nc: float = 2.21359e18,
    Nv: float = 2.21359e18,
    T: float = 300.0,
    G_per_sun: float = 1.0e21,
    calc_voc: bool = False,
    plqy_in_percent: bool = False,
    intensity_col: str = 'intensity_suns',
    plqy_col: str = 'plqy',
    a0:float = 1.0,
    Eu:float = None,
) -> pd.DataFrame:
    """
    Convert raw SSPL data (intensity in suns + PLQY) into a DataFrame that
    contains a ``qfls`` column [eV], ready for use with :func:`fit_multitrap`.

    Two methods are supported (select with ``use_sq_limit``):

    **Method 1 — krad-based** (``use_sq_limit=False``, requires ``krad``)::

        QFLS = kT · ln( PLQY × G / (krad × ni²) )

        where  G = intensity_suns × G_per_sun  and  ni² = Nc·Nv·exp(−Eg/kT).

    **Method 2 — Shockley-Queisser limit** (``use_sq_limit=True``)::

        QFLS = V_oc,rad(1 sun) + kT·ln(suns) + kT·ln(PLQY)

        V_oc,rad is computed from blackbody integrals via :func:`sq_voc_rad`;
        no knowledge of krad, Nc, or Nv is required.

    Parameters
    ----------
    df : pd.DataFrame
        Raw SSPL data.
    Eg : float
        Bandgap [eV].
    krad : float or None
        Bimolecular radiative recombination coefficient [cm³/s].
        Required when ``use_sq_limit=False``; ignored otherwise.
    Nc, Nv : float
        Effective DOS in CB and VB [cm⁻³].  Only used when
        ``use_sq_limit=False``.
    T : float
        Temperature [K].  Default 300 K.
    G_per_sun : float
        Generation rate per 1-sun [cm⁻³ s⁻¹].  Only used when
        ``use_sq_limit=False``.  Default 10²¹.
    use_sq_limit : bool
        If ``True``, use the Shockley-Queisser formula (recommended —
        no material parameters beyond Eg needed).  Default ``False``.
    plqy_in_percent : bool
        Set to ``True`` if the PLQY column is in percent (0–100).
    intensity_col, plqy_col : str
        Column names in *df*.

    Returns
    -------
    pd.DataFrame
        Columns: ``intensity_suns``, ``plqy`` (fraction), ``qfls`` [eV].
        Rows with non-positive intensity or PLQY are dropped.
    """
    kT = 8.617333e-5 * T                   # thermal energy [eV]

    intensity = df[intensity_col].values.astype(float).copy()
    plqy      = df[plqy_col].values.astype(float).copy()

    if plqy_in_percent:
        plqy = plqy / 100.0

    valid_mask = (intensity > 0) & (plqy > 0)

    if calc_voc:
        # SQ-limit: QFLS = V_oc,rad + kT·ln(suns) + kT·ln(PLQY)
        V_oc = voc(Eg, T=T)
        with np.errstate(divide='ignore', invalid='ignore'):
            qfls = np.where(
                valid_mask,
                V_oc,
                np.nan,
            )
        print(f"preprocess_sspl: V_oc,rad = {V_oc:.4f} eV  "
              f"(Eg = {Eg} eV, T = {T} K)")
    else:
        if krad is None:
            raise ValueError(
                "krad must be provided when use_sq_limit=False. "
                "Pass a bimolecular rate coefficient [cm³/s] or set "
                "use_sq_limit=True to use the Shockley-Queisser limit."
            )
        # krad-based: QFLS = kT·ln(PLQY·G / (krad·ni²))
        ni2   = Nc * Nv * np.exp(-Eg / kT)
        G     = intensity * G_per_sun
        R_rad = plqy * G
        with np.errstate(divide='ignore', invalid='ignore'):
            qfls = np.where(R_rad > 0, kT * np.log(R_rad / (krad * ni2)), np.nan)

    out = pd.DataFrame({
        'intensity_suns': intensity,
        'plqy':           plqy,
        'qfls':           qfls,
    })

    valid  = valid_mask & np.isfinite(qfls)
    n_drop = int((~valid).sum())
    if n_drop:
        print(f"preprocess_sspl: dropped {n_drop} row(s) with non-positive "
              f"intensity or PLQY.")

    return out[valid].reset_index(drop=True)


def export_fit_results(results, output_dir, prefix="fit"):
    """
    Export fit results to separate files for easy analysis and archiving.
    
    Parameters
    ----------
    results : dict
        Results dictionary from fit_multitrap_simple containing:
        - 'params': fitted trap parameters DataFrame
        - 'krad_cm3s': radiative recombination coefficient
        - 'rmse_total', 'rmse_sspl', 'rmse_trpl': error metrics
        - 'n_evals', 'n_gens': optimization statistics
        - 'solve_time': fitting duration
        - 'history': optimization history
        - 'x_opt', 'x0': optimal and initial normalized parameters
        - 'bounds': parameter bounds dictionary
        - 'sspl_fit': steady-state PL fit DataFrame
        - 'trpl_fit': time-resolved PL fit DataFrame
    output_dir : str or Path
        Directory where files will be saved
    prefix : str
        Prefix for output files (default: "fit")
    
    Returns
    -------
    dict
        Dictionary with paths to all exported files
    
    Files Created
    -------------
    {prefix}_sspl.csv : Steady-state PL data and fit
    {prefix}_trpl.csv : Time-resolved PL data and fit
    {prefix}_parameters.json : All fit parameters, bounds, and metadata
    {prefix}_history.npz : Optimization history (X and F arrays)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    exported_files = {}
    
    # ========== Export TRPL fit ==========
    trpl_path = output_dir / f"{prefix}_trpl.csv"
    results['trpl_fit'].to_csv(trpl_path, index=False, float_format='%.6e')
    exported_files['trpl'] = str(trpl_path)
    print(f"✓ Exported TRPL fit to: {trpl_path}")

    prefix = prefix.replace("fit", "simulation")  # Update prefix for SSPL file
    trpl_path = output_dir / f"{prefix}_trpl.csv"
    results['trpl_sim'].to_csv(trpl_path, index=False, float_format='%.6e')
    exported_files['trpl_sim'] = str(trpl_path)
    print(f"✓ Exported TRPL sim to: {trpl_path}")
    
    # ========== Export parameters and metadata ==========
    params_dict = {
        # Trap parameters
        'trap_parameters': results['params'].to_dict(orient='records'),
        
        # Radiative recombination
        'krad_cm3s': float(results['krad_cm3s']),
        
        # Error metrics
        'error': {
            'total': float(results['error_total']),
            'trpl': float(results['error_trpl'])
        },
        
        # Optimization statistics
        'optimization': {
            'n_evaluations': int(results['n_evals']),
            'n_generations': int(results['n_gens']),
            'solve_time_seconds': float(results['solve_time'])
        },

        
        # Parameter bounds (per-trap lists of [lo, hi])
        'bounds': {
            'Et_eV':     [list(b) for b in results['bounds']['Et_bounds']],
            'Nt_cm3':    [list(b) for b in results['bounds']['Nt_bounds']],
            'tau_n_s':   [list(b) for b in results['bounds']['tau_n_bounds']],
            'tau_p_s':   [list(b) for b in results['bounds']['tau_p_bounds']],
            'krad_cm3s': list(results['bounds']['krad_bounds']),
        },
        
        # Normalized parameters
        'normalized_parameters': {
            'x_opt': results['x_opt'].tolist() if hasattr(results['x_opt'], 'tolist') else list(results['x_opt']),
            'x0': results['x0'].tolist() if hasattr(results['x0'], 'tolist') else list(results['x0'])
        },

        # Start point values (unnormalized)
        'initial_parameters': {
            **{f'Et_{i}_eV': float(results['x0_unnormalized'][0][i]) for i in range(len(results['x0_unnormalized'][0]))},
            **{f'Nt_{i}_1/cm3': float(results['x0_unnormalized'][1][i]) for i in range(len(results['x0_unnormalized'][1]))},
            **{f'tau_n_{i}_s': float(results['x0_unnormalized'][2][i]) for i in range(len(results['x0_unnormalized'][2]))},
            **{f'tau_p_{i}_s': float(results['x0_unnormalized'][3][i]) for i in range(len(results['x0_unnormalized'][3]))},
            'krad_cm3s': float(results['x0_unnormalized'][4])
        },
        
        # Metadata
        'metadata': {
            'num_traps': len(results['params']),
            'export_timestamp': pd.Timestamp.now().isoformat()
        }
    }
    
    params_path = output_dir / f"{prefix}_parameters.json"
    with open(params_path, 'w') as f:
        json.dump(params_dict, f, indent=2)
    exported_files['parameters'] = str(params_path)
    print(f"✓ Exported parameters to: {params_path}")
    
    # ========== Export optimization history ==========
    if results['history'] is not None:
        try:
            history = results['history']
            X_hist = [entry.pop.get("X") for entry in history]
            F_hist = [entry.pop.get("F") for entry in history]
            
            # Flatten
            X_all = np.vstack(X_hist)
            F_all = np.vstack(F_hist).flatten()
            
            # Also save generation information
            n_per_gen = [len(entry.pop.get("X")) for entry in history]
            gen_indices = np.repeat(np.arange(len(n_per_gen)), n_per_gen)
            
            history_path = output_dir / f"{prefix}_history.npz"
            np.savez_compressed(
                history_path,
                X=X_all,  # All evaluated points (normalized parameters)
                F=F_all,  # Objective function values (RMSE)
                generation=gen_indices,  # Generation number for each point
                n_generations=len(history),
                n_evaluations=len(X_all)
            )
            exported_files['history'] = str(history_path)
            print(f"✓ Exported optimization history to: {history_path}")
            print(f"  - {len(X_all)} evaluations across {len(history)} generations")
        except Exception as e:
            print(f"⚠ Warning: Could not export history: {e}")
            exported_files['history'] = None
    else:
        print("⚠ No optimization history available to export")
        exported_files['history'] = None
    
    # ========== Create summary file ==========
    summary_path = output_dir / f"{prefix}_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write(f"FIT RESULTS SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Export Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("TRAP PARAMETERS:\n")
        f.write("-" * 70 + "\n")
        for idx, row in results['params'].iterrows():
            f.write(f"Trap {int(row['trap'])}:\n")
            f.write(f"  Et    = {row['Et_eV']:.4f} eV\n")
            f.write(f"  Nt    = {row['Nt_1/cm3']:.4e} cm⁻³\n")
            f.write(f"  τₙ    = {row['tau_n_s']:.4e} s\n")
            f.write(f"  τₚ    = {row['tau_p_s']:.4e} s\n")
            f.write("\n")
        
        f.write(f"RADIATIVE RECOMBINATION:\n")
        f.write("-" * 70 + "\n")
        f.write(f"krad = {results['krad_cm3s']:.4e} cm³/s\n\n")
        
        f.write("FIT QUALITY:\n")
        f.write("-" * 70 + "\n")
        f.write(f"Total error  = {results['error_total']:.6f}\n")
        f.write(f"TRPL error   = {results['error_trpl']:.6f}\n\n")

        f.write("OPTIMIZATION STATISTICS:\n")
        f.write("-" * 70 + "\n")
        f.write(f"Evaluations  = {results['n_evals']}\n")
        f.write(f"Generations  = {results['n_gens']}\n")
        f.write(f"Solve Time   = {results['solve_time']:.2f} s\n\n")

        f.write("Start point values:\n")
        f.write("-" * 70 + "\n")
        for key, value in params_dict['initial_parameters'].items():
            f.write(f"{key:12s} : {value:.4e}\n")

        
        f.write("EXPORTED FILES:\n")
        f.write("-" * 70 + "\n")
        for key, path in exported_files.items():
            if path is not None:
                f.write(f"{key:12s} : {Path(path).name}\n")
        f.write("\n")
        f.write("=" * 70 + "\n")
    
    exported_files['summary'] = str(summary_path)
    print(f"✓ Exported summary to: {summary_path}")
    
    print(f"\n✅ All results exported to: {output_dir}")
    
    return exported_files



def calculate_excited_carrier_concentration(laser_wavelength_nm, sample_thickness_nm, pulse_energy_measured_nJ, laser_spot_diameter_mm):
    """
    Estimate the photoexcited carrier density injected by a single laser pulse.

    Assumes every absorbed photon generates exactly one electron-hole pair and
    that the sample absorbs the full pulse (no reflection or transmission loss).
    For partial absorption, multiply the result by the absorptance fraction.

    Parameters
    ----------
    laser_wavelength_nm : float
        Laser wavelength [nm].  Used to compute the photon energy E = hc/λ.
    sample_thickness_nm : float
        Absorber layer thickness [nm].  Converts areal photon density to
        volumetric carrier density.
    pulse_energy_measured_nJ : float
        Measured energy per pulse [nJ].
    laser_spot_diameter_mm : float
        1/e² (full) laser spot diameter [mm].  The circular spot area is
        A = π/4 · d².

    Returns
    -------
    float
        Excited carrier density immediately after the pulse [cm⁻³].
    """
    # Example calculation:
    # Convert laser power mW to W
    A_spot_cm2=np.pi/4*(laser_spot_diameter_mm*1e-1)**2
    fluence = pulse_energy_measured_nJ/A_spot_cm2
    laser_photon_energy_nJ=1e9*H_JS*C/(1e-9 * laser_wavelength_nm)
    photons_area_pulse= fluence/laser_photon_energy_nJ
    photons_volume_pulse = photons_area_pulse / (sample_thickness_nm *1e-7) #photons/cm^3    
    return photons_volume_pulse 



def trim_and_normalize(df):
    """
    Trim a TRPL transient to start at the PL peak, normalize PL to 1, and
    shift the time index so that the peak is at t = 0.

    Parameters
    ----------
    df : pd.DataFrame
        Raw TRPL DataFrame.  The index must be time in nanoseconds.
        Must contain a ``'pl'`` column with PL intensity values.

    Returns
    -------
    pd.DataFrame
        Trimmed and normalized DataFrame.  The ``'pl'`` column is divided by
        its maximum value.  The time index is shifted so that the peak
        appears at index 0.  All data before the peak are discarded.
    """
    max_idx = df["pl"].idxmax()    # Find index of max amplitude
    df["pl"] = df["pl"] / df["pl"].loc[max_idx]
    trimmed = df.loc[max_idx:]                # Discard everything before the peak
    trimmed.index = trimmed.index - trimmed.index[0] #shift data to 0
    return trimmed


def normalize(df):
    """
    Trim a TRPL transient to start at the PL peak and normalize PL to 1,
    but keep the original absolute time values in the index (no shift to 0).

    Identical to :func:`trim_and_normalize` except that the time index is
    not shifted — the peak time retains its original value.  Use this when
    the absolute timing relative to the laser pulse is needed downstream.

    Parameters
    ----------
    df : pd.DataFrame
        Raw TRPL DataFrame.  The index must be time in nanoseconds.
        Must contain a ``'pl'`` column with PL intensity values.

    Returns
    -------
    pd.DataFrame
        Trimmed and normalized DataFrame.  ``'pl'`` is divided by its
        maximum value.  Data before the peak are discarded.  The time index
        retains its original (unshifted) values.
    """
    max_idx = df["pl"].idxmax()    # Find index of max amplitude
    df["pl"] = df["pl"] / df["pl"].loc[max_idx]
    trimmed = df.loc[max_idx:]                # Discard everything before the peak
    trimmed.index = trimmed.index #shift data to 0
    return trimmed



def t_pl_to_qfls_tau(t_pl: pd.DataFrame, n_pulse: float, Eg: float):
    """
    Convert a (time, PL) TRPL transient into (QFLS, τ_diff) representation.

    The QFLS is derived from PL intensity via:

        QFLS(t) ≈ 2 kT ln(n_pulse / ni) + kT ln(PL(t) / PL_max)

    which assumes equal electron and hole densities (n ≈ p ≈ n_pulse at t=0)
    and uses the constants NC and NV from ``constants.py``.

    The differential lifetime is computed from the logarithmic derivative of PL:

        τ_diff = −f / (d ln PL / dt)

    where f = 2 (ambipolar factor for equal electron/hole densities).

    Parameters
    ----------
    t_pl : pd.DataFrame
        TRPL DataFrame with time **in nanoseconds** as the index and a
        ``'pl'`` column containing normalised PL intensity (peak = 1).
        The data should already be trimmed to start at the PL peak.
    n_pulse : float
        Photoexcited carrier density at t = 0 [cm⁻³].  Used to set the
        absolute QFLS offset.
    Eg : float
        Bandgap energy [eV].  Used to compute the intrinsic carrier
        concentration ni.

    Returns
    -------
    pd.DataFrame
        Same index as *t_pl* (time in ns).  Columns:
        ``'pl'`` (clipped to ≥ 1e-20), ``'qfls'`` [eV], ``'tau_diff'`` [s].
    """
    t = t_pl.index.to_numpy() * 1e-9  # Convert ns → s for derivative
    pl = t_pl["pl"].to_numpy()

    # Avoid zero or negative PL values to prevent log issues
    pl = np.clip(pl, 1e-20, None)

    f = 2.0
    log_pl = np.log(pl)

    # Gradient of log(PL) w.r.t time
    dlogpl_dt = np.gradient(log_pl, t, edge_order=2)
    tau_diff = -f / dlogpl_dt  # τ_diff in seconds

    ni = np.sqrt(NC * NV * np.exp(-Eg / VT))  
    qfls = VT * np.log(pl / np.nanmax(pl)) + 2 * VT * np.log(n_pulse / ni)

    df = pd.DataFrame({
        "pl": pl,
        "qfls": qfls,
        "tau_diff": tau_diff
    }, index=t_pl.index)
    #df["tau_diff"] = df["tau_diff"].replace([np.inf, -np.inf], np.nan)
    #df = df.dropna() #cleans row where tau_diff is NaN

    return df


def qfls_tau_to_t_pl(df_data: pd.DataFrame):
    """
    Reconstruct a (time, PL) transient from a (QFLS, τ_diff) dataset.

    This is the approximate inverse of :func:`t_pl_to_qfls_tau`.  The PL is
    reconstructed as PL ∝ exp(QFLS / kT), and cumulative time is computed by
    integrating the differential lifetime:

        dt ≈ −(τ_diff / 2) · dPL / PL

    The QFLS and τ_diff arrays are reversed before integration so that the
    earliest time point (lowest QFLS) comes first.

    Parameters
    ----------
    df_data : pd.DataFrame
        DataFrame containing at least:
        ``'qfls'`` [eV] — quasi-Fermi level splitting, and
        ``'tau_diff'`` [s] — differential carrier lifetime.
        Rows should be ordered from high QFLS (early time) to low QFLS (late time),
        as produced by :func:`t_pl_to_qfls_tau`.

    Returns
    -------
    pd.DataFrame
        Columns ``'time'``, ``'pl'``, ``'qfls'``, ``'tau_diff'`` with
        ``'time'`` as the index.  PL is normalised to 1.  Time is in the
        same units as τ_diff (seconds if τ_diff is in seconds).
    """
# ---- Vectorized time and PL ----
    tau_diff = df_data["tau_diff"].values  # convert to NumPy array
    qfls = df_data["qfls"].values 
    n_timesteps = len(qfls) 
    qfls_rev = qfls[::-1]                    # 1D, same for all samples
    pl_base = np.exp(qfls_rev / VT) - 1
    pl_base /= np.max(pl_base)               # normalize
    pl = pl_base

    tau_rev = tau_diff[::-1] 

    # Compute cumulative time
    dt = - (tau_rev[:-1] / 2) * (pl[1:] - pl[:-1]) / pl[:-1]
    t = np.concatenate([[0], np.cumsum(dt)])  # single column of length n_timesteps
    df = pd.DataFrame({
        "time": t,
        "pl": pl,
        "qfls": qfls_rev,
        "tau_diff": tau_rev
    }).set_index("time")
    return df

def minmax_scale(tau_diff, tau_min, tau_max):
    """
    Apply min-max normalisation to scale values into [0, 1].

    Parameters
    ----------
    tau_diff : array-like
        Values to normalise.  Typically differential lifetime [s].
    tau_min : float
        Value that maps to 0.
    tau_max : float
        Value that maps to 1.

    Returns
    -------
    np.ndarray
        Normalised values in [0, 1] (values outside the range are not clipped).
    """
    tau_diff_norm = (tau_diff - tau_min)/(tau_max - tau_min)
    return tau_diff_norm


def reverse_minmaxscale(tau, tau_min, tau_max):
    """
    Invert min-max normalisation to recover original-scale values.

    Parameters
    ----------
    tau : array-like
        Normalised values in [0, 1] as produced by :func:`minmax_scale`.
    tau_min : float
        Original minimum (mapped to 0 during normalisation).
    tau_max : float
        Original maximum (mapped to 1 during normalisation).

    Returns
    -------
    np.ndarray
        Values in the original physical scale.
    """
    return tau.copy()*(tau_max - tau_min) + tau_min


def first_not_none(*args):
    """
    Return the first argument that is not None, or None if all are None.

    Parameters
    ----------
    *args
        Any number of positional arguments.

    Returns
    -------
    object or None
        The first non-None value encountered, scanning left to right.
    """
    return next((a for a in args if a is not None), None)

def spline_torch(df, n_points=11, degree=3, n_pred=256):
    """
    Fit a constrained B-spline to a TRPL transient in log-log space and
    return a smooth, monotone prediction on a uniform log-spaced grid.

    The spline is fitted in (log10 time, log10 PL) space with two
    constraints enforced via the ``splinetorch`` library:

    * **Monotone decreasing**: the first derivative is forced ≤ 0,
      preventing unphysical oscillations.
    * **Pinned start point**: the spline passes exactly through the first
      data point so that the peak value is preserved.

    After fitting, predictions are made on a log-spaced time grid spanning
    the original data range, then converted back to linear time and PL.
    The time axis is shifted so that the earliest point is at t = 0.

    Parameters
    ----------
    df : pd.DataFrame
        TRPL DataFrame with time **in nanoseconds** as the index and a
        ``'pl'`` column.  Data must already be trimmed to the PL peak and
        normalised (peak = 1).  The first time point need not be exactly 0 —
        the function internally shifts the time axis to start at 1 ns before
        fitting.
    n_points : int, optional
        Number of equally-spaced knots for the B-spline (default 11).
        More knots allow tighter fitting to the data but risk overfitting.
    degree : int, optional
        Spline polynomial degree (default 3 = cubic).  Currently the
        ``BSpline`` backend always uses degree 3 regardless of this value.
    n_pred : int, optional
        Number of prediction points on the output log-spaced time grid
        (default 256).

    Returns
    -------
    pd.DataFrame
        Smoothed transient with time (shifted to start at 0 ns) as the
        index and a ``'pl'`` column.  Rows with PL > 1.01 are dropped to
        remove any artefacts at the boundary.
    """
    time = df.index.to_numpy(dtype=float)[:] #time
    pl   = df["pl"].to_numpy(dtype=float)[:]

    # Shift the first point instead of dropping
    time = time - time[0] + 1 #shift everything to start at 1ns


    # Convert to log10
    x = torch.tensor(np.log10(time), dtype=torch.float32).view(-1,1)
    y = torch.tensor(np.log10(pl), dtype=torch.float32).view(-1,1)

    # Knot setup
    xmax = torch.max(x)
    xmin = torch.min(x)
    knots = torch.linspace(xmin, xmax, steps=n_points)
    knots_t = torch.tensor(knots, dtype=torch.float32)


    spline = BSpline(x=x, y=y, knots=knots_t,degree=3)
    point_constraints = { 0: {'=': (torch.tensor([[x[0], ]]), torch.tensor([[y[0]]]))} }
    derivative_constraints = {1: {'<=': torch.tensor(0.0)}}  # 1st derivative <= 0

    # Force monotone decreasing function

    spline.fit(x, y, derivative_constraints=derivative_constraints, point_constraints=point_constraints)

    x_pred = torch.log10(torch.logspace(
        torch.tensor(np.log10(time[0])),
        torch.tensor(np.log10(time[-1])),
        n_pred
        )
    )
    x_pred = x_pred.view(-1, 1)
    y_pred = spline.predict(x_pred)

    x_export= 10**x_pred.numpy().flatten()
    x_export_shifted = x_export - x_export[0]
    y_export = 10**y_pred.numpy().flatten()
    
    df_spline = pd.DataFrame({
        "time": x_export_shifted,
        "pl": y_export
    }).set_index("time")
    mask = df_spline["pl"] <= 1.01
    df_spline = df_spline[mask]
    return df_spline

