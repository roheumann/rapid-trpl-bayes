import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from constants import *
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from typing import Tuple, List

VT = 0.0258  # example value, replace with your thermal voltage

def determine_trap_regime(
    df: pd.DataFrame,
    window_size: int = 3,
    deep_nid_max_min: List[float] = [7.5, 50],
    shallow_nid_max_min: List[float] = [1.8, 2.2],
    contiguous_window_size: int = 5,
    diffusion_correction: float = 0.06
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Determine deep and shallow trap regimes from a TRPL dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing at least columns:
        - 'tau_diff': differential lifetime values
        - 'qfls': quasi-Fermi level splitting values
    window_size : int, optional
        Number of points for sliding window averaging (default is 3).
    deep_nid_max_min : list of float, optional
        Minimum and maximum ideality factor (nid) to define the deep trap regime (default [7.5, 50]).
    shallow_nid_max_min : list of float, optional
        Minimum and maximum ideality factor (nid) to define the shallow trap regime (default [1.8, 2.2]).
    contiguous_window_size : int, optional
        Minimum number of consecutive points to consider a coherent trap regime (default 5).

    Returns
    -------
    mask_deep_clean : np.ndarray of bool
        Boolean mask indicating coherent deep trap regime points.
    mask_shallow_clean : np.ndarray of bool
        Boolean mask indicating coherent shallow trap regime points.
    """

    # Compute sliding window averages
    tau_window = np.mean(sliding_window_view(df.tau_diff, window_size), axis=-1)
    qfls_window = np.mean(sliding_window_view(df.qfls, window_size), axis=-1)

    # Estimate curvature and normalized trap density
    curvature_window = -np.gradient(np.log(tau_window), qfls_window)
    nid_window = 1 / (VT * curvature_window)

    # Initial masks based on nid ranges
    mask_deep = (nid_window >= deep_nid_max_min[0]) & (nid_window <= deep_nid_max_min[1])
    mask_shallow = (
        (nid_window >= shallow_nid_max_min[0]) &
        (nid_window <= shallow_nid_max_min[1]) &
        (qfls_window <= np.max(qfls_window) - diffusion_correction)  # correction due to diffusion
    )

    # Helper: identify contiguous regions of True values
    def contiguous_windows(mask: np.ndarray, min_len: int = 5) -> List[np.ndarray]:
        idx = np.where(mask)[0]
        if len(idx) == 0:
            return []
        splits = np.where(np.diff(idx) > 1)[0] + 1
        groups = np.split(idx, splits)
        return [g for g in groups if len(g) >= min_len]

    # Helper: convert list of windows back into boolean mask
    def windows_to_mask(windows: List[np.ndarray], size: int) -> np.ndarray:
        mask = np.zeros(size, dtype=bool)
        for w in windows:
            mask[w] = True
        return mask

    # Apply contiguous window filtering
    deep_windows = contiguous_windows(mask_deep, min_len=contiguous_window_size)
    shallow_windows = contiguous_windows(mask_shallow, min_len=contiguous_window_size)

    # Convert windows back to boolean masks
    mask_deep_clean = windows_to_mask(deep_windows, len(nid_window))
    mask_shallow_clean = windows_to_mask(shallow_windows, len(nid_window))

    # sliding_window_view shrinks the array by (window_size - 1); pad back to original length
    pad_before = (window_size - 1) // 2
    pad_after = window_size - 1 - pad_before
    mask_deep_clean = np.pad(mask_deep_clean, (pad_before, pad_after), constant_values=False)
    mask_shallow_clean = np.pad(mask_shallow_clean, (pad_before, pad_after), constant_values=False)

    return mask_deep_clean, mask_shallow_clean

def start_point_estimator(
    df_trpl: pd.DataFrame,
    trap_regime_mask: np.ndarray,
    Eg: float,
    trap_type: str = "shallow",
    photodoping: bool = True,
    dEt: float = 0.05,
    Nt_n1_factor: float = 10,
    Ntrap: float = 1e14
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate initial parameters (start points) for shallow or deep trap regimes in TRPL.

    Parameters
    ----------
    df_trpl : pd.DataFrame
        TRPL dataset containing columns:
        - 'qfls' : quasi-Fermi level splitting
        - 'tau_diff' : differential carrier lifetime
    trap_regime_mask : np.ndarray
        Boolean mask indicating the Fermi-level-range of the trap regime to fit
    Eg : float
        Bandgap energy in eV
    trap_type : str, optional
        "shallow" or "deep" trap (default "shallow")
    photodoping : bool, optional
        True if photodoping traps should be considered (default True)
    dEt : float, optional
        Energy difference for shallow trap (default 0.05 eV)
    Nt_n1_factor : float, optional
        Factor to scale trap density relative to thermal occupation n1 (default 10)
    Ntrap : float, optional
        Trap density for deep trap (default 1e14)

    Returns
    -------
    krad : np.ndarray
        Radiative recombination rate (s^-1)
    Etrap : np.ndarray
        Trap energy (eV)
    Ntrap : np.ndarray
        Trap density (cm^-3)
    taun : np.ndarray
        Electron lifetime (s)
    taup : np.ndarray
        Hole lifetime (s)
    """

    # Subset TRPL dataframe to the selected trap regime
    df_fit_regime = df_trpl[trap_regime_mask].copy()
    df_fit_regime.sort_values('qfls', ascending=True, inplace=True)

    # Set a default radiative recombination rate
    krad = 1e-11

    if trap_type == "shallow":
        # -----------------------------
        # Initial guesses for shallow trap
        # -----------------------------
        x = df_fit_regime.qfls.values
        y = df_fit_regime.tau_diff.values

        # slope term for initial tau0 estimation
        slope_term = x / (2 * VT)
        tau_0_shallow = np.exp(np.mean(np.log(y) + slope_term))
        print("Fitted tau0 shallow:", tau_0_shallow)

        # intrinsic carrier concentration
        ni = np.sqrt(NC * NV) * np.exp(-Eg / (2 * VT))

        # Trap energy and occupation depending on photodoping
        Etrap = Eg - dEt
        n1 = NC * np.exp((Etrap - Eg) / VT)
        Ntrap = Nt_n1_factor * n1

        a = np.sqrt(n1 * Ntrap)/ni if photodoping else n1/ni
        b = krad * n1
        taup = tau_0_shallow / (a - b * tau_0_shallow)
        taun = taup

    elif trap_type == "deep":
        # -----------------------------
        # Deep trap: simple approximation
        # -----------------------------
        Etrap = 0.5 * Eg
        tau_SRH = df_fit_regime.tau_diff.mean()
        taun = tau_SRH
        taup = tau_SRH
        print("Fitted tau_SRH:", tau_SRH)

    # -----------------------------
    # Plot experimental data and fit
    # -----------------------------
    plt.close('all')
    fig, ax = plt.subplots(figsize=(7,5))

    # Experimental TRPL
    ax.semilogy(
        df_trpl['qfls'],
        df_trpl["tau_diff"],
        'o',
        markersize=6,
        alpha=0.8,
        color="black",
        label="Experiment"
    )

    # Fit curve
    if trap_type == "shallow":
        ax.semilogy(
            df_fit_regime.qfls,
            tau_0_shallow * np.exp(-df_fit_regime.qfls / (2*VT)),
            color='red',
            linewidth=2,
            label="Shallow trap fit"
        )
    elif trap_type == "deep":
        ax.semilogy(
            df_fit_regime.qfls,
            np.full_like(df_fit_regime.qfls, tau_SRH),
            color='red',
            linewidth=2,
            label="Deep trap fit"
        )

    ax.set_xlabel(r"$\Delta E_\mathrm{F}$ (eV)", fontsize=12)
    ax.set_ylabel(r"$\tau_\mathrm{diff}$ (s)", fontsize=12)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend(fontsize=10)
    fig.tight_layout()
    plt.show()

    return (np.asarray(krad),
            np.asarray(Etrap),
            np.asarray(Ntrap),
            np.asarray(taun),
            np.asarray(taup))

def start_point_estimator_old(df_trpl, qfls_min, qfls_max, Eg, trap_type="shallow", photodoping=True, dEt = 0.05, Ntrap = 1e14):


    df_fit_regime = df_trpl[(df_trpl.qfls >= qfls_min) & (df_trpl.qfls <= qfls_max)]
    df_fit_regime.sort_values('qfls', ascending=True)
    krad = 1e-11
    if trap_type == "shallow":
        # initial guesses
        #p0 = [1e4]  # [tau0_guess]

        x = df_fit_regime.qfls.values
        y = df_fit_regime.tau_diff.values
        slope_term = x / (2*VT)

        tau_0_shallow = np.exp(np.mean(np.log(y) + slope_term))

        #params, cov = curve_fit(model_fixed_slope, df_fit_regime.qfls, df_fit_regime.tau_diff, p0=p0)
        #print(f"params {params}")
        #tau_0_shallow = params[0]
        print("Fitted tau0 shallow:", tau_0_shallow)
        ni = np.sqrt(NC * NV) * np.exp(-Eg / (2 * VT)) 
        if photodoping==True:
            Etrap = Eg - dEt #1.409/2 + Eg - kT*np.log(Nc/ni)
            n1 = NC  * np.exp((Etrap-Eg)/VT)
            Ntrap =  10 * n1
            a = np.sqrt(n1*Ntrap)/ni
            b = krad*n1
            taup = tau_0_shallow/(a-b*tau_0_shallow)
            taun = taup
        else:
            Etrap = Eg - dEt #1.409/2 + Eg - kT*np.log(Nc/ni)
            n1 = NC  * np.exp((Etrap-Eg)/VT)
            Ntrap =  0.1 * n1
            a = n1/ni
            b = krad*n1
            taup = tau_0_shallow/(a-b*tau_0_shallow)
            a = n1/ni
            taun = taup
        

    if trap_type == "deep":
        Etrap = 0.5*Eg
        Ntrap = Ntrap
        tau_SRH = df_fit_regime.tau_diff.mean()
        taun = tau_SRH
        taup = taun 
        print("Fitted tau_SRH:", tau_SRH)

    plt.close('all')  # Prevents plotting over previous notebook figures

    fig, ax = plt.subplots(figsize=(7,5))

    # Experimental data
    ax.semilogy(
        df_trpl['qfls'],
        df_trpl["tau_diff"],
        'o',
        markersize=6,
        label="Experiment",
        alpha=0.8,
        color = "black"
    )
    if trap_type == "shallow":
        # Shallow trap fit
        ax.semilogy(
            df_fit_regime.qfls,
            tau_0_shallow * np.exp(-df_fit_regime.qfls / (2*VT)),
            color='red',
            linewidth=2,
            label="Shallow trap fit"
        )
    elif trap_type == "deep":
        # Deep trap fit
        ax.semilogy(
        df_fit_regime.qfls,
        np.full_like(df_fit_regime.qfls, tau_SRH),
        color='red',
        linewidth=2,
        label="Deep trap fit"
        )

    ax.set_xlabel(r"$\Delta E_\mathrm{F}$ (s)", fontsize=12)
    ax.set_ylabel(r"$\tau_\mathrm{diff}$ (s)", fontsize=12)

    ax.grid(True, which="both", linestyle="--", alpha=0.4)

    ax.legend(fontsize=10)

    fig.tight_layout()

    plt.show()
    
    return np.asarray(krad), np.asarray(Etrap), np.asarray(Ntrap), np.asarray(taun), np.asarray(taup)


def model_fixed_slope(x, tau0):
    return tau0 * np.exp(-x / (2*VT))