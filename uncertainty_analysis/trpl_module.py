"Authors Robin Heumann and Chris Dreessen, 22/01/2026"

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
import matplotlib.pyplot as plt


def calcequil_general(Eg, Nc, Nv, Nt=None, Et=None, T=300):
    """
    CALCEQUIL_GENERAL
    Computes thermal equilibrium carrier densities (n0, p0) and trap occupancies (nt0)
    for an arbitrary number of trap states using charge neutrality.
    
    INPUTS:
        Eg  - bandgap [eV]
        Nc  - conduction band effective DOS [cm^-3]
        Nv  - valence band effective DOS [cm^-3]
        Nt  - array of trap densities [cm^-3] (optional, default=0)
        Et  - array of trap energies relative to Ev [eV] (optional, default=Eg/2)
        T   - temperature [K] (optional, default=300 K)
    
    OUTPUTS:
        n0   - equilibrium electron density [cm^-3]
        p0   - equilibrium hole density [cm^-3]
        nt0  - array of equilibrium trapped carrier densities [cm^-3]
    """
    # Handle defaults
    if Nt is None:
        Nt = np.array([0.0])
    if Et is None:
        Et = np.array([Eg/2])
    
    # Ensure Nt and Et are numpy arrays
    Nt = np.atleast_1d(Nt)
    Et = np.atleast_1d(Et)
    
    # Constants
    kT = 8.617e-5 * T  # [eV]
    
    # Handle trivial case (no traps)
    if np.all(Nt == 0):
        ni = np.sqrt(Nc * Nv * np.exp(-Eg/kT))
        n0 = ni
        p0 = ni
        nt0 = np.zeros_like(Nt)
        Ef = Eg/2  # intrinsic Fermi level
        return n0, p0, nt0, Ef
    
    # Initial guess: midgap
    Ef_guess = Eg/2
    
    # Solve for Ef using fsolve
    result = brentq(neutrality_equation, 0.0, Eg, 
                args=(Eg, Nc, Nv, Nt, Et, kT), 
                xtol=1e-12, full_output=True)
    Ef = result[0]
    
    # Once Ef is found, compute equilibrium populations
    n0 = Nc * np.exp(-(Eg - Ef)/kT)
    p0 = Nv * np.exp(-Ef/kT)
    nt0 = Nt / (1 + np.exp((Et - Ef)/kT))
    
    return n0, p0, nt0


def neutrality_equation(Ef, Eg, Nc, Nv, Nt, Et, kT):
    """
    Charge-neutrality residual used as the root function for equilibrium Ef.

    Evaluates p(Ef) − n(Ef) − Σ nt_j(Ef), which equals zero when the Fermi
    level Ef satisfies charge neutrality at thermal equilibrium.  Passed to
    ``scipy.optimize.brentq`` in :func:`calcequil_general`.

    Parameters
    ----------
    Ef : float
        Trial Fermi level position relative to the valence band [eV].
    Eg : float
        Bandgap [eV].
    Nc : float
        Effective density of states in conduction band [cm⁻³].
    Nv : float
        Effective density of states in valence band [cm⁻³].
    Nt : ndarray
        Trap densities [cm⁻³], shape (M,).
    Et : ndarray
        Trap energy levels relative to VB [eV], shape (M,).
    kT : float
        Thermal energy [eV].

    Returns
    -------
    float
        Residual p − n − Σ nt [cm⁻³].  Zero at the equilibrium Fermi level.
    """
    n = Nc * np.exp(-(Eg - Ef)/kT)
    p = Nv * np.exp(-Ef/kT)
    nt = Nt / (1 + np.exp((Et - Ef)/kT))
    res = p - n - np.sum(nt)  # neutrality condition
    return res


def dyna_SRH_reduced_general(t, x, pars):
    """
    Right-hand side of the reduced SRH carrier-dynamics ODE system.

    The state vector ``x`` tracks only the free-electron density and the
    trap occupancies; the hole density is reconstructed algebraically from
    charge neutrality (p = n + Σ nt + Kneutral), which eliminates one ODE.

    For each trap *j* the net electron and hole capture rates are::

        Rn_j = bn_j · n · (Nt_j − nt_j) − en_j · nt_j
        Rp_j = bp_j · p · nt_j − ep_j · (Nt_j − nt_j)

    The full system is::

        dn/dt   = −krad·(n·p − ni²) − Σ_j Rn_j + bias
        dnt_j/dt = Rn_j − Rp_j          for each trap j

    Parameters
    ----------
    t : float
        Current time [s].  Not used in the equations (autonomous system),
        but required by the ``solve_ivp`` calling convention.
    x : ndarray, shape (1 + M,)
        Current state vector.  ``x[0]`` is the free-electron density n
        [cm⁻³]; ``x[1:]`` are the trap occupancies nt [cm⁻³] for each of
        the M traps.
    pars : dict
        Parameter dictionary with the following keys:
        ``'Nt'`` [ndarray, cm⁻³], ``'bn'`` [ndarray, cm³/s],
        ``'bp'`` [ndarray, cm³/s], ``'en'`` [ndarray, s⁻¹],
        ``'ep'`` [ndarray, s⁻¹], ``'krad'`` [float, cm³/s],
        ``'ni'`` [float, cm⁻³], ``'Kneutral'`` [float, cm⁻³],
        ``'bias'`` [float, cm⁻³ s⁻¹].

    Returns
    -------
    ndarray, shape (1 + M,)
        Time derivatives [dn/dt, dnt_1/dt, …] in [cm⁻³ s⁻¹].
    """
    n = x[0]
    nt = x[1:]
    
    Nt = pars['Nt']
    bn = pars['bn']
    bp = pars['bp']
    en = pars['en']
    ep = pars['ep']
    
    p = n + np.sum(nt) + pars['Kneutral']
    
    Rrad = pars['krad'] * (n * p - pars['ni']**2)
    
    Rn = bn * n * (Nt - nt) - en * nt
    Rp = bp * p * nt - ep * (Nt - nt)
    
    dn = -Rrad - np.sum(Rn) + pars['bias']
    dnt = Rn - Rp
    
    return np.concatenate([[dn], dnt])

def ode_func(t, x, pars):
        return dyna_SRH_reduced_general(t, x, pars)
    
def stop_event(t, x, pars):
    """Event function for solve_ivp - only takes (t, x) as arguments"""
    n = x[0]
    nt = x[1:]
    
    # Extract values directly (no dictionary access)
    n_pulse = pars['n_pulse']
    n0_val = pars['n0']
    p0_val = pars['p0']
    Kneutral = pars['Kneutral']

    p = n + np.sum(nt) + Kneutral
    
    # Return value: integration stops when this crosses zero from positive to negative
    # We want to stop when carriers decay significantly
    if n * p / n_pulse**2 <= 1e-20:
        return -1  # Stop
    elif n / n0_val <= 1.5 and p / p0_val <= 1.5:
        return -1  # Stop
    else:
        return 1  # Continue

def solve_transient(tarr, npulse, Eg, Nc, Nv, krad, Nt, Et, taun, taup, T=300, bias=0):
    """
    Simulates the time-dependent carrier and trap dynamics in a
    semiconductor (e.g., perovskite layer) using Shockley-Read-Hall (SRH)
    formalism for an arbitrary number of trap states.
    
    Parameters
    ----------
    tarr : array_like (N,)
        Time array for integration [s]
    npulse : float
        Excess carrier density at t=0 [cm^-3]
    Eg : float
        Bandgap energy [eV]
    Nc : float
        Effective density of states in CB [cm^-3]
    Nv : float
        Effective density of states in VB [cm^-3]
    krad : float
        Radiative recombination coefficient [cm^3/s]
    Nt : array_like (M,)
        Vector of trap densities [cm^-3]
    Et : array_like (M,)
        Vector of trap energies relative to VB [eV]
    taun : array_like (M,)
        Vector of electron lifetimes [s]
    taup : array_like (M,)
        Vector of hole lifetimes [s]
    T : float, optional
        Temperature [K], default = 300 K
    bias : float, optional
        External generation rate [cm^-3 s^-1], default = 0
    
    Returns
    -------
    t : ndarray (N,)
        Time array
    n_tr : ndarray (N,)
        Electron density vs. time [cm^-3]
    p_tr : ndarray (N,)
        Hole density vs. time [cm^-3]
    nt_tr : ndarray (N, M)
        Trap occupancies vs. time [cm^-3]
    QFLS_tr : ndarray (N,)
        Quasi-Fermi level splitting vs. time [eV]
    tau_tr : ndarray (N,)
        Differential lifetime from PL decay [s]
    tau_tr_n : ndarray (N,)
        Electron lifetime [s]
    tau_tr_p : ndarray (N,)
        Hole lifetime [s]
    PL_tr : ndarray (N,)
        PL signal ∝ (n·p - ni²)
    e_trapping : ndarray (N, M)
        Electron trapping flux per trap [cm^-3 s^-1]
    e_detrapping : ndarray (N, M)
        Electron detrapping flux per trap [cm^-3 s^-1]
    h_trapping : ndarray (N, M)
        Hole trapping flux per trap [cm^-3 s^-1]
    h_detrapping : ndarray (N, M)
        Hole detrapping flux per trap [cm^-3 s^-1]
    
    Notes
    -----
    - Hole density is reconstructed from charge neutrality
    - Equilibrium initial condition calculated via calcequil_general
    - Differential lifetime computed from derivative of log(PL)
    
    Author: Chris Dreessen and Robin Heumann
    Date: 19/01/2026
    """
    
    # Convert inputs to numpy arrays
    Nt = np.asarray(Nt).flatten()
    Et = np.asarray(Et).flatten()
    taun = np.asarray(taun).flatten()
    taup = np.asarray(taup).flatten()
    bn = 1 / (taun * Nt)    # electron capture coefficients [cm^3/s]
    bp = 1 / (taup * Nt)    # hole capture coefficients [cm^3/s]
    tarr = np.asarray(tarr).flatten()
    
    # === Thermal parameters ===
    kT = 8.617e-5 * T
    ni = np.sqrt(Nc * Nv * np.exp(-Eg / kT))
    
    M = len(Nt)  # Number of traps
    
    # === Equilibrium ===
    n0, p0, nt0 = calcequil_general(Eg, Nc, Nv, Nt, Et, T)
    
    # Emission rates from detailed balance
    n1 = Nc * np.exp((Et - Eg) / kT)
    p1 = Nv * np.exp(-Et / kT)
    en = bn * n1
    ep = bp * p1
    
    # === Initial condition with npulse ===
    # Charge neutrality offset
    Kneutral = 0#p0 - n0 - np.sum(nt0)
    
    
    n_init = n0 + npulse
    nt_init = nt0  # Traps initially unchanged
    x0 = np.concatenate([[n_init], nt_init])
    
    # === Parameters for ODE ===
    pars = {
        'ni': ni,
        'n_pulse': npulse,
        'n0': n0,
        'p0': p0,
        'krad': krad,
        'Nt': Nt,
        'bn': bn,
        'bp': bp,
        'en': en,
        'ep': ep,
        'Kneutral': Kneutral,
        'bias': bias
    }
    
    # === Solve ODE ===

    
    # Mark as terminal event
    stop_event.terminal = True
    stop_event.direction = -1  # Trigger when going from positive to negative
    
    sol = solve_ivp(ode_func, [tarr[0], tarr[-1]], x0, 
                    args=(pars,), method='LSODA',
                    rtol=1.0E-5, atol=1.0E-7, events=stop_event)
    
    t = sol.t
    x = sol.y.T
    
    # === Reconstruct outputs ===
    n_tr = x[:, 0]
    nt_tr = x[:, 1:]
    p_tr = n_tr + np.sum(nt_tr, axis=1) + Kneutral
    
    # PL and QFLS
    PL_tr = n_tr * p_tr - ni**2
    QFLS_tr = kT * np.log(n_tr * p_tr / ni**2)
    #V_start_est = kT * np.log(npulse**2 / ni**2)
    #PL_tr = n_tr * p_tr - ni**2
    #QFLS_tr = V_start_est + kT * np.log(PL_tr / np.max(PL_tr))
    
    
    # Differential lifetime
    grad = np.real(np.gradient(np.log(PL_tr / np.max(PL_tr)), t))
    mask = grad != 0.0
    masked_grad = grad[mask]
    tau_tr = -2.0 / masked_grad

    t = t[mask]*1e9 #ns
    n_tr = n_tr[mask]
    p_tr = p_tr[mask]
    nt_tr = nt_tr[mask]
    QFLS_tr = QFLS_tr[mask]
    PL_tr = PL_tr[mask] / np.max(PL_tr[mask])
    

    # === Trap fluxes ===
    e_trapping = np.zeros((len(t), M))
    e_detrapping = np.zeros((len(t), M))
    h_trapping = np.zeros((len(t), M))
    h_detrapping = np.zeros((len(t), M))
    
    for j in range(M):
        e_trapping[:, j] = bn[j] * n_tr * (Nt[j] - nt_tr[:, j])
        e_detrapping[:, j] = en[j] * nt_tr[:, j]
        h_trapping[:, j] = bp[j] * p_tr * nt_tr[:, j]
        h_detrapping[:, j] = ep[j] * (Nt[j] - nt_tr[:, j])

    
    krad_rate = krad * n_tr * p_tr
    
    # === Lifetimes ===
    R_n = krad * n_tr * p_tr + np.sum(e_trapping - e_detrapping, axis=1)
    R_p = krad * n_tr * p_tr + np.sum(h_trapping - h_detrapping, axis=1)
    
    tau_tr_n = n_tr / R_n
    tau_tr_p = p_tr / R_p
    
    return (t, n_tr, p_tr, nt_tr, QFLS_tr, tau_tr, tau_tr_n, tau_tr_p, PL_tr,
            e_trapping, e_detrapping, h_trapping, h_detrapping, krad_rate)


def plot_sim(experiment, result):
    """
    Quick diagnostic plot comparing experimental TRPL data with a simulation.

    Produces a 2×2 panel figure:

    * Top-left:  PL vs time (semi-log y)
    * Top-right: PL vs time (log-log)
    * Bottom-left: τ_diff vs time (log-log)
    * Bottom-right: τ_diff vs QFLS (semi-log y)

    Parameters
    ----------
    experiment : pd.DataFrame
        Experimental TRPL data.  The index must be time [ns].  Required
        columns: ``'pl'`` (normalised PL intensity), ``'tau_diff'`` [s],
        ``'qfls'`` [eV].
    result : dict
        Result dictionary as returned by :func:`solve_transient` wrapped in
        a results dict, or as returned by the fitting functions.  Must
        contain a ``'trpl_sim'`` key whose value is a DataFrame with
        columns ``'time_tr_ns'`` [ns], ``'pl_tr'``, ``'qfls_tr_eV'`` [eV],
        and ``'tau_tr_s'`` [s].
    """
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    ax = axes[0,0]
    ax.semilogy(experiment.index, experiment.pl, 'o', label="experiment")
    ax.semilogy(result["trpl_sim"]['time_tr_ns'], result["trpl_sim"]['pl_tr'], label="fit")
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("PL (normalized)")
    ax.legend()
    ax.set_xlim(np.nanmin(experiment.index), np.nanmax(experiment.index))
    ax.set_ylim(np.nanmin(experiment.pl), np.nanmax(experiment.pl))

    ax = axes[0,1]
    ax.loglog(experiment.index, experiment.pl, 'o', label="experiment")
    ax.loglog(result["trpl_sim"]['time_tr_ns'], result["trpl_sim"]['pl_tr'], label="fit")
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("PL (normalized)")
    ax.legend()
    ax.set_xlim(np.nanmin(experiment.index), np.nanmax(experiment.index))
    ax.set_ylim(np.nanmin(experiment.pl), np.nanmax(experiment.pl))

    ax = axes[1,0]
    ax.loglog(experiment.index, experiment.tau_diff, 'o', label="experiment")
    ax.loglog(result["trpl_sim"]['time_tr_ns'], result["trpl_sim"]['tau_tr_s'], label="fit")
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("PL (normalized)")
    ax.legend()
    ax.set_xlim(np.nanmin(experiment.index), np.nanmax(experiment.index))
    ax.set_ylim(np.nanmin(experiment.tau_diff), np.nanmax(experiment.tau_diff))

    ax = axes[1,1]
    ax.semilogy(experiment.qfls, experiment.tau_diff, 'o', label="experiment")
    ax.semilogy(result["trpl_sim"]['qfls_tr_eV'], result["trpl_sim"]['tau_tr_s'], label="fit")
    ax.set_xlabel("Quasi Fermi-level splitting (eV)")
    ax.set_ylabel("Diff. decay time (s)")
    ax.legend()
    ax.set_xlim(np.nanmin(experiment.qfls), np.nanmax(experiment.qfls))
    ax.set_ylim(np.nanmin(experiment.tau_diff), np.nanmax(experiment.tau_diff))

    plt.show()