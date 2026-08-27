"Authors Robin Heumann and Chris Dreessen, 22/01/2026"

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from numba import njit


@njit
def dyna_SRH_reduced_general(t, x, krad, ni, Nt, bn, bp, en, ep, Kneutral, bias):
    """
    ODE right-hand side for the reduced SRH system.
    All parameters are passed individually (no dict) so Numba can JIT-compile this.
    """
    n  = x[0]
    nt = x[1:]

    p    = n + np.sum(nt) + Kneutral
    Rrad = krad * (n * p - ni**2)
    Rn   = bn * n * (Nt - nt) - en * nt
    Rp   = bp * p * nt - ep * (Nt - nt)
    dn   = -Rrad - np.sum(Rn) + bias
    dnt  = Rn - Rp

    return np.concatenate((np.array([dn]), dnt))


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
    Nt   = np.asarray(Nt,   dtype=np.float64).flatten()
    Et   = np.asarray(Et,   dtype=np.float64).flatten()
    taun = np.asarray(taun, dtype=np.float64).flatten()
    taup = np.asarray(taup, dtype=np.float64).flatten()
    tarr = np.asarray(tarr, dtype=np.float64).flatten()

    bn = 1.0 / (taun * Nt)    # electron capture coefficients [cm^3/s]
    bp = 1.0 / (taup * Nt)    # hole capture coefficients [cm^3/s]

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
    Kneutral = 0.0
    n_init   = n0 + npulse
    nt_init  = nt0
    x0       = np.concatenate([[n_init], nt_init])

    # === ODE closure (captures local scalars/arrays; dyna_SRH_reduced_general is @njit) ===
    def _ode(t, x):
        return dyna_SRH_reduced_general(t, x, krad, ni, Nt, bn, bp, en, ep, Kneutral, bias)

    def _stop(t, x):
        n  = x[0]
        nt = x[1:]
        p  = n + np.sum(nt) + Kneutral
        if n * p / npulse**2 <= 1e-20:
            return -1
        if n / n0 <= 1.5 and p / p0 <= 1.5:
            return -1
        return 1

    _stop.terminal  = True
    _stop.direction = -1

    # === Solve ODE ===
    sol = solve_ivp(_ode, [tarr[0], tarr[-1]], x0,
                    method='LSODA', rtol=1.0e-5, atol=1.0e-7, events=_stop)

    t = sol.t
    x = sol.y.T

    # === Reconstruct outputs ===
    n_tr  = x[:, 0]
    nt_tr = x[:, 1:]
    p_tr  = n_tr + np.sum(nt_tr, axis=1) + Kneutral

    # PL and QFLS
    PL_tr   = n_tr * p_tr - ni**2
    QFLS_tr = kT * np.log(n_tr * p_tr / ni**2)

    # Differential lifetime
    grad        = np.real(np.gradient(np.log(PL_tr / np.max(PL_tr)), t))
    mask        = np.isfinite(grad)
    masked_grad = grad[mask]
    tau_tr      = -2.0 / masked_grad

    t       = t[mask]
    n_tr    = n_tr[mask]
    p_tr    = p_tr[mask]
    nt_tr   = nt_tr[mask]
    QFLS_tr = QFLS_tr[mask]
    PL_tr   = PL_tr[mask]

    # === Trap fluxes ===
    e_trapping   = np.zeros((len(t), M))
    e_detrapping = np.zeros((len(t), M))
    h_trapping   = np.zeros((len(t), M))
    h_detrapping = np.zeros((len(t), M))

    for j in range(M):
        e_trapping[:, j]   = bn[j] * n_tr * (Nt[j] - nt_tr[:, j])
        e_detrapping[:, j] = en[j] * nt_tr[:, j]
        h_trapping[:, j]   = bp[j] * p_tr * nt_tr[:, j]
        h_detrapping[:, j] = ep[j] * (Nt[j] - nt_tr[:, j])

    krad_rate = krad * n_tr * p_tr

    # === Lifetimes ===
    R_n = krad * n_tr * p_tr + np.sum(e_trapping   - e_detrapping, axis=1)
    R_p = krad * n_tr * p_tr + np.sum(h_trapping   - h_detrapping, axis=1)

    tau_tr_n = n_tr / R_n
    tau_tr_p = p_tr / R_p

    return (t, n_tr, p_tr, nt_tr, QFLS_tr, tau_tr, tau_tr_n, tau_tr_p, PL_tr,
            e_trapping, e_detrapping, h_trapping, h_detrapping, krad_rate)


def calcequil_general(Eg, Nc, Nv, Nt=None, Et=None, T=300):
    """
    Computes thermal equilibrium carrier densities (n0, p0) and trap occupancies (nt0)
    for an arbitrary number of trap states using charge neutrality.
    """
    if Nt is None:
        Nt = np.array([0.0])
    if Et is None:
        Et = np.array([Eg / 2])

    Nt = np.atleast_1d(np.asarray(Nt, dtype=np.float64))
    Et = np.atleast_1d(np.asarray(Et, dtype=np.float64))

    kT = 8.617e-5 * T

    if np.all(Nt == 0):
        ni = np.sqrt(Nc * Nv * np.exp(-Eg / kT))
        return ni, ni, np.zeros_like(Nt)

    result = brentq(neutrality_equation, 0.0, Eg,
                    args=(Eg, Nc, Nv, Nt, Et, kT),
                    xtol=1e-12, full_output=True)
    Ef = result[0]

    n0  = Nc * np.exp(-(Eg - Ef) / kT)
    p0  = Nv * np.exp(-Ef / kT)
    nt0 = Nt / (1 + np.exp((Et - Ef) / kT))

    return n0, p0, nt0


@njit
def neutrality_equation(Ef, Eg, Nc, Nv, Nt, Et, kT):
    """Neutrality residual (pure math — JIT compiled)."""
    n  = Nc * np.exp(-(Eg - Ef) / kT)
    p  = Nv * np.exp(-Ef / kT)
    nt = Nt / (1 + np.exp((Et - Ef) / kT))
    return p - n - np.sum(nt)
