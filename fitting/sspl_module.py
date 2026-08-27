"Author Robin Heumann and Chris Dreessen, 22/01/2026"
import numpy as np
from scipy.optimize import fsolve, brentq

def solve_steady_state(Npoints, Eg, Nc, Nv, krad, Nt, Et, taun, taup, T=300):
    """
    solve_steady_state
    
    This function solves the **steady-state carrier and trap populations** in
    a semiconductor under uniform generation for an **arbitrary number of
    trap states**, using the Shockley–Read–Hall (SRH) formalism. The steady
    state is defined by **charge neutrality** and **steady carrier balance**
    (generation = recombination). The solution is obtained by sweeping the
    electron concentration over a logarithmic grid and solving for the hole
    density using Newton–Raphson iteration.
    
    For each carrier density, the function computes trap occupancies, radiative
    and SRH recombination rates, quasi-Fermi level splitting (QFLS),
    photoluminescence quantum yield (PLQY), and various lifetime definitions.
    Additionally, it provides electron/hole trapping and detrapping fluxes for
    each trap.
    
    -----------------------
    INPUTS:
        Npoints [int]      - Number of points in logarithmic electron density sweep
        Eg      [float]    - Bandgap energy [eV]
        Nc      [float]    - Effective density of states in CB [cm^-3]
        Nv      [float]    - Effective density of states in VB [cm^-3]
        krad    [float]    - Radiative recombination coefficient [cm^3/s]
        Nt      [array]    - Vector of trap densities [cm^-3]
        Et      [array]    - Vector of trap energies relative to VBM [eV]
        taun    [array]    - Vector of electron lifetimes [s]
        taup    [array]    - Vector of hole lifetimes [s]
        T       [float]    - (Optional) Temperature [K], default = 300 K
    
    -----------------------
    OUTPUTS:
        narr        [array]  - Electron density sweep [cm^-3]
        parr        [array]  - Hole densities from charge neutrality [cm^-3]
        ntmat       [array]  - Trap occupancies for each trap [cm^-3]
        Rtotarr     [array]  - Total recombination rate [cm^-3 s^-1]
        Rradarr     [array]  - Radiative recombination rate [cm^-3 s^-1]
        Rsrhmat     [array]  - Trap-specific SRH recombination rates [cm^-3 s^-1]
        QFLS        [array]  - Quasi-Fermi level splitting [eV]
        PLQY        [array]  - Photoluminescence quantum yield (Rrad/Rtot)
        tau_ss      [array]  - Effective lifetime sqrt(n·p)/Rtot [s]
        tau_ss_n    [array]  - Electron lifetime n/Rtot [s]
        tau_ss_p    [array]  - Hole lifetime p/Rtot [s]
        nid         [array]  - Ideality factor extracted from QFLS vs Rtot
        e_trapping  [array]  - Electron trapping flux per trap [cm^-3 s^-1]
        e_detrapping[array]  - Electron detrapping flux per trap [cm^-3 s^-1]
        h_trapping  [array]  - Hole trapping flux per trap [cm^-3 s^-1]
        h_detrapping[array]  - Hole detrapping flux per trap [cm^-3 s^-1]
    
    -----------------------
    NOTES:
    - The function uses `solve_p_newton_general` to determine the hole density
    for each n, ensuring charge neutrality including trap charge.
    - Trap occupancies follow the SRH steady-state occupation probability.
    - This routine provides steady-state characteristics for direct
    comparison with time-dependent simulations from solve_transient.
    - The n-sweep typically spans from equilibrium n0 to Nc.
    
    Author: Chris Dreessen and Robin Heumann
    Date: 19/01/2026
    """
    
    # Ensure inputs are numpy arrays
    Nt = np.atleast_1d(Nt)
    Et = np.atleast_1d(Et)
    taun = np.atleast_1d(taun)
    taup = np.atleast_1d(taup)

    bn = 1 / (taun * Nt)    # electron capture coefficients [cm^3/s]
    bp = 1 / (taup * Nt)    # hole capture coefficients [cm^3/s]
  
    
    # Constants
    kT = 8.617e-5 * T  # [eV]
    
    ni = np.sqrt(Nc * Nv * np.exp(-Eg/kT))
    M = len(Nt)  # number of traps
    
    # --- Step 1: equilibrium concentrations via generalized routine
    n0, p0, nt0 = calcequil_general(Eg, Nc, Nv, Nt, Et, T)
    
    # --- Step 2: define sweep grid
    narr = np.logspace(np.log10(n0), np.log10(Nc), Npoints)
    parr = np.zeros(Npoints)
    ntmat = np.zeros((M, Npoints))
    Rtotarr = np.zeros(Npoints)
    Rradarr = np.zeros(Npoints)
    Rsrhmat = np.zeros((M, Npoints))
    QFLS = np.zeros(Npoints)
    
    # Precompute equilibrium constants for traps
    n1 = Nc * np.exp((Et - Eg)/kT)
    p1 = Nv * np.exp(-Et/kT)
    en = bn* n1
    ep = bp * p1
    
    # --- Step 3: loop over n grid ---
    for ii in range(Npoints):
        n = narr[ii]
        # solve neutrality for holes
        parr[ii] = solve_p_newton_general(n, Nt, bn, bp, en, ep)
        
        p = parr[ii]
        
        # trap occupancies
        for j in range(M):
            f = trap_occ(n, p, bn[j], bp[j], en[j], ep[j])
            ntmat[j, ii] = Nt[j] * f
        
        # recombination channels
        Rrad = krad * n * p
        Rradarr[ii] = Rrad
        
        Rsrh_j = np.zeros(M)
        for j in range(M):
            Rsrh_j[j] = (Nt[j] * bn[j] * bp[j] * (n * p) / 
                        (n * bn[j] + p * bp[j] + en[j] + ep[j]))
        Rsrhmat[:, ii] = Rsrh_j
        Rtotarr[ii] = Rrad + np.sum(Rsrh_j)
        
        # QFLS
        QFLS[ii] = kT * np.log(n * p / ni**2)
    
    # --- Step 4: derived quantities ---
    PLQY = Rradarr / Rtotarr
    tau_ss_n = narr / Rtotarr
    tau_ss_p = parr / Rtotarr
    tau_ss = np.sqrt(narr * parr) / Rtotarr
    nid = (1/kT) * np.gradient(QFLS, np.log(Rtotarr))
    
    # --- Step 5: carrier capture/emission rates per trap ---
    e_trapping = np.zeros((M, Npoints))
    e_detrapping = np.zeros((M, Npoints))
    h_trapping = np.zeros((M, Npoints))
    h_detrapping = np.zeros((M, Npoints))
    
    for ii in range(Npoints):
        n = narr[ii]
        p = parr[ii]
        nt = ntmat[:, ii]
        for j in range(M):
            e_trapping[j, ii] = bn[j] * n * (Nt[j] - nt[j])
            e_detrapping[j, ii] = bn[j] * n1[j] * nt[j]
            h_trapping[j, ii] = bp[j] * p * nt[j]
            h_detrapping[j, ii] = bp[j] * p1[j] * (Nt[j] - nt[j])
    
    return (narr, parr, ntmat, Rtotarr, Rradarr, Rsrhmat,
            QFLS, PLQY, tau_ss, tau_ss_n, tau_ss_p, nid,
            e_trapping, e_detrapping, h_trapping, h_detrapping)





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


def solve_p_newton_general(n, Nt, bn, bp, en, ep):
    """
    Newton iteration for neutrality condition
    
    INPUTS:
        n   - electron density [cm^-3]
        Nt  - array of trap densities [cm^-3]
        bn  - array of electron capture coefficients [cm^3/s]
        bp  - array of hole capture coefficients [cm^3/s]
        en  - array of electron emission rates [s^-1]
        ep  - array of hole emission rates [s^-1]
    
    OUTPUTS:
        p   - hole density from charge neutrality [cm^-3]
    """
    M = len(Nt)
    
    # Initial guess
    p = n + np.sum(Nt / (1 + bp/bn + en/(bn*n)))
    
    for it in range(30):
        f = np.zeros(M)
        df = np.zeros(M)
        
        for j in range(M):
            f[j] = trap_occ(n, p, bn[j], bp[j], en[j], ep[j])  # f is trap occupancy
            df[j] = -(n*bn[j] + ep[j])*bp[j] / (n*bn[j] + p*bp[j] + en[j] + ep[j])**2
            # change in trap occupancy when changing p
        
        F = p - n - np.sum(f * Nt)
        dFdp = 1 - np.sum(Nt * df)
        # change of charge neutrality when changing p
        
        dp = -F / dFdp  # Newton-Raphson method
        p = p + dp
        if abs(dp/p) < 1e-8:
            break
    
    if p < 0:
        p = n
    
    return p


def trap_occ(n, p, bn, bp, en, ep):
    """
    Calculate trap occupancy
    
    INPUTS:
        n   - electron density [cm^-3]
        p   - hole density [cm^-3]
        bn  - electron capture coefficient [cm^3/s]
        bp  - hole capture coefficient [cm^3/s]
        en  - electron emission rate [s^-1]
        ep  - hole emission rate [s^-1]
    
    OUTPUTS:
        fs  - trap occupancy (fraction)
    """
    fs = (n*bn + ep) / (n*bn + p*bp + en + ep)
    return fs


