"""
dd_model.py — drift-diffusion ("mobility") tr-PL model, Python port of
``pdex1_sproul_Chris_frequency.m`` (MATLAB pdepe, slab, no Poisson equation)
=========================================================================

States per mesh node: u = [n, p, nt_1, ..., nt_K] (any number K of traps).

    dn/dt   = d/dx (Dn dn/dx) - (n - ni)/taub - krad (n p - ni^2) - sum_k [bn_k n (N_k - nt_k) - en_k nt_k]
    dp/dt   = d/dx (Dp dp/dx) - (p - ni)/taub - krad (n p - ni^2) - sum_k [bp_k p nt_k - ep_k (N_k - nt_k)]
    dnt_k/dt =  bn_k n (N_k - nt_k) - bp_k p nt_k - en_k nt_k + ep_k (N_k - nt_k)

    bn = 1/(taun Nt), bp = 1/(taup Nt), en = bn n1, ep = bp p1,
    n1 = Nc exp((Et - Eg)/kT), p1 = Nv exp(-Et/kT), D = kT mu.

Boundary conditions (pdex1bc): D du/dx = S_l (u - ni)/2 at x = 0 and D du/dx = -S_r (u - ni)/2 at x = d
for n and p (Robin, "note factor 2"); traps carry no flux.  S_l = S_r = 0 is zero flux.

Initial condition (pdex1ic): n = u00 exp(-alpha x) + n0, p = u00 exp(-alpha x) + p0,
u00 = alpha d n_pulse / (1 - exp(-alpha d)) (volume average n_pulse), traps at equilibrium.
n0, p0, nt0 come from the legacy ``trpl_module.calcequil_general`` (imported, never edited).

Discretisation: method of lines on the pdepe mesh (x = linspace(0, d, nx)) with the
Skeel-Berzins finite-volume scheme pdepe uses for m = 0: for node i with control volume
[x_{i-1/2}, x_{i+1/2}],

    vol_i du_i/dt = f_{i+1/2} - f_{i-1/2} + (h_{i-1}/2) s(U_{i-1/2}) + (h_i/2) s(U_{i+1/2}),

f_{i+1/2} = D (u_{i+1} - u_i)/h_i, U_{i+1/2} = (u_i + u_{i+1})/2 ("midpoint" source quadrature,
as pdepe; ``quadrature="nodal"`` evaluates s(u_i) at the nodes instead).  Time integration:
scipy ``solve_ivp`` BDF with a block-tridiagonal ``jac_sparsity``, states scaled to O(1)
(n, p by n_pulse; nt_k by N_k), tight rtol/atol.  The solution starts at t[0] (as pdepe).

Post-processing (main loop of the .m file):

    PL(t)  = int_E int_x alphaE(E) Phibb(E) exp(-alphaE(E) x) n p dx/d dE   (trapz in x and E)
             Earr = linspace(1.51, 1.79, 400), Phibb = E^2 exp(-E/kT), alphaE from alpha_vs_E_SSPL.dat
    PL     = PL/max(PL) n_pulse^2 - ni^2
    QFLS   = kT ln(PL/ni^2 + 1)          (= 2 kT ln(n_pulse/ni) + kT ln(PL/PL_max): the "nominal" axis
                                          of inference/ode.py)
    tau    = -2 gradient(t, log(PL))     (MATLAB gradient with coordinates log(PL): ``matlab_gradient``)

Constants: the MATLAB set (``MATLAB_CONSTANTS``: kT = 0.0258, Nc = Nv = 2.2e18, Eg = 1.625) is only
for comparisons with the MATLAB CSVs; every DD-vs-0D comparison uses the inference / legacy-ODE set
(``INFERENCE_CONSTANTS``: T = 300 K, Nc = Nv = 2.21359e18, kT = 8.617e-5 T as in trpl_module).

Public functions
----------------
simulate_dd(...)         -> dict with t, PL, PL_norm, QFLS, tau_diff, n_avg, p_avg, nt_avg (+ profiles)
log10_tau_on_qfls(...)   log10 tau_diff of a simulate_dd result at given QFLS points
nominal_qfls(res, vt)    QFLS axis (MATLAB Vint, or the inference nominal axis for a given Vt)
initial_pl_enhancement() PL(t0) of the exp(-alpha x) profile / PL of a flat one (QFLS-axis offset kT ln(.))
curve_log10_tau(...)     the same for raw (QFLS, tau, PL) arrays (e.g. a MATLAB CSV)
dd_log10_tau(...)        simulate_dd + log10_tau_on_qfls in one call
matlab_gradient(F, h)    MATLAB gradient(F, h) for a coordinate vector h
"""
from __future__ import annotations

import importlib.util
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.integrate import solve_ivp
from scipy.interpolate import PchipInterpolator

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
#: legacy rate-equation module (byte-identical copy of nn-training/trpl_module.py); imported only
LEGACY_MODULE_PATH = REPO / "network-training" / "trpl_module.py"
#: absorption coefficient vs photon energy used for the PL emission weighting (MATLAB alpha_exp)
ALPHA_FILE = HERE / "alpha_vs_E_SSPL.dat"

#: Boltzmann constant of the legacy solver (trpl_module: kT = 8.617e-5 * T)
KB_LEGACY = 8.617e-5
#: constants of pdex1_sproul_Chris_frequency.m (only for comparisons with the MATLAB CSVs)
MATLAB_CONSTANTS = dict(kT=0.0258, Nc=2.2e18, Nv=2.2e18, Eg=1.625)
#: constants of the inference / legacy ODE (inference/ode.py -> trpl_module.solve_transient)
INFERENCE_CONSTANTS = dict(T=300.0, Nc=2.21359e18, Nv=2.21359e18)
#: MATLAB time grid t = logspace(-11, -2, 512)
T_MATLAB = np.logspace(-11, -2, 512)
#: MATLAB energy grid of the PL integral
E_MATLAB = np.linspace(1.51, 1.79, 400)

_LEGACY = None


def legacy_module():
    """``network-training/trpl_module.py`` loaded once under a private name (as inference/ode.py does)."""
    global _LEGACY
    if _LEGACY is None:
        spec = importlib.util.spec_from_file_location("dd_legacy_trpl_module", LEGACY_MODULE_PATH)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _LEGACY = mod
    return _LEGACY


# ----------------------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------------------
def matlab_gradient(F, h) -> np.ndarray:
    """
    MATLAB ``gradient(F, h)`` with a coordinate vector h: (F[i+1] - F[i-1]) / (h[i+1] - h[i-1]) inside,
    one-sided first differences at both ends (unlike numpy's second-order non-uniform formula).
    """
    F = np.asarray(F, np.float64)
    h = np.asarray(h, np.float64)
    g = np.empty_like(F)
    with np.errstate(divide="ignore", invalid="ignore"):
        g[0] = (F[1] - F[0]) / (h[1] - h[0])
        g[-1] = (F[-1] - F[-2]) / (h[-1] - h[-2])
        g[1:-1] = (F[2:] - F[:-2]) / (h[2:] - h[:-2])
    return g


def trapz_weights(x) -> np.ndarray:
    """Weights w with sum(w * f) == trapz(x, f)."""
    x = np.asarray(x, np.float64)
    dx = np.diff(x)
    w = np.zeros_like(x)
    w[:-1] += dx / 2
    w[1:] += dx / 2
    return w


def emission_weights(x, d, kT, E=None, alpha_file=ALPHA_FILE) -> np.ndarray:
    """
    Weights W(x) with PL = sum_x W(x) n(x) p(x) for the MATLAB double integral
    trapz_E( trapz_x(alphaE Phibb exp(-alphaE x) n p) / d ).  Both integrals are linear in n p,
    so the order of summation is changed only (identical up to rounding).
    """
    E = E_MATLAB if E is None else np.asarray(E, np.float64)
    a = np.loadtxt(alpha_file)
    if E.min() < a[:, 0].min() or E.max() > a[:, 0].max():
        raise ValueError("energy grid outside the range of the absorption data (MATLAB interp1 would give NaN)")
    alphaE = np.interp(E, a[:, 0], a[:, 1])                       # interp1(..., 'linear')
    phibb = E ** 2 * np.exp(-E / kT)
    wE = trapz_weights(E) * alphaE * phibb                        # (nE,)
    kernel = wE @ np.exp(-np.outer(alphaE, np.asarray(x, np.float64)))   # (nx,)
    return kernel * trapz_weights(x) / d


def initial_pl_enhancement(alpha: float, d: float = 450e-7, nx: int = 150, kT: float = KB_LEGACY * 300.0,
                           pl_mode: str = "spectral", E=None, alpha_file=ALPHA_FILE) -> float:
    """
    PL(t[0]) of the MATLAB initial profile u00 exp(-alpha x) divided by the PL of a flat profile with
    density n_pulse (high injection, same emission weighting).  Because the DD PL is normalised to its
    maximum (PL/max(PL) n_pulse^2), the DD QFLS axis lies kT ln(ratio) below the physical QFLS once the
    carrier profile has flattened, i.e. log10 tau(QFLS) is shifted along the QFLS axis by that amount.
    """
    x = np.linspace(0.0, d, nx)
    ad = alpha * d
    g = np.ones(nx) if ad == 0 else ad / (-np.expm1(-ad)) * np.exp(-alpha * x)
    w = emission_weights(x, d, kT, E, alpha_file) if pl_mode == "spectral" else trapz_weights(x) / d
    return float(w @ g ** 2 / w.sum())


def _block_tridiag_sparsity(nx: int, nc: int):
    """Jacobian sparsity for node-interleaved states: full nc x nc blocks on the block tridiagonal."""
    band = sparse.diags([np.ones(nx - 1), np.ones(nx), np.ones(nx - 1)], [-1, 0, 1], format="csr")
    return sparse.kron(band, np.ones((nc, nc)), format="csc")


# ----------------------------------------------------------------------------------------------------------
# model
# ----------------------------------------------------------------------------------------------------------
def simulate_dd(Et, Nt, taun, taup, krad, n_pulse, alpha, *,
                d: float = 450e-7, mu: float = 1.0, mu_p: float | None = None,
                Eg: float = 1.625, Nc: float = INFERENCE_CONSTANTS["Nc"], Nv: float = INFERENCE_CONSTANTS["Nv"],
                T: float = INFERENCE_CONSTANTS["T"], kT: float | None = None,
                Sl: float = 0.0, Sr: float = 0.0, taub: float = np.inf,
                nx: int = 150, t=None, profile: str = "exp", pl_mode: str = "spectral",
                E_grid=None, alpha_file=ALPHA_FILE, quadrature: str = "midpoint",
                rtol: float = 1e-8, atol_np: float | None = None, atol_nt: float | None = None,
                return_profiles: bool = False) -> dict:
    """
    Drift-diffusion tr-PL transient (port of pdex1_sproul_Chris_frequency.m).

    Parameters
    ----------
    Et, Nt, taun, taup : sequences (K,) — trap energy above the VB [eV], density [cm^-3], electron and hole
                         capture lifetimes [s] (K = 0 allowed; traps with Nt = 0 are inactive, as in MATLAB)
    krad      : radiative coefficient [cm^3/s]
    n_pulse   : volume-averaged excess carrier density at t[0] [cm^-3] (MATLAB nint0)
    alpha     : absorption coefficient at the excitation wavelength [1/cm] (IC profile exp(-alpha x))
    d, mu     : thickness [cm], mobility [cm^2/Vs] (D = kT mu; ``mu_p`` for holes, default = mu)
    Eg, Nc, Nv: band gap [eV], effective densities of states [cm^-3]
    T, kT     : temperature [K] or thermal energy [eV]; kT = 8.617e-5 T (legacy) unless kT is given,
                then T = kT / 8.617e-5 for calcequil_general
    Sl, Sr    : surface recombination velocities [cm/s] (Robin BC S (u - ni)/2), taub bulk lifetime [s]
    nx        : number of mesh nodes (x = linspace(0, d, nx))
    t         : output times [s], the integration starts at t[0] (default logspace(-11, -2, 512))
    profile   : "exp" (MATLAB IC) or "flat" (n = n_pulse + n0 everywhere: 0D limit)
    pl_mode   : "spectral" (MATLAB PL integral with alphaE escape weighting) or "volume" (int n p dx / d)
    quadrature: "midpoint" (pdepe) or "nodal" source evaluation
    rtol, atol_np, atol_nt : solver tolerances; atol_np [cm^-3] for n, p (default 1e-13 n_pulse),
                atol_nt as a fraction of N_k (default 1e-13)
    return_profiles : also return n, p (nt_out, nx) and nt (K, nt_out, nx)

    Returns
    -------
    dict with t, PL (= PL/max(PL) n_pulse^2 - ni^2), PL_norm (= PL/max(PL), the MATLAB CSV column),
    QFLS [eV], tau_diff [s] (MATLAB gradient), n_avg, p_avg (trapz(x, .)/d), nt_avg (K, nt_out), x,
    info (solver status, nfev/njev/nlu, runtime_s, n0, p0, nt0, ni, kT, ...) and the profiles if requested.
    """
    t0_wall = time.perf_counter()
    Et = np.atleast_1d(np.asarray(Et, np.float64)).ravel()
    Nt = np.atleast_1d(np.asarray(Nt, np.float64)).ravel()
    taun = np.atleast_1d(np.asarray(taun, np.float64)).ravel()
    taup = np.atleast_1d(np.asarray(taup, np.float64)).ravel()
    K = len(Nt)
    if not (len(Et) == len(taun) == len(taup) == K):
        raise ValueError("Et, Nt, taun, taup must have the same length")
    if profile not in ("exp", "flat"):
        raise ValueError("profile must be 'exp' or 'flat'")
    if pl_mode not in ("spectral", "volume"):
        raise ValueError("pl_mode must be 'spectral' or 'volume'")
    if quadrature not in ("midpoint", "nodal"):
        raise ValueError("quadrature must be 'midpoint' or 'nodal'")

    if kT is None:
        kT = KB_LEGACY * T
    else:
        T = kT / KB_LEGACY
    ni = np.sqrt(Nc * Nv * np.exp(-Eg / kT))
    Dn = kT * mu
    Dp = kT * (mu if mu_p is None else mu_p)
    t_out = T_MATLAB if t is None else np.asarray(t, np.float64)

    # --- trap coefficients (inactive traps: Nt = 0 -> all rates 0, as MATLAB's beta = 0) ---
    active = Nt > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        bn = np.where(active, 1.0 / (taun * Nt), 0.0)
        bp = np.where(active, 1.0 / (taup * Nt), 0.0)
    n1 = Nc * np.exp((Et - Eg) / kT)
    p1 = Nv * np.exp(-Et / kT)
    en, ep = bn * n1, bp * p1

    # --- equilibrium from the legacy solver (never edited) ---
    leg = legacy_module()
    if np.any(active):
        n0, p0, nt0_act = leg.calcequil_general(Eg, Nc, Nv, Nt[active], Et[active], T)
        nt0 = np.zeros(K)
        nt0[active] = nt0_act
    else:
        n0, p0, _ = leg.calcequil_general(Eg, Nc, Nv, None, None, T)
        nt0 = np.zeros(K)

    # --- mesh, control volumes, initial condition ---
    x = np.linspace(0.0, d, nx)
    h = np.diff(x)
    half = h / 2                                   # half-interval widths (left/right parts of control volumes)
    vol = trapz_weights(x)                         # control volume widths (== trapz weights)
    nc = 2 + K
    if profile == "exp":
        ad = alpha * d
        u00 = n_pulse if ad == 0 else ad * n_pulse / (-np.expm1(-ad))
        g = u00 * np.exp(-alpha * x)
    else:
        g = np.full(nx, float(n_pulse))
    U0 = np.empty((nx, nc))
    U0[:, 0] = g + n0
    U0[:, 1] = g + p0
    U0[:, 2:] = nt0[None, :]

    scale = np.concatenate([[n_pulse, n_pulse], np.where(active, Nt, 1.0)])
    if atol_np is None:
        atol_np = 1e-13 * n_pulse
    if atol_nt is None:
        atol_nt = 1e-13
    atol = np.tile(np.concatenate([[atol_np / n_pulse] * 2, [atol_nt] * K]), nx)
    Nt_row = Nt[None, :]
    ni2 = ni * ni
    bulk = np.isfinite(taub)

    def source(u):
        n = u[:, 0]
        p = u[:, 1]
        nt = u[:, 2:]
        rrad = krad * (n * p - ni2)
        empty = Nt_row - nt
        capn = bn * n[:, None] * empty - en * nt          # net electron capture per trap
        capp = bp * p[:, None] * nt - ep * empty          # net hole capture per trap
        s = np.empty_like(u)
        s[:, 0] = -rrad - capn.sum(axis=1)
        s[:, 1] = -rrad - capp.sum(axis=1)
        if bulk:
            s[:, 0] -= (n - ni) / taub
            s[:, 1] -= (p - ni) / taub
        s[:, 2:] = capn - capp
        return s

    def rhs(_t, y):
        u = y.reshape(nx, nc) * scale
        acc = np.zeros((nx, nc))
        fn = Dn * (u[1:, 0] - u[:-1, 0]) / h
        fp = Dp * (u[1:, 1] - u[:-1, 1]) / h
        acc[:-1, 0] += fn
        acc[1:, 0] -= fn
        acc[:-1, 1] += fp
        acc[1:, 1] -= fp
        if Sl:
            acc[0, :2] -= Sl * (u[0, :2] - ni) / 2       # f(x_l) = -p_l/q_l = Sl (u - ni)/2
        if Sr:
            acc[-1, :2] -= Sr * (u[-1, :2] - ni) / 2     # f(x_r) = -p_r/q_r = -Sr (u - ni)/2
        if quadrature == "midpoint":
            sm = source(0.5 * (u[1:] + u[:-1])) * half[:, None]
            acc[:-1] += sm
            acc[1:] += sm
        else:
            acc += source(u) * vol[:, None]
        return (acc / vol[:, None] / scale).ravel()

    sol = solve_ivp(rhs, (t_out[0], t_out[-1]), (U0 / scale).ravel(), method="BDF", t_eval=t_out,
                    rtol=rtol, atol=atol, jac_sparsity=_block_tridiag_sparsity(nx, nc))
    t_solve = time.perf_counter() - t0_wall
    if sol.status != 0 or sol.y.shape[1] != len(t_out):
        raise RuntimeError(f"DD solve failed: {sol.message}")

    U = sol.y.T.reshape(len(t_out), nx, nc) * scale        # (nt, nx, nc)
    n, p = U[:, :, 0], U[:, :, 1]
    npx = n * p
    wx = vol / d
    if pl_mode == "spectral":
        pl_raw = npx @ emission_weights(x, d, kT, E_grid, alpha_file)
    else:
        pl_raw = npx @ wx
    PL = pl_raw / np.max(pl_raw) * n_pulse ** 2 - ni2
    with np.errstate(divide="ignore", invalid="ignore"):
        qfls = kT * np.log(PL / ni2 + 1.0)
        tau = -2.0 * matlab_gradient(t_out, np.log(PL))
    out = dict(t=t_out, PL=PL, PL_norm=PL / np.max(PL), QFLS=qfls, tau_diff=tau,
               n_avg=n @ wx, p_avg=p @ wx, nt_avg=np.einsum("tik,i->kt", U[:, :, 2:], wx), x=x,
               info=dict(status=int(sol.status), message=sol.message, nfev=int(sol.nfev), njev=int(sol.njev),
                         nlu=int(sol.nlu), runtime_s=time.perf_counter() - t0_wall, solve_s=t_solve,
                         n0=float(n0), p0=float(p0), nt0=nt0.tolist(), ni=float(ni), kT=float(kT), T=float(T),
                         n_pulse=float(n_pulse), Eg=float(Eg), Nc=float(Nc), Nv=float(Nv), alpha=float(alpha),
                         Dn=Dn, Dp=Dp, nx=nx, K=K, rtol=rtol, atol_np=atol_np, atol_nt=atol_nt,
                         profile=profile, pl_mode=pl_mode, quadrature=quadrature))
    if return_profiles:
        out.update(n=n, p=p, nt=np.moveaxis(U[:, :, 2:], 2, 0))
    return out


# ----------------------------------------------------------------------------------------------------------
# tau(QFLS) on given points
# ----------------------------------------------------------------------------------------------------------
def curve_log10_tau(qfls, tau, pl, qfls_points, pl_decay_magnitude: float | None = 12.0,
                    kind: str = "pchip") -> np.ndarray:
    """
    log10 tau_diff at ``qfls_points`` from a decay curve (arrays in time order), post-processed as
    inference/ode.py does for the ODE: drop non-finite tau / PL, cap the PL ``pl_decay_magnitude`` decades
    below its maximum (None: no cap), reverse to ascending QFLS, de-duplicate, interpolate ("pchip" as
    ode.py, or "linear" as MATLAB interp1).  NaN outside the simulated range — never extrapolated.
    """
    qfls = np.asarray(qfls, np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        lt = np.log10(np.asarray(tau, np.float64))
        lp = np.log10(np.asarray(pl, np.float64))
    ok = np.isfinite(lt) & np.isfinite(lp) & np.isfinite(qfls)
    q, lt, lp = qfls[ok], lt[ok], lp[ok]
    pts = np.asarray(qfls_points, np.float64)
    if len(q) < 2:
        return np.full(pts.shape, np.nan)
    if pl_decay_magnitude is not None:
        lp = lp - np.max(lp)
        cap = np.nonzero(lp < -float(pl_decay_magnitude))[0]
        if len(cap):
            q, lt = q[:cap[0]], lt[:cap[0]]
    qa, idx = np.unique(q[::-1], return_index=True)
    la = lt[::-1][idx]
    if len(qa) < 2:
        return np.full(pts.shape, np.nan)
    if kind == "pchip":
        return PchipInterpolator(qa, la, extrapolate=False)(pts)
    if kind == "linear":
        return np.interp(pts, qa, la, left=np.nan, right=np.nan)
    raise ValueError("kind must be 'pchip' or 'linear'")


def nominal_qfls(res: dict, vt: float | None = None) -> np.ndarray:
    """
    QFLS axis of a ``simulate_dd`` result.  vt = None: the MATLAB Vint (model kT).  vt given: the nominal
    axis of inference/ode.py, 2 vt ln(n_pulse/ni) + vt ln(PL/PL_max) with ni = sqrt(Nc Nv) exp(-Eg/(2 vt))
    (inference passes constants["Vt"] = k_B T/q = 0.025852 V, the legacy kT is 8.617e-5 T = 0.025851 eV).
    """
    if vt is None:
        return res["QFLS"]
    i = res["info"]
    ni = np.sqrt(i["Nc"] * i["Nv"]) * np.exp(-i["Eg"] / (2.0 * vt))
    with np.errstate(divide="ignore", invalid="ignore"):
        return 2.0 * vt * np.log(i["n_pulse"] / ni) + vt * np.log(res["PL"] / np.max(res["PL"]))


def log10_tau_on_qfls(res: dict, qfls_points, pl_decay_magnitude: float | None = 12.0,
                      kind: str = "pchip", vt: float | None = None) -> np.ndarray:
    """
    log10 tau_diff of a ``simulate_dd`` result at ``qfls_points`` (NaN outside the simulated range).
    For comparisons with inference/ode.py (``ode_log10_tau``, axis "nominal") pass vt=constants["Vt"].
    """
    return curve_log10_tau(nominal_qfls(res, vt), res["tau_diff"], res["PL"], qfls_points,
                           pl_decay_magnitude, kind)


def dd_log10_tau(qfls_points, *args, pl_decay_magnitude: float | None = 12.0, kind: str = "pchip",
                 vt: float | None = None, **kwargs) -> np.ndarray:
    """``simulate_dd(*args, **kwargs)`` followed by ``log10_tau_on_qfls``."""
    return log10_tau_on_qfls(simulate_dd(*args, **kwargs), qfls_points, pl_decay_magnitude, kind, vt)


def to_matlab_table(res: dict):
    """DataFrame with the MATLAB CSV columns time_s, PL_norm, QFLS_eV, tau_diff_s."""
    import pandas as pd
    return pd.DataFrame(dict(time_s=res["t"], PL_norm=res["PL_norm"], QFLS_eV=res["QFLS"],
                             tau_diff_s=res["tau_diff"]))
