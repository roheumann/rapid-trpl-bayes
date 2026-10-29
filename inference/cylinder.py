"""
cylinder.py — eigenvector-cylinder slice inference around the ODE best fit
=========================================================================
14/10/2026  (revision; decided 2026-10-14, plan ~/.claude/plans/floofy-purring-fiddle.md, Part 2)

Idea
----
The Laplace covariance of the ODE best fit θ* (``cov_theta_curve_fit``, scipy curve_fit with
absolute_sigma, σ·√N per point = the per-point equivalent of the curve-level σ) is diagonalised in *decade* coordinates

    x = (log10 n1_k, log10 N_t,k, log10 τ_n,k, log10 τ_p,k, [log10 k_rad])  of the FREE parameters,
    log10 n1 = log10 N_c + (E_t − E_g)/(kT ln10)  (linear in E_t → the uniform prior stays uniform),

so that every axis is measured in decades.  For every eigenvector v_k (σ_k = √λ_k) a cylinder

    C_k = { θ* + Σ_j z_j σ_j v_j :  z_k on the chord of the box,  Σ_{j≠k} z_j² ≤ R² }   (R = 2)

is filled with scrambled-Sobol points (uniform inside C_k ∩ box ∩ prior support).  The network log-likelihood
is evaluated at every point and the points are importance-weighted with the mixture density of all cylinders
(balance heuristic):

    q(z) = Σ_k (N_k/N) · 1[z ∈ C_k ∩ B] / vol(C_k ∩ B),     w = L(θ) π(θ) / q(z),  π = const on the support.

Everything is computed in whitened coordinates z; the Jacobian to decades is the constant Π σ_j and
cancels in the normalised weights.  The same weighted samples give both corner plots (eigen coordinates
z and physical decades, see corner_export.py).

Coordinates used here
---------------------
theta   : prior coordinates (log10 k_rad, log10 N_t, E_t/E_g, log10 τ_n, log10 τ_p per trap), all D
x       : decades of the free parameters (E_t/E_g -> log10 n1), D_free
z       : whitened coordinates, x = x* + V diag(σ) z
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import gammaln, ndtri
from scipy.stats import qmc

SIGMA_1 = 0.6827
SIGMA_2 = 0.9545
LN10 = math.log(10.0)

# physical box of the decision (2026-10-14); intersected with the prior (= training box ∩ overrides)
PHYS_BOX = {"Et_rel": (0.5, 1.0), "log10_Nt": (12.0, None), "log10_taun": (-12.0, -4.0),
            "log10_taup": (-12.0, -4.0)}


# ─────────────────────────────────────────────────────────────────────────────
# parameter bookkeeping
# ─────────────────────────────────────────────────────────────────────────────
def _kind(name: str) -> str:
    """'log10_krad' -> 'log10_krad', 'log10_Nt_1' -> 'log10_Nt', 'Et_rel_2' -> 'Et_rel', ..."""
    return name if name == "log10_krad" else name.rsplit("_", 1)[0]


def _trap(name: str) -> int:
    return 0 if name == "log10_krad" else int(name.rsplit("_", 1)[1])


@dataclass
class DecadeMap:
    """Linear map theta (all D) <-> x (free parameters in decades)."""
    prior: object

    def __post_init__(self):
        p = self.prior
        self.free = list(p.free_idx)
        self.names = [p.names[i] for i in self.free]
        self.is_et = np.array([_kind(n) == "Et_rel" for n in self.names])
        self.slope = p.Eg / (p.kT * LN10)                     # d log10 n1 / d Et_rel
        self.offset = math.log10(p.Nc) - self.slope           # log10 n1 = offset + slope * Et_rel
        self.A = np.where(self.is_et, self.slope, 1.0)        # diagonal Jacobian dx/dtheta_free
        self.keys = [n.replace("Et_rel", "log10_n1") for n in self.names]

    def to_x(self, theta):
        th = np.atleast_2d(theta)[:, self.free]
        return np.where(self.is_et, self.offset + self.slope * th, th)

    def to_theta(self, x, theta_ref):
        """x (B, D_free) -> theta (B, D); fixed parameters from theta_ref."""
        x = np.atleast_2d(x)
        th_free = np.where(self.is_et, (x - self.offset) / self.slope, x)
        out = np.repeat(np.asarray(theta_ref, float)[None], len(x), axis=0)
        out[:, self.free] = th_free
        return out

    def cov_to_x(self, cov_free):
        return self.A[:, None] * cov_free * self.A[None, :]

    def box(self):
        """(lo, hi) in x: prior box ∩ physical box of the decision."""
        p = self.prior
        lo, hi = p.lo[self.free].copy(), p.hi[self.free].copy()
        log_nc = math.log10(p.Nc)
        for i, n in enumerate(self.names):
            k = _kind(n)
            if k in PHYS_BOX:
                a, b = PHYS_BOX[k]
                b = log_nc if (k == "log10_Nt" and b is None) else b
                lo[i], hi[i] = max(lo[i], a), min(hi[i], b)
        xl = np.where(self.is_et, self.offset + self.slope * lo, lo)
        xh = np.where(self.is_et, self.offset + self.slope * hi, hi)
        return xl, xh


# ─────────────────────────────────────────────────────────────────────────────
# sigma and covariance
# ─────────────────────────────────────────────────────────────────────────────
def nn_test_mse_log10(nn_dir) -> float:
    """
    Whole-curve test MSE of a network in log10 tau_diff units (decision 2026-10-14):
    training_details.json results.test_mse is on the min-max scaled output -> divide by scale_^2.
    """
    import glob
    import joblib
    nn_dir = Path(nn_dir)
    det = json.loads((nn_dir / "training_details.json").read_text())["results"]
    if "test_mse_log10" in det:                         # train_conv_v2.py stores it directly
        return float(det["test_mse_log10"])
    sc = joblib.load(sorted(glob.glob(str(nn_dir / "*output_scaler.joblib")))[0])
    s = np.atleast_1d(sc.scale_).astype(float)
    if not np.allclose(s, s[0]):
        raise ValueError("output scaler is not global (per-point scales) — use test_mse_log10")
    return float(det["test_mse"]) / s[0] ** 2


def fit_mse_ode(run_dir) -> float:
    """MSE of log10 tau_diff of the ODE best fit vs the data (6_best_fit_at_data_points.csv)."""
    d = pd.read_csv(Path(run_dir) / "6_best_fit_at_data_points.csv")
    r = np.log10(d["tau_diff_ode_s"].values) - np.log10(d["tau_diff_data_s"].values)
    return float(np.mean(r[np.isfinite(r)] ** 2))


def sigma_new(spec: dict, run_dir, nn_dir) -> dict:
    """
    tr-PL sigma of the cylinder inference.
      spec = {"kind": "nn_test"}           sigma = sqrt(fit_MSE + NN_test_MSE)       (synthetic data)
      spec = {"kind": "base", "base": s}   sigma = sqrt(s^2 + fit_MSE)               (experimental data)
      spec = {"kind": "fixed", "value": s} sigma = s
      spec = {"kind": "run"}               sigma = sigma_trpl of the run itself (the sigma of its NUTS / emcee / SMC,
                                            e.g. sigma_rule 'model_plus_nn'; 20/10/2026)
    """
    fm = fit_mse_ode(run_dir)
    out = {"kind": spec["kind"], "fit_mse_ode": fm}
    if spec["kind"] == "nn_test":
        out["nn_test_mse_log10"] = nn_test_mse_log10(nn_dir)
        out["sigma_trpl"] = math.sqrt(fm + out["nn_test_mse_log10"])
    elif spec["kind"] == "base":
        out["base"] = float(spec["base"])
        out["sigma_trpl"] = math.sqrt(out["base"] ** 2 + fm)
    elif spec["kind"] == "fixed":
        out["sigma_trpl"] = float(spec["value"])
    elif spec["kind"] == "run":
        from .runs import run_sigmas
        s = run_sigmas(run_dir)["trpl"]
        out["sigma_trpl"] = float(s[0] if isinstance(s, (list, tuple)) else s)
    else:
        raise ValueError(spec)
    return out


def gauss_newton_cov(predict, theta_star, free_idx, sigma_points, h: float = 1e-3):
    """
    Laplace (Gauss-Newton) covariance of the free parameters (theta coordinates) for the MSE likelihood of the
    workflow, log L = -sum_d MSE_d / (2 sigma_d^2): F = J^T diag(1/s_i^2) J with s_i = sigma_d sqrt(N_d) — the
    weights and the covariance of curve_fit (absolute_sigma) in main.ipynb — with J from central differences.

    predict      : (B, D) -> (B, n) model values at the data points (NaN = not covered; such points are dropped)
    sigma_points : (n,) s_i
    Returns (cov (D_free, D_free), number of points used).
    """
    theta_star = np.asarray(theta_star, float)
    ths = [theta_star]
    for i in free_idx:
        for sgn in (+1, -1):
            t = theta_star.copy()
            t[i] += sgn * h
            ths.append(t)
    Y = np.asarray(predict(np.array(ths)), float)
    J = np.stack([(Y[1 + 2 * j] - Y[2 + 2 * j]) / (2 * h) for j in range(len(free_idx))], axis=1)
    ok = np.all(np.isfinite(J), axis=1) & np.isfinite(Y[0])
    Jw = J[ok] / np.asarray(sigma_points, float)[ok, None]
    return np.linalg.pinv(Jw.T @ Jw), int(ok.sum())


def laplace_cov(log_post, theta_star, art=None, npulse=None, T: float = 300.0, n_jobs: int = 6, h: float = 1e-3):
    """
    Gauss-Newton covariance (free parameters, theta coordinates) at theta_star with the CURRENT sigmas of
    log_post, for the modes without a usable curve_fit covariance:
      'joint' : tr-PL from the ODE (refine.OdeFitProblem, as curve_fit) + exact ss-PL model (numpy)
      'sspl'  : exact ss-PL model (TF)
      'trpl'  : tr-PL from the ODE (same as the cross-check of cylinder.ipynb)
    Returns (cov, info).
    """
    import math as _m
    free = list(log_post.prior.free_idx)
    if log_post.trpl_terms:
        from . import ode, refine
        prob = refine.make_problem(log_post, art, npulse, T)

        def predict(TH):
            y = ode.ode_predictions(prob, TH, n_jobs=n_jobs, verbose=False)["y"]
            if prob.sspl is not None:
                y = np.hstack([y, prob.sspl.predict(TH)])
            return y
        cov, n_ok = gauss_newton_cov(predict, theta_star, free, prob.sigma_points, h)
        return cov, {"method": "Gauss-Newton, tr-PL with the ODE" + (" + exact ss-PL model" if prob.sspl is not None else ""),
                     "n_points_used": n_ok, "n_points": len(prob.y_obs), "fd_step": h}
    import tensorflow as tf
    obs = log_post.sspl_term.observation
    s = log_post.sigmas()["sspl"] * _m.sqrt(len(obs.y_obs))
    cov, n_ok = gauss_newton_cov(lambda TH: obs.predict(tf.constant(TH, tf.float64)).numpy(), theta_star, free,
                                 np.full(len(obs.y_obs), s), h)
    return cov, {"method": "Gauss-Newton, exact ss-PL model", "n_points_used": n_ok, "n_points": len(obs.y_obs),
                 "fd_step": h}


def conditional_cov(cov, keep):
    """Covariance of the kept parameters CONDITIONAL on the others (Schur complement of the precision)."""
    P = np.linalg.pinv(cov)
    return np.linalg.pinv(P[np.ix_(keep, keep)])


@dataclass
class EigenSystem:
    x_star: np.ndarray          # (D,)
    cov_x: np.ndarray           # (D, D) rescaled covariance in decades
    V: np.ndarray               # (D, D) columns = eigenvectors, sorted by sigma descending
    sigma: np.ndarray           # (D,) sqrt(eigenvalues), floored
    sigma_raw: np.ndarray       # (D,) before the floor
    floored: np.ndarray         # (D,) bool
    keys: list

    @property
    def D(self):
        return len(self.sigma)

    def to_z(self, x):
        return ((np.atleast_2d(x) - self.x_star) @ self.V) / self.sigma

    def to_x(self, z):
        return self.x_star + (np.atleast_2d(z) * self.sigma) @ self.V.T


def eigen_system(x_star, cov_x, keys, sigma_min: float = 0.02) -> EigenSystem:
    cov_x = 0.5 * (cov_x + cov_x.T)
    lam, V = np.linalg.eigh(cov_x)
    order = np.argsort(lam)[::-1]
    lam, V = lam[order], V[:, order]
    V = V * np.sign(V[np.argmax(np.abs(V), axis=0), np.arange(V.shape[1])])   # largest component > 0
    s_raw = np.sqrt(np.clip(lam, 0.0, None))
    s = np.maximum(s_raw, sigma_min)
    return EigenSystem(np.asarray(x_star, float), cov_x, V, s, s_raw, s_raw < sigma_min, list(keys))


def chord(x_star, v, lo, hi):
    """t-range of x* + t v inside the box [lo, hi]."""
    with np.errstate(divide="ignore", invalid="ignore"):
        t1, t2 = (lo - x_star) / v, (hi - x_star) / v
    tl = np.where(np.abs(v) > 1e-14, np.minimum(t1, t2), -np.inf)
    th = np.where(np.abs(v) > 1e-14, np.maximum(t1, t2), np.inf)
    return float(np.max(tl)), float(np.min(th))


# ─────────────────────────────────────────────────────────────────────────────
# cylinders
# ─────────────────────────────────────────────────────────────────────────────
def log_ball_volume(n: int, r: float) -> float:
    return 0.5 * n * math.log(math.pi) + n * math.log(r) - gammaln(0.5 * n + 1.0)


@dataclass
class Cylinders:
    es: EigenSystem
    lo: np.ndarray
    hi: np.ndarray
    radius: float = 2.0
    chords_t: np.ndarray = field(default=None)      # (K, 2) in decades along v_k
    chords_z: np.ndarray = field(default=None)      # (K, 2) in z_k units

    def __post_init__(self):
        es = self.es
        self.chords_t = np.array([chord(es.x_star, es.V[:, k], self.lo, self.hi) for k in range(es.D)])
        self.chords_z = self.chords_t / es.sigma[:, None]
        # range of every whitened coordinate z_j over the box (extremes at the box corners)
        dlo, dhi = (self.lo - es.x_star)[:, None] * es.V, (self.hi - es.x_star)[:, None] * es.V    # (D_x, K)
        self.zbox_lo = np.minimum(dlo, dhi).sum(axis=0) / es.sigma
        self.zbox_hi = np.maximum(dlo, dhi).sum(axis=0) / es.sigma

    def member(self, z) -> np.ndarray:
        """(B, K) bool: z inside cylinder k (box not checked)."""
        z = np.atleast_2d(z)
        r2 = np.sum(z ** 2, axis=1, keepdims=True) - z ** 2            # Σ_{j≠k} z_j²
        on_axis = (z >= self.chords_z[:, 0]) & (z <= self.chords_z[:, 1])
        return (r2 <= self.radius ** 2) & on_axis

    def proposal(self, k: int):
        """
        Proposal region of cylinder k (contains C_k ∩ box): z_k on the chord; perpendicular directions whose
        box range covers [-R, R] form a ball of radius R ('stiff'), the others are drawn uniformly over
        [-R, R] ∩ (box range of z_j) ('sloppy': 2σ_j larger than the box, the ball would lie mostly outside).
        Returns (stiff dims, sloppy dims, sloppy ranges (n, 2), log volume of the proposal).
        """
        R = self.radius
        perp = [j for j in range(self.es.D) if j != k]
        stiff = [j for j in perp if self.zbox_lo[j] <= -R and self.zbox_hi[j] >= R]
        sloppy = [j for j in perp if j not in stiff]
        rng = np.array([[max(-R, self.zbox_lo[j]), min(R, self.zbox_hi[j])] for j in sloppy]).reshape(-1, 2)
        a, b = self.chords_z[k]
        lv = math.log(b - a) + (log_ball_volume(len(stiff), R) if stiff else 0.0) \
            + float(np.sum(np.log(np.maximum(rng[:, 1] - rng[:, 0], 1e-300))))
        return stiff, sloppy, rng, lv

    def sample(self, k: int, m: int, seed: int):
        """2^m scrambled-Sobol points uniform in the proposal of cylinder k (z coordinates) + its log volume.
        Points outside the cylinder / box / support are rejected by the caller (draw)."""
        D = self.es.D
        stiff, sloppy, rng, lv = self.proposal(k)
        ns = len(stiff)
        d = 1 + len(sloppy) + (ns + 1 if ns else 0)
        u = np.clip(qmc.Sobol(d=d, scramble=True, seed=seed).random_base2(m), 1e-12, 1 - 1e-12)
        a, b = self.chords_z[k]
        z = np.empty((len(u), D))
        z[:, k] = a + (b - a) * u[:, 0]
        if sloppy:
            z[:, sloppy] = rng[:, 0] + (rng[:, 1] - rng[:, 0]) * u[:, 1:1 + len(sloppy)]
        if ns:
            g = ndtri(u[:, 1 + len(sloppy):1 + len(sloppy) + ns])
            g /= np.linalg.norm(g, axis=1, keepdims=True)
            r = self.radius * u[:, -1] ** (1.0 / ns)
            z[:, stiff] = g * r[:, None]
        return z, lv


def in_box(x, lo, hi):
    return np.all((x >= lo) & (x <= hi), axis=1)


@dataclass
class CylinderResult:
    z: np.ndarray            # (N, D) accepted points
    x: np.ndarray            # (N, D) decades
    theta: np.ndarray        # (N, D_all)
    cyl: np.ndarray          # (N,) cylinder the point was drawn from
    loglik: np.ndarray       # (N,)
    log_q: np.ndarray        # (N,)
    log_w: np.ndarray        # (N,) unnormalised
    w: np.ndarray            # (N,) normalised
    n_drawn: np.ndarray      # (K,)
    n_accepted: np.ndarray   # (K,)
    log_vol: np.ndarray      # (K,) log vol(C_k ∩ box ∩ support), z units
    ess: float
    n_evals: int


def draw(cyl: Cylinders, dmap: DecadeMap, prior, theta_ref, m: int = 14, seed: int = 0):
    """Sobol points of all cylinders, restricted to box ∩ prior support; acceptance -> volumes."""
    Z, C, nd, na, lv = [], [], [], [], []
    for k in range(cyl.es.D):
        z, lv_prop = cyl.sample(k, m, seed + 1000 * k)
        z = z[cyl.member(z)[:, k]]                                   # inside cylinder k (ball in all ⊥ dims)
        n_prop = 2 ** m
        x = cyl.es.to_x(z)
        ok = in_box(x, cyl.lo, cyl.hi)
        th = dmap.to_theta(x[ok], theta_ref)
        sup = prior.in_support(th) if len(th) else np.zeros(0, bool)
        idx = np.flatnonzero(ok)[sup]
        nd.append(n_prop); na.append(len(idx))
        lv.append(lv_prop + (math.log(len(idx) / n_prop) if len(idx) else -np.inf))
        Z.append(z[idx]); C.append(np.full(len(idx), k))
    return np.vstack(Z), np.concatenate(C), np.array(nd), np.array(na), np.array(lv)


def mixture_log_q(cyl: Cylinders, z, n_accepted, log_vol):
    """log q(z) of the cylinder mixture (balance heuristic), z inside box ∩ support."""
    N = n_accepted.sum()
    M = cyl.member(z)
    use = (n_accepted > 0)[None, :]                                                  # empty cylinders: weight 0
    comp = np.where(M & use, np.log(np.maximum(n_accepted, 1) / N) - np.where(use[0], log_vol, 0.0), -np.inf)
    mx = np.max(comp, axis=1, keepdims=True)
    inside = np.isfinite(mx[:, 0])
    out = np.full(len(comp), -np.inf)                                                # outside every cylinder: q = 0
    out[inside] = (mx[inside] + np.log(np.sum(np.exp(comp[inside] - mx[inside]), axis=1, keepdims=True)))[:, 0]
    return out


def run(cyl: Cylinders, dmap: DecadeMap, prior, theta_ref, log_lik_fn, m: int = 14, seed: int = 0,
        chunk: int = 65536) -> CylinderResult:
    """Draw, evaluate log L (log_lik_fn: theta (B, D_all) -> (B,)), weight."""
    z, cid, nd, na, lv = draw(cyl, dmap, prior, theta_ref, m, seed)
    x = cyl.es.to_x(z)
    th = dmap.to_theta(x, theta_ref)
    ll = np.concatenate([np.asarray(log_lik_fn(th[i:i + chunk]), float) for i in range(0, len(th), chunk)])
    lq = mixture_log_q(cyl, z, na, lv)
    lw = np.where(np.isfinite(ll), ll - lq, -np.inf)
    w = np.exp(lw - np.max(lw))
    w /= w.sum()
    ess = float(1.0 / np.sum(w ** 2))
    return CylinderResult(z, x, th, cid, ll, lq, lw, w, nd, na, lv, ess, len(th))


def slices(cyl: Cylinders, dmap: DecadeMap, prior, theta_ref, log_lik_fn, n: int = 201):
    """log L along every eigen-axis (chord), NaN outside the prior support. Returns dict k -> DataFrame."""
    out = {}
    for k in range(cyl.es.D):
        t = np.linspace(*cyl.chords_t[k], n)
        x = cyl.es.x_star + t[:, None] * cyl.es.V[:, k][None]
        th = dmap.to_theta(x, theta_ref)
        ll = np.full(n, np.nan)
        sup = prior.in_support(th)
        if sup.any():
            ll[sup] = np.asarray(log_lik_fn(th[sup]), float)
        out[k] = pd.DataFrame({"t": t, "z": t / cyl.es.sigma[k], "loglik": ll})
    return out


def fraction_inside(cyl: Cylinders, x):
    """Fraction of points (decades) inside the union of the cylinders and the box."""
    z = cyl.es.to_z(x)
    return float(np.mean(cyl.member(z).any(axis=1) & in_box(x, cyl.lo, cyl.hi)))
