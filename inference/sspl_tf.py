"""
sspl_tf.py — steady-state PL forward model in TensorFlow (float64, batched)
==========================================================================

Physics (identical to the ss-PL grid code of the original submission and, up
to the negligible -n_i^2 terms, to its fitting module ``sspl_module.py``):

* Sweep the electron density n over a fixed logarithmic grid.
* Charge neutrality with acceptor-like traps:  p = n + sum_k n_t,k(n, p), with
      n_t,k = N_t,k (tau_p,k n + tau_n,k p_1,k) /
              (tau_p,k (n + n_1,k) + tau_n,k (p + p_1,k))
* R_rad = k_rad (n p - n_i^2),
  R_SRH,k = (n p - n_i^2) / ((n + n_1,k) tau_p,k + (p + p_1,k) tau_n,k),
  G = R_tot = R_rad + sum_k R_SRH,k  (steady state),
  PLQY = R_rad / R_tot,  QFLS = kT ln(n p / n_i^2).

Solver
------
* 1 trap: closed-form root of the quadratic for n_t, written in the
  cancellation-free form  n_t = 2 |c| / (b + sqrt(b^2 + 4 a |c|)).
* N traps: g(p) = p - n - sum_k n_t,k(p) is strictly increasing in p and has
  its root in [n, n + sum N_t].  The root is found by bisection in log p
  (robust, fixed number of steps), polished with Newton steps, and the
  gradient is obtained from one final differentiable Newton step around the
  stop-gradient root:  p = p* - g(p*, theta) / g'(p*).  This is exact
  implicit differentiation (dp/dtheta = -dg/dtheta / g'), so autodiff never
  has to differentiate through the iterations.

The model is run in float64: n p reaches ~5e36, close to the float32 limit.

Interpolation to the measured points is done PER ROW (one interpolation
bracket per parameter set) — the batched equivalent of the bug fix 1.1.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def _pow10(x):
    """10**x for float64 tensors (tf.pow(10.0, x) would promote the base to float32)."""
    import tensorflow as tf
    return tf.exp(tf.convert_to_tensor(x, tf.float64) * math.log(10.0))

KB_EV = 8.617333262e-5

# Accepted comparison modes (y vs x) and legacy aliases from 03_sspl_errors.py
SSPL_MODES = ("qfls_vs_suns", "plqy_vs_suns", "plqy_vs_qfls", "qfls_vs_G", "plqy_vs_G")
_MODE_ALIASES = {"suns_vs_qfls": "qfls_vs_suns", "suns_vs_plqy": "plqy_vs_suns",
                 "intensity_suns_qfls": "qfls_vs_suns", "intensity_suns_plqy": "plqy_vs_suns",
                 "qfls_plqy": "plqy_vs_qfls", "g_qfls": "qfls_vs_G", "g_plqy": "plqy_vs_G"}


def canonical_mode(mode: str) -> str:
    m = _MODE_ALIASES.get(mode, mode)
    if m not in SSPL_MODES:
        raise ValueError(f"Unknown ss-PL comparison mode '{mode}'. Use one of {SSPL_MODES}.")
    return m


# ─────────────────────────────────────────────────────────────────────────────
# Row-wise interpolation (TF)
# ─────────────────────────────────────────────────────────────────────────────

def interp_rows(x_rows, y_rows, x_new):
    """
    Piecewise-linear interpolation of each row of y_rows(x_rows) at x_new.

    x_rows, y_rows : (B, N) tensors, x_rows increasing along axis 1
    x_new          : (n,) or (B, n)
    Returns (y_new (B, n), inside (B, n) bool).  Points outside a row's range
    are extrapolated linearly with the first/last segment, which keeps the
    likelihood smooth (``inside`` reports where that happened).
    """
    import tensorflow as tf
    B = tf.shape(x_rows)[0]
    N = tf.shape(x_rows)[1]
    x_new = tf.convert_to_tensor(x_new, x_rows.dtype)
    if x_new.shape.rank == 1:
        x_new = tf.broadcast_to(x_new[None, :], [B, tf.shape(x_new)[0]])
    # bracket per row: last index with x_rows[b, idx] <= x_new[b, i]
    idx = tf.searchsorted(x_rows, x_new, side="right") - 1
    inside = (idx >= 0) & (idx <= N - 2) | tf.equal(x_new, x_rows[:, -1:])
    idx = tf.clip_by_value(idx, 0, N - 2)
    x0 = tf.gather(x_rows, idx, batch_dims=1)
    x1 = tf.gather(x_rows, idx + 1, batch_dims=1)
    y0 = tf.gather(y_rows, idx, batch_dims=1)
    y1 = tf.gather(y_rows, idx + 1, batch_dims=1)
    dx = x1 - x0
    dx = tf.where(tf.abs(dx) > 1e-300, dx, tf.ones_like(dx) * 1e-300)
    w = (x_new - x0) / dx
    return y0 + w * (y1 - y0), inside


def sort_rows(x_rows, *ys):
    """Sort every row of x_rows (and the companion arrays) in increasing order."""
    import tensorflow as tf
    order = tf.argsort(x_rows, axis=1)
    out = [tf.gather(x_rows, order, batch_dims=1)]
    out += [tf.gather(y, order, batch_dims=1) for y in ys]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Forward model
# ─────────────────────────────────────────────────────────────────────────────

class SsplModel:
    """
    Steady-state PL of the SRH model for a batch of parameter vectors theta.

    Parameters
    ----------
    Eg : float               bandgap [eV]
    n_traps : int
    Nc, Nv : float           effective densities of states [cm^-3]
    T : float                temperature [K] (kT = k_B T unless ``kT`` is given)
    n_sweep : (lo, hi, n)    electron-density sweep [cm^-3] (default 1e8 … Nv, 160 points)
    solver : 'auto' | 'analytic' | 'root'
        'auto' uses the closed form for 1 trap and the root solver otherwise.
    n_bisect, n_newton : int iterations of the N-trap root solver
    """

    def __init__(self, Eg: float, n_traps: int, Nc: float = 2.21359e18,
                 Nv: float = 2.21359e18, T: float = 300.0, kT: float | None = None,
                 n_sweep=(1e8, None, 160), solver: str = "auto",
                 n_bisect: int = 64, n_newton: int = 3):
        import tensorflow as tf
        self.Eg = float(Eg)
        self.n_traps = int(n_traps)
        self.Nc, self.Nv = float(Nc), float(Nv)
        self.kT = float(kT) if kT is not None else KB_EV * float(T)
        lo, hi, n = n_sweep
        hi = self.Nv if hi is None else hi
        self.n_sweep = np.logspace(np.log10(lo), np.log10(hi), int(n))
        self._n = tf.constant(self.n_sweep[None, :], tf.float64)        # (1, N)
        self.ni2 = self.Nc * self.Nv * math.exp(-self.Eg / self.kT)
        if solver == "auto":
            solver = "analytic" if self.n_traps == 1 else "root"
        if solver == "analytic" and self.n_traps != 1:
            raise ValueError("The analytic solver is only valid for one trap.")
        self.solver = solver
        self.n_bisect = int(n_bisect)
        self.n_newton = int(n_newton)

    # ------------------------------------------------------------------
    def _trap_arrays(self, theta):
        """Physical trap parameters as (B, 1) tensors, one list entry per trap."""
        import tensorflow as tf
        theta = tf.convert_to_tensor(theta, tf.float64)
        krad = _pow10(theta[:, 0:1])
        traps = []
        for k in range(self.n_traps):
            c = 1 + 4 * k
            Nt = _pow10(theta[:, c:c + 1])
            Et = theta[:, c + 1:c + 2] * self.Eg
            taun = _pow10(theta[:, c + 2:c + 3])
            taup = _pow10(theta[:, c + 3:c + 4])
            n1 = self.Nc * tf.exp((Et - self.Eg) / self.kT)
            p1 = self.Nv * tf.exp(-Et / self.kT)
            traps.append(dict(Nt=Nt, taun=taun, taup=taup, n1=n1, p1=p1))
        return krad, traps

    @staticmethod
    def _nt(n, p, t):
        """Trapped-electron density of one trap, broadcast to (B, N)."""
        return t["Nt"] * (t["taup"] * n + t["taun"] * t["p1"]) / (
            t["taup"] * (n + t["n1"]) + t["taun"] * (p + t["p1"]))

    def _g_and_dg(self, p, n, traps):
        """Neutrality residual g(p) = p - n - sum n_t and its derivative dg/dp > 0."""
        g = p - n
        dg = 1.0
        for t in traps:
            num = t["taup"] * n + t["taun"] * t["p1"]
            den = t["taup"] * (n + t["n1"]) + t["taun"] * (p + t["p1"])
            g = g - t["Nt"] * num / den
            dg = dg + t["Nt"] * num * t["taun"] / den ** 2
        return g, dg

    def _p_analytic(self, n, t):
        """One trap: positive root of a n_t^2 + b n_t + c = 0 (stable form)."""
        import tensorflow as tf
        a = t["taun"]
        b = t["taup"] * t["n1"] + (t["taun"] + t["taup"]) * n + t["taun"] * t["p1"]
        c_abs = t["Nt"] * (t["taup"] * n + t["taun"] * t["p1"])          # = -c > 0
        nt = 2.0 * c_abs / (b + tf.sqrt(b * b + 4.0 * a * c_abs))
        nt = tf.minimum(nt, t["Nt"])
        return n + nt

    def _p_root(self, n, traps):
        """N traps: bracketed root of g(p) with implicit-function gradient."""
        import tensorflow as tf
        sg = tf.stop_gradient
        traps_sg = [{k: sg(v) for k, v in t.items()} for t in traps]
        Nt_sum = tf.add_n([t["Nt"] for t in traps_sg])
        shape = tf.shape(n + Nt_sum)
        log_lo = tf.broadcast_to(tf.math.log(n), shape)
        log_hi = tf.broadcast_to(tf.math.log(n + Nt_sum), shape)

        def body(i, lo, hi):
            mid = 0.5 * (lo + hi)
            g, _ = self._g_and_dg(tf.exp(mid), n, traps_sg)
            lo = tf.where(g < 0.0, mid, lo)
            hi = tf.where(g < 0.0, hi, mid)
            return i + 1, lo, hi

        _, log_lo, log_hi = tf.while_loop(lambda i, lo, hi: i < self.n_bisect, body,
                                          (tf.constant(0), log_lo, log_hi))
        p = tf.exp(0.5 * (log_lo + log_hi))
        p_min, p_max = tf.broadcast_to(n, shape), n + Nt_sum
        for _ in range(self.n_newton):                    # polish (no gradient)
            g, dg = self._g_and_dg(p, n, traps_sg)
            p = tf.clip_by_value(p - g / dg, p_min, p_max)
        p = sg(p)
        # one differentiable Newton step = implicit differentiation
        g, dg = self._g_and_dg(p, n, traps)
        return p - g / sg(dg)

    # ------------------------------------------------------------------
    def solve(self, theta) -> dict:
        """
        Steady state on the n sweep for every row of theta.

        Returns dict of (B, N) tensors: n, p, qfls [eV], plqy, G [cm^-3 s^-1], nt_<k> [cm^-3] (trapped electrons
        of trap k, k = 1 … n_traps; p = n + sum_k nt_<k>).
        """
        import tensorflow as tf
        krad, traps = self._trap_arrays(theta)
        n = self._n
        p = self._p_analytic(n, traps[0]) if self.solver == "analytic" else self._p_root(n, traps)
        np_ = n * p
        excess = np_ - self.ni2
        R_rad = krad * excess
        R_srh = tf.add_n([excess / ((n + t["n1"]) * t["taup"] + (p + t["p1"]) * t["taun"])
                          for t in traps])
        G = R_rad + R_srh
        return {"n": tf.broadcast_to(n, tf.shape(p)), "p": p,
                "qfls": self.kT * tf.math.log(np_ / self.ni2),
                "plqy": R_rad / G, "G": G,
                **{f"nt_{k + 1}": self._nt(n, p, t) for k, t in enumerate(traps)}}


# ─────────────────────────────────────────────────────────────────────────────
# Measured ss-PL data + comparison mode
# ─────────────────────────────────────────────────────────────────────────────

class SsplObservation:
    """
    Model prediction of one measured ss-PL data set.

    Parameters
    ----------
    model : SsplModel
    mode : str
        'qfls_vs_suns'  QFLS [eV] vs intensity [suns]  (residual in eV)
        'plqy_vs_suns'  log10 PLQY vs intensity [suns]
        'plqy_vs_qfls'  log10 PLQY vs QFLS [eV]
        'qfls_vs_G', 'plqy_vs_G'  as above, x = generation rate [cm^-3 s^-1]
        (legacy names such as 'suns_vs_qfls' are accepted)
    x_exp, y_exp : arrays
        Measured abscissa (suns, G or QFLS) and ordinate (QFLS [eV] or PLQY
        as a FRACTION — log10 is taken here).
    G_per_sun : float
        Volumetric generation rate at 1 sun [cm^-3 s^-1] (only *_vs_suns).
    """

    def __init__(self, model: SsplModel, mode: str, x_exp, y_exp, G_per_sun: float | None = None):
        import tensorflow as tf
        self.model = model
        self.mode = canonical_mode(mode)
        x_exp = np.asarray(x_exp, np.float64)
        y_exp = np.asarray(y_exp, np.float64)
        ok = np.isfinite(x_exp) & np.isfinite(y_exp)
        if self.mode != "plqy_vs_qfls":
            ok &= x_exp > 0
        if self.mode.startswith("plqy"):
            ok &= y_exp > 0
        self.x_exp, self.y_exp_raw = x_exp[ok], y_exp[ok]
        self.n_points = int(ok.sum())
        if self.mode.endswith("_suns"):
            if G_per_sun is None:
                raise ValueError(f"mode '{self.mode}' needs G_per_sun")
            self.G_per_sun = float(G_per_sun)
            x_model = np.log10(self.x_exp * self.G_per_sun)          # log10 G
        elif self.mode.endswith("_G"):
            self.G_per_sun = None
            x_model = np.log10(self.x_exp)
        else:
            self.G_per_sun = None
            x_model = self.x_exp                                       # QFLS
        self._x_model = tf.constant(x_model, tf.float64)
        # observed y in the residual space
        self.y_obs = np.log10(self.y_exp_raw) if self.mode.startswith("plqy") else self.y_exp_raw.copy()
        self.residual_unit = "log10 PLQY" if self.mode.startswith("plqy") else "eV"

    def predict(self, theta, return_inside: bool = False):
        """Model y at the measured x for every row of theta, (B, n_points)."""
        import tensorflow as tf
        s = self.model.solve(theta)
        y = tf.math.log(s["plqy"]) / math.log(10.0) if self.mode.startswith("plqy") else s["qfls"]
        x = s["qfls"] if self.mode == "plqy_vs_qfls" else tf.math.log(s["G"]) / math.log(10.0)
        x, y = sort_rows(x, y)
        y_new, inside = interp_rows(x, y, self._x_model)
        return (y_new, inside) if return_inside else y_new

    def model_curve(self, theta) -> dict:
        """Full simulated curves (numpy) in the units of this mode, for plotting."""
        s = {k: v.numpy() for k, v in self.model.solve(theta).items()}
        if self.mode.endswith("_suns"):
            x = s["G"] / self.G_per_sun
        elif self.mode.endswith("_G"):
            x = s["G"]
        else:
            x = s["qfls"]
        y = s["plqy"] if self.mode.startswith("plqy") else s["qfls"]
        return {"x": x, "y": y, **s}


# ─────────────────────────────────────────────────────────────────────────────
# numpy twin (for CPU worker processes, e.g. the ODE refinement with curve_fit)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SsplSpecNumpy:
    """
    Picklable numpy version of SsplModel + SsplObservation (same physics,
    same solver strategy, same per-row interpolation incl. linear
    extrapolation).  Create it with ``SsplObservation.numpy_spec()``.
    """
    Eg: float
    n_traps: int
    Nc: float
    Nv: float
    kT: float
    n_sweep: np.ndarray
    solver: str
    mode: str
    x_model: np.ndarray            # log10 G or QFLS of the measured points
    n_bisect: int = 64
    n_newton: int = 3

    def solve(self, theta) -> dict:
        """Steady state on the n sweep for every row of theta (numpy)."""
        th = np.atleast_2d(np.asarray(theta, np.float64))
        n = self.n_sweep[None, :]
        krad = 10.0 ** th[:, 0:1]
        traps = []
        for k in range(self.n_traps):
            c = 1 + 4 * k
            Et = th[:, c + 1:c + 2] * self.Eg
            traps.append(dict(Nt=10.0 ** th[:, c:c + 1], taun=10.0 ** th[:, c + 2:c + 3],
                              taup=10.0 ** th[:, c + 3:c + 4],
                              n1=self.Nc * np.exp((Et - self.Eg) / self.kT),
                              p1=self.Nv * np.exp(-Et / self.kT)))

        def g_dg(p):
            g, dg = p - n, 1.0
            for t in traps:
                num = t["taup"] * n + t["taun"] * t["p1"]
                den = t["taup"] * (n + t["n1"]) + t["taun"] * (p + t["p1"])
                g = g - t["Nt"] * num / den
                dg = dg + t["Nt"] * num * t["taun"] / den ** 2
            return g, dg

        if self.solver == "analytic" or (self.solver == "auto" and self.n_traps == 1):
            t = traps[0]
            a = t["taun"]
            b = t["taup"] * t["n1"] + (t["taun"] + t["taup"]) * n + t["taun"] * t["p1"]
            c_abs = t["Nt"] * (t["taup"] * n + t["taun"] * t["p1"])
            p = n + np.minimum(2.0 * c_abs / (b + np.sqrt(b * b + 4.0 * a * c_abs)), t["Nt"])
        else:
            Nt_sum = sum(t["Nt"] for t in traps)
            lo = np.broadcast_to(np.log(n), (len(th), n.shape[1])).copy()
            hi = np.log(n + Nt_sum) * np.ones_like(lo)
            for _ in range(self.n_bisect):
                mid = 0.5 * (lo + hi)
                g, _ = g_dg(np.exp(mid))
                neg = g < 0.0
                lo = np.where(neg, mid, lo)
                hi = np.where(neg, hi, mid)
            p = np.exp(0.5 * (lo + hi))
            for _ in range(self.n_newton):
                g, dg = g_dg(p)
                p = np.clip(p - g / dg, n, n + Nt_sum)
        ni2 = self.Nc * self.Nv * math.exp(-self.Eg / self.kT)
        excess = n * p - ni2
        R_rad = krad * excess
        R_srh = sum(excess / ((n + t["n1"]) * t["taup"] + (p + t["p1"]) * t["taun"]) for t in traps)
        G = R_rad + R_srh
        return {"n": np.broadcast_to(n, p.shape), "p": p, "qfls": self.kT * np.log(n * p / ni2),
                "plqy": R_rad / G, "G": G,
                **{f"nt_{k + 1}": SsplModel._nt(n, p, t) for k, t in enumerate(traps)}}

    def predict(self, theta) -> np.ndarray:
        """Model y at the measured x (rows of theta), linear extrapolation outside."""
        s = self.solve(theta)
        y = np.log10(s["plqy"]) if self.mode.startswith("plqy") else s["qfls"]
        x = s["qfls"] if self.mode == "plqy_vs_qfls" else np.log10(s["G"])
        out = np.empty((len(x), len(self.x_model)))
        for r in range(len(x)):
            o = np.argsort(x[r])
            xr, yr = x[r][o], y[r][o]
            j = np.clip(np.searchsorted(xr, self.x_model, side="right") - 1, 0, len(xr) - 2)
            w = (self.x_model - xr[j]) / (xr[j + 1] - xr[j])
            out[r] = yr[j] + w * (yr[j + 1] - yr[j])
        return out


def _numpy_spec(self) -> "SsplSpecNumpy":
    """Picklable numpy twin of this observation (see SsplSpecNumpy)."""
    m = self.model
    return SsplSpecNumpy(Eg=m.Eg, n_traps=m.n_traps, Nc=m.Nc, Nv=m.Nv, kT=m.kT,
                         n_sweep=np.asarray(m.n_sweep), solver=m.solver, mode=self.mode,
                         x_model=np.asarray(self._x_model.numpy()),
                         n_bisect=m.n_bisect, n_newton=m.n_newton)


SsplObservation.numpy_spec = _numpy_spec
