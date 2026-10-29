"""
prior.py — prior support, ordering constraint and unconstrained coordinates
===========================================================================

Prior (REVISION_PLAN.md, Section 2 item 5)
------------------------------------------
* Uniform on a box in the NN input coordinates theta
  (log10 for k_rad, N_t, tau_n, tau_p; E_t/Eg linear).
* The box is the training box of the network (``lb``/``ub`` of the training
  HDF5): prior = optimisation = sampling = training bounds (R1 comment 6,
  Table S1).  Narrower bounds may be given explicitly; wider bounds are
  refused because they would require extrapolation of the surrogate.
* For several traps the trap energies are ordered, E_t1 < E_t2 < ... .
  This removes the label-switching mirror modes (which break R-hat and SBC)
  and — for the current networks — is also required to stay inside the
  training domain: the 2-trap Sobol designs contain ONLY E_t1 < E_t2.
* Optional "photodoping" constraint N_t,k > n_1,k = N_C exp((E_t,k - E_g)/kT)
  for selected traps (in the legacy optimiser this was a penalty; here it is
  part of the prior support).
* Fixed parameters (e.g. k_rad = 1e-11 cm^3/s) are represented by a
  degenerate interval of half-width FIXED_HALF_WIDTH around the value; all
  samplers/optimisers work unchanged and ``Prior.free_idx`` lists the
  parameters that are actually inferred (plots and tables skip the others).

Unconstrained coordinates
-------------------------
NUTS cannot handle hard walls and L-BFGS/CMA-ES work best without bounds, so
all of them operate on u in R^D.  ``Prior.forward(u)`` maps u to theta and
returns log|det J|; the density in u is  p(u) = pi(theta(u)) |J(u)|.

* box dimension:     theta = lo + (hi - lo) * sigmoid(u)
* ordered energies:  e_k = e_{k-1} + (b - e_{k-1}) * sigmoid(u_k),  e_0 = a
                     (implemented through r_k = log(b - e_k), which is
                     numerically stable for u -> +-inf)
* photodoping N_t:   theta = L(e_k) + (hi - L(e_k)) * sigmoid(u),
                     L(e) = max(lo, log10 n_1(e))
The Jacobian is triangular (each theta_i depends on its own u_i and on
earlier energies only), so log|J| is the sum of the diagonal terms.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

KB_EV = 8.617333262e-5          # Boltzmann constant [eV/K]
FIXED_HALF_WIDTH = 1e-9         # half-width of the interval of a fixed parameter (theta units)
LN10 = math.log(10.0)

# position of each trap parameter inside a 4-block
_TRAP_FIELDS = ("Nt", "Et", "taun", "taup")


def param_names(n_traps: int) -> list:
    """Names of the theta components (see module docstring of the package)."""
    names = ["log10_krad"]
    for k in range(1, n_traps + 1):
        names += [f"log10_Nt_{k}", f"Et_rel_{k}", f"log10_taun_{k}", f"log10_taup_{k}"]
    return names


def display_names(n_traps: int) -> list:
    """Axis labels for plots/tables (display coordinates: E_t in eV)."""
    labels = [r"$\log_{10} k_\mathrm{rad}$"]
    for k in range(1, n_traps + 1):
        s = str(k) if n_traps > 1 else ""
        labels += [rf"$\log_{{10}} N_{{t{s}}}$", rf"$E_{{t{s}}}$ [eV]",
                   rf"$\log_{{10}} \tau_{{n{s}}}$", rf"$\log_{{10}} \tau_{{p{s}}}$"]
    return labels


def display_keys(n_traps: int) -> list:
    """Machine-readable names in display coordinates (ArviZ variables, CSV columns)."""
    keys = ["log10_krad"]
    for k in range(1, n_traps + 1):
        keys += [f"log10_Nt_{k}", f"Et_{k}_eV", f"log10_taun_{k}", f"log10_taup_{k}"]
    return keys


def theta_index(field: str, trap: int | None = None) -> int:
    """Index of a parameter in theta.  trap is 1-based; field in krad/Nt/Et/taun/taup."""
    if field == "krad":
        return 0
    return 1 + 4 * (int(trap) - 1) + _TRAP_FIELDS.index(field)


class Prior:
    """
    Uniform prior on the (constrained) training box, with the bijection to R^D.

    Parameters
    ----------
    lo, hi : array (D,)
        Box bounds in theta coordinates.
    n_traps : int
    Eg : float
        Bandgap [eV] (converts E_t/Eg <-> E_t and enters n_1).
    order_traps : bool
        Enforce E_t1 < E_t2 < ... (default True for n_traps > 1).  All traps
        must then share the same E_t bounds.
    photodoping_traps : sequence of int
        1-based trap indices with the constraint N_t > n_1 (default none).
    Nc : float
        Effective density of states of the conduction band [cm^-3] (for n_1).
    T : float
        Temperature [K] (for n_1).
    """

    def __init__(self, lo, hi, n_traps: int, Eg: float, order_traps: bool | None = None,
                 photodoping_traps=(), Nc: float = 2.21359e18, T: float = 300.0):
        self.lo = np.asarray(lo, np.float64).copy()
        self.hi = np.asarray(hi, np.float64).copy()
        self.n_traps = int(n_traps)
        self.dim = 1 + 4 * self.n_traps
        if self.lo.shape != (self.dim,) or self.hi.shape != (self.dim,):
            raise ValueError(f"lo/hi must have shape ({self.dim},)")
        if np.any(self.hi <= self.lo):
            raise ValueError("Every prior dimension needs hi > lo.")
        self.Eg = float(Eg)
        self.Nc = float(Nc)
        self.T = float(T)
        self.kT = KB_EV * self.T
        self.names = param_names(self.n_traps)
        self.labels = display_names(self.n_traps)
        self.display_keys = display_keys(self.n_traps)
        self.order_traps = (self.n_traps > 1) if order_traps is None else bool(order_traps)
        self.photodoping_traps = tuple(sorted(int(k) for k in photodoping_traps))

        self.idx_Et = [theta_index("Et", k) for k in range(1, self.n_traps + 1)]
        self.idx_Nt = [theta_index("Nt", k) for k in range(1, self.n_traps + 1)]
        if self.order_traps and self.n_traps > 1:
            if not (np.allclose(self.lo[self.idx_Et], self.lo[self.idx_Et[0]]) and
                    np.allclose(self.hi[self.idx_Et], self.hi[self.idx_Et[0]])):
                raise ValueError("Trap ordering requires identical E_t bounds for all traps.")
        for k in self.photodoping_traps:
            if not 1 <= k <= self.n_traps:
                raise ValueError(f"photodoping trap {k} out of range")
            # the constraint must leave room below the upper N_t bound everywhere
            if self._log10_n1(self.hi[self.idx_Et[k - 1]]) >= self.hi[self.idx_Nt[k - 1]]:
                raise ValueError(f"photodoping constraint for trap {k} is infeasible "
                                 f"at the upper E_t bound.")
        # dims that use the plain sigmoid map
        special = set(self.idx_Et if self.order_traps else [])
        special |= {self.idx_Nt[k - 1] for k in self.photodoping_traps}
        self._box_dims = [i for i in range(self.dim) if i not in special]
        self.log_volume = self._compute_log_volume()
        # fixed parameters = degenerate intervals (see module docstring)
        self.fixed_mask = (self.hi - self.lo) < 1e-6
        self.free_idx = [i for i in range(self.dim) if not self.fixed_mask[i]]
        if self.order_traps and np.any(self.fixed_mask[self.idx_Et]):
            raise ValueError("Fixing a trap energy together with trap ordering is not supported.")

    # ------------------------------------------------------------------
    @classmethod
    def from_artifacts(cls, artifacts, Eg: float, order_traps: bool | None = None,
                       photodoping_traps=(), bounds_override: dict | None = None,
                       fixed: dict | None = None, T: float = 300.0) -> "Prior":
        """
        Prior on the training box of a network.

        ``bounds_override`` maps theta names to (lo, hi) to NARROW individual
        bounds, e.g. ``{"log10_Nt_1": (12, 18.345)}``.  Bounds outside the
        training box raise an error (extrapolation of the surrogate).
        ``fixed`` maps theta names to values that are held fixed, e.g.
        ``{"log10_krad": -11.0}`` (must lie inside the training box).
        """
        lo = np.asarray(artifacts.lb[2:], np.float64).copy()
        hi = np.asarray(artifacts.ub[2:], np.float64).copy()
        names = param_names(artifacts.n_traps)
        bounds_override = dict(bounds_override or {})
        for name, v in (fixed or {}).items():
            bounds_override[name] = (float(v) - FIXED_HALF_WIDTH, float(v) + FIXED_HALF_WIDTH)
        for name, (l, h) in bounds_override.items():
            i = names.index(name)
            if l < lo[i] - 1e-9 or h > hi[i] + 1e-9:
                raise ValueError(f"{name}: override ({l}, {h}) exceeds the training "
                                 f"bounds ({lo[i]}, {hi[i]}).")
            lo[i], hi[i] = float(l), float(h)
        return cls(lo, hi, artifacts.n_traps, Eg, order_traps=order_traps,
                   photodoping_traps=photodoping_traps, Nc=artifacts.constants["Nc"], T=T)

    # ------------------------------------------------------------------
    # physics helpers
    # ------------------------------------------------------------------
    def _log10_n1(self, et_rel):
        """log10 n_1 = log10 Nc + (E_t - E_g) / (kT ln10)  for E_t = et_rel * Eg."""
        return math.log10(self.Nc) + (et_rel - 1.0) * self.Eg / (self.kT * LN10)

    def _pd_lower(self, k: int, et_rel):
        """Lower N_t bound of photodoping trap k (1-based), works for numpy and TF."""
        lo = self.lo[self.idx_Nt[k - 1]]
        l1 = math.log10(self.Nc) + (et_rel - 1.0) * (self.Eg / (self.kT * LN10))
        try:
            import tensorflow as tf
            if tf.is_tensor(et_rel):
                return tf.maximum(tf.constant(lo, tf.float64), l1)
        except ImportError:
            pass
        return np.maximum(lo, l1)

    # ------------------------------------------------------------------
    # volume of the support (needed only for normalised densities / evidence)
    # ------------------------------------------------------------------
    def _compute_log_volume(self) -> float:
        lv = float(np.sum(np.log(self.hi[self._box_dims] - self.lo[self._box_dims])))
        if self.order_traps:
            a, b = self.lo[self.idx_Et[0]], self.hi[self.idx_Et[0]]
            lv += self.n_traps * math.log(b - a) - math.lgamma(self.n_traps + 1)
        if self.photodoping_traps:
            # Monte Carlo over the (ordered) E_t region of the N_t ranges
            rng = np.random.default_rng(12345)
            e = rng.uniform(self.lo[self.idx_Et], self.hi[self.idx_Et], size=(200_000, self.n_traps))
            if self.order_traps:
                e = np.sort(e, axis=1)
            width = np.ones(len(e))
            for k in self.photodoping_traps:
                width *= self.hi[self.idx_Nt[k - 1]] - self._pd_lower(k, e[:, k - 1])
            lv += math.log(float(np.mean(width)))
        return lv

    # ------------------------------------------------------------------
    # bijection u (R^D) -> theta
    # ------------------------------------------------------------------
    @staticmethod
    def _ordered_chain(u_cols, a, b):
        """
        Map unconstrained u_1..u_m to a < e_1 < ... < e_m < b.

        Uses r_k = log(b - e_k) = log(b - a) + sum_{j<=k} log sigmoid(-u_j),
        e_k = b - exp(r_k); the log-Jacobian of step k is
        r_{k-1} + log sigmoid(u_k) + log sigmoid(-u_k).
        a, b may be floats or (B,) tensors.
        """
        import tensorflow as tf
        r_prev = tf.math.log(b - a)
        e_list, logdet = [], 0.0
        for uk in u_cols:
            ls_pos = tf.math.log_sigmoid(uk)
            ls_neg = tf.math.log_sigmoid(-uk)
            logdet = logdet + r_prev + ls_pos + ls_neg
            r_k = r_prev + ls_neg
            e_list.append(b - tf.exp(r_k))
            r_prev = r_k
        return e_list, logdet

    def forward(self, u, fixed: tuple | None = None):
        """
        theta(u) and log|det J|.

        Parameters
        ----------
        u : (B, D) tensor (float64)
        fixed : (j, value) or None
            Hold theta_j at ``value`` (scalar or (B,)) — used by the profile
            likelihood.  Column j of u is then ignored and the remaining
            parameters are mapped onto the support conditional on theta_j
            (for an ordered energy: the other energies stay ordered around it).

        Returns
        -------
        theta : (B, D) tensor, log_det : (B,) tensor
        """
        import tensorflow as tf
        u = tf.convert_to_tensor(u, tf.float64)
        B = tf.shape(u)[0]
        cols = [None] * self.dim
        logdet = tf.zeros([B], tf.float64)
        j_fix = None
        if fixed is not None:
            j_fix = int(fixed[0])
            v_fix = tf.broadcast_to(tf.convert_to_tensor(fixed[1], tf.float64), [B])
            if j_fix in [self.idx_Nt[k - 1] for k in self.photodoping_traps]:
                raise NotImplementedError("Profiling a photodoping-constrained N_t is not supported.")
            cols[j_fix] = v_fix

        # 1) plain box dimensions
        for i in self._box_dims:
            if i == j_fix:
                continue
            ui = u[:, i]
            lo, hi = self.lo[i], self.hi[i]
            cols[i] = lo + (hi - lo) * tf.sigmoid(ui)
            logdet += math.log(hi - lo) + tf.math.log_sigmoid(ui) + tf.math.log_sigmoid(-ui)

        # 2) ordered trap energies
        if self.order_traps:
            a = tf.constant(self.lo[self.idx_Et[0]], tf.float64)
            b = tf.constant(self.hi[self.idx_Et[0]], tf.float64)
            if j_fix in self.idx_Et:
                m = self.idx_Et.index(j_fix)
                below = [u[:, i] for i in self.idx_Et[:m]]
                above = [u[:, i] for i in self.idx_Et[m + 1:]]
                e_lo, ld_lo = self._ordered_chain(below, a, v_fix)
                e_hi, ld_hi = self._ordered_chain(above, v_fix, b)
                for i, e in zip(self.idx_Et[:m], e_lo):
                    cols[i] = e
                for i, e in zip(self.idx_Et[m + 1:], e_hi):
                    cols[i] = e
                logdet += ld_lo + ld_hi
            else:
                e_all, ld = self._ordered_chain([u[:, i] for i in self.idx_Et], a, b)
                for i, e in zip(self.idx_Et, e_all):
                    cols[i] = e
                logdet += ld

        # 3) photodoping N_t (lower bound depends on the trap's own energy)
        for k in self.photodoping_traps:
            i = self.idx_Nt[k - 1]
            L = self._pd_lower(k, cols[self.idx_Et[k - 1]])
            hi = tf.constant(self.hi[i], tf.float64)
            ui = u[:, i]
            cols[i] = L + (hi - L) * tf.sigmoid(ui)
            logdet += tf.math.log(hi - L) + tf.math.log_sigmoid(ui) + tf.math.log_sigmoid(-ui)

        return tf.stack(cols, axis=1), logdet

    def forward_np(self, u) -> np.ndarray:
        """
        theta(u) in numpy (no TensorFlow), same map as :meth:`forward`.
        Used by the ODE refinement, which runs in CPU worker processes.
        """
        u = np.atleast_2d(np.asarray(u, np.float64))
        th = np.zeros_like(u)

        def _sig(x):
            return 0.5 * (1.0 + np.tanh(0.5 * x))

        for i in self._box_dims:
            th[:, i] = self.lo[i] + (self.hi[i] - self.lo[i]) * _sig(u[:, i])
        if self.order_traps:
            a, b = self.lo[self.idx_Et[0]], self.hi[self.idx_Et[0]]
            r = np.full(len(u), np.log(b - a))
            for i in self.idx_Et:
                r = r - np.logaddexp(0.0, u[:, i])          # += log sigmoid(-u)
                th[:, i] = b - np.exp(r)
        for k in self.photodoping_traps:
            i = self.idx_Nt[k - 1]
            L = self._pd_lower(k, th[:, self.idx_Et[k - 1]])
            th[:, i] = L + (self.hi[i] - L) * _sig(u[:, i])
        return th

    def inverse(self, theta, eps: float = 1e-9) -> np.ndarray:
        """
        u(theta) (numpy).  Points on or outside the support boundary are
        clipped to a fraction ``eps`` inside it.
        """
        theta = np.atleast_2d(np.asarray(theta, np.float64))
        u = np.zeros_like(theta)

        def _logit(s):
            s = np.clip(s, eps, 1.0 - eps)
            return np.log(s) - np.log1p(-s)

        for i in self._box_dims:
            u[:, i] = _logit((theta[:, i] - self.lo[i]) / (self.hi[i] - self.lo[i]))
        if self.order_traps:
            a, b = self.lo[self.idx_Et[0]], self.hi[self.idx_Et[0]]
            e_prev = np.full(len(theta), a)
            for i in self.idx_Et:
                s = (theta[:, i] - e_prev) / (b - e_prev)
                u[:, i] = _logit(s)
                # continue the chain from the value actually represented by u
                e_prev = e_prev + (b - e_prev) * np.clip(s, eps, 1.0 - eps)
        for k in self.photodoping_traps:
            i = self.idx_Nt[k - 1]
            L = self._pd_lower(k, theta[:, self.idx_Et[k - 1]])
            u[:, i] = _logit((theta[:, i] - L) / (self.hi[i] - L))
        return u

    # ------------------------------------------------------------------
    # density in theta coordinates
    # ------------------------------------------------------------------
    def in_support(self, theta, tol: float = 0.0) -> np.ndarray:
        """Boolean (B,): theta inside the box and satisfying all constraints."""
        theta = np.atleast_2d(np.asarray(theta, np.float64))
        ok = np.all((theta >= self.lo - tol) & (theta <= self.hi + tol), axis=1)
        if self.order_traps:
            e = theta[:, self.idx_Et]
            ok &= np.all(np.diff(e, axis=1) > 0, axis=1)
        for k in self.photodoping_traps:
            ok &= theta[:, self.idx_Nt[k - 1]] >= self._pd_lower(k, theta[:, self.idx_Et[k - 1]]) - tol
        return ok

    def log_prob(self, theta):
        """log pi(theta): -log(volume) inside the support, -inf outside (TF, (B,))."""
        import tensorflow as tf
        theta = tf.convert_to_tensor(theta, tf.float64)
        ok = tf.reduce_all((theta >= self.lo) & (theta <= self.hi), axis=1)
        if self.order_traps:
            e = tf.gather(theta, self.idx_Et, axis=1)
            ok &= tf.reduce_all(e[:, 1:] > e[:, :-1], axis=1)
        for k in self.photodoping_traps:
            ok &= theta[:, self.idx_Nt[k - 1]] >= self._pd_lower(k, theta[:, self.idx_Et[k - 1]])
        return tf.where(ok, tf.constant(-self.log_volume, tf.float64),
                        tf.constant(-np.inf, tf.float64))

    # ------------------------------------------------------------------
    # draws
    # ------------------------------------------------------------------
    def _finalise_draws(self, th: np.ndarray) -> np.ndarray:
        """Sort energies (uniform on the ordered region) and flag photodoping violations."""
        if self.order_traps:
            # Sorting only the energy values is valid because all other trap
            # parameters are iid uniform and independent of the energies.
            th[:, self.idx_Et] = np.sort(th[:, self.idx_Et], axis=1)
        return th

    def sample(self, n: int, rng=None) -> np.ndarray:
        """n independent prior draws (numpy, (n, D))."""
        rng = np.random.default_rng(rng)
        out = []
        while sum(len(o) for o in out) < n:
            th = self._finalise_draws(rng.uniform(self.lo, self.hi, size=(2 * n, self.dim)))
            out.append(th[self.in_support(th)])
        return np.concatenate(out)[:n]

    def sobol(self, n: int, seed: int = 0) -> np.ndarray:
        """n scrambled-Sobol points over the prior (space-filling optimiser starts)."""
        from scipy.stats import qmc
        m = int(np.ceil(np.log2(max(2, 2 * n))))
        pts = qmc.Sobol(d=self.dim, scramble=True, seed=seed).random_base2(m)
        th = self._finalise_draws(self.lo + (self.hi - self.lo) * pts)
        th = th[self.in_support(th)]
        if len(th) < n:
            th = np.concatenate([th, self.sample(n - len(th), rng=seed)])
        return th[:n]

    def project(self, theta, margin: float = 1e-4) -> np.ndarray:
        """
        Move parameter vectors into the support (used for start values).

        Values are clipped to a relative ``margin`` inside the box; with
        ordering, whole trap blocks are re-labelled so that E_t increases
        (this only renames the traps, it does not change the physics); a
        photodoping N_t below n_1 is raised to n_1.
        """
        th = np.atleast_2d(np.asarray(theta, np.float64)).copy()
        span = self.hi - self.lo
        th = np.clip(th, self.lo + margin * span, self.hi - margin * span)
        if self.order_traps:
            for r in range(len(th)):
                blocks = th[r, 1:].reshape(self.n_traps, 4)
                blocks = blocks[np.argsort(blocks[:, 1])]
                # separate identical energies slightly
                for k in range(1, self.n_traps):
                    if blocks[k, 1] <= blocks[k - 1, 1]:
                        blocks[k, 1] = blocks[k - 1, 1] + margin * span[self.idx_Et[0]]
                th[r, 1:] = blocks.ravel()
        for k in self.photodoping_traps:
            i = self.idx_Nt[k - 1]
            L = self._pd_lower(k, th[:, self.idx_Et[k - 1]])
            th[:, i] = np.maximum(th[:, i], L + margin * (self.hi[i] - L))
        return th

    # ------------------------------------------------------------------
    # conversions
    # ------------------------------------------------------------------
    def to_display(self, theta) -> np.ndarray:
        """Display coordinates: theta with E_t/Eg converted to E_t [eV]."""
        d = np.atleast_2d(np.asarray(theta, np.float64)).copy()
        d[..., self.idx_Et] *= self.Eg
        return d

    def display_bounds(self) -> np.ndarray:
        """(D, 2) prior bounds in display coordinates."""
        b = np.stack([self.lo, self.hi], axis=1).copy()
        b[self.idx_Et] *= self.Eg
        return b

    def to_physical(self, theta) -> pd.DataFrame:
        """Physical units: k_rad [cm^3/s], N_t [cm^-3], E_t [eV], tau [s]."""
        th = np.atleast_2d(np.asarray(theta, np.float64))
        out = {"krad_cm3s": 10 ** th[:, 0]}
        for k in range(1, self.n_traps + 1):
            out[f"Et_{k}_eV"] = th[:, theta_index("Et", k)] * self.Eg
            out[f"Nt_{k}_cm-3"] = 10 ** th[:, theta_index("Nt", k)]
            out[f"tau_n_{k}_s"] = 10 ** th[:, theta_index("taun", k)]
            out[f"tau_p_{k}_s"] = 10 ** th[:, theta_index("taup", k)]
        return pd.DataFrame(out)

    def from_physical(self, Et_eV, Nt, taun, taup, krad) -> np.ndarray:
        """theta (D,) from physical values (Et_eV, Nt, taun, taup: one per trap)."""
        Et_eV, Nt, taun, taup = (np.atleast_1d(np.asarray(x, np.float64)) for x in (Et_eV, Nt, taun, taup))
        th = np.empty(self.dim)
        th[0] = np.log10(float(krad))
        for k in range(1, self.n_traps + 1):
            th[theta_index("Nt", k)] = np.log10(Nt[k - 1])
            th[theta_index("Et", k)] = Et_eV[k - 1] / self.Eg
            th[theta_index("taun", k)] = np.log10(taun[k - 1])
            th[theta_index("taup", k)] = np.log10(taup[k - 1])
        return th

    def describe(self) -> pd.DataFrame:
        """Bounds table (theta and physical units) — the prior part of Table S1."""
        rows = []
        for i, name in enumerate(self.names):
            lo, hi = self.lo[i], self.hi[i]
            if name.startswith("Et_rel"):
                phys = (f"{lo * self.Eg:.4f} – {hi * self.Eg:.4f} eV", "linear")
            elif name == "log10_krad":
                phys = (f"{10**lo:.3g} – {10**hi:.3g} cm^3/s", "log10")
            elif name.startswith("log10_Nt"):
                phys = (f"{10**lo:.3g} – {10**hi:.3g} cm^-3", "log10")
            else:
                phys = (f"{10**lo:.3g} – {10**hi:.3g} s", "log10")
            if self.fixed_mask[i]:
                phys = ("fixed at " + phys[0].split(" – ")[0] + phys[0].split(" – ")[1][phys[0].split(" – ")[1].find(" "):],
                        "fixed")
            rows.append({"parameter": name, "theta_lo": lo, "theta_hi": hi,
                         "physical_range": phys[0], "scale": phys[1]})
        df = pd.DataFrame(rows)
        notes = []
        if self.order_traps:
            notes.append("E_t1 < ... < E_tK")
        if self.photodoping_traps:
            notes.append(f"N_t > n_1 for traps {list(self.photodoping_traps)}")
        df.attrs["constraints"] = ", ".join(notes) if notes else "none"
        return df
