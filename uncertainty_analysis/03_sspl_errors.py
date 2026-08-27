"""
03_sspl_errors.py  —  SSPL error computation (physics-based)
=============================================================
Reads a grid CSV and computes the MSE between the simulated SSPL curve
and the experimental data for every grid point.

Two physics solvers:
  analytic   Quadratic analytical solution for the 1-trap charge-neutrality
             equation (fast, exact, identical to run_sspl_gridsearch.py).
             Only valid for num_traps == 1.
  numeric    Fixed-point iteration for p = n + Σ_k nt_k.  Works for any
             number of traps (including 1).  Slightly slower.

Three comparison modes (config["sspl"]["compare"]):
  plqy_vs_qfls  (default) Compare log10(PLQY) vs QFLS.
                Experimental CSV must have columns: qfls_eV, plqy.
  plqy_vs_G     Compare log10(PLQY) vs generation rate G.
                Interpolation is done on the log10(G) axis.
                Experimental CSV must have columns: G_cm3s, plqy.
  qfls_vs_G     Compare QFLS (eV, linear) vs generation rate G.
                Interpolation is done on the log10(G) axis.
                Experimental CSV must have columns: G_cm3s, qfls_eV.
  suns_vs_qfls  Compare QFLS vs intensity (suns→G conversion via AM1.5G).
                Tab-separated file with units row at row 2 (skipped).
                Experimental CSV must have columns: intensity, qfls.
  suns_vs_plqy  Compare log10(PLQY) vs intensity (suns→G conversion via AM1.5G).
                Tab-separated file with units row at row 2 (skipped).
                Experimental CSV must have columns: intensity, plqy.

The solver is selected by config["sspl"]["solver"] and automatically falls
back to "numeric" when num_traps > 1.

Output
------
  <output_dir>/errors_sspl.csv
      all parameter columns  +  error_sspl_mse

Usage
-----
  python 03_sspl_errors.py --config config.json --grid grid.csv
"""

import argparse, json, os, sys, time
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.integrate import trapezoid as _trapz

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _p(msg): print(msg, flush=True)


# ── Physics constants / sweep ─────────────────────────────────────────────────

def _ni2(NC, NV, Eg, kT):
    return NC * NV * np.exp(-Eg / kT)


def _n_sweep(lo, hi, n):
    return np.logspace(np.log10(lo), np.log10(hi), n)


# ── 1-trap analytic solver (quadratic) — matches run_sspl_gridsearch.py ───────

def _sspl_analytic(Et, Nt, taun, taup, krad, n_sw, NC, NV, Eg, kT, ni2_val):
    """
    Vectorised for M grid points × N_N sweep points.
    Et, Nt, taun, taup, krad: (M,)
    n_sw:  (N_N,)
    Returns qfls_sim (M, N_N), plqy_sim (M, N_N).
    """
    Et_c   = Et  [:, None];  Nt_c   = Nt  [:, None]
    taun_c = taun[:, None];  taup_c = taup[:, None];  krad_c = krad[:, None]
    n = n_sw[None, :]

    n1 = NC * np.exp((Et_c - Eg) / kT)
    p1 = NV * np.exp(-Et_c / kT)

    aa  =  taun_c
    bb  =  taup_c * n1  + (taun_c + taup_c) * n + taun_c * p1
    cc  = -Nt_c   * (taup_c * n + taun_c * p1)
    disc = np.maximum(bb**2 - 4.0 * aa * cc, 0.0)
    nt  = (-bb + np.sqrt(disc)) / (2.0 * aa)
    nt  = np.clip(nt, 0.0, Nt_c)

    p = n + nt

    R_rad = krad_c * (n * p - ni2_val)
    R_srh = (n * p - ni2_val) / ((n + n1) * taup_c + (p + p1) * taun_c)
    R_tot = R_rad + R_srh

    plqy_sim = R_rad / R_tot
    qfls_sim = kT * np.log(n * p / ni2_val)
    return qfls_sim, plqy_sim, R_tot


# ── N-trap numeric solver (fixed-point iteration) ─────────────────────────────

def _sspl_numeric(trap_params, krad, n_sw, NC, NV, Eg, kT, ni2_val,
                  max_iter=60, rtol=1e-8):
    """
    trap_params: list of dicts  {Et:(M,), Nt:(M,), taun:(M,), taup:(M,)}
    krad: (M,)
    n_sw: (N_N,)
    Returns qfls_sim (M, N_N), plqy_sim (M, N_N).
    """
    n = n_sw[None, :]   # (1, N_N)
    p = n.copy() * np.ones((len(krad), 1))   # (M, N_N)  initial guess

    n1_list = []
    p1_list = []
    for t in trap_params:
        Et_c = t["Et"][:, None]
        n1_list.append(NC * np.exp((Et_c - Eg) / kT))
        p1_list.append(NV * np.exp(-Et_c / kT))

    for _ in range(max_iter):
        p_old   = p
        nt_sum  = np.zeros_like(p)
        for t, n1, p1 in zip(trap_params, n1_list, p1_list):
            Nt_c   = t["Nt"][:, None]
            taun_c = t["taun"][:, None]
            taup_c = t["taup"][:, None]
            denom = taup_c * (n + n1) + taun_c * (p + p1)
            denom = np.where(denom > 0, denom, 1e-300)
            nt_k  = np.clip(Nt_c * (taup_c * n + taun_c * p1) / denom, 0.0, Nt_c)
            nt_sum += nt_k
        p = n + nt_sum
        if np.max(np.abs(p - p_old) / (p_old + 1e-300)) < rtol:
            break

    krad_c = krad[:, None]
    R_rad  = krad_c * (n * p - ni2_val)
    R_srh  = np.zeros_like(p)
    for t, n1, p1 in zip(trap_params, n1_list, p1_list):
        taun_c = t["taun"][:, None];  taup_c = t["taup"][:, None]
        R_srh += (n * p -ni2_val) / ((n + n1) * taup_c + (p + p1) * taun_c)

    R_tot    = R_rad + R_srh
    plqy_sim = R_rad / R_tot
    qfls_sim = kT * np.log(n * p / ni2_val)
    return qfls_sim, plqy_sim, R_tot


# ── MSE vs experimental SSPL data ─────────────────────────────────────────────

def _sspl_error(qfls_sim, plqy_sim, qfls_exp, log10_plqy_exp):
    """
    qfls_sim, plqy_sim: (M, N_N)
    Returns err_mse: (M,)
    """
    n_n = qfls_sim.shape[1]

    sort_idx     = np.argsort(qfls_sim[0])
    qfls_sorted  = qfls_sim[:, sort_idx]
    log10_p_sim  = np.log10(np.maximum(plqy_sim[:, sort_idx], 1e-300))

    idx    = np.searchsorted(qfls_sorted[0], qfls_exp) - 1
    in_rng = (idx >= 0) & (idx < n_n - 1)
    idx_s  = np.clip(idx, 0, n_n - 2)

    x0 = qfls_sorted[:, idx_s];  x1 = qfls_sorted[:, idx_s + 1]
    dx = np.where(np.abs(x1 - x0) > 1e-30, x1 - x0, 1e-30)
    w  = np.clip((qfls_exp - x0) / dx, 0.0, 1.0)

    y0 = log10_p_sim[:, idx_s];  y1 = log10_p_sim[:, idx_s + 1]
    log10_fit = y0 + w * (y1 - y0)
    log10_fit[:, ~in_rng] = np.nan

    res2 = (log10_fit - log10_plqy_exp) ** 2
    with np.errstate(all="ignore"):
        return np.nanmean(res2, axis=1)


# ── G-based error helpers ─────────────────────────────────────────────────────

def _interp_on_logG(G_sim, y_sim, G_exp):
    """
    Interpolate y_sim (M, N_N) at experimental G_exp points using a log10(G)
    x-axis.  Returns y_fit (M, N_exp) with NaN where G_exp is out of range.
    The sort order is determined from row 0 (G is monotonically increasing
    with n_sweep for all physically reasonable parameter sets).
    """
    logG_sim = np.log10(np.maximum(G_sim, 1e-300))
    sort_idx = np.argsort(logG_sim[0])
    logG_sorted = logG_sim[:, sort_idx]
    y_sorted    = y_sim[:, sort_idx]

    logG_exp = np.log10(np.maximum(G_exp, 1e-300))
    n_n = G_sim.shape[1]

    idx   = np.searchsorted(logG_sorted[0], logG_exp) - 1
    in_rng = (idx >= 0) & (idx < n_n - 1)
    idx_s  = np.clip(idx, 0, n_n - 2)

    x0 = logG_sorted[:, idx_s];  x1 = logG_sorted[:, idx_s + 1]
    dx = np.where(np.abs(x1 - x0) > 1e-30, x1 - x0, 1e-30)
    w  = np.clip((logG_exp - x0) / dx, 0.0, 1.0)

    y0 = y_sorted[:, idx_s];  y1 = y_sorted[:, idx_s + 1]
    y_fit = y0 + w * (y1 - y0)
    y_fit[:, ~in_rng] = np.nan
    return y_fit


def _sspl_error_plqy_vs_G(G_sim, plqy_sim, G_exp, log10_plqy_exp):
    """
    MSE in log10(PLQY) space, interpolated on the log10(G) axis.
    G_sim, plqy_sim: (M, N_N); G_exp, log10_plqy_exp: (N_exp,)
    Returns err_mse (M,).
    """
    log10_plqy_fit = np.log10(np.maximum(
        _interp_on_logG(G_sim, plqy_sim, G_exp), 1e-300))
    res2 = (log10_plqy_fit - log10_plqy_exp) ** 2
    with np.errstate(all="ignore"):
        return np.nanmean(res2, axis=1)


def _sspl_error_qfls_vs_G(G_sim, qfls_sim, G_exp, qfls_exp):
    """
    MSE in QFLS (eV, linear) space, interpolated on the log10(G) axis.
    G_sim, qfls_sim: (M, N_N); G_exp, qfls_exp: (N_exp,) in eV.
    Returns err_mse (M,) in eV^2.
    """
    res2 = (_interp_on_logG(G_sim, qfls_sim, G_exp) - qfls_exp) ** 2
    with np.errstate(all="ignore"):
        return np.nanmean(res2, axis=1)


# ── AM1.5G 1-sun generation rate ─────────────────────────────────────────────

def _compute_G_per_sun(Eg, d_cm, am15g_path=None):
    """Return 1-sun generation rate [cm⁻³ s⁻¹] for a film of thickness d_cm."""
    if am15g_path is None:
        am15g_path = os.path.join(_SCRIPT_DIR, "..", "data", "AM15G.dat")
    am15g = pd.read_csv(am15g_path, sep=r'\s+', header=None,
                        names=['E', 'sunspectrum'], engine='python')
    E = np.linspace(am15g['E'].min(), am15g['E'].max(), 1000)
    phi = PchipInterpolator(am15g['E'], am15g['sunspectrum'], extrapolate=False)(E)
    mask = E >= Eg
    return _trapz(phi[mask], E[mask]) / d_cm


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--grid",   default=None)
    ap.add_argument("--out",    default=None)
    args = ap.parse_args()

    cfg_path = args.config if os.path.isabs(args.config) else \
               os.path.join(os.getcwd(), args.config)
    with open(cfg_path) as fh:
        cfg = json.load(fh)

    base    = os.path.dirname(cfg_path)
    out_dir = cfg["output_dir"]
    if not os.path.isabs(out_dir):
        out_dir = os.path.join(base, out_dir)
    os.makedirs(out_dir, exist_ok=True)

    grid_mode = cfg["grid"].get("mode", "full")
    default_grid = "grid_slices.csv" if grid_mode == "slice" else "grid.csv"
    grid_csv = args.grid or os.path.join(out_dir, default_grid)
    out_csv  = args.out  or os.path.join(out_dir, "errors_sspl.csv")

    # ── Physics constants ──────────────────────────────────────────────────────
    ph  = cfg["physics"]
    Eg  = ph["Eg_eV"];  NC = ph["NC_cm3"];  NV = ph["NV_cm3"]
    kT  = 8.617e-5 * ph["T_K"]
    ni2_val = _ni2(NC, NV, Eg, kT)
    num_traps = ph["num_traps"]

    sc  = cfg.get("sspl", {})
    n_sw = _n_sweep(sc.get("n_sweep_lo", 1e8), sc.get("n_sweep_hi", NV),
                    sc.get("n_sweep_n", 80))
    solver  = sc.get("solver", "analytic")
    compare = sc.get("compare", "plqy_vs_qfls")
    if num_traps > 1 and solver == "analytic":
        _p("  num_traps>1 → switching solver to 'numeric'")
        solver = "numeric"
    if compare not in ("plqy_vs_qfls", "plqy_vs_G", "qfls_vs_G",
                       "suns_vs_qfls", "suns_vs_plqy"):
        raise ValueError(f"sspl.compare must be 'plqy_vs_qfls', 'plqy_vs_G', "
                         f"'qfls_vs_G', 'suns_vs_qfls', or 'suns_vs_plqy', "
                         f"got '{compare}'")

    # ── Load experimental SSPL data ────────────────────────────────────────────
    sspl_path = cfg["data"]["sspl_csv"]
    if not os.path.isabs(sspl_path):
        sspl_path = os.path.join(base, sspl_path)
    if compare in ("suns_vs_qfls", "suns_vs_plqy"):
        # File has a units row as row 2 — skip it
        df_sspl   = pd.read_csv(sspl_path, sep='\t', header=0, skiprows=[1])
        suns_exp  = df_sspl["intensity"].values.astype(float)
        d_cm      = ph.get("thickness_nm", 450.0) * 1e-7
        G_per_sun = _compute_G_per_sun(Eg, d_cm) * sc.get("G_factor", 1.0)
        if compare == "suns_vs_qfls":
            qfls_exp = df_sspl["qfls"].values.astype(float)
            mask     = np.isfinite(suns_exp) & (suns_exp > 0) & np.isfinite(qfls_exp)
            suns_exp = suns_exp[mask];  qfls_exp = qfls_exp[mask]
            G_exp    = suns_exp * G_per_sun
        else:  # suns_vs_plqy
            plqy_exp = df_sspl["plqy"].values.astype(float)
            mask     = np.isfinite(suns_exp) & (suns_exp > 0) \
                       & np.isfinite(plqy_exp) & (plqy_exp > 0)
            suns_exp = suns_exp[mask];  plqy_exp = plqy_exp[mask]
            G_exp    = suns_exp * G_per_sun
            log10_plqy_exp = np.log10(plqy_exp)
        _p(f"SSPL data : {len(G_exp)} valid pts  compare={compare}  solver={solver}")
        _p(f"  thickness={d_cm*1e7:.0f} nm  G_per_sun={G_per_sun:.4e} cm⁻³ s⁻¹")
    else:
        df_sspl = pd.read_csv(sspl_path)

    if compare == "plqy_vs_qfls":
        qfls_exp = df_sspl["qfls_eV"].values
        plqy_exp = df_sspl["plqy"].values
        mask = np.isfinite(qfls_exp) & np.isfinite(plqy_exp) & (plqy_exp > 0)
        qfls_exp = qfls_exp[mask];  plqy_exp = plqy_exp[mask]
        log10_plqy_exp = np.log10(plqy_exp)
        _p(f"SSPL data : {len(qfls_exp)} valid pts  compare={compare}  solver={solver}")
    elif compare == "plqy_vs_G":
        G_exp    = df_sspl["G_cm3s"].values
        plqy_exp = df_sspl["plqy"].values
        mask = np.isfinite(G_exp) & (G_exp > 0) & np.isfinite(plqy_exp) & (plqy_exp > 0)
        G_exp    = G_exp[mask];  plqy_exp = plqy_exp[mask]
        log10_plqy_exp = np.log10(plqy_exp)
        _p(f"SSPL data : {len(G_exp)} valid pts  compare={compare}  solver={solver}")
    elif compare == "qfls_vs_G":
        G_exp    = df_sspl["G_cm3s"].values
        qfls_exp = df_sspl["qfls_eV"].values
        mask = np.isfinite(G_exp) & (G_exp > 0) & np.isfinite(qfls_exp)
        G_exp    = G_exp[mask];  qfls_exp = qfls_exp[mask]
        _p(f"SSPL data : {len(G_exp)} valid pts  compare={compare}  solver={solver}")

    # ── Process chunks ─────────────────────────────────────────────────────────
    chunk_size = int(cfg["grid"].get("chunk_size", 100_000))
    _p(f"Grid CSV  : {grid_csv}")
    _p(f"Output    : {out_csv}")
    t0 = time.time();  rows_done = 0;  first = True

    for chunk in pd.read_csv(grid_csv, chunksize=chunk_size):
        krad = chunk["krad_cm3s"].values.astype(np.float64)

        if solver == "analytic":
            Et   = chunk["Et_1_eV"].values.astype(np.float64)
            Nt   = chunk["Nt_1_cm-3"].values.astype(np.float64)
            taun = chunk["tau_n_1_s"].values.astype(np.float64)
            taup = chunk["tau_p_1_s"].values.astype(np.float64)
            qfls_sim, plqy_sim, G_sim = _sspl_analytic(
                Et, Nt, taun, taup, krad, n_sw, NC, NV, Eg, kT, ni2_val)
        else:
            trap_params = []
            for k in range(1, num_traps + 1):
                trap_params.append({
                    "Et":   chunk[f"Et_{k}_eV"].values.astype(np.float64),
                    "Nt":   chunk[f"Nt_{k}_cm-3"].values.astype(np.float64),
                    "taun": chunk[f"tau_n_{k}_s"].values.astype(np.float64),
                    "taup": chunk[f"tau_p_{k}_s"].values.astype(np.float64),
                })
            qfls_sim, plqy_sim, G_sim = _sspl_numeric(
                trap_params, krad, n_sw, NC, NV, Eg, kT, ni2_val)

        if compare == "plqy_vs_qfls":
            err_mse = _sspl_error(qfls_sim, plqy_sim, qfls_exp, log10_plqy_exp)
        elif compare in ("plqy_vs_G", "suns_vs_plqy"):
            err_mse = _sspl_error_plqy_vs_G(G_sim, plqy_sim, G_exp, log10_plqy_exp)
        elif compare == "qfls_vs_G":
            err_mse = _sspl_error_qfls_vs_G(G_sim, qfls_sim, G_exp, qfls_exp)
        else:  # suns_vs_qfls — G_exp already converted from suns
            err_mse = _sspl_error_qfls_vs_G(G_sim, qfls_sim, G_exp, qfls_exp)

        chunk["error_sspl_mse"] = err_mse
        chunk.to_csv(out_csv, mode="w" if first else "a",
                     header=first, index=False, float_format="%.6e")
        first = False;  rows_done += len(chunk)
        _p(f"  {rows_done:>12,} rows  ({time.time()-t0:.1f}s)")

    _p(f"\nDone. {rows_done:,} rows in {time.time()-t0:.1f}s → {out_csv}")


if __name__ == "__main__":
    main()
