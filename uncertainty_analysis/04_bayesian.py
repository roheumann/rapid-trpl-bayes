"""
04_bayesian.py  —  Bayesian posterior inference
================================================
Reads error CSV(s) produced by 02/03_errors scripts, computes the posterior
P(Θ|data) ∝ L(data|Θ), and writes 2-D and 1-D marginal distributions plus
a corner plot.

Modes
-----
  trpl    Uses error_trpl_mse only
  sspl    Uses error_sspl_mse only
  joint   Combines both independently (log-likelihood is the sum)

Grid modes
----------
  full    Streams through the large grid CSV, accumulates 2-D/1-D marginals
          into pre-allocated numpy arrays, then plots a corner plot.
  slice   Loads slice CSV entirely (small), plots per-pair 2-D heatmaps and
          1-D profiles around the best-fit point.

Output (in <output_dir>/bayesian_<mode>/)
  panels_2d/*.csv        2-D marginal (or slice) tables
  panels_1d/*.csv        1-D marginal (or profile) tables
  credible_intervals.csv 95 % HDR per parameter
  cornerplot.png         corner plot figure
  metadata.json

Usage
-----
  # full grid, joint posterior
  python 04_bayesian.py --config config.json --mode joint

  # slice grid, TRPL only
  python 04_bayesian.py --config config.json --mode trpl --grid-mode slice
  python 04_bayesian.py --config config.json --mode trpl --grid-mode slice \\
      --trpl-errors errors_trpl_slices.csv
"""

import argparse, json, os, sys, time
import gc
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.special import logsumexp

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def _p(msg): print(msg, flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _param_keys(num_traps):
    keys = []
    for k in range(1, num_traps + 1):
        keys.extend([f"Et_{k}", f"Nt_{k}", f"tau_n_{k}", f"tau_p_{k}"])
    keys.append("krad")
    return keys


def _csv_col(key):
    if key.startswith("Et_"):  return key + "_eV"
    if key.startswith("Nt_"):  return key + "_cm-3"
    if key.startswith("tau_"): return key + "_s"
    if key == "krad":           return "krad_cm3s"
    return key


def _is_log(key):
    return not key.startswith("Et_")


def _log_col(phys_col, is_log):
    return f"log10_{phys_col.replace('_cm-3','').replace('_eV','').replace('_s','').replace('krad_cm3s','krad').replace('-','')}" \
           if is_log else phys_col


def _safe(name):
    return (name.replace("_cm-3","").replace("_eV","").replace("_s","")
                .replace("krad_cm3s","krad").replace("-",""))


def _log10_safe(x):
    """
    log10(x) with NaN for non-positive values — no artificial floor.

    Avoids the 1e-300 clip trick: zero-probability cells get NaN (blank in
    plots / empty in CSV) rather than a fake −300 floor.
    Uses a double np.where so that log10 is never evaluated on non-positive
    inputs (which would emit a RuntimeWarning).
    """
    a   = np.asarray(x, dtype=np.float64)
    pos = a > 0
    return np.where(pos, np.log10(np.where(pos, a, 1.0)), np.nan)


def _log_likelihood(error_arr, sigma):
    """Gaussian log-likelihood from MSE:  -MSE/(2σ²) - log(σ√2π)."""
    return (-error_arr.astype(np.float64) / (2.0 * sigma**2)
            - np.log(sigma * np.sqrt(2.0 * np.pi)))


def _find_mse_min(csv_path, col, chunk_size=500_000):
    """Scan CSV for the minimum finite value of `col`. Returns (min_val, n_rows)."""
    mse_min = np.inf
    n_rows  = 0
    for chunk in pd.read_csv(csv_path, usecols=[col], chunksize=chunk_size,
                              dtype={col: np.float32}):
        v = chunk[col].values
        finite = v[np.isfinite(v)]
        if len(finite):
            mse_min = min(mse_min, float(finite.min()))
        n_rows += len(chunk)
    return float(mse_min), n_rows


def _sigma_eff(sigma_nn, mse_min):
    """Combine NN base sigma with grid MSE_min in quadrature."""
    return float(np.sqrt(sigma_nn**2 + mse_min))


def _write_summary(out_dir, cfg, mode, grid_mode,
                   sigma_info, log_Z, ci_rows):
    """
    Write a comprehensive summary of the Bayesian run:
      summary.json  — all metadata, sigma values, evidence, CIs
      summary.csv   — flat table: one row = sigma/evidence summary +
                      one row per parameter with CI bounds
    """
    run_name  = cfg.get("run_name", "")  if cfg else ""
    num_traps = cfg["physics"]["num_traps"] if cfg else None

    # ── JSON ────────────────────────────────────────────────────────────────
    summary = {
        "run_name":    run_name,
        "mode":        mode,
        "grid_mode":   grid_mode,
        "num_traps":   num_traps,
        "sigma": {
            "sigma_nn_trpl":      sigma_info.get("sigma_nn_trpl"),
            "sigma_nn_sspl":      sigma_info.get("sigma_nn_sspl"),
            "MSE_grid_min_trpl":  sigma_info.get("MSE_grid_min_trpl"),
            "MSE_grid_min_sspl":  sigma_info.get("MSE_grid_min_sspl"),
            "sigma_eff_trpl":     sigma_info.get("sigma_eff_trpl"),
            "sigma_eff_sspl":     sigma_info.get("sigma_eff_sspl"),
        },
        "evidence": {
            "log_Z":       float(log_Z) if log_Z is not None else None,
            "log10_Z":     float(log_Z * np.log10(np.e)) if log_Z is not None else None,
        },
        "credible_intervals": ci_rows,
    }
    json_path = os.path.join(out_dir, "summary.json")
    with open(json_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    _p(f"  Summary JSON → {json_path}")

    # ── Flat CSV ─────────────────────────────────────────────────────────────
    rows = []
    # Sigma / evidence row
    rows.append({
        "section":           "sigma_evidence",
        "parameter":         "",
        "sigma_nn_trpl":     sigma_info.get("sigma_nn_trpl"),
        "sigma_nn_sspl":     sigma_info.get("sigma_nn_sspl"),
        "MSE_grid_min_trpl": sigma_info.get("MSE_grid_min_trpl"),
        "MSE_grid_min_sspl": sigma_info.get("MSE_grid_min_sspl"),
        "sigma_eff_trpl":    sigma_info.get("sigma_eff_trpl"),
        "sigma_eff_sspl":    sigma_info.get("sigma_eff_sspl"),
        "log10_Z":           float(log_Z * np.log10(np.e)) if log_Z is not None else None,
        "center_phys":  None, "mode_phys":    None,
        "ci1s_lo_phys": None, "ci1s_hi_phys": None,
        "ci2s_lo_phys": None, "ci2s_hi_phys": None,
        "true_value":   None, "true_in_ci1s": None, "true_in_ci2s": None,
    })
    # Per-parameter rows
    for ci in ci_rows:
        rows.append({
            "section":           "credible_interval",
            "parameter":         ci["parameter"],
            "sigma_nn_trpl":     None,
            "sigma_nn_sspl":     None,
            "MSE_grid_min_trpl": None,
            "MSE_grid_min_sspl": None,
            "sigma_eff_trpl":    None,
            "sigma_eff_sspl":    None,
            "log10_Z":           None,
            "center_phys":  ci.get("center_phys"),
            "mode_phys":    ci.get("mode_phys"),
            "ci1s_lo_phys": ci.get("ci1s_lo_phys"),
            "ci1s_hi_phys": ci.get("ci1s_hi_phys"),
            "ci2s_lo_phys": ci.get("ci2s_lo_phys"),
            "ci2s_hi_phys": ci.get("ci2s_hi_phys"),
            "true_value":   ci.get("true_value"),
            "true_in_ci1s": ci.get("true_in_ci1s"),
            "true_in_ci2s": ci.get("true_in_ci2s"),
        })
    csv_path = os.path.join(out_dir, "summary.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False, float_format="%.6e")
    _p(f"  Summary CSV  → {csv_path}")


def _load_cfg(cfg_path):
    with open(cfg_path) as fh:
        return json.load(fh)


# ─────────────────────────────────────────────────────────────────────────────
# 1-D credible interval (95 % HDR)
# ─────────────────────────────────────────────────────────────────────────────

def _ci_1sigma(prob_mass, theta_axis, theta_best=None, fraction=0.6827):
    """
    Smallest symmetric interval around theta_best (or the mode) that encloses
    at least `fraction` of mass.

    Expands by k grid steps on both sides of best_idx simultaneously:
    [best_idx-k, best_idx+k], clamped to grid boundaries.  Stops at the first k
    where the enclosed mass >= fraction.

    sigma_minus = best_fit - lo, sigma_plus = hi - best_fit (axis units).
    On a uniform grid both are approximately equal (truly symmetric error bar).

    Returns (lo, hi, mode, sigma_minus, sigma_plus, in_ci_mask).
    """
    prob_mass = prob_mass / prob_mass.sum()          # safe-normalise
    mode      = float(theta_axis[np.argmax(prob_mass)])
    center    = theta_best if theta_best is not None else mode

    n        = len(theta_axis)
    best_idx = int(np.argmin(np.abs(theta_axis - center)))
    prefix   = np.concatenate([[0.0], np.cumsum(prob_mass)])   # prefix[i] = sum[0:i]

    best_l, best_r = best_idx, best_idx          # fallback: single cell
    for k in range(n):
        l    = max(0, best_idx - k)
        r    = min(n - 1, best_idx + k)
        mass = prefix[r + 1] - prefix[l]
        if mass >= fraction:
            best_l, best_r = l, r
            break

    in_ci    = np.zeros(n, dtype=bool)
    in_ci[best_l:best_r + 1] = True
    # Symmetric interval: half-width = max distance from center to either grid boundary.
    # This anchors lo/hi on center, not on the nearest grid point, so sigma_minus == sigma_plus.
    delta       = max(center - float(theta_axis[best_l]),
                      float(theta_axis[best_r]) - center)
    lo          = center - delta
    hi          = center + delta
    sigma_minus = delta
    sigma_plus  = delta
    return lo, hi, mode, sigma_minus, sigma_plus, in_ci


# 2-D Gaussian-equivalent probability fractions: P = 1 - exp(-n²/2)
_SIGMA_LEVELS = [("1σ", 1 - np.exp(-0.5)), ("2σ", 1 - np.exp(-2.0)), ("3σ", 1 - np.exp(-4.5))]


def _mass_contour_levels(mass_flat, p_bf=None, mass_2d=None, bf_ij=None):
    """
    2-D contour levels at 1σ / 2σ / 3σ using the pure HDR approach:
    sort cells by probability (descending), accumulate until the desired
    fraction is reached, and use that probability as the contour threshold.

    Spatial symmetric expansion (bf_ij) cannot be used for probability
    contours: block.min() is dominated by the low-probability best-fit cell
    whenever the best-fit is in the tail, collapsing the threshold to ~0 and
    producing a contour that spans the entire panel.  The best-fit is already
    shown as a separate marker on the plot.

    Unused parameters (p_bf, mass_2d, bf_ij) are accepted for API compatibility.
    """
    pos = mass_flat[np.isfinite(mass_flat) & (mass_flat > 0)]
    if len(pos) < 2:
        return None, None
    sort_idx = np.argsort(pos)[::-1]
    cumsum   = np.cumsum(pos[sort_idx])
    cumsum  /= cumsum[-1]
    levels, labels = [], {}
    for lbl, frac in _SIGMA_LEVELS:
        idx = int(np.searchsorted(cumsum, frac))
        idx = min(idx, len(pos) - 1)
        lv  = float(pos[sort_idx[idx]])
        if lv > 0 and lv not in labels:
            levels.append(lv)
            labels[lv] = lbl
    if not levels:
        return None, None
    return np.array(sorted(levels)), labels


def _panel_p_bf(agg, ca, ma, cb, mb, best_fit_params):
    """Return the probability at the best-fit grid point in a 2-D panel."""
    if not best_fit_params:
        return None
    pa, pb = ma["phys"], mb["phys"]
    if pa not in best_fit_params or pb not in best_fit_params:
        return None
    bfv_a = float(best_fit_params[pa])
    bfv_b = float(best_fit_params[pb])
    bf_ax_a = np.log10(bfv_a) if ma["log"] else bfv_a
    bf_ax_b = np.log10(bfv_b) if mb["log"] else bfv_b
    x_vals = np.sort(agg[ca].unique())
    y_vals = np.sort(agg[cb].unique())
    x_near = x_vals[int(np.argmin(np.abs(x_vals - bf_ax_a)))]
    y_near = y_vals[int(np.argmin(np.abs(y_vals - bf_ax_b)))]
    sub = agg[(np.abs(agg[ca] - x_near) < 1e-9) & (np.abs(agg[cb] - y_near) < 1e-9)]
    return float(sub["probability"].iloc[0]) if len(sub) > 0 else None


# ─────────────────────────────────────────────────────────────────────────────
# Parameter metadata
# ─────────────────────────────────────────────────────────────────────────────

def _build_param_meta(num_traps):
    """Return ordered list of dicts with axis/label/log info."""
    meta = []
    for k in range(1, num_traps + 1):
        meta += [
            {"key": f"Et_{k}",    "phys": f"Et_{k}_eV",    "log": False,
             "label": rf"$E_t^{{{k}}}$ (eV)"},
            {"key": f"Nt_{k}",    "phys": f"Nt_{k}_cm-3",  "log": True,
             "label": rf"$\log_{{10}}(N_t^{{{k}}}\,/\,\mathrm{{cm}}^{{-3}})$"},
            {"key": f"tau_n_{k}", "phys": f"tau_n_{k}_s",  "log": True,
             "label": rf"$\log_{{10}}(\tau_n^{{{k}}}\,/\,\mathrm{{s}})$"},
            {"key": f"tau_p_{k}", "phys": f"tau_p_{k}_s",  "log": True,
             "label": rf"$\log_{{10}}(\tau_p^{{{k}}}\,/\,\mathrm{{s}})$"},
        ]
    meta.append({"key": "krad", "phys": "krad_cm3s", "log": True,
                 "label": r"$\log_{10}(k_\mathrm{rad}\,/\,\mathrm{cm}^3\mathrm{s}^{-1})$"})
    for m in meta:
        m["axis_col"] = (_log_col(m["phys"], m["log"])
                          if m["log"] else m["phys"])
    return meta


# ─────────────────────────────────────────────────────────────────────────────
# FULL-GRID mode
# ─────────────────────────────────────────────────────────────────────────────

def _run_full(cfg, mode, trpl_csv, sspl_csv, out_dir, true_params, param_meta,
             best_fit_params=None):
    """Stream full grid, accumulate marginals, plot corner plot."""

    sigma_nn_trpl = float(cfg["bayesian"]["sigma_trpl"])
    sigma_nn_sspl = float(cfg["bayesian"]["sigma_sspl"])
    chunk_size    = int(cfg["bayesian"].get("chunk_size", 1_000_000))
    phys_cols     = [m["phys"] for m in param_meta]

    _p(f"\nMode=full  posterior={mode}  chunk={chunk_size:,}")

    # ── Pre-scan: find MSE_min for dynamic sigma_eff computation ─────────────
    _p("Pre-scan: finding MSE grid minimum …")
    mse_min_trpl = mse_min_sspl = None
    if mode in ("trpl", "joint"):
        mse_min_trpl, n_rows = _find_mse_min(trpl_csv, "error_trpl_mse", chunk_size)
        _p(f"  MSE_min_trpl = {mse_min_trpl:.6e}  ({n_rows:,} rows)")
    if mode in ("sspl", "joint"):
        mse_min_sspl, n_rows = _find_mse_min(sspl_csv, "error_sspl_mse", chunk_size)
        _p(f"  MSE_min_sspl = {mse_min_sspl:.6e}  ({n_rows:,} rows)")

    sigma_eff_trpl = _sigma_eff(sigma_nn_trpl, mse_min_trpl or 0.0)
    sigma_eff_sspl = _sigma_eff(sigma_nn_sspl, mse_min_sspl or 0.0)
    _p(f"  sigma_eff_trpl = {sigma_eff_trpl:.6e}  "
       f"(σ_NN={sigma_nn_trpl:.4e}, MSE_min={mse_min_trpl})")
    _p(f"  sigma_eff_sspl = {sigma_eff_sspl:.6e}  "
       f"(σ_NN={sigma_nn_sspl:.4e}, MSE_min={mse_min_sspl})")

    # ── Main pass: accumulate log-likelihoods using sigma_eff ─────────────────
    _p("Main pass: loading log-likelihoods …")
    param_list = [];  logL_list = []

    if mode in ("trpl", "joint"):
        trpl_iter = pd.read_csv(trpl_csv, chunksize=chunk_size,
                                dtype={"error_trpl_mse": np.float32})
    if mode in ("sspl", "joint"):
        sspl_iter = pd.read_csv(sspl_csv, chunksize=chunk_size,
                                dtype={"error_sspl_mse": np.float32})

    n_rows = 0
    for _ in range(10**9):
        try:
            if mode == "trpl":
                chunk = next(iter(trpl_iter))
                logL  = _log_likelihood(chunk["error_trpl_mse"].values, sigma_eff_trpl)
            elif mode == "sspl":
                chunk = next(iter(sspl_iter))
                logL  = _log_likelihood(chunk["error_sspl_mse"].values, sigma_eff_sspl)
            else:   # joint
                tc    = next(iter(trpl_iter))
                sc    = next(iter(sspl_iter))
                logL  = (_log_likelihood(tc["error_trpl_mse"].values, sigma_eff_trpl)
                        + _log_likelihood(sc["error_sspl_mse"].values, sigma_eff_sspl))
                chunk = tc
        except StopIteration:
            break

        logL = np.where(np.isfinite(logL), logL, -np.inf)
        param_list.append(chunk[phys_cols].values.astype(np.float32))
        logL_list.append(logL.astype(np.float64))
        n_rows += len(chunk)
        if n_rows % (chunk_size * 5) == 0:
            _p(f"  {n_rows:,} rows")

    params_arr = np.concatenate(param_list, axis=0)
    logL_arr   = np.concatenate(logL_list)
    del param_list, logL_list;  gc.collect()
    _p(f"  Total: {len(params_arr):,} rows  logL max={logL_arr.max():.2f}")

    # ── Normalise ─────────────────────────────────────────────────────────────
    log_Z  = logsumexp(logL_arr)
    L_norm = np.exp(logL_arr - log_Z).astype(np.float32)
    del logL_arr;  gc.collect()

    df = pd.DataFrame(params_arr, columns=phys_cols)
    del params_arr;  gc.collect()
    for m in param_meta:
        if m["log"]:
            df[m["axis_col"]] = np.log10(df[m["phys"]]).astype(np.float32)
    df["_L"] = L_norm
    del L_norm;  gc.collect()

    sigma_info = {
        "sigma_nn_trpl":     sigma_nn_trpl,
        "sigma_nn_sspl":     sigma_nn_sspl,
        "MSE_grid_min_trpl": mse_min_trpl,
        "MSE_grid_min_sspl": mse_min_sspl,
        "sigma_eff_trpl":    sigma_eff_trpl,
        "sigma_eff_sspl":    sigma_eff_sspl,
    }
    _write_panels_and_plot(df, param_meta, out_dir, mode, true_params, log_Z,
                           grid_mode="full", sigma_info=sigma_info, cfg=cfg,
                           best_fit_params=best_fit_params)


# ─────────────────────────────────────────────────────────────────────────────
# SLICE mode
# ─────────────────────────────────────────────────────────────────────────────

def _marginalize_1d_from_slices(panels_2d, param_meta):
    """
    For each parameter, collect all 2-D slice panels that contain it and
    marginalise out the complementary axis by an exact discrete sum: each
    panel's "probability" column is already a normalised pmf over its 2-D
    grid (Σ = 1 by construction, see log_Z_slice in _run_slice), so summing
    over the companion axis gives the exact marginal mass — no trapezoidal
    integration (and no dependence on the companion axis's grid spacing)
    is needed. The per-panel marginal profiles are then combined via a
    plain average: since every panel is already normalised to 1, there is
    no genuine per-panel "evidence" left to weight by (a previous version
    weighted by each panel's raw pre-normalisation trapezoid mass, but that
    mass is dominated by the companion grid step size, not by the amount of
    posterior mass the panel carries, so it doesn't measure evidence).

    Returns dict: phys_col → DataFrame(axis_col, phys_col, probability,
                                       probability_density)
    where probability_density = probability / grid_step_on_axis_col,
    so that sum(probability_density * delta) = 1 (integrates to 1 over the axis).
    """
    out = {}
    for m in param_meta:
        phys = m["phys"]
        ax   = m["axis_col"]
        contribs  = []
        axis_vals = None

        for (px, py), panel in panels_2d.items():
            if phys not in (px, py):
                continue

            prof = panel.groupby(ax, sort=True)["probability"].sum()
            vals = prof.index.to_numpy(float)
            prof = prof.to_numpy(float)

            if axis_vals is None:
                axis_vals = vals

            total = prof.sum()
            contribs.append(prof / total if total > 0 else prof)

        if not contribs:
            continue

        avg = np.mean(np.vstack(contribs), axis=0)

        delta = float(np.median(np.diff(axis_vals))) if len(axis_vals) > 1 else 1.0
        df = pd.DataFrame({ax: axis_vals, "probability": avg})
        df["probability_density"] = avg / delta
        df["log10_P"] = _log10_safe(avg)                                      # log10 of mass
        df["log10_probability_density"] = df["log10_P"].values - np.log10(delta)  # subtract log10(Δx)
        df[phys] = 10 ** df[ax].values if m["log"] else df[ax].values
        out[phys] = df

    return out


def _run_slice(cfg, mode, trpl_csv, sspl_csv, trpl_1d, sspl_1d,
               out_dir, true_params, param_meta, best_fit_params=None):
    """Load slice CSVs, compute per-pair likelihoods, plot."""

    sigma_nn_trpl = float(cfg["bayesian"]["sigma_trpl"])
    sigma_nn_sspl = float(cfg["bayesian"]["sigma_sspl"])
    phys_cols     = [m["phys"] for m in param_meta]

    _p(f"\nMode=slice  posterior={mode}")

    # ── Find MSE_min across slice CSVs for dynamic sigma_eff ─────────────────
    mse_min_trpl = mse_min_sspl = None
    if mode in ("trpl", "joint") and trpl_csv and os.path.isfile(trpl_csv):
        mse_min_trpl, _ = _find_mse_min(trpl_csv, "error_trpl_mse")
        _p(f"  MSE_min_trpl = {mse_min_trpl:.6e}")
    if mode in ("sspl", "joint") and sspl_csv and os.path.isfile(sspl_csv):
        mse_min_sspl, _ = _find_mse_min(sspl_csv, "error_sspl_mse")
        _p(f"  MSE_min_sspl = {mse_min_sspl:.6e}")

    sigma_eff_trpl = _sigma_eff(sigma_nn_trpl, mse_min_trpl or 0.0)
    sigma_eff_sspl = _sigma_eff(sigma_nn_sspl, mse_min_sspl or 0.0)
    _p(f"  sigma_eff_trpl = {sigma_eff_trpl:.6e}")
    _p(f"  sigma_eff_sspl = {sigma_eff_sspl:.6e}")

    sigma_info = {
        "sigma_nn_trpl":     sigma_nn_trpl,
        "sigma_nn_sspl":     sigma_nn_sspl,
        "MSE_grid_min_trpl": mse_min_trpl,
        "MSE_grid_min_sspl": mse_min_sspl,
        "sigma_eff_trpl":    sigma_eff_trpl,
        "sigma_eff_sspl":    sigma_eff_sspl,
    }

    def _load_logL(csv_trpl, csv_sspl):
        """Load error CSVs, compute combined log-likelihood using sigma_eff."""
        src = []
        if mode in ("trpl","joint") and csv_trpl and os.path.isfile(csv_trpl):
            src.append((csv_trpl, "error_trpl_mse", sigma_eff_trpl))
        if mode in ("sspl","joint") and csv_sspl and os.path.isfile(csv_sspl):
            src.append((csv_sspl, "error_sspl_mse", sigma_eff_sspl))
        if not src:
            raise FileNotFoundError("No valid error CSV found for slice mode.")
        df   = pd.read_csv(src[0][0])
        logL = np.zeros(len(df), dtype=np.float64)
        for path, ecol, sigma in src:
            tmp   = pd.read_csv(path)
            logL += _log_likelihood(tmp[ecol].values, sigma)
        logL = np.where(np.isfinite(logL), logL, -np.inf)
        df["_logL"] = logL
        return df

    # ── 2-D slices ─────────────────────────────────────────────────────────────
    df_slices = _load_logL(trpl_csv, sspl_csv)
    df_slices = df_slices[df_slices["slice_pair"].notna()].copy()

    for m in param_meta:
        if m["log"] and m["phys"] in df_slices.columns:
            df_slices[m["axis_col"]] = np.log10(df_slices[m["phys"]])

    # ── Global logL reference: optimizer best-fit grid point ─────────────────
    # For each slice panel find the grid point nearest to the optimizer best-fit
    # (the other parameters are already fixed at best-fit values by construction).
    # Using this single shared reference makes relative_prob = exp(logL - logL_ref)
    # directly comparable across all panels: the optimizer maps to 1.0 everywhere,
    # and grid points that beat the optimizer get relative_prob > 1.
    if best_fit_params:
        logL_refs = []
        for pair_label, grp_ref in df_slices.groupby("slice_pair"):
            cols_ref = pair_label.split("|")
            phys_x_r, phys_y_r = cols_ref[0], cols_ref[1]
            mx_r = next((m for m in param_meta if m["phys"] == phys_x_r), None)
            my_r = next((m for m in param_meta if m["phys"] == phys_y_r), None)
            if (mx_r is None or my_r is None
                    or phys_x_r not in best_fit_params
                    or phys_y_r not in best_fit_params):
                continue
            bfx_r = float(best_fit_params[phys_x_r])
            bfy_r = float(best_fit_params[phys_y_r])
            ax_r, ay_r = mx_r["axis_col"], my_r["axis_col"]
            bf_ax_r = np.log10(bfx_r) if mx_r["log"] else bfx_r
            bf_ay_r = np.log10(bfy_r) if my_r["log"] else bfy_r
            xv_r = np.sort(grp_ref[ax_r].unique())
            yv_r = np.sort(grp_ref[ay_r].unique())
            xn_r = float(xv_r[np.argmin(np.abs(xv_r - bf_ax_r))])
            yn_r = float(yv_r[np.argmin(np.abs(yv_r - bf_ay_r))])
            sub_r = grp_ref[(np.abs(grp_ref[ax_r] - xn_r) < 1e-9) &
                            (np.abs(grp_ref[ay_r] - yn_r) < 1e-9)]
            if len(sub_r):
                logL_refs.append(float(sub_r["_logL"].iloc[0]))
        logL_ref = float(max(logL_refs)) if logL_refs else float(df_slices["_logL"].max())
    else:
        logL_ref = float(df_slices["_logL"].max())
    _p(f"  logL reference (optimizer best-fit grid approx) = {logL_ref:.6f}")

    panels_2d_dir = os.path.join(out_dir, "panels_2d")
    os.makedirs(panels_2d_dir, exist_ok=True)

    panels_2d = {}
    for pair_label, grp in df_slices.groupby("slice_pair"):
        cols   = pair_label.split("|")
        phys_x, phys_y = cols[0], cols[1]
        mx = next(m for m in param_meta if m["phys"] == phys_x)
        my = next(m for m in param_meta if m["phys"] == phys_y)
        ax_col, ay_col = mx["axis_col"], my["axis_col"]

        log_Z_slice = logsumexp(grp["_logL"].values)
        grp = grp.copy()
        grp["log10_P"]    = (grp["_logL"].values - log_Z_slice) * np.log10(np.e)
        grp["probability"] = np.exp(grp["_logL"].values - log_Z_slice)

        # relative_prob: likelihood ratio to optimizer best-fit (global reference).
        # Values > 1 mean the grid point beats the optimizer.
        grp["relative_prob"] = np.exp(grp["_logL"].values - logL_ref)
        grp["log10_relative_prob"] = (grp["_logL"].values - logL_ref) * np.log10(np.e)

        dx = float(np.median(np.diff(np.sort(grp[ax_col].unique()))))
        dy = float(np.median(np.diff(np.sort(grp[ay_col].unique()))))
        grp["probability_density"] = grp["probability"].values / (dx * dy)
        # Stay in log-space: log10(P/(dx·dy)) = log10_P − log10(dx·dy).
        # This avoids NaNs from exp() underflow (probability can be 0 for
        # low-likelihood cells even though log10_P is finite).
        grp["log10_probability_density"] = grp["log10_P"].values - np.log10(dx * dy)

        fname = os.path.join(panels_2d_dir,
                             f"slice_{_safe(phys_x)}_vs_{_safe(phys_y)}.csv")
        grp[[phys_x, phys_y,
             "log10_relative_prob", "log10_probability_density"]].to_csv(
            fname, index=False, float_format="%.6e")
        panels_2d[(phys_x, phys_y)] = grp
        _p(f"  2D slice {_safe(phys_x)} vs {_safe(phys_y)}: {len(grp)} pts")

    # ── Marginalise 2-D slices → 1-D distributions ────────────────────────────
    panels_1d_dir = os.path.join(out_dir, "panels_1d")
    os.makedirs(panels_1d_dir, exist_ok=True)

    _p("Marginalising 2-D slices → 1-D …")
    panels_1d = _marginalize_1d_from_slices(panels_2d, param_meta)
    for m in param_meta:
        phys = m["phys"]
        if phys not in panels_1d:
            continue
        df_m = panels_1d[phys]
        fname = os.path.join(panels_1d_dir, f"margslice_{_safe(phys)}.csv")
        df_m[[m["axis_col"], phys, "probability", "probability_density",
              "log10_P", "log10_probability_density"]].to_csv(
            fname, index=False, float_format="%.6e")
        _p(f"  1-D marginal {_safe(phys)}: {len(df_m)} pts")

    # ── Explicit 1-D profile CSVs (overrides marginalised if provided) ────────
    if trpl_1d is not None or sspl_1d is not None:
        df_1d = _load_logL(trpl_1d, sspl_1d)
        df_1d = df_1d[df_1d["varied_param"].notna()].copy()

        for m in param_meta:
            if m["log"] and m["phys"] in df_1d.columns:
                df_1d[m["axis_col"]] = np.log10(df_1d[m["phys"]])

        for param_col, grp in df_1d.groupby("varied_param"):
            mx = next((m for m in param_meta if m["phys"] == param_col), None)
            if mx is None:
                continue
            ax_col  = mx["axis_col"]
            log_Z_p = logsumexp(grp["_logL"].values)
            grp     = grp.copy()
            grp["log10_P"]     = (grp["_logL"].values - log_Z_p) * np.log10(np.e)
            grp["probability"] = np.exp(grp["_logL"].values - log_Z_p)
            _ax_vals = np.sort(grp[ax_col].unique())
            _delta   = float(np.median(np.diff(_ax_vals))) if len(_ax_vals) > 1 else 1.0
            grp["probability_density"] = grp["probability"].values / _delta
            grp["log10_probability_density"] = grp["log10_P"].values - np.log10(_delta)
            fname = os.path.join(panels_1d_dir,
                                 f"profile_{_safe(param_col)}.csv")
            grp[[ax_col, param_col, "probability", "probability_density",
                 "log10_P", "log10_probability_density"]].to_csv(
                fname, index=False, float_format="%.6e")
            panels_1d[param_col] = grp
            _p(f"  1-D profile {_safe(param_col)}: {len(grp)} pts")

    # ── 1-σ / 2-σ credible intervals from 1-D panels ────────────────────────
    ci_rows = [];  hpd_rows = []
    for m in param_meta:
        p = m["phys"]
        if p not in panels_1d:
            continue

        agg    = panels_1d[p].copy()
        ax_col = m["axis_col"]
        bf_ax = None
        if best_fit_params and p in best_fit_params:
            bfv = float(best_fit_params[p])
            bf_ax = np.log10(bfv) if m["log"] else bfv
        lo_ax,  hi_ax,  mode_ax, sm_ax,  sp_ax,  in_ci1 = _ci_1sigma(
            agg["probability"].values, agg[ax_col].values, theta_best=bf_ax, fraction=0.6827)
        lo_ax2, hi_ax2, _,       sm_ax2, sp_ax2, in_ci2 = _ci_1sigma(
            agg["probability"].values, agg[ax_col].values, theta_best=bf_ax, fraction=0.9545)
        center_ax = bf_ax if bf_ax is not None else mode_ax
        agg["in_ci1s"] = in_ci1
        agg["in_ci2s"] = in_ci2
        panels_1d[p]   = agg

        lo_p     = 10**lo_ax     if m["log"] else lo_ax
        hi_p     = 10**hi_ax     if m["log"] else hi_ax
        lo2_p    = 10**lo_ax2    if m["log"] else lo_ax2
        hi2_p    = 10**hi_ax2    if m["log"] else hi_ax2
        mode_p   = 10**mode_ax   if m["log"] else mode_ax
        center_p = 10**center_ax if m["log"] else center_ax
        tv       = (true_params[p] if true_params and p in true_params else None) or None
        tv_ax    = (np.log10(tv) if m["log"] else tv) if tv is not None else None
        in_ci1_tv = None if tv_ax is None else bool(lo_ax  <= tv_ax <= hi_ax)
        in_ci2_tv = None if tv_ax is None else bool(lo_ax2 <= tv_ax <= hi_ax2)

        hpd_rows.append({
            "parameter":          p,
            "center_phys":        center_p,
            "mode_phys":          mode_p,
            "ci1s_lo_phys":       lo_p,    "ci1s_hi_phys":      hi_p,
            "ci1s_lo_axis":       lo_ax,   "ci1s_hi_axis":      hi_ax,
            "sigma1s_minus_axis": sm_ax,   "sigma1s_plus_axis": sp_ax,
            "ci2s_lo_phys":       lo2_p,   "ci2s_hi_phys":      hi2_p,
            "ci2s_lo_axis":       lo_ax2,  "ci2s_hi_axis":      hi_ax2,
            "sigma2s_minus_axis": sm_ax2,  "sigma2s_plus_axis": sp_ax2,
            "true_value":         tv,
            "true_in_ci1s":       in_ci1_tv,
            "true_in_ci2s":       in_ci2_tv,
        })
        ci_rows.append({
            "parameter":    p,
            "center_phys":  center_p,
            "mode_phys":    mode_p,
            "ci1s_lo_phys": lo_p,  "ci1s_hi_phys": hi_p,
            "ci2s_lo_phys": lo2_p, "ci2s_hi_phys": hi2_p,
            "true_value":   tv,
            "true_in_ci1s": in_ci1_tv,
            "true_in_ci2s": in_ci2_tv,
        })

    if hpd_rows:
        pd.DataFrame(hpd_rows).to_csv(
            os.path.join(out_dir, "hpd_intervals.csv"), index=False,
            float_format="%.6e")
        pd.DataFrame(ci_rows).to_csv(
            os.path.join(out_dir, "credible_intervals.csv"), index=False)
        _write_modes_csv(hpd_rows, best_fit_params, out_dir)

    _write_contour_csvs(panels_2d, param_meta, out_dir,
                        best_fit_params=best_fit_params)
    _write_summary(out_dir, cfg, mode, "slice", sigma_info, None, ci_rows)
    _tau0_c = _tau0_curves(cfg, (len(param_meta) - 1) // 4, panels_2d,
                            panels_1d, best_fit_params, out_dir) if (cfg and mode in ("trpl", "joint")) else {}
    _plot_slice_corner(panels_2d, panels_1d, param_meta,
                       true_params, out_dir, mode, cfg=cfg,
                       best_fit_params=best_fit_params, tau0_curves=_tau0_c)


# ─────────────────────────────────────────────────────────────────────────────
# Shared: marginals, 1-D CIs, panel export, corner plot (full-grid)
# ─────────────────────────────────────────────────────────────────────────────

def _write_panels_and_plot(df, param_meta, out_dir, mode, true_params,
                            log_Z, grid_mode, sigma_info=None, cfg=None,
                            best_fit_params=None):
    panels_2d_dir = os.path.join(out_dir, "panels_2d")
    panels_1d_dir = os.path.join(out_dir, "panels_1d")
    os.makedirs(panels_2d_dir, exist_ok=True)
    os.makedirs(panels_1d_dir, exist_ok=True)

    phys_cols = [m["phys"] for m in param_meta]
    pairs     = list(combinations(range(len(param_meta)), 2))

    # ── Grid step sizes for probability density ───────────────────────────────
    # Uniform log-prior + uniform log-space grid → prior×ΔV cancels; no
    # Jacobian needed for the posterior masses.  Dividing by the grid step size
    # on the axis column converts mass → density so the PDF integrates to 1.
    grid_delta = {}
    for m in param_meta:
        col_vals = np.sort(df[m["axis_col"]].unique())
        grid_delta[m["phys"]] = float(np.median(np.diff(col_vals))) if len(col_vals) > 1 else 1.0

    # ── 2-D marginals ─────────────────────────────────────────────────────────
    _p("Computing 2-D marginals …")
    panels_2d = {}
    for i, j in pairs:
        mi, mj   = param_meta[i], param_meta[j]
        ci, cj   = mi["axis_col"], mj["axis_col"]
        pi, pj   = mi["phys"], mj["phys"]
        agg = df.groupby([ci, cj], sort=True)["_L"].sum().reset_index()
        agg.rename(columns={"_L": "probability"}, inplace=True)
        agg["log10_P"] = _log10_safe(agg["probability"].values)
        agg[pi] = (10**agg[ci].values) if mi["log"] else agg[ci].values
        agg[pj] = (10**agg[cj].values) if mj["log"] else agg[cj].values
        agg["probability_density"] = agg["probability"].values / (grid_delta[pi] * grid_delta[pj])
        agg["log10_probability_density"] = agg["log10_P"].values - np.log10(grid_delta[pi] * grid_delta[pj])
        fname = os.path.join(panels_2d_dir,
                             f"marginal_{_safe(pi)}_vs_{_safe(pj)}.csv")
        agg[[ci, cj, pi, pj, "probability", "probability_density",
             "log10_P", "log10_probability_density"]].to_csv(
            fname, index=False, float_format="%.6e")
        panels_2d[(mi["phys"], mj["phys"])] = agg
        _p(f"  {_safe(pi)} vs {_safe(pj)}: {len(agg)} cells")

    # ── 1-D marginals + 1-σ / 2-σ credible intervals ─────────────────────────
    _p("Computing 1-D marginals …")
    panels_1d = {};  ci_rows = [];  hpd_rows = []
    for m in param_meta:
        c   = m["axis_col"];  p = m["phys"]
        agg = df.groupby(c, sort=True)["_L"].sum().reset_index()
        agg.rename(columns={"_L": "probability"}, inplace=True)
        agg["log10_P"] = _log10_safe(agg["probability"].values)
        agg[p] = (10**agg[c].values) if m["log"] else agg[c].values
        agg["probability_density"] = agg["probability"].values / grid_delta[p]
        agg["log10_probability_density"] = agg["log10_P"].values - np.log10(grid_delta[p])

        bf_ax = None
        if best_fit_params and p in best_fit_params:
            bfv = float(best_fit_params[p])
            bf_ax = np.log10(bfv) if m["log"] else bfv
        lo_ax,  hi_ax,  mode_ax, sm_ax,  sp_ax,  in_ci1 = _ci_1sigma(
            agg["probability"].values, agg[c].values, theta_best=bf_ax, fraction=0.6827)
        lo_ax2, hi_ax2, _,       sm_ax2, sp_ax2, in_ci2 = _ci_1sigma(
            agg["probability"].values, agg[c].values, theta_best=bf_ax, fraction=0.9545)
        center_ax = bf_ax if bf_ax is not None else mode_ax
        agg["in_ci1s"] = in_ci1
        agg["in_ci2s"] = in_ci2

        fname = os.path.join(panels_1d_dir, f"marginal_{_safe(p)}.csv")
        agg[[c, p, "probability", "probability_density",
             "log10_P", "log10_probability_density",
             "in_ci1s", "in_ci2s"]].to_csv(fname, index=False, float_format="%.6e")
        panels_1d[p] = agg

        lo_p      = 10**lo_ax      if m["log"] else lo_ax
        hi_p      = 10**hi_ax      if m["log"] else hi_ax
        lo2_p     = 10**lo_ax2     if m["log"] else lo_ax2
        hi2_p     = 10**hi_ax2     if m["log"] else hi_ax2
        mode_p    = 10**mode_ax    if m["log"] else mode_ax
        center_p  = 10**center_ax  if m["log"] else center_ax
        tv        = (true_params[p] if true_params and p in true_params else None) or None
        tv_ax     = (np.log10(tv) if m["log"] else tv) if tv is not None else None
        in_ci1_tv = None if tv_ax is None else bool(lo_ax  <= tv_ax <= hi_ax)
        in_ci2_tv = None if tv_ax is None else bool(lo_ax2 <= tv_ax <= hi_ax2)

        hpd_rows.append({
            "parameter":          p,
            "center_phys":        center_p,
            "mode_phys":          mode_p,
            "ci1s_lo_phys":       lo_p,    "ci1s_hi_phys":      hi_p,
            "ci1s_lo_axis":       lo_ax,   "ci1s_hi_axis":      hi_ax,
            "sigma1s_minus_axis": sm_ax,   "sigma1s_plus_axis": sp_ax,
            "ci2s_lo_phys":       lo2_p,   "ci2s_hi_phys":      hi2_p,
            "ci2s_lo_axis":       lo_ax2,  "ci2s_hi_axis":      hi_ax2,
            "sigma2s_minus_axis": sm_ax2,  "sigma2s_plus_axis": sp_ax2,
            "true_value":         tv,
            "true_in_ci1s":       in_ci1_tv,
            "true_in_ci2s":       in_ci2_tv,
        })
        ci_rows.append({
            "parameter":    p,
            "center_phys":  center_p,
            "mode_phys":    mode_p,
            "ci1s_lo_phys": lo_p,  "ci1s_hi_phys": hi_p,
            "ci2s_lo_phys": lo2_p, "ci2s_hi_phys": hi2_p,
            "true_value":   tv,
            "true_in_ci1s": in_ci1_tv,
            "true_in_ci2s": in_ci2_tv,
        })
        tv_str = f"  true={'in 1σ' if in_ci1_tv else 'OUT OF 1σ'}" \
                 if in_ci1_tv is not None else ""
        _p(f"  {p:20s}  1σ=[{lo_p:.4g},{hi_p:.4g}]  2σ=[{lo2_p:.4g},{hi2_p:.4g}]"
           f"  center={center_p:.4g}  mode={mode_p:.4g}{tv_str}")

    pd.DataFrame(ci_rows).to_csv(
        os.path.join(out_dir, "credible_intervals.csv"), index=False)
    pd.DataFrame(hpd_rows).to_csv(
        os.path.join(out_dir, "hpd_intervals.csv"), index=False,
        float_format="%.6e")
    _write_modes_csv(hpd_rows, best_fit_params, out_dir)
    _p(f"  1σ/2σ intervals → {os.path.join(out_dir, 'hpd_intervals.csv')}")

    _write_contour_csvs(panels_2d, param_meta, out_dir,
                        best_fit_params=best_fit_params)

    df.drop(columns=["_L"], inplace=True, errors="ignore")
    gc.collect()

    _write_summary(out_dir, cfg=cfg, mode=mode, grid_mode=grid_mode,
                   sigma_info=sigma_info or {}, log_Z=log_Z, ci_rows=ci_rows)
    _tau0_c = _tau0_curves(cfg, (len(param_meta) - 1) // 4, panels_2d,
                            panels_1d, best_fit_params, out_dir) if (cfg and mode in ("trpl", "joint")) else {}
    _plot_full_corner(panels_2d, panels_1d, param_meta,
                      true_params, out_dir, mode, log_Z, cfg=cfg,
                      best_fit_params=best_fit_params, tau0_curves=_tau0_c)


# ─────────────────────────────────────────────────────────────────────────────
# Contour-path export
# ─────────────────────────────────────────────────────────────────────────────

def _write_contour_csvs(panels_2d, param_meta, out_dir, best_fit_params=None):
    """
    For every 2-D panel, extract the 1σ/2σ/3σ mass-contour paths and write
    one CSV per panel to <out_dir>/contours/.

    CSV columns:
      x_phys   – physical x value (first key in panel dict)
      y_phys   – physical y value (second key in panel dict)
      sigma    – sigma level (1, 2, or 3)

    Disconnected contour segments within the same sigma level are separated by
    a row of NaN, so OriginLab line plots automatically show gaps.
    Contour levels are anchored to the best-fit point when best_fit_params
    is provided (see _mass_contour_levels / _panel_p_bf).
    """
    contours_dir = os.path.join(out_dir, "contours")
    os.makedirs(contours_dir, exist_ok=True)

    meta_by_phys = {m["phys"]: m for m in param_meta}
    _SIGMA_NUM   = {"1σ": 1, "2σ": 2, "3σ": 3}

    for (pa, pb), agg in panels_2d.items():
        ma = meta_by_phys.get(pa)   # pa = x-axis param (first/smaller index)
        mb = meta_by_phys.get(pb)   # pb = y-axis param
        if ma is None or mb is None:
            continue

        ca = ma["axis_col"]   # x-axis column
        cb = mb["axis_col"]   # y-axis column

        x_vals = np.sort(agg[ca].unique())
        y_vals = np.sort(agg[cb].unique())
        try:
            Z_mass = agg.pivot(index=cb, columns=ca,
                               values="probability").values
        except Exception:
            continue

        bf_ij = None
        if best_fit_params:
            bfv_x = best_fit_params.get(ma["phys"])
            bfv_y = best_fit_params.get(mb["phys"])
            if bfv_x is not None and bfv_y is not None:
                bx = np.log10(float(bfv_x)) if ma["log"] else float(bfv_x)
                by = np.log10(float(bfv_y)) if mb["log"] else float(bfv_y)
                bf_ij = (int(np.argmin(np.abs(y_vals - by))),
                         int(np.argmin(np.abs(x_vals - bx))))
        m_lvls, m_lbls = _mass_contour_levels(
            Z_mass.ravel(), mass_2d=Z_mass, bf_ij=bf_ij)
        if m_lvls is None:
            continue

        # extract paths via a temporary (never shown) figure
        fig_tmp, ax_tmp = plt.subplots()
        try:
            cs = ax_tmp.contour(x_vals, y_vals, Z_mass, levels=m_lvls)
        except Exception:
            plt.close(fig_tmp)
            continue

        rows = []
        nan_row = {f"{pa}_phys": np.nan, f"{pb}_phys": np.nan, "sigma": np.nan}
        for lv, segs in zip(cs.levels, cs.allsegs):
            lbl      = m_lbls.get(float(lv), "")
            sigma_n  = _SIGMA_NUM.get(lbl, np.nan)
            first_seg = True
            for seg in segs:
                if len(seg) == 0:
                    continue
                if not first_seg:
                    rows.append(nan_row.copy())
                first_seg = False
                for pt in seg:
                    ax_x, ax_y = float(pt[0]), float(pt[1])
                    phys_x = 10**ax_x if ma["log"] else ax_x
                    phys_y = 10**ax_y if mb["log"] else ax_y
                    rows.append({f"{pa}_phys": phys_x,
                                 f"{pb}_phys": phys_y,
                                 "sigma": sigma_n})
            if not first_seg:    # separate sigma levels with NaN too
                rows.append(nan_row.copy())

        plt.close(fig_tmp)

        if rows:
            fname = os.path.join(contours_dir,
                                 f"contour_{_safe(pa)}_vs_{_safe(pb)}.csv")
            pd.DataFrame(rows).to_csv(fname, index=False, float_format="%.6e")
            _p(f"  Contours → {fname}")


def _write_modes_csv(hpd_rows, best_fit_params, out_dir):
    """
    Write modes.csv: mode position (posterior mode) and optimizer best-fit
    for each parameter, in physical units.
    """
    rows = []
    for h in hpd_rows:
        p   = h["parameter"]
        bfv = best_fit_params.get(p) if best_fit_params else None
        rows.append({
            "parameter":    p,
            "mode_phys":    h["mode_phys"],
            "best_fit_phys": bfv,
        })
    if rows:
        fname = os.path.join(out_dir, "modes.csv")
        pd.DataFrame(rows).to_csv(fname, index=False, float_format="%.6e")
        _p(f"  Modes → {fname}")


# ─────────────────────────────────────────────────────────────────────────────
# tau0 theoretical iso-lines (Et vs taup, NT vs taup)
# ─────────────────────────────────────────────────────────────────────────────

def _tau0_curves(cfg, num_traps, panels_2d, panels_1d, best_fit_params, out_dir):
    """
    Compute and export theoretical tau0 iso-lines on Et-taup and NT-taup panels.

    Requires config key  physics.tau0_s  (seconds).  Returns {} when absent.

    For each trap k the function uses a representative NT_k and Et_k (taken from
    best_fit_params first, then the posterior mode) and sweeps taup over the full
    panel axis range.

    Photodoping condition:  NT > n1  where  n1 = NC * exp(-(Eg - Et) / kT)
    Transition taup:        taup_cross = tau0 * ni / NT_k

    Et vs taup:
        photodoping    (taup < taup_cross):
            Et = Eg + kT * ln( (tau0*ni/taup)^2 / (NT_k * NC) )
        non-photodoping (taup >= taup_cross):
            Et = Eg + kT * ln( tau0*ni / (taup * NC) )

    NT vs taup (photodoping only, taup < tau0*ni/n1_k):
        NT = (tau0*ni/taup)^2 / n1_k,   n1_k = NC * exp(-(Eg - Et_k) / kT)

    CSVs are written to  <out_dir>/tau0_curves/ .
    Returns  {k: {"Et_taup": df, "Nt_taup": df}}  for each trap k.
    """
    phys_cfg = cfg.get("physics", {}) if cfg else {}
    tau0 = phys_cfg.get("tau0_s")
    if tau0 is None:
        return {}

    tau0 = float(tau0)
    Eg   = float(phys_cfg["Eg_eV"])
    NC   = float(phys_cfg["NC_cm3"])
    NV   = float(phys_cfg.get("NV_cm3", NC))
    T_K  = float(phys_cfg.get("T_K", 300))
    kB   = 8.617333e-5          # eV / K
    kT   = kB * T_K
    ni   = np.sqrt(NC * NV) * np.exp(-Eg / (2.0 * kT))

    _p(f"\ntau0 curves: tau0={tau0:.3e} s  ni={ni:.3e} cm⁻³  kT={kT:.4f} eV")

    curves_dir = os.path.join(out_dir, "tau0_curves")
    os.makedirs(curves_dir, exist_ok=True)

    result = {}

    for k in range(1, num_traps + 1):
        Et_phys   = f"Et_{k}_eV"
        Nt_phys   = f"Nt_{k}_cm-3"
        taup_phys = f"tau_p_{k}_s"
        Nt_ax     = f"log10_Nt_{k}"
        taup_ax   = f"log10_tau_p_{k}"

        # ── actual taup grid points ───────────────────────────────────────────
        taup_log = None
        if taup_phys in panels_1d and taup_ax in panels_1d[taup_phys].columns:
            taup_log = np.sort(panels_1d[taup_phys][taup_ax].values.astype(float))
        if taup_log is None:
            for (pa, pb), agg in panels_2d.items():
                if taup_ax in agg.columns:
                    taup_log = np.sort(agg[taup_ax].unique().astype(float))
                    break
        if taup_log is None:
            _p(f"  tau0 curves trap {k}: taup axis not found, skipping")
            continue

        taup = 10.0 ** taup_log

        # ── Et and NT grid bounds (used to clip computed curves) ──────────────
        Et_lo = Et_hi = None
        if Et_phys in panels_1d and Et_phys in panels_1d[Et_phys].columns:
            Et_lo = float(panels_1d[Et_phys][Et_phys].min())
            Et_hi = float(panels_1d[Et_phys][Et_phys].max())

        Nt_ax_lo = Nt_ax_hi = None
        if Nt_phys in panels_1d and Nt_ax in panels_1d[Nt_phys].columns:
            Nt_ax_lo = float(panels_1d[Nt_phys][Nt_ax].min())
            Nt_ax_hi = float(panels_1d[Nt_phys][Nt_ax].max())

        # ── best-fit NT_k and Et_k (from optimizer JSON; posterior mode as fallback) ──
        NT_k = None
        if best_fit_params and Nt_phys in best_fit_params:
            NT_k = float(best_fit_params[Nt_phys])
        elif Nt_phys in panels_1d and Nt_ax in panels_1d[Nt_phys].columns:
            df1n = panels_1d[Nt_phys]
            idx  = int(df1n["probability"].values.argmax())
            NT_k = 10.0 ** float(df1n.iloc[idx][Nt_ax])

        Et_k = None
        if best_fit_params and Et_phys in best_fit_params:
            Et_k = float(best_fit_params[Et_phys])
        elif Et_phys in panels_1d and Et_phys in panels_1d[Et_phys].columns:
            df1e = panels_1d[Et_phys]
            idx  = int(df1e["probability"].values.argmax())
            Et_k = float(df1e.iloc[idx][Et_phys])

        # ── Et vs taup curve ──────────────────────────────────────────────────
        df_et = pd.DataFrame()
        if NT_k is not None and NT_k > 0:
            taup_cross = tau0 * ni / NT_k      # transition taup (s)
            Et_vals    = np.full(len(taup), np.nan)
            branches   = np.empty(len(taup), dtype=object)

            # photodoping (NT > n1) ↔ taup > tau0·ni/NT_k
            pd_mask = taup > taup_cross
            np_mask = ~pd_mask

            if pd_mask.any():
                arg = (tau0 * ni / taup[pd_mask]) ** 2 / (NT_k * NC)
                Et_vals[pd_mask] = np.where(arg > 0, Eg + kT * np.log(arg), np.nan)
                branches[pd_mask] = "photodoping"

            if np_mask.any():
                arg = tau0 * ni / (taup[np_mask] * NC)
                Et_vals[np_mask] = np.where(arg > 0, Eg + kT * np.log(arg), np.nan)
                branches[np_mask] = "non_photodoping"

            in_grid = np.isfinite(Et_vals)
            if Et_lo is not None:
                in_grid &= (Et_vals >= Et_lo) & (Et_vals <= Et_hi)
            df_et = pd.DataFrame({
                Et_phys:   Et_vals[in_grid],
                taup_phys: taup[in_grid],
                taup_ax:   taup_log[in_grid],
                "branch":  branches[in_grid],
            })
            csv_et = os.path.join(curves_dir, f"tau0_curve_Et{k}_vs_taup{k}.csv")
            df_et.to_csv(csv_et, index=False, float_format="%.6e")
            _p(f"  tau0 curve Et_{k} vs tau_p_{k} → {csv_et}  "
               f"(NT_k={NT_k:.3e} cm⁻³, taup_cross={taup_cross:.3e} s, "
               f"{in_grid.sum()}/{len(taup)} pts in grid)")
        else:
            _p(f"  tau0 curves trap {k}: NT_k not available, skipping Et-taup curve")

        # ── NT vs taup curve (photodoping only) ───────────────────────────────
        df_nt = pd.DataFrame()
        if Et_k is not None:
            n1_k = NC * np.exp(-(Eg - Et_k) / kT)
            if n1_k > 0:
                taup_cross_nt = tau0 * ni / n1_k
                pd_mask_nt    = taup < taup_cross_nt
                if pd_mask_nt.any():
                    tp_pd     = taup[pd_mask_nt]
                    NT_vals   = (tau0 * ni / tp_pd) ** 2 / n1_k
                    NT_log    = np.log10(np.clip(NT_vals, 1e-300, None))
                    in_grid_nt = np.isfinite(NT_log)
                    if Nt_ax_lo is not None:
                        in_grid_nt &= (NT_log >= Nt_ax_lo) & (NT_log <= Nt_ax_hi)
                    df_nt = pd.DataFrame({
                        Nt_phys:   NT_vals[in_grid_nt],
                        Nt_ax:     NT_log[in_grid_nt],
                        taup_phys: tp_pd[in_grid_nt],
                        taup_ax:   taup_log[pd_mask_nt][in_grid_nt],
                        "branch":  "photodoping",
                    })
                csv_nt = os.path.join(curves_dir, f"tau0_curve_Nt{k}_vs_taup{k}.csv")
                df_nt.to_csv(csv_nt, index=False, float_format="%.6e")
                _p(f"  tau0 curve Nt_{k} vs tau_p_{k} → {csv_nt}  "
                   f"(n1_k={n1_k:.3e} cm⁻³, taup_cross={taup_cross_nt:.3e} s)")
        else:
            _p(f"  tau0 curves trap {k}: Et_k not available, skipping NT-taup curve")

        # ── NT vs taup non-photodoping boundary: horizontal line at taup_cross ─
        # taup_cross = tau0·ni/n1_k  (n1_k fixed from best-fit Et_k).
        # This is a constant taup value — a horizontal line spanning the NT axis.
        df_nt_boundary = pd.DataFrame()
        if Et_k is not None and Nt_phys in panels_1d and Nt_ax in panels_1d[Nt_phys].columns:
            n1_k_bnd       = NC * np.exp(-(Eg - Et_k) / kT)
            taup_cross_bnd = tau0 * ni / n1_k_bnd
            tp_log_bnd     = float(np.log10(taup_cross_bnd)) if taup_cross_bnd > 0 else None
            Nt_log_grid    = np.sort(panels_1d[Nt_phys][Nt_ax].values.astype(float))
            NT_grid        = 10.0 ** Nt_log_grid
            if tp_log_bnd is not None:
                in_grid_bnd = (NT_grid <= n1_k_bnd) & \
                              (Nt_log_grid >= (Nt_ax_lo if Nt_ax_lo is not None else -np.inf)) & \
                              (Nt_log_grid <= (Nt_ax_hi if Nt_ax_hi is not None else  np.inf))
                df_nt_boundary = pd.DataFrame({
                    Nt_phys:   NT_grid[in_grid_bnd],
                    Nt_ax:     Nt_log_grid[in_grid_bnd],
                    taup_phys: np.full(in_grid_bnd.sum(), taup_cross_bnd),
                    taup_ax:   np.full(in_grid_bnd.sum(), tp_log_bnd),
                    "branch":  "non_photodoping",
                })
            csv_bnd = os.path.join(curves_dir, f"tau0_boundary_Nt{k}_vs_taup{k}.csv")
            df_nt_boundary.to_csv(csv_bnd, index=False, float_format="%.6e")
            _p(f"  tau0 boundary Nt_{k} vs tau_p_{k} → {csv_bnd}  "
               f"(taup_cross={taup_cross_bnd:.3e} s, n1_k={n1_k_bnd:.3e} cm⁻³)")

        # ── Nt vs Et curve ───────────────────────────────────────────────────
        # Fix taup at best-fit; sweep Nt grid.
        # Photodoping (Nt >= Nt_cross = tau0·ni/taup_bf):
        #   Et = Eg + kT·ln((tau0·ni/taup_bf)² / (NC·Nt))
        # Non-photodoping (Nt < Nt_cross): Et = Et_cross (constant, vertical line)
        df_nt_et = pd.DataFrame()
        taup_bf = None
        if best_fit_params and taup_phys in best_fit_params:
            taup_bf = float(best_fit_params[taup_phys])
        elif taup_phys in panels_1d and taup_ax in panels_1d[taup_phys].columns:
            df1tp  = panels_1d[taup_phys]
            idx_tp = int(df1tp["probability"].values.argmax())
            taup_bf = 10.0 ** float(df1tp.iloc[idx_tp][taup_ax])

        if taup_bf is not None and taup_bf > 0:
            Nt_log_grid_et = None
            if Nt_phys in panels_1d and Nt_ax in panels_1d[Nt_phys].columns:
                Nt_log_grid_et = np.sort(panels_1d[Nt_phys][Nt_ax].values.astype(float))
            if Nt_log_grid_et is None:
                for (pa, pb), agg in panels_2d.items():
                    if Nt_ax in agg.columns:
                        Nt_log_grid_et = np.sort(agg[Nt_ax].unique().astype(float))
                        break
            if Nt_log_grid_et is not None:
                NT_grid_et    = 10.0 ** Nt_log_grid_et
                Nt_cross_et   = tau0 * ni / taup_bf   # boundary Nt
                Et_vals_et    = np.full(len(NT_grid_et), np.nan)
                branches_et   = np.empty(len(NT_grid_et), dtype=object)
                pd_mask_et    = NT_grid_et >= Nt_cross_et
                if pd_mask_et.any():
                    arg = (tau0 * ni / taup_bf) ** 2 / (NC * NT_grid_et[pd_mask_et])
                    Et_vals_et[pd_mask_et] = np.where(arg > 0, Eg + kT * np.log(arg), np.nan)
                    branches_et[pd_mask_et] = "photodoping"
                np_mask_et = ~pd_mask_et
                if np_mask_et.any():
                    # Et at Nt = Nt_cross = Eg + kT·ln(tau0·ni / (taup_bf·NC))
                    Et_at_cross = Eg + kT * np.log(tau0 * ni / (taup_bf * NC))
                    Et_vals_et[np_mask_et] = Et_at_cross
                    branches_et[np_mask_et] = "non_photodoping"
                in_grid_et = np.isfinite(Et_vals_et)
                if Et_lo is not None:
                    in_grid_et &= (Et_vals_et >= Et_lo) & (Et_vals_et <= Et_hi)
                if Nt_ax_lo is not None:
                    in_grid_et &= (Nt_log_grid_et >= Nt_ax_lo) & (Nt_log_grid_et <= Nt_ax_hi)
                if in_grid_et.any():
                    df_nt_et = pd.DataFrame({
                        Et_phys:  Et_vals_et[in_grid_et],
                        Nt_phys:  NT_grid_et[in_grid_et],
                        Nt_ax:    Nt_log_grid_et[in_grid_et],
                        "branch": branches_et[in_grid_et],
                    })
                    csv_nt_et = os.path.join(curves_dir, f"tau0_curve_Nt{k}_vs_Et{k}.csv")
                    df_nt_et.to_csv(csv_nt_et, index=False, float_format="%.6e")
                    _p(f"  tau0 curve Nt_{k} vs Et_{k} → {csv_nt_et}  "
                       f"(taup_bf={taup_bf:.3e} s, Nt_cross={Nt_cross_et:.3e} cm⁻³, "
                       f"{in_grid_et.sum()}/{len(NT_grid_et)} pts in grid)")
            else:
                _p(f"  tau0 curves trap {k}: Nt grid not found, skipping Nt-Et curve")
        else:
            _p(f"  tau0 curves trap {k}: taup_bf not available, skipping Nt-Et curve")

        result[k] = {"Et_taup": df_et, "Nt_taup": df_nt,
                     "Nt_taup_boundary": df_nt_boundary,
                     "Nt_Et": df_nt_et}

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Percentile-based color levels
# ─────────────────────────────────────────────────────────────────────────────

_PCTS_BEST = [90, 80, 70, 60, 50, 40, 30, 20, 10, 5, 2, 1]


def _percentile_levels(panels_2d, out_dir, col="probability_density"):
    """
    Compute discrete contour levels from 'best N%' percentiles of `col`
    pooled across all 2-D panels.

    For slice corner plots pass col="relative_prob" so the color scale is
    anchored to the optimizer best-fit (relative_prob = 1.0 at best-fit,
    > 1 if a grid point beats the optimizer).  For full-grid corner plots
    the default col="probability_density" is used.

    Writes percentile_thresholds.csv to out_dir.
    Returns (levels, norm, cmap, tick_vals, tick_lbls).
    """
    all_dens = np.concatenate([agg[col].values
                               for agg in panels_2d.values()
                               if col in agg.columns])
    pos = all_dens[np.isfinite(all_dens) & (all_dens > 0)]

    # "best N%" boundary = (100-N)th percentile of the positive densities
    thresholds = np.array([np.percentile(pos, 100 - p) for p in _PCTS_BEST])

    levels = np.unique(np.concatenate([[0.0], thresholds, [float(pos.max())]]))
    n_bands = len(levels) - 1
    cmap = matplotlib.colormaps["viridis"].resampled(n_bands)
    norm = matplotlib.colors.BoundaryNorm(levels, cmap.N)

    # Colorbar ticks: one per unique threshold, labelled "best N%"
    seen, tick_vals, tick_lbls = set(), [], []
    for p, t in zip(_PCTS_BEST, thresholds):
        t = float(t)
        if t not in seen:
            seen.add(t)
            tick_vals.append(t)
            tick_lbls.append(f"best {p}%")

    # Export thresholds to CSV
    rows = [{
        "label":                               f"best_{p}pct",
        "best_pct":                             p,
        "probability_density_threshold":        float(t),
        "log10_probability_density_threshold":  float(np.log10(t)) if t > 0 else np.nan,
    } for p, t in zip(_PCTS_BEST, thresholds)]
    csv_path = os.path.join(out_dir, "percentile_thresholds.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False, float_format="%.6e")
    _p(f"  Percentile thresholds → {csv_path}")

    return levels, norm, cmap, tick_vals, tick_lbls


# ─────────────────────────────────────────────────────────────────────────────
# Corner plots
# ─────────────────────────────────────────────────────────────────────────────

def _plot_full_corner(panels_2d, panels_1d, param_meta,
                      true_params, out_dir, mode, log_Z, cfg=None,
                      best_fit_params=None, tau0_curves=None):
    levels, norm, cmap, tick_vals, tick_lbls = _percentile_levels(panels_2d, out_dir)

    N = len(param_meta)
    #fig, axes = plt.subplots(N, N, figsize=(3.2 * N, 3.2 * N))
    fig, axes = plt.subplots(N, N, figsize=(3.2 * N, 3.2 * N))
    fig.suptitle(f"Bayesian posterior — {mode.upper()}  "
                 rf"log$_{{10}}Z$ = {log_Z*np.log10(np.e):.2f}",
                 fontsize=13, y=1.01)

    for i, mi in enumerate(param_meta):
        for j, mj in enumerate(param_meta):
            ax = axes[i, j]
            if j > i:
                ax.set_visible(False)
                continue

            def _ax_val(params, m):
                if not params or m["phys"] not in params: return None
                v = params[m["phys"]]
                if v is None: return None
                v = float(v)
                return np.log10(v) if m["log"] else v

            xi_true = _ax_val(true_params, mi)
            xi_bf   = _ax_val(best_fit_params, mi)

            if i == j:
                agg    = panels_1d[mi["phys"]]
                x      = agg[mi["axis_col"]].values
                y      = agg["probability_density"].values
                in_ci1 = agg["in_ci1s"].values if "in_ci1s" in agg.columns \
                         else np.ones(len(x), dtype=bool)
                in_ci2 = agg["in_ci2s"].values if "in_ci2s" in agg.columns \
                         else np.ones(len(x), dtype=bool)
                mode_ax  = float(x[np.argmax(agg["probability"].values)])
                center   = xi_bf if xi_bf is not None else mode_ax
                lo1 = float(x[in_ci1].min()) if in_ci1.any() else x[0]
                hi1 = float(x[in_ci1].max()) if in_ci1.any() else x[-1]
                lo2 = float(x[in_ci2].min()) if in_ci2.any() else x[0]
                hi2 = float(x[in_ci2].max()) if in_ci2.any() else x[-1]
                delta1 = max(center - lo1, hi1 - center)
                delta2 = max(center - lo2, hi2 - center)
                lo_ci1 = center - delta1;  hi_ci1 = center + delta1
                lo_ci2 = center - delta2;  hi_ci2 = center + delta2
                ax.axvspan(lo_ci2, hi_ci2, alpha=0.12, color="steelblue", label="2σ")
                ax.axvspan(lo_ci1, hi_ci1, alpha=0.25, color="steelblue", label="1σ")
                ax.plot(x, y, color="steelblue", lw=1.5)
                ax.axvline(mode_ax,  color="black",     lw=1.2, ls="-")
                ax.axvline(lo_ci1,   color="steelblue", lw=0.7, ls="--")
                ax.axvline(hi_ci1,   color="steelblue", lw=0.7, ls="--")
                ax.axvline(lo_ci2,   color="steelblue", lw=0.5, ls=":")
                ax.axvline(hi_ci2,   color="steelblue", lw=0.5, ls=":")
                ax.text(0.97, 0.95, f"$1\\sigma:\\pm{delta1:.2f}$\n$2\\sigma:\\pm{delta2:.2f}$",
                        transform=ax.transAxes, fontsize=5,
                        ha="right", va="top", color="steelblue")
                if xi_true is not None:
                    ax.axvline(xi_true, color="red",    lw=1.2, ls="--")
                if xi_bf is not None:
                    ax.axvline(xi_bf,   color="orange", lw=1.2, ls=":")
                ax.set_ylim(bottom=0)
                ax.set_xlabel(mi["label"], fontsize=7)
                ax.set_ylabel("PDF", fontsize=7)
                if mi["phys"] == "krad_cm3s":
                    ax.xaxis.set_major_locator(
                        matplotlib.ticker.MaxNLocator(integer=True))
            else:
                key = (mj["phys"], mi["phys"]) \
                      if (mj["phys"], mi["phys"]) in panels_2d \
                      else (mi["phys"], mj["phys"])
                agg = panels_2d[key]
                cx = mj["axis_col"];  cy = mi["axis_col"]
                x_vals = np.sort(agg[cx].unique())
                y_vals = np.sort(agg[cy].unique())
                Z_dens = agg.pivot(index=cy, columns=cx,
                                   values="probability_density").values
                ax.contourf(x_vals, y_vals, Z_dens, levels=levels, norm=norm,
                            cmap=cmap, extend="max")
                Z_mass = agg.pivot(index=cy, columns=cx,
                                   values="probability").values
                bf_ij = None
                if best_fit_params:
                    bfv_x = best_fit_params.get(mj["phys"])
                    bfv_y = best_fit_params.get(mi["phys"])
                    if bfv_x is not None and bfv_y is not None:
                        bx = np.log10(float(bfv_x)) if mj["log"] else float(bfv_x)
                        by = np.log10(float(bfv_y)) if mi["log"] else float(bfv_y)
                        bf_ij = (int(np.argmin(np.abs(y_vals - by))),
                                 int(np.argmin(np.abs(x_vals - bx))))
                m_lvls, m_lbls = _mass_contour_levels(
                    Z_mass.ravel(), mass_2d=Z_mass, bf_ij=bf_ij)
                if m_lvls is not None:
                    try:
                        cs = ax.contour(x_vals, y_vals, Z_mass, levels=m_lvls,
                                        colors="white", linewidths=0.5, alpha=0.7)
                        ax.clabel(cs, levels=m_lvls, fmt=m_lbls,
                                  fontsize=5, inline=True)
                    except Exception:
                        pass

                # ── tau0 theoretical iso-line overlay ────────────────────────
                if tau0_curves:
                    _n_tr = (len(param_meta) - 1) // 4
                    for _k in range(1, _n_tr + 1):
                        _c   = tau0_curves.get(_k, {})
                        _Et  = f"Et_{_k}_eV"
                        _Nt  = f"Nt_{_k}_cm-3"
                        _tp  = f"tau_p_{_k}_s"
                        _tax = f"log10_tau_p_{_k}"
                        _nax = f"log10_Nt_{_k}"
                        # Et vs tau_p panel: x = Et (eV), y = log10(taup)
                        if mj["phys"] == _Et and mi["phys"] == _tp:
                            _df = _c.get("Et_taup", pd.DataFrame())
                            if not _df.empty and _Et in _df.columns and _tax in _df.columns:
                                for _br, _col, _ls in [
                                    ("photodoping",     "lime", "-"),
                                    ("non_photodoping", "cyan", "--"),
                                ]:
                                    _m = _df["branch"] == _br
                                    if _m.any():
                                        ax.plot(_df.loc[_m, _Et].values,
                                                _df.loc[_m, _tax].values,
                                                color=_col, lw=1.5, ls=_ls,
                                                zorder=5)
                        # Nt vs tau_p panel: x = log10(Nt), y = log10(taup)
                        if mj["phys"] == _Nt and mi["phys"] == _tp:
                            _df = _c.get("Nt_taup", pd.DataFrame())
                            if not _df.empty and _nax in _df.columns and _tax in _df.columns:
                                ax.plot(_df[_nax].values, _df[_tax].values,
                                        color="lime", lw=1.5, ls="-", zorder=5)
                            _dfb = _c.get("Nt_taup_boundary", pd.DataFrame())
                            if not _dfb.empty and _nax in _dfb.columns and _tax in _dfb.columns:
                                ax.plot(_dfb[_nax].values, _dfb[_tax].values,
                                        color="cyan", lw=1.5, ls="--", zorder=5)
                        # Nt vs Et panel: x = Et (eV), y = log10(Nt)
                        if mj["phys"] == _Et and mi["phys"] == _Nt:
                            _df = _c.get("Nt_Et", pd.DataFrame())
                            if not _df.empty and _Et in _df.columns and _nax in _df.columns:
                                for _br, _col, _ls in [
                                    ("photodoping",     "lime", "-"),
                                    ("non_photodoping", "cyan", "--"),
                                ]:
                                    _m = _df["branch"] == _br
                                    if _m.any():
                                        ax.plot(_df.loc[_m, _Et].values,
                                                _df.loc[_m, _nax].values,
                                                color=_col, lw=1.5, ls=_ls,
                                                zorder=5)

                xj_true = _ax_val(true_params, mj)
                if xi_true is not None: ax.axhline(xi_true, color="red", lw=1.0, ls="--")
                if xj_true is not None: ax.axvline(xj_true, color="red", lw=1.0, ls="--")
                ax.set_xlabel(mj["label"], fontsize=7)
                ax.set_ylabel(mi["label"], fontsize=7)
                if mj["phys"] == "krad_cm3s":
                    ax.xaxis.set_major_locator(
                        matplotlib.ticker.MaxNLocator(integer=True))
                if mi["phys"] == "krad_cm3s":
                    ax.yaxis.set_major_locator(
                        matplotlib.ticker.MaxNLocator(integer=True))
            ax.tick_params(labelsize=6)


    sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.tight_layout()
    plt.subplots_adjust(right=0.82)
    cax = fig.add_axes([0.85, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    cb = fig.colorbar(sm, cax=cax)
    cb.set_ticks(tick_vals)
    cb.set_ticklabels(tick_lbls)
    cb.ax.tick_params(labelsize=6)

    out_png = os.path.join(out_dir, "cornerplot.png")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    _p(f"  Corner plot → {out_png}")


def _plot_slice_corner(panels_2d, panels_1d, param_meta,
                       true_params, out_dir, mode, cfg=None,
                       best_fit_params=None, tau0_curves=None):
    """Corner-plot-like figure where each off-diagonal panel is a 2-D slice."""
    levels, norm, cmap, tick_vals, tick_lbls = _percentile_levels(
        panels_2d, out_dir, col="relative_prob")

    N   = len(param_meta)
    fig, axes = plt.subplots(N, N, figsize=(3.2 * N, 3.2 * N))
    fig.suptitle(f"Slice posteriors — {mode.upper()}", fontsize=13, y=1.01)

    for i, mi in enumerate(param_meta):
        for j, mj in enumerate(param_meta):
            ax = axes[i, j]
            if j > i:
                ax.set_visible(False)
                continue

            def _ax_val(params, m):
                if not params or m["phys"] not in params: return None
                v = params[m["phys"]]
                if v is None: return None
                v = float(v)
                return np.log10(v) if m["log"] else v

            xi_true = _ax_val(true_params, mi)
            xi_bf   = _ax_val(best_fit_params, mi)

            if i == j:
                # 1-D profile
                if mi["phys"] in panels_1d:
                    agg    = panels_1d[mi["phys"]]
                    x      = agg[mi["axis_col"]].values
                    y      = agg["probability_density"].values
                    in_ci1 = agg["in_ci1s"].values if "in_ci1s" in agg.columns \
                             else np.ones(len(x), dtype=bool)
                    in_ci2 = agg["in_ci2s"].values if "in_ci2s" in agg.columns \
                             else np.ones(len(x), dtype=bool)
                    mode_ax  = float(x[np.argmax(agg["probability"].values)])
                    center   = xi_bf if xi_bf is not None else mode_ax
                    lo1 = float(x[in_ci1].min()) if in_ci1.any() else x[0]
                    hi1 = float(x[in_ci1].max()) if in_ci1.any() else x[-1]
                    lo2 = float(x[in_ci2].min()) if in_ci2.any() else x[0]
                    hi2 = float(x[in_ci2].max()) if in_ci2.any() else x[-1]
                    delta1 = max(center - lo1, hi1 - center)
                    delta2 = max(center - lo2, hi2 - center)
                    lo_ci1 = center - delta1;  hi_ci1 = center + delta1
                    lo_ci2 = center - delta2;  hi_ci2 = center + delta2
                    ax.axvspan(lo_ci2, hi_ci2, alpha=0.12, color="steelblue", label="2σ")
                    ax.axvspan(lo_ci1, hi_ci1, alpha=0.25, color="steelblue", label="1σ")
                    ax.plot(x, y, color="steelblue", lw=1.5)
                    ax.axvline(mode_ax,  color="black",     lw=1.2, ls="-")
                    ax.axvline(lo_ci1,   color="steelblue", lw=0.7, ls="--")
                    ax.axvline(hi_ci1,   color="steelblue", lw=0.7, ls="--")
                    ax.axvline(lo_ci2,   color="steelblue", lw=0.5, ls=":")
                    ax.axvline(hi_ci2,   color="steelblue", lw=0.5, ls=":")
                    ax.text(0.97, 0.95, f"$1\\sigma:\\pm{delta1:.2f}$\n$2\\sigma:\\pm{delta2:.2f}$",
                            transform=ax.transAxes, fontsize=5,
                            ha="right", va="top", color="steelblue")
                if xi_true is not None:
                    ax.axvline(xi_true, color="red",    lw=1.2, ls="--")
                if xi_bf is not None:
                    ax.axvline(xi_bf,   color="orange", lw=1.2, ls=":")
                ax.set_ylim(bottom=0)
                ax.set_xlabel(mi["label"], fontsize=7)
                ax.set_ylabel("PDF", fontsize=7)
                if mi["phys"] == "krad_cm3s":
                    ax.xaxis.set_major_locator(
                        matplotlib.ticker.MaxNLocator(integer=True))
            else:
                # 2-D slice
                key = None
                for k in [(mj["phys"], mi["phys"]), (mi["phys"], mj["phys"])]:
                    if k in panels_2d:
                        key = k; break
                if key is None:
                    ax.set_visible(False); continue

                agg = panels_2d[key]
                cx = mj["axis_col"];  cy = mi["axis_col"]
                x_vals = np.sort(agg[cx].unique())
                y_vals = np.sort(agg[cy].unique())
                try:
                    Z_rel = agg.pivot(index=cy, columns=cx,
                                      values="relative_prob").values
                    ax.contourf(x_vals, y_vals, Z_rel, levels=levels, norm=norm,
                                cmap=cmap, extend="max")
                    Z_mass = agg.pivot(index=cy, columns=cx,
                                       values="probability").values
                    bf_ij = None
                    if best_fit_params:
                        bfv_x = best_fit_params.get(mj["phys"])
                        bfv_y = best_fit_params.get(mi["phys"])
                        if bfv_x is not None and bfv_y is not None:
                            bx = np.log10(float(bfv_x)) if mj["log"] else float(bfv_x)
                            by = np.log10(float(bfv_y)) if mi["log"] else float(bfv_y)
                            bf_ij = (int(np.argmin(np.abs(y_vals - by))),
                                     int(np.argmin(np.abs(x_vals - bx))))
                    m_lvls, m_lbls = _mass_contour_levels(
                        Z_mass.ravel(), mass_2d=Z_mass, bf_ij=bf_ij)
                    if m_lvls is not None:
                        cs = ax.contour(x_vals, y_vals, Z_mass, levels=m_lvls,
                                        colors="white", linewidths=0.5, alpha=0.7)
                        ax.clabel(cs, levels=m_lvls, fmt=m_lbls,
                                  fontsize=5, inline=True)
                except Exception:
                    ax.text(0.5, 0.5, "pivot error", transform=ax.transAxes,
                            ha="center", va="center", fontsize=7)

                # ── tau0 theoretical iso-line overlay ────────────────────────
                if tau0_curves:
                    _n_tr = (len(param_meta) - 1) // 4
                    for _k in range(1, _n_tr + 1):
                        _c   = tau0_curves.get(_k, {})
                        _Et  = f"Et_{_k}_eV"
                        _Nt  = f"Nt_{_k}_cm-3"
                        _tp  = f"tau_p_{_k}_s"
                        _tax = f"log10_tau_p_{_k}"
                        _nax = f"log10_Nt_{_k}"
                        # Et vs tau_p panel: x = Et (eV), y = log10(taup)
                        if mj["phys"] == _Et and mi["phys"] == _tp:
                            _df = _c.get("Et_taup", pd.DataFrame())
                            if not _df.empty and _Et in _df.columns and _tax in _df.columns:
                                for _br, _col, _ls in [
                                    ("photodoping",     "lime", "-"),
                                    ("non_photodoping", "cyan", "--"),
                                ]:
                                    _m = _df["branch"] == _br
                                    if _m.any():
                                        ax.plot(_df.loc[_m, _Et].values,
                                                _df.loc[_m, _tax].values,
                                                color=_col, lw=1.5, ls=_ls,
                                                zorder=5)
                        # Nt vs tau_p panel: x = log10(Nt), y = log10(taup)
                        if mj["phys"] == _Nt and mi["phys"] == _tp:
                            _df = _c.get("Nt_taup", pd.DataFrame())
                            if not _df.empty and _nax in _df.columns and _tax in _df.columns:
                                ax.plot(_df[_nax].values, _df[_tax].values,
                                        color="lime", lw=1.5, ls="-", zorder=5)
                            _dfb = _c.get("Nt_taup_boundary", pd.DataFrame())
                            if not _dfb.empty and _nax in _dfb.columns and _tax in _dfb.columns:
                                ax.plot(_dfb[_nax].values, _dfb[_tax].values,
                                        color="cyan", lw=1.5, ls="--", zorder=5)
                        # Nt vs Et panel: x = Et (eV), y = log10(Nt)
                        if mj["phys"] == _Et and mi["phys"] == _Nt:
                            _df = _c.get("Nt_Et", pd.DataFrame())
                            if not _df.empty and _Et in _df.columns and _nax in _df.columns:
                                for _br, _col, _ls in [
                                    ("photodoping",     "lime", "-"),
                                    ("non_photodoping", "cyan", "--"),
                                ]:
                                    _m = _df["branch"] == _br
                                    if _m.any():
                                        ax.plot(_df.loc[_m, _Et].values,
                                                _df.loc[_m, _nax].values,
                                                color=_col, lw=1.5, ls=_ls,
                                                zorder=5)

                xj_true = _ax_val(true_params, mj)
                if xi_true is not None: ax.axhline(xi_true, color="red", lw=1.0, ls="--")
                if xj_true is not None: ax.axvline(xj_true, color="red", lw=1.0, ls="--")
                ax.set_xlabel(mj["label"], fontsize=7)
                ax.set_ylabel(mi["label"], fontsize=7)
                if mj["phys"] == "krad_cm3s":
                    ax.xaxis.set_major_locator(
                        matplotlib.ticker.MaxNLocator(integer=True))
                if mi["phys"] == "krad_cm3s":
                    ax.yaxis.set_major_locator(
                        matplotlib.ticker.MaxNLocator(integer=True))
            ax.tick_params(labelsize=6)

    sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.tight_layout()
    plt.subplots_adjust(right=0.82)
    cax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_ticks(tick_vals)
    cb.set_ticklabels(tick_lbls)
    cb.ax.tick_params(labelsize=6)

    out_png = os.path.join(out_dir, "cornerplot_slice.png")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    _p(f"  Slice corner plot → {out_png}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",      default="config.json")
    ap.add_argument("--mode",        choices=["trpl", "sspl", "joint"],
                    default="joint", help="Which error(s) to use")
    ap.add_argument("--grid-mode",   choices=["full", "slice"], default=None,
                    help="Override grid.mode from config")
    ap.add_argument("--trpl-errors", default=None,
                    help="TRPL errors CSV (default: <out>/errors_trpl.csv)")
    ap.add_argument("--sspl-errors", default=None,
                    help="SSPL errors CSV (default: <out>/errors_sspl.csv)")
    ap.add_argument("--trpl-1d",     default=None,
                    help="TRPL 1-D profile errors (slice mode)")
    ap.add_argument("--sspl-1d",     default=None,
                    help="SSPL 1-D profile errors (slice mode)")
    args = ap.parse_args()

    cfg_path = args.config if os.path.isabs(args.config) else \
               os.path.join(os.getcwd(), args.config)
    cfg = _load_cfg(cfg_path)
    base = os.path.dirname(cfg_path)

    out_dir_root = cfg["output_dir"]
    if not os.path.isabs(out_dir_root):
        out_dir_root = os.path.join(base, out_dir_root)

    out_dir = os.path.join(out_dir_root, f"bayesian_{args.mode}")
    os.makedirs(out_dir, exist_ok=True)

    grid_mode = args.grid_mode or cfg["grid"]["mode"]
    num_traps = cfg["physics"]["num_traps"]

    trpl_csv = args.trpl_errors or os.path.join(out_dir_root, "errors_trpl.csv")
    sspl_csv = args.sspl_errors or os.path.join(out_dir_root, "errors_sspl.csv")
    trpl_1d  = args.trpl_1d
    sspl_1d  = args.sspl_1d

    true_params = cfg.get("true_params", None)
    param_meta  = _build_param_meta(num_traps)

    best_fit_params = None
    bf_cfg = cfg.get("best_fit", {})
    if "json_path" in bf_cfg:
        bf_path = bf_cfg["json_path"]
        if not os.path.isabs(bf_path):
            bf_path = os.path.join(base, bf_path)
        if os.path.isfile(bf_path):
            with open(bf_path) as fh:
                bf_raw = json.load(fh)
            # Flatten nested trap structure → same keys as true_params / param_meta
            best_fit_params = {"krad_cm3s": bf_raw.get("krad_cm3s")}
            traps_list = bf_raw.get("traps") or bf_raw.get("trap_parameters", [])
            for k, trap in enumerate(traps_list, start=1):
                best_fit_params[f"Et_{k}_eV"]   = trap.get("Et_eV")
                best_fit_params[f"Nt_{k}_cm-3"] = trap.get("Nt_cm3") or trap.get("Nt_1/cm3")
                best_fit_params[f"tau_n_{k}_s"] = trap.get("tau_n_s")
                best_fit_params[f"tau_p_{k}_s"] = trap.get("tau_p_s")
            best_fit_params = {k: v for k, v in best_fit_params.items()
                               if v is not None}
            _p(f"Best-fit params loaded: {bf_path}")
        else:
            _p(f"  (best_fit JSON not found: {bf_path})")

    _p("=" * 60)
    _p(f"04_bayesian  mode={args.mode}  grid={grid_mode}  "
       f"num_traps={num_traps}")
    _p(f"  Output dir : {out_dir}")
    _p(f"  TRPL csv   : {trpl_csv}")
    _p(f"  SSPL csv   : {sspl_csv}")
    _p("=" * 60)

    # ── Save metadata ──────────────────────────────────────────────────────────
    meta = {
        "mode": args.mode, "grid_mode": grid_mode,
        "sigma_trpl": cfg["bayesian"]["sigma_trpl"],
        "sigma_sspl": cfg["bayesian"]["sigma_sspl"],
        "trpl_csv": trpl_csv, "sspl_csv": sspl_csv,
        "true_params": true_params,
    }
    with open(os.path.join(out_dir, "metadata.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    t0 = time.time()
    if grid_mode == "full":
        _run_full(cfg, args.mode, trpl_csv, sspl_csv,
                  out_dir, true_params, param_meta,
                  best_fit_params=best_fit_params)
    else:
        _run_slice(cfg, args.mode, trpl_csv, sspl_csv,
                   trpl_1d, sspl_1d, out_dir, true_params, param_meta,
                   best_fit_params=best_fit_params)

    _p(f"\nTotal elapsed: {time.time()-t0:.1f}s")
    _p("Done.")


if __name__ == "__main__":
    main()
