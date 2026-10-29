"""
data.py — import and (optional) preprocessing of tr-PL and ss-PL data
=====================================================================

tr-PL
-----
Two input formats:

* raw transient   : two columns time [ns], PL (a.u.)  →  ``load_trpl_raw``
                    then ``preprocess_trpl`` (trim at the peak, normalise,
                    monotone B-spline in log–log space, convert to
                    tau_diff vs QFLS) — wraps ``preprocessing/data_utils.py``.
* preprocessed    : columns containing 'qfls' and 'tau'  →  ``load_trpl_processed``

Both end in a DataFrame with columns ``qfls`` [eV] and ``tau_diff`` [s];
``clean_trpl`` removes non-finite / non-positive tau_diff and sorts by QFLS.

ss-PL
-----
``load_sspl`` reads either the tab-separated "suns / QFLS" file with a units
row (e.g. ``85-15_film/sspl_suns_qfls.dat``) or a CSV with any of the columns
intensity_suns / qfls(_eV) / plqy / G_cm3s, and returns a DataFrame with the
standard columns ``intensity_suns``, ``qfls``, ``plqy``, ``G_cm3s`` (missing
ones are absent).  No conversion PLQY -> QFLS is needed: the likelihood can
compare log10 PLQY directly (modes 'plqy_vs_suns', 'plqy_vs_qfls').

``g_per_sun`` gives the volumetric generation rate at 1 sun from the AM1.5G
spectrum (same numerics as the grid pipeline of the original submission).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import DATA_DIR, add_legacy_paths


def _find_col(columns, *keywords):
    """First column whose normalised name contains one of the keywords."""
    def _norm(s):
        return str(s).lower().replace(" ", "").replace("_", "").replace("(", "").replace(")", "")
    for kw in keywords:
        for col in columns:
            if _norm(kw) in _norm(col):
                return col
    return None


# ─────────────────────────────────────────────────────────────────────────────
# tr-PL
# ─────────────────────────────────────────────────────────────────────────────

def read_numeric_table(path, skiprows="auto"):
    """
    Read a text table with numeric columns separated by commas, semicolons,
    tabs or spaces.

    skiprows : 'auto' skips blank lines and every line that does not start
        with two numbers (header / units rows); an int skips exactly that many
        lines first (and then behaves like 'auto').
    Returns (data (n, m) float array, header tokens of the first skipped
    non-numeric line or None).
    """
    import re
    split = re.compile(r"[,;\t]+|\s{2,}|\s+")
    with open(path, encoding="utf-8", errors="replace") as fh:
        lines = fh.read().splitlines()
    if isinstance(skiprows, int):
        lines = lines[skiprows:]
    header, rows = None, []
    for ln in lines:
        if not ln.strip():
            continue
        tok = [t for t in re.split(r"[,;\t]|\s{2,}", ln.strip()) if t.strip()]
        if len(tok) < 2:
            tok = ln.split()
        try:
            rows.append([float(t) for t in tok])
        except ValueError:
            if header is None and not rows:
                header = [t.strip() for t in tok]
    width = min(len(r) for r in rows)
    return np.array([r[:width] for r in rows], dtype=np.float64), header


def load_trpl_raw(path, time_col: int = 0, pl_col: int = 1, skiprows="auto") -> pd.DataFrame:
    """
    Raw transient: time [ns] as index and a 'pl' column.

    Works for comma/tab/space separated files with or without header.
    NOTE: the 85:15 file ``data/85-15_film/trpl_transient.csv`` has no
    header, but the original notebooks read it with ``skiprows=1`` (dropping
    the first data point); its config sets ``skiprows: 1`` to reproduce the
    paper's preprocessed curve exactly.
    """
    arr, _ = read_numeric_table(path, skiprows=skiprows)
    out = pd.DataFrame({"time": arr[:, time_col], "pl": arr[:, pl_col]}).set_index("time")
    return out[np.isfinite(out["pl"])]


def load_trpl_processed(path) -> pd.DataFrame:
    """Preprocessed tr-PL file → DataFrame with columns qfls [eV], tau_diff [s]."""
    df = pd.read_csv(path)
    q = _find_col(df.columns, "qfls")
    t = _find_col(df.columns, "tau_diff", "taudiff", "tau")
    if q is None or t is None:
        raise ValueError(f"Cannot find QFLS / tau_diff columns in {list(df.columns)}")
    out = pd.DataFrame({"qfls": df[q].astype(float).values,
                        "tau_diff": df[t].astype(float).values})
    for extra in ("time", "pl"):
        if extra in df.columns:
            out[extra] = df[extra].values
    return out


def preprocess_trpl(df_raw: pd.DataFrame, npulse: float, Eg: float,
                    n_knots: int = 5, n_pred: int = 256, verbose: bool = False) -> dict:
    """
    Raw transient → tau_diff vs QFLS, using the legacy preprocessing chain of
    ``preprocessing/data_utils.py`` (identical to the original Figure notebooks):

    1. ``trim_and_normalize``  — start at the PL maximum, PL_max = 1, t(peak) = 0
    2. ``spline_torch``        — monotone (dPL/dt <= 0) cubic B-spline in
                                 log10(t)–log10(PL), pinned at the first point,
                                 evaluated at ``n_pred`` log-spaced times
    3. ``t_pl_to_qfls_tau``    — QFLS = Vt ln(PL/PL_max) + 2 Vt ln(n_pulse/n_i),
                                 tau_diff = -2 / (d ln PL / dt)

    Returns dict with the intermediate DataFrames ``trimmed``, ``spline``,
    ``processed`` (spline-based, used for the fit) and ``raw_converted``
    (raw points converted the same way, for plots only).
    """
    import contextlib
    import io
    add_legacy_paths()
    import data_utils as utils

    trimmed = utils.trim_and_normalize(df_raw.copy())
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink) if not verbose else contextlib.nullcontext():
        spline = utils.spline_torch(trimmed, n_points=n_knots, degree=3, n_pred=n_pred)
    processed = utils.t_pl_to_qfls_tau(spline, npulse, Eg)
    raw_converted = utils.t_pl_to_qfls_tau(trimmed, npulse, Eg)
    return {"trimmed": trimmed, "spline": spline, "processed": processed,
            "raw_converted": raw_converted}


def clean_trpl(df: pd.DataFrame, qfls_range=None) -> pd.DataFrame:
    """
    Keep finite points with tau_diff > 0 (optionally inside a QFLS window),
    sorted by increasing QFLS.  Prints how many points were dropped.
    """
    q = df["qfls"].to_numpy(float)
    t = df["tau_diff"].to_numpy(float)
    ok = np.isfinite(q) & np.isfinite(t) & (t > 0)
    if qfls_range is not None:
        ok &= (q >= qfls_range[0]) & (q <= qfls_range[1])
    n_drop = int((~ok).sum())
    if n_drop:
        print(f"clean_trpl: dropped {n_drop} of {len(df)} points "
              f"(non-finite, tau_diff <= 0{' or outside QFLS window' if qfls_range else ''}).")
    out = pd.DataFrame({"qfls": q[ok], "tau_diff": t[ok]})
    return out.sort_values("qfls").reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# ss-PL
# ─────────────────────────────────────────────────────────────────────────────

def load_sspl(path, fmt: str = "auto", plqy_in_percent: bool = False) -> pd.DataFrame:
    """
    Load ss-PL data into standard columns.

    The column meaning is taken from the header line (keywords 'intensity' /
    'suns', 'qfls', 'plqy', 'G'/'generation'); unit rows and blank lines are
    skipped.  Without a header, two columns are read as (intensity_suns, plqy).
    fmt : kept for backward compatibility ('auto', 'suns_qfls_dat', 'csv').
    plqy_in_percent : divide the PLQY column by 100.
    """
    arr, header = read_numeric_table(path)
    out = pd.DataFrame()
    if header is None:
        header = ["intensity_suns", "plqy"][:arr.shape[1]]
    for j, name in enumerate(header[:arr.shape[1]]):
        n = name.lower().replace(" ", "").replace("_", "")
        if "intensity" in n or "suns" in n:
            out["intensity_suns"] = arr[:, j]
        elif "qfls" in n:
            out["qfls"] = arr[:, j]
        elif "plqy" in n:
            out["plqy"] = arr[:, j] / (100.0 if plqy_in_percent else 1.0)
        elif "generation" in n or n.startswith("g"):
            out["G_cm3s"] = arr[:, j]
    if out.empty:
        raise ValueError(f"No known ss-PL columns in header {header}")
    return out


def sspl_xy(df: pd.DataFrame, mode: str):
    """(x, y) arrays of a standardised ss-PL DataFrame for a comparison mode."""
    from .sspl_tf import canonical_mode
    m = canonical_mode(mode)
    xcol = {"suns": "intensity_suns", "G": "G_cm3s", "qfls": "qfls"}[m.split("_vs_")[1]]
    ycol = m.split("_vs_")[0]
    for col in (xcol, ycol):
        if col not in df.columns:
            raise ValueError(f"mode '{m}' needs column '{col}' in the ss-PL data")
    return df[xcol].to_numpy(float), df[ycol].to_numpy(float)


def g_per_sun(Eg: float, thickness_cm: float, absorptance: float = 1.0, am15g_path=None) -> float:
    """
    Volumetric generation rate at 1 sun [cm^-3 s^-1]:
        G = absorptance * int_{E >= Eg} Phi_AM1.5G(E) dE / d
    (step-function absorption; same numerics as 03_sspl_errors.py).
    """
    from scipy.integrate import trapezoid
    from scipy.interpolate import PchipInterpolator
    if am15g_path is None:
        am15g_path = DATA_DIR / "AM15G.dat"
    am = pd.read_csv(am15g_path, sep=r"\s+", header=None, names=["E", "phi"], engine="python")
    E = np.linspace(am["E"].min(), am["E"].max(), 1000)
    phi = PchipInterpolator(am["E"], am["phi"], extrapolate=False)(E)
    mask = E >= Eg
    return float(absorptance * trapezoid(phi[mask], E[mask]) / thickness_cm)


def g_per_sun_from_voc_rad(voc_rad: float, krad: float, Eg: float, Nc: float, Nv: float, T: float = 300.0) -> float:
    """
    1-sun generation rate [cm^-3 s^-1] for which the model's radiative limit equals a measured Voc,rad:
    in the radiative limit G = k_rad (n p - n_i^2) and QFLS = kT ln(n p / n_i^2), so
        Voc,rad = kT ln(1 + G / (k_rad n_i^2))   <=>   G = k_rad n_i^2 (exp(Voc,rad / kT) - 1).
    kT and n_i as in the ss-PL model (sspl_tf.SsplModel).  With this G the measured QFLS = Voc,rad + kT ln PLQY
    and the model's QFLS = kT ln(n p / n_i^2) share the same calibration.
    """
    from .sspl_tf import KB_EV
    kT = KB_EV * float(T)
    ni2 = Nc * Nv * np.exp(-Eg / kT)
    return float(krad * ni2 * np.expm1(voc_rad / kT))


def sspl_g_per_sun(cfg: dict, constants: dict, prior=None):
    """
    1-sun generation rate of a config and the Voc,rad check -> (G, info).

    ``sspl.voc_rad_V`` set (needs k_rad fixed in ``prior.fixed``): G from the measured Voc,rad
    (``g_per_sun_from_voc_rad``); otherwise the step-absorption estimate ``g_per_sun`` with ``sspl.absorptance``.
    info: Voc,rad of both estimates and the absorptance that the Voc,rad-based G corresponds to.
    """
    from .sspl_tf import KB_EV
    ss, ph = cfg["sspl"], cfg["physics"]
    Eg, T, d_cm = float(ph["Eg_eV"]), float(ph.get("T_K", 300.0)), float(ph["thickness_nm"]) * 1e-7
    Nc, Nv = float(constants["Nc"]), float(constants["Nv"])
    kT = KB_EV * T
    ni2 = Nc * Nv * np.exp(-Eg / kT)
    g_abs = g_per_sun(Eg, d_cm, absorptance=ss["absorptance"])
    fixed = (cfg.get("prior") or {}).get("fixed") or {}
    krad = 10.0 ** fixed["log10_krad"] if "log10_krad" in fixed else None
    info = {"G_absorptance": g_abs, "absorptance": ss["absorptance"], "krad_fixed": krad}
    if krad is not None:
        info["voc_rad_model_absorptance_V"] = float(kT * np.log1p(g_abs / (krad * ni2)))
    if ss.get("voc_rad_V") is None:
        info.update({"source": "absorptance", "G_per_sun": g_abs})
        return g_abs, info
    if krad is None:
        raise ValueError("sspl.voc_rad_V needs k_rad fixed (prior.fixed.log10_krad)")
    g = g_per_sun_from_voc_rad(float(ss["voc_rad_V"]), krad, Eg, Nc, Nv, T)
    info.update({"source": "voc_rad", "voc_rad_V": float(ss["voc_rad_V"]), "G_per_sun": g,
                 "voc_rad_model_V": float(kT * np.log1p(g / (krad * ni2))),
                 "absorptance_equivalent": float(ss["absorptance"] * g / g_abs)})
    return g, info
