"""
corner_export.py — plot-ready corner data in ONE HDF5 file (Origin on Windows + matplotlib preview)
=================================================================================================
14/10/2026  (revision, plan ~/.claude/plans/floofy-purring-fiddle.md, Part 3)

A corner plot is one HDF5 group (e.g. /corner/physical, /corner/eigen) with

  attrs   layout ('triangle' | '2trap_mirrored'), param_order, hpd_levels, sigma_levels, colormap, colors_json,
          diag_mode ('marginal' | 'slice'), bins_1d, bins_2d, smooth_bins, coordinates
  params/<key>                attrs: label_origin, label_mpl, unit, scale ('log10' | 'linear' | 'z'),
                                     lo, hi, best_fit, truth, start, et_minus_eg_axis, Eg, offset, slope
  layout                      structured table: row, col, kind (1d|2d|empty), kx, ky, xside, yside, extra
  panels/<kx>__<ky>/hpd       (ny, nx) posterior mass level 0..1 (0 = densest cell, 1 = outside 99 %);
                    /density  (ny, nx) normalised density per (unit x · unit y)
                    /x, /y    bin centres (plot coordinates: decades for log10 parameters)
                    /sigma1, /sigma2   (n, 2) contour polylines of the 68.27 / 95.45 % HPD regions,
                                       NaN rows between segments (Origin draws gaps)
  panels/<k>/x, /pdf, /hpd_band  (n,) 1D marginal per unit of x; hpd_band 2 = inside 1σ, 1 = inside 2σ, 0 = outside
            attrs median, mean; datasets intervals68 / intervals95 (m, 2) HPD intervals (may be disjoint)
  diag/<k>/z, /dlogL_nn, /dlogL_ode, /parabola   (diag_mode = 'slice', eigen corner)
  overlays/eigen_lines (K, n, D), eigen_marks (K, 4, D) [t = -2σ, -σ, σ, 2σ; NaN outside the chord],
           eigen_sigma (K,)

Comparison corner (write_comparison_corner, 15/10/2026): the schema above for the FILL posterior (one sampler) plus
  attrs   fill_method, methods (json list), method_dash_json (json: name -> solid | dash | dot | dashdot)
  methods/<name>/<kx>__<ky>/sigma1, /sigma2     1σ / 2σ HPD contours of every sampler (same bins and smoothing)
  methods/<name>/<k>/x, /pdf, /intervals68, /intervals95     its 1D marginals;  methods/<name> attrs: label, dash,
                                                              n_samples, ess, time_s, ... (sampler info)
  panels/<kx>__<ky>/scatter   (M, 3) equal-weight samples of the fill posterior: x, y, posterior-mass level of the fill
                              map at the sample (same colour levels as hpd), outermost first (densest drawn last)

Slice corner (scripts/paper_slices.py, 19/10/2026): panels log10_n1_k__log10_taup_k may hold the iso-tau_0 lines
  tau0_photodoping, tau0_non_photodoping   (m, 2) [x, y] polylines, NaN rows = gaps (inference/slices.tau0_lines)
  attrs   tau0_json (per trap: tau0_s, source, N_t, k_rad, n_i, ...), tau0_dash_json (regime -> dash)

Plot coordinates: the value stored is log10 of the physical quantity for scale = 'log10' (10**x gives cm^-3 / s),
the value itself for 'linear' and 'z'.  The colour levels are identical in every 2D panel (posterior mass),
so one colour bar is valid for the whole figure.
"""
from __future__ import annotations

import json

import numpy as np
from scipy.ndimage import gaussian_filter

HPD_LEVELS = (0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.99)
METHOD_DASH = {"NUTS": "solid", "emcee": "dash", "SMC": "dot", "AMIS": "dashdot", "cylinder IS": "dashdot",
               "ODE": "solid", "NN axis head": "dot", "NN analytic axis": "dash",     # these 3: low-injection test
               "ODE SMC": "dashdot"}                                                   # SMC with the rate equations
SIGMA_LEVELS = (0.6827, 0.9545)
COLORS = {"sigma1": "#56B4E9", "sigma2": "#D9D9D9", "best_fit": "#E41A1C", "truth": "#00CED1",
          "start": "#4DAF4A", "eigen_axes": "#FFFFFF", "line_1d": "#000000", "tau0": "#7CFC00"}
TAU0_DASH = {"photodoping": "dash", "non_photodoping": "dashdot"}     # iso-tau_0 lines of the slice corner
COLORMAP = "Inferno"


# ─────────────────────────────────────────────────────────────────────────────
# labels (Origin rich text as in corner-plot/corner-Plot-*.otpu, and matplotlib)
# ─────────────────────────────────────────────────────────────────────────────
def labels(key: str, n_traps: int):
    """key in decade coordinates (log10_n1_1, log10_Nt_1, log10_taun_1, log10_taup_1, log10_krad) or z_k."""
    if key.startswith("z_"):
        k = key[2:]
        return (rf"\i(z)\-({k}) = \i(t)\-({k})/\f:Symbol(s)\-({k})", rf"$z_{{{k}}}=t_{{{k}}}/\sigma_{{{k}}}$", "", "z")
    if key == "log10_krad":
        return (r"\i(k)\-(rad) (cm\+(3) s\+(-1))", r"$k_\mathrm{rad}$ (cm$^3$ s$^{-1}$)", "cm^3/s", "log10")
    kind, t = key.rsplit("_", 1)
    ti = t if n_traps > 1 else ""
    sub = f",{ti}" if ti else ""
    if kind == "log10_n1":
        return (rf"\i(n)\-(1{sub}) (cm\+(-3))", rf"$n_{{1{sub}}}$ (cm$^{{-3}}$)", "cm^-3", "log10")
    if kind == "log10_Nt":
        return (rf"\i(N)\-(t{ti}) (cm\+(-3))", rf"$N_{{t{ti}}}$ (cm$^{{-3}}$)", "cm^-3", "log10")
    if kind == "log10_taun":
        return (rf"\i(\f:Symbol(t))\-(SRH,n{ti}) (s)", rf"$\tau_{{\mathrm{{SRH,n{ti}}}}}$ (s)", "s", "log10")
    if kind == "log10_taup":
        return (rf"\i(\f:Symbol(t))\-(SRH,p{ti}) (s)", rf"$\tau_{{\mathrm{{SRH,p{ti}}}}}$ (s)", "s", "log10")
    raise KeyError(key)


def et_minus_eg_label(key: str, n_traps: int):
    t = key.rsplit("_", 1)[1] if n_traps > 1 else ""
    return (rf"\i(E)\-(t{t}) \- \i(E)\-(g) (eV)", rf"$E_{{t{t}}}-E_g$ (eV)")


# ─────────────────────────────────────────────────────────────────────────────
# densities, HPD levels, contours
# ─────────────────────────────────────────────────────────────────────────────
def band_colors_hex():
    """Hex colour of every posterior-mass band (index 0 = densest, <= 1 %; last = outside 99 %), discrete Inferno
    exactly as corner_plot.band_cmap — stored in the HDF5 file for Origin (dots of the combined corner)."""
    import matplotlib as mpl
    from matplotlib.colors import to_hex
    n = len(HPD_LEVELS) + 1
    return [to_hex(c) for c in mpl.colormaps["inferno"](np.linspace(0.98, 0.0, n))]


def hpd_level_map(density):
    """Mass level of every cell: posterior mass of all cells with a density >= this cell (0 = densest)."""
    d = np.nan_to_num(np.asarray(density, float), nan=0.0).ravel()
    tot = d.sum()
    if tot <= 0:
        return np.ones_like(density, dtype=float)
    order = np.argsort(d)[::-1]
    cum = np.empty_like(d)
    cum[order] = np.cumsum(d[order]) / tot
    cum[d <= 0] = 1.0
    return cum.reshape(np.shape(density))


def density_1d(v, w, lo, hi, bins=120, smooth=1.5):
    h, e = np.histogram(v, bins=bins, range=(lo, hi), weights=w)
    if smooth:
        h = gaussian_filter(h.astype(float), smooth, mode="constant")
    dx = e[1] - e[0]
    s = h.sum()
    return 0.5 * (e[1:] + e[:-1]), (h / (s * dx) if s > 0 else h)


def density_2d(vx, vy, w, xr, yr, bins=60, smooth=1.0):
    h, ex, ey = np.histogram2d(vx, vy, bins=bins, range=[xr, yr], weights=w)
    if smooth:
        h = gaussian_filter(h, smooth, mode="constant")
    h = h.T                                                       # (ny, nx)
    a = (ex[1] - ex[0]) * (ey[1] - ey[0])
    s = h.sum()
    return 0.5 * (ex[1:] + ex[:-1]), 0.5 * (ey[1:] + ey[:-1]), (h / (s * a) if s > 0 else h)


def contour_polyline(x, y, level_map, level):
    """Contour of the HPD level map at `level` as one (n, 2) array with NaN rows between segments."""
    import contourpy
    gen = contourpy.contour_generator(x, y, level_map, line_type="Separate")
    segs = gen.lines(level)
    if not segs:
        return np.zeros((0, 2))
    out = []
    for s in segs:
        out.append(s)
        out.append(np.full((1, 2), np.nan))
    return np.vstack(out[:-1])


def hpd_intervals_1d(x, pdf, mass):
    lev = hpd_level_map(pdf)
    inside = lev <= mass
    dx = x[1] - x[0]
    runs, start = [], None
    for i, f in enumerate(inside):
        if f and start is None:
            start = i
        if (not f or i == len(inside) - 1) and start is not None:
            end = i if f else i - 1
            runs.append((x[start] - dx / 2, x[end] + dx / 2))
            start = None
    return np.array(runs).reshape(-1, 2)


def weighted_quantile(v, w, q):
    o = np.argsort(v)
    c = np.cumsum(w[o]); c /= c[-1]
    return float(np.interp(q, c, v[o]))


# ─────────────────────────────────────────────────────────────────────────────
# layouts
# ─────────────────────────────────────────────────────────────────────────────
LAYOUT_DTYPE = np.dtype([("row", "i4"), ("col", "i4"), ("kind", "S8"), ("kx", "S32"), ("ky", "S32"),
                         ("xside", "S8"), ("yside", "S8"), ("extra", "S32")])


def layout_triangle(keys, et_cols=()):
    """Lower triangle; x tick labels on the bottom row, y on the first column, 1D y axis on the right.
    et_cols: keys that get an extra top axis E_t - E_g on their diagonal panel."""
    rows = []
    D = len(keys)
    for r in range(D):
        for c in range(r + 1):
            if r == c:
                rows.append((r, c, b"1d", keys[c].encode(), b"", b"bottom" if r == D - 1 else b"none", b"right",
                             b"top:Et-Eg" if keys[c] in et_cols else b""))
            else:
                rows.append((r, c, b"2d", keys[c].encode(), keys[r].encode(),
                             b"bottom" if r == D - 1 else b"none", b"left" if c == 0 else b"none", b""))
    return np.array(rows, dtype=LAYOUT_DTYPE)


def layout_2trap_mirrored(trap1, trap2):
    """
    Layout of corner-plot/corner-2traps.png (5 rows x 4 columns):
    trap 1 (n1, tau_p, tau_n, N_t) lower-left, diagonal at row = col + 1, axes bottom/left;
    trap 2 (N_t, tau_n, tau_p, n1) upper-right, diagonal at row = col, axes top/right.
    The E_t - E_g axis is an offset outer axis: below the n1,1 labels (bottom) and above the n1,2 labels (top).
    """
    rows = []
    for r in range(5):
        for c in range(4):
            if r == c + 1:          # trap-1 diagonal
                k = trap1[c]
                rows.append((r, c, b"1d", k.encode(), b"", b"bottom" if r == 4 else b"none", b"right",
                             b"bottom:Et-Eg" if (r == 4 and "n1" in k) else b""))
            elif r > c + 1:         # trap-1 2D
                rows.append((r, c, b"2d", trap1[c].encode(), trap1[r - 1].encode(),
                             b"bottom" if r == 4 else b"none", b"left" if c == 0 else b"none",
                             b"bottom:Et-Eg" if (r == 4 and "n1" in trap1[c]) else b""))
            elif r == c:            # trap-2 diagonal
                k = trap2[c]
                rows.append((r, c, b"1d", k.encode(), b"", b"top" if r == 0 else b"none", b"left",
                             b"top:Et-Eg" if (r == 0 and "n1" in k) else b""))
            else:                   # trap-2 2D (c > r)
                rows.append((r, c, b"2d", trap2[c].encode(), trap2[r].encode(),
                             b"top" if r == 0 else b"none", b"right" if c == 3 else b"none",
                             b"top:Et-Eg" if (r == 0 and "n1" in trap2[c]) else b""))
    return np.array(rows, dtype=LAYOUT_DTYPE)


def physical_order(keys):
    """Order of the physical corner: per trap n1, tau_p, tau_n, N_t; k_rad last."""
    traps = sorted({int(k.rsplit("_", 1)[1]) for k in keys if k != "log10_krad"})
    out = [f"{p}_{t}" for t in traps for p in ("log10_n1", "log10_taup", "log10_taun", "log10_Nt")
           if f"{p}_{t}" in keys]
    return out + (["log10_krad"] if "log10_krad" in keys else [])


# ─────────────────────────────────────────────────────────────────────────────
# writer
# ─────────────────────────────────────────────────────────────────────────────
def write_corner(group, samples, weights, keys, meta, layout, ranges, bins_1d=120, bins_2d=60, smooth=1.0,
                 diag_mode="marginal", diag=None, eigen=None, coordinates="decades", attrs=None):
    """
    group    : open h5py group (e.g. f.create_group('corner/physical'))
    samples  : (N, D) in plot coordinates, columns in the order of `keys`
    weights  : (N,) normalised importance weights
    meta     : dict key -> dict(label_origin, label_mpl, unit, scale, best_fit, truth, start, extra attrs)
    layout   : structured array (LAYOUT_DTYPE)
    ranges   : dict key -> (lo, hi)
    diag     : dict key -> dict of 1D arrays (z, dlogL_nn, dlogL_ode, parabola) for diag_mode 'slice'
    eigen    : dict(lines (K, n, D), marks (K, 4, D), sigma (K,)) in the same coordinates
    """
    w = np.asarray(weights, float)
    group.attrs.update({"layout_kind": str(attrs.get("layout_kind", "")) if attrs else "",
                        "param_order": json.dumps(list(keys)), "hpd_levels": np.array(HPD_LEVELS),
                        "sigma_levels": np.array(SIGMA_LEVELS), "colormap": COLORMAP,
                        "colors_json": json.dumps(COLORS), "band_colors_json": json.dumps(band_colors_hex()),
                        "diag_mode": diag_mode, "bins_1d": bins_1d,
                        "bins_2d": bins_2d, "smooth_bins": smooth, "coordinates": coordinates,
                        "n_samples": len(w), "ess": float(1.0 / np.sum(w ** 2))})
    for k, v in (attrs or {}).items():
        group.attrs[k] = v
    gp = group.create_group("params")
    for i, k in enumerate(keys):
        g = gp.create_group(k)
        g.attrs["index"] = i
        g.attrs["lo"], g.attrs["hi"] = ranges[k]
        for a, v in meta[k].items():
            g.attrs[a] = np.nan if v is None else v
    group.create_dataset("layout", data=layout)

    col = {k: i for i, k in enumerate(keys)}
    pn = group.create_group("panels")
    for row in layout:
        kx, ky = row["kx"].decode(), row["ky"].decode()
        kind = row["kind"].decode()
        if kind == "1d" and kx not in pn:
            v = samples[:, col[kx]]
            x, pdf = density_1d(v, w, *ranges[kx], bins=bins_1d, smooth=1.5 * smooth)
            lev = hpd_level_map(pdf)
            g = pn.create_group(kx)
            g["x"], g["pdf"] = x, pdf
            g["hpd_band"] = np.where(lev <= SIGMA_LEVELS[0], 2, np.where(lev <= SIGMA_LEVELS[1], 1, 0)).astype("i1")
            g["intervals68"] = hpd_intervals_1d(x, pdf, SIGMA_LEVELS[0])
            g["intervals95"] = hpd_intervals_1d(x, pdf, SIGMA_LEVELS[1])
            g.attrs["median"] = weighted_quantile(v, w, 0.5)
            g.attrs["mean"] = float(w @ v)
        elif kind == "2d":
            name = f"{kx}__{ky}"
            if name in pn:
                continue
            x, y, dens = density_2d(samples[:, col[kx]], samples[:, col[ky]], w, ranges[kx], ranges[ky],
                                    bins=bins_2d, smooth=smooth)
            lev = hpd_level_map(dens)
            g = pn.create_group(name)
            g["x"], g["y"] = x, y
            g["density"] = dens.astype("f4")
            g["hpd"] = lev.astype("f4")
            g["sigma1"] = contour_polyline(x, y, lev, SIGMA_LEVELS[0])
            g["sigma2"] = contour_polyline(x, y, lev, SIGMA_LEVELS[1])
    if diag is not None:
        gd = group.create_group("diag")
        for k, d in diag.items():
            g = gd.create_group(k)
            for a, v in d.items():
                g[a] = np.asarray(v, float)
    if eigen is not None:
        go = group.create_group("overlays")
        go["eigen_lines"] = eigen["lines"]
        go["eigen_marks"] = eigen["marks"]
        go["eigen_sigma"] = eigen["sigma"]
    return group


# ─────────────────────────────────────────────────────────────────────────────
# physical corner of a prior (n1 instead of E_t) and the comparison corner of several samplers
# ─────────────────────────────────────────────────────────────────────────────
def physical_corner_spec(prior, best_theta=None, truth_theta=None, start_theta=None):
    """
    Keys, labels, ranges and layout of the physical-decade corner (n1 instead of E_t, E_t - E_g axis) of a prior.
    Returns dict(dmap, keys, perm, meta, ranges, layout, layout_kind); samples in plot coordinates are
    dmap.to_x(theta)[:, perm].
    """
    import math as _m
    from .cylinder import DecadeMap
    dmap = DecadeMap(prior)
    keys_p = physical_order(dmap.keys)
    perm = [dmap.keys.index(k) for k in keys_p]
    xlo = dmap.to_x(prior.lo[None])[0]
    xhi = dmap.to_x(prior.hi[None])[0]
    to_x = (lambda th: None if th is None else dmap.to_x(np.asarray(th, float)[None])[0])
    xb, xt, xs = to_x(best_theta), to_x(truth_theta), to_x(start_theta)
    meta, ranges = {}, {}
    for k in keys_p:
        i = dmap.keys.index(k)
        lab_o, lab_m, unit, scale_ = labels(k, prior.n_traps)
        m = {"label_origin": lab_o, "label_mpl": lab_m, "unit": unit, "scale": scale_,
             "best_fit": None if xb is None else xb[i], "truth": None if xt is None else xt[i],
             "start": None if xs is None else xs[i]}
        if k.startswith("log10_n1"):
            eo, em = et_minus_eg_label(k, prior.n_traps)
            m.update(label_origin_et=eo, label_mpl_et=em, log10_Nc=_m.log10(prior.Nc),
                     kT_ln10=prior.kT * _m.log(10), Eg=prior.Eg)
        meta[k], ranges[k] = m, (min(xlo[i], xhi[i]), max(xlo[i], xhi[i]))
    if prior.n_traps == 2 and "log10_krad" not in keys_p:
        t1 = [k for k in keys_p if k.endswith("_1")]
        t2 = [k for k in keys_p if k.endswith("_2")]
        layout, kind = layout_2trap_mirrored(t1, [t2[3], t2[2], t2[1], t2[0]]), "2trap_mirrored"
    else:
        layout, kind = layout_triangle(keys_p, et_cols=[k for k in keys_p if k.startswith("log10_n1")]), "triangle"
    return {"dmap": dmap, "keys": keys_p, "perm": perm, "meta": meta, "ranges": ranges, "layout": layout,
            "layout_kind": kind}


def level_at(x, y, xc, yc, level_map):
    """Posterior-mass level of the map cell containing each point (x, y); 1 (outside 99 %) outside the map."""
    dx, dy = xc[1] - xc[0], yc[1] - yc[0]
    ix = np.floor((np.asarray(x) - (xc[0] - dx / 2)) / dx).astype(int)
    iy = np.floor((np.asarray(y) - (yc[0] - dy / 2)) / dy).astype(int)
    inside = (ix >= 0) & (ix < len(xc)) & (iy >= 0) & (iy < len(yc))
    out = np.ones(len(ix))
    out[inside] = np.asarray(level_map)[iy[inside], ix[inside]]
    return out


def write_comparison_corner(group, fill_name, methods, keys, meta, layout, ranges, scatter=None, method_info=None,
                            bins_1d=120, bins_2d=60, smooth=1.0, attrs=None):
    """
    One corner with the posterior-mass map of the fill sampler and the 1σ / 2σ HPD contours (and 1D marginals)
    of every sampler, all from the same bins and smoothing.

    group     : open h5py group (e.g. f.create_group('corner/physical'))
    fill_name : key of `methods` whose posterior gives the colour map (hpd / density) and the scatter levels
    methods   : dict name -> (samples (N, D) in plot coordinates, columns in the order of `keys`, weights (N,) or None)
    scatter   : (M, D) equal-weight samples of the fill posterior (drawn coloured by the posterior-mass level)
    method_info : dict name -> dict of attrs (label, time_s, n_evals, ess, r_hat, log_z, ...)
    """
    norm_w = (lambda S, W: np.full(len(S), 1.0 / len(S)) if W is None else np.asarray(W, float) / np.sum(W))
    S_f, W_f = methods[fill_name]
    a = dict(attrs or {})
    write_corner(group, S_f, norm_w(S_f, W_f), keys, meta, layout, ranges, bins_1d, bins_2d, smooth,
                 diag_mode="marginal", coordinates="decades", attrs=a)
    group.attrs.update({"fill_method": fill_name, "methods": json.dumps(list(methods)),
                        "method_dash_json": json.dumps({m: METHOD_DASH.get(m, "dashdot") for m in methods})})
    col = {k: i for i, k in enumerate(keys)}
    root = group.create_group("methods")
    for name, (S, W) in methods.items():
        w = norm_w(S, W)
        gm = root.create_group(name)
        gm.attrs.update({"dash": METHOD_DASH.get(name, "dashdot"), "n_samples": len(w), "ess": float(1.0 / np.sum(w ** 2)),
                         "label": name})
        for k_, v in ((method_info or {}).get(name) or {}).items():
            gm.attrs[k_] = np.nan if v is None else v
        for row in layout:
            kx, ky, kind = row["kx"].decode(), row["ky"].decode(), row["kind"].decode()
            if kind == "1d" and kx not in gm:
                v = S[:, col[kx]]
                x, pdf = density_1d(v, w, *ranges[kx], bins=bins_1d, smooth=1.5 * smooth)
                g = gm.create_group(kx)
                g["x"], g["pdf"] = x, pdf
                g["intervals68"] = hpd_intervals_1d(x, pdf, SIGMA_LEVELS[0])
                g["intervals95"] = hpd_intervals_1d(x, pdf, SIGMA_LEVELS[1])
                g.attrs["median"] = weighted_quantile(v, w, 0.5)
            elif kind == "2d" and f"{kx}__{ky}" not in gm:
                x, y, dens = density_2d(S[:, col[kx]], S[:, col[ky]], w, ranges[kx], ranges[ky], bins=bins_2d,
                                        smooth=smooth)
                lev = hpd_level_map(dens)
                g = gm.create_group(f"{kx}__{ky}")
                g["sigma1"] = contour_polyline(x, y, lev, SIGMA_LEVELS[0])
                g["sigma2"] = contour_polyline(x, y, lev, SIGMA_LEVELS[1])
    if scatter is not None:
        for name, g in group["panels"].items():
            if "__" not in name:
                continue
            kx, ky = name.split("__")
            sx, sy = scatter[:, col[kx]], scatter[:, col[ky]]
            lv = level_at(sx, sy, g["x"][()], g["y"][()], g["hpd"][()])
            o = np.argsort(-lv, kind="stable")                    # outermost first: the densest points end on top
            g["scatter"] = np.column_stack([sx[o], sy[o], lv[o]]).astype("f4")
    return group

