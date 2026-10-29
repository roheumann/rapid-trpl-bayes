"""
build_corner.py — corner plot in Origin (2021+, Windows) from ONE HDF5 file written on rekenbeest
================================================================================================
Input : results/<run>/cylinder_<net>/cylinder.h5   (inference/corner_export.py, schema_version 1)
        or results/<run>/corner_combined/corner_combined.h5 (scripts/combined_corner.py: NUTS, emcee and SMC in one
        corner; FIGURE = 'physical')
        or results/<run>/slices_paper/slices_paper.h5 (scripts/paper_slices.py: sliced grid of the paper, no diagonal,
        simulation panels in the upper right; additionally two graphs from your templates TEMPLATES, see below)
Output: next to the .h5 file: corner_<figure>.opju (data + graph, editable), corner_<figure>.pdf/.png;
        in Origin's User Files folder (German Origin: Anwenderdateien) the template corner-<layout>-<figure>-v2.otpu
        (the look of this figure, reusable by hand).

Run inside Origin (embedded Python, originpro built in):
    Window -> Script Window:   run -pyf "C:\\...\\corner-plot\\origin\\build_corner.py"
or open in Code Builder and press F5.  One-time setup: Connectivity -> Python Packages -> install  h5py
(numpy comes with originpro).

Settings: H5 (None = file dialog), FIGURE ('physical' | 'eigen' | 'loadings'), BACKGROUND, STYLE.

What it does
  1. reads the corner group (params, layout table, panels, overlays) with h5py
  2. writes every panel into an Origin workbook / matrix book (the data stay in the project)
  3. builds ONE graph page with one layer per panel at the position given by the layout table:
       2D: posterior-mass map as a heatmap with the SAME levels (0, 1, 2, 5, ..., 99, 100 %) and the SAME 14 fixed
           Inferno colours in every panel (set level by level, no palette file needed), one discrete colour bar
           (own layer right of the grid), 1σ (sky blue) / 2σ (light grey) HPD contour lines,
           best fit (red dotted), truth (cyan), start point (green), eigen-axes (white), white ticks
           (plus one linked frame layer per 2D panel that keeps the axis lines black);
           slice corner: iso-tau_0 lines (green; dashed N_t >= 10 n1, dash-dot N_t <= 0.1 n1) in the n1 - tau_p panels
       1D: marginal (black) with 2σ / 1σ HPD bands (light grey / sky blue), best fit / truth lines;
           eigen corner: Δln L along v_k (NN solid, ODE dashed) and the Laplace parabola
       E_t − E_g: extra linear axis on the n1 panels named in the layout table, and on top of every other n1 column
           (n1,2 of the slice corner), formula-linked to the panel: E_t − E_g = kT ln10 (log10 n1 − log10 N_c)
       combined corner (group has methods/): 1σ / 2σ contours and 1D marginals of every sampler in the same colours,
           dash pattern per sampler (NUTS solid, emcee dash, SMC dot); BACKGROUND = 'samples': the samples of the
           fill sampler as dots coloured by their posterior-mass band on black (the 'outside 99 %' colour),
           'bands': the band map
  4. saves .opju, exports PDF/PNG and saves the page as template corner-<layout>-<figure>-v2 (User Files folder)

Every Origin-specific call sits in a small helper below; if one fails on your Origin version the script prints
which one and continues, so the rest of the figure is still built.  Plot coordinates: log10 parameters are drawn
in decades on linear axes (the posterior maps are regular grids in decades — no re-triangulation on log axes).
"""
import json
import math
import os
import re
import traceback

import numpy as np
import originpro as op

H5 = None                 # None -> file dialog; or r"C:\data\cylinder.h5"
FIGURE = "physical"       # 'physical' | 'eigen' | 'loadings'
BACKGROUND = "samples"    # combined corner only: 'samples' (coloured dots) | 'bands' (posterior-mass map)

# slices_paper.h5 only: graphs from YOUR templates, one curve set per layer.  A template is looked up as
# <name>.otpu / .otp in Origin's User Files folder (whatever its name: User Files, Anwenderdateien, ...) and in
# TEMPLATE_DIRS, subfolders included.
# Layer contents (in this order; a template with fewer layers gets the rest in its last layer):
#   'pl_t' PL/PL0 vs t, 'tau_qfls' tau_diff vs QFLS, 'conc' n, p, n_t vs QFLS, 'rates' trapping/detrapping vs QFLS
#   'tau_t' tau_diff vs t; '_us' after pl_t / tau_t: time in microseconds.  The script prints each template layer's
#   axis titles, so a wrong mapping is easy to spot.  t_pl_qfls_tau_diff_paper: (a) PL vs t (us, log-log),
#   (b) PL vs t (s, linear t), (c) tau_diff vs t, (d) tau_diff vs QFLS.
TEMPLATE_GRAPHS = False   # slices_paper.h5: also build the graphs from TEMPLATES (False: corner plot only)
TEMPLATES = {"t_pl_qfls_tau_diff_paper": ["pl_t_us", "pl_t", "tau_t", "tau_qfls"], "concentration-rates": ["conc", "rates"]}
TEMPLATE_DIRS = [r"C:\Users\ro.heumann\Documents\OriginLab\Anwenderdateien"]
CURVE_COLORS = {"best_fit": "#E41A1C", "truth": "#00CED1", "data": "#000000", "ode": "#00CED1"}

STYLE = {                 # cm and pt; replace by the numbers in old_template_style.json (inspect_template.py)
    "panel_cm": 2.4, "gap_cm": 0.15,                   # square panels, gap between them
    "margin_left_cm": 2.4, "margin_bottom_cm": 2.2, "margin_top_cm": 1.8,
    "cbar_gap_cm": 1.0, "cbar_w_cm": 0.45,             # colour bar right of the grid
    "font": "Arial", "tick_pt": 9, "title_pt": 11,     # real point sizes (fonts are not scaled with the panels)
    "label_gap_cm": 0.12,                              # axis -> 10^n labels -> title
    "line_1d_width": 1.5, "contour_width": 1.2, "overlay_width": 1.2,
    "sample_symbol_size": 1.5,                         # pt, dots of the combined corner
    "cbar": None,             # colour bar [left, top, width, height] in cm; None: right of the grid
    "cbar_pt": 9,             # font size of the colour bar labels and legends
    "symbol_size": 3, "data_symbol_size": 2,           # pt: start point / data points (PPC panels of build_corner_modes)
    "fit_data_symbol_size": 4,                         # pt: measured data in the fit panels (open circles, under the fits)
    "layer_factor": 1.0,      # Origin's element scale factor of every layer
    "interpolate": True,      # 2D maps as Origin colour-fill contours (interpolated between grid points); False: heatmap
    "upsample": 3,            # heatmap fallback of a failed contour: grid refined 3x (bilinear) so it looks the same
    "merged_page_cm": 50.0,   # height of the merged page before pfit2l fits it to the figure
    "et_axis_cm": 1.35,       # room above a panel with a top E_t - E_g axis (tick labels + title): the PL(t) panel
                              # of a slice corner ends this far above the n1,2 column's axis
    "et_pl_shift_cm": 0.4,    # ... and moves right by this, so its x tick labels clear the E_t title
    "x_label_rotate": 45,     # degrees: x tick labels of the corner panels (0 = horizontal); E_t and fit panels stay flat
    "x_label_rotate_top": None,  # degrees for x tick labels on TOP of a panel; None = x_label_rotate
    "x_label_edge_cm": 0.3,   # tilted 10^n labels: no tick label closer than this to the panel edge they lean over
    "data_spacing_cm": 0.2,   # measured data in the fit panels: one symbol per this distance along the curve
}
# Fallback for files without band_colors_json: Inferno, densest band (≤ 1 %) first, 'outside 99 %' (black) last
BAND_COLORS = ["#f5f992", "#f5d949", "#fcb418", "#f98e09", "#ef6e21", "#de5238", "#c63d4d", "#a82e5f", "#8a226a",
               "#6c186e", "#4c0c6b", "#2b0b57", "#0e092b", "#000004"]
# Colour bar titles (Origin rich text) for the colour_bar_title attribute of the file; cylinder / combined: mass
CBAR_TITLES = {"(p(best fit) - p) / p(best fit)": r"(\i(p)\-(best fit) - \i(p)) / \i(p)\-(best fit)",
               "relative p: best N % of the slice points": r"relative \i(p): best N % of the slice points",
               "": "posterior mass of higher density"}
DASH = {"solid": 0, "dash": 1, "dot": 2, "dashdot": 3}
COLORS_FALLBACK = {"sigma1": "#56B4E9", "sigma2": "#D9D9D9", "best_fit": "#E41A1C", "truth": "#00CED1",
                   "start": "#4DAF4A", "eigen_axes": "#FFFFFF", "line_1d": "#000000", "tau0": "#7CFC00"}

LOG = []


def step(name):
    """Decorator-free helper: run f, log failures, continue."""
    def wrap(f, *a, **k):
        try:
            return f(*a, **k)
        except Exception as e:                          # noqa: BLE001  (report and continue)
            LOG.append(f"[{name}] {type(e).__name__}: {e}")
            print(LOG[-1])
            traceback.print_exc()
            return None
    return wrap


def _s(v):
    return v.decode() if isinstance(v, bytes) else str(v)


def rgb(hexcol):
    h = hexcol.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


LT_FAILED = {}            # LabTalk statement (numbers masked) -> [count, first full statement]


def split_lt(cmd):
    """Split a LabTalk line into statements at ';' outside quotes, braces and parentheses."""
    out, cur, depth, quote = [], [], 0, False
    for ch in cmd:
        if ch == '"':
            quote = not quote
        elif not quote and ch in "({":
            depth += 1
        elif not quote and ch in ")}":
            depth -= 1
        if ch == ";" and not quote and depth == 0:
            out.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    out.append("".join(cur).strip())
    return [s for s in out if s]


def lt(cmd):
    """Run LabTalk statement by statement, so one failing statement does not silently skip the rest of the line.
    Failures (op.lt_exec returning False) are collected in LT_FAILED and listed at the end of the run."""
    ok = True
    for s in split_lt(cmd):
        if op.lt_exec(s + ";") is False:
            ok = False
            key = re.sub(r"-?\d+(\.\d+)?(e[-+]?\d+)?", "#", s)[:120]
            if key not in LT_FAILED:
                LT_FAILED[key] = [0, s]
                print(f"LabTalk failed: {s}")
            LT_FAILED[key][0] += 1
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# reading
# ─────────────────────────────────────────────────────────────────────────────
def pick_file():
    if H5:
        return H5
    lt('dlgfile group:="*.h5" title:="cylinder / corner_combined / slices_paper .h5 from rekenbeest"; string __f$ = fname$;')
    return op.get_lt_str("__f$")


def read_group(path, figure):
    import h5py
    with h5py.File(path, "r") as f:
        if figure == "loadings":
            g = f["loadings"]
            return {"V": g["V"][()], "sigma": g["sigma"][()], "cols": json.loads(g.attrs["columns_origin"])}
        g = f[f"corner/{figure}"]
        keys = json.loads(g.attrs["param_order"])
        params = {k: {a: (_s(v) if isinstance(v, (bytes, np.bytes_)) else v) for a, v in g["params"][k].attrs.items()}
                  for k in keys}
        panels = {}
        for name, pg in g["panels"].items():
            panels[name] = {d: pg[d][()] for d in pg.keys()}
            panels[name].update({a: pg.attrs[a] for a in pg.attrs})
        diag = {k: {d: v[()] for d, v in dg.items()} for k, dg in g["diag"].items()} if "diag" in g else {}
        ov = {d: v[()] for d, v in g["overlays"].items()} if "overlays" in g else {}
        methods = {}
        if "methods" in g:                       # combined corner: contours / 1D marginals of every sampler
            dash = json.loads(g.attrs["method_dash_json"])
            for m in json.loads(g.attrs["methods"]):
                gm = g["methods"][m]
                methods[m] = {"dash": DASH.get(dash.get(m, "solid"), 0), "label": _s(gm.attrs.get("label", m)),
                              "panels": {n: {d: v[()] for d, v in pg.items()} for n, pg in gm.items()}}
        curves = {}
        if "curves" in f:                        # slices_paper.h5: data, best_fit/{nn, ode, at_data}, truth/...
            def tree(h):
                return {k: (tree(v) if hasattr(v, "keys") else v[()]) for k, v in h.items()}
            curves = tree(f["curves"])
            curves["_attrs"] = {k: {a: (_s(v) if isinstance(v, (bytes, np.bytes_)) else v) for a, v in f["curves"][k].attrs.items()}
                                for k in f["curves"].keys()}
            sm = f["curves"].attrs.get("sspl_mode")
            curves["_sspl_mode"] = _s(sm) if sm is not None else ""
        return {"keys": keys, "params": params, "layout": g["layout"][()], "panels": panels, "diag": diag,
                "curves": curves,
                "overlays": ov, "hpd_levels": np.array(g.attrs["hpd_levels"]), "diag_mode": _s(g.attrs["diag_mode"]),
                "colors": json.loads(g.attrs["colors_json"]) if "colors_json" in g.attrs else COLORS_FALLBACK,
                "layout_kind": _s(g.attrs.get("layout_kind", "triangle")), "methods": methods,
                "fill_method": _s(g.attrs.get("fill_method", "")),
                "colour_bar_title": _s(g.attrs.get("colour_bar_title", "")),   # e.g. slices: (p(best fit) - p)/p(best fit)
                # slices: iso-tau_0 lines in the n1 - tau_p panels (panels/<kx>__<ky>/tau0_<regime>)
                "tau0": json.loads(g.attrs["tau0_json"]) if "tau0_json" in g.attrs else {},
                "tau0_dash": json.loads(g.attrs["tau0_dash_json"]) if "tau0_dash_json" in g.attrs else {},
                "band_colors": json.loads(g.attrs["band_colors_json"]) if "band_colors_json" in g.attrs else None}


def band_index(hpd, levels):
    """Posterior-mass level -> band index 0 (densest, ≤1 %) ... len(levels) (outside 99 %): same in every panel."""
    return np.searchsorted(np.asarray(levels), np.asarray(hpd, float), side="left").astype(float)


def colour_bounds(levels):
    """Edges of the colour levels in %: 0, 1, 2, 5, ..., 99, 100 -> len(levels) + 1 bands."""
    return np.array([0.0] + [100.0 * float(v) for v in levels] + [100.0])


def band_z(hpd, levels):
    """Z written to the matrix: the centre (in %) of the band of each point.  The band is decided here, exactly as in
    the matplotlib preview, so Origin's level boundary convention (< or ≤) cannot move a point into another band."""
    b = colour_bounds(levels)
    return (0.5 * (b[:-1] + b[1:]))[band_index(hpd, levels).astype(int)]


# ─────────────────────────────────────────────────────────────────────────────
# Origin helpers (each one small; adapt here if your Origin version differs)
# ─────────────────────────────────────────────────────────────────────────────
def new_sheet_xy(book, name, cols):
    """cols: list of (long name, array) -> worksheet in `book`; returns worksheet."""
    wks = book.add_sheet(name[:31]) if book is not None else op.new_sheet("w", lname=name)
    for i, (lname, arr) in enumerate(cols):
        wks.from_list(i, [float(v) for v in np.asarray(arr, float)], lname=lname)
    return wks


def new_matrix(name, Z, x, y):
    ms = op.new_sheet("m", lname=name)
    ms.from_np(np.asarray(Z, float))
    ms.xymap = float(x[0]), float(x[-1]), float(y[0]), float(y[-1])
    return ms


def place_layer(gp, idx, left, top, width, height, unit=3):
    """Layer idx (0-based) at a position in cm (unit 3; 1 = % of the page).  Fonts / symbols are first decoupled from the layer size
    (layer.fixed = 1): otherwise Origin shrinks every text with the (small) panel."""
    lt(f"win -a {gp.name}; page.active = {idx + 1}; layer.fixed = 1; layer.factor = {STYLE['layer_factor']};"
       f" layer.unit = {unit}; layer.left = {left:.3f}; layer.top = {top:.3f}; layer.width = {width:.3f};"
       f" layer.height = {height:.3f};")


def add_layer(gp):
    lt(f"win -a {gp.name}; layer -n;")                 # new, NON-linked layer with bottom X / left Y axes
    return gp[int(op.lt_float("page.nlayers")) - 1]


def set_axes(gl, xr, yr):
    gl.set_xlim(float(xr[0]), float(xr[1]))
    gl.set_ylim(float(yr[0]), float(yr[1]))


def label_sides(xb, xt, yl, yr):
    """Tick labels per side (0/1).  Sets the old layer.x/y.showLabels and the per-axis switches of Origin 2020+
    (layer.x / x2 / y / y2 .showlabel): showLabels alone had no visible effect in the third Origin run."""
    return (f"layer.x.showLabels = {xb + 2 * xt}; layer.y.showLabels = {yl + 2 * yr}; layer.x.showlabel = {xb};"
            f" layer.x2.showlabel = {xt}; layer.y.showlabel = {yl}; layer.y2.showlabel = {yr};")


def axis_sides(gp, idx, xside, yside, xtitle, ytitle):
    """Show tick labels and titles only on the requested sides (bottom/top, left/right), ticks inward on all."""
    a = f"win -a {gp.name}; page.active = {idx + 1};"
    a += " layer.x.showAxes = 3; layer.y.showAxes = 3; layer.x.ticks = 5; layer.y.ticks = 5;"     # both sides, in
    a += " layer.x2.ticks = 5; layer.y2.ticks = 5; "
    a += label_sides(int(xside == "bottom"), int(xside == "top"), int(yside == "left"), int(yside == "right"))
    lt(a)
    step("tick label size")(lt, f"layer.x.label.pt = {STYLE['tick_pt']}; layer.y.label.pt = {STYLE['tick_pt']};"
                                f" layer.x.label.font = font({STYLE['font']}); layer.y.label.font = font({STYLE['font']});")
    if idx == 0:                                         # only the template's first layer has default titles
        op.lt_exec("xb.show = 0;")                       # op.lt_exec: a missing title is no error
        op.lt_exec("yl.show = 0;")
    t = STYLE["title_pt"]
    if xside in ("bottom", "top"):
        o = "xb" if xside == "bottom" else "xt"
        lt(f'label -{o} {xtitle}; {o}.text$ = "{xtitle}"; {o}.show = 1; {o}.fsize = {t};')
    if yside in ("left", "right") and ytitle:
        o = "yl" if yside == "left" else "yr"
        lt(f'label -{o} {ytitle}; {o}.text$ = "{ytitle}"; {o}.show = 1; {o}.fsize = {t};')


def lt_rgb(hexcol):
    return "color({}, {}, {})".format(*rgb(hexcol))


def colormap_plot(gp, idx, gl, ms, bounds, colors, contour=False):
    """Colour map of matrix ms in layer idx, coloured by set_cmap: a colour-fill contour (226; Origin interpolates
    between the grid points, contour lines and labels hidden) or a heatmap (105; one block per cell)."""
    p = None
    if contour:
        p = step("contour plot")(gl.add_mplot, ms, 0, type=226)
    if p is None:
        p = gl.add_mplot(ms, 0, type=105)
        contour = False
    ok = cmap_ready(gp, idx) and step("colour levels")(set_cmap, gp, idx, bounds, colors)
    if not ok:                                          # once more after a redraw (Origin sometimes refuses a contour)
        op.lt_exec("doc -uw;")
        op.lt_exec("sec -p 0.5;")
        ok = cmap_ready(gp, idx) and step("colour levels")(set_cmap, gp, idx, bounds, colors)
    if ok and contour:
        step("hide contour lines")(lt, f"win -a {gp.name}; page.active = {idx + 1}; layer.cmap.showLines(3);"
                                       " layer.cmap.showLabels(3); layer.cmap.updateScale();")
    op.lt_exec(f"win -a {gp.name};")
    op.lt_exec(f"page.active = {idx + 1};")
    op.lt_exec("label -r legend;")                      # Origin's auto legend of the colour map
    return bool(ok)


def regrid(Z, x, y, lo_x, hi_x, lo_y, hi_y, factor=1):
    """Bilinear resampling of Z (rows = y) from the cell centres x, y onto an even grid that ends exactly at the axis
    limits, so the contour map fills the panel to its edges (values beyond the outer centres are held constant)."""
    xn, yn = np.linspace(lo_x, hi_x, factor * len(x)), np.linspace(lo_y, hi_y, factor * len(y))
    Zx = np.array([np.interp(xn, x, row) for row in Z])
    return np.array([np.interp(yn, y, col) for col in Zx.T]).T, xn, yn


CMAP_CHECKED = []


def cmap_ready(gp, idx):
    """True if the layer's colour map can be accessed (one quiet check instead of 20 Origin error lines)."""
    op.lt_exec(f"win -a {gp.name};")
    op.lt_exec(f"page.active = {idx + 1};")
    try:
        v = float(op.lt_float("layer.cmap.numColors"))
    except Exception:                                   # noqa: BLE001  (cannot tell: let set_cmap try)
        return True
    return v == v                                       # NaN -> no colour map in this layer


def set_cmap(gp, idx, bounds, colors):
    """Levels at the band edges (0, 1, 2, 5, ..., 99, 100 %) and one fixed colour per level, written level by level
    (layer.cmap.zN / colorN), so the result does not depend on a palette file (Inferno.pal is not in every Origin).
    The colour of level i is taken from the Z value Origin reports for it, so an off-by-one in Origin's level count
    cannot shift the colours.  The first call prints the levels Origin reports."""
    n = len(bounds) - 1
    lt(f"win -a {gp.name}; page.active = {idx + 1};")                      # the heatmap is the layer's only plot
    lt("layer.cmap.linkpal = 0;")                                          # colours are set by hand, not by palette
    lt(f"layer.cmap.numMajorLevels = {n}; layer.cmap.numMinorLevels = 0; layer.cmap.zmin = {bounds[0]:g};"
       f" layer.cmap.zmax = {bounds[-1]:g}; layer.cmap.SetLevels(1);")
    # level by level; no LabTalk dataset (first Origin run: only panel 1 got its colours; re-declaring the dataset is
    # the suspect).  Origin rejects a zN that is not strictly between its neighbours ("invalid value", 2026-10-20:
    # z2 = 0.6827 above the old z3 = 0.667 of the even start levels), so each level is written once the levels stay
    # ascending: the current levels are read (even spacing if Origin does not report them)
    want = [float(v) for v in bounds[:-1]]
    cur = []
    for i in range(n):
        try:
            v = float(op.lt_float(f"layer.cmap.z{i + 1}"))
        except Exception:                               # noqa: BLE001
            v = float("nan")
        cur.append(v if v == v else bounds[0] + i * (bounds[-1] - bounds[0]) / n)
    todo, order = set(range(n)), []
    while todo:
        ok_i = [i for i in sorted(todo) if (i == 0 or cur[i - 1] < want[i]) and (i == n - 1 or want[i] < cur[i + 1])]
        if not ok_i:                                    # cannot happen for ascending bounds; write the rest as before
            ok_i = sorted(todo)
        for i in ok_i:
            cur[i] = want[i]
            order.append(i)
            todo.discard(i)
    lt(" ".join(f"layer.cmap.z{i + 1} = {want[i]:g};" for i in order))
    nc = int(round(op.lt_float("layer.cmap.numColors")))
    zs = [op.lt_float(f"layer.cmap.z{i}") for i in range(1, nc + 1)]
    for i, z in enumerate(zs, 1):
        k = int(np.clip(np.searchsorted(bounds, z + 1e-9, side="right") - 1, 0, n - 1))
        lt(f"layer.cmap.color{i} = {lt_rgb(colors[k])};")
    lt(f"layer.cmap.colorBelow = {lt_rgb(colors[0])}; layer.cmap.colorAbove = {lt_rgb(colors[-1])};"
       f" layer.cmap.colorMiss = {lt_rgb(colors[-1])}; layer.cmap.updateScale();")
    zr = [round(z, 3) for z in zs]
    if not CMAP_CHECKED:
        CMAP_CHECKED.append(1)
        print(f"colour map: Origin reports {nc} levels, z = {zr}; wanted {[float(v) for v in bounds[:-1]]}")
    if nc < n or not np.allclose(zs[:n], bounds[:n], atol=1e-6):
        LOG.append(f"[colour levels] layer {idx + 1}: Origin levels {zr} differ from {[float(v) for v in bounds]}")
    return True


def colour_bar_page(box, bounds, colors):
    """Discrete colour bar as layer 1 of its own page (only a page's first layer took the colour map in Origin):
    a heatmap of the band values, densest band on top, 'outside 99 %' at the bottom, no ticks or labels.  The text
    is added after the merge (colour_bar_text)."""
    n = len(bounds) - 1
    gp = op.new_graph(lname="colour_bar")
    gl = gp[0]
    place_layer(gp, 0, 0.5, 0.5, box[2], box[3])         # final position set after the merge
    vals = (0.5 * (bounds[:-1] + bounds[1:]))[::-1]    # matrix row 0 = bottom
    ms = new_matrix("colour_bar", np.repeat(vals[:, None], 2, axis=1), [0.25, 0.75], [0.5, n - 0.5])
    colormap_plot(gp, 0, gl, ms, bounds, colors)
    set_axes(gl, (0, 1), (0, n))
    lt(f"win -a {gp.name}; page.active = 1; layer.x.showAxes = 3; layer.y.showAxes = 3; layer.x.ticks = 0;"
       " layer.y.ticks = 0; layer.x2.ticks = 0; layer.y2.ticks = 0; " + label_sides(0, 0, 0, 0))
    op.lt_exec("xb.show = 0;")
    op.lt_exec("yl.show = 0;")
    return gp


def colour_bar_text(gp, idx, box, bounds, levels, title):
    """Level values at the band edges and the rotated title, in the colour-bar layer idx of the merged page.  Labels
    are created at a positive position and then moved ('-0.8' after label -a is read as a switch).  Returns the y
    (axis units of the bar layer) below which legends can be stacked."""
    n = len(bounds) - 1
    lt(f"win -a {gp.name}; page.active = {idx + 1};")
    x0, names = 1.35, []
    for k in range(1, n):                               # edge k: between band k-1 (above) and band k (below)
        nm, y = f"cbar{k}", n - k
        lt(f"label -a {x0} {y} -n {nm} {100 * float(levels[k - 1]):g} %;")
        lt(f"{nm}.fsize = {STYLE['cbar_pt']}; {nm}.x = {x0} + {nm}.dx / 2; {nm}.y = {y};")
        names.append(nm)
    w = max(op.lt_float(f"{nm}.dx") for nm in names) if names else 1.0
    unit_cm = box[2]                                    # one x unit of the bar layer in cm (x runs 0..1)
    xt = x0 + w + 0.3 + 0.7 * STYLE["title_pt"] * 0.03528 / unit_cm
    step("colour bar title")(lt, f"label -a {xt} {n / 2} -n cbartitle {title}; cbartitle.rotate = 90;"
                                 f" cbartitle.fsize = {STYLE['title_pt']}; cbartitle.x = {xt}; cbartitle.y = {n / 2};")
    return -0.8


def bar_text(gp, idx, name, text, y_top):
    """Multi-line text in the colour-bar layer, left-aligned with the bar, top edge at y_top; returns its bottom."""
    lt(f"win -a {gp.name}; page.active = {idx + 1};")
    text = text.replace("=", "\\(61)")                 # the tau0 legend with '=' failed twice (likely read as an assignment)
    lt(f"label -s -sa -a 0 1 -n {name} {text};")
    lt(f"{name}.fsize = {STYLE['cbar_pt']}; {name}.x = {name}.dx / 2; {name}.y = {y_top} - {name}.dy / 2;")
    return y_top - op.lt_float(f"{name}.dy") - 0.6


def line_plot(gl, wks, colx, coly, color, width, dash=0, symbol=False, size=None, interior=0):
    p = gl.add_plot(wks, coly=coly, colx=colx, type="s" if symbol else "l")
    p.color = color
    if symbol:                                          # circle, size in pt; interior 0 solid, 1 open (2 = dot centre:
        p.set_cmd(f"-k 2 -kf {interior} -z {size or STYLE['symbol_size']}")   # hid the data under the fit lines)
    else:
        p.set_cmd(f"-w {int(width * 500)}")              # line width (1 pt = 500)
        if dash:
            p.set_cmd(f"-d {dash}")                      # 1 dash, 2 dot, 3 dash-dot
    return p


def scatter_bands(gl, wks, n_bands, colors, size):
    """One scatter series per posterior-mass band (columns 2b, 2b+1 of wks), outermost band first."""
    for b in range(n_bands, -1, -1):
        if wks.cols <= 2 * b + 1:
            continue
        p = gl.add_plot(wks, coly=2 * b + 1, colx=2 * b, type="s")
        p.color = colors[b]
        p.set_cmd(f"-k 2 -kf 2 -z {size}")              # filled circle, size in pt


def fill_area(gl, wks, colx, coly, color):
    p = gl.add_plot(wks, coly=coly, colx=colx, type="l")
    p.color = color
    lt("set %C -pfm 1;")                                # fill area under the curve (to zero) with the line colour
    return p


REPLAY = {}              # (page j, layer k) -> LabTalk formatting, applied again after merge_graph


def panel_axes(gp, idx, kx, ky, px, py, xside, yside, j=None):
    """Axes of a 2D panel in three layers: the panel itself (linear in decades) shows no axes; a tick layer over it
    with true log10 axes (10^lo .. 10^hi, Origin's own log minor ticks) for the log10_ parameters, linear axes
    otherwise, white major + minor ticks on all four sides; a frame layer on top with the black axis lines.  Log axes
    get 10^n labels and their title as text in the panel layer after the merge: returns those tasks."""
    lx, ly = kx.startswith("log10_"), ky.startswith("log10_")
    panel_s = ("layer.x.showAxes = 0; layer.y.showAxes = 0; layer.x.ticks = 0; layer.y.ticks = 0; layer.x2.ticks = 0;"
               " layer.y2.ticks = 0; " + label_sides(0, 0, 0, 0))
    lt(f"win -a {gp.name}; page.active = {idx + 1}; " + panel_s)
    op.lt_exec("xb.show = 0;")
    op.lt_exec("yl.show = 0;")
    lt("layer.unit = 3;")
    geo = [op.lt_float(f"layer.{p}") for p in ("left", "top", "width", "height")]
    frame = (f"layer.fixed = 1; layer.factor = {STYLE['layer_factor']}; layer.unit = 3; layer.left = {geo[0]};"
             f" layer.top = {geo[1]}; layer.width = {geo[2]}; layer.height = {geo[3]};")
    # tick layer
    lt(f"win -a {gp.name}; layer -n;")
    k_tick = int(op.lt_float('page.nlayers'))
    lt(f"page.active = {k_tick}; " + frame)
    tick_s = "layer.x.link = 0; layer.y.link = 0;"
    for ax, log, p_ in (("x", lx, px), ("y", ly, py)):
        lo, hi = float(p_["lo"]), float(p_["hi"])
        if log:
            tick_s += (f" layer.{ax}.type = 2; layer.{ax}.from = {10 ** lo:.6g}; layer.{ax}.to = {10 ** hi:.6g};"
                       f" layer.{ax}.inc = 1; layer.{ax}.minorTicks = 8;")
        else:
            st = linear_step(lo, hi, int(p_.get("nmax", 6)))   # Origin's numbers on the drawn ticks (tick_segments)
            tick_s += (f" layer.{ax}.type = 1; layer.{ax}.from = {lo:.6g}; layer.{ax}.to = {hi:.6g};"
                       f" layer.{ax}.inc = {st:g}; layer.{ax}.minorTicks = 1; layer.{ax}.firstTick = "
                       f"{math.ceil(lo / st - 1e-9) * st:.6g};")
    tick_s += (" layer.x.showAxes = 3; layer.y.showAxes = 3; layer.x.ticks = 5; layer.x2.ticks = 5; layer.y.ticks = 5;"
               " layer.y2.ticks = 5; layer.x.color = color(white); layer.y.color = color(white);"
               " layer.x2.color = color(white); layer.y2.color = color(white);"
               " layer.x.ticks = 0; layer.x2.ticks = 0; layer.y.ticks = 0; layer.y2.ticks = 0;")   # ticks: panel lines
    if not (lx and ly):                                  # linear axes keep Origin's numeric labels
        # the far-side labels (x2 top, y2 right) too: they follow the white axis colour otherwise (1-trap modes
        # corner, 2026-10-20: top / right numbers invisible)
        tick_s += (f" layer.x.label.color = color(black); layer.y.label.color = color(black);"
                   f" layer.x.label.pt = {STYLE['tick_pt']}; layer.y.label.pt = {STYLE['tick_pt']};"
                   f" layer.x.label.rotate = {float(px.get('rotate', STYLE['x_label_rotate'])):g};")
        if xside == "top" and not lx:
            tick_s += (f" layer.x2.label.color = color(black); layer.x2.label.pt = {STYLE['tick_pt']};"
                       f" layer.x2.label.rotate = {top_rotate():g};")
        if yside == "right" and not ly:
            tick_s += f" layer.y2.label.color = color(black); layer.y2.label.pt = {STYLE['tick_pt']};"
    tick_s += " " + label_sides(int(xside == "bottom" and not lx), int(xside == "top" and not lx),
                                int(yside == "left" and not ly), int(yside == "right" and not ly))   # last
    lt(tick_s)
    t = STYLE["title_pt"]
    if xside in ("bottom", "top") and not lx:            # linear axes: Origin's numeric labels and titles
        o = "xb" if xside == "bottom" else "xt"
        lt(f'label -{o} {px["label_origin"]}; {o}.fsize = {t};')
    if yside == "left" and not ly:                       # (right: text after the merge, see decade_labels)
        lt(f'label -yl {py["label_origin"]}; yl.fsize = {t};')
    # frame layer: black axis lines, no ticks, no labels
    lt(f"win -a {gp.name}; layer -n;")
    k_frame = int(op.lt_float('page.nlayers'))
    frame_s = ("layer.x.link = 0; layer.y.link = 0; layer.x.showAxes = 3; layer.y.showAxes = 3; layer.x.ticks = 0;"
               " layer.y.ticks = 0; layer.x2.ticks = 0; layer.y2.ticks = 0; layer.x.color = color(black);"
               " layer.y.color = color(black); layer.x2.color = color(black); layer.y2.color = color(black); "
               + label_sides(0, 0, 0, 0))
    lt(f"page.active = {k_frame}; " + frame + " " + frame_s)
    if j is not None:
        REPLAY.update({(j, idx + 1): panel_s, (j, k_tick): tick_s, (j, k_frame): frame_s})
    tasks = []
    if xside in ("bottom", "top") and lx:
        tasks.append(("x", px, py, geo, xside, px["label_origin"]))
    if yside in ("left", "right") and ly:
        tasks.append(("y", py, px, geo, yside, py["label_origin"]))
    elif yside == "right":                               # linear: Origin's yr title landed far off the panel
        tasks.append(("y", py, px, geo, yside, py["label_origin"], True))
    return tasks


def linear_step(lo, hi, nmax=6):
    """Major tick step of a linear axis: the smallest of 1, 2, 2.5, 5 x 10^k with at most nmax major ticks (the old
    10^floor(log10(span / 4)) gave 0.01 on a 0.24 eV QFLS axis: 24 majors, a comb of ticks).  An axis dict may carry
    its own 'nmax' (the density axes of the 1D panels: 3)."""
    span = abs(hi - lo) or 1.0
    base = 10 ** math.floor(math.log10(span))
    for m in (0.1, 0.2, 0.25, 0.5, 1.0, 2.0, 2.5, 5.0):
        if span / (m * base) <= nmax:
            return m * base
    return 10 * base


def tick_segments(p_, q_, geo_len_cm, log, side_lo=True):
    """White tick marks of one axis as NaN-separated segments in panel coordinates (decades): major ticks at every
    decade, 8 minor ticks at log10(2..9) in between (linear axes: Origin-like even ticks); on both sides of the panel.
    Returns (along, across) arrays; geo_len_cm is the panel size along the other axis (tick length conversion)."""
    lo, hi = float(p_["lo"]), float(p_["hi"])
    qlo, qhi = float(q_["lo"]), float(q_["hi"])
    per_cm = (qhi - qlo) / geo_len_cm
    if log:
        major = [float(n) for n in range(int(math.ceil(lo)), int(math.floor(hi)) + 1)]
        minor = [n + math.log10(m) for n in range(int(math.floor(lo)), int(math.ceil(hi)) + 1) for m in range(2, 10)]
    else:
        step_ = linear_step(lo, hi, int(p_.get("nmax", 6)))
        major = list(np.arange(math.ceil(lo / step_ - 1e-9) * step_, hi + 1e-12, step_))
        minor = [m + step_ / 2 for m in major]
    a, b = [], []
    for vals, L in ((major, 0.14), (minor, 0.07)):
        for v in vals:
            if lo <= v <= hi:
                for base, sgn in ((qlo, 1), (qhi, -1)):
                    a += [v, v, np.nan]
                    b += [base, base + sgn * L * per_cm, np.nan]
    return np.array(a), np.array(b)


def top_rotate():
    v = STYLE.get("x_label_rotate_top")
    return float(STYLE["x_label_rotate"] if v is None else v)


def decade_labels(gp, idx, ax, p_, q_, geo, side, title, linear=False):
    """10^n tick labels of the log axis 'ax' and its title, as text in panel layer idx (coordinates in decades).  q_ is
    the other axis, along which the offsets run; geo the panel frame in cm.  Labels are created at (0, 0) and then
    moved: a negative number after 'label -a' is read as a switch.  linear=True: a linear axis whose numbers Origin
    draws; only the title is placed, beyond the estimated width of those numbers."""
    lo, hi = float(p_["lo"]), float(p_["hi"])
    qlo, qhi = float(q_["lo"]), float(q_["hi"])
    span = hi - lo
    far = side in ("top", "right")
    rot = (top_rotate() if far else float(STYLE["x_label_rotate"])) if ax == "x" else 0.0
    if ax == "x" and "rotate" in p_:                     # the axis' own angle (PPC panels of build_corner_modes: 0)
        rot = float(p_["rotate"])
    if linear and not title:
        return
    if linear:
        per_cm = (qhi - qlo) / (geo[3] if ax == "x" else geo[2])
        gap = STYLE["label_gap_cm"] * per_cm
        pt_cm = STYLE["tick_pt"] * 0.03528
        nch = max(len(f"{v:.2f}".rstrip("0").rstrip(".")) for v in (lo, hi, (lo + hi) / 2)) + 1
        ext = (pt_cm if ax == "x" else 0.56 * pt_cm * nch) * per_cm
        th = 1.25 * STYLE["title_pt"] * 0.03528 * per_cm
        pos = (qhi if far else qlo) + (1 if far else -1) * (gap + ext + gap + th / 2)
        nm = f"ttl{ax}"
        lt(f"win -a {gp.name}; page.active = {idx + 1};")
        lt(f"label -a 0 0 -n {nm} {title}; {nm}.fsize = {STYLE['title_pt']}; {nm}.attach = 2;")
        if ax == "x":
            lt(f"{nm}.x = {(lo + hi) / 2}; {nm}.y = {pos};")
        else:
            lt(f"{nm}.rotate = 90; {nm}.x = {pos}; {nm}.y = {(lo + hi) / 2};")
        return
    # tilted labels reach past their tick (left below the panel, right above it): keep them off that panel edge, or
    # they run into the neighbouring panel's last label
    e_lo = e_hi = 0.01 * span
    if rot:
        wide = STYLE["x_label_edge_cm"] * span / geo[2]
        e_lo, e_hi = (e_lo, max(e_hi, wide)) if far else (max(e_lo, wide), e_hi)
    elif ax == "x":                                      # flat labels are centred on the tick: half a label off both edges
        e_lo = e_hi = max(e_lo, STYLE["x_label_edge_cm"] * span / geo[2])
    def nums(k):
        return [n for n in range(int(math.ceil(lo / k)) * k, int(math.floor(hi)) + 1, k) if lo + e_lo <= n <= hi - e_hi]
    k = max([k for k in (1, 2, 3, 4, 5, 6) if len(nums(k)) >= 3] or [1])         # at least 3 labels
    ns = [(n, f"10\\+({n})") for n in nums(k)]
    # fewer than 3 decades inside: m x 10^n labels too, decades first, then the multiples in order, each kept only if it
    # is at least one label (x: ~1 cm, y: ~0.45 cm) from every label already kept; the first set giving 3 labels wins
    min_sep = (1.0 if ax == "x" else 0.45) * span / (geo[2] if ax == "x" else geo[3])
    for mult in ((3,), (2, 5), (2, 3), (2, 4, 6, 8), (2, 3, 4, 5, 6, 7, 8, 9)):
        if len(ns) >= 3:
            break
        cand = [(n + math.log10(m), f"{m}×10\\+({n})") for m in mult
                for n in range(int(math.floor(lo)) - 1, int(math.ceil(hi)) + 1)
                if lo + e_lo <= n + math.log10(m) <= hi - e_hi]
        keep = [(n, f"10\\+({n})") for n in nums(1)]
        for c_ in cand:
            if all(abs(c_[0] - q[0]) >= min_sep for q in keep):
                keep.append(c_)
        ns = sorted(keep)
    want = STYLE.get("x_label_count") if ax == "x" else None
    if want and len(ns) > want and all(t.startswith("10") for _, t in ns):   # decades only: exactly 'want' of them,
        dec = nums(1)                                    # every k-th one, the largest k that gives that many
        sel = next((d for k_ in range(6, 0, -1) for off in range(k_)
                    for d in [[n for n in dec if (n - off) % k_ == 0]] if len(d) == want), None)
        if sel:
            ns = [(n, f"10\\+({n})") for n in sel]
    if ax == "x" and p_.get("ticks_x") is not None:      # decades given by hand (build_corner_modes_1trap.py)
        ns = [(int(n), f"10\\+({int(n)})") for n in p_["ticks_x"] if lo <= n <= hi]
    per_cm = (qhi - qlo) / (geo[3] if ax == "x" else geo[2])      # other-axis units per cm
    gap = STYLE["label_gap_cm"] * per_cm
    lt(f"win -a {gp.name}; page.active = {idx + 1};")
    ext = 0.0
    for i, (n, text) in enumerate(ns):
        nm = f"dl{ax}{i}"
        lt(f"label -a 0 0 -n {nm} {text}; {nm}.fsize = {STYLE['tick_pt']}; {nm}.attach = 2;")
        if rot:                                          # tilted: the text ends at the tick (bottom) / starts there (top)
            lt(f"{nm}.rotate = {rot:g};")
        e = op.lt_float(f"{nm}.dy" if ax == "x" else f"{nm}.dx")
        off = gap + e / 2
        pos = qhi + off if far else qlo - off
        if ax == "x":
            sh = (1 if far else -1) * op.lt_float(f"{nm}.dx") / 2 if rot else 0.0   # bounding box corner on the tick
            lt(f"{nm}.x = {n + sh}; {nm}.y = {pos};")
        else:
            lt(f"{nm}.x = {pos}; {nm}.y = {n};")
        ext = max(ext, e)
    th = 1.25 * STYLE["title_pt"] * 0.03528 * per_cm               # title line height in other-axis units
    off = gap + ext + gap + th / 2
    pos = qhi + off if far else qlo - off
    nm = f"ttl{ax}"
    lt(f"label -a 0 0 -n {nm} {title}; {nm}.fsize = {STYLE['title_pt']}; {nm}.attach = 2;")
    if ax == "x":
        lt(f"{nm}.x = {(lo + hi) / 2}; {nm}.y = {pos};")
    else:
        lt(f"{nm}.rotate = 90; {nm}.x = {pos}; {nm}.y = {(lo + hi) / 2};")


ET_MINOR = {0.05: 4, 0.1: 1, 0.2: 1, 0.25: 4, 0.3: 2, 0.4: 3, 0.5: 4, 1.0: 4}   # major step (eV) -> minor ticks


def et_ticks(e_lo, e_hi, width_cm):
    """Major step of the E_t - E_g axis: the smallest round step whose tick labels (width estimated at tick_pt, Arial)
    keep 0.25 cm between them; first major tick on a multiple of the step.  The 0.25 eV steps of the first Origin run
    gave '-0.75-0.50-0.25' run together on a 2.4 cm panel."""
    span = abs(e_hi - e_lo)
    pt_cm = STYLE["tick_pt"] * 0.03528
    inc = 1.0
    for v in ET_MINOR:
        dec = len(f"{v:g}".partition(".")[2])
        w = 0.56 * pt_cm * (len(f"{-max(abs(e_lo), abs(e_hi)):.{dec}f}") - 0.5)
        if v / span * width_cm >= w + 0.25:
            inc = v
            break
    return inc, ET_MINOR[inc], math.ceil(min(e_lo, e_hi) / inc - 1e-9) * inc


def et_axis(gp, idx, where, lo, hi, log10_nc, kT_ln10, title):
    """Extra axis E_t - E_g (eV) on panel layer idx (x in decades of n1):
        E_t - E_g = kT ln(n1 / N_c) = kT ln10 (log10 n1 - log10 N_c),
    linear in the panel's decade axis.  So the axis is a LINEAR axis in its own layer, linked to the panel layer by
    this formula (layadd, custom X link: X1 = kT ln10 (X1 - log10 N_c), X2 likewise): its range always matches the n1
    axis, also after rescaling the panel in Origin.  If the formula link fails, the converted range is set directly.
    Ticks point outward (inward black ticks were invisible on the dark maps).  Call it on the final (merged) page: a
    link made on a panel page would point to a wrong layer number after merge_graph."""
    e_lo, e_hi = (lo - log10_nc) * kT_ln10, (hi - log10_nc) * kT_ln10
    side = 2 if where == "top" else 1
    lt(f"win -a {gp.name}; page.active = {idx + 1}; layer.unit = 3;")
    geo = [op.lt_float(f"layer.{p}") for p in ("left", "top", "width", "height")]
    n0 = int(op.lt_float("page.nlayers"))
    f1, f2 = (f"{kT_ln10:.10g}*({x}-({log10_nc:.10g}))" for x in ("X1", "X2"))
    linked = lt(f'layadd userdef:=1 bottom:={int(side == 1)} top:={int(side == 2)} left:=0 right:=0'
                f' linkto:={idx + 1} xaxis:=custom x1:="{f1}" x2:="{f2}" yaxis:=straight;')
    linked = linked and int(op.lt_float("page.nlayers")) == n0 + 1
    if not linked:                                       # no formula link: unlinked layer with the converted range
        LOG.append(f"[E_t - E_g axis] formula link (layadd) failed on layer {idx + 1}; range set directly")
        lt(f"win -a {gp.name}; page.active = {idx + 1}; layer -n X;")
    k = int(op.lt_float("page.nlayers"))
    lt(f"page.active = {k}; layer.fixed = 1; layer.factor = {STYLE['layer_factor']}; layer.unit = 3;"
       f" layer.left = {geo[0]:.3f}; layer.top = {geo[1]:.3f}; layer.width = {geo[2]:.3f}; layer.height = {geo[3]:.3f};"
       " layer.x.type = 1;")
    if linked:
        op.lt_exec("doc -uw;")
        got = (op.lt_float("layer.x.from"), op.lt_float("layer.x.to"))
        if max(abs(got[0] - e_lo), abs(got[1] - e_hi)) > 1e-4:
            LOG.append(f"[E_t - E_g axis] formula link gave {got[0]:.4f} .. {got[1]:.4f} eV instead of "
                       f"{e_lo:.4f} .. {e_hi:.4f}; link removed, range set directly")
            linked = False
    if not linked:
        lt(f"layer.x.link = 0; layer.x.from = {e_lo:.6f}; layer.x.to = {e_hi:.6f};")
    inc, minor, first = et_ticks(e_lo, e_hi, geo[2])
    lt(f"layer.x.inc = {inc}; layer.x.minorTicks = {minor}; layer.x.showAxes = {side}; layer.y.showAxes = 0;"
       " layer.x.ticks = 10; layer.x2.ticks = 10; layer.x.color = color(black); layer.x2.color = color(black);")
    step("first tick")(lt, f"layer.x.firstTick = {first:.6g};")
    step("tick label size")(lt, f"layer.x.label.pt = {STYLE['tick_pt']}; layer.x2.label.pt = {STYLE['tick_pt']};"
                                f" layer.x.label.font = font({STYLE['font']}); layer.x2.label.font = font({STYLE['font']});")
    lt(label_sides(int(side == 1), int(side == 2), 0, 0))
    title = re.sub(r"\\-(?!\()", "-", title)             # '\-' without '(' is a minus, not a subscript
    o = "xt" if where == "top" else "xb"
    lt(f'label -{o} {title}; {o}.text$ = "{title}"; {o}.show = 1; {o}.fsize = {STYLE["title_pt"]};')
    print(f"E_t - E_g axis on layer {idx + 1}: {e_lo:.3f} .. {e_hi:.3f} eV, step {inc} eV"
          f" ({'formula link' if linked else 'fixed range'})")


def et_axes_of(G):
    """Layout rows (index -> 'top' | 'bottom') that get an E_t - E_g axis: those named in the file's layout, plus the
    topmost panel of every n1 column the file names none for (slice files written before 19/10/2026 had it only on
    the n1,1 column)."""
    out, cols = {}, {}
    for li, row in enumerate(G["layout"]):
        kind, kx, xside, extra = (_s(row[k]) for k in ("kind", "kx", "xside", "extra"))
        if kind not in ("2d", "1d") or "log10_Nc" not in G["params"].get(kx, {}):
            continue
        if extra.endswith("Et-Eg"):
            out[li] = extra.split(":")[0]
        cols.setdefault(kx, []).append((int(row["row"]), li, xside, extra.endswith("Et-Eg")))
    for kx, rows in cols.items():
        if not any(r[3] for r in rows):
            r, li, xside, _ = min(rows)
            if xside != "top":
                out[li] = "top"
    return out


def draw_et(gp, idx, where, px):
    """et_axis for the n1 parameter px (attributes lo, hi, log10_Nc, kT_ln10, label_origin_et of the file)."""
    et_axis(gp, idx, where, float(px["lo"]), float(px["hi"]), float(px["log10_Nc"]), float(px["kT_ln10"]),
            px.get("label_origin_et", "E\\-(t) - E\\-(g) (eV)"))


# ─────────────────────────────────────────────────────────────────────────────
# figure
# ─────────────────────────────────────────────────────────────────────────────
def geometry(layout):
    """Square panels: (left, top, width, height) in cm of grid cell (r, c), spanning rs x cs cells."""
    s = STYLE
    p, g = s["panel_cm"], s["gap_cm"]
    def pos(r, c, rs=1, cs=1):
        return (s["margin_left_cm"] + c * (p + g), s["margin_top_cm"] + r * (p + g), cs * p + (cs - 1) * g,
                rs * p + (rs - 1) * g)
    return pos


def page_height(gp, cm):
    """Page height in cm (the width follows: Origin keeps the page's aspect ratio)."""
    lt(f"win -a {gp.name}; page.height = {cm / 2.54:.4f} * page.resy;")


def page_size_check(gp):
    """Print the page size after pfit2l (page fitted to the layers)."""
    lt(f"win -a {gp.name};")
    got = [op.lt_float(e) * 2.54 for e in ("page.width / page.resx", "page.height / page.resy")]
    print(f"page: {got[0]:.1f} x {got[1]:.1f} cm")


def tag_layers(gp, j, geo):
    """Name every layer of source page j 'M<j>L<k>' and return the final geometry of each (all layers of a page
    share the panel frame geo, in cm): merge_graph lays all layers out on a grid, so each one is put back by name."""
    lt(f"win -a {gp.name};")
    n = int(op.lt_float("page.nlayers"))
    for k in range(1, n + 1):
        lt(f'page.active = {k}; layer.name$ = "M{j}L{k}";')
    return [tuple(geo)] * n


def find_graph(name):
    try:
        return op.find_graph(name)
    except Exception:                                   # noqa: BLE001
        return next(g for g in op.pages("g") if g.name == name)


def merge_pages(pages):
    """merge_graph of all panel pages into one new page (sources deleted, layers re-placed afterwards); fonts are not
    rescaled (scalemode fixed, factor 1)."""
    names = "+char(10)$+".join(f'"{p.name}"' for p in pages)
    cmd = f"merge_graph option:=specified graphs:={names} keep:=0 groupgraph:=0"   # groupgraph 1 relinked the layers
    if not lt(f"{cmd} scalemode:=fixed fixfactor:=1;"):
        print("merge_graph with scale options failed; merging with defaults")
        lt(cmd + ";")
    op.lt_exec('string __mg$ = "%H";')
    return find_graph(op.get_lt_str("__mg$"))


MERGED = {}               # (page j, layer k) -> 0-based layer index on the merged page


def place_merged(gp, geos):
    """Put every merged layer back at the geometry of its source layer (found by the name M<j>L<k>).  Returns the
    merged layer index (0-based) of each source page's layer 1."""
    lt(f"win -a {gp.name};")
    n = int(op.lt_float("page.nlayers"))
    first, seen = {}, 0
    for i in range(1, n + 1):
        lt(f"page.active = {i}; string __ln$ = layer.name$;")
        m = re.fullmatch(r"M(\d+)L(\d+)", op.get_lt_str("__ln$").strip())
        if not m:
            continue
        j, k = int(m.group(1)), int(m.group(2))
        g = geos[j][k - 1]
        lt(f"page.active = {i}; layer.unit = 3; layer.left = {g[0]:.3f}; layer.top = {g[1]:.3f};"
           f" layer.width = {g[2]:.3f}; layer.height = {g[3]:.3f};")
        MERGED[(j, k)] = i - 1
        if k == 1:
            first[j] = i - 1
        seen += 1
    want = sum(len(g) for g in geos)
    print(f"merged page: {n} layers, {seen} placed by name (expected {want})")
    if seen == 0 and n == want:                         # names lost in the merge: assume the merge kept the order
        i = 0
        for j, gj in enumerate(geos):
            first[j] = i
            for g in gj:
                i += 1
                lt(f"page.active = {i}; layer.unit = 3; layer.left = {g[0]:.3f}; layer.top = {g[1]:.3f};"
                   f" layer.width = {g[2]:.3f}; layer.height = {g[3]:.3f};")
        LOG.append("[merge] layer names lost in merge_graph; layers placed in merge order")
    elif seen != want:
        LOG.append(f"[merge] {seen} of {want} layers found by name; layer names after merge_graph differ")
    return first


def cbar_box(layout):
    """[left, top, width, height] of the colour bar in cm: STYLE['cbar'], else right of the grid, over the middle
    60 % of its height (as in the matplotlib preview)."""
    s = STYLE
    if s.get("cbar"):
        return tuple(float(v) for v in s["cbar"])
    nrow, ncol = int(layout["row"].max()) + 1, int(layout["col"].max()) + 1
    p, g = s["panel_cm"], s["gap_cm"]
    grid_w, grid_h = ncol * p + (ncol - 1) * g, nrow * p + (nrow - 1) * g
    return s["margin_left_cm"] + grid_w + s["cbar_gap_cm"], s["margin_top_cm"] + 0.15 * grid_h, s["cbar_w_cm"], 0.6 * grid_h


def build_corner(G, h5_path, figure):
    book = op.new_book("w", lname=f"corner_{figure}_data")
    pages = []                                   # one graph page per panel: only layer 1 of a page takes a colour map
    tasks = {}                                   # page -> 10^n labels / titles to write after the merge
    final = {}                                   # page -> final (left, top, width, height) in cm of all its layers
    pos = geometry(G["layout"])
    C, P, levels = G["colors"], G["params"], G["hpd_levels"]
    n_bands = len(levels)
    bounds = colour_bounds(levels)
    band_cols = G["band_colors"] if G["band_colors"] and len(G["band_colors"]) == n_bands + 1 else BAND_COLORS
    if len(band_cols) != n_bands + 1:
        LOG.append(f"[colours] {n_bands + 1} bands but {len(band_cols)} colours")
    ov = G["overlays"]
    ets = et_axes_of(G)                          # layout row -> 'top' | 'bottom': E_t - E_g axes, drawn after the merge
    et_cells = [pos(int(G["layout"][i]["row"]), int(G["layout"][i]["col"])) for i, w in ets.items() if w == "top"]
    et_pages = {}                                # page -> (where, n1 parameter)
    for li, row in enumerate(G["layout"]):
        r, c, kind = int(row["row"]), int(row["col"]), _s(row["kind"])
        kx, ky, xside, yside, extra = (_s(row[k]) for k in ("kx", "ky", "xside", "yside", "extra"))
        if kind == "empty" or (kind == "curve" and not G["curves"]):
            continue
        lname = f"panel_{kx}__{ky}" if ky else f"panel_{kx}"
        gp = op.new_graph(lname=lname)                   # default-size page; final position set after the merge
        gl, idx = gp[0], 0
        pages.append(gp)
        rs, cs = (int(v) for v in (extra.split("span:")[1] if "span:" in extra else "1x1").split("x")) \
            if kind == "curve" else (1, 1)
        if kind == "curve" and cs > 1:                   # wide fit panel: one panel width free on its left
            c, cs = c + 1, cs - 1
        final[len(pages) - 1] = pos(r, c, rs, cs)
        ncol_ = int(G["layout"]["col"].max()) + 1
        if kind == "curve" and ncol_ >= 6:              # taller simulation panels in the empty upper right
            P_, g_ = STYLE["panel_cm"], STYLE["gap_cm"]
            if cs > 1:                                   # tau(QFLS) / ss-PL: 2 rows tall, cols 3 .. end
                final[len(pages) - 1] = pos(0, 3, 2, ncol_ - 3)
            else:                                        # PL(t): 2 x 2 under it, room for the x title above
                l_, t_, w_, h_ = pos(2, ncol_ - 2, 2, 2)
                t_, h_ = t_ + 0.6 * P_, h_ - 0.6 * P_
                # E_t - E_g axis on top of a panel below-left (n1,2 column): end PL(t) above that axis' labels and
                # title, and move it right so its own x tick labels clear them
                et_t = [et_[1] for et_ in et_cells if et_[1] > t_ and et_[0] + 2 * P_ > l_]
                if et_t and t_ + h_ > min(et_t) - STYLE["et_axis_cm"]:
                    h_ = min(et_t) - STYLE["et_axis_cm"] - t_
                    l_, w_ = l_ + STYLE["et_pl_shift_cm"], w_ - STYLE["et_pl_shift_cm"]
                final[len(pages) - 1] = (l_, t_, w_, h_)
        step("place layer")(place_layer, gp, idx, 0.5, 0.5, *final[len(pages) - 1][2:])
        if kind == "curve":                      # simulation panel of a slice file (tau(QFLS), PL(t), ss-PL)
            step(f"curve panel {kx}")(curve_layer, gp, idx, gl, book, G["curves"], kx,
                                      size_cm=final[len(pages) - 1][2:])
            continue
        px = P[kx]
        if kind == "2d":
            py = P[ky]
            d = G["panels"][f"{kx}__{ky}"]
            dots = BACKGROUND == "samples" and "scatter" in d and G["band_colors"]
            interp = bool(STYLE["interpolate"]) and not dots
            mx, my = d["x"], d["y"]
            if interp:                                   # continuous value in %; Origin's levels make the bands
                Z = np.clip(100.0 * np.asarray(d["hpd"], float), 0.0, 100.0)
                Z = np.where(np.isfinite(Z), Z, 100.0)
                Z, mx, my = regrid(Z, np.asarray(mx, float), np.asarray(my, float), float(px["lo"]), float(px["hi"]),
                                   float(py["lo"]), float(py["hi"]))
            else:
                Z = band_z(d["hpd"], levels)
            if dots:                                     # black background (= the 'outside 99 %' band) under the dots
                Z = np.full_like(Z, 0.5 * (bounds[-2] + bounds[-1]))
            ms = step("matrix")(new_matrix, f"{kx}__{ky}", Z, mx, my)
            ok = ms is not None and step("colour map")(colormap_plot, gp, idx, gl, ms, bounds, band_cols, contour=interp)
            if not ok and interp:                        # contour failed: same panel as a finely resampled heatmap
                op.lt_exec(f"win -cd {gp.name};")
                gp = op.new_graph(lname=lname)
                gl = gp[0]
                pages[-1] = gp
                step("place layer")(place_layer, gp, idx, 0.5, 0.5, *final[len(pages) - 1][2:])
                Zf, fx, fy = regrid(Z, np.asarray(mx, float), np.asarray(my, float), float(px["lo"]), float(px["hi"]),
                                    float(py["lo"]), float(py["hi"]), int(STYLE["upsample"]))
                ms = step("matrix")(new_matrix, f"{kx}__{ky}_hm", Zf, fx, fy)
                ok = ms is not None and step("colour map")(colormap_plot, gp, idx, gl, ms, bounds, band_cols)
                LOG.append(f"[colour map] {kx}__{ky}: contour failed, heatmap {'used' if ok else 'failed too'}")
            if dots:
                sc = d["scatter"]
                b = band_index(sc[:, 2], levels).astype(int)
                cols = []
                for k in range(n_bands + 1):
                    cols += [(f"x band {k}", sc[b == k, 0]), (f"y band {k}", sc[b == k, 1])]
                w = step("sheet")(new_sheet_xy, book, f"samples_{kx}__{ky}", cols)
                if w is not None:
                    step("samples")(scatter_bands, gl, w, n_bands, G["band_colors"], STYLE["sample_symbol_size"])
            contour_sets = ([(m, M["panels"][f"{kx}__{ky}"], M["dash"]) for m, M in G["methods"].items()]
                            if G["methods"] else [("", d, 0)])
            for m, dm, dash_ in contour_sets:
                for s_ in ("sigma2", "sigma1"):
                    if len(dm[s_]):
                        w = step("sheet")(new_sheet_xy, book, f"{s_}_{m}_{kx}__{ky}", [("x", dm[s_][:, 0]), ("y", dm[s_][:, 1])])
                        if w is not None:
                            step("contour line")(line_plot, gl, w, 0, 1, C[s_], STYLE["contour_width"], dash_)
            for reg, dname in G["tau0_dash"].items():           # iso-tau_0 lines (slice corner, n1 - tau_p panels)
                v = d.get(f"tau0_{reg}")
                if v is not None and len(v):
                    w = step("sheet")(new_sheet_xy, book, f"t0_{'pd' if reg == 'photodoping' else 'npd'}_{kx}__{ky}",
                                      [("x", v[:, 0]), ("y", v[:, 1])])
                    if w is not None:
                        step("tau0 line")(line_plot, gl, w, 0, 1, C.get("tau0", COLORS_FALLBACK["tau0"]),
                                          STYLE["overlay_width"], DASH.get(dname, 1))
            lines = []
            for name, dash in (("truth", 0), ("best_fit", 2)):
                vx, vy = px.get(name, np.nan), py.get(name, np.nan)
                if np.isfinite(vx):
                    lines.append((name, dash, [vx, vx], list(py_range(py))))
                if np.isfinite(vy):
                    lines.append((name, dash, list(py_range(px)), [vy, vy]))
            for j, (name, dash, xs, ys) in enumerate(lines):
                w = step("sheet")(new_sheet_xy, book, f"{name}{j}_{kx}__{ky}", [("x", xs), ("y", ys)])
                if w is not None:
                    step("overlay line")(line_plot, gl, w, 0, 1, C[name], STYLE["overlay_width"], dash)
            if np.isfinite(px.get("start", np.nan)) and np.isfinite(py.get("start", np.nan)):
                w = step("sheet")(new_sheet_xy, book, f"start_{kx}__{ky}", [("x", [px["start"]]), ("y", [py["start"]])])
                if w is not None:
                    step("start point")(line_plot, gl, w, 0, 1, C["start"], 1, symbol=True)
            if "eigen_lines" in ov:
                ix, iy = int(px["index"]), int(py["index"])
                L = ov["eigen_lines"]
                xs = np.concatenate([np.append(L[k, :, ix], np.nan) for k in range(L.shape[0])])
                ys = np.concatenate([np.append(L[k, :, iy], np.nan) for k in range(L.shape[0])])
                w = step("sheet")(new_sheet_xy, book, f"eig_{kx}__{ky}", [("x", xs), ("y", ys)])
                if w is not None:
                    step("eigen axes")(line_plot, gl, w, 0, 1, C["eigen_axes"], 0.5)
            step("axes")(set_axes, gl, (px["lo"], px["hi"]), (py["lo"], py["hi"]))
            g_ = final[len(pages) - 1]
            for ax, pa, pb, L in (("x", px, py, g_[3]), ("y", py, px, g_[2])):
                a, b = tick_segments(pa, pb, L, (kx if ax == "x" else ky).startswith("log10_"))
                cols = [("x", a), ("y", b)] if ax == "x" else [("x", b), ("y", a)]
                w = step("sheet")(new_sheet_xy, book, f"ticks{ax}_{kx}__{ky}", cols)
                if w is not None:
                    step("white ticks")(line_plot, gl, w, 0, 1, "#FFFFFF", 0.8)
            tasks[len(pages) - 1] = step("panel axes")(panel_axes, gp, idx, kx, ky, px, py, xside, yside,
                                                       len(pages) - 1) or []
        else:
            if G["diag_mode"] == "slice" and kx in G["diag"]:
                d = G["diag"][kx]
                w = step("sheet")(new_sheet_xy, book, f"slice_{kx}", [("z", d["z"]), ("NN", d["dlogL_nn"]),
                                                                     ("ODE", d["dlogL_ode"]), ("Laplace", d["parabola"])])
                if w is not None:
                    step("slice NN")(line_plot, gl, w, 0, 1, "#000000", STYLE["line_1d_width"])
                    step("slice ODE")(line_plot, gl, w, 0, 2, "#000000", STYLE["line_1d_width"], 1)
                    step("slice Laplace")(line_plot, gl, w, 0, 3, "#808080", STYLE["line_1d_width"], 3)
                step("axes")(set_axes, gl, (px["lo"], px["hi"]), (-8, 1))
                ytitle = r"\g(D)ln \i(L)"
            else:
                d = G["panels"][kx]
                band = d["hpd_band"]
                pdf = d["pdf"]
                w = step("sheet")(new_sheet_xy, book, f"pdf_{kx}", [
                    ("x", d["x"]), ("pdf", pdf), ("2 sigma band", np.where(band >= 1, pdf, np.nan)),
                    ("1 sigma band", np.where(band >= 2, pdf, np.nan))])
                if w is not None:
                    step("2σ band")(fill_area, gl, w, 0, 2, C["sigma2"])
                    step("1σ band")(fill_area, gl, w, 0, 3, C["sigma1"])
                    if not G["methods"]:
                        step("pdf")(line_plot, gl, w, 0, 1, C.get("line_1d", "#000000"), STYLE["line_1d_width"])
                top = float(np.nanmax(pdf))
                for m, M in G["methods"].items():           # combined corner: every sampler, own dash pattern
                    dm = M["panels"][kx]
                    w3 = step("sheet")(new_sheet_xy, book, f"pdf_{m}_{kx}", [("x", dm["x"]), (f"pdf {m}", dm["pdf"])])
                    if w3 is not None:
                        step("pdf")(line_plot, gl, w3, 0, 1, C.get("line_1d", "#000000"), STYLE["line_1d_width"], M["dash"])
                    top = max(top, float(np.nanmax(dm["pdf"])))
                ymax = 1.15 * top if top > 0 else 1.0
                for j, (name, dash) in enumerate((("truth", 0), ("best_fit", 2))):
                    v = px.get(name, np.nan)
                    if np.isfinite(v):
                        w2 = step("sheet")(new_sheet_xy, book, f"{name}_{kx}", [("x", [v, v]), ("y", [0, ymax])])
                        if w2 is not None:
                            step("overlay line")(line_plot, gl, w2, 0, 1, C[name], STYLE["overlay_width"], dash)
                step("axes")(set_axes, gl, (px["lo"], px["hi"]), (0, ymax))
                ytitle = r"PDF (per decade)"
            step("sides")(axis_sides, gp, idx, xside, yside, px["label_origin"], ytitle)
        if li in ets:
            et_pages[len(pages) - 1] = (ets[li], px)
        op.lt_exec(f"win -a {gp.name}; page.active = 1;")
        op.lt_exec("label -r legend;")                  # the auto legend of the page template (1D panels)
    # colour bar (own page), merge all pages into one, put every layer back, then the bar text and the legends
    box = cbar_box(G["layout"])
    cbp = step("colour bar")(colour_bar_page, box, bounds, band_cols)
    if cbp is not None:
        pages.append(cbp)
        final[len(pages) - 1] = box
    geos = [step("tag layers")(tag_layers, p, j, final[j]) or [] for j, p in enumerate(pages)]
    merged = step("merge")(merge_pages, pages)
    if merged is None:
        LOG.append("[merge] panels left as separate graphs")
        for j, (where, px) in et_pages.items():
            step("E_t - E_g axis")(draw_et, pages[j], 0, where, px)
        return pages[0]
    gp = merged
    step("page height")(page_height, gp, STYLE["merged_page_cm"])          # room for the layout in cm
    first = step("place layers")(place_merged, gp, geos) or {}
    for (j, k), sc in REPLAY.items():                   # axis formatting again: merge_graph reset it on layers 2, 3
        if (j, k) in MERGED:
            step("axis format after merge")(lt, f"win -a {gp.name}; page.active = {MERGED[(j, k)] + 1}; {sc}")
    for j, tl in tasks.items():                         # 10^n labels and titles of the log axes (panel layers)
        for t in tl:
            if j in first:
                step("10^n labels")(decade_labels, gp, first[j], *t)
    for j, (where, px) in et_pages.items():             # E_t - E_g axes, formula-linked to the merged panel layers
        if j in first:
            step("E_t - E_g axis")(draw_et, gp, first[j], where, px)
    title = CBAR_TITLES.get(G.get("colour_bar_title", ""), G.get("colour_bar_title", ""))
    if cbp is None or (len(pages) - 1) not in first:
        step("fit page")(lt, f"win -a {gp.name}; pfit2l margin:=0 borderwid:=2;")
        return gp
    cb_idx = first[len(pages) - 1]
    y = step("colour bar text")(colour_bar_text, gp, cb_idx, box, bounds, levels, title)
    y = -0.8 if y is None else y
    names = {0: "solid", 1: "dashed", 2: "dotted", 3: "dash-dot"}
    if G["methods"]:                                     # legend: sampler <-> dash pattern
        lines = []
        for m, M in G["methods"].items():           # 'SMC (4.8 s, log Z = ...)' -> two lines (narrow margin)
            head, _, rest = M["label"].partition(" (")
            lines.append(f"{head}: {names[M['dash']]}")
            if rest:
                lines.append(f"  ({rest}")
            if m == G["fill_method"]:
                lines.append("(colour map + samples)" if BACKGROUND == "samples" else "(colour map)")
        lines += ["1\\g(s) HPD: sky blue", "2\\g(s) HPD: light grey"]
        y = step("legend")(bar_text, gp, cb_idx, "legend_samplers", "\\n".join(lines), y) or y
    if G.get("tau0") and any(k.startswith("tau0_") for d in G["panels"].values() for k in d):
        vals = ", ".join(sorted({f"{t['tau0_s']:.3g}" for t in G["tau0"].values()}))
        dd = {r: names[DASH.get(n, 1)] for r, n in G["tau0_dash"].items()}
        txt = "\\n".join([r"iso-\g(t)\-(0) (green)", rf"\g(t)\-(0) = {vals} s",
                          rf"{dd.get('photodoping', 'dashed')}: \i(N)\-(t) >= 10 \i(n)\-(1)",
                          rf"{dd.get('non_photodoping', 'dash-dot')}: \i(N)\-(t) <= 0.1 \i(n)\-(1)"])
        step("tau0 label")(bar_text, gp, cb_idx, "legend_tau0", txt, y)
    step("fit page")(lt, f"win -a {gp.name}; pfit2l margin:=0 borderwid:=2;")
    step("page size check")(page_size_check, gp)
    return gp


CURVE_TITLES = {"pl_t": ("time (s)", r"PL/PL\-(0)"),
                "tau_t": ("time (s)", r"\i(\g(t))\-(diff) (s)"),
                "tau_qfls": (r"Fermi-level splitting \g(D)\i(E)\-(F) (eV)", r"\i(\g(t))\-(diff) (s)"),
                "conc": (r"\g(D)\i(E)\-(F) (eV)", r"concentration (cm\+(-3))"),
                "rates": (r"\g(D)\i(E)\-(F) (eV)", r"rate (cm\+(-3) s\+(-1))"),
                "sspl": (r"\g(D)\i(E)\-(F) (eV)", r"log\-(10) PLQY")}


def clip(x, y, lo=-np.inf, hi=np.inf, positive=True):
    """Points with lo <= x <= hi (and y > 0 for log axes): the ODE runs to QFLS 0.3 eV / 72 s, far past the data."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y) & (x >= lo) & (x <= hi)
    if positive:
        m &= y > 0
    return x[m], y[m]


def curve_columns(C, name):
    """[(label, x, y, colour, dash, symbol)] of one curve panel from the /curves tree.  Model curves are clipped to the
    x range of the data.  A '_us' suffix (pl_t_us, tau_t_us) gives the time in microseconds."""
    out, d = [], C["data"]
    sets = [k for k in ("best_fit", "truth") if k in C]
    A = C["_attrs"]
    base, us = (name[:-3], 1e6) if name.endswith("_us") else (name, 1.0)
    t_lo, t_hi = float(np.nanmin(d["t"][d["t"] > 0])), float(np.nanmax(d["t"]))
    q_lo, q_hi = float(np.nanmin(d["qfls"])) - 0.01, float(np.nanmax(d["qfls"])) + 0.01
    if base == "tau_qfls":
        out.append(("data", *clip(d["qfls"], d["tau_diff"]), CURVE_COLORS["data"], 0, True))
        for k in sets:
            lab = A[k].get("label", k)
            out.append((f"{lab} NN (eps_rms {A[k]['eps_rms_nn']:.1e})", *clip(C[k]["nn"]["qfls"], C[k]["nn"]["tau_diff"], q_lo, q_hi),
                        CURVE_COLORS[k], 0, False))
            out.append((f"{lab} ODE (eps_rms {A[k]['eps_rms_ode']:.1e})", *clip(C[k]["ode"]["qfls_eV"], C[k]["ode"]["tau_diff_s"], q_lo, q_hi),
                        CURVE_COLORS["ode"], 1, False))
    elif base in ("pl_t", "tau_t"):
        yd, yn, yo = ("pl", "pl", "pl_norm") if base == "pl_t" else ("tau_diff", "tau_diff", "tau_diff_s")
        x, y = clip(d["t"], d[yd])
        out.append(("data", x * us, y, CURVE_COLORS["data"], 0, True))
        for k in sets:
            lab = A[k].get("label", k)
            x, y = clip(C[k]["nn"]["t"], C[k]["nn"][yn], t_lo, t_hi)
            out.append((f"{lab} NN", x * us, y, CURVE_COLORS[k], 0, False))
            x, y = clip(C[k]["ode"]["time_s"], C[k]["ode"][yo], t_lo, t_hi)
            out.append((f"{lab} ODE", x * us, y, CURVE_COLORS["ode"], 1, False))
    elif base == "conc":
        cols = {"n_cm3": "#1F77B4", "p_cm3": "#D62728"}
        for k in sets:
            o = C[k]["ode"]
            for c_ in [c for c in o if c in cols or c.startswith("nt_")]:
                out.append((f"{c_} {k}", *clip(o["qfls_eV"], o[c_]), cols.get(c_, "#2CA02C"), 0 if k == "best_fit" else 1, False))
    elif base == "rates":
        cols = {"e_capture": "#1F77B4", "e_emission": "#17BECF", "h_capture": "#D62728", "h_emission": "#FF7F0E"}
        nice = {"e_capture": "e trapping", "e_emission": "e detrapping", "h_capture": "h trapping", "h_emission": "h detrapping"}
        for k in sets:
            o = C[k]["ode"]
            for c_ in [c for c in o if c.rsplit("_", 1)[0] in cols]:
                b_ = c_.rsplit("_", 1)[0]
                out.append((f"{nice[b_]} {c_.rsplit('_', 1)[1]} {k}", *clip(o["qfls_eV"], o[c_]), cols[b_],
                            0 if k == "best_fit" else 1, False))
    elif base == "sspl":
        g0 = C[sets[0]]["sspl"]
        out.append(("ss-PL data", g0["x_exp"], g0["y_obs"], CURVE_COLORS["data"], 0, True))
        for k in sets:
            cy = C[k]["sspl"]["curve_y"]
            cy = np.log10(cy) if C.get("_sspl_mode", "").startswith("plqy") else cy
            out.append((f"{A[k].get('label', k)} ss-PL model", C[k]["sspl"]["curve_x"], cy, CURVE_COLORS[k], 0, False))
    return out


def thin_points(x, y, logx, logy, size_cm, step_cm):
    """Indices of the points kept when one symbol is drawn per step_cm along the curve, in plot coordinates (decades
    on log axes, the data range spread over the panel size in cm): the 256 measured points would merge into a band."""
    fx = np.log10(x) if logx else np.asarray(x, float)
    fy = np.log10(y) if logy else np.asarray(y, float)
    u = (fx - fx.min()) / max(np.ptp(fx), 1e-300) * size_cm[0]
    v = (fy - fy.min()) / max(np.ptp(fy), 1e-300) * size_cm[1]
    s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(u), np.diff(v)))])
    keep, last = [0], 0.0
    for i in range(1, len(s)):
        if s[i] - last >= step_cm:
            keep.append(i)
            last = s[i]
    if keep[-1] != len(s) - 1:
        keep.append(len(s) - 1)
    return np.array(keep)


def curve_layer(gp, idx, gl, book, C, name, titles=True, template=False, size_cm=(10.0, 5.0)):
    """Draw one curve panel into layer idx.  Own panel: log axes as in the paper figure + titles.  Template layer
    (template=True): the template's axis types, titles and annotations are kept; only the plots are added.
    Measured data: open circles under the fit lines, thinned to one per STYLE['data_spacing_cm'] along the curve
    (the sheet keeps all points in its first two columns); size_cm = panel width, height."""
    base = name[:-3] if name.endswith("_us") else name
    logx, logy = base in ("pl_t", "tau_t"), base != "sspl"
    for j, (lab, x, y, col, dash, sym) in enumerate(curve_columns(C, name)):
        if sym and base in ("pl_t", "tau_t", "tau_qfls") and len(x) > 1:
            k = thin_points(x, y, logx, logy, size_cm, STYLE["data_spacing_cm"])
            w = step("sheet")(new_sheet_xy, book, f"{name}_{j}", [("x", x), (lab, y), ("x plotted", x[k]),
                                                                 (f"{lab} plotted", y[k])])
            if w is not None:
                step("data")(line_plot, gl, w, 2, 3, col, 1, symbol=True, size=STYLE["fit_data_symbol_size"], interior=1)
            continue
        w = step("sheet")(new_sheet_xy, book, f"{name}_{j}", [("x", x), (lab, y)])
        if w is not None:
            step("curve")(line_plot, gl, w, 0, 1, col, STYLE["line_1d_width"], dash, symbol=sym,
                          size=STYLE["fit_data_symbol_size"], interior=1 if sym else 0)
    lt(f"win -a {gp.name}; page.active = {idx + 1};")
    if not template:
        lt(f"layer.x.type = {2 if logx else 1}; layer.y.type = {2 if logy else 1};"
           " layer.x.showAxes = 3; layer.y.showAxes = 3; layer.x.ticks = 5; layer.y.ticks = 5;")
    step("rescale")(gl.rescale)
    if not template:
        step("curve tick labels")(lt, f"layer.x.label.pt = {STYLE['tick_pt']}; layer.y.label.pt = {STYLE['tick_pt']};"
                                      f" layer.x.label.font = font({STYLE['font']}); layer.y.label.font = font({STYLE['font']});")
        for ax, log in (("x", logx), ("y", logy)):
            if log:                                      # 10^n labels (format 2, as in your paper template), <= 4 decades
                lo, hi = (op.lt_float(f"layer.{ax}.{e}") for e in ("from", "to"))
                span = math.log10(hi / lo) if lo > 0 and hi > 0 else 1.0
                k = next((k for k in (1, 2, 3, 4, 5, 6, 10) if span / k <= 4), 20)
                step("10^n format")(lt, f"layer.{ax}.label.numFormat = 2; layer.{ax}.inc = {k};")
        # same tick options on bottom/top and left/right (the 'use same options' box of the Ticks tab, set per axis)
        step("thin ticks")(lt, " ".join(f"layer.{a}.thickness = 0.8; layer.{a}.tickthickness = 0.8;"
                                         f" layer.{a}.ticklength = 4; layer.{a}.mticklength = 2; layer.{a}.ticks = 5;"
                                         f" layer.{a}.mtickthickness = 0.8;" for a in ("x", "y", "x2", "y2")))
        step("same opposite ticks")(lt, "layer.x2.minorTicks = layer.x.minorTicks; layer.y2.minorTicks = layer.y.minorTicks;"
                                        " layer.x2.inc = layer.x.inc; layer.y2.inc = layer.y.inc;")
        lt(label_sides(1, 0, 1, 0))
    if titles:
        xt, yt = CURVE_TITLES[base]
        if name.endswith("_us"):
            xt = r"time (\g(m)s)"
        step("curve titles")(lt, f'label -xb {xt}; label -yl {yt}; xb.fsize = {STYLE["title_pt"]}; yl.fsize = {STYLE["title_pt"]};')
    step("legend")(lt, "legend -r;")
    if not template:
        step("legend size")(lt, f"legend.fsize = {STYLE['cbar_pt']};")


def layer_check(gp, n_lay):
    """Per layer of a template page after plotting: number of plots, axis ranges and types, links (to find out why a
    layer stays empty)."""
    out = []
    for i in range(1, n_lay + 1):
        lt(f"win -a {gp.name}; page.active = {i};")
        f = {k: op.lt_float(f"layer.{k}") for k in ("x.from", "x.to", "x.type", "y.from", "y.to", "y.type", "link",
                                                    "x.link", "y.label.numFormat", "y.labelSubtype")}
        try:
            n_pl = len(gp[i - 1].plot_list())
        except Exception:                               # noqa: BLE001
            n_pl = "?"
        out.append(f"layer {i}: {n_pl} plots, x {f['x.from']:.3g}..{f['x.to']:.3g} (type {f['x.type']:g}),"
                   f" y {f['y.from']:.3g}..{f['y.to']:.3g} (type {f['y.type']:g}), linked to layer {f['link']:g},"
                   f" x link {f['x.link']:g}, y label format {f['y.label.numFormat']:g} / {f['y.labelSubtype']:g}")
    return out


def plain_panels(name, panels, book, C):
    """Fallback for a missing template: one page, the panels side by side, log axes and titles as in the preview."""
    gp = op.new_graph(lname=name)
    n = len(panels)
    w = (100 - 12 - 6 - (n - 1) * 10) / n
    for j, pn in enumerate(panels):
        gl = gp[0] if j == 0 else add_layer(gp)
        place_layer(gp, j, 12 + j * (w + 10), 12, w, 70, unit=1)
        curve_layer(gp, j, gl, book, C, pn, titles=True)
    return gp


def layer_titles(gp, n_lay):
    """Axis titles of every layer of a template page, to check the TEMPLATES mapping."""
    out = []
    for i in range(1, n_lay + 1):
        lt(f"win -a {gp.name}; page.active = {i};")
        t = []
        for o in ("xb", "yl", "yr"):                    # op.lt_exec, not lt: a missing title is no error
            op.lt_exec('string __t$ = "";')
            op.lt_exec(f"__t$ = {o}.text$;")
            t.append(op.get_lt_str("__t$"))
        out.append(f"layer {i}: x '{t[0]}', y '{t[1] or t[2]}'")
    return out


def build_template_graphs(G, h5_path):
    """Worksheets with the column names of the paper export + one graph per template in TEMPLATES."""
    C = G["curves"]
    book = op.new_book("w", lname="simulations_best_fit_truth")
    for k in [k for k in ("best_fit", "truth") if k in C]:      # paper export columns (fitting/trpl_fitting.py)
        o = C[k]["ode"]
        cols = [("time_tr_s", o["time_s"]), ("pl_tr", o["pl_norm"]), ("qfls_tr_eV", o["qfls_eV"]), ("tau_tr_s", o["tau_diff_s"]),
                ("n_tr_1/cm3", o["n_cm3"]), ("p_tr_1/cm3", o["p_cm3"])]
        cols += [(f"nt_tr_{c.split('_')[1]}_1/cm3", o[c]) for c in o if c.startswith("nt_")]
        for base, nm in (("e_capture", "e_trapping"), ("e_emission", "e_detrapping"), ("h_capture", "h_trapping"), ("h_emission", "h_detrapping")):
            cols += [(f"{nm}_tr_trap{c.rsplit('_', 1)[1]}_1/cm3/s", o[c]) for c in o if c.rsplit("_", 1)[0] == base]
        step("sheet")(new_sheet_xy, book, f"trpl_sim_{k}", cols)
        nn = C[k]["nn"]
        step("sheet")(new_sheet_xy, book, f"nn_sim_{k}", [("time_nn_s", nn["t"]), ("pl_nn", nn["pl"]),
                                                          ("qfls_nn_eV", nn["qfls"]), ("tau_nn_s", nn["tau_diff"])])
    d = C["data"]
    step("sheet")(new_sheet_xy, book, "data", [("time_s", d["t"]), ("pl", d["pl"]), ("qfls_eV", d["qfls"]), ("tau_diff_s", d["tau_diff"])])
    pages = []
    for tmpl, panels in TEMPLATES.items():
        path = find_template(tmpl)
        gp = step(f"template {tmpl}")(op.new_graph, template=path, lname=tmpl) if path else None
        if gp is None:
            print(f"template '{tmpl}' not found (searched {template_dirs()}): plain graph with {len(panels)} panels;"
                  f" copy {tmpl}.otpu into the User Files folder to use your template")
            gp = step(f"plain graph {tmpl}")(plain_panels, tmpl, panels, book, C)
            if gp is not None:
                pages.append(gp)
            continue
        print(f"template '{tmpl}': {path}")
        lt(f"win -a {gp.name};")
        n_lay = int(op.lt_float("page.nlayers"))
        print(f"{tmpl}: {n_lay} layer(s); panels {panels}")
        for line in step("template layer titles")(layer_titles, gp, n_lay) or []:
            print("   ", line)
        for j, name in enumerate(panels):
            li = min(j, n_lay - 1)
            step(f"{tmpl}: {name}")(curve_layer, gp, li, gp[li], book, C, name, titles=False, template=True)
        for line in step("template layer check")(layer_check, gp, n_lay) or []:
            print("   ", line)
        pages.append(gp)
    base = os.path.dirname(h5_path)
    for gp in pages:
        step("export pdf")(gp.save_fig, os.path.join(base, f"{gp.lname}.pdf"))
        step("export png")(gp.save_fig, os.path.join(base, f"{gp.lname}.png"), width=2400)
    return pages


def user_files():
    """Origin's User Files folder, whatever its name in the installed language (User Files, Anwenderdateien)."""
    try:
        return op.path("u")
    except Exception:                                   # noqa: BLE001
        lt('string __u$ = "%Y";')
        return op.get_lt_str("__u$")


def template_dirs():
    u = step("user files folder")(user_files)
    return [d for d in [u] + list(TEMPLATE_DIRS) if d and os.path.isdir(d)]


def find_template(name):
    """Full path of <name>.otpu / .otp in the User Files folder or TEMPLATE_DIRS (subfolders included), else None."""
    want = {f"{name}{e}".lower() for e in (".otpu", ".otp")}
    for d in template_dirs():
        for root, _, files in os.walk(d):
            for f in files:
                if f.lower() in want:
                    return os.path.join(root, f)
    return None


def save_project(fname):
    """Save the whole project as fname (originpro, else LabTalk 'save'); raise if no file appears."""
    try:
        op.save(fname)
    except Exception as e:                              # noqa: BLE001
        print(f"op.save failed ({e}); trying LabTalk save")
        lt(f'save "{fname}";')
    if not os.path.isfile(fname):
        raise IOError(f"project file not written: {fname}")


def py_range(p):
    return float(p["lo"]), float(p["hi"])


def build_loadings(L):
    V, sig, cols = L["V"], L["sigma"], L["cols"]
    ms = new_matrix("loadings", V, [1, V.shape[1]], [1, V.shape[0]])
    gp = op.new_graph(template="heatmap", lname="corner_loadings")
    p = gp[0].add_mplot(ms, 0, type=105)
    step("palette")(lambda: setattr(p, "colormap", "BlueWhiteRed.pal"))
    lt("layer.cmap.zmin = -1; layer.cmap.zmax = 1;")
    lt('xb.text$ = "eigenvector"; yl.text$ = "parameter";')
    print("rows (parameters):", cols, "| sigma_k:", np.round(sig, 3))
    return gp


def main():
    path = pick_file()
    G = read_group(path, FIGURE)
    out = os.path.splitext(path)[0]
    gp = build_loadings(G) if FIGURE == "loadings" else build_corner(G, path, FIGURE)
    if gp is None:
        return
    if TEMPLATE_GRAPHS and FIGURE != "loadings" and G.get("curves"):     # slices_paper.h5: graphs from the user's templates
        step("template graphs")(build_template_graphs, G, path)
    base = os.path.join(os.path.dirname(path), f"corner_{FIGURE}")
    # project first, so a failing export / template step cannot keep it from being saved (no __file__ needed:
    # it is not always defined when Origin runs a script)
    step("project")(save_project, base + ".opju")
    step("export pdf")(gp.save_fig, base + ".pdf")
    step("export png")(gp.save_fig, base + ".png", width=2400)
    tmpl = f"corner-{G.get('layout_kind', 'loadings')}-{FIGURE}-v2"
    step("template")(lt, f'template_saveas pg:=[{gp.name}] template:="{tmpl}";')      # into the User Files folder
    print(f"done: {base}.opju / .pdf / .png | template {tmpl}.otpu in {step('user files folder')(user_files)}")
    if LOG:
        print("\nsteps that failed (the rest of the figure was built):")
        for m in LOG:
            print("  ", m)
    if LT_FAILED:
        print("\nLabTalk statements that failed (count, first example) - please send this list back:")
        for n, ex in LT_FAILED.values():
            print(f"   {n:4d} x  {ex}")


if not globals().get("_NO_MAIN"):                       # build_corner_traps.py loads this file as a library
    main()
