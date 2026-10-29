"""
build_corner_modes.py — tr-PL, ss-PL and joint SMC posteriors in one corner plot (Origin)
=========================================================================================
Input : results/<base>_joint/corner_modes/corner_modes.h5 (scripts/modes_corner.py on rekenbeest)
Output: corner_modes.opju / .pdf / .png next to the .h5 file

  * lower triangle: the fill mode of the file (tr-PL, blue) as filled 1σ (dark) / 2σ (light) HPD regions, the other
    modes (ss-PL grey, joint red) as 1σ solid / 2σ dashed contour lines on top; diagonal: 1D marginals per mode
  * upper right: posterior predictive check tau_diff(QFLS): 68 % / 95 % bands of the NN curves in the same style
    (fill mode filled, the others solid / dashed lines), the measured data (open circles) and the joint median
    (solid bright red, MEDIAN_COLOR); legend in the empty lower left of the panel
  * below it: the same for the ss-PL data (QFLS vs log illumination, or log PLQY vs QFLS: axes from the file), from the full ss-PL model curves
  * 1 trap (no room in the upper triangle): both PPC panels as squares in a row above the corner, legend at the top of
    the free upper triangle
  * 2 traps (TRAP_SETTINGS[2]): the PPC panels as squares in a row below the corner (PPC_ROW_BELOW) with the best-fit
    curves, the slice panels (SLICE_PANELS) stacked in the upper right with one colour bar, the legend below them;
    joint median as open squares (2D) and dashed lines (1D), best fit as stars and dotted lines
Square panels, 10^n labels, log minor ticks, merge and fitted page as build_corner.py, which must lie in the same
folder (BUILD_CORNER).  Run inside Origin:  run -pyf "C:\\...\\origin\\build_corner_modes.py"
"""
import json
import os

import numpy as np

H5 = None                  # None -> file dialog
BUILD_CORNER = r"C:\Users\ro.heumann\Documents\PhDProject\NN_Paper\Data-Final\Second-Submission\origin\build_corner.py"
TICK_COLOR = "#000000"     # ticks drawn in the panels (the fills are light, so black)
SLICE_TICK_COLOR = "#FFFFFF"   # ... in the slice panels (dark Inferno fill): white
LINE_WIDTH = {"sigma1": 1.4, "sigma2": 1.0, "fill_edge": 0.6, "pdf": 1.6, "median": 1.6}
DASH = {"sigma1": 0, "sigma2": 1, "median": 0}               # Origin -d: 0 solid, 1 dash, 2 dot
PPC_CELLS = 2              # 1 trap: the two PPC panels are squares of this many corner cells, in a row ABOVE the corner
PPC_TOP_CM = 0.6           # ... starting this far from the top of the page (tau_diff left, ss-PL right)
PPC_HGAP_CM = 0.6          # ... this far apart (no labels between them: the ss-PL y axis is on its right)
PPC_BELOW_CM = 3.2         # ... the corner starts this far below them (their x labels + title, the E_t - E_g axis)
DENSITY_TITLE = r"\i(p) (dec\+(-1))"   # title of the 1D density axes (bottom-right diagonal panel only)
MEDIAN_COLOR = "#006400"   # joint median in the PPC panels: solid red (best-fit red of build_corner), brighter than the
                           # dark red joint band lines
RANGES = {                 # axis ranges by hand per number of traps, in decades, e.g. {"log10_n1_1": (8.0, 17.8)}:
                           # every panel with that parameter (corner, marginal, slice panels, E_t - E_g axis); fills and
                           # lines are cut to it. None keeps that end as it is in the file
    1: {
        "log10_n1_1": (15.0, None),
        "log10_taup_1": (-10.0, -5.0),
        "log10_taun_1": (-10.0, -6.0),
    },
    2: {
        "log10_taun_1": (-11.0, None),
        "log10_n1_2": (10.0, None),
        "log10_taup_2": (-10.0, None),
        "log10_taun_2": (-11.0, -7.0),
    },
}
SLICE_RANGES = {           # per number of traps, per slice panel <kx>__<ky>, applied after RANGES (overrides it in
    1: {},                 # that panel only)
    2: {"log10_n1_1__log10_taup_1": {"log10_n1_1": (15.69897, 16.69897),          # 5e15 .. 5e16 cm^-3
                                     "log10_taup_1": (-8.30103, -7.30103)},       # 5e-9 .. 5e-8 s
        "log10_n1_2__log10_taup_2": {"log10_n1_2": (16.0, None), "log10_taup_2": (-9.0, -6.0)}},
}
SHOW_TRUTH = True          # truth of synthetic data (file: params/<k> attr truth): dashed lines in the 2D, 1D and slice panels
SHOW_BEST_FIT = False       # joint best fit (attr best_fit): a star in the 2D and slice panels
SHOW_SLICE = True          # the slice panel of the file (/slice): 1 trap at the right end of row 0, 2 traps in the upper
                           # right (3 x 3 cells); both with a colour bar
SLICE_PANELS = None        # slice panels <kx>__<ky> to show, in this order (None: all of the file, /slice, /slice_2, ...)
PPC_ON_TOP = False         # 2 traps: PPC panels in a row above the corner (as for 1 trap) instead of the upper triangle
PPC_ROW_BELOW = False      # ... that row below the corner instead of above it
PPC_UNDER_CM = 2.2         # ... starting this far below the corner's bottom panels (their x labels + titles)
X_LABEL_ROTATE = None      # angle of the corner's x tick labels (None: build_corner's STYLE, 45)
TRUTH_COLOR = "#FF7F00"    # orange
BEST_COLOR = "#E41A1C"     # red: best-fit star in the 2D panels, dotted line in the 1D panels
BEST_1D_DASH = 2           # Origin -d of that line: 2 dot, 6 short dot
BEST_1D_WIDTH = None       # ... and its width in pt (None: STYLE overlay_width)
SHOW_BEST_PPC = True       # best-fit curves in the PPC panels (ppc/best_x, best_y in the file), BEST_COLOR
BEST_PPC_DASH = 0          # ... Origin -d (0 solid)
BEST_PPC_WIDTH = 1.6       # ... width in pt
SHOW_MEDIAN = False        # joint posterior median (params attr median): open square in the 2D panels, dashed line in
MEDIAN_DASH_1D = 1         # the 1D panels (Origin -d 1 dash), both in MEDIAN_COLOR
MEDIAN_SQUARE_CM = 0.11    # half side of the square
MEDIAN_LINE_PT = 1.5       # line width of the square
TAU0_WIDTH = None          # width of the iso-tau_0 lines in pt (None: STYLE overlay_width)
PPC_TAU_RANGE = None       # (lo, hi) of the tau_diff PPC axis in log10 s, None ends kept; bands, median and data cut to it
X_LABEL_COUNT = None       # corner x axes with only decade labels: exactly this many (every k-th decade)
PPC_TITLES = {}            # axis titles of the PPC panels instead of the file's: keys tau_x, tau_y, sspl_x, sspl_y
LETTERS = {}               # panel letters: {"corner" | "ppc" | "sspl": (text, fx, fy, colour), "slices": [...]} at
                           # fractions fx, fy of that panel (corner: its top-left 1D panel; > 1 = right of it)
LETTER_PT = 14
PPC_TICK_PT = None         # tick-label size of the PPC panels in pt (None: STYLE tick_pt as the corner)
STYLE_SET = {}             # build_corner STYLE entries to change for this figure, e.g. font sizes {"tick_pt": 11}
LEGEND_W_CM = 4.0          # 2 traps: width of the legend below the slice panels
SLICE_BEST_COLOR = "#FFFF00"   # best fit in the slice panel (the slice's centre): yellow star with a black outline
SLICE_BEST_EDGE = "#000000"
STAR_FILLED = True         # False: the best-fit stars as outlines only (STAR_LINE_PT)
STAR_LINE_PT = 1.2
PPC_HALF_PAGE = False      # PPC row on top: each panel a square half the corner's width (else PPC_CELLS cells)
SLICE_SCALE = None         # 2 traps: side of the slice panels in corner cells, top right of the corner (None: 3 x 3 cells),
                           # stacked (SLICE_PANELS order) with one colour bar; its title in line with the title of the
                           # bottom-right density axis
SLICE_VGAP_CM = 1.6        # ... between them (x labels + title of the upper one)
SLICE_ALIGN_CM = 0.0       # ... shift of the slice panels and colour bar to the right, if the title is not quite in line
STAR_RADIUS_CM = 0.15      # best-fit star (drawn as nested outlines: symbol plots do not render in the merged pages)
TAU0_COLOR = {"photodoping": "#CC7722", "non_photodoping": "#FFFFFF",   # iso-tau_0 lines in the slice panel: ochre /
              "photodoping_ntratio": "#7CFC00", "non_photodoping_ntratio": "#FF77FF",       # white, dash / dash-dot,
              "taun_plus_taup": "#00E5FF"}   # labelled in the panel in the same colour; taun_plus_taup: (tau_n, tau_p)
                                             # slice, deep-trap line tau_n + tau_p = best fit (modes_corner.py)
TAU0_DASH = {"photodoping": 1, "non_photodoping": 3, "photodoping_ntratio": 2,   # ntratio: trap-2 slice,
             "non_photodoping_ntratio": 2, "taun_plus_taup": 1}                 # both formulas at tau_0 N_t1/N_t2 (file)
TAU0_TEXT = {"photodoping": "photodoping", "non_photodoping": "no photodoping",
             "photodoping_ntratio": r"photodoping \x(00D7) \i(N)\-(t1)/\i(N)\-(t2)",
             "non_photodoping_ntratio": r"no photodoping \x(00D7) \i(N)\-(t1)/\i(N)\-(t2)",
             "taun_plus_taup": r"\g(t)\-(n) + \g(t)\-(p) \(61) const"}
TAU0_TEXT_AT = 0.5         # each iso-tau_0 label at this fraction along its line
TRAP_SETTINGS = {          # per number of traps: overrides of the settings above (1 trap keeps them)
    2: {"SHOW_BEST_FIT": True, "BEST_COLOR": "#1F3FFF", "PPC_ON_TOP": True, "PPC_ROW_BELOW": True,
        "SHOW_MEDIAN": True, "BEST_PPC_WIDTH": 2.0,
        "TAU0_COLOR": {**TAU0_COLOR, "photodoping": "#00E5FF"},
        "LINE_WIDTH": {**LINE_WIDTH, "pdf": 2.0, "median": 2.0}, "BEST_1D_WIDTH": 2.0, "TAU0_WIDTH": 2.0,
        "PPC_TAU_RANGE": (-8.30103, None), "X_LABEL_COUNT": 3,
        "PPC_HALF_PAGE": True, "SLICE_PANELS": None, "SLICE_SCALE": 1.5, "X_LABEL_ROTATE": 0,
        "TAU0_TEXT_AT": 0.25, "STAR_FILLED": False, "SLICE_BEST_COLOR": "#1F3FFF", "SLICE_BEST_EDGE": None,
        "DATA_RADIUS_CM": 0.09,
        "STYLE_SET": {"tick_pt": 10.5, "title_pt": 12.5, "cbar_pt": 10.5}, "LETTER_PT": 16, "PPC_TICK_PT": 14,
        "LEGEND_ROW_CM": 0.45, "LEGEND_W_CM": 4.2, "PPC_BELOW_CM": 3.6, "SLICE_VGAP_CM": 1.8, "MEDIAN_COLOR": "#CC9900", "BEST_1D_DASH": 6,
        "PPC_TITLES": {"tau_y": r"differential decay time \i(\g(t))\-(diff) (s)",
                       "sspl_y": r"Fermi-level splitting \g(D)\i(E)\-(F) (eV)"},
        "LETTERS": {"corner": ("(a)", 1.6, 0.85, "#000000"), "slices": [("(b)", 0.88, 0.85, "#FFFFFF"),
                                                                      ("(c)", 0.88, 0.85, "#FFFFFF")],
                    "ppc": ("(d)", 0.9, 0.92, "#000000"), "sspl": ("(e)", 0.1, 0.92, "#000000")}},
}
LEGEND_ROW_CM = 0.38       # row height of the legend (cbar_pt text)
LEGEND_POS = (0.03, 0.40)  # legend in the tau_diff panel: left edge, top edge as fractions of the panel (lower left,
                           # below the bands); adjust if it hits a band


def load_build_corner():
    path = BUILD_CORNER
    if not os.path.isfile(path) and "__file__" in globals():
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "build_corner.py")
    bc = {"__name__": "build_corner", "_NO_MAIN": True}
    with open(path, encoding="utf-8") as fh:
        exec(compile(fh.read(), path, "exec"), bc)
    return bc


def read(path):
    import h5py
    D = {}
    with h5py.File(path, "r") as f:
        s = lambda v: v.decode() if isinstance(v, bytes) else v
        D["keys"] = json.loads(f.attrs["param_order"])
        D["modes"] = json.loads(f.attrs["modes"])
        D["labels"] = json.loads(f.attrs["mode_labels_json"])
        D["colors"] = json.loads(f.attrs["mode_colors_json"])
        D["colors27"] = json.loads(f.attrs["colors27_json"])
        D["fill"] = s(f.attrs.get("fill_mode", "trpl"))
        mi = D["modes"].index(D["fill"])
        D["fill_colors"] = (json.loads(f.attrs["fill_colors_json"]) if "fill_colors_json" in f.attrs
                            else [D["colors27"][3 ** mi * lv] for lv in (0, 1, 2)])     # outside, 2σ, 1σ
        D["sigma"] = [float(v) for v in f.attrs.get("sigma_levels", (0.6827, 0.9545))]
        D["layout"] = f["layout"][()]
        D["params"] = {k: {a: s(v) for a, v in f["params"][k].attrs.items()} for k in D["keys"]}
        D["panels"] = {n: {d: g[d][()] for d in g} for n, g in f["panels"].items()}
        D["ppc"] = {d: f["ppc"][d][()] for d in f["ppc"]}
        D["ppc_attrs"] = {a: s(v) for a, v in f["ppc"].attrs.items()}
        if "sspl_ppc" in f:
            D["sspl"] = {d: f["sspl_ppc"][d][()] for d in f["sspl_ppc"]}
            D["sspl_attrs"] = {a: s(v) for a, v in f["sspl_ppc"].attrs.items()}
        D["slices"] = []                                     # /slice, /slice_2, ... in that order
        names = sorted((n for n in f if n == "slice" or n.startswith("slice_")),
                       key=lambda n: 1 if n == "slice" else int(n.split("_")[1]))
        for n in names:
            g = f[n]
            D["slices"].append({"d": {d: g[d][()] for d in g if d != "params"},
                                "attrs": {a: s(v) for a, v in g.attrs.items()},
                                "params": {k: {a: s(v) for a, v in g["params"][k].attrs.items()} for k in g["params"]}})
    return D


def ppc_cells(D):
    """Grid cells of the predictive-check panel in the empty upper right of the triangle: (r0, c0, rs, cs)."""
    n = len(D["keys"])
    rs, cs = max(1, n // 2 - 1), n // 2
    return 0, n - cs, rs, cs


def tick_lines(bc, gl, book, name, px, py, geo, logx, logy, color=None):
    for ax, pa, pb, L, log in (("x", px, py, geo[3], logx), ("y", py, px, geo[2], logy)):
        a, b = bc["tick_segments"](pa, pb, L, log)
        cols = [("x", a), ("y", b)] if ax == "x" else [("x", b), ("y", a)]
        w = bc["step"]("sheet")(bc["new_sheet_xy"], book, f"ticks{ax}_{name}", cols)
        if w is not None:
            bc["step"]("ticks")(bc["line_plot"], gl, w, 0, 1, color or TICK_COLOR, 0.8)


def legend(bc, gp, idx, D, px, py, x_frac, top_frac, height_cm, width_cm):
    """Legend as line pieces + text in layer idx (data coordinates px, py; layer size in cm): the fill mode as dark /
    light swatches, the other modes as solid / dashed pieces, data (open circle), joint median (dotted), and a line
    saying which is 68 / 95 %.  Left edge x_frac, top top_frac of the layer."""
    op, lt, S = bc["op"], bc["lt"], bc["STYLE"]
    x0, x1, y0, y1 = float(px["lo"]), float(px["hi"]), float(py["lo"]), float(py["hi"])
    dx, dy = x1 - x0, y1 - y0
    cx, cy = dx / width_cm, dy / height_cm                   # data units per cm
    row, xl, yt = LEGEND_ROW_CM * cy, x0 + x_frac * dx, y0 + top_frac * dy
    gl = gp[idx]
    book = op.new_book("w", lname="legend_modes")
    C, fm = D["colors"], D["fill"]
    entries = [(D["labels"][fm], [(D["fill_colors"][2], 6, 0), (D["fill_colors"][1], 6, 0)])]
    entries += [(D["labels"][m], [(C[m]["s1"], LINE_WIDTH["sigma1"], 0), (C[m]["s1"], LINE_WIDTH["sigma2"], 1)])
                for m in D["modes"] if m != fm]
    entries += [("data", None), (f"median ({D['labels']['joint']})", "median" if SHOW_MEDIAN else
                                  [(MEDIAN_COLOR, LINE_WIDTH["median"], DASH["median"])])]
    if SHOW_TRUTH and any(_val(p_, "truth") is not None for p_ in D["params"].values()):
        entries.append(("true value", [(TRUTH_COLOR, bc["STYLE"]["overlay_width"], 1)]))
    if SHOW_BEST_FIT and any(_val(p_, "best_fit") is not None for p_ in D["params"].values()):
        entries.append((f"best fit ({D['labels']['joint']})", "star"))
    if SHOW_BEST_FIT and SHOW_SLICE and D["slices"] and (SLICE_BEST_COLOR != BEST_COLOR or SLICE_BEST_EDGE):
        entries.append(("best fit (slice centre)", "star_slice"))
    seg = 0.32 * cx                                          # one swatch piece
    for i, (text, sw) in enumerate(entries):
        y = yt - (i + 0.5) * row
        if sw is None:
            ox, oy = circles([xl + seg], [y], cx, cy)            # the data symbol, drawn as lines like the data
            w = bc["new_sheet_xy"](book, f"lg{i}", [("x", ox), ("y", oy)])
            bc["line_plot"](gl, w, 0, 1, "#000000", DATA_LINE_PT)
        elif sw == "median":                                 # PPC line with the 2D panels' square on it
            w = bc["new_sheet_xy"](book, f"lg{i}", [("x", [xl, xl + 2 * seg]), ("y", [y, y])])
            bc["line_plot"](gl, w, 0, 1, MEDIAN_COLOR, LINE_WIDTH["median"], dash=DASH["median"])
            ox, oy = square_xy(xl + seg, y, cx, cy)
            w = bc["new_sheet_xy"](book, f"lg{i}s", [("x", ox), ("y", oy)])
            bc["line_plot"](gl, w, 0, 1, MEDIAN_COLOR, MEDIAN_LINE_PT)
        elif sw in ("star", "star_slice"):
            if sw == "star" and SHOW_BEST_PPC and "best_x" in D["ppc"]:    # the PPC curve behind the star
                w = bc["new_sheet_xy"](book, f"lg{i}l", [("x", [xl, xl + 2 * seg]), ("y", [y, y])])
                bc["line_plot"](gl, w, 0, 1, BEST_COLOR, BEST_PPC_WIDTH, dash=BEST_PPC_DASH)
            ox, oy = star_xy(xl + seg, y, cx, cy)
            w = bc["new_sheet_xy"](book, f"lg{i}", [("x", ox), ("y", oy)])
            bc["line_plot"](gl, w, 0, 1, BEST_COLOR if sw == "star" else SLICE_BEST_COLOR,
                            1.0 if STAR_FILLED else STAR_LINE_PT)
            if sw == "star_slice" and SLICE_BEST_EDGE:
                ox, oy = star_xy(xl + seg, y, cx, cy, scales=(1.0,))
                w = bc["new_sheet_xy"](book, f"lg{i}e", [("x", ox), ("y", oy)])
                bc["line_plot"](gl, w, 0, 1, SLICE_BEST_EDGE, 0.5)
        else:
            for k, (col, wid, dash) in enumerate(sw):
                xs = [xl + k * seg, xl + (k + 1) * seg - 0.04 * cx] if len(sw) > 1 else [xl, xl + 2 * seg]
                w = bc["new_sheet_xy"](book, f"lg{i}_{k}", [("x", xs), ("y", [y, y])])
                bc["line_plot"](gl, w, 0, 1, col, wid, dash=dash)
        nm = f"lgt{i}"
        lt(f"win -a {gp.name}; page.active = {idx + 1};")
        lt(f"label -a 0 0 -n {nm} {text}; {nm}.fsize = {S['cbar_pt']}; {nm}.attach = 2;")
        lt(f"{nm}.x = {xl + 2 * seg + 0.15 * cx} + {nm}.dx / 2; {nm}.y = {y};")
    foot = ["68 %: dark / solid, 95 %: light / dashed"] if width_cm > 5 else ["68 %: dark / solid", "95 %: light / dashed"]
    for i, text in enumerate(foot):                          # narrow legend layer (1 trap): two lines
        nm = f"lgh{i}"
        lt(f"label -a 0 0 -n {nm} {text}; {nm}.fsize = {S['cbar_pt']}; {nm}.attach = 2;")
        lt(f"{nm}.x = {xl} + {nm}.dx / 2; {nm}.y = {yt - (len(entries) + i + 0.5) * row};")


def fill_map(D, g):
    """(Z, bounds, colours) of the filled mode in panel / PPC group g: the continuous HPD mass when the file
    has it (levels at the 1σ / 2σ mass), otherwise its level 0 / 1 / 2 out of the category matrix."""
    fc, fm = D["fill_colors"], D["fill"]
    if f"{fm}_mass" in g:
        s1, s2 = D["sigma"]
        return np.asarray(g[f"{fm}_mass"], float), np.array([0.0, s1, s2, 1.0001]), [fc[2], fc[1], fc[0]]
    lv = (np.asarray(g["cat"], int) // 3 ** D["modes"].index(fm)) % 3
    return lv.astype(float), np.array([-0.5, 0.5, 1.5, 2.5]), list(fc)


def band_lines(bc, gl, book, name, D, x, Q):
    """68 % (solid) and 95 % (dashed) band edges of the modes that are not filled, joint last (on top)."""
    step = bc["step"]
    for m in [m for m in D["modes"] if m != D["fill"]]:
        for q, s_ in (("2.5", "sigma2"), ("97.5", "sigma2"), ("16", "sigma1"), ("84", "sigma1")):
            w = step("sheet")(bc["new_sheet_xy"], book, f"{name}_{m}_q{q}", [("x", x), ("y", Q[f"{m}_q{q}"])])
            if w is not None:
                step("band line")(bc["line_plot"], gl, w, 0, 1, D["colors"][m]["s1"], LINE_WIDTH[s_], dash=DASH[s_])


DATA_RADIUS_CM = 0.06      # measured data: open circles of this radius ...
DATA_LINE_PT = 2.0         # ... and line width, drawn as line pieces: symbol plots stayed invisible in the merged
                           # pages of this figure (Origin runs 1 and 2, 2026-10-20: -kf 0 and -kf 1), lines do not


def circles(x, y, ux, uy, r_cm=None, nv=16):
    """Open circles around (x, y) as one NaN-separated polyline; ux, uy: data units per cm of the panel."""
    r_cm = DATA_RADIUS_CM if r_cm is None else r_cm
    t = np.linspace(0.0, 2 * np.pi, nv + 1)
    cx, cy = [], []
    for a, b in zip(x, y):
        cx += list(a + r_cm * ux * np.cos(t)) + [np.nan]
        cy += list(b + r_cm * uy * np.sin(t)) + [np.nan]
    return np.array(cx), np.array(cy)


def data_points(bc, gl, book, name, x, y, geo, xr, yr):
    """Measured data as open circles, thinned to one per data_spacing_cm along the curve.  The sheet keeps all points
    (columns 1-2) and the thinned ones (3-4); the circles themselves are columns 5-6."""
    S = bc["STYLE"]
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    ux, uy = (xr[1] - xr[0]) / geo[2], (yr[1] - yr[0]) / geo[3]
    # panel coordinates in cm, so the thinning follows the axis ranges, not the data range
    u, v = (x - xr[0]) / ux, (y - yr[0]) / uy
    k = bc["thin_points"](u, v, False, False, (np.ptp(u) or 1.0, np.ptp(v) or 1.0), S["data_spacing_cm"]) if len(x) > 2 \
        else np.arange(len(x))
    ox, oy = circles(x[k], y[k], ux, uy)
    w = bc["step"]("sheet")(bc["new_sheet_xy"], book, name, [("x all", x), ("y all", y), ("x", x[k]), ("y", y[k]),
                                                             ("circle x", ox), ("circle y", oy)])
    if w is not None:
        bc["step"]("data")(bc["line_plot"], gl, w, 4, 5, "#000000", DATA_LINE_PT)


def best_curve(bc, gl, book, name, Q, xname):
    """Best-fit curve of a PPC panel (best_x, best_y in the file), BEST_COLOR."""
    if not (SHOW_BEST_FIT and SHOW_BEST_PPC and "best_x" in Q):
        return
    w = bc["step"]("sheet")(bc["new_sheet_xy"], book, name, [(xname, Q["best_x"]), ("best fit", Q["best_y"])])
    if w is not None:
        bc["step"]("best fit")(bc["line_plot"], gl, w, 0, 1, BEST_COLOR, BEST_PPC_WIDTH, dash=BEST_PPC_DASH)


def square_xy(x, y, ux, uy, r_cm=None):
    """Open square around (x, y), half side r_cm (MEDIAN_SQUARE_CM)."""
    r_cm = MEDIAN_SQUARE_CM if r_cm is None else r_cm
    a, b = r_cm * ux, r_cm * uy
    return np.array([x - a, x + a, x + a, x - a, x - a]), np.array([y - b, y - b, y + b, y + b, y - b])


def _cut_lines(v, col, lo, hi):
    v = np.array(v, float)
    if len(v):
        v[(v[:, col] < lo) | (v[:, col] > hi)] = np.nan
    return v


def _resolve(p_, lo, hi):
    """(lo, hi) with None replaced by the parameter's current end."""
    return (float(p_["lo"]) if lo is None else float(lo)), (float(p_["hi"]) if hi is None else float(hi))


def range_slice(sl, k, lo, hi):
    """New range of parameter k in one slice panel: hpd grid cropped, iso-tau_0 and sigma lines cut."""
    Sd, SA, SP = sl["d"], sl["attrs"], sl["params"]
    if k not in SP:
        return
    lo, hi = _resolve(SP[k], lo, hi)
    SP[k]["lo"], SP[k]["hi"] = lo, hi
    for axis, key, col in (("x", SA["kx"], 0), ("y", SA["ky"], 1)):
        if key != k:
            continue
        keep = (Sd[axis] >= lo) & (Sd[axis] <= hi)
        Sd[axis] = Sd[axis][keep]
        Sd["hpd"] = Sd["hpd"][:, keep] if axis == "x" else Sd["hpd"][keep, :]
        for d in [d for d in Sd if d.startswith("tau0_") or d in ("sigma1", "sigma2")]:
            Sd[d] = _cut_lines(Sd[d], col, lo, hi)


def apply_ranges(D, ranges, slice_ranges=None):
    """RANGES: new lo / hi of parameters (None: keep that end); the grids of the fills are cropped to cell centres
    inside, lines outside set to NaN (Origin draws gaps), so nothing is drawn beyond the panel frame.  slice_ranges
    {<kx>__<ky>: {k: (lo, hi)}}: afterwards, in that slice panel only."""
    cut_lines = _cut_lines
    for k, (lo, hi) in ranges.items():
        for sl in D["slices"]:
            range_slice(sl, k, lo, hi)
        if k not in D["params"]:
            continue
        lo, hi = _resolve(D["params"][k], lo, hi)
        D["params"][k]["lo"], D["params"][k]["hi"] = lo, hi
        for name, g in D["panels"].items():
            if "__" not in name:                        # 1D marginal
                if name == k:
                    for m in D["modes"]:
                        x = np.asarray(g[f"{m}_x"], float)
                        g[f"{m}_pdf"] = np.where((x >= lo) & (x <= hi), g[f"{m}_pdf"], np.nan)
                continue
            kx, ky = name.split("__")
            panels = [(g, kx, ky)]
            for g_, kx_, ky_ in panels:
                for axis, key, col in (("x", kx_, 0), ("y", ky_, 1)):
                    if key != k:
                        continue
                    keep = (g_[axis] >= lo) & (g_[axis] <= hi)
                    g_[axis] = g_[axis][keep]
                    for d in [d for d in g_ if d == "cat" or d.endswith("_mass")]:
                        g_[d] = g_[d][:, keep] if axis == "x" else g_[d][keep, :]
                    for d in [d for d in g_ if d.endswith("_sigma1") or d.endswith("_sigma2")]:
                        g_[d] = cut_lines(g_[d], col, lo, hi)
    for sl in D["slices"]:
        for k, (lo, hi) in (slice_ranges or {}).get(f"{sl['attrs']['kx']}__{sl['attrs']['ky']}", {}).items():
            range_slice(sl, k, lo, hi)


def crop_ppc(Q, lo, hi):
    """tau_diff PPC axis cut to (lo, hi) in log10 s (None: keep): category grid rows cropped, band edges, median and data
    outside set to NaN (Origin draws gaps; the data circles are left out)."""
    y = np.asarray(Q["y"], float)
    lo = y[0] if lo is None else float(lo)
    hi = y[-1] if hi is None else float(hi)
    keep = (y >= lo) & (y <= hi)
    Q["y"] = y[keep]
    for d in [d for d in Q if d == "cat" or d.endswith("_mass")]:
        Q[d] = np.asarray(Q[d])[keep, :]
    # band edges <mode>_q<quantile> (not data_qfls, the data's x), the median and the data
    for d in [d for d in Q if d.rsplit("_q", 1)[-1].replace(".", "").isdigit() or d in ("median_y", "data_y", "best_y")]:
        v = np.array(Q[d], float)
        v[(v < lo) | (v > hi)] = np.nan
        Q[d] = v


def star_xy(x, y, ux, uy, r_cm=STAR_RADIUS_CM, scales=None):
    """A five-pointed star at (x, y) as nested outlines (scales 1 ... 0.15), so it reads as filled; NaN-separated."""
    t = np.pi / 2 + np.arange(11) * np.pi / 5
    rr = np.where(np.arange(11) % 2 == 0, 1.0, 0.42)
    cx, cy = [], []
    if scales is None:
        scales = (1.0, 0.8, 0.6, 0.42, 0.25, 0.12) if STAR_FILLED else (1.0,)
    for sc in scales:
        cx += list(x + sc * r_cm * ux * rr * np.cos(t)) + [np.nan]
        cy += list(y + sc * r_cm * uy * rr * np.sin(t)) + [np.nan]
    return np.array(cx), np.array(cy)


class ppc_ticks:
    """with ppc_ticks(S): STYLE tick_pt set to PPC_TICK_PT inside (PPC axes and their 10^n labels), restored after."""
    def __init__(self, S, on=True):
        self.S, self.on = S, on and PPC_TICK_PT is not None

    def __enter__(self):
        if self.on:
            self.old, self.S["tick_pt"] = self.S["tick_pt"], PPC_TICK_PT

    def __exit__(self, *a):
        if self.on:
            self.S["tick_pt"] = self.old


def _val(p_, k):
    v = p_.get(k)
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def marks(bc, gl, book, name, px, py, geo, star=True, color=None, edge=None):
    """Truth (orange dashed lines through the panel) and best fit (star, BEST_COLOR or color, outlined in edge) of a 2D
    panel, if switched on and in the file."""
    step = bc["step"]
    tx, ty = _val(px, "truth"), _val(py, "truth")
    if SHOW_TRUTH:
        segs = []
        if tx is not None:
            segs.append(([tx, tx], [float(py["lo"]), float(py["hi"])]))
        if ty is not None:
            segs.append(([float(px["lo"]), float(px["hi"])], [ty, ty]))
        for i, (xs, ys) in enumerate(segs):
            w = step("sheet")(bc["new_sheet_xy"], book, f"truth{i}_{name}", [("x", xs), ("y", ys)])
            if w is not None:
                step("truth")(bc["line_plot"], gl, w, 0, 1, TRUTH_COLOR, bc["STYLE"]["overlay_width"], dash=1)
    mx, my = _val(px, "median"), _val(py, "median")
    if SHOW_MEDIAN and mx is not None and my is not None:
        ux = (float(px["hi"]) - float(px["lo"])) / geo[2]
        uy = (float(py["hi"]) - float(py["lo"])) / geo[3]
        qx_, qy_ = square_xy(mx, my, ux, uy)
        w = step("sheet")(bc["new_sheet_xy"], book, f"median_{name}", [("x", qx_), ("y", qy_)])
        if w is not None:
            step("median")(bc["line_plot"], gl, w, 0, 1, MEDIAN_COLOR, MEDIAN_LINE_PT)
    bx, by = _val(px, "best_fit"), _val(py, "best_fit")
    if SHOW_BEST_FIT and star and bx is not None and by is not None:
        ux = (float(px["hi"]) - float(px["lo"])) / geo[2]
        uy = (float(py["hi"]) - float(py["lo"])) / geo[3]
        sx_, sy_ = star_xy(bx, by, ux, uy)
        w = step("sheet")(bc["new_sheet_xy"], book, f"best_{name}", [("x", sx_), ("y", sy_)])
        if w is not None:
            step("best fit")(bc["line_plot"], gl, w, 0, 1, color or BEST_COLOR, 1.0 if STAR_FILLED else STAR_LINE_PT)
        if edge:
            sx_, sy_ = star_xy(bx, by, ux, uy, scales=(1.0,))
            w = step("sheet")(bc["new_sheet_xy"], book, f"best_edge_{name}", [("x", sx_), ("y", sy_)])
            if w is not None:
                step("best fit")(bc["line_plot"], gl, w, 0, 1, edge, 0.5)


def cbar_text(bc, gp, idx, box, bounds, levels, title, xt_min=None, every=2):
    """Edge values (every 2nd: 13 edges on a short bar) and the rotated title in the colour-bar layer of the merged page;
    xt_min: the title at least this far right (bar widths), to line it up with another axis title."""
    lt, S = bc["lt"], bc["STYLE"]
    n = len(bounds) - 1
    lt(f"win -a {gp.name}; page.active = {idx + 1};")
    x0, w = 1.35, 0.0
    for k in range(1, n, every):
        nm, y = f"scb{k}", n - k
        lt(f"label -a {x0} {y} -n {nm} {100 * float(levels[k - 1]):g} %;")
        lt(f"{nm}.fsize = {S['cbar_pt']}; {nm}.x = {x0} + {nm}.dx / 2; {nm}.y = {y};")
        w = max(w, bc["op"].lt_float(f"{nm}.dx"))
    xt = x0 + w + 0.3 + 0.7 * S["cbar_pt"] * 0.03528 / box[2]
    if xt_min is not None:
        xt = max(xt, xt_min)
    lt(f"label -a {xt} {n / 2} -n scbtitle {title}; scbtitle.rotate = 90; scbtitle.fsize = {S['cbar_pt']};"
       f" scbtitle.x = {xt}; scbtitle.y = {n / 2};")


def main():
    bc = load_build_corner()
    op, step, S = bc["op"], bc["step"], bc["STYLE"]
    if H5:
        bc["H5"] = H5
    path = bc["pick_file"]()
    D = read(path)
    n_traps = sum(k.startswith("log10_n1_") for k in D["params"])
    globals().update(TRAP_SETTINGS.get(n_traps, {}))
    S.update(STYLE_SET)
    if SLICE_PANELS:                                         # only these slice panels, in this order
        by_name = {f"{sl['attrs']['kx']}__{sl['attrs']['ky']}": sl for sl in D["slices"]}
        D["slices"] = [by_name[k] for k in SLICE_PANELS if k in by_name]
    if X_LABEL_ROTATE is not None:
        S["x_label_rotate"] = X_LABEL_ROTATE
    if X_LABEL_COUNT:
        S["x_label_count"] = X_LABEL_COUNT
    if PPC_TAU_RANGE:
        crop_ppc(D["ppc"], *PPC_TAU_RANGE)
    if RANGES.get(n_traps) or SLICE_RANGES.get(n_traps):
        apply_ranges(D, RANGES.get(n_traps, {}), SLICE_RANGES.get(n_traps, {}))
    layout, P = D["layout"], D["params"]
    cmaps = []                                               # (page, bounds, colours): second colour pass
    pos = bc["geometry"](layout)
    n0 = len(D["keys"])
    _, _, rs0, _ = ppc_cells(D)
    # 1 trap: no room for the PPC panels in the upper triangle; 2 traps: PPC_ON_TOP
    small = "sspl" in D and (n0 - 2 - rs0 < 2 or PPC_ON_TOP)
    if small:                                            # -> a row of two squares above the corner (pos reads the margin)
        side = PPC_CELLS * S["panel_cm"] + (PPC_CELLS - 1) * S["gap_cm"]
        if PPC_HALF_PAGE:                                # each half the corner's width
            side = (n0 * S["panel_cm"] + (n0 - 1) * S["gap_cm"] - PPC_HGAP_CM) / 2
        l0 = S["margin_left_cm"]
        if PPC_ROW_BELOW:                                # below the corner's bottom row (its x labels and titles)
            _, t_b, _, h_b = pos(n0 - 1, 0)
            top_ = t_b + h_b + PPC_UNDER_CM
        else:
            top_ = PPC_TOP_CM
            S["margin_top_cm"] = PPC_TOP_CM + side + PPC_BELOW_CM
        small_geo = ((l0, top_, side, side), (l0 + side + PPC_HGAP_CM, top_, side, side))
    book = op.new_book("w", lname="corner_modes_data")
    pages, final, tasks = [], {}, {}
    et_pages = {}                                            # page -> (where, n1 parameter): E_t axes after the merge
    spots = []                                               # panel letters: (LETTERS key, page, x range, y range, index)
    # diagonal panels left of a diagonal panel with an E_t axis on top: no density numbers (they ran into that axis)
    et_next = {int(r_["col"]) - 1 for r_ in layout if bc["_s"](r_["kind"]) == "1d" and bc["_s"](r_["extra"]).endswith("Et-Eg")}

    def fill_panel(name, gp, gl, g, zbc=None):
        """Filled mode of the page's first (only) layer; returns the page and layer, which are new if Origin refused the
        colour-fill contour (Origin run 2026-10-20: 2 of 8 pages without colour map, then solid red): the page is made
        again with a 3x resampled heatmap, as build_corner does."""
        Z, bnd, cols = fill_map(D, g) if zbc is None else zbc
        x, y = np.asarray(g["x"], float), np.asarray(g["y"], float)
        contour = bool(S["interpolate"])
        ms = step("matrix")(bc["new_matrix"], name, Z, x, y)
        ok = ms is not None and step("colour map")(bc["colormap_plot"], gp, 0, gl, ms, bnd, cols, contour=contour)
        if not ok and contour:
            j = len(pages) - 1
            lname = gp.lname
            op.lt_exec(f"win -cd {gp.name};")
            gp = op.new_graph(lname=lname)
            gl = gp[0]
            pages[j] = gp
            step("place layer")(bc["place_layer"], gp, 0, 0.5, 0.5, *final[j][2:])
            Zf, fx, fy = bc["regrid"](Z, x, y, x[0], x[-1], y[0], y[-1], int(S["upsample"]))
            ms = step("matrix")(bc["new_matrix"], f"{name}_hm", Zf, fx, fy)
            ok = ms is not None and step("colour map")(bc["colormap_plot"], gp, 0, gl, ms, bnd, cols)
            bc["LOG"].append(f"[colour map] {name}: contour failed, heatmap {'used' if ok else 'failed too'}")
            contour = False
        if ok:
            cmaps.append((gp, bnd, cols, contour))
        return gp, gl

    def new_page(name, geo):
        gp = op.new_graph(lname=name)
        pages.append(gp)
        final[len(pages) - 1] = geo
        step("place layer")(bc["place_layer"], gp, 0, 0.5, 0.5, *geo[2:])
        return gp, gp[0], len(pages) - 1

    for row in layout:
        r, c, kind = int(row["row"]), int(row["col"]), bc["_s"](row["kind"])
        kx, ky, xside, yside, extra = (bc["_s"](row[k]) for k in ("kx", "ky", "xside", "yside", "extra"))
        px = P[kx]
        if kind == "2d":
            py = P[ky]
            g = D["panels"][f"{kx}__{ky}"]
            gp, gl, j = new_page(f"panel_{kx}__{ky}", pos(r, c))
            gp, gl = fill_panel(f"{kx}__{ky}", gp, gl, g)
            fm = D["fill"]                                   # thin edge of the filled 1σ region, then the line modes
            for m, s_, wid, dash in [(fm, "sigma1", LINE_WIDTH["fill_edge"], 0)] + \
                    [(m, s_, LINE_WIDTH[s_], DASH[s_]) for m in D["modes"] if m != fm for s_ in ("sigma2", "sigma1")]:
                v = g.get(f"{m}_{s_}")
                if v is not None and len(v):
                    w = step("sheet")(bc["new_sheet_xy"], book, f"{m}_{s_}_{kx}__{ky}", [("x", v[:, 0]), ("y", v[:, 1])])
                    if w is not None:
                        step("contour line")(bc["line_plot"], gl, w, 0, 1, D["colors"][m]["s1"], wid, dash=dash)
            marks(bc, gl, book, f"{kx}__{ky}", px, py, final[j])
            tick_lines(bc, gl, book, f"{kx}__{ky}", px, py, final[j], kx.startswith("log10_"), ky.startswith("log10_"))
            step("axes")(bc["set_axes"], gl, (px["lo"], px["hi"]), (py["lo"], py["hi"]))
            tasks[j] = step("panel axes")(bc["panel_axes"], gp, 0, kx, ky, px, py, xside, yside, j) or []
        else:
            g = D["panels"][kx]
            gp, gl, j = new_page(f"panel_{kx}", pos(r, c))
            ymax = 1.1 * max(float(np.nanmax(g[f"{m}_pdf"])) for m in D["modes"])
            for m in D["modes"]:
                w = step("sheet")(bc["new_sheet_xy"], book, f"pdf_{m}_{kx}", [("x", g[f"{m}_x"]), ("y", g[f"{m}_pdf"])])
                if w is not None:
                    step("pdf")(bc["line_plot"], gl, w, 0, 1, D["colors"][m]["s1"], LINE_WIDTH["pdf"])
            last = r == c == len(D["keys"]) - 1                  # one title, at the bottom-right panel (page margin)
            pyd = {"lo": 0.0, "hi": ymax, "label_origin": DENSITY_TITLE if last else "", "nmax": 4}
            tv = _val(px, "truth")
            if SHOW_TRUTH and tv is not None:
                w = step("sheet")(bc["new_sheet_xy"], book, f"truth_{kx}", [("x", [tv, tv]), ("y", [0.0, ymax])])
                if w is not None:
                    step("truth")(bc["line_plot"], gl, w, 0, 1, TRUTH_COLOR, S["overlay_width"], dash=1)
            mv = _val(px, "median")
            if SHOW_MEDIAN and mv is not None:
                w = step("sheet")(bc["new_sheet_xy"], book, f"median_{kx}", [("x", [mv, mv]), ("y", [0.0, ymax])])
                if w is not None:
                    step("median")(bc["line_plot"], gl, w, 0, 1, MEDIAN_COLOR, LINE_WIDTH["median"], dash=MEDIAN_DASH_1D)
            bv = _val(px, "best_fit")
            if SHOW_BEST_FIT and bv is not None:
                w = step("sheet")(bc["new_sheet_xy"], book, f"best_{kx}", [("x", [bv, bv]), ("y", [0.0, ymax])])
                if w is not None:
                    step("best fit")(bc["line_plot"], gl, w, 0, 1, BEST_COLOR, BEST_1D_WIDTH or S["overlay_width"],
                                     dash=BEST_1D_DASH)
            tick_lines(bc, gl, book, f"pdf_{kx}", px, pyd, final[j], kx.startswith("log10_"), False)
            step("axes")(bc["set_axes"], gl, (px["lo"], px["hi"]), (0.0, ymax))
            ysd = "none" if c in et_next else "right"
            tasks[j] = step("panel axes")(bc["panel_axes"], gp, 0, kx, "pdf", px, pyd, xside, ysd, j) or []
            if r == c == 0:
                spots.append(("corner", j, (float(px["lo"]), float(px["hi"])), (0.0, ymax), 0))
            if extra.endswith("Et-Eg") and "log10_Nc" in px:
                et_pages[j] = (extra.split(":")[0], px)
        op.lt_exec(f"win -a {gp.name}; page.active = 1;")
        op.lt_exec("label -r legend;")

    # posterior predictive check (upper right)
    Q, QA = D["ppc"], D["ppc_attrs"]
    qx = {"lo": float(Q["x"][0]), "hi": float(Q["x"][-1]), "label_origin": QA["x_label_origin"]}
    qy = {"lo": float(Q["y"][0]), "hi": float(Q["y"][-1]), "label_origin": QA["y_label_origin"]}
    qx["label_origin"] = PPC_TITLES.get("tau_x", qx["label_origin"])
    qy["label_origin"] = PPC_TITLES.get("tau_y", qy["label_origin"])
    if PPC_TAU_RANGE:                                        # the axis exactly at the range asked for
        qy["lo"], qy["hi"] = [float(v) if v is not None else qy[k] for v, k in zip(PPC_TAU_RANGE, ("lo", "hi"))]
    r0, c0, rs, cs = ppc_cells(D)
    n = len(D["keys"])
    P_ = S["panel_cm"]
    # Few parameters (1 trap, n = 4): the upper triangle has no room for the PPC panels.  Then both are squares in a row
    # above the corner (tau_diff left, ss-PL right, geometry set before the corner pages), x labels below them and
    # horizontal, the ss-PL y axis on its right; the legend at the top of the free upper triangle.
    leg_geo = None
    if small:
        (ppc_geo, ss_geo), qx["rotate"] = small_geo, 0
        ppc_xside, ss_xside, ss_yside = "bottom", "bottom", "right"
        with_slice = SHOW_SLICE and len(D["slices"]) > 0 and n >= 4
        if with_slice and n > 4:                     # 2 traps: slice squares at the top right of the corner, legend below
            l_, t_, w_, _ = pos(0, n - 1)
            right = l_ + w_                          # right edge of the corner
            side_s = SLICE_SCALE * P_ if SLICE_SCALE else 3 * P_ + 2 * S["gap_cm"]
            # colour-bar title centre (cm right of the corner) = centre of the bottom-right density title, which
            # decade_labels places at gap + numbers + gap + half a title line (numbers estimated from their length)
            ymax = 1.1 * max(float(np.nanmax(D["panels"][D["keys"][-1]][f"{m}_pdf"])) for m in D["modes"])
            nch = max(len(f"{v:.2f}".rstrip("0").rstrip(".")) for v in (0.0, ymax, ymax / 2)) + 1
            pt_cm, cb_pt_cm = S["tick_pt"] * 0.03528, S["cbar_pt"] * 0.03528
            title_x = right + 2 * S["label_gap_cm"] + 0.56 * pt_cm * nch + 0.625 * S["title_pt"] * 0.03528
            # bar: 0.25 cm gap, 0.4 cm wide, labels from 1.35 bar widths ("99 %": ~4 characters), title 0.3 further
            bar_l = title_x - (0.4 * (1.35 + 0.3) + 0.56 * cb_pt_cm * 4 + 0.7 * cb_pt_cm) + SLICE_ALIGN_CM
            slice_geos = [(bar_l - 0.25 - side_s, t_ + i * (side_s + SLICE_VGAP_CM), side_s, side_s)
                          for i in range(len(D["slices"]))]
            cb_title_x = title_x + SLICE_ALIGN_CM
            off = SLICE_VGAP_CM
            lw_ = LEGEND_W_CM                        # right-aligned with the corner, below the last slice's labels
            leg_geo = (right - lw_, slice_geos[-1][1] + side_s + off, lw_, 7 * LEGEND_ROW_CM)
        elif with_slice:                             # legend in the free cells below the slice panel (col n - 1)
            l_, t_, w_, _ = pos(1, n - 1)
            off = 1.7                                # the slice panel's tilted x labels + title
            dx_ = 0.85                               # right of the density numbers of the diagonal panel (n-2, n-2)
            leg_geo = (l_ + dx_, t_ + off, w_, 2 * P_ + S["gap_cm"] - off)
        elif n >= 4:
            leg_geo = pos(0, n - 2, 2, 2)
    else:
        l_, t_, w_, h_ = pos(rs, n - 2, 2, 2)
        ppc_geo, ss_geo = pos(r0, c0, rs, cs), (l_, t_ + 0.6 * P_, w_, h_ - 0.6 * P_)
        ppc_xside, ss_xside, ss_yside = "bottom", "bottom", "left"
    gp, gl, jp = new_page("ppc", ppc_geo)
    gp, gl = fill_panel("ppc", gp, gl, Q)
    band_lines(bc, gl, book, "ppc", D, Q["data_qfls"], Q)
    w = step("sheet")(bc["new_sheet_xy"], book, "ppc_median_joint", [("QFLS (eV)", Q["median_x"]), ("median", Q["median_y"])])
    if w is not None:
        step("median")(bc["line_plot"], gl, w, 0, 1, MEDIAN_COLOR, LINE_WIDTH["median"], dash=DASH["median"])
    best_curve(bc, gl, book, "ppc_best", Q, "QFLS (eV)")
    data_points(bc, gl, book, "ppc_data", Q["data_qfls"], Q["data_y"], ppc_geo, (qx["lo"], qx["hi"]), (qy["lo"], qy["hi"]))
    for m in D["modes"]:                                     # the bands as numbers too (sheet only)
        step("sheet")(bc["new_sheet_xy"], book, f"ppc_{m}", [("QFLS (eV)", Q["data_qfls"])] +
                      [(f"q{q}", Q[f"{m}_q{q}"]) for q in ("2.5", "16", "50", "84", "97.5")])
    tick_lines(bc, gl, book, "ppc", qx, qy, final[jp], False, True)
    step("axes")(bc["set_axes"], gl, (qx["lo"], qx["hi"]), (qy["lo"], qy["hi"]))
    with ppc_ticks(S):
        tasks[jp] = step("panel axes")(bc["panel_axes"], gp, 0, "qfls", "log10_tau", qx, qy, ppc_xside, "left", jp) or []
    spots.append(("ppc", jp, (qx["lo"], qx["hi"]), (qy["lo"], qy["hi"]), 0))
    op.lt_exec("label -r legend;")

    # ss-PL predictive check, below the tau_diff(QFLS) panel
    if "sspl" in D:
        Q2, A2 = D["sspl"], D["sspl_attrs"]
        sx = {"lo": float(Q2["x"][0]), "hi": float(Q2["x"][-1]), "label_origin": A2["x_label_origin"]}
        if small:
            sx["rotate"] = 0
        sy = {"lo": float(Q2["y"][0]), "hi": float(Q2["y"][-1]), "label_origin": A2["y_label_origin"]}
        sx["label_origin"] = PPC_TITLES.get("sspl_x", sx["label_origin"])
        sy["label_origin"] = PPC_TITLES.get("sspl_y", sy["label_origin"])
        xlog, ylog = int(A2.get("x_log10", 1)), int(A2.get("y_log10", 0))   # plqy_vs_qfls: linear QFLS x, log PLQY y
        kx2 = "log10_illumination" if xlog else "qfls"
        ky2 = "log10_plqy" if ylog else "sspl_y"
        gp, gl, js = new_page("sspl_ppc", ss_geo)
        gp, gl = fill_panel("sspl_ppc", gp, gl, Q2)
        xn = "log10 illumination" if xlog else "QFLS (eV)"
        band_lines(bc, gl, book, "sspl", D, Q2["x"], Q2)
        w = step("sheet")(bc["new_sheet_xy"], book, "sspl_median_joint", [(xn, Q2["median_x"]), ("median", Q2["median_y"])])
        if w is not None:
            step("median")(bc["line_plot"], gl, w, 0, 1, MEDIAN_COLOR, LINE_WIDTH["median"], dash=DASH["median"])
        best_curve(bc, gl, book, "sspl_best", Q2, xn)
        data_points(bc, gl, book, "sspl_data", Q2["data_x"], Q2["data_y"], ss_geo, (sx["lo"], sx["hi"]), (sy["lo"], sy["hi"]))
        for m in D["modes"]:
            step("sheet")(bc["new_sheet_xy"], book, f"sspl_{m}", [(xn, Q2["x"])] +
                          [(f"q{q}", Q2[f"{m}_q{q}"]) for q in ("2.5", "16", "50", "84", "97.5")])
        tick_lines(bc, gl, book, "sspl_ppc", sx, sy, final[js], bool(xlog), bool(ylog))
        step("axes")(bc["set_axes"], gl, (sx["lo"], sx["hi"]), (sy["lo"], sy["hi"]))
        with ppc_ticks(S):
            tasks[js] = step("panel axes")(bc["panel_axes"], gp, 0, kx2, ky2, sx, sy, ss_xside, ss_yside, js) or []
        spots.append(("sspl", js, (sx["lo"], sx["hi"]), (sy["lo"], sy["hi"]), 0))
        op.lt_exec("label -r legend;")

    # slice panel: 1 trap at the right end of row 0, 2 traps a 3 x 3 square in the upper right; its colour bar right of
    # it, iso-tau_0 lines labelled in the panel
    jcb, slice_text, cb, cb_info = None, [], None, None      # slice_text: (page j, regime, text, x, y, side)

    def draw_slice(i, sl, geo_s, rotate=None, text_at=0.5):
        Sd, SA, SP = sl["d"], sl["attrs"], sl["params"]
        kx_s, ky_s = SA["kx"], SA["ky"]
        px_s, py_s = dict(SP[kx_s]), dict(SP[ky_s])
        if rotate is not None:
            px_s["rotate"] = rotate
        levels = np.array(json.loads(str(SA["hpd_levels"]).replace(" ", ",").replace(",,", ",").replace("[,", "["))
                          if isinstance(SA["hpd_levels"], str) else SA["hpd_levels"], float).ravel()
        bnd = bc["colour_bounds"](levels)
        bcols = json.loads(SA["band_colors_json"]) if "band_colors_json" in SA else list(bc["BAND_COLORS"])
        if len(bcols) != len(bnd) - 1:
            bcols = list(bc["BAND_COLORS"])
        x_, y_ = np.asarray(Sd["x"], float), np.asarray(Sd["y"], float)
        Z = np.clip(100.0 * np.asarray(Sd["hpd"], float), 0.0, 100.0)
        Z = np.where(np.isfinite(Z), Z, 100.0)
        lo_x, hi_x, lo_y, hi_y = (float(px_s["lo"]), float(px_s["hi"]), float(py_s["lo"]), float(py_s["hi"]))
        Z, x_, y_ = bc["regrid"](Z, x_, y_, lo_x, hi_x, lo_y, hi_y)
        nm_ = "slice" if i == 0 else f"slice_{i + 1}"
        gp, gl, js = new_page(nm_, geo_s)
        gp, gl = fill_panel(nm_, gp, gl, {"x": x_, "y": y_}, (Z, bnd, bcols))
        ux = (hi_x - lo_x) / geo_s[2]
        for reg in ("photodoping", "non_photodoping", "photodoping_ntratio", "non_photodoping_ntratio",
                    "taun_plus_taup"):
            v = Sd.get(f"tau0_{reg}")
            if v is None or not len(v):
                continue
            w = step("sheet")(bc["new_sheet_xy"], book, f"{nm_}_tau0_{reg}",
                              [(f"x {kx_s}", v[:, 0]), (f"y {ky_s}", v[:, 1])])
            if w is not None:
                step("tau0 line")(bc["line_plot"], gl, w, 0, 1, TAU0_COLOR[reg], TAU0_WIDTH or S["overlay_width"],
                                  dash=TAU0_DASH[reg])
            ok = v[np.isfinite(v[:, 0]) & np.isfinite(v[:, 1])]
            if len(ok):                              # text next to the middle of the line, 0.15 cm beside it, on the
                mx, my = ok[int(text_at * (len(ok) - 1))]   # side towards the panel centre (stays inside the panel)
                sgn = 1 if mx < 0.5 * (lo_x + hi_x) else -1
                slice_text.append((js, reg, TAU0_TEXT[reg], mx + sgn * 0.15 * ux, my, sgn))
        marks(bc, gl, book, nm_, px_s, py_s, geo_s, color=SLICE_BEST_COLOR, edge=SLICE_BEST_EDGE)
        tick_lines(bc, gl, book, nm_, px_s, py_s, geo_s, kx_s.startswith("log10_"), ky_s.startswith("log10_"),
                   color=SLICE_TICK_COLOR)
        step("axes")(bc["set_axes"], gl, (lo_x, hi_x), (lo_y, hi_y))
        tasks[js] = step("panel axes")(bc["panel_axes"], gp, 0, kx_s, ky_s, px_s, py_s, "bottom", "left", js) or []
        spots.append(("slices", js, (lo_x, hi_x), (lo_y, hi_y), i))
        op.lt_exec("label -r legend;")
        title = bc["CBAR_TITLES"].get(SA.get("colour_bar_title", ""), SA.get("colour_bar_title", ""))
        return bnd, bcols, (bnd, levels, title)

    def slice_bar(box, bnd, bcols):
        cbp = step("colour bar")(bc["colour_bar_page"], box, bnd, bcols)
        if cbp is None:
            return None
        pages.append(cbp)
        final[len(pages) - 1] = box
        cmaps.append((cbp, bnd, bcols, False))
        return len(pages) - 1

    if small and SHOW_SLICE and D["slices"] and n == 4:
        geo_s = pos(0, n - 1)
        bnd, bcols, cb_info = draw_slice(0, D["slices"][0], geo_s, text_at=TAU0_TEXT_AT)
        cb = (geo_s[0] + geo_s[2] + 0.25, geo_s[1], 0.4, P_ + S["gap_cm"] + 1.5)     # bar down to the legend
        jcb = slice_bar(cb, bnd, bcols)
    elif small and SHOW_SLICE and D["slices"]:
        for i, sl in enumerate(D["slices"]):
            bnd, bcols, cb_info = draw_slice(i, sl, slice_geos[i], rotate=0, text_at=TAU0_TEXT_AT)
        g0, g1 = slice_geos[0], slice_geos[-1]
        cb = (g0[0] + g0[2] + 0.25, g0[1], 0.4, g1[1] + g1[3] - g0[1])     # one bar along all slice panels
        jcb = slice_bar(cb, bnd, bcols)
        cb_info = cb_info + ((cb_title_x - cb[0]) / cb[2],)                  # title position in bar widths

    # legend in a layer of its own (small figures: no room in the PPC panel)
    jl = None
    if leg_geo is not None:
        gp, gl, jl = new_page("legend", leg_geo)
        step("axes")(bc["set_axes"], gl, (0.0, 1.0), (0.0, 1.0))
        hide = ("layer.x.showAxes = 0; layer.y.showAxes = 0; layer.x.ticks = 0; layer.y.ticks = 0; layer.x2.ticks = 0;"
                " layer.y2.ticks = 0; " + bc["label_sides"](0, 0, 0, 0))
        step("legend axes")(bc["lt"], f"win -a {gp.name}; page.active = 1; " + hide)
        op.lt_exec("xb.show = 0;")
        op.lt_exec("yl.show = 0;")
        op.lt_exec("label -r legend;")
        bc["REPLAY"][(jl, 1)] = hide

    # Second colour pass: in the first Origin run every 4th colour map stayed in Origin's default (solid red) although
    # no statement failed; now that all pages are drawn, the levels and colours are written once more on every page
    for gp_, bnd, cols, contour in cmaps:
        op.lt_exec(f"win -a {gp_.name};")
        op.lt_exec("doc -uw;")
        if not bc["cmap_ready"](gp_, 0):                 # never written over a layer without a colour map
            bc["LOG"].append(f"[colour levels (2nd pass)] {gp_.lname}: no colour map, skipped")
            continue
        step("colour levels (2nd pass)")(bc["set_cmap"], gp_, 0, bnd, cols)
        if contour:
            op.lt_exec("layer.cmap.showLines(3);")
            op.lt_exec("layer.cmap.showLabels(3);")
            op.lt_exec("layer.cmap.updateScale();")
    print(f"colour maps: {len(cmaps)} pages, levels and colours written twice")

    # merge, put every layer back, formatting again, 10^n labels, legend, fit the page
    geos = [step("tag layers")(bc["tag_layers"], p, j, final[j]) or [] for j, p in enumerate(pages)]
    merged = step("merge")(bc["merge_pages"], pages)
    if merged is None:
        print("merge failed: panels left as separate graphs")
        return
    gp = merged
    step("page height")(bc["page_height"], gp, S["merged_page_cm"])
    first = step("place layers")(bc["place_merged"], gp, geos) or {}
    for (j, k), sc in bc["REPLAY"].items():
        if (j, k) in bc["MERGED"]:
            step("axis format after merge")(bc["lt"], f"win -a {gp.name}; page.active = {bc['MERGED'][(j, k)] + 1}; {sc}")
    ppc_pages = {jp, js} if "sspl" in D else {jp}
    for j, tl in tasks.items():
        for t in tl:
            if j in first:
                with ppc_ticks(S, j in ppc_pages):
                    step("10^n labels")(bc["decade_labels"], gp, first[j], *t)
    for j, (where, px) in et_pages.items():                  # formula-linked to the merged panel layer (build_corner)
        if j in first:
            step("E_t - E_g axis")(bc["draw_et"], gp, first[j], where, px)
    for i, (js_, reg, text, tx_, ty_, sgn) in enumerate(slice_text):
        if js_ in first:
            nm = f"t0txt{i}"
            step("tau0 text")(bc["lt"], f"win -a {gp.name}; page.active = {first[js_] + 1};"
                                        f" label -a 0 0 -n {nm} {text}; {nm}.fsize = {S['cbar_pt']}; {nm}.attach = 2;"
                                        f" {nm}.color = {bc['lt_rgb'](TAU0_COLOR[reg])};"
                                        f" {nm}.x = {tx_} {'+' if sgn > 0 else '-'} {nm}.dx / 2; {nm}.y = {ty_};")
    for i, (key, j_, xr, yr, k_) in enumerate(spots):          # panel letters, in data units of that panel
        spec = LETTERS.get(key)
        if isinstance(spec, list):
            spec = spec[k_] if k_ < len(spec) else None
        if not spec or j_ not in first:
            continue
        text, fx, fy, col = spec
        nm = f"letter{i}"
        step("panel letter")(bc["lt"], f"win -a {gp.name}; page.active = {first[j_] + 1};"
                                       f" label -a 0 0 -n {nm} {text}; {nm}.text$ = \"{text}\";"
                                       f" {nm}.fsize = {LETTER_PT}; {nm}.attach = 2;"
                                       f" {nm}.color = {bc['lt_rgb'](col)};"
                                       f" {nm}.x = {xr[0] + fx * (xr[1] - xr[0])}; {nm}.y = {yr[0] + fy * (yr[1] - yr[0])};")
    if jcb is not None and jcb in first:
        step("colour bar text")(cbar_text, bc, gp, first[jcb], cb, *cb_info)
    unit = {"lo": 0.0, "hi": 1.0}
    if jl is not None and jl in first:
        step("legend")(legend, bc, gp, first[jl], D, unit, unit, 0.05, 0.97, leg_geo[3], leg_geo[2])
    elif jp in first:
        step("legend")(legend, bc, gp, first[jp], D, qx, qy, *LEGEND_POS, ppc_geo[3], ppc_geo[2])
    step("fit page")(bc["lt"], f"win -a {gp.name}; pfit2l margin:=0 borderwid:=2;")
    step("page size")(bc["page_size_check"], gp)

    base = os.path.join(os.path.dirname(path), "corner_modes")
    step("project")(bc["save_project"], base + ".opju")
    step("export pdf")(gp.save_fig, base + ".pdf")
    step("export png")(gp.save_fig, base + ".png", width=2400)
    print(f"done: {base}.opju / .pdf / .png")
    if bc["LOG"]:
        print("\nsteps that failed (the rest of the figure was built):")
        for m in bc["LOG"]:
            print("  ", m)
    if bc["LT_FAILED"]:
        print("\nLabTalk statements that failed (count, first example):")
        for n, ex in bc["LT_FAILED"].values():
            print(f"   {n:4d} x  {ex}")


if not globals().get("_NO_MAIN"):                       # build_corner_nn_vs_ode.py loads this file as a library
    main()
