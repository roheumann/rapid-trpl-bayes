"""
build_sbc.py — simulation-based calibration (SBC) figure for the SI in Origin (2021+, Windows): 1 or 2 traps
=============================================================================================================
Input : the output folder of scripts/sbc_compare.py (ecdf_curves.csv, ecdf_bands.csv) and its summary CSV
        (<folder>/*_summary.csv or <folder>_summary.csv next to the folder), written on rekenbeest, e.g.
          1 trap : results/1trap_synthetic_sigtr_GaAs/synthetic/sbc_1trap/
          2 traps: results/sbc_2trap_85-15/  (summary: results/sbc_2trap_85-15_summary.csv)
        200 data sets per mode, 100 posterior draws each; variant A = NN data, B = ODE data (paired), where present
Output: <folder>/<folder name>.opju (workbook + graph), .pdf, .png

Run inside Origin:   run -pyf "C:\\...\\corner-plot\\origin\\build_sbc.py"
A file dialog asks for ecdf_curves.csv of the folder (CURVES = None); no extra Python packages needed (csv + numpy).

Figure: one row per measurement mode (tr-PL, ss-PL, joint) and trap (2 traps: two rows per mode, "trap 1" / "trap 2"),
columns = N_t, E_t, tau_n, tau_p of that trap + coverage.
  ECDF panels  ECDF of the fractional rank of the true value minus the uniform ECDF (0 = perfectly calibrated), with
               the simultaneous 95 % band of a uniform rank sample (grey; Saeilynoja et al. 2022).  NN data solid,
               ODE data (B, where it exists) dashed in the same colour; p = simultaneous ECDF p-value (A / B).
               A curve leaving the band = miscalibration: a U shape = too narrow posteriors, an inverted U = too wide,
               a one-sided bulge = bias.
  coverage     empirical minus nominal coverage of the central 50 / 68 / 95 % credible intervals, +- 1 binomial SE,
               one marker per parameter (0 = nominal); ODE data as lighter markers just right of the NN-data ones.
Every panel is layer 1 of its own graph page; the pages are merged with merge_graph and each layer is put back by
name, then its axis formatting is applied again and all texts (titles, p-values, row names, legend) are added - the
build_corner.py recipe.  Origin runs of 2026-10-26 with one page and `layer -n` layers: the tick-label and showAxes
switches only took effect in layer 1, so the inner panels kept their tick labels.  Also learned there: '=' in label
text fails (written as \\(61)); symbol plots stay invisible (coverage markers are line outlines); the band is filled
with vertical line segments (as the bars of build_error_histogram.py) plus thin edge lines.
"""
import csv
import glob
import os
import re
import traceback

import numpy as np
import originpro as op

CURVES = None              # None -> file dialog (ecdf_curves.csv); e.g.
# r"R:\simulations\rapid-trpl-bayes\results\sbc_2trap_85-15\ecdf_curves.csv"
MODES = [("trpl", "tr-PL", "#2166AC"), ("sspl", "ss-PL", "#525252"), ("joint", "joint", "#B2182B")]   # modes corner colours
PARAMS = [("log10_Nt_{k}", r"\i(N)\-(t)"), ("Et_{k}_eV", r"\i(E)\-(t)"), ("log10_taun_{k}", r"\g(t)\-(n)"),
          ("log10_taup_{k}", r"\g(t)\-(p)")]            # per trap k
VARIANTS = [("A", "NN data", 0), ("B", "ODE data", 1)]   # (summary suffix, legend text, Origin -d: 0 solid, 1 dash)
B_LIGHTEN = 0.5            # ODE-data coverage markers: mode colour mixed with this much white
MARKERS = ["circle", "square", "triangle", "diamond"]   # coverage-panel marker per parameter (line outlines)
MARKER_CM = 0.055          # marker radius
BAND = "simultaneous"      # or "pointwise"
Y_ECDF = (-0.13, 0.13)     # y range of the ECDF panels (band reaches +-0.10 for 200 data sets)
X_ECDF = (-0.04, 1.04)     # x range: a little room so the 1.0 / 0.0 labels of neighbouring panels do not touch
Y_COV = (-0.12, 0.12)      # y range of the coverage panels (empirical - nominal)
LEVELS = [(50, 0.50), (68, 0.68), (95, 0.95)]
COV_DX = {1: 0.03, 2: 0.045}   # x offset between the parameters' markers at one level, without / with ODE data
COV_DX_B = 0.016           # ODE-data marker this far right of the NN-data marker of the same parameter
X_COV = (0.38, 1.06)       # x range of the coverage panels
STYLE = {"font": "Arial", "tick_pt": 9, "title_pt": 11, "text_pt": 8, "legend_pt": 8, "row_pt": 9,
         "lw_curve": 1.5, "lw_curve_b": 1.2, "lw_edge": 0.5, "lw_zero": 0.6, "lw_err": 1.0, "lw_marker": 1.0}
BAND_FILL, BAND_EDGE = "#DCDCDC", "#A6A6A6"
PANEL_CM = (3.0, 2.4)      # width, height of every panel
GAP_CM = (0.25, 0.35)      # horizontal, vertical gap between panels
TRAP_GAP_CM = 0.45         # 2 traps: extra gap between the row pairs of two modes
COV_GAP_CM = 0.9           # extra gap left of the coverage column (its y labels sit on the right)
ORIGIN_CM = (2.9, 1.2)     # left, top of the first panel on the page
ROW_LABEL_CM = 2.4         # mode name this far left of the first column (centre of the rotated text)
TRAP_LABEL_CM = 1.95       # 2 traps: "trap 1" / "trap 2" this far left
YTITLE_CM = 1.2            # y titles this far left (ECDF) / right (coverage) of the frame (centre of the rotated text)
XTITLE_CM = 0.85           # x titles this far below the bottom row
COLTITLE_CM = 0.35         # column titles this far above the top row
LEGEND_CM = (0.0, 1.5, 3.0, 2.2)   # legend layer: offset left / below the bottom coverage panel, width, height
MERGED_PAGE_CM = 60.0      # height of the merged page before pfit2l fits it to the figure

LOG, LT_FAILED = [], {}


def step(name):
    def wrap(f, *a, **k):
        try:
            return f(*a, **k)
        except Exception as e:                          # noqa: BLE001  (report and continue)
            LOG.append(f"[{name}] {type(e).__name__}: {e}")
            print(LOG[-1])
            traceback.print_exc()
            return None
    return wrap


def lt(cmd):
    """LabTalk statement by statement; failures are listed at the end of the run."""
    ok = True
    for s in [s.strip() for s in cmd.split(";") if s.strip()]:
        if op.lt_exec(s + ";") is False:
            ok = False
            key = re.sub(r"-?\d+(\.\d+)?(e[-+]?\d+)?", "#", s)[:120]
            LT_FAILED.setdefault(key, [0, s])[0] += 1
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# reading
# ─────────────────────────────────────────────────────────────────────────────
def pick_curves():
    if CURVES and os.path.isfile(CURVES):
        return CURVES
    lt('dlgfile group:="ecdf_curves.csv" title:="ecdf_curves.csv in the sbc_compare.py output folder";'
       ' string __f$ = fname$;')
    return op.get_lt_str("__f$")


def read_rows(path):
    """CSV rows as dicts; BOM stripped and ';' accepted (a copy re-saved by Excel with a German locale)."""
    with open(path, newline="", encoding="utf-8-sig") as fh:
        text = fh.read()
    head = text.split("\n", 1)[0]
    delim = ";" if head.count(";") > head.count(",") else ","
    rows = list(csv.DictReader(text.splitlines(), delimiter=delim))
    fix = (lambda v: v.strip().replace(",", ".")) if delim == ";" else (lambda v: v.strip())   # decimal commas
    return [{k.strip(): (fix(v) if isinstance(v, str) else v) for k, v in r.items() if k} for r in rows]


SUMMARY_COLS = ("quantity", "mode", "subset", "cov50_A")


def summary_csv(folder):
    """The sbc_compare.py summary: <folder>/*_summary.csv (1-trap layout) or <folder>_summary.csv next to it
    (2-trap layout).  Only files whose header has the summary columns count, so other *_summary.csv are skipped."""
    beside = folder.rstrip("\\/") + "_summary.csv"
    cands = sorted(glob.glob(os.path.join(folder, "*_summary.csv"))) + ([beside] if os.path.isfile(beside) else [])
    for f in cands:
        rows = read_rows(f)
        if rows and all(c in rows[0] for c in SUMMARY_COLS):
            return f
        LOG.append(f"[summary] skipped {f}: columns {list(rows[0])[:6] if rows else 'none'} ...")
    raise FileNotFoundError(f"no sbc_compare.py summary CSV (columns {SUMMARY_COLS}) in or next to {folder}; "
                            f"checked {cands}")


def read_data(curves_csv):
    folder = os.path.dirname(curves_csv)
    curves, bands, stats = {}, {}, {}
    for r in read_rows(curves_csv):
        c = curves.setdefault((r["mode"], r["variant"][0], r["quantity"]), ([], []))
        c[0].append(float(r["z"])); c[1].append(float(r["ecdf_minus_uniform"]))
    for r in read_rows(os.path.join(folder, "ecdf_bands.csv")):
        if r["band"] == BAND:
            b = bands.setdefault(r["mode"], ([], [], []))
            b[0].append(float(r["z"])); b[1].append(float(r["lo"])); b[2].append(float(r["hi"]))
    for r in read_rows(summary_csv(folder)):
        if r["subset"] == "all":
            stats[(r["mode"], r["quantity"])] = r
    return folder, curves, bands, stats


def fval(s, key):
    """Float of a summary cell, None if the column is missing or empty (variant B not run for this mode)."""
    v = (s or {}).get(key, "")
    return float(v) if v not in ("", None) else None


def lighten(hexcol, f):
    c = [int(hexcol[i:i + 2], 16) for i in (1, 3, 5)]
    return "#%02X%02X%02X" % tuple(int(round(v + f * (255 - v))) for v in c)


# ─────────────────────────────────────────────────────────────────────────────
# Origin helpers (one page per panel, merge_graph, layers put back by name: as build_corner.py)
# ─────────────────────────────────────────────────────────────────────────────
def sheet(book, name, cols):
    """cols: list of (long name, array) -> new worksheet in book."""
    wks = book.add_sheet(name[:31])
    for i, (lname, arr) in enumerate(cols):
        wks.from_list(i, [float(v) for v in np.asarray(arr, float)], lname=lname)
    return wks


class Layer:
    def __init__(self, gp, idx):
        self.gp, self.idx, self.gl = gp, idx, gp[idx]

    def activate(self):
        lt(f"win -a {self.gp.name}; page.active = {self.idx + 1};")

    def line(self, wks, cx, cy, color, width, dash=0):
        p = self.gl.add_plot(wks, coly=cy, colx=cx, type="l")
        p.color = color
        p.set_cmd(f"-w {int(width * 500)}")             # 1 pt = 500
        if dash:
            p.set_cmd(f"-d {dash}")
        return p


def label_sides(xb, xt, yl, yr):
    return (f"layer.x.showLabels = {xb + 2 * xt}; layer.y.showLabels = {yl + 2 * yr}; layer.x.showlabel = {xb};"
            f" layer.x2.showlabel = {xt}; layer.y.showlabel = {yl}; layer.y2.showlabel = {yr};")


def axes_cmd(x, y, xinc, yinc, xlabels, ylabels, yright=False):
    """LabTalk of one panel's axes: linear scales, ticks inward on all four sides, tick labels only where asked (last)."""
    s = (f"layer.x.type = 1; layer.x.from = {x[0]:.6g}; layer.x.to = {x[1]:.6g}; layer.x.inc = {xinc:.6g};"
         f" layer.y.type = 1; layer.y.from = {y[0]:.6g}; layer.y.to = {y[1]:.6g}; layer.y.inc = {yinc:.6g};"
         " layer.x.showAxes = 3; layer.y.showAxes = 3;"
         f" layer.x.label.pt = {STYLE['tick_pt']}; layer.y.label.pt = {STYLE['tick_pt']};"
         f" layer.x.label.font = font({STYLE['font']}); layer.y.label.font = font({STYLE['font']}); ")
    s += " ".join(f"layer.{a}.thickness = 0.8; layer.{a}.tickthickness = 0.8; layer.{a}.ticklength = 4;"
                  f" layer.{a}.mticklength = 2; layer.{a}.ticks = 5; layer.{a}.mtickthickness = 0.8;"
                  for a in ("x", "y", "x2", "y2"))
    s += (" layer.x2.minorTicks = layer.x.minorTicks; layer.y2.minorTicks = layer.y.minorTicks;"
          " layer.x2.inc = layer.x.inc; layer.y2.inc = layer.y.inc; ")
    return s + label_sides(int(xlabels), 0, int(ylabels and not yright), int(ylabels and yright))


HIDDEN_AXES = ("layer.x.from = 0; layer.x.to = 1; layer.y.from = 0; layer.y.to = 1; layer.x.showAxes = 0;"
               " layer.y.showAxes = 0; layer.x.ticks = 0; layer.y.ticks = 0; layer.x2.ticks = 0; layer.y2.ticks = 0; "
               + label_sides(0, 0, 0, 0))


def text(L, name, txt, x, y, pt, rotate=0, color=None):
    """Text centred at (x, y) in the axis units of layer L (may lie outside the frame)."""
    L.activate()
    lt(f"label -s -sa -n {name} {txt}; {name}.fsize = {pt};")
    if rotate:
        lt(f"{name}.rotate = {rotate};")
    if color:
        lt(f"{name}.color = color({color});")
    lt(f"{name}.x = {x:.6g}; {name}.y = {y:.6g};")


def segments(x0, y0, x1, y1):
    """Many line segments as one NaN-separated polyline (Origin breaks a line plot at missing values)."""
    xs, ys = [], []
    for a, b, c, d in zip(x0, y0, x1, y1):
        xs += [a, c, np.nan]; ys += [b, d, np.nan]
    return np.array(xs), np.array(ys)


def outline(kind, x, y, ux, uy, r_cm=MARKER_CM):
    """Closed marker outline around (x, y) as a polyline; ux, uy: axis units per cm of the panel."""
    if kind == "circle":
        t = np.linspace(0, 2 * np.pi, 25)
        px, py = np.cos(t), np.sin(t)
    else:
        n, t0 = {"square": (4, np.pi / 4), "triangle": (3, np.pi / 2), "diamond": (4, np.pi / 2)}[kind]
        t = t0 + np.arange(n + 1) * 2 * np.pi / n
        px, py = np.cos(t), np.sin(t)
        if kind == "square":
            px, py = px / np.sqrt(2) * 1.25, py / np.sqrt(2) * 1.25
    return x + r_cm * ux * px, y + r_cm * uy * py


def outlines(kind, xs, ys, ux, uy):
    """Several outlines as one NaN-separated polyline."""
    ox, oy = [], []
    for x, y in zip(xs, ys):
        a, b = outline(kind, x, y, ux, uy)
        ox += list(a) + [np.nan]; oy += list(b) + [np.nan]
    return np.array(ox), np.array(oy)


def find_graph(name):
    try:
        return op.find_graph(name)
    except Exception:                                   # noqa: BLE001
        return next(g for g in op.pages("g") if g.name == name)


def merge(pages, geos):
    """merge_graph of the panel pages (sources deleted, fonts not rescaled), every layer put back at its geometry by
    the name M<j> given before the merge.  Returns the merged page and {j: 0-based layer index}."""
    for j, p in enumerate(pages):
        lt(f'win -a {p.name}; page.active = 1; layer.name$ = "M{j}";')
    names = "+char(10)$+".join(f'"{p.name}"' for p in pages)
    cmd = f"merge_graph option:=specified graphs:={names} keep:=0 groupgraph:=0"
    if not lt(f"{cmd} scalemode:=fixed fixfactor:=1;"):
        print("merge_graph with scale options failed; merging with defaults")
        lt(cmd + ";")
    op.lt_exec('string __mg$ = "%H";')
    gp = find_graph(op.get_lt_str("__mg$"))
    lt(f"win -a {gp.name}; page.height = {MERGED_PAGE_CM / 2.54:.4f} * page.resy;")
    n = int(op.lt_float("page.nlayers"))
    where = {}
    for i in range(1, n + 1):
        lt(f"page.active = {i}; string __ln$ = layer.name$;")
        m = re.fullmatch(r"M(\d+)", op.get_lt_str("__ln$").strip())
        if m:
            where[int(m.group(1))] = i - 1
    if len(where) != len(pages) and n == len(pages):     # names lost: assume merge order
        LOG.append("[merge] layer names lost in merge_graph; layers placed in merge order")
        where = {j: j for j in range(n)}
    for j, i in where.items():
        l, t, w, h = geos[j]
        lt(f"page.active = {i + 1}; layer.fixed = 1; layer.factor = 1; layer.unit = 3; layer.left = {l:.3f};"
           f" layer.top = {t:.3f}; layer.width = {w:.3f}; layer.height = {h:.3f};")
    print(f"merged page: {n} layers, {len(where)} of {len(pages)} placed by name")
    return gp, where


# ─────────────────────────────────────────────────────────────────────────────
# figure
# ─────────────────────────────────────────────────────────────────────────────
COV_UNITS = ((X_COV[1] - X_COV[0]) / PANEL_CM[0], (Y_COV[1] - Y_COV[0]) / PANEL_CM[1])   # coverage axis units per cm
ECDF_UNITS = ((X_ECDF[1] - X_ECDF[0]) / PANEL_CM[0], (Y_ECDF[1] - Y_ECDF[0]) / PANEL_CM[1])


def build(path):
    folder, curves, bands, stats = read_data(path)
    name = os.path.basename(folder.rstrip("\\/"))
    n_traps = 2 if any(q == "log10_Nt_2" for _, q in stats) else 1
    modes = [m for m in MODES if any(k[0] == m[0] for k in stats)]
    rows = [(m, k) for m in modes for k in range(1, n_traps + 1)]          # (mode tuple, trap)
    has_b = any(k[1] == "B" for k in curves)
    ncol, last = len(PARAMS) + 1, len(rows) - 1
    cov_dx = COV_DX[2 if has_b else 1]

    def frame(r, c):
        w, h = PANEL_CM
        left = ORIGIN_CM[0] + c * (w + GAP_CM[0]) + (COV_GAP_CM if c == len(PARAMS) else 0.0)
        top = ORIGIN_CM[1] + r * (h + GAP_CM[1]) + (r // n_traps) * (TRAP_GAP_CM if n_traps > 1 else 0.0)
        return left, top, w, h

    book = op.new_book("w", lname=name)
    pages, geos, fmts, keys = [], [], [], []

    def new_panel(key, geo, fmt):
        gp = op.new_graph(lname=f"{name}_{key}")
        lt(f"win -a {gp.name}; page.active = 1; layer.fixed = 1; layer.factor = 1; layer.unit = 3; layer.left = 0.5;"
           f" layer.top = 0.5; layer.width = {geo[2]:.3f}; layer.height = {geo[3]:.3f};")
        op.lt_exec("xb.show = 0;")                      # the template's axis titles (texts are added after the merge)
        op.lt_exec("yl.show = 0;")
        op.lt_exec("label -r legend;")
        pages.append(gp); geos.append(geo); fmts.append(fmt); keys.append(key)
        return Layer(gp, 0)

    for r, ((mode, mlabel, mcol), k) in enumerate(rows):
        z, lo, hi = (np.asarray(v) for v in bands.get(mode, ([], [], [])))
        wb = sheet(book, f"band_{mode}_{k}", [("rank fraction", z), ("band low", lo), ("band high", hi),
                                              *zip(("fill x", "fill y"), segments(z, lo, z, hi))])
        fill_pt = 1.4 * (z[1] - z[0]) / ECDF_UNITS[0] / 0.03528 if len(z) > 1 else 1.0   # band point spacing + overlap
        for c, (qf, _) in enumerate(PARAMS):
            q = qf.format(k=k)
            lay = new_panel((r, c), frame(r, c), axes_cmd(X_ECDF, Y_ECDF, 0.5, 0.1, r == last, c == 0))
            lay.line(wb, 3, 4, BAND_FILL, fill_pt)
            lay.line(wb, 0, 1, BAND_EDGE, STYLE["lw_edge"])
            lay.line(wb, 0, 2, BAND_EDGE, STYLE["lw_edge"])
            cols = [("zero x", list(X_ECDF)), ("zero y", [0, 0])]
            for v, _, _ in VARIANTS:
                zc, dc = curves.get((mode, v, q), ([], []))
                cols += [(f"rank fraction {v}", zc), (f"ECDF - uniform {v}", dc)]
            wc = sheet(book, f"ecdf_{mode}_{q}"[:31], cols)
            lay.line(wc, 0, 1, "#000000", STYLE["lw_zero"], 1)
            for i, (v, _, dash) in reversed(list(enumerate(VARIANTS))):   # A last = on top
                if (mode, v, q) in curves:
                    lay.line(wc, 2 + 2 * i, 3 + 2 * i, mcol, STYLE["lw_curve" if v == "A" else "lw_curve_b"], dash)
            lt(f"win -a {lay.gp.name}; page.active = 1; " + fmts[-1])
        cov = new_panel((r, ncol - 1), frame(r, ncol - 1), axes_cmd(X_COV, Y_COV, 0.2, 0.05, r == last, True, True))
        cols, plots = [], []
        for j, (qf, _) in enumerate(PARAMS):
            q = qf.format(k=k)
            s = stats.get((mode, q))
            for v, _, _ in VARIANTS:
                y = [fval(s, f"cov{p}_{v}") for p, _ in LEVELS]
                if None in y:
                    continue
                dx = (j - (len(PARAMS) - 1) / 2) * cov_dx + (COV_DX_B if v == "B" else 0.0)
                x = [lv + dx for _, lv in LEVELS]
                y = [yy - lv for yy, (_, lv) in zip(y, LEVELS)]
                se = [fval(s, f"cov{p}_se_{v}") or 0.0 for p, _ in LEVELS]
                ex, ey = segments(x, np.subtract(y, se), x, np.add(y, se))
                mx, my = outlines(MARKERS[j % len(MARKERS)], x, y, *COV_UNITS)
                n0 = len(cols)
                cols += [(f"{q} {v} nominal", x), (f"{q} {v} emp - nom", y), (f"{q} {v} err x", ex),
                         (f"{q} {v} err y", ey), (f"{q} {v} marker x", mx), (f"{q} {v} marker y", my)]
                plots.append((n0, mcol if v == "A" else lighten(mcol, B_LIGHTEN)))
        wv = sheet(book, f"coverage_{mode}_{k}", cols + [("zero x", list(X_COV)), ("zero y", [0, 0])])
        cov.line(wv, len(cols), len(cols) + 1, "#000000", STYLE["lw_zero"], 1)
        for n0, colr in plots:
            cov.line(wv, n0 + 2, n0 + 3, colr, STYLE["lw_err"])
            cov.line(wv, n0 + 4, n0 + 5, colr, STYLE["lw_marker"])
        lt(f"win -a {cov.gp.name}; page.active = 1; " + fmts[-1])

    # legend: parameter markers, and NN / ODE data line styles if both exist (panel with hidden axes 0..1)
    l0, t0, _, h0 = frame(last, ncol - 1)
    lw_, lh_ = LEGEND_CM[2], LEGEND_CM[3]
    leg = new_panel("legend", (l0 + LEGEND_CM[0], t0 + h0 + LEGEND_CM[1], lw_, lh_), HIDDEN_AXES)
    n_leg = len(PARAMS) + (len(VARIANTS) if has_b else 0)
    ys = [1 - (i + 0.5) / n_leg for i in range(n_leg)]
    cols = []
    for j in range(len(PARAMS)):
        mx, my = outline(MARKERS[j % len(MARKERS)], 0.08, ys[j], 1 / lw_, 1 / lh_)
        cols += [(f"legend {j} x", mx), (f"legend {j} y", my)]
    if has_b:
        for i in range(len(VARIANTS)):
            cols += [(f"legend {VARIANTS[i][0]} x", [0.0, 0.18]), (f"legend {VARIANTS[i][0]} y", [ys[len(PARAMS) + i]] * 2)]
    wl = sheet(book, "legend", cols)
    for j in range(len(PARAMS)):
        leg.line(wl, 2 * j, 2 * j + 1, "#000000", STYLE["lw_marker"])
    if has_b:
        for i, (_, _, dash) in enumerate(VARIANTS):
            c0 = 2 * (len(PARAMS) + i)
            leg.line(wl, c0, c0 + 1, "#000000", STYLE["lw_curve"], dash)
    lt(f"win -a {leg.gp.name}; page.active = 1; " + fmts[-1])

    # ── merge, formatting again, texts ───────────────────────────────────────
    gp, where = merge(pages, geos)
    for j, i in where.items():
        lt(f"win -a {gp.name}; page.active = {i + 1}; {fmts[j]}")
    L = {keys[j]: Layer(gp, i) for j, i in where.items()}
    t = STYLE["title_pt"]
    mid = (len(rows) - 1) / 2                                   # y titles centred on the row block
    for r, ((mode, mlabel, mcol), k) in enumerate(rows):
        for c, (qf, qlabel) in enumerate(PARAMS):
            lay = L.get((r, c))
            if lay is None:
                continue
            s = stats.get((mode, qf.format(k=k)))
            ps = [fval(s, f"p_ecdf_sim_{v}") for v, _, _ in VARIANTS]
            ps = [f"{p:.2f}" if p >= 0.01 else f"{p:.3f}" for p in ps if p is not None]
            if ps:
                text(lay, f"p{r}_{c}", "\\i(p) \\(61) " + " / ".join(ps), sum(X_ECDF) / 2,
                     Y_ECDF[1] - 0.15 * (Y_ECDF[1] - Y_ECDF[0]), STYLE["text_pt"])
            if r == 0:
                text(lay, f"ct{c}", qlabel, sum(X_ECDF) / 2, Y_ECDF[1] + COLTITLE_CM * ECDF_UNITS[1], t)
            if r == last:
                text(lay, f"xt{c}", "fractional rank", sum(X_ECDF) / 2, Y_ECDF[0] - XTITLE_CM * ECDF_UNITS[1], t)
            if c == 0:
                gap_y = Y_ECDF[0] - (GAP_CM[1] / 2) * ECDF_UNITS[1]      # middle of the gap below this row
                if n_traps > 1:                                  # trap name per row, mode name between its two rows
                    text(lay, f"trap{r}", f"trap {k}", X_ECDF[0] - TRAP_LABEL_CM * ECDF_UNITS[0], 0.0,
                         STYLE["row_pt"], rotate=90, color=mcol)
                    if k == 1:
                        text(lay, f"row{r}", mlabel, X_ECDF[0] - ROW_LABEL_CM * ECDF_UNITS[0], gap_y, t,
                             rotate=90, color=mcol)
                else:
                    text(lay, f"row{r}", mlabel, X_ECDF[0] - ROW_LABEL_CM * ECDF_UNITS[0], 0.0, t,
                         rotate=90, color=mcol)
                if r == int(mid):                                # y title centred on the whole row block
                    text(lay, "yt", r"ECDF \x(2212) uniform", X_ECDF[0] - YTITLE_CM * ECDF_UNITS[0],
                         0.0 if len(rows) % 2 else gap_y, t, rotate=90)
        cov = L.get((r, ncol - 1))
        if cov is None:
            continue
        if r == 0:
            text(cov, "ct_cov", "coverage", sum(X_COV) / 2, Y_COV[1] + COLTITLE_CM * COV_UNITS[1], t)
        if r == int(mid):
            dy = 0.0 if len(rows) % 2 else Y_COV[0] - (GAP_CM[1] / 2) * COV_UNITS[1]
            text(cov, "yt_cov", "coverage error", X_COV[1] + YTITLE_CM * COV_UNITS[0], dy, t, rotate=90)
        if r == last:
            text(cov, "xt_cov", "nominal level", sum(X_COV) / 2, Y_COV[0] - XTITLE_CM * COV_UNITS[1], t)
    leg = L.get("legend")
    if leg is not None:
        for j, (_, lab) in enumerate(PARAMS):
            text(leg, f"lg{j}", lab, 0.32, ys[j], STYLE["legend_pt"])
        if has_b:
            for i, (_, lab, _) in enumerate(VARIANTS):
                text(leg, f"lgv{i}", lab, 0.45, ys[len(PARAMS) + i], STYLE["legend_pt"])
    lt(f"win -a {gp.name}; pfit2l margin:=0 borderwid:=2;")    # page = the layers + their labels
    return folder, name, gp


def main():
    path = pick_curves()
    res = step("build")(build, path)
    if res is None:
        return
    folder, name, gp = res
    base = os.path.join(folder, name)
    try:
        op.save(base + ".opju")
    except Exception as e:                              # noqa: BLE001
        print(f"op.save failed ({e}); trying LabTalk save")
        lt(f'save "{base}.opju";')
    step("export pdf")(gp.save_fig, base + ".pdf")
    step("export png")(gp.save_fig, base + ".png", width=2400)
    print(f"done: {base}.opju / .pdf / .png")
    if LOG:
        print("\nsteps that failed (the rest of the figure was built):")
        for m in LOG:
            print("  ", m)
    if LT_FAILED:
        print("\nLabTalk statements that failed (count, first example) - please send this list back:")
        for c, ex in LT_FAILED.values():
            print(f"   {c:4d} x  {ex}")


if not globals().get("_NO_MAIN"):
    main()
