"""
build_dd_vs_0d.py — drift-diffusion vs 0D tau_diff(QFLS) at the chosen alpha in Origin (2021+, Windows)
=======================================================================================================
Input : <sigma_model folder>/origin_dd_vs_0d/ written on rekenbeest by
            python scripts/export_dd_vs_0d_origin.py <sigma_model folder>
        (data_points.csv, raw_points.csv, curve_0d.csv, curve_dd.csv, axes.json)
Output: next to those files: dd_vs_0d.opju (workbook + graph), dd_vs_0d.pdf / .png

Run inside Origin:   run -pyf "C:\\...\\corner-plot\\origin\\build_dd_vs_0d.py"
A file dialog asks for axes.json (FOLDER = None); no extra Python packages needed (csv + numpy).

Top panel   : tau_diff on a log axis (10^n labels): RAW measured points (open grey circles), 0D rate equations
              (cyan), drift-diffusion at alpha (red dashed); legend top right.
Bottom panel: residuals DD - 0D (log10 tau_diff) at the spline points (red), zero line (black dotted),
              RMSE = sigma_model.  x title: Fermi-level splitting Delta E_F (eV).
Texts in the log panel are placed at real tau values (Origin object coordinates are axis values, not decades).
Recipe of build_sbc.py / build_corner.py: every panel is layer 1 of its own page, the pages are merged with
merge_graph, the layers are put back by name and formatted again, and all texts are added after the merge.  The
circles are line outlines (symbol plots stayed invisible in the SBC run), '=' in a label is written as \\(61).
"""
import csv
import json
import os
import re
import traceback

import numpy as np
import originpro as op

FOLDER = None              # None -> file dialog (pick axes.json)
STYLE = {"font": "Arial", "tick_pt": 12, "title_pt": 12, "legend_pt": 12, "text_pt": 12,
         "lw_data": 0.6, "lw_0d": 1.5, "lw_dd": 1.5, "lw_delta": 1.2, "lw_zero": 0.6}
COLORS = {"data": "#666666", "0d": "#00CED1", "dd": "#D62728", "zero": "#000000"}
DASH = {"solid": 0, "dash": 1, "dot": 2}
MARKER_CM = 0.07           # circle radius
LEFT_CM, TOP_CM, WIDTH_CM = 2.7, 0.6, 11.0
TOP_H_CM, BOTTOM_H_CM, GAP_CM = 8.0, 3.5, 0.25
MERGED_PAGE_CM = 30.0      # page height before pfit2l

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
def pick_folder():
    if FOLDER:
        return FOLDER
    lt('dlgfile group:="axes.json" title:="origin_dd_vs_0d/axes.json"; string __f$ = fname$;')
    return os.path.dirname(op.get_lt_str("__f$"))


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return {k: np.array([float(r[k]) if r[k] not in ("", "nan") else np.nan for r in rows]) for k in rows[0]}


# ─────────────────────────────────────────────────────────────────────────────
# Origin helpers (as build_sbc.py)
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


def axes_cmd(x, xinc, y, yinc, xlabels, ylog=False):
    """LabTalk of one panel's axes: x linear, y linear or log10 (y given in decades, 10^n labels), ticks inward on
    all four sides, the first major tick on a multiple of the step (x starts at 0.93), tick labels bottom (if
    xlabels) and left."""
    if ylog:
        ys = (f" layer.y.type = 2; layer.y.from = {10 ** y[0]:.6g}; layer.y.to = {10 ** y[1]:.6g}; layer.y.inc = 1;"
              " layer.y.label.numFormat = 2;")
    else:
        ys = (f" layer.y.type = 1; layer.y.from = {y[0]:.6g}; layer.y.to = {y[1]:.6g}; layer.y.inc = {yinc:.6g};"
              f" layer.y.firstTick = {np.ceil(y[0] / yinc - 1e-9) * yinc:.6g}; layer.y.minorTicks = 1;")
    s = (f"layer.x.type = 1; layer.x.from = {x[0]:.6g}; layer.x.to = {x[1]:.6g}; layer.x.inc = {xinc:.6g};"
         f" layer.x.firstTick = {np.ceil(x[0] / xinc - 1e-9) * xinc:.6g};" + ys +
         " layer.x.showAxes = 3; layer.y.showAxes = 3; layer.x.minorTicks = 1;"
         f" layer.x.label.pt = {STYLE['tick_pt']}; layer.y.label.pt = {STYLE['tick_pt']};"
         f" layer.x.label.font = font({STYLE['font']}); layer.y.label.font = font({STYLE['font']}); ")
    s += " ".join(f"layer.{a}.thickness = 0.8; layer.{a}.tickthickness = 0.8; layer.{a}.ticklength = 4;"
                  f" layer.{a}.mticklength = 2; layer.{a}.ticks = 5; layer.{a}.mtickthickness = 0.8;"
                  for a in ("x", "y", "x2", "y2"))
    s += (" layer.x2.minorTicks = layer.x.minorTicks; layer.y2.minorTicks = layer.y.minorTicks;"
          " layer.x2.inc = layer.x.inc; layer.y2.inc = layer.y.inc;"
          " layer.x2.firstTick = layer.x.firstTick; ")
    if not ylog:
        s += "layer.y2.firstTick = layer.y.firstTick; "
    return s + label_sides(int(xlabels), 0, 1, 0)


def text(L, name, txt, x, y, pt, rotate=0, left=False):
    """Text at (x, y) in the axis units of layer L (may lie outside the frame): centred, or left-aligned at x."""
    L.activate()
    lt(f"label -s -sa -n {name} {txt}; {name}.fsize = {pt};")
    if rotate:
        lt(f"{name}.rotate = {rotate};")
    lt(f"{name}.x = {x:.6g}" + (f" + {name}.dx / 2" if left else "") + f"; {name}.y = {y:.6g};")


def circles(xs, ys, ux, uy, r_cm=MARKER_CM):
    """Open circles as one NaN-separated polyline; ux, uy: axis units per cm of the panel (for a log axis: ys in
    decades and uy in decades per cm, convert the y outline with 10 ** afterwards)."""
    t = np.linspace(0, 2 * np.pi, 17)
    ox, oy = [], []
    for x, y in zip(xs, ys):
        if np.isfinite(x) and np.isfinite(y):
            ox += list(x + r_cm * ux * np.cos(t)) + [np.nan]
            oy += list(y + r_cm * uy * np.sin(t)) + [np.nan]
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


def sci(v):
    """5.07e+04 -> 5.07×10^4 in Origin text."""
    e = int(np.floor(np.log10(v)))
    return f"{v / 10 ** e:.3g}\\x(00D7)10\\+({e})"


# ─────────────────────────────────────────────────────────────────────────────
# figure
# ─────────────────────────────────────────────────────────────────────────────
def build(folder):
    A = json.load(open(os.path.join(folder, "axes.json"), encoding="utf-8"))
    P = read_csv(os.path.join(folder, "data_points.csv"))
    R = read_csv(os.path.join(folder, "raw_points.csv"))
    C0 = read_csv(os.path.join(folder, "curve_0d.csv"))
    CD = read_csv(os.path.join(folder, "curve_dd.csv"))
    X, Y, DY = A["x"], A["y_log10"], A["delta_y"]          # Y in decades
    geo_top = (LEFT_CM, TOP_CM, WIDTH_CM, TOP_H_CM)
    geo_bot = (LEFT_CM, TOP_CM + TOP_H_CM + GAP_CM, WIDTH_CM, BOTTOM_H_CM)
    ux = (X[1] - X[0]) / WIDTH_CM                       # axis units per cm
    uy_top, uy_bot = (Y[1] - Y[0]) / TOP_H_CM, (DY[1] - DY[0]) / BOTTOM_H_CM        # decades / cm in the top panel
    fmt_top = axes_cmd(X, A["x_inc"], Y, 1, False, ylog=True)
    fmt_bot = axes_cmd(X, A["x_inc"], DY, A["delta_inc"], True)

    # legend geometry (top right of the top panel, in its axis units)
    lg_x0 = X[1] - 7.7 * ux
    lg_y = [Y[1] - (0.6 + 0.6 * i) * uy_top for i in range(3)]      # decades
    lg_line = [lg_x0, lg_x0 + 0.8 * ux]

    book = op.new_book("w", lname="dd_vs_0d")
    w_pts = book[0]
    for i, k in enumerate(("qfls_eV", "log10_tau_data", "log10_tau_0d", "log10_tau_dd", "delta_dd_minus_0d")):
        w_pts.from_list(i, [float(v) for v in P[k]], lname=k)
    w_raw = sheet(book, "raw_points", [("QFLS (eV)", R["qfls_eV"]), ("tau_diff (s)", R["tau_diff_s"])])
    w_cur = sheet(book, "curves", [("QFLS 0D (eV)", C0["qfls_eV"]), ("tau_diff 0D (s)", C0["tau_diff_0d_s"]),
                                   ("QFLS DD (eV)", CD["qfls_eV"]), ("tau_diff DD (s)", CD["tau_diff_dd_s"])])
    cx, cy = circles(R["qfls_eV"], np.log10(R["tau_diff_s"]), ux, uy_top)
    kx, ky = circles([np.mean(lg_line)], [lg_y[0]], ux, uy_top)
    w_aux = sheet(book, "plot_helpers", [("data circles x", cx), ("data circles y", 10 ** cy),
                                         ("zero x", X), ("zero y", [0.0, 0.0]),
                                         ("legend circle x", kx), ("legend circle y", 10 ** ky),
                                         ("legend line x", lg_line), ("legend 0D y", [10 ** lg_y[1]] * 2),
                                         ("legend DD y", [10 ** lg_y[2]] * 2)])

    pages, geos, fmts = [], [], []

    def new_panel(key, geo, fmt):
        gp = op.new_graph(lname=f"dd_vs_0d_{key}")
        lt(f"win -a {gp.name}; page.active = 1; layer.fixed = 1; layer.factor = 1; layer.unit = 3; layer.left = 0.5;"
           f" layer.top = 0.5; layer.width = {geo[2]:.3f}; layer.height = {geo[3]:.3f};")
        op.lt_exec("xb.show = 0;")                      # the template's axis titles (texts are added after the merge)
        op.lt_exec("yl.show = 0;")
        pages.append(gp); geos.append(geo); fmts.append(fmt)
        return Layer(gp, 0)

    top = new_panel("top", geo_top, fmt_top)
    top.line(w_aux, 0, 1, COLORS["data"], STYLE["lw_data"])
    top.line(w_cur, 0, 1, COLORS["0d"], STYLE["lw_0d"])
    top.line(w_cur, 2, 3, COLORS["dd"], STYLE["lw_dd"], DASH["dash"])
    top.line(w_aux, 4, 5, COLORS["data"], STYLE["lw_data"])                 # legend keys
    top.line(w_aux, 6, 7, COLORS["0d"], STYLE["lw_0d"])
    top.line(w_aux, 6, 8, COLORS["dd"], STYLE["lw_dd"], DASH["dash"])
    op.lt_exec("label -r legend;")
    lt(f"win -a {top.gp.name}; page.active = 1; " + fmt_top)

    bot = new_panel("bottom", geo_bot, fmt_bot)
    bot.line(w_aux, 2, 3, COLORS["zero"], STYLE["lw_zero"], DASH["dot"])
    bot.line(w_pts, 0, 4, COLORS["dd"], STYLE["lw_delta"])
    op.lt_exec("label -r legend;")
    lt(f"win -a {bot.gp.name}; page.active = 1; " + fmt_bot)

    # ── merge, formatting again, texts ───────────────────────────────────────
    gp, where = merge(pages, geos)
    for j, i in where.items():
        lt(f"win -a {gp.name}; page.active = {i + 1}; {fmts[j]}")
    T, B = Layer(gp, where.get(0, 0)), Layer(gp, where.get(1, 1))
    t, lp = STYLE["title_pt"], STYLE["legend_pt"]
    xm = (X[0] + X[1]) / 2
    text(T, "ytop", "\\g(t)\\-(diff) (s)", X[0] - 1.85 * ux, 10 ** ((Y[0] + Y[1]) / 2), t, rotate=90)
    text(B, "ybot", "Residuals (dec)", X[0] - 1.85 * ux, (DY[0] + DY[1]) / 2, t, rotate=90)
    text(B, "xbot", "Fermi-level splitting \\g(D)\\i(E)\\-(F) (eV)", xm, DY[0] - 1.15 * uy_bot, t)
    tx = lg_line[1] + 0.25 * ux
    text(T, "lg0", "data", tx, 10 ** lg_y[0], lp, left=True)
    text(T, "lg1", "0D rate equations", tx, 10 ** lg_y[1], lp, left=True)
    text(T, "lg2", f"drift-diffusion, \\g(a) \\(61) {sci(A['alpha_cm'])} cm\\+(\\x(2212)1)", tx, 10 ** lg_y[2], lp,
         left=True)
    text(B, "rmse", f"RMSE \\(61) {A['sigma_model_dec']:.3f} dec", X[1] - 2.5 * ux, DY[0] + 0.5 * uy_bot,
         STYLE["text_pt"])
    lt(f"win -a {gp.name}; pfit2l margin:=0 borderwid:=2;")
    return gp


def main():
    folder = pick_folder()
    gp = step("build")(build, folder)
    if gp is None:
        return
    base = os.path.join(folder, "dd_vs_0d")
    try:
        op.save(base + ".opju")
    except Exception as e:                              # noqa: BLE001
        print(f"op.save failed ({e}); trying LabTalk save")
        lt(f'save "{base}.opju";')
    step("export pdf")(gp.save_fig, base + ".pdf")
    step("export png")(gp.save_fig, base + ".png", width=2400)
    print(f"done: {base}.opju / .pdf / .png")
    for m in LOG:
        print("failed step:", m)
    if LT_FAILED:
        print("\nLabTalk statements that failed (count, first example) - please send this list back:")
        for c, ex in LT_FAILED.values():
            print(f"   {c:4d} x  {ex}")


main()
