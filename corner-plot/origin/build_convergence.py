"""
build_convergence.py — start-point convergence study with ABSOLUTE -log L in Origin (2021+, Windows)
====================================================================================================
Input : <run>/start_point_study/origin_convergence/ written on rekenbeest by
            python scripts/export_convergence_origin.py <run>
        (cmaes_histories.csv, curve_fit_histories.csv, starts_endpoints.csv, final_sorted.csv, axes.json)
Output: next to those files: convergence_abs.opju (workbook + graph), convergence_abs.pdf / .png

Run inside Origin:   run -pyf "C:\\...\\corner-plot\\origin\\build_convergence.py"
A file dialog asks for axes.json (FOLDER = None); no extra Python packages needed (csv + numpy).

The figure is origin_convergence/preview.png (matplotlib) rebuilt in Origin:
  column 1  CMA-ES on the NN likelihood: best -log L so far vs NN evaluations (log x), end = black-edged circle
  column 2  curve_fit with the ODE: -log L vs ODE solves, start = circle (CMA-ES result), end = star
  rows      split y axis, because -log L < 0 near the optimum: upper row log y [Y_SPLIT, Y_LOG_HI],
            lower row linear y [Y_LIN_LO, Y_SPLIT] (ranges from axes.json)
  column 3  final ODE -log L of every start that reached the best-fit basin, sorted (zoomed y): shows which starts
            end better than the physics-informed start (seed 1 = black ring, red dotted line)
  dotted    black: best fit of the run (ODE -log L); grey (CMA-ES panel only): NN maximum
Red = the 16 CMA-ES seeds of the physics-informed start, viridis = Sobol starts (colours from the CSV).
"""
import csv
import json
import os
import re
import traceback

import numpy as np
import originpro as op

FOLDER = None              # None -> file dialog (pick axes.json)
STYLE = {"font": "Arial", "tick_pt": 9, "title_pt": 11, "legend_pt": 8,
         "lw_sobol": 0.8, "lw_physics": 2.0, "lw_ref": 1.5,
         "sym_sobol": 4, "sym_physics": 6, "star_extra": 3}
SHAPE = {"circle": 2, "star": 14}          # Origin symbol shapes (14 = 5-point star; change if your version differs)
DASH = {"solid": 0, "dash": 1, "dot": 2}
RED = "#D62728"
NN_GREY = "#737373"
# layer frames in cm on the page: (left, top, width, height); the page is fitted to them at the end (pfit2l)
FRAMES = {"cma_log": (2.5, 1.0, 11.0, 3.5), "cma_lin": (2.5, 4.5, 11.0, 6.5),
          "cf_log": (13.9, 1.0, 5.5, 3.5), "cf_lin": (13.9, 4.5, 5.5, 6.5),
          "zoom": (20.6, 4.5, 4.8, 6.5)}
LEGEND_TOP_CM = 0.9        # legend: top edge on the page, centred over the zoom panel
PAGE_START_CM = 30.0       # page height before pfit2l (big enough for every frame; Origin keeps the aspect ratio)

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
    lt('dlgfile group:="axes.json" title:="origin_convergence/axes.json of the start-point study"; string __f$ = fname$;')
    return os.path.dirname(op.get_lt_str("__f$"))


def read_csv(path):
    """-> dict column -> list (float where possible, NaN for empty cells)."""
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    head, body = rows[0], rows[1:]
    out = {}
    for j, h in enumerate(head):
        vals = [r[j] if j < len(r) else "" for r in body]
        try:
            out[h] = [float(v) if v != "" else float("nan") for v in vals]
        except ValueError:
            out[h] = vals
    return out


def finite(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    k = np.isfinite(x) & np.isfinite(y)
    return x[k], y[k]


# ─────────────────────────────────────────────────────────────────────────────
# Origin helpers
# ─────────────────────────────────────────────────────────────────────────────
def sheet(book, name, cols):
    """cols: list of (long name, array) -> new worksheet in book."""
    wks = book.add_sheet(name[:31])
    for i, (lname, arr) in enumerate(cols):
        wks.from_list(i, [float(v) for v in np.asarray(arr, float)], lname=lname)
    return wks


class Layer:
    """A graph layer that counts its plots (for the \\l(layer.plot) legend entries)."""
    def __init__(self, gp, idx):
        self.gp, self.idx, self.gl, self.n = gp, idx, gp[idx], 0

    def activate(self):
        lt(f"win -a {self.gp.name}; page.active = {self.idx + 1};")

    def line(self, wks, cx, cy, color, width, dash=0):
        p = self.gl.add_plot(wks, coly=cy, colx=cx, type="l")
        p.color = color
        p.set_cmd(f"-w {int(width * 500)}")             # 1 pt = 500
        if dash:
            p.set_cmd(f"-d {dash}")
        self.n += 1
        return f"{self.idx + 1}.{self.n}"

    def marker(self, wks, cx, cy, color, size, shape=SHAPE["circle"], edge=True):
        """Filled symbol in `color`; a black open symbol of the same size on top gives the black edge."""
        p = self.gl.add_plot(wks, coly=cy, colx=cx, type="s")
        p.color = color
        p.set_cmd(f"-k {shape} -kf 0 -z {size}")
        self.n += 1
        ref = f"{self.idx + 1}.{self.n}"
        if edge:
            q = self.gl.add_plot(wks, coly=cy, colx=cx, type="s")
            q.color = "#000000"
            q.set_cmd(f"-k {shape} -kf 1 -z {size}")
            self.n += 1
        return ref


def new_layer(gp):
    lt(f"win -a {gp.name}; layer -n;")
    return int(op.lt_float("page.nlayers")) - 1


def place(L, frame):
    l, t, w, h = frame
    L.activate()
    lt(f"layer.fixed = 1; layer.factor = 1; layer.unit = 3; layer.left = {l}; layer.top = {t};"
       f" layer.width = {w}; layer.height = {h};")


def label_sides(xb, xt, yl, yr):
    """Tick labels per side (0/1), as build_corner.label_sides (old showLabels + the per-axis switches)."""
    return (f"layer.x.showLabels = {xb + 2 * xt}; layer.y.showLabels = {yl + 2 * yr}; layer.x.showlabel = {xb};"
            f" layer.x2.showlabel = {xt}; layer.y.showlabel = {yl}; layer.y2.showlabel = {yr};")


def axes(L, x, y, xlog, ylog, xlabels, ylabels, xtitle="", ytitle="", yright=False, xinc=None, yinc=None):
    """Scales, ticks inward on all four sides, labels only where asked; log axes as 10^n (build_corner style).
    The label switches are set last: setting from / to / inc afterwards brought the opposite labels back."""
    L.activate()
    s = (f"layer.x.type = {2 if xlog else 1}; layer.x.from = {x[0]:.8g}; layer.x.to = {x[1]:.8g};"
         f" layer.y.type = {2 if ylog else 1}; layer.y.from = {y[0]:.8g}; layer.y.to = {y[1]:.8g};"
         " layer.x.showAxes = 3; layer.y.showAxes = 3;"
         f" layer.x.label.pt = {STYLE['tick_pt']}; layer.y.label.pt = {STYLE['tick_pt']};"
         f" layer.x.label.font = font({STYLE['font']}); layer.y.label.font = font({STYLE['font']});")
    for a, log, inc in (("x", xlog, xinc), ("y", ylog, yinc)):
        if log:
            s += f" layer.{a}.label.numFormat = 2; layer.{a}.inc = 1;"
        elif inc:
            s += f" layer.{a}.inc = {inc};"
    lt(s)
    lt(" ".join(f"layer.{a}.thickness = 0.8; layer.{a}.tickthickness = 0.8; layer.{a}.ticklength = 4;"
                f" layer.{a}.mticklength = 2; layer.{a}.ticks = 5; layer.{a}.mtickthickness = 0.8;"
                for a in ("x", "y", "x2", "y2")))
    lt("layer.x2.minorTicks = layer.x.minorTicks; layer.y2.minorTicks = layer.y.minorTicks;"
       " layer.x2.inc = layer.x.inc; layer.y2.inc = layer.y.inc;")
    lt(label_sides(int(xlabels), 0, int(ylabels and not yright), int(ylabels and yright)))
    t = STYLE["title_pt"]
    for o, txt in (("xb", xtitle), ("yr" if yright else "yl", ytitle)):
        if txt:
            lt(f'label -{o} {txt}; {o}.text$ = "{txt}"; {o}.show = 1; {o}.fsize = {t};')
        elif L.idx == 0:                                # only the first layer has default titles (new layers: none)
            op.lt_exec(f"{o}.show = 0;")
    op.lt_exec("label -r legend;")                     # no auto legend


# ─────────────────────────────────────────────────────────────────────────────
# figure
# ─────────────────────────────────────────────────────────────────────────────
def build(folder):
    A = json.load(open(os.path.join(folder, "axes.json"), encoding="utf-8"))
    cma = read_csv(os.path.join(folder, "cmaes_histories.csv"))
    cf = read_csv(os.path.join(folder, "curve_fit_histories.csv"))
    ep = read_csv(os.path.join(folder, "starts_endpoints.csv"))
    fs = read_csv(os.path.join(folder, "final_sorted.csv"))
    labels, colors = ep["label"], ep["color"]
    phys = [lb.startswith("physics") for lb in labels]
    order = [i for i in range(len(labels)) if not phys[i]] + [i for i in range(len(labels)) if phys[i]]   # red on top

    # ── workbook ───────────────────────────────────────────────────────────────
    book = op.new_book("w", lname="convergence_abs")
    W = {}
    for i in range(len(labels)):
        lb = labels[i]
        x, y = finite(cma[f"{lb}_evals"], cma[f"{lb}_nll_nn"])
        xo, yo = finite(cf[f"{lb}_solves"], cf[f"{lb}_nll_ode"])
        W[lb] = step(f"sheet {lb}")(sheet, book, lb, [
            ("NN evaluations", x), ("-log L (NN), CMA-ES", y),
            ("CMA-ES end: evals", x[-1:]), ("CMA-ES end: -log L (NN)", y[-1:]),
            ("ODE solves", xo), ("-log L (ODE), curve_fit", yo),
            ("curve_fit start: solves", xo[:1]), ("curve_fit start: -log L (ODE)", yo[:1]),
            ("curve_fit end: solves", xo[-1:]), ("curve_fit end: -log L (ODE)", yo[-1:])])
    xc, xf = A["x_cmaes"], A["x_curve_fit"]
    ref = sheet(book, "reference_lines", [
        ("x CMA-ES", xc), ("best fit (ODE)", [A["best_fit_nll_ode"]] * 2), ("NN maximum", [A["nn_max_nll_nn"]] * 2),
        ("x curve_fit", xf), ("best fit (ODE) ", [A["best_fit_nll_ode"]] * 2),
        ("x zoom", [0, A["zoom_n"] + 1]), ("best fit (ODE)  ", [A["best_fit_nll_ode"]] * 2),
        ("physics start, seed 1", [A["physics_seed1_nll_ode"]] * 2)])
    n = A["zoom_n"]
    zs = sheet(book, "final_sorted_zoom", [("rank", fs["rank"][:n]), ("-log L (ODE) final", np.r_[
        [v if np.isfinite(v) else w for v, w in zip(fs["nll_ode_physics"][:n], fs["nll_ode_sobol"][:n])]])])
    zcols = []                                           # one 1-point series per start (own colour)
    for r in range(n):
        zcols += [(f"rank {fs['label'][r]}", [fs["rank"][r]]),
                  (fs["label"][r], [fs["nll_ode_sobol"][r] if np.isfinite(fs["nll_ode_sobol"][r])
                                    else fs["nll_ode_physics"][r]])]
    zp = sheet(book, "zoom_points", zcols)
    seed1 = [i for i in range(n) if fs["label"][i] == "physics"]
    ep_cols = [(k, ep[k]) for k in ep if k not in ("label", "color", "is_physics_seed", "better_than_physics_seed1")]
    sheet(book, "starts_endpoints", ep_cols)

    # ── graph: 4 data layers + zoom + legend ──────────────────────────────────
    gp = op.new_graph(lname="convergence_abs")
    lt(f"win -a {gp.name}; page.height = {PAGE_START_CM / 2.54:.4f} * page.resy;")
    idx = {"cma_log": 0}
    for k in ("cma_lin", "cf_log", "cf_lin", "zoom"):
        idx[k] = new_layer(gp)
    L = {k: Layer(gp, i) for k, i in idx.items()}
    leg = {}
    for i in order:
        lb, col, hl = labels[i], colors[i], phys[i]
        w = W[lb]
        if w is None:
            continue
        lw = STYLE["lw_physics"] if hl else STYLE["lw_sobol"]
        sz = STYLE["sym_physics"] if hl else STYLE["sym_sobol"]
        for row in ("log", "lin"):
            a, b = L[f"cma_{row}"], L[f"cf_{row}"]
            r1 = a.line(w, 0, 1, col, lw)
            r2 = a.marker(w, 2, 3, col, sz)
            r3 = b.line(w, 4, 5, col, lw, DASH["dash"])
            b.marker(w, 6, 7, col, sz)
            r4 = b.marker(w, 8, 9, col, sz + STYLE["star_extra"], SHAPE["star"])
            if row == "lin":
                if hl:
                    leg.setdefault("physics", r1)
                else:
                    leg.setdefault("cma", r1); leg.setdefault("switch", r2)
                    leg.setdefault("cf", r3); leg.setdefault("end", r4)
    for row in ("log", "lin"):
        r = L[f"cma_{row}"].line(ref, 0, 1, "#000000", STYLE["lw_ref"], DASH["dot"])
        g = L[f"cma_{row}"].line(ref, 0, 2, NN_GREY, STYLE["lw_ref"], DASH["dot"])
        L[f"cf_{row}"].line(ref, 3, 4, "#000000", STYLE["lw_ref"], DASH["dot"])
        if row == "lin":
            leg["best"], leg["nnmax"] = r, g
    Z = L["zoom"]
    Z.line(ref, 5, 6, "#000000", STYLE["lw_ref"], DASH["dot"])
    leg["seed1line"] = Z.line(ref, 5, 7, RED, 1.0, DASH["dot"])
    for r in range(n):
        lb = fs["label"][r]
        col = RED if lb.startswith("physics") else fs["color"][r]
        Z.marker(zp, 2 * r, 2 * r + 1, col, STYLE["sym_physics"] if lb.startswith("physics") else STYLE["sym_sobol"] + 1)
    for r in seed1:                                     # black ring around the first physics seed
        p = Z.gl.add_plot(zp, coly=2 * r + 1, colx=2 * r, type="s")
        p.color = "#000000"
        p.set_cmd(f"-k {SHAPE['circle']} -kf 1 -z {STYLE['sym_physics'] + 5}")
        Z.n += 1
        leg["seed1"] = f"{Z.idx + 1}.{Z.n}"

    # ── axes ─────────────────────────────────────────────────────────────────
    ylin, ylog, yz = A["y_lin"], A["y_log"], A["y_zoom"]
    ylin = (ylin[0], ylin[1] - 0.01)                     # no linear "1.0" label under the log "10^0" at the split
    for k, lay in L.items():
        place(lay, FRAMES[k])
    axes(L["cma_log"], xc, ylog, True, True, False, True)
    axes(L["cma_lin"], xc, ylin, True, False, True, True, "NN evaluations (CMA-ES)", "\\x(2212)log \\i(L)", yinc=0.5)
    axes(L["cf_log"], xf, ylog, False, True, False, False, xinc=100)
    axes(L["cf_lin"], xf, ylin, False, False, True, False, "ODE solves (curve_fit)", xinc=100, yinc=0.5)
    axes(L["zoom"], (0, n + 1), yz, False, False, True, True, "rank of final \\x(2212)log \\i(L)",
         "final \\x(2212)log \\i(L) (ODE)", yright=True, xinc=10)
    lt("layer.y.label.numFormat = 1; layer.y.label.decPlaces = 4;")        # zoom: 4 decimals, no offset

    # ── legend (\l(layer.plot) = the line / symbol of that plot) ──────────────
    bf, nm = A["best_fit_nll_ode"], A["nn_max_nll_nn"]
    entries = [("cma", "CMA-ES (NN likelihood)"), ("switch", "switch to curve_fit"),
               ("cf", "curve_fit (ODE likelihood)"), ("end", "end"),
               ("physics", f"physics-informed start ({A['n_physics_seeds']} CMA-ES seeds)"),
               ("seed1", "physics start, seed 1"),
               ("best", f"best fit (ODE), {bf:.4f}"), ("nnmax", f"NN maximum, {nm:.4f}")]
    text = "\\n".join(f"\\l({leg[k]}) {t}" for k, t in entries if k in leg)
    L["zoom"].activate()                                # legend text in the zoom layer, placed above its frame
    lt(f'label -s -sa -n convleg {text}; convleg.fsize = {STYLE["legend_pt"]};')
    l, t, w, h = FRAMES["zoom"]
    per_cm = (yz[1] - yz[0]) / h                         # zoom y units per cm
    lt(f"convleg.x = {(n + 1) / 2:.6g}; convleg.y = {yz[1] + (t - LEGEND_TOP_CM) * per_cm:.8g} - convleg.dy / 2;")
    lt(f"win -a {gp.name}; pfit2l margin:=0 borderwid:=2;")    # page = the layers + their labels
    return gp


def main():
    folder = pick_folder()
    gp = step("build")(build, folder)
    if gp is None:
        return
    base = os.path.join(folder, "convergence_abs")
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


main()
