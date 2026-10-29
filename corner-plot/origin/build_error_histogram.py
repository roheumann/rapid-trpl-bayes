"""
build_error_histogram.py — histogram of -log L after N CMA-ES evaluations in Origin (2021+, Windows)
====================================================================================================
Input : <run>/start_point_study/origin_convergence/error_histogram_<N>evals.csv, written on rekenbeest by
            python scripts/export_error_histogram_origin.py <run> [--evals 1000]
Output: next to it: error_histogram_<N>evals.opju (worksheet + graph), .pdf, .png

Run inside Origin:   run -pyf "C:\\...\\corner-plot\\origin\\build_error_histogram.py"
A file dialog asks for the CSV (CSV = None); no extra Python packages needed.

Bars side by side per bin: physics-informed start (red, its CMA-ES seeds) and random Sobol starts (orange); height
= percent of the group (HEIGHT = "percent") or the number of starts (HEIGHT = "count"); the count is written on every
non-empty bar.  The bins are not equal-width, so the x axis is the bin index and every bin gets its range as text.
The bars are not an Origin column plot (grouped column plots need per-plot colours, which LabTalk sets unreliably for
grouped plots) and not a fill under a line ('set %C -pfm 1' left them empty in the Origin run of 2026-10-21): each bar
is FILL_LINES_PER_PT vertical line segments per pt of bar width, FILL_LINE_PT wide, so they overlap into a solid
fill, plus a black outline.  The legend shows a filled square per group (one point outside the x range).

error_histogram_<N>evals_logdelta.csv (export_error_histogram_origin.py --log-delta, 24/10/2026): the bins are
log-equal in Delta(-log L) to the NN maximum, so equal spacing on the bin index IS a log axis.  The edge values are
then written at the bin boundaries (like the ticks of a log axis), the first bin is labelled "converged" (< its upper
edge), and the x title is Delta(-log L) to the NN maximum.
"""
import csv
import os
import re
import traceback

import numpy as np
import originpro as op

CSV = None                 # None -> file dialog
HEIGHT = "percent"         # "percent" (of each group; 16 vs 64 starts) or "count"
GROUPS = [("physics", "#D62728", "physics-informed start"), ("random", "#FF7F0E", "random (Sobol) starts")]
FILL_LINE_PT = 1.0         # width of the vertical fill lines (pt)
FILL_LINES_PER_PT = 2      # fill lines per pt of bar width (spacing 0.5 pt < line width: no gaps)
STYLE = {"font": "Arial", "tick_pt": 9, "title_pt": 11, "legend_pt": 9, "count_pt": 7, "bin_pt": 8,
         "bar_width": 0.38, "edge_pt": 0.6}
FRAME_CM = (2.2, 0.8, 14.0, 7.0)          # layer left, top, width, height (cm)
PAGE_START_CM = 20.0

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
    ok = True
    for s in [s.strip() for s in cmd.split(";") if s.strip()]:
        if op.lt_exec(s + ";") is False:
            ok = False
            key = re.sub(r"-?\d+(\.\d+)?(e[-+]?\d+)?", "#", s)[:120]
            LT_FAILED.setdefault(key, [0, s])[0] += 1
    return ok


def rgb(hexcol):
    h = hexcol.lstrip("#")
    return "color({}, {}, {})".format(*(int(h[i:i + 2], 16) for i in (0, 2, 4)))


def pick_csv():
    if CSV:
        return CSV
    lt('dlgfile group:="error_histogram_*.csv" title:="error_histogram_<N>evals.csv"; string __f$ = fname$;')
    return op.get_lt_str("__f$")


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return {k: [r[k] for r in rows] for k in rows[0]}


def bars(i, h, w):
    """Outline of bars centred at i with heights h (closed at y = 0 between bars)."""
    x, y = [], []
    for c, v in zip(i, h):
        x += [c - w / 2, c - w / 2, c + w / 2, c + w / 2]
        y += [0.0, v, v, 0.0]
    return x, y


def fill_lines(c, v, w, x_per_pt):
    """One bar as a serpentine (up, across, down, across, ...) of vertical segments, inset by half a line width so
    the fill stays inside the outline; one plot per bar, so no connection ever runs between bars."""
    inset = 0.5 * FILL_LINE_PT * x_per_pt
    k = max(2, int(np.ceil(w / x_per_pt * FILL_LINES_PER_PT)) + 1)
    x, y = [], []
    for j, xv in enumerate(np.linspace(c - w / 2 + inset, c + w / 2 - inset, k)):
        x += [xv, xv]
        y += [0.0, v] if j % 2 == 0 else [v, 0.0]
    return x, y


def label_of(lo, hi):
    """'-2.24 to -2.23' with Origin's minus (a leading '-' after 'label -a' would be read as a switch)."""
    f = lambda v: f"{float(v):g}".replace("-", "\\x(2212)")
    return f"{f(lo)} to {f(hi)}"


def edge_of(v):
    """Edge value of a log-delta bin, 2 significant digits (0.02, 0.063, 0.2, 6.3, 20, 630)."""
    v = float(v)
    return f"{float(f'{v:.2g}'):g}" if v < 100 else f"{v:.0f}"


def text(name, s, x, y, pt):
    """Text label in axis coordinates (created at 0 0 and moved: a negative number after 'label -a' is a switch)."""
    lt(f"label -a 0 0 -n {name} {s}; {name}.fsize = {pt}; {name}.attach = 2; {name}.x = {x:.6g}; {name}.y = {y:.6g};")


def build(path):
    H = read_csv(path)
    n_ev = re.search(r"(\d+)evals", os.path.basename(path))
    n_ev = n_ev.group(1) if n_ev else "N"
    nb = len(H["bin"])
    i = np.arange(nb, dtype=float)
    logd = "_logdelta" in os.path.basename(path)
    labels = [label_of(a, b) for a, b in zip(H["lo"], H["hi"])]
    hv = {g: np.array(H[f"{HEIGHT}_{g}"], float) for g, _, _ in GROUPS}
    cnt = {g: np.array(H[f"count_{g}"], float).astype(int) for g, _, _ in GROUPS}
    n_g = {g: int(cnt[g].sum()) for g in cnt}

    book = op.new_book("w", lname=f"error_histogram_{n_ev}evals")
    wks = book[0]
    wks.from_list(0, [s.replace("\u2212", "-") for s in H["label"]],
                  lname="bin (Delta -log L to NN max)" if logd else "bin (-log L, NN)")
    for k, key in enumerate(("lo", "hi", "count_physics", "count_random", "percent_physics", "percent_random")):
        wks.from_list(k + 1, [float(v) for v in H[key]], lname=key)
    w = STYLE["bar_width"]
    shapes = book.add_sheet("bar_outlines")
    l, t, wd, ht = FRAME_CM
    x_from, x_to = -0.6, nb - 0.4
    x_per_pt = (x_to - x_from) / (wd / 0.03528)          # x units per pt of page
    for gi, (g, _, _) in enumerate(GROUPS):              # outlines: columns 2 gi, 2 gi + 1
        x, y = bars(i + (gi - 0.5) * w, hv[g], w)
        shapes.from_list(2 * gi, x, lname=f"x {g}")
        shapes.from_list(2 * gi + 1, y, lname=f"{HEIGHT} {g}")
    fills = book.add_sheet("bar_fills")                  # one column pair per non-empty bar
    fill_cols = []
    for gi, (g, color, _) in enumerate(GROUPS):
        for b in range(nb):
            if hv[g][b] > 0:
                x, y = fill_lines(i[b] + (gi - 0.5) * w, hv[g][b], w, x_per_pt)
                k = 2 * len(fill_cols)
                fills.from_list(k, [float(v) for v in x], lname=f"x {g} bin {b}")
                fills.from_list(k + 1, [float(v) for v in y], lname=f"{g} bin {b}")
                fill_cols.append((k, color))
    keys = book.add_sheet("legend_keys")                 # one point per group outside the x range
    for gi in range(len(GROUPS)):
        keys.from_list(2 * gi, [x_from - 10.0], lname="x")
        keys.from_list(2 * gi + 1, [1.0], lname=GROUPS[gi][0])

    gp = op.new_graph(lname=f"error_histogram_{n_ev}evals")
    gl = gp[0]
    lt(f"win -a {gp.name}; page.height = {PAGE_START_CM / 2.54:.4f} * page.resy;")
    refs, n_pl = [], 0
    for k, color in fill_cols:                           # fills
        p = gl.add_plot(fills, coly=k + 1, colx=k, type="l")
        p.color = color
        p.set_cmd(f"-w {int(FILL_LINE_PT * 500)}")
        n_pl += 1
    for gi in range(len(GROUPS)):                        # black outlines on top
        p = gl.add_plot(shapes, coly=2 * gi + 1, colx=2 * gi, type="l")
        p.color = "#000000"
        p.set_cmd(f"-w {int(STYLE['edge_pt'] * 500)}")
        n_pl += 1
    for gi, (g, color, _) in enumerate(GROUPS):          # legend keys: filled squares (shape 1), never visible
        p = gl.add_plot(keys, coly=2 * gi + 1, colx=2 * gi, type="s")
        p.color = color
        p.set_cmd("-k 1 -kf 0 -z 9")
        n_pl += 1
        refs.append(n_pl)
    op.lt_exec("label -r legend;")

    ymax = max(float(v.max()) for v in hv.values())
    ytop = (np.ceil(ymax * 1.15 / 10) * 10) if HEIGHT == "percent" else np.ceil(ymax * 1.15)
    lt(f"page.active = 1; layer.fixed = 1; layer.factor = 1; layer.unit = 3; layer.left = {l}; layer.top = {t};"
       f" layer.width = {wd}; layer.height = {ht};")
    lt(f"layer.x.type = 1; layer.x.from = {x_from}; layer.x.to = {x_to}; layer.y.type = 1; layer.y.from = 0;"
       f" layer.y.to = {ytop:g}; layer.x.showAxes = 3; layer.y.showAxes = 3; layer.x.ticks = 0; layer.x2.ticks = 0;"
       " layer.y.ticks = 5; layer.y2.ticks = 5;"
       f" layer.y.label.pt = {STYLE['tick_pt']}; layer.y.label.font = font({STYLE['font']});"
       " layer.y2.inc = layer.y.inc; layer.y2.minorTicks = layer.y.minorTicks;")
    lt("layer.x.showLabels = 0; layer.y.showLabels = 1; layer.x.showlabel = 0; layer.x2.showlabel = 0;"
       " layer.y.showlabel = 1; layer.y2.showlabel = 0;")

    per_cm = ytop / ht                                   # y units per cm
    if logd:                                             # edge values at the boundaries, "converged" under bin 0
        text("bin0", "converged", i[0], -0.35 * per_cm, STYLE["bin_pt"])
        for b in range(1, nb):
            text(f"edge{b}", edge_of(H["lo"][b]), i[b] - 0.5, -0.35 * per_cm, STYLE["bin_pt"])
        text(f"edge{nb}", edge_of(H["hi"][nb - 1]), i[nb - 1] + 0.5, -0.35 * per_cm, STYLE["bin_pt"])
        for b in range(1, nb + 1):                       # short tick marks at the edges (bin boundaries)
            lt(f"draw -n etk{b} -l {{{i[0] + b - 0.5:.6g}, 0, {i[0] + b - 0.5:.6g}, {-0.12 * per_cm:.6g}}};")
    else:
        for b in range(nb):                              # bin ranges below the axis
            text(f"bin{b}", labels[b], i[b], -0.35 * per_cm, STYLE["bin_pt"])
    for gi, (g, _, _) in enumerate(GROUPS):              # counts on the bars
        for b in range(nb):
            if cnt[g][b]:
                text(f"n{g[0]}{b}", str(cnt[g][b]), i[b] + (gi - 0.5) * w, hv[g][b] + 0.22 * per_cm, STYLE["count_pt"])
    xt = f"\\x(2212)log \\i(L) (NN) after {n_ev} CMA-ES evaluations"
    if logd:
        xt = f"\\x(0394)(\\x(2212)log \\i(L)) to the NN maximum after {n_ev} CMA-ES evaluations (log bins)"
    yt = "starts of the group (%)" if HEIGHT == "percent" else "number of starts"
    op.lt_exec("xb.show = 0;")                          # x title as text below the bin labels
    text("xttl", xt, (nb - 1) / 2, -1.0 * per_cm, STYLE["title_pt"])
    lt(f'label -yl {yt}; yl.text$ = "{yt}"; yl.show = 1; yl.fsize = {STYLE["title_pt"]};')
    entries = [f"\\l({r}) {name} ({n_g[g]})" for r, (g, _, name) in zip(refs, GROUPS)]
    legend = "\\n".join(entries)
    lt(f"label -s -sa -n histleg {legend}; histleg.fsize = {STYLE['legend_pt']}; histleg.attach = 2;"
       f" histleg.x = {nb - 0.6} - histleg.dx / 2; histleg.y = {ytop:g} - histleg.dy / 2 - 0.3 * {per_cm:.6g};")
    lt(f"win -a {gp.name}; pfit2l margin:=0 borderwid:=2;")
    return gp


def main():
    path = pick_csv()
    gp = step("build")(build, path)
    if gp is None:
        return
    base = os.path.splitext(path)[0]
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
