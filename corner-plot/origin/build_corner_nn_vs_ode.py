"""
build_corner_nn_vs_ode.py — NN SMC and ODE SMC posteriors of one run in one corner plot (Origin)
=================================================================================================
Input : <run>/ode_vs_nn/corner_nn_vs_ode/corner_nn_vs_ode.h5 (scripts/nn_vs_ode_corner.py on rekenbeest)
Output: corner_nn_vs_ode.opju / .pdf / .png next to the .h5 file

  * lower triangle: NN SMC (blue) as filled 1σ (dark, solid outline) / 2σ (light, fill only) HPD regions, ODE SMC
    (wine red) as 1σ solid / 2σ dashed contour lines of equal width on top; true value as dashed black lines, best
    fit as an open red star
  * font sizes and flat x tick labels of the 1-trap modes corner (STYLE_SET, X_LABEL_ROTATE); 2 traps: 3 decade labels
    per x axis (X_LABEL_COUNT) so the flat labels do not run into each other
  * diagonal: 1D marginals of both (density per decade, right axis, one title on the bottom-right panel)
  * legend in the free upper triangle
The look of the modes corner (build_corner_modes.py) without the predictive-check panels: square panels, 10^n labels,
log minor ticks, E_t - E_g axis on the n1 column, merge and fitted page.  build_corner.py and build_corner_modes.py
must lie in the folder of BUILD_CORNER.  Run inside Origin:  run -pyf "C:\\...\\origin\\build_corner_nn_vs_ode.py"
"""
import json
import os

import numpy as np

H5 = None                  # None -> file dialog
BUILD_CORNER = r"C:\Users\ro.heumann\Documents\PhDProject\NN_Paper\Data-Final\Second-Submission\origin\build_corner.py"
TRUTH_COLOR = "#000000"    # true value: dashed lines through every panel
TRUTH_WIDTH = 0.8
SHOW_TRUTH = True
BEST_COLOR = "#E41A1C"     # best fit: open star (drawn as lines: symbol plots stay invisible in merged pages)
BEST_RADIUS_CM = 0.13
BEST_WIDTH = 0.9
SHOW_BEST = True
LEGEND_SHIFT_CM = 0.9      # legend layer right of the density axis labels of the diagonal panel below-left of it
SIGMA2_DASH = {"nn": 2, "ode": 1}   # Origin -d of the 2σ contour per method: 2 dot, 1 dash; 1σ solid
NN_LINES = ("sigma1",)     # NN contour lines drawn on its fill: the 2σ region is the light fill only (() -> thin 1σ edge)
SIGMA2_SAME_WIDTH = True   # 2σ (dashed) lines as thick as the 1σ (solid) lines
STYLE_SET = {"tick_pt": 9, "title_pt": 11, "cbar_pt": 9}   # as the 1-trap modes corner (build_corner STYLE defaults)
X_LABEL_ROTATE = 0         # flat x tick labels, as the 1-trap modes corner
X_LABEL_COUNT = {2: 3}     # per number of traps: exactly this many decade labels per x axis (None: build_corner's choice)
OUT_NAME = "corner_nn_vs_ode"   # name of the .opju / .pdf / .png next to the .h5 file (build_corner_nuts_vs_smc.py)


def load(name, bc_path, extra=None):
    """build_corner.py / build_corner_modes.py as a library (their main() is not run)."""
    path = os.path.join(os.path.dirname(bc_path), name)
    if not os.path.isfile(path) and "__file__" in globals():
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), name)
    g = {"__name__": name[:-3], "_NO_MAIN": True, **(extra or {})}
    with open(path, encoding="utf-8") as fh:
        exec(compile(fh.read(), path, "exec"), g)
    return g


def read(path):
    import h5py
    D = {}
    with h5py.File(path, "r") as f:
        s = lambda v: v.decode() if isinstance(v, bytes) else v
        D["keys"] = json.loads(f.attrs["param_order"])
        D["modes"] = json.loads(f.attrs["modes"])
        D["labels"] = json.loads(f.attrs["mode_labels_json"])
        D["colors"] = json.loads(f.attrs["mode_colors_json"])
        D["fill"] = s(f.attrs["fill_mode"])
        D["fill_colors"] = json.loads(f.attrs["fill_colors_json"])          # outside, 2σ, 1σ
        D["sigma"] = [float(v) for v in f.attrs["sigma_levels"]]
        D["layout"] = f["layout"][()]
        D["params"] = {k: {a: s(v) for a, v in f["params"][k].attrs.items()} for k in D["keys"]}
        D["panels"] = {n: {d: g[d][()] for d in g} for n, g in f["panels"].items()}
    return D


def star(x, y, ux, uy, r_cm=BEST_RADIUS_CM):
    """Open five-pointed star around (x, y) as one closed polyline; ux, uy: data units per cm of the panel."""
    t = np.pi / 2 + np.arange(11) * np.pi / 5
    r = np.where(np.arange(11) % 2 == 0, r_cm, 0.42 * r_cm)
    return x + r * ux * np.cos(t), y + r * uy * np.sin(t)


def markers(bc, gl, book, name, px, py, geo, diag=False):
    """Truth (dashed lines) and best fit (star) of one panel; geo: panel (left, top, width, height) in cm."""
    step = bc["step"]
    tx, ty = px.get("truth"), None if diag else py.get("truth")
    if SHOW_TRUTH and tx is not None:
        w = step("sheet")(bc["new_sheet_xy"], book, f"truth_x_{name}", [("x", [tx, tx]), ("y", [py["lo"], py["hi"]])])
        if w is not None:
            step("truth")(bc["line_plot"], gl, w, 0, 1, TRUTH_COLOR, TRUTH_WIDTH, dash=1)
    if SHOW_TRUTH and ty is not None:
        w = step("sheet")(bc["new_sheet_xy"], book, f"truth_y_{name}", [("x", [px["lo"], px["hi"]]), ("y", [ty, ty])])
        if w is not None:
            step("truth")(bc["line_plot"], gl, w, 0, 1, TRUTH_COLOR, TRUTH_WIDTH, dash=1)
    bx, by = px.get("best_fit"), py.get("best_fit")
    if SHOW_BEST and not diag and bx is not None and by is not None:
        ux = (float(px["hi"]) - float(px["lo"])) / geo[2]
        uy = (float(py["hi"]) - float(py["lo"])) / geo[3]
        sx, sy = star(float(bx), float(by), ux, uy)
        w = step("sheet")(bc["new_sheet_xy"], book, f"best_{name}", [("x", sx), ("y", sy)])
        if w is not None:
            step("best fit")(bc["line_plot"], gl, w, 0, 1, BEST_COLOR, BEST_WIDTH)


def legend(bc, bm, gp, idx, D, height_cm, width_cm):
    """Legend in layer idx (axes 0..1): NN as dark / light swatches, ODE as solid / dashed pieces, truth, best fit,
    and which is 68 / 95 %."""
    op, lt, S = bc["op"], bc["lt"], bc["STYLE"]
    cx, cy = 1.0 / width_cm, 1.0 / height_cm
    row, xl, yt = bm["LEGEND_ROW_CM"] * cy, 0.03, 0.97
    gl = gp[idx]
    book = op.new_book("w", lname="legend_nn_vs_ode")
    C, fm, LW = D["colors"], D["fill"], bm["LINE_WIDTH"]
    w2 = LW["sigma1"] if SIGMA2_SAME_WIDTH else LW["sigma2"]
    entries = [(D["labels"][fm], [(D["fill_colors"][2], 6, 0), (D["fill_colors"][1], 6, 0)])]
    entries += [(D["labels"][m], [(C[m]["s1"], LW["sigma1"], 0), (C[m]["s1"], w2, SIGMA2_DASH.get(m, 1))])
                for m in D["modes"] if m != fm]
    if SHOW_TRUTH:
        entries.append(("true value", [(TRUTH_COLOR, TRUTH_WIDTH, 1)]))
    if SHOW_BEST:
        entries.append(("best fit", "star"))
    seg = 0.32 * cx
    for i, (text, sw) in enumerate(entries):
        y = yt - (i + 0.5) * row
        if sw == "star":
            sx, sy = star(xl + seg, y, cx, cy)
            w = bc["new_sheet_xy"](book, f"lg{i}", [("x", sx), ("y", sy)])
            bc["line_plot"](gl, w, 0, 1, BEST_COLOR, BEST_WIDTH)
        else:
            for k, (col, wid, dash) in enumerate(sw):
                xs = [xl + k * seg, xl + (k + 1) * seg - 0.04 * cx] if len(sw) > 1 else [xl, xl + 2 * seg]
                w = bc["new_sheet_xy"](book, f"lg{i}_{k}", [("x", xs), ("y", [y, y])])
                bc["line_plot"](gl, w, 0, 1, col, wid, dash=dash)
        nm = f"lgt{i}"
        nseg = 2 if sw == "star" else max(2, len(sw))      # text right of all swatches (NN: 2 fills + 2 lines)
        lt(f"win -a {gp.name}; page.active = {idx + 1};")
        lt(f"label -a 0 0 -n {nm} {text}; {nm}.fsize = {S['cbar_pt']}; {nm}.attach = 2;")
        lt(f"{nm}.x = {xl + nseg * seg + 0.15 * cx} + {nm}.dx / 2; {nm}.y = {y};")
    for i, text in enumerate(["68 %: dark / solid", "95 %: light / dashed"]):
        nm = f"lgh{i}"
        lt(f"label -a 0 0 -n {nm} {text}; {nm}.fsize = {S['cbar_pt']}; {nm}.attach = 2;")
        lt(f"{nm}.x = {xl} + {nm}.dx / 2; {nm}.y = {yt - (len(entries) + i + 0.5) * row};")


def main():
    bc = load("build_corner.py", BUILD_CORNER)
    bm = load("build_corner_modes.py", BUILD_CORNER)
    op, step, S = bc["op"], bc["step"], bc["STYLE"]
    if H5:
        bc["H5"] = H5
    path = bc["pick_file"]()
    D = read(path)
    layout, P = D["layout"], D["params"]
    LW = bm["LINE_WIDTH"]
    lw = lambda s_: LW["sigma1"] if SIGMA2_SAME_WIDTH else LW[s_]
    S.update(STYLE_SET)
    if X_LABEL_ROTATE is not None:
        S["x_label_rotate"] = X_LABEL_ROTATE
    n_traps = sum(k.startswith("log10_n1_") for k in P)
    if X_LABEL_COUNT.get(n_traps):
        S["x_label_count"] = X_LABEL_COUNT[n_traps]
    cmaps = []                                               # (page, bounds, colours, contour): second colour pass
    pos = bc["geometry"](layout)
    n = len(D["keys"])
    book = op.new_book("w", lname="corner_nn_vs_ode_data")
    pages, final, tasks, et_pages = [], {}, {}, {}

    def new_page(name, geo):
        gp = op.new_graph(lname=name)
        pages.append(gp)
        final[len(pages) - 1] = geo
        step("place layer")(bc["place_layer"], gp, 0, 0.5, 0.5, *geo[2:])
        return gp, gp[0], len(pages) - 1

    def fill_panel(name, gp, gl, g):
        """NN fill as colour-fill contour; heatmap on a new page if Origin refuses the contour (as build_corner_modes)."""
        Z, bnd, cols = bm["fill_map"](D, g)
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

    for row in layout:
        r, c, kind = int(row["row"]), int(row["col"]), bc["_s"](row["kind"])
        kx, ky, xside, yside, extra = (bc["_s"](row[k]) for k in ("kx", "ky", "xside", "yside", "extra"))
        px = P[kx]
        if kind == "2d":
            py = P[ky]
            g = D["panels"][f"{kx}__{ky}"]
            gp, gl, j = new_page(f"panel_{kx}__{ky}", pos(r, c))
            gp, gl = fill_panel(f"{kx}__{ky}", gp, gl, g)
            fm = D["fill"]                                   # NN lines (or a thin 1σ edge) on its fill, then the ODE lines
            lines = [(fm, s_, lw(s_), 0 if s_ == "sigma1" else SIGMA2_DASH.get(fm, 1))
                     for s_ in ("sigma2", "sigma1") if s_ in NN_LINES] if NN_LINES else [(fm, "sigma1", LW["fill_edge"], 0)]
            lines += [(m, s_, lw(s_), 0 if s_ == "sigma1" else SIGMA2_DASH.get(m, 1))
                      for m in D["modes"] if m != fm for s_ in ("sigma2", "sigma1")]
            for m, s_, wid, dash in lines:
                v = g.get(f"{m}_{s_}")
                if v is not None and len(v):
                    w = step("sheet")(bc["new_sheet_xy"], book, f"{m}_{s_}_{kx}__{ky}", [("x", v[:, 0]), ("y", v[:, 1])])
                    if w is not None:
                        step("contour line")(bc["line_plot"], gl, w, 0, 1, D["colors"][m]["s1"], wid, dash=dash)
            markers(bc, gl, book, f"{kx}__{ky}", px, py, final[j])
            bm["tick_lines"](bc, gl, book, f"{kx}__{ky}", px, py, final[j], kx.startswith("log10_"), ky.startswith("log10_"))
            step("axes")(bc["set_axes"], gl, (px["lo"], px["hi"]), (py["lo"], py["hi"]))
            tasks[j] = step("panel axes")(bc["panel_axes"], gp, 0, kx, ky, px, py, xside, yside, j) or []
        else:
            g = D["panels"][kx]
            gp, gl, j = new_page(f"panel_{kx}", pos(r, c))
            ymax = 1.1 * max(float(np.nanmax(g[f"{m}_pdf"])) for m in D["modes"])
            for m in D["modes"]:
                w = step("sheet")(bc["new_sheet_xy"], book, f"pdf_{m}_{kx}", [("x", g[f"{m}_x"]), ("y", g[f"{m}_pdf"])])
                if w is not None:
                    step("pdf")(bc["line_plot"], gl, w, 0, 1, D["colors"][m]["s1"], LW["pdf"])
            last = r == c == n - 1                           # one title, at the bottom-right panel (page margin)
            pyd = {"lo": 0.0, "hi": ymax, "label_origin": bm["DENSITY_TITLE"] if last else "", "nmax": 4}
            markers(bc, gl, book, f"pdf_{kx}", px, pyd, final[j], diag=True)
            bm["tick_lines"](bc, gl, book, f"pdf_{kx}", px, pyd, final[j], kx.startswith("log10_"), False)
            step("axes")(bc["set_axes"], gl, (px["lo"], px["hi"]), (0.0, ymax))
            tasks[j] = step("panel axes")(bc["panel_axes"], gp, 0, kx, "pdf", px, pyd, xside, "right", j) or []
            if extra.endswith("Et-Eg") and "log10_Nc" in px:
                et_pages[j] = (extra.split(":")[0], px)
        op.lt_exec(f"win -a {gp.name}; page.active = 1;")
        op.lt_exec("label -r legend;")

    # legend in a layer of its own, upper right (free upper triangle)
    l_, t_, w_, h_ = pos(0, n - 2, 2, 2)
    leg_geo = (l_ + LEGEND_SHIFT_CM, t_, w_ - LEGEND_SHIFT_CM, h_)
    gp, gl, jl = new_page("legend", leg_geo)
    step("axes")(bc["set_axes"], gl, (0.0, 1.0), (0.0, 1.0))
    hide = ("layer.x.showAxes = 0; layer.y.showAxes = 0; layer.x.ticks = 0; layer.y.ticks = 0; layer.x2.ticks = 0;"
            " layer.y2.ticks = 0; " + bc["label_sides"](0, 0, 0, 0))
    step("legend axes")(bc["lt"], f"win -a {gp.name}; page.active = 1; " + hide)
    op.lt_exec("xb.show = 0;")
    op.lt_exec("yl.show = 0;")
    op.lt_exec("label -r legend;")
    bc["REPLAY"][(jl, 1)] = hide

    # second colour pass (some colour maps stayed solid red in earlier Origin runs although no statement failed)
    for gp_, bnd, cols, contour in cmaps:
        op.lt_exec(f"win -a {gp_.name};")
        op.lt_exec("doc -uw;")
        if not bc["cmap_ready"](gp_, 0):
            bc["LOG"].append(f"[colour levels (2nd pass)] {gp_.lname}: no colour map, skipped")
            continue
        step("colour levels (2nd pass)")(bc["set_cmap"], gp_, 0, bnd, cols)
        if contour:
            op.lt_exec("layer.cmap.showLines(3);")
            op.lt_exec("layer.cmap.showLabels(3);")
            op.lt_exec("layer.cmap.updateScale();")
    print(f"colour maps: {len(cmaps)} pages, levels and colours written twice")

    # merge, put every layer back, formatting again, 10^n labels, E_t - E_g axis, legend, fit the page
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
    for j, tl in tasks.items():
        for t in tl:
            if j in first:
                step("10^n labels")(bc["decade_labels"], gp, first[j], *t)
    for j, (where, px) in et_pages.items():
        if j in first:
            step("E_t - E_g axis")(bc["draw_et"], gp, first[j], where, px)
    if jl in first:
        step("legend")(legend, bc, bm, gp, first[jl], D, leg_geo[3], leg_geo[2])
    step("fit page")(bc["lt"], f"win -a {gp.name}; pfit2l margin:=0 borderwid:=2;")
    step("page size")(bc["page_size_check"], gp)

    base = os.path.join(os.path.dirname(path), OUT_NAME)
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
        for k, (cnt, ex) in bc["LT_FAILED"].items():
            print(f"   {cnt:4d} x  {ex}")


if not globals().get("_NO_MAIN"):
    main()
