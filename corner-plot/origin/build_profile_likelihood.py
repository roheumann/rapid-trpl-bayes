"""
build_profile_likelihood.py — N_t profile likelihood of the synthetic 1-trap case in Origin (2021+, Windows)
===========================================================================================================
Input : the folder results/1trap_synthetic_sigtr_GaAs/degeneracy_search/ written on rekenbeest by
            python scripts/degeneracy_search_1trap.py      (profile_Nt.csv, setup.json)
            python scripts/degeneracy_plot_1trap.py        (curves_trpl.csv, curves_sspl.csv, residuals_at_data.csv)
        plus data_trpl.csv / data_sspl.csv (copies of the synthetic data files).  Copy the whole folder to Windows.
Output: next to those files: profile_likelihood.opju (workbook + graphs) and one .pdf / .png per graph.

Run inside Origin:   run -pyf "C:\\...\\corner-plot\\origin\\build_profile_likelihood.py"
A file dialog asks for profile_Nt.csv (FOLDER = None); no extra Python packages needed (csv + numpy).

Graphs (each graph page has ONE layer: on merged / multi-layer pages, symbol plots and the tick-label switches did
not work in earlier Origin runs of this project):
  profile_logL     -log L (joint, profile over E_t, tau_n, tau_p) vs N_t (log x); truth = dashed vertical line;
                   dotted horizontal lines at Delta(-log L) = 0.5 and 1.92 (68 % / 95 % of a 1-parameter profile)
  profile_tau      profiled tau_n and tau_p vs N_t (log-log), truth as open symbols
  profile_Et       profiled E_t vs N_t
  trpl_curves      tau_diff vs QFLS: synthetic data (open circles), truth (black) and the chosen profile points
  trpl_residuals   ODE - data (dec) at the 90 data points, dashed lines at +-sigma_trPL
  sspl_curves      QFLS vs suns: data and curves
  sspl_residuals   model - data (meV) at the 10 intensities, dashed lines at +-sigma_ssPL
"""
import csv
import json
import os
import re
import traceback

import numpy as np
import originpro as op

FOLDER = None              # None -> file dialog (pick profile_Nt.csv)
STYLE = {"font": "Arial", "tick_pt": 9, "title_pt": 11, "legend_pt": 8, "line_pt": 1.5, "ref_pt": 1.0, "sym": 6}
FRAME_CM = (2.2, 0.8, 9.0, 6.5)            # layer left, top, width, height (cm)
PAGE_START_CM = 20.0
TRUTH_COLOR = "#000000"
SET_COLORS = ["#FF7F0E", "#2CA02C", "#D62728", "#9467BD", "#8C564B", "#E377C2"]   # profile points (order of the CSV)
DATA_COLOR = "#595959"
DASH = {"solid": 0, "dash": 1, "dot": 2}
SHAPE = {"square": 1, "circle": 2}

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


def pick_folder():
    if FOLDER:
        return FOLDER
    lt('dlgfile group:="profile_Nt.csv" title:="profile_Nt.csv of degeneracy_search"; string __f$ = fname$;')
    return os.path.dirname(op.get_lt_str("__f$"))


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return {k: [r[k] for r in rows] for k in rows[0]}


def col(T, k, mask=None):
    v = np.array([float(x) if x not in ("", "nan") else np.nan for x in T[k]])
    return v if mask is None else v[mask]


def sets_of(T):
    out = []
    for s in T["set"]:
        if s not in out:
            out.append(s)
    return out


def minus(s):
    """Origin minus sign (a leading '-' after 'label' would be read as a switch)."""
    return s.replace("-", "\\x(2212)")


# ─────────────────────────────────────────────────────────────────────────────
# Origin helpers
# ─────────────────────────────────────────────────────────────────────────────
NAMES = []                 # long names of the graph pages (export file names)


def new_page(name):
    NAMES.append(name)
    gp = op.new_graph(lname=name)
    lt(f"win -a {gp.name}; page.height = {PAGE_START_CM / 2.54:.4f} * page.resy;")
    l, t, w, h = FRAME_CM
    lt(f"page.active = 1; layer.fixed = 1; layer.factor = 1; layer.unit = 3; layer.left = {l}; layer.top = {t};"
       f" layer.width = {w}; layer.height = {h};")
    return gp, gp[0]


def line_plot(gl, wks, cx, cy, color, width_pt, dash=0):
    p = gl.add_plot(wks, coly=cy, colx=cx, type="l")
    p.color = color
    p.set_cmd(f"-w {int(width_pt * 500)} -d {dash}")
    return p


def sym_plot(gl, wks, cx, cy, color, shape=2, size=6, line_and_sym=False):
    p = gl.add_plot(wks, coly=cy, colx=cx, type="y" if line_and_sym else "s")
    p.color = color
    p.set_cmd(f"-k {shape} -kf 0 -z {size}")        # -kf 0: open symbol
    if line_and_sym:
        p.set_cmd(f"-w {int(STYLE['line_pt'] * 500)}")
    return p


def axes(xr, yr, xlog=False, ylog=False):
    """Ranges, scale types, ticks on all sides, labels left/bottom only; 10^n labels on log axes."""
    lt(f"layer.x.type = {2 if xlog else 1}; layer.x.from = {xr[0]:.8g}; layer.x.to = {xr[1]:.8g};"
       f" layer.y.type = {2 if ylog else 1}; layer.y.from = {yr[0]:.8g}; layer.y.to = {yr[1]:.8g};"
       " layer.x.showAxes = 3; layer.y.showAxes = 3; layer.x.ticks = 5; layer.x2.ticks = 5; layer.y.ticks = 5;"
       " layer.y2.ticks = 5;")
    for ax in ("x", "y"):
        lt(f"layer.{ax}.label.pt = {STYLE['tick_pt']}; layer.{ax}.label.font = font({STYLE['font']});")
    if xlog:
        lt("layer.x.inc = 1; layer.x.label.numFormat = 2;")
    if ylog:
        lt("layer.y.inc = 1; layer.y.label.numFormat = 2;")
    lt("layer.x.showlabel = 1; layer.y.showlabel = 1; layer.x2.showlabel = 0; layer.y2.showlabel = 0;")


def titles(xt, yt):
    lt(f'label -xb {xt}; xb.text$ = "{xt}"; xb.fsize = {STYLE["title_pt"]};'
       f' label -yl {yt}; yl.text$ = "{yt}"; yl.fsize = {STYLE["title_pt"]};')


def legend(name, entries, corner="tr"):
    """
    entries: list of (plot index in the layer, text).  The legend stays where Origin puts it (inside the frame,
    top); move it by hand if it covers a curve.  `corner` is kept only as a hint in the printed summary.
    """
    txt = "\\n".join(f"\\l({i}) {t}" for i, t in entries)
    lt(f"label -s -sa -n {name} {txt}; {name}.fsize = {STYLE['legend_pt']};")
    print(f"legend {name}: suggested corner {corner}")


def finish(gp):
    lt(f"win -a {gp.name}; pfit2l margin:=0 borderwid:=2;")


def nice_range(v, pad=0.06, log=False):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if log:
        return 10 ** np.floor(np.log10(v.min())), 10 ** np.ceil(np.log10(v.max()))
    lo, hi = v.min(), v.max()
    d = (hi - lo) or abs(hi) or 1.0
    return lo - pad * d, hi + pad * d


# ─────────────────────────────────────────────────────────────────────────────
# build
# ─────────────────────────────────────────────────────────────────────────────
def build(folder):
    P = read_csv(os.path.join(folder, "profile_Nt.csv"))
    TR = read_csv(os.path.join(folder, "curves_trpl.csv"))
    SS = read_csv(os.path.join(folder, "curves_sspl.csv"))
    RES = read_csv(os.path.join(folder, "residuals_at_data.csv"))
    DT = read_csv(os.path.join(folder, "data_trpl.csv"))
    DS = read_csv(os.path.join(folder, "data_sspl.csv"))
    setup = json.load(open(os.path.join(folder, "setup.json"), encoding="utf-8"))
    truth = setup["truth"]
    sig_tr, sig_ss = float(setup["sigma_trpl"]), 1e3 * float(setup["sigma_sspl"])
    nt_true, et_true = float(truth["Nt_cm3"][0]), float(truth["Et_eV"][0])
    taun_true, taup_true = float(truth["tau_n_s"][0]), float(truth["tau_p_s"][0])

    nt, nll = col(P, "Nt_cm3"), col(P, "neg_logL")
    o = np.argsort(nt)
    nt, nll = nt[o], nll[o]
    et, tn, tp = col(P, "Et_eV")[o], col(P, "taun_s")[o], col(P, "taup_s")[o]
    sets = sets_of(TR)
    colors = {s: (TRUTH_COLOR if s == "truth" else SET_COLORS[(i - 1) % len(SET_COLORS)]) for i, s in enumerate(sets)}

    def set_label(s):
        if s == "truth":
            return "truth"
        k = int(np.argmin(np.abs(np.log10(nt) - float(s.split("1e")[1]))))
        return (f"\\i(N)\\-(t) {nt[k]:.2g} cm\\+(\\x(2212)3), \\i(E)\\-(t) {et[k]:.3f} eV "
                f"(\\x(2212)log \\i(L) {nll[k]:.2f})").replace("e+", "e")

    # workbook
    book = op.new_book("w", lname="profile_likelihood")
    wp = book[0]
    wp.name = "profile"
    for k, (v, nm) in enumerate([(nt, "N_t (cm^-3)"), (nll, "-log L joint"), (et, "E_t (eV)"), (tn, "tau_n (s)"),
                                 (tp, "tau_p (s)"), (col(P, "rmse_trpl_dec")[o], "tr-PL RMSE (dec)"),
                                 (col(P, "rmse_sspl_meV")[o], "ss-PL RMSE (meV)")]):
        wp.from_list(k, [float(x) for x in v], lname=nm)
    yl_lo, yl_hi = -0.05 * max(nll.max(), 2.0), 1.08 * nll.max()
    ref = book.add_sheet("reference_lines")
    xr = (10 ** np.floor(np.log10(nt.min())), 10 ** np.ceil(np.log10(nt.max())))
    for k, (x, y, nm) in enumerate([([nt_true, nt_true], [yl_lo, yl_hi], "truth N_t"),
                                    (list(xr), [0.5, 0.5], "Delta 0.5 (68 %)"),
                                    (list(xr), [1.92, 1.92], "Delta 1.92 (95 %)")]):
        ref.from_list(2 * k, [float(v) for v in x], lname=f"x {nm}")
        ref.from_list(2 * k + 1, [float(v) for v in y], lname=nm)
    tsh = book.add_sheet("truth_point")
    for k, (v, nm) in enumerate([([nt_true], "N_t"), ([taun_true], "tau_n"), ([taup_true], "tau_p"), ([et_true], "E_t")]):
        tsh.from_list(k, [float(x) for x in v], lname=nm)

    ct = book.add_sheet("trpl_curves")
    cs = book.add_sheet("sspl_curves")
    rt = book.add_sheet("trpl_residuals")
    rs = book.add_sheet("sspl_residuals")
    dd = book.add_sheet("data")
    dd.from_list(0, [float(v) for v in DT["qfls_eV"]], lname="QFLS (eV)")
    dd.from_list(1, [float(v) for v in DT["tau_diff_s"]], lname="tau_diff (s)")
    dd.from_list(2, [float(v) for v in DS["intensity_suns"]], lname="suns")
    dd.from_list(3, [float(v) for v in DS["qfls_eV"]], lname="QFLS ss-PL (eV)")
    for i, s in enumerate(sets):
        m = np.array([x == s for x in TR["set"]])
        ct.from_list(2 * i, [float(v) for v in col(TR, "qfls_eV", m)], lname=f"QFLS {s}")
        ct.from_list(2 * i + 1, [float(v) for v in col(TR, "tau_diff_s", m)], lname=f"tau_diff {s}")
        m = np.array([x == s for x in SS["set"]])
        cs.from_list(2 * i, [float(v) for v in col(SS, "intensity_suns", m)], lname=f"suns {s}")
        cs.from_list(2 * i + 1, [float(v) for v in col(SS, "qfls_eV", m)], lname=f"QFLS {s}")
        m = np.array([x == s for x in RES["set"]])
        q, r = col(RES, "qfls_eV", m), col(RES, "resid_log10_tau_dec", m)
        k = np.isfinite(q) & np.isfinite(r)
        rt.from_list(2 * i, [float(v) for v in q[k]], lname=f"QFLS {s}")
        rt.from_list(2 * i + 1, [float(v) for v in r[k]], lname=f"ODE - data {s} (dec)")
        x, r = col(RES, "intensity_suns", m), col(RES, "resid_qfls_meV", m)
        k = np.isfinite(x) & np.isfinite(r)
        rs.from_list(2 * i, [float(v) for v in x[k]], lname=f"suns {s}")
        rs.from_list(2 * i + 1, [float(v) for v in r[k]], lname=f"model - data {s} (meV)")
    bands = book.add_sheet("sigma_lines")
    qd = col(DT, "qfls_eV")
    sd = col(DS, "intensity_suns")
    qx = [float(qd.min() - 0.03), float(qd.max() + 0.03)]
    sx = [float(sd.min() / 2), float(sd.max() * 2)]
    for k, (x, y, nm) in enumerate([(qx, [sig_tr, sig_tr], "+sigma_tr"), (qx, [-sig_tr, -sig_tr], "-sigma_tr"),
                                    (sx, [sig_ss, sig_ss], "+sigma_ss"), (sx, [-sig_ss, -sig_ss], "-sigma_ss"),
                                    (qx, [0, 0], "zero tr"), (sx, [0, 0], "zero ss")]):
        bands.from_list(2 * k, x, lname=f"x {nm}")
        bands.from_list(2 * k + 1, y, lname=nm)

    pages = []
    # 1. profile -log L
    gp, gl = new_page("profile_logL")
    line_plot(gl, ref, 2, 3, "#808080", STYLE["ref_pt"], DASH["dot"])
    line_plot(gl, ref, 4, 5, "#808080", STYLE["ref_pt"], DASH["dot"])
    line_plot(gl, ref, 0, 1, TRUTH_COLOR, STYLE["ref_pt"], DASH["dash"])
    sym_plot(gl, wp, 0, 1, "#1F77B4", SHAPE["circle"], STYLE["sym"], line_and_sym=True)
    axes(xr, (yl_lo, yl_hi), xlog=True)
    titles("\\i(N)\\-(t) (cm\\+(\\x(2212)3))", "\\x(2212)log \\i(L) (joint, profiled)")
    legend("lgp", [(4, "profile (\\i(E)\\-(t), \\i(\\g(t))\\-(n), \\i(\\g(t))\\-(p) refitted, ODE)"),
                   (3, "truth"), (1, "\\x(0394) = 0.5 / 1.92 (68 % / 95 %)")], corner="tl")
    finish(gp); pages.append((gp, NAMES[-1]))

    # 2. profiled tau_n, tau_p
    gp, gl = new_page("profile_tau")
    sym_plot(gl, wp, 0, 3, "#1F77B4", SHAPE["circle"], STYLE["sym"], line_and_sym=True)
    sym_plot(gl, wp, 0, 4, "#D62728", SHAPE["square"], STYLE["sym"], line_and_sym=True)
    sym_plot(gl, tsh, 0, 1, "#1F77B4", SHAPE["circle"], STYLE["sym"] + 4)
    sym_plot(gl, tsh, 0, 2, "#D62728", SHAPE["square"], STYLE["sym"] + 4)
    lo, hi = nice_range(np.r_[tn, tp], log=True)
    axes(xr, (lo, hi), xlog=True, ylog=True)
    titles("\\i(N)\\-(t) (cm\\+(\\x(2212)3))", "profiled lifetime (s)")
    legend("lgt", [(1, "\\i(\\g(t))\\-(n)"), (2, "\\i(\\g(t))\\-(p)"), (3, "truth (large open symbols)")], corner="tr")
    finish(gp); pages.append((gp, NAMES[-1]))

    # 3. profiled E_t
    gp, gl = new_page("profile_Et")
    sym_plot(gl, wp, 0, 2, "#1F77B4", SHAPE["circle"], STYLE["sym"], line_and_sym=True)
    sym_plot(gl, tsh, 0, 3, "#000000", SHAPE["circle"], STYLE["sym"] + 4)
    axes(xr, nice_range(np.r_[et, et_true]), xlog=True)
    titles("\\i(N)\\-(t) (cm\\+(\\x(2212)3))", "profiled \\i(E)\\-(t) (eV)")
    legend("lge", [(1, "profile"), (2, "truth")], corner="tl")
    finish(gp); pages.append((gp, NAMES[-1]))

    # 4. tr-PL curves
    gp, gl = new_page("trpl_curves")
    sym_plot(gl, dd, 0, 1, DATA_COLOR, SHAPE["circle"], STYLE["sym"] - 2)
    ent = [(1, "synthetic data")]
    for i, s in enumerate(sets):
        line_plot(gl, ct, 2 * i, 2 * i + 1, colors[s], STYLE["line_pt"])
        ent.append((i + 2, set_label(s)))
    lo, hi = nice_range(col(DT, "tau_diff_s"), log=True)
    axes((qx[0], qx[1]), (lo, hi), ylog=True)
    titles("QFLS (eV)", "\\i(\\g(t))\\-(diff) (s)")
    legend("lgc", ent, corner="tr")
    finish(gp); pages.append((gp, NAMES[-1]))

    # 5. tr-PL residuals
    gp, gl = new_page("trpl_residuals")
    line_plot(gl, bands, 0, 1, "#808080", STYLE["ref_pt"], DASH["dash"])
    line_plot(gl, bands, 2, 3, "#808080", STYLE["ref_pt"], DASH["dash"])
    line_plot(gl, bands, 8, 9, "#808080", 0.5, DASH["solid"])
    for i, s in enumerate(sets):
        sym_plot(gl, rt, 2 * i, 2 * i + 1, colors[s], SHAPE["circle"], STYLE["sym"] - 3, line_and_sym=True)
    r_all = col(RES, "resid_log10_tau_dec")
    m = np.nanmax(np.abs(np.r_[r_all[np.isfinite(r_all)], sig_tr]))
    axes((qx[0], qx[1]), (-1.1 * m, 1.1 * m))
    titles("QFLS (eV)", "ODE \\x(2212) data (dec)")
    legend("lgr", [(1, f"\\+-\\i(\\g(s))\\-(trPL) = {sig_tr:.3f} dec")], corner="tl")
    finish(gp); pages.append((gp, NAMES[-1]))

    # 6. ss-PL curves
    gp, gl = new_page("sspl_curves")
    sym_plot(gl, dd, 2, 3, DATA_COLOR, SHAPE["circle"], STYLE["sym"])
    ent = [(1, "synthetic data")]
    for i, s in enumerate(sets):
        line_plot(gl, cs, 2 * i, 2 * i + 1, colors[s], STYLE["line_pt"])
        ent.append((i + 2, set_label(s)))
    axes(nice_range(col(SS, "intensity_suns"), log=True), nice_range(col(SS, "qfls_eV")), xlog=True)
    titles("illumination (suns)", "QFLS (eV)")
    legend("lgs", ent, corner="tl")
    finish(gp); pages.append((gp, NAMES[-1]))

    # 7. ss-PL residuals
    gp, gl = new_page("sspl_residuals")
    line_plot(gl, bands, 4, 5, "#808080", STYLE["ref_pt"], DASH["dash"])
    line_plot(gl, bands, 6, 7, "#808080", STYLE["ref_pt"], DASH["dash"])
    line_plot(gl, bands, 10, 11, "#808080", 0.5, DASH["solid"])
    for i, s in enumerate(sets):
        sym_plot(gl, rs, 2 * i, 2 * i + 1, colors[s], SHAPE["circle"], STYLE["sym"] - 2, line_and_sym=True)
    r_all = col(RES, "resid_qfls_meV")
    m = np.nanmax(np.abs(np.r_[r_all[np.isfinite(r_all)], sig_ss]))
    axes((sx[0], sx[1]), (-1.15 * m, 1.15 * m), xlog=True)
    titles("illumination (suns)", "model \\x(2212) data (meV)")
    legend("lgq", [(1, f"\\+-\\i(\\g(s))\\-(ssPL) = {sig_ss:.2f} meV")], corner="tl")
    finish(gp); pages.append((gp, NAMES[-1]))
    return pages


def main():
    folder = pick_folder()
    pages = step("build")(build, folder)
    base = os.path.join(folder, "profile_likelihood")
    try:
        op.save(base + ".opju")
    except Exception as e:                              # noqa: BLE001
        print(f"op.save failed ({e}); trying LabTalk save")
        lt(f'save "{base}.opju";')
    for gp, nm in pages or []:
        step(f"export {nm}")(gp.save_fig, os.path.join(folder, nm + ".pdf"))
        step(f"export {nm} png")(gp.save_fig, os.path.join(folder, nm + ".png"), width=2400)
    print(f"done: {base}.opju and one .pdf / .png per graph")
    for m in LOG:
        print("failed step:", m)
    if LT_FAILED:
        print("\nLabTalk statements that failed (count, first example) - please send this list back:")
        for c, ex in LT_FAILED.values():
            print(f"   {c:4d} x  {ex}")


main()
