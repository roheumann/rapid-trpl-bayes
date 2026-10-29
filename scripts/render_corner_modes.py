"""
render_corner_modes.py — matplotlib rendering of a corner_modes.h5 in the layout of the paper's 2-trap joint figure
====================================================================================================================
23/10/2026 (Robin, 77:23 film).  Same content as the Origin figure (corner-plot/origin/build_corner_modes.py), from the
same HDF5 written by scripts/modes_corner.py (--modes joint, two slice panels, extras):

  (a)  joint SMC posterior corner: 68 % (dark) / 95 % (light) HPD regions, 1-D marginals (p per decade, E_t - E_g axis
       on top of the n1 columns), median (gold square / dashed) and best fit (blue star / dotted)
  (b)  sliced grid n1,1 - tau_p,1 through the best fit, colour (p_best - p)/p_best, iso-tau_0 lines
  (c)  the same for trap 2, plus the iso-tau_0 lines scaled by N_t1/N_t2
  (d)  tau_diff(QFLS): data, joint posterior predictive 68/95 % bands, median curve, best fit (ODE)
  (e)  ss-PL QFLS vs illumination: the same

    python scripts/render_corner_modes.py <run>/corner_modes/corner_modes.h5 [--out <dir>]

Writes corner_modes_figure.png/.pdf next to the h5 file (or into --out).
"""
import argparse
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap

GOLD, BLUE = "#D4A017", "#1F3FFF"
LABEL = {"log10_n1_1": r"$n_{1,1}$ (cm$^{-3}$)", "log10_taup_1": r"$\tau_\mathrm{SRH,p1}$ (s)",
         "log10_taun_1": r"$\tau_\mathrm{SRH,n1}$ (s)", "log10_Nt_1": r"$N_\mathrm{t1}$ (cm$^{-3}$)",
         "log10_n1_2": r"$n_{1,2}$ (cm$^{-3}$)", "log10_taup_2": r"$\tau_\mathrm{SRH,p2}$ (s)",
         "log10_taun_2": r"$\tau_\mathrm{SRH,n2}$ (s)", "log10_Nt_2": r"$N_\mathrm{t2}$ (cm$^{-3}$)"}


def attr(g, k, default=np.nan):
    v = g.attrs.get(k, default)
    try:
        return float(v)
    except (TypeError, ValueError):
        return v


def log_ticks(ax, lo, hi, which="x", step=None):
    """10^n labels on a log10 axis, every `step` decades (chosen from the range)."""
    span = hi - lo
    step = step or (1 if span <= 4 else 2 if span <= 8 else 4)
    t = np.arange(np.ceil(lo / step) * step, hi + 1e-9, step)
    lab = [rf"$10^{{{int(v)}}}$" for v in t]
    (ax.set_xticks if which == "x" else ax.set_yticks)(t)
    (ax.set_xticklabels if which == "x" else ax.set_yticklabels)(lab)


def corner(fig, f, gs_box):
    keys = json.loads(f.attrs["param_order"])
    fills = json.loads(f.attrs["fill_colors_json"])
    P = {k: f["params"][k] for k in keys}
    n = len(keys)
    x0, y0, w, h = gs_box
    pad = 0.004
    cw, ch = w / n, h / n
    axes = {}
    for i in range(n):          # row
        for j in range(i + 1):  # col
            ax = fig.add_axes([x0 + j * cw + pad, y0 + (n - 1 - i) * ch + pad, cw - 2 * pad, ch - 2 * pad])
            axes[i, j] = ax
            kx = keys[j]
            lox, hix = attr(P[kx], "lo"), attr(P[kx], "hi")
            if i == j:
                g = f["panels"][kx]
                ax.plot(g["joint_x"][()], g["joint_pdf"][()], color="#B2182B", lw=1.2)
                ax.axvline(attr(P[kx], "median"), color=GOLD, ls="--", lw=1.4)
                ax.axvline(attr(P[kx], "best_fit"), color=BLUE, ls=":", lw=1.4)
                ax.set_ylim(0, None)
                ax.yaxis.tick_right(); ax.tick_params(axis="y", labelsize=6)
                if kx.startswith("log10_n1"):       # E_t - E_g on top
                    kTl, lNc = attr(P[kx], "kT_ln10"), attr(P[kx], "log10_Nc")
                    top = ax.secondary_xaxis("top", functions=(lambda x, a=kTl, b=lNc: a * (x - b),
                                                               lambda e, a=kTl, b=lNc: e / a + b))
                    top.set_xlabel(rf"$E_\mathrm{{t{kx[-1]}}} - E_\mathrm{{g}}$ (eV)", fontsize=7)
                    top.tick_params(labelsize=6)
            else:
                ky = keys[i]
                g = f["panels"][f"{kx}__{ky}"]
                lev = g["cat"][()] // 9 if len(json.loads(f.attrs["modes"])) > 1 else g["cat"][()]   # l_joint
                ax.pcolormesh(g["x"][()], g["y"][()], lev, cmap=ListedColormap(fills), vmin=-0.5, vmax=2.5,
                              shading="nearest", rasterized=True)
                ax.plot(attr(P[kx], "median"), attr(P[ky], "median"), "s", mfc="none", mec=GOLD, ms=5, mew=1.2)
                ax.plot(attr(P[kx], "best_fit"), attr(P[ky], "best_fit"), "*", mfc="none", mec=BLUE, ms=7, mew=1.0)
                ax.set_ylim(attr(P[ky], "lo"), attr(P[ky], "hi"))
                if j == 0:
                    ax.set_ylabel(LABEL.get(ky, ky), fontsize=7); log_ticks(ax, attr(P[ky], "lo"), attr(P[ky], "hi"), "y")
                    ax.tick_params(axis="y", labelsize=6)
                else:
                    ax.set_yticklabels([])
            ax.set_xlim(lox, hix)
            if i == n - 1:
                ax.set_xlabel(LABEL.get(kx, kx), fontsize=7); log_ticks(ax, lox, hix, "x")
                ax.tick_params(axis="x", labelsize=6, rotation=45)
            else:
                ax.set_xticklabels([])
            ax.tick_params(direction="in", length=2)
    axes[n - 1, n - 1].set_ylabel(r"$p$ (dec$^{-1}$)", fontsize=7); axes[n - 1, n - 1].yaxis.set_label_position("right")
    return axes


def slice_panel(ax, g, cax=None, ntratio=False):
    kx, ky = attr(g, "kx"), attr(g, "ky")
    lev = np.concatenate([[0.0], np.array(json.loads(str(g.attrs["hpd_levels"]).replace("  ", " ").replace(" ", ",")
                                                     .replace("[,", "[").replace(",]", "]"))), [1.0]])
    z = g["hpd"][()]                              # z[iy, ix] (rows = y), as pcolormesh expects
    cmap = plt.get_cmap("inferno_r", len(lev) - 1)
    m = ax.pcolormesh(g["x"][()], g["y"][()], z, cmap=cmap, norm=BoundaryNorm(lev, cmap.N), shading="nearest",
                      rasterized=True)
    styles = [("tau0_photodoping", "#00CED1", "--", "photodoping"), ("tau0_non_photodoping", "white", "-.", "no photodoping"),
              ("tau0_taun_plus_taup", "#00CED1", "--", r"$\tau_\mathrm{n} + \tau_\mathrm{p}$ = best fit")]
    if ntratio:
        styles += [("tau0_photodoping_ntratio", "#7CFC00", ":", r"photodoping $\times N_\mathrm{t1}/N_\mathrm{t2}$"),
                   ("tau0_non_photodoping_ntratio", "#FF77FF", ":", r"no photodoping $\times N_\mathrm{t1}/N_\mathrm{t2}$")]
    for key, col, ls, lab in styles:
        if key in g and np.isfinite(g[key][()]).all(axis=-1).any():     # no legend entry for an empty line
            xy = g[key][()]
            ax.plot(xy[:, 0], xy[:, 1], color=col, ls=ls, lw=1.2, label=lab)
    px, py = g["params"][kx], g["params"][ky]
    ax.plot(attr(px, "best_fit"), attr(py, "best_fit"), "*", mfc="none", mec=BLUE, ms=8)
    xb, yb = attr(px, "best_fit"), attr(py, "best_fit")
    # zoom on the region with p > 10 % of the best fit (as the paper panels)
    X, Y = np.meshgrid(g["x"][()], g["y"][()])     # same [iy, ix] layout as z
    if (z < 0.9).any():
        xs, ys = X[z < 0.9], Y[z < 0.9]
        def lim(lo, hi, full, min_span=2.0):      # at least min_span decades (e.g. a fit at the prior edge)
            lo, hi = lo - 0.1, hi + 0.1
            if hi - lo < min_span:
                c = 0.5 * (lo + hi); lo, hi = c - min_span / 2, c + min_span / 2
            return max(lo, full.min()), min(hi, full.max())
        ax.set_xlim(*lim(min(xs.min(), xb), max(xs.max(), xb), g["x"][()]))
        ax.set_ylim(*lim(min(ys.min(), yb), max(ys.max(), yb), g["y"][()]))
    ax.set_xlabel(LABEL.get(kx, kx), fontsize=8); ax.set_ylabel(LABEL.get(ky, ky), fontsize=8)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: rf"$10^{{{v:.1f}}}$"))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: rf"$10^{{{v:.1f}}}$"))
    ax.tick_params(labelsize=6)
    ax.legend(fontsize=5, loc="upper right", framealpha=0.3)
    if cax is not None:
        cb = plt.colorbar(m, cax=cax, ticks=[0.01, 0.05, 0.2, 0.4, 0.6, 0.8, 0.99])
        cb.ax.set_yticklabels(["1 %", "5 %", "20 %", "40 %", "60 %", "80 %", "99 %"], fontsize=6)
        cb.ax.invert_yaxis()
        cb.set_label(r"$(p_\mathrm{best\,fit} - p)/p_\mathrm{best\,fit}$", fontsize=7)


def ppc_panel(ax, g, xlog=False, ylog10=True):
    x = g["data_qfls"][()] if "data_qfls" in g else None
    fills = ("#F4A582", "#B2182B")
    if "data_qfls" in g:          # tr-PL: bands at the data QFLS points
        xb = g["data_qfls"][()]
        tr = (lambda v: 10 ** v)
        dx, dy = g["data_qfls"][()], g["data_y"][()]
    else:                         # ss-PL: bands on the x grid
        xb = g["x"][()]
        tr = (lambda v: v)
        dx, dy = g["data_x"][()], g["data_y"][()]
    xp = (lambda v: 10 ** v) if xlog else (lambda v: v)
    o = np.argsort(xb)
    ax.fill_between(xp(xb[o]), tr(g["joint_q2.5"][()][o]), tr(g["joint_q97.5"][()][o]), color=fills[0], lw=0, label="joint 95 %")
    ax.fill_between(xp(xb[o]), tr(g["joint_q16"][()][o]), tr(g["joint_q84"][()][o]), color=fills[1], lw=0, label="joint 68 %")
    if "median_x" in g:
        mo = np.argsort(g["median_x"][()])
        ax.plot(xp(g["median_x"][()][mo]), tr(g["median_y"][()][mo]), color=GOLD, lw=1.5, label="median (joint)")
    if "best_x" in g:
        bo = np.argsort(g["best_x"][()])
        ax.plot(xp(g["best_x"][()][bo]), tr(g["best_y"][()][bo]), color=BLUE, lw=1.3, label="best fit (joint, ODE)")
    ax.plot(xp(dx), tr(dy), "o", mfc="none", mec="k", ms=4, mew=1.2, label="data")
    if ylog10 and "data_qfls" in g:
        ax.set_yscale("log")
    if xlog:
        ax.set_xscale("log")
    ax.set_xlim(xp(dx.min()) * (0.8 if xlog else 1) - (0 if xlog else 0.02), xp(dx.max()) * (1.25 if xlog else 1) + (0 if xlog else 0.02))
    ax.tick_params(labelsize=7)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("h5")
    ap.add_argument("--out", default=None)
    ap.add_argument("--title", default=None)
    a = ap.parse_args()
    out = Path(a.out) if a.out else Path(a.h5).parent
    out.mkdir(parents=True, exist_ok=True)
    with h5py.File(a.h5, "r") as f:
        fig = plt.figure(figsize=(10.5, 15.5))
        corner(fig, f, (0.07, 0.36, 0.72, 0.58))
        fig.text(0.205, 0.905, "(a)", fontsize=11)
        cax = fig.add_axes([0.93, 0.70, 0.015, 0.23])
        if "slice" in f:
            axb = fig.add_axes([0.74, 0.82, 0.16, 0.11]); slice_panel(axb, f["slice"], cax=cax)
            axb.set_title("(b)", fontsize=10, loc="right")
        if "slice_2" in f:
            axc = fig.add_axes([0.74, 0.66, 0.16, 0.11]); slice_panel(axc, f["slice_2"], ntratio=True)
            axc.set_title("(c)", fontsize=10, loc="right")
        axd = fig.add_axes([0.08, 0.05, 0.42, 0.26]); ppc_panel(axd, f["ppc"])
        axd.set_xlabel(r"Fermi-level splitting $\Delta E_\mathrm{F}$ (eV)"); axd.set_ylabel(r"differential decay time $\tau_\mathrm{diff}$ (s)")
        axd.set_title("(d)", loc="right")
        axe = fig.add_axes([0.55, 0.05, 0.37, 0.26]); ppc_panel(axe, f["sspl_ppc"], xlog=True, ylog10=False)
        axe.set_xlabel("illumination (suns)"); axe.set_ylabel(r"Fermi-level splitting $\Delta E_\mathrm{F}$ (eV)")
        axe.yaxis.set_label_position("right"); axe.yaxis.tick_right(); axe.set_title("(e)", loc="left")
        axd.legend(fontsize=7, loc="lower left")
        h = [plt.Rectangle((0, 0), 1, 1, color="#B2182B"), plt.Rectangle((0, 0), 1, 1, color="#F4A582"),
             plt.Line2D([], [], marker="o", mfc="none", mec="k", ls=""),
             plt.Line2D([], [], color=GOLD, marker="s", mfc="none"), plt.Line2D([], [], color=BLUE, marker="*", mfc="none")]
        fig.legend(h, ["joint 68 %", "joint 95 %", "data", "median (joint)", "best fit (joint)"],
                   loc="center", bbox_to_anchor=(0.86, 0.56), fontsize=8, frameon=False)
        if a.title:
            fig.suptitle(a.title, y=0.995, fontsize=10)
        for ext in ("png", "pdf"):
            fig.savefig(out / f"corner_modes_figure.{ext}", dpi=200)
    print("written:", out / "corner_modes_figure.png")


if __name__ == "__main__":
    main()
