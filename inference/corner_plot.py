"""
corner_plot.py — matplotlib preview of a corner group written by corner_export.write_corner
==========================================================================================
Reads ONLY the HDF5 group (no samples), so the preview also checks that the file contains everything the
Origin script needs.  Style of corner-plot/corner-*.png: Inferno fill on fixed posterior-mass levels
(same in every panel, one colour bar), 1σ sky blue / 2σ light grey, best fit red dotted, truth cyan,
start point green, eigen-axes white; white tick marks on the 2D (off-diagonal) panels.

Comparison corners (corner_export.write_comparison_corner): the 1σ / 2σ contours and 1D marginals of every sampler
in the same colours, told apart by the dash pattern (NUTS solid, emcee dashed, SMC dotted); the fill posterior either
as the band map (background='bands') or as its samples coloured by the same posterior-mass levels on the colour of
the outermost band (background='scatter').
"""
from __future__ import annotations

import json

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator

from .corner_export import HPD_LEVELS, TAU0_DASH


def _s(v):
    return v.decode() if isinstance(v, bytes) else str(v)


def band_cmap():
    """Discrete Inferno: densest band (≤1 %) pale yellow ... outside 99 % black."""
    edges = np.concatenate([[0.0], HPD_LEVELS, [1.0 + 1e-9]])
    cols = mpl.colormaps["inferno"](np.linspace(0.98, 0.0, len(edges) - 1))
    return ListedColormap(cols), BoundaryNorm(edges, len(edges) - 1), edges


def _log_fmt(scale):
    if scale == "log10":
        return FuncFormatter(lambda v, _: rf"$10^{{{v:.0f}}}$" if abs(v - round(v)) < 1e-6 else "")
    return None


MPL_DASH = {"solid": "-", "dash": "--", "dot": ":", "dashdot": "-."}


def _param(group, key):
    a = dict(group["params"][key].attrs)
    return {k: (_s(v) if isinstance(v, (bytes, np.bytes_)) else v) for k, v in a.items()}


def plot_corner(group, panel_size=1.55, show_eigen=True, title=None, background="auto", point_size=0.6, curves=None,
                cbar_label="posterior mass of higher density"):
    """background: 'bands' (posterior-mass map), 'scatter' (samples coloured by the mass level; needs
    panels/*/scatter) or 'auto' (scatter if present).  curves: the /curves group of a slice file (layout rows of
    kind 'curve': tau(QFLS), PL(t), ss-PL; 'span:<rows>x<cols>' in the extra column)."""
    lay = group["layout"][()]
    nrow, ncol = int(lay["row"].max()) + 1, int(lay["col"].max()) + 1
    colors = json.loads(group.attrs["colors_json"])
    diag_mode = _s(group.attrs["diag_mode"])
    keys = json.loads(group.attrs["param_order"])
    P = {k: _param(group, k) for k in keys}
    cmap, norm, edges = band_cmap()
    methods = json.loads(group.attrs["methods"]) if "methods" in group.attrs else []
    dash = json.loads(group.attrs["method_dash_json"]) if methods else {}
    has_scatter = any("scatter" in g for n, g in group["panels"].items() if "__" in n)
    mode_bg = ("scatter" if has_scatter else "bands") if background == "auto" else background
    fig, axs = plt.subplots(nrow, ncol, figsize=(panel_size * ncol + 1.3, panel_size * nrow + 0.4), squeeze=False)
    for ax in axs.ravel():
        ax.set_visible(False)
    fig.subplots_adjust(left=0.09, right=0.86, bottom=0.08, top=0.88, wspace=0.12, hspace=0.12)   # room for E_t - E_g
    eig = group.get("overlays")
    tau0_drawn = set()
    for row in lay:
        r, c = int(row["row"]), int(row["col"])
        kind, kx, ky = _s(row["kind"]), _s(row["kx"]), _s(row["ky"])
        xside, yside, extra = _s(row["xside"]), _s(row["yside"]), _s(row["extra"])
        if kind == "curve":
            if curves is not None:
                rs, cs = (int(v) for v in (extra.split("span:")[1] if "span:" in extra else "1x1").split("x"))
                b0, b1 = axs[r, c].get_position(), axs[r + rs - 1, c + cs - 1].get_position()
                padx = 0.32 * (b1.x1 - b1.x0)                                  # room for the y labels on the left
                pady = 0.24 * (b1.y1 - b1.y0)                                  # and for the x labels below
                cax_ = fig.add_axes([b0.x0 + padx, b1.y0 + pady, b1.x1 - b0.x0 - padx, b0.y1 - b1.y0 - pady])
                draw_curve_panel(cax_, curves, kx, legend=kx == "tau_qfls", small=True)   # one legend: the tau panel
            continue
        if kind == "empty":
            continue
        ax = axs[r, c]
        ax.set_visible(True)
        px = P[kx]
        ax.set_xlim(px["lo"], px["hi"])
        ax.tick_params(direction="in", top=True, right=True, labelsize=6.5, length=2.5)
        if kind == "2d":
            ax.tick_params(which="both", color="w")                  # white ticks on the posterior map
            py = P[ky]
            g = group["panels"][f"{kx}__{ky}"]
            if mode_bg == "scatter" and "scatter" in g:
                sc = g["scatter"][()]
                ax.set_facecolor(cmap(norm(1.0)))                       # colour of 'outside 99 %'
                ax.scatter(sc[:, 0], sc[:, 1], c=sc[:, 2], cmap=cmap, norm=norm, s=point_size, lw=0,
                           rasterized=True)
            else:
                ax.pcolormesh(g["x"][()], g["y"][()], g["hpd"][()], cmap=cmap, norm=norm, shading="nearest",
                              rasterized=True)
            if methods:                                                   # every sampler: same colours, own dashes
                for m in methods:
                    gm = group["methods"][m][f"{kx}__{ky}"]
                    for s in ("sigma2", "sigma1"):
                        d = gm[s][()]
                        if len(d):
                            ax.plot(d[:, 0], d[:, 1], color=colors[s], lw=0.9, ls=MPL_DASH[dash[m]])
            else:
                for s, lw in (("sigma2", 0.9), ("sigma1", 0.9)):
                    d = g[s][()]
                    if len(d):
                        ax.plot(d[:, 0], d[:, 1], color=colors[s], lw=lw)
            for reg, dsh in TAU0_DASH.items():                             # iso-tau_0 lines (slice corner)
                d = g[f"tau0_{reg}"][()] if f"tau0_{reg}" in g else np.zeros((0, 2))
                if len(d):
                    ax.plot(d[:, 0], d[:, 1], color=colors.get("tau0", "#7CFC00"), lw=1.0, ls=MPL_DASH[dsh])
                    tau0_drawn.add(reg)
            ax.set_ylim(py["lo"], py["hi"])
            for name, ls, lw in (("truth", "-", 0.8), ("best_fit", ":", 1.0)):
                vx, vy = px.get(name, np.nan), py.get(name, np.nan)
                if np.isfinite(vx):
                    ax.axvline(vx, color=colors[name], ls=ls, lw=lw)
                if np.isfinite(vy):
                    ax.axhline(vy, color=colors[name], ls=ls, lw=lw)
            if np.isfinite(px.get("start", np.nan)) and np.isfinite(py.get("start", np.nan)):
                ax.plot(px["start"], py["start"], "o", ms=3, color=colors["start"])
            if show_eigen and eig is not None:
                L, M = eig["eigen_lines"][()], eig["eigen_marks"][()]
                ix, iy = int(px["index"]), int(py["index"])
                for k in range(L.shape[0]):
                    ax.plot(L[k, :, ix], L[k, :, iy], color=colors["eigen_axes"], lw=0.5, alpha=0.8)
                    ax.plot(M[k, [1, 2], ix], M[k, [1, 2], iy], "o", ms=2, color=colors["eigen_axes"])
                    ax.plot(M[k, [0, 3], ix], M[k, [0, 3], iy], "o", ms=2.5, mfc="none", mew=0.6,
                            color=colors["eigen_axes"])
            ax.yaxis.set_major_formatter(_log_fmt(py["scale"]) or ax.yaxis.get_major_formatter())
            if py["scale"] == "log10":
                ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=3))
        else:
            if diag_mode == "slice" and "diag" in group and kx in group["diag"]:
                g = group["diag"][kx]
                z = g["z"][()]
                ax.plot(z, g["parabola"][()], color="0.55", lw=1.0, ls="-.", label="Laplace")
                if "dlogL_ode" in g:
                    ax.plot(z, g["dlogL_ode"][()], color="k", lw=1.0, ls="--", label="ODE")
                ax.plot(z, g["dlogL_nn"][()], color="k", lw=1.2, label="NN")
                for m in (-2, -1, 1, 2):
                    ax.axvline(m, color=colors["sigma1" if abs(m) == 1 else "sigma2"], lw=0.7)
                ax.set_ylim(-8, 1)
                ax.set_ylabel(r"$\Delta\ln L$", fontsize=7)
                if r == 0:
                    ax.legend(fontsize=5, frameon=False, loc="lower center")
            else:
                g = group["panels"][kx]
                x, pdf, band = g["x"][()], g["pdf"][()], g["hpd_band"][()]
                ax.fill_between(x, 0, pdf, where=band >= 1, color=colors["sigma2"], step="mid", lw=0)
                ax.fill_between(x, 0, pdf, where=band >= 2, color=colors["sigma1"], step="mid", lw=0)
                ymax = np.nanmax(pdf)
                if methods:
                    for m in methods:
                        gm = group["methods"][m][kx]
                        ax.plot(gm["x"][()], gm["pdf"][()], color="k", lw=1.0, ls=MPL_DASH[dash[m]])
                        ymax = max(ymax, np.nanmax(gm["pdf"][()]))
                else:
                    ax.plot(x, pdf, color="k", lw=1.0)
                ax.set_ylim(0, 1.15 * ymax if ymax > 0 else 1)
                for name, ls in (("truth", "-"), ("best_fit", ":")):
                    if np.isfinite(px.get(name, np.nan)):
                        ax.axvline(px[name], color=colors[name], ls=ls, lw=1.0)
            if yside == "right":
                ax.yaxis.tick_right(); ax.yaxis.set_label_position("right")
                ax.tick_params(left=True, labelleft=False, labelright=True)
        # axis labels / sides
        ax.xaxis.set_major_formatter(_log_fmt(px["scale"]) or ax.xaxis.get_major_formatter())
        if px["scale"] == "log10":
            ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=3))
        if xside == "bottom":
            ax.set_xlabel(px["label_mpl"], fontsize=7.5)
        elif xside == "top":
            ax.xaxis.set_label_position("top"); ax.tick_params(labeltop=True, labelbottom=False)
            ax.set_xlabel(px["label_mpl"], fontsize=7.5)
        else:
            ax.tick_params(labelbottom=False)
        if kind == "2d":
            if yside == "left":
                ax.set_ylabel(P[ky]["label_mpl"], fontsize=7.5)
            elif yside == "right":
                ax.yaxis.set_label_position("right"); ax.tick_params(labelright=True, labelleft=False)
                ax.set_ylabel(P[ky]["label_mpl"], fontsize=7.5)
            else:
                ax.tick_params(labelleft=False)
        if extra.endswith("Et-Eg") and "kT_ln10" in px:
            # E_t - E_g = (log10 n1 - log10 N_c) kT ln10 : linear in the log10-n1 axis
            where = extra.split(":")[0]
            if where == "top":
                loc = 1.45 if xside == "top" else 1.0          # offset above existing top labels
            else:
                loc = -0.55 if xside == "bottom" else 0.0      # offset below existing bottom labels
            f = lambda v, p=px: (v - p["log10_Nc"]) * p["kT_ln10"]
            fi = lambda e, p=px: e / p["kT_ln10"] + p["log10_Nc"]
            sec = ax.secondary_xaxis(loc, functions=(f, fi))
            sec.tick_params(direction="in", labelsize=6.5, length=2.5)
            sec.set_xlabel(px["label_mpl_et"], fontsize=7.5)
    cax = fig.add_axes([0.9, 0.25, 0.018, 0.5])
    cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, boundaries=edges, ticks=edges[1:-1])
    cb.ax.set_yticklabels([f"{100 * v:g} %" for v in edges[1:-1]], fontsize=6)
    cb.ax.invert_yaxis()
    cb.set_label(cbar_label, fontsize=7)
    if tau0_drawn:                                                        # iso-tau_0 lines: legend below the colour bar
        t0 = json.loads(group.attrs["tau0_json"]) if "tau0_json" in group.attrs else {}
        vals = sorted({f"{v['tau0_s']:.2e}" for v in t0.values()})
        h = [Line2D([], [], color=colors.get("tau0", "#7CFC00"), lw=1.0, ls=MPL_DASH[TAU0_DASH[reg]], label=lab)
             for reg, lab in (("photodoping", r"$N_t \geq 10\,n_1$"), ("non_photodoping", r"$N_t \leq 0.1\,n_1$"))
             if reg in tau0_drawn]
        fig.legend(handles=h, loc="lower right", bbox_to_anchor=(0.995, 0.02), fontsize=6, frameon=False,
                   title=r"iso-$\tau_0$" + (f"\n$\\tau_0$ = {', '.join(vals)} s" if vals else ""), title_fontsize=6)
    if methods:                                                           # legend in the empty upper right
        h = []
        for m in methods:
            a = dict(group["methods"][m].attrs)
            lab = _s(a.get("label", m))
            if m == _s(group.attrs["fill_method"]):
                lab += " — colour map" + (" and samples" if mode_bg == "scatter" else "")
            h.append(Line2D([], [], color="k", ls=MPL_DASH[dash[m]], lw=1.0, label=lab))
        h += [Line2D([], [], color=colors["sigma1"], lw=1.4, label="1σ (68.3 %) HPD"),
              Line2D([], [], color=colors["sigma2"], lw=1.4, label="2σ (95.4 %) HPD")]
        P0 = P[keys[0]]
        if np.isfinite(P0.get("truth", np.nan)):
            h.append(Line2D([], [], color=colors["truth"], lw=1.0, label="truth"))
        if np.isfinite(P0.get("best_fit", np.nan)):
            h.append(Line2D([], [], color=colors["best_fit"], lw=1.0, ls=":", label="best fit"))
        if np.isfinite(P0.get("start", np.nan)):
            h.append(Line2D([], [], color=colors["start"], marker="o", ls="", ms=3, label="start point"))
        fig.legend(handles=h, loc="upper right", bbox_to_anchor=(0.86, 0.86), fontsize=6.5, frameon=False)
    if title:
        fig.suptitle(title, fontsize=9, y=0.995)
    return fig


CURVE_STYLE = {"best_fit": ("#E41A1C", "best fit"), "truth": ("#00CED1", "true parameters")}


def draw_curve_panel(ax, curves, name, legend=True, small=False):
    """
    One simulation panel of a slice file (/curves): 'tau_qfls' (tau_diff vs QFLS), 'pl_t' (PL vs t) or 'sspl'.
    Data: black circles; best fit red, true parameters cyan; NN solid, ODE dashed (labels with eps_rms).
    """
    fs = 6 if small else 8
    d = curves["data"]
    sets = [k for k in ("best_fit", "truth") if k in curves]
    if name == "tau_qfls":
        ax.semilogy(d["qfls"][()], d["tau_diff"][()], "o", mfc="none", mec="k", ms=2.5 if small else 3.5, mew=0.6,
                    label="data")
        q_lo, q_hi = d["qfls"][()].min(), d["qfls"][()].max()
        for k in sets:
            g, (col, lab) = curves[k], CURVE_STYLE[k]
            ax.semilogy(g["nn/qfls"][()], g["nn/tau_diff"][()], "-", color=col, lw=1.2,
                        label=rf"{lab} NN, $\varepsilon_\mathrm{{rms}}$ = {g.attrs['eps_rms_nn']:.1e}")
            ax.semilogy(g["ode/qfls_eV"][()], g["ode/tau_diff_s"][()], "--", color=col, lw=1.0,
                        label=rf"{lab} ODE, $\varepsilon_\mathrm{{rms}}$ = {g.attrs['eps_rms_ode']:.1e}")
        ax.set_xlim(q_lo - 0.03, q_hi + 0.03)
        ys = d["tau_diff"][()]
        ax.set_ylim(ys.min() / 300, ys.max() * 5)                        # room for the legend (lower left)
        ax.set_xlabel(r"Fermi-level splitting $\Delta E_\mathrm{F}$ (eV)", fontsize=fs + 1)
        ax.set_ylabel(r"$\tau_\mathrm{diff}$ (s)", fontsize=fs + 1)
    elif name == "pl_t":
        ax.loglog(d["t"][()], d["pl"][()], "o", mfc="none", mec="k", ms=2.5 if small else 3.5, mew=0.6, label="data")
        for k in sets:
            g, (col, lab) = curves[k], CURVE_STYLE[k]
            ax.loglog(g["nn/t"][()], g["nn/pl"][()], "-", color=col, lw=1.2, label=f"{lab} NN")
            ax.loglog(g["ode/time_s"][()], g["ode/pl_norm"][()], "--", color=col, lw=1.0, label=f"{lab} ODE")
        t = d["t"][()]
        ax.set_xlim(max(1e-11, t.min() / 3), t.max() * 3)
        ax.set_ylim(max(1e-14, d["pl"][()].min() / 10), 2)
        ax.set_xlabel("time (s)", fontsize=fs + 1)
        ax.set_ylabel(r"PL / PL$_0$", fontsize=fs + 1)
    elif name == "sspl":
        mode = curves.attrs.get("sspl_mode", "plqy_vs_qfls")
        mode = mode.decode() if isinstance(mode, bytes) else str(mode)
        g0 = curves[sets[0]]["sspl"]
        ax.plot(g0["x_exp"][()], g0["y_obs"][()], "o", mfc="none", mec="k", ms=2.5 if small else 3.5, mew=0.6,
                label="ss-PL data")
        for k in sets:
            g, (col, lab) = curves[k]["sspl"], CURVE_STYLE[k]
            cx, cy = g["curve_x"][()], g["curve_y"][()]
            if mode.startswith("plqy"):
                cy = np.log10(cy)
            ax.plot(cx, cy, "-", color=col, lw=1.2, label=f"{lab} (ss-PL model)")
        x = g0["x_exp"][()]
        ax.set_xlim(x.min() - 0.03 * np.ptp(x), x.max() + 0.03 * np.ptp(x))
        y = g0["y_obs"][()]
        ax.set_ylim(y.min() - 0.3 * np.ptp(y), y.max() + 0.3 * np.ptp(y))
        ax.set_xlabel(mode.split("_vs_")[1].replace("qfls", r"$\Delta E_\mathrm{F}$ (eV)"), fontsize=fs + 1)
        ax.set_ylabel(r"$\log_{10}$ PLQY" if mode.startswith("plqy") else "QFLS (eV)", fontsize=fs + 1)
    ax.tick_params(direction="in", top=True, right=True, which="both", labelsize=fs)
    if legend:
        ax.legend(fontsize=fs - 0.5, frameon=False, loc={"tau_qfls": "lower left", "pl_t": "lower left"}.get(name, "best"))
    return ax


def plot_loadings(group, title=None):
    """Heat map of the eigenvectors (rows v_k with sigma_k) in the physical decade coordinates."""
    V = group["V"][()]
    sig = group["sigma"][()]
    cols = json.loads(group.attrs["columns_mpl"])
    fig, ax = plt.subplots(figsize=(1.0 + 0.62 * V.shape[0], 0.8 + 0.42 * V.shape[0]))
    im = ax.imshow(V, cmap="RdBu_r", vmin=-1, vmax=1)          # rows = parameters, columns = v_k
    for i in range(V.shape[0]):
        for j in range(V.shape[1]):
            ax.text(j, i, f"{V[i, j]:+.2f}", ha="center", va="center", fontsize=6,
                    color="w" if abs(V[i, j]) > 0.6 else "k")
    ax.set_xticks(range(V.shape[1]), [f"$v_{{{k + 1}}}$\n$\\sigma$={s:.2g}" for k, s in enumerate(sig)], fontsize=6.5)
    ax.set_yticks(range(V.shape[0]), cols, fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.8, label="loading")
    if title:
        ax.set_title(title, fontsize=8)
    fig.tight_layout()
    return fig
