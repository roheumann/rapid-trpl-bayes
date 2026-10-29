"""
modes_corner.py — one HDF5 with the SMC posteriors of the tr-PL, ss-PL and joint runs of a sample, for Origin
==============================================================================================================
19/10/2026.  Input: results/<base>_{trpl,sspl,joint}/smc/smc.h5 (equal-weight SMC draws, theta coordinates).

    python scripts/modes_corner.py results/2trap_85-15_experimental_v3 [--draws 2000] [--grid 240] [--gpu 1]

One run only (22/10/2026, e.g. the raw-tr-PL joint run of the 85:15 film), filled, with two slice panels from
another run's slice file and both iso-tau_0 formulas over the whole panel:

    python scripts/modes_corner.py results/2trap_85-15_film/2trap_85-15_film --modes joint \
        --slice-file <run>/slices_joint_.../slices_joint_....h5 \
        --slice-panel log10_n1_1__log10_taup_1,log10_n1_2__log10_taup_2 --tau0-full

Writes results/<base>_joint/corner_modes/corner_modes.h5 (+ preview corner_modes_preview.png), read by
corner-plot/origin/build_corner_modes.py:

  /params/<key>            attrs label_origin, lo, hi (decades), [log10_Nc, kT_ln10, label_origin_et],
                           [truth (config true_params), best_fit (joint run's 6_best_fit.json theta_best)]
  /layout                  lower triangle (diagonal = 1D marginals), same dtype as corner_export
  /panels/<kx>__<ky>/      cat (G, G) uint8 = l_trpl + 3 l_sspl + 9 l_joint, l = 2 inside 1σ, 1 inside 2σ, 0 outside
                           (HPD regions of each mode), x, y (cell centres, decades), <mode>_sigma1 / _sigma2 (N, 2)
  /panels/<k>/             <mode>_x, <mode>_pdf (1D marginals)
  /ppc/                    cat (G, G) of the 68 / 95 % bands of the NN tau_diff curves (log10 s) of each mode,
                           x (QFLS, eV), y (log10 tau_diff), data_qfls, data_y, median_x, median_y (joint),
                           <mode>_q{2.5,16,50,84,97.5}
  /sspl_ppc/               the same for the ss-PL data (x = log10 illumination / G, or QFLS for plqy_vs_qfls; y = QFLS / log10 PLQY; attrs x_log10, y_log10),
                           from the full ss-PL model curves of the draws
  /slice, /slice_2, ...     (--slice-panel a,b: one group per panel, in that order) one panel of the paper slices of the --slice-mode run (default joint) (slices_paper/slices_paper.h5, --slice-panel):
                           hpd, x, y, sigma1, sigma2, tau0_photodoping / tau0_non_photodoping (iso-tau_0 lines in this
                           plane at the slice's n1, (m, 2) [x, y]); attrs kx, ky, hpd_levels, band_colors_json,
                           colour_bar_title, tau0_json; /slice/params/<key> attrs lo, hi, label_origin, truth, best_fit
  extras (write_extras)    /params/<key> attr median; /ppc, /sspl_ppc best_x, best_y (best-fit curves); trap-2 slice
                           tau0_(non_)photodoping_ntratio (--extras-only: only these, into an existing file)
  attrs                    param_order, modes, colors27_json (blended colour of every category), mode_colors_json

The category colours are the mode colours composited over white in the order ss-PL, tr-PL, joint
(2σ light, 1σ dark, alpha 0.60 / 0.85), so overlapping regions stay visible.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MODES = ("trpl", "sspl", "joint")
MODE_LABEL = {"trpl": "tr-PL", "sspl": "ss-PL", "joint": "joint"}
COLORS = {"trpl": {"s1": "#2166AC", "s2": "#92C5DE"},      # blue
          "sspl": {"s1": "#525252", "s2": "#BDBDBD"},      # grey
          "joint": {"s1": "#B2182B", "s2": "#F4A582"}}     # red
ALPHA = {1: 0.60, 2: 0.85}                                 # level 1 = 2σ region, 2 = 1σ region
SIGMA = (0.6827, 0.9545)
SMOOTH = 3.5                                               # Gaussian smoothing of the 2D histograms (bins)
FILL_MODE = "trpl"        # the one mode drawn as filled regions / bands; the others as 1σ solid / 2σ dashed lines


def hex2rgb(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], float)


def blended_colors():
    """Colour of category l_trpl + 3 l_sspl + 9 l_joint: white, then ss-PL, tr-PL, joint composited on top."""
    out = []
    for cat in range(27):
        lv = {"trpl": cat % 3, "sspl": (cat // 3) % 3, "joint": cat // 9}
        c = np.array([255.0, 255.0, 255.0])
        for m in ("sspl", "trpl", "joint"):
            if lv[m]:
                col = hex2rgb(COLORS[m]["s1" if lv[m] == 2 else "s2"])
                c = (1 - ALPHA[lv[m]]) * c + ALPHA[lv[m]] * col
        out.append("#%02x%02x%02x" % tuple(int(round(v)) for v in c))
    return out


def fill_colors(m=FILL_MODE):
    """Colours of the filled mode: outside, 2σ region, 1σ region (as in blended_colors, without the other modes)."""
    return [blended_colors()[3 ** MODES.index(m) * lv] for lv in (0, 1, 2)]


def interp2(Z, x, y, xf, yf):
    """Bilinear interpolation of Z (rows = y) from centres x, y onto xf, yf (held constant outside)."""
    Zx = np.array([np.interp(xf, x, row) for row in Z])
    return np.array([np.interp(yf, y, col) for col in Zx.T]).T


def levels(mass, s1=SIGMA[0], s2=SIGMA[1]):
    return np.where(mass <= s1, 2, np.where(mass <= s2, 1, 0)).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base", help="results/<base> without the _trpl/_sspl/_joint suffix")
    ap.add_argument("--smc", default="smc")
    ap.add_argument("--draws", type=int, default=2000, help="posterior draws per mode for the predictive bands")
    ap.add_argument("--grid", type=int, default=240, help="resolution of the filled regions")
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--slice-panel", default="log10_n1_1__log10_taup_1",
                    help="panel <kx>__<ky> of the tr-PL run's slices_paper.h5 to copy ('' or --no-slice: none)")
    ap.add_argument("--no-slice", action="store_true")
    ap.add_argument("--slice-mode", default="joint", choices=MODES, help="run whose slices_paper.h5 gives the slice panel")
    ap.add_argument("--slice-file", default=None, help="slice h5 to copy from instead of <base>_<slice-mode>/slices_paper")
    ap.add_argument("--tau0-full", action="store_true",
                    help="n1 - tau_p slices: both iso-tau_0 formulas over the whole n1 range of the panel (not only "
                         "inside their regimes), at the slice's N_t")
    ap.add_argument("--modes", default=",".join(MODES), help="comma-separated subset of trpl,sspl,joint (joint required)")
    ap.add_argument("--fill-mode", default=None, help=f"mode drawn filled (default {FILL_MODE}, or the only mode)")
    ap.add_argument("--best-fit-run", default=None,
                    help="run folder whose 6_best_fit.json / best-fit curves give the best fit (default: the joint run)")
    ap.add_argument("--extras-only", action="store_true",
                    help="only (re)write medians, best-fit curves and the N_t1/N_t2 photodoping line into the existing "
                         "corner_modes.h5 (write_extras), nothing else recomputed")
    a = ap.parse_args()
    modes = [m for m in MODES if m in a.modes.split(",")]
    assert "joint" in modes, "the joint run gives the best fit and the median"
    fm = a.fill_mode or (FILL_MODE if FILL_MODE in modes else modes[-1])
    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import h5py
    from inference import corner_export as ce, ppc, runs, surrogate
    from inference.prior import Prior
    surrogate.configure_tensorflow()

    base = Path(a.base)
    run = {m: Path(f"{base}_{m}") for m in modes}
    out = run["joint"] / "corner_modes"
    out.mkdir(parents=True, exist_ok=True)

    theta, priors, cfgs, arts = {}, {}, {}, {}
    for m in modes:
        cfg = json.loads((run[m] / "settings.json").read_text())["config"]
        art = surrogate.load_network(Path(cfg["nn"]["dir"]), cfg["nn"].get("stem"))
        prior = Prior.from_artifacts(art, Eg=cfg["physics"]["Eg_eV"], order_traps=cfg["prior"]["order_traps"],
                                     photodoping_traps=cfg["prior"]["photodoping_traps"],
                                     bounds_override=cfg["prior"]["bounds_override"], fixed=cfg["prior"].get("fixed", {}),
                                     T=cfg["physics"]["T_K"])
        with h5py.File(run[m] / a.smc / "smc.h5", "r") as f:
            names = json.loads(f["inference"].attrs["names"])
            assert names == list(prior.names), (m, names, prior.names)
            theta[m] = f["inference/theta_equal_weight"][()]
        cfgs[m], arts[m], priors[m] = cfg, art, prior
        print(f"{m}: {len(theta[m])} SMC draws")

    best_dir = Path(a.best_fit_run) if a.best_fit_run else run["joint"]
    best_theta = np.asarray(json.loads((best_dir / "6_best_fit.json").read_text())["theta_best"], float)
    truth_theta = None
    tp = cfgs["joint"].get("true_params")
    if tp:                                                   # synthetic data (as build_smc_notebook.py)
        truth_theta = np.atleast_2d(priors["joint"].from_physical(tp["Et_eV"], tp["Nt_cm3"], tp["tau_n_s"], tp["tau_p_s"],
                                                                  tp["krad_cm3s"]))[0]
    spec = ce.physical_corner_spec(priors["joint"], best_theta=best_theta, truth_theta=truth_theta)
    keys, perm, ranges, meta = spec["keys"], spec["perm"], spec["ranges"], spec["meta"]
    X = {m: spec["dmap"].to_x(theta[m])[:, perm] for m in modes}
    layout = ce.layout_triangle(keys, et_cols=[k for k in keys if k.startswith("log10_n1")])
    if a.extras_only:
        with h5py.File(out / "corner_modes.h5", "r+") as f:
            write_extras(f, best_dir, keys, X["joint"], meta)
        print("extras written:", out / "corner_modes.h5")
        return
    G = a.grid
    colors27 = blended_colors()

    with h5py.File(out / "corner_modes.h5", "w") as f:
        f.attrs.update({"param_order": json.dumps(keys), "modes": json.dumps(modes),
                        "mode_labels_json": json.dumps(MODE_LABEL), "mode_colors_json": json.dumps(COLORS),
                        "colors27_json": json.dumps(colors27), "sigma_levels": np.array(SIGMA),
                        "fill_mode": fm, "fill_colors_json": json.dumps(fill_colors(fm)),
                        "category": " + ".join(f"{3 ** i} l_{m}" for i, m in enumerate(modes))
                                    + "; l = 2 inside 1 sigma, 1 inside 2 sigma"})
        f["layout"] = layout
        gp = f.create_group("params")
        for k in keys:
            g = gp.create_group(k)
            g.attrs.update({"lo": ranges[k][0], "hi": ranges[k][1], "label_origin": meta[k]["label_origin"]})
            for e in ("truth", "best_fit"):
                if meta[k].get(e) is not None:
                    g.attrs[e] = float(meta[k][e])
            for e in ("log10_Nc", "kT_ln10", "label_origin_et"):
                if e in meta[k]:
                    g.attrs[e] = meta[k][e]
        pan = f.create_group("panels")
        col = {k: i for i, k in enumerate(keys)}
        for row in layout:
            kx, ky, kind = row["kx"].decode(), row["ky"].decode(), row["kind"].decode()
            if kind == "1d":
                g = pan.create_group(kx)
                for m in modes:
                    x, pdf = ce.density_1d(X[m][:, col[kx]], np.full(len(X[m]), 1 / len(X[m])), *ranges[kx])
                    g[f"{m}_x"], g[f"{m}_pdf"] = x, pdf
                continue
            g = pan.create_group(f"{kx}__{ky}")
            xf = np.linspace(*ranges[kx], G)
            yf = np.linspace(*ranges[ky], G)
            cat = np.zeros((G, G), np.uint8)
            for mi, m in enumerate(modes):
                w = np.full(len(X[m]), 1 / len(X[m]))
                x, y, dens = ce.density_2d(X[m][:, col[kx]], X[m][:, col[ky]], w, ranges[kx], ranges[ky], smooth=SMOOTH)
                lev = ce.hpd_level_map(dens)
                cat += (3 ** mi * levels(interp2(lev, x, y, xf, yf))).astype(np.uint8)
                if m == fm:                                       # continuous mass: smooth fills in Origin
                    g[f"{m}_mass"] = interp2(lev, x, y, xf, yf).astype(np.float32)
                g[f"{m}_sigma1"] = ce.contour_polyline(x, y, lev, SIGMA[0])
                g[f"{m}_sigma2"] = ce.contour_polyline(x, y, lev, SIGMA[1])
            g["cat"], g["x"], g["y"] = cat, xf, yf

        # posterior predictive bands of tau_diff(QFLS), NN surrogate, the same data for every mode
        bands, data = {}, None
        for m in modes:
            sig = runs.run_sigmas(run[m]).get("trpl") or 1.0          # the prediction does not depend on sigma
            log_post, _ = runs.make_log_post(cfgs[m], run[m], arts[m], priors[m], "trpl", sig, None)
            c = ppc.nn_curves(log_post, ppc.pick_draws(theta[m], a.draws, seed=20261019))
            bands[m] = np.quantile(c["f"], [0.025, 0.16, 0.5, 0.84, 0.975], axis=0)
            if data is None:
                data = (np.asarray(c["qfls"], float), np.asarray(c["y_obs"], float))
            print(f"{m}: predictive bands from {len(c['f'])} draws")
        qx, qy = data
        o = np.argsort(qx)
        qx, qy = qx[o], qy[o]
        bands = {m: b[:, o] for m, b in bands.items()}
        lo = min(np.nanmin(qy), min(b[0].min() for b in bands.values())) - 0.3
        hi = max(np.nanmax(qy), max(b[4].max() for b in bands.values())) + 0.3
        xf, yf = np.linspace(qx[0], qx[-1], G), np.linspace(lo, hi, G)
        cat = np.zeros((G, G), np.uint8)
        for mi, m in enumerate(modes):
            q = [np.interp(xf, qx, b) for b in bands[m]]
            Y = yf[:, None]
            lv = np.where((Y >= q[1]) & (Y <= q[3]), 2, np.where((Y >= q[0]) & (Y <= q[4]), 1, 0))
            cat += (3 ** mi * lv).astype(np.uint8)
        g = f.create_group("ppc")
        g["cat"], g["x"], g["y"] = cat, xf, yf
        g["data_qfls"], g["data_y"] = qx, qy
        g["median_x"], g["median_y"] = qx, bands["joint"][2]
        for m in modes:
            for q, b in zip(("2.5", "16", "50", "84", "97.5"), bands[m]):
                g[f"{m}_q{q}"] = b
        g.attrs.update({"x_label_origin": r"Fermi-level splitting \g(D)\i(E)\-(F) (eV)",
                        "y_label_origin": r"\i(\g(t))\-(diff) (s)", "y_coordinates": "log10 s"})
        sspl_ppc(f, modes, run, cfgs, arts, priors, theta, a.draws, G)
        if a.slice_panel and not a.no_slice:
            src = Path(a.slice_file) if a.slice_file else Path(f"{base}_{a.slice_mode}") / "slices_paper" / "slices_paper.h5"
            for i, panel in enumerate(p_ for p_ in a.slice_panel.split(",") if p_):
                name = "slice" if i == 0 else f"slice_{i + 1}"
                copy_slice(f, src, panel, name, a.tau0_full)
                if name in f:
                    f[name].attrs["mode"] = a.slice_mode
        write_extras(f, best_dir, keys, X["joint"], meta)
    print("written:", out / "corner_modes.h5")
    preview(out / "corner_modes.h5", out / "corner_modes_preview.png")


def _put(g, name, v):
    if name in g:
        del g[name]
    g[name] = v


def write_extras(f, run_dir, keys, Xj, meta=None):
    """23/10/2026.  /params/<k> attr median (joint posterior median); /ppc and /sspl_ppc best_x, best_y: the best-fit
    curves of the joint run (ODE tau_diff of 6_best_fit_ode_simulation.csv in log10 s; ss-PL curve of
    6_best_fit_sspl_curve.csv, log10 suns), cut to the panel's x range; in a trap-2 (log10 n1, log10 tau_p) slice
    tau0_photodoping_ntratio / tau0_non_photodoping_ntratio: both iso-tau_0 lines at N_t1 / N_t2 times the data's tau_0
    (best-fit N_t of the slice's run, tau0_json; each line shifted by about N_t1 / N_t2 in tau_p), attr tau0_ntratio.
    run_dir: the run giving the best fit (--best-fit-run); its best_fit attrs are written again from meta."""
    from inference import slices
    for i, k in enumerate(keys):
        f["params"][k].attrs["median"] = float(np.median(Xj[:, i]))
        if meta is not None and meta[k].get("best_fit") is not None:
            f["params"][k].attrs["best_fit"] = float(meta[k]["best_fit"])
    f.attrs["best_fit_source"] = str(run_dir / "6_best_fit.json")
    csv = lambda name: np.genfromtxt(run_dir / name, delimiter=",", names=True)
    if (run_dir / "6_best_fit_ode_simulation.csv").is_file():
        d = csv("6_best_fit_ode_simulation.csv")
        x, t = np.asarray(d["qfls_eV"], float), np.asarray(d["tau_diff_s"], float)
        xr = f["ppc/x"][()]
        ok = np.isfinite(x) & np.isfinite(t) & (t > 0) & (x >= xr[0]) & (x <= xr[-1])
        _put(f["ppc"], "best_x", x[ok])
        _put(f["ppc"], "best_y", np.log10(t[ok]))
        f["ppc"].attrs["best_source"] = "6_best_fit_ode_simulation.csv (ODE at the best fit)"
        print(f"ppc: best-fit ODE curve, {ok.sum()} points")
    if "sspl_ppc" in f and (run_dir / "6_best_fit_sspl_curve.csv").is_file():
        g = f["sspl_ppc"]
        if str(g.attrs.get("mode", "")) == "qfls_vs_suns":
            d = csv("6_best_fit_sspl_curve.csv")
            xs, y = np.asarray(d["x_suns"], float), np.asarray(d["y_qfls"], float)
            xr = g["x"][()]
            ok = np.isfinite(xs) & np.isfinite(y) & (xs > 0)
            x = np.log10(np.where(ok, xs, 1.0))
            ok &= (x >= xr[0]) & (x <= xr[-1])
            _put(g, "best_x", x[ok])
            _put(g, "best_y", y[ok])
            g.attrs["best_source"] = "6_best_fit_sspl_curve.csv"
            print(f"sspl_ppc: best-fit curve, {ok.sum()} points")
        else:
            print(f"sspl_ppc: mode {g.attrs.get('mode')}, no best-fit curve (only qfls_vs_suns)")
    for name in [n for n in f if n == "slice" or n.startswith("slice_")]:
        g = f[name]
        kx, ky = str(g.attrs["kx"]), str(g.attrs["ky"])
        if not (kx == "log10_n1_2" and ky == "log10_taup_2") or "tau0_json" not in g.attrs:
            continue
        t0 = {str(v.get("trap")): v for v in json.loads(g.attrs["tau0_json"]).values()}
        if "1" not in t0 or "2" not in t0:
            continue
        t, r = t0["2"], t0["1"]["Nt_cm3"] / t0["2"]["Nt_cm3"]
        lx = np.linspace(float(g["params"][kx].attrs["lo"]), float(g["params"][kx].attrs["hi"]), 1000)
        yr = (float(g["params"][ky].attrs["lo"]), float(g["params"][ky].attrs["hi"]))
        for reg in ("photodoping", "non_photodoping"):
            ly = np.log10(slices.taup_of_tau0(10.0 ** lx, r * t["tau0_s"], t["Nt_cm3"], t["krad_cm3s"], t["ni_cm3"],
                                              reg == "photodoping"))
            ly = np.where((ly >= yr[0]) & (ly <= yr[1]), ly, np.nan)
            _put(g, f"tau0_{reg}_ntratio", np.column_stack([lx, ly]))
            print(f"{name}: {reg} line at tau_0 x N_t1/N_t2 = {r:.3g}, {np.isfinite(ly).sum()} points in the panel")
        g.attrs["tau0_ntratio"] = r


def copy_slice(f, src, panel, name="slice", tau0_full=False):
    """One panel of the paper slice corner (inference/slices.py, scripts/paper_slices.py) into /slice with its
    iso-tau_0 lines: copied for an (n1, tau_p) panel (the slice file has them), computed for a (log10 tau_p, log10 N_t)
    panel: at the slice's fixed n1 (the best fit,
    tau0_json) tau_p(N_t) = tau_0 / (a - k_rad n1 tau_0), a = sqrt(n1 N_t)/n_i for N_t >= 10 n1 (photodoping) and
    a = n1/n_i for N_t <= 0.1 n1 (no photodoping: tau_p independent of N_t, a vertical line).
    tau0_full: in an (n1, tau_p) panel both lines are computed over the whole n1 range at the slice's N_t, also outside
    their regimes (attr tau0_full = 1)."""
    import h5py
    from inference import slices
    if not src.is_file():
        print(f"slice: {src} not found, no slice panel")
        return
    kx, ky = panel.split("__")
    with h5py.File(src, "r") as h:
        G = h["corner/physical"]
        if f"panels/{panel}" not in G:
            print(f"slice: no panel {panel} in {src}")
            return
        P = G[f"panels/{panel}"]
        g = f.create_group(name)
        for d in ("hpd", "x", "y", "sigma1", "sigma2", "tau0_photodoping", "tau0_non_photodoping"):
            if d in P:
                g[d] = P[d][()]
        for a_ in ("hpd_levels", "band_colors_json", "colour_bar_title", "colors_json", "tau0_json", "fill_quantity"):
            if a_ in G.attrs:
                g.attrs[a_] = G.attrs[a_]
        g.attrs.update({"kx": kx, "ky": ky, "source": str(src)})
        gp = g.create_group("params")
        for k in (kx, ky):
            q = gp.create_group(k)
            for a_ in ("lo", "hi", "label_origin", "truth", "best_fit"):
                if a_ in G[f"params/{k}"].attrs:
                    q.attrs[a_] = G[f"params/{k}"].attrs[a_]
        t0 = json.loads(G.attrs["tau0_json"]) if "tau0_json" in G.attrs else {}
    trap = kx.rsplit("_", 1)[-1]
    t = next((v for v in t0.values() if str(v.get("trap")) == trap), None)
    if ({kx.rsplit("_", 1)[0], ky.rsplit("_", 1)[0]} == {"log10_taun", "log10_taup"}
            and ky.rsplit("_", 1)[-1] == trap):
        # (tau_n, tau_p) panel of one trap (27/10/2026, Robin): the deep-trap line tau_n + tau_p = const through the
        # best fit, stored as tau0_taun_plus_taup so that render_corner_modes / build_corner_modes draw it like the
        # iso-tau_0 lines.  Parametrised by s = log10(x share / y share), so both asymptotes are sampled.
        bx, by = (float(g["params"][k].attrs["best_fit"]) for k in (kx, ky))
        c = np.log10(10.0 ** bx + 10.0 ** by)
        s = np.linspace(-9.0, 9.0, 2000)
        lx, ly = c - np.log10(1.0 + 10.0 ** -s), c - np.log10(1.0 + 10.0 ** s)
        inside = ((lx >= float(g["params"][kx].attrs["lo"])) & (lx <= float(g["params"][kx].attrs["hi"]))
                  & (ly >= float(g["params"][ky].attrs["lo"])) & (ly <= float(g["params"][ky].attrs["hi"])))
        g["tau0_taun_plus_taup"] = np.column_stack([lx, ly])[inside]
        g.attrs["taun_plus_taup_s"] = float(10.0 ** c)
        print(f"slice {name}: {panel} with the line tau_n + tau_p = {10.0 ** c:.3g} s (best fit), "
              f"{int(inside.sum())} points in the panel")
        return
    if tau0_full and t is not None and kx.startswith("log10_n1") and ky.startswith("log10_taup"):
        lx = np.linspace(float(g["params"][kx].attrs["lo"]), float(g["params"][kx].attrs["hi"]), 1000)
        yr = (float(g["params"][ky].attrs["lo"]), float(g["params"][ky].attrs["hi"]))
        for reg in ("photodoping", "non_photodoping"):
            ly = np.log10(slices.taup_of_tau0(10.0 ** lx, t["tau0_s"], t["Nt_cm3"], t["krad_cm3s"], t["ni_cm3"],
                                              reg == "photodoping"))
            ly = np.where((ly >= yr[0]) & (ly <= yr[1]), ly, np.nan)
            if f"tau0_{reg}" in g:
                del g[f"tau0_{reg}"]
            g[f"tau0_{reg}"] = np.column_stack([lx, ly])
        g.attrs["tau0_full"] = 1
        print(f"slice {name}: {panel}, both iso-tau_0 formulas over the whole panel (tau_0 {t['tau0_s']:.3g} s, "
              f"N_t {t['Nt_cm3']:.3g} cm^-3): photodoping {np.isfinite(g['tau0_photodoping'][:, 1]).sum()}, "
              f"no photodoping {np.isfinite(g['tau0_non_photodoping'][:, 1]).sum()} points in the panel")
        return
    if "tau0_photodoping" in g:                          # n1 - tau_p panel: the slice file has the iso-tau_0 lines
        print(f"slice {name}: {panel} with the slice file's iso-tau_0 lines")
        return
    if t is None or not (kx.startswith("log10_taup") and ky.startswith("log10_Nt")):
        print(f"slice: {panel} copied (no iso-tau_0 lines for this plane)")
        return
    xr = (float(g["params"][kx].attrs["lo"]), float(g["params"][kx].attrs["hi"]))
    yr = (float(g["params"][ky].attrs["lo"]), float(g["params"][ky].attrs["hi"]))
    n1, tau0, krad, ni = t["n1_cm3"], t["tau0_s"], t["krad_cm3s"], t["ni_cm3"]
    for reg, (a_, b_) in {"photodoping": (max(yr[0], np.log10(10 * n1)), yr[1]),
                          "non_photodoping": (yr[0], min(yr[1], np.log10(0.1 * n1)))}.items():
        line = np.zeros((0, 2))
        if b_ > a_:
            ly = np.linspace(a_, b_, 400)
            lx = np.broadcast_to(np.log10(slices.taup_of_tau0(n1, tau0, 10.0 ** ly, krad, ni, reg == "photodoping")),
                                 ly.shape)                   # no photodoping: one tau_p for all N_t
            ok = np.flatnonzero(np.isfinite(lx) & (lx >= xr[0]) & (lx <= xr[1]))
            if len(ok):
                line = np.column_stack([lx, ly])[ok[0]:ok[-1] + 1]
        g[f"tau0_{reg}"] = line
    print(f"slice: {panel} with iso-tau_0 lines (tau_0 {tau0:.3g} s, n1 {n1:.3g} cm^-3): "
          f"photodoping {len(g['tau0_photodoping'])} points, no photodoping {len(g['tau0_non_photodoping'])} points")


def sspl_ppc(f, modes, run, cfgs, arts, priors, theta, n_draws, G):
    """Posterior predictive bands of the ss-PL data for every mode: the model curve of each draw interpolated onto a
    common grid (log10 illumination / G; linear QFLS for 'plqy_vs_qfls'); 68 / 95 % bands, joint median, measured
    points.  y_obs is already in residual space (QFLS in eV or log10 PLQY), the model curves are PLQY as a fraction."""
    from inference import ppc, runs
    bands, data, xg = {}, None, None
    for m in modes:
        sig = runs.run_sigmas(run[m]).get("sspl") or 1.0
        lp, _ = runs.make_log_post(cfgs[m], run[m], arts[m], priors[m], "sspl", None, sig)
        th = ppc.pick_draws(theta[m], n_draws, seed=20261019)
        xs, ys = [], []
        for i in range(0, len(th), 500):
            p = lp.predict_sspl(th[i:i + 500])
            xs.append(np.asarray(p["curve"]["x"], float)); ys.append(np.asarray(p["curve"]["y"], float))
            if data is None:
                data = (np.asarray(p["x_exp"], float), np.asarray(p["y_obs"], float))
                mode_ss = lp.sspl_term.observation.mode
                xlog = mode_ss != "plqy_vs_qfls"
                fx = np.log10 if xlog else (lambda v: v)
        X, Y = np.concatenate(xs), np.concatenate(ys)
        if xg is None:
            pad = 0.3 if xlog else 0.02
            xg = np.linspace(fx(data[0].min()) - pad, fx(data[0].max()) + pad, G)
        cur = np.full((len(X), len(xg)), np.nan)
        for s_ in range(len(X)):
            ok = np.isfinite(X[s_]) & np.isfinite(Y[s_]) & ((X[s_] > 0) if xlog else True)
            if ok.sum() < 2:
                continue
            lx, yy = fx(X[s_][ok]), Y[s_][ok]
            o = np.argsort(lx)
            v = np.interp(xg, lx[o], yy[o], left=np.nan, right=np.nan)
            cur[s_] = v
        bands[m] = np.nanquantile(cur, [0.025, 0.16, 0.5, 0.84, 0.975], axis=0)
        print(f"{m}: ss-PL bands from {len(X)} draws")
    plqy = mode_ss.startswith("plqy")
    ydat = data[1]                                           # already log10 PLQY for the plqy modes
    bands = {m: (np.log10(b) if plqy else b) for m, b in bands.items()}
    lo = np.nanmin([np.nanmin(b[0]) for b in bands.values()] + [ydat.min()])
    hi = np.nanmax([np.nanmax(b[4]) for b in bands.values()] + [ydat.max()])
    pad = 0.05 * (hi - lo)
    yg = np.linspace(lo - pad, hi + pad, G)
    cat = np.zeros((G, G), np.uint8)
    for mi, m in enumerate(modes):
        q = [np.nan_to_num(b, nan=-1e9) for b in bands[m]]
        Y = yg[:, None]
        lv = np.where((Y >= q[1]) & (Y <= q[3]), 2, np.where((Y >= q[0]) & (Y <= q[4]), 1, 0))
        cat += (3 ** mi * lv).astype(np.uint8)
    g = f.create_group("sspl_ppc")
    g["cat"], g["x"], g["y"] = cat, xg, yg
    g["data_x"], g["data_y"] = fx(data[0]), ydat
    g["median_x"], g["median_y"] = xg, bands["joint"][2]
    for m in modes:
        for q, b in zip(("2.5", "16", "50", "84", "97.5"), bands[m]):
            g[f"{m}_q{q}"] = b
    x_lab = {"suns": "illumination (suns)", "G": r"generation rate (cm\+(-3) s\+(-1))"}
    g.attrs.update({"mode": mode_ss, "x_log10": int(xlog), "y_log10": int(plqy),
                    "x_label_origin": x_lab["suns" if mode_ss.endswith("_suns") else "G"] if xlog else r"\g(D)\i(E)\-(F) (eV)",
                    "y_label_origin": "PLQY" if plqy else r"\g(D)\i(E)\-(F) (eV)"})


def preview(h5, png):
    """Quick matplotlib check of the HDF5 content in the figure's style: FILL_MODE filled, the others as 1σ solid /
    2σ dashed lines, in the corner and in both predictive-check panels (not the Origin look)."""
    import h5py
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    with h5py.File(h5, "r") as f:
        keys = json.loads(f.attrs["param_order"])
        cols = json.loads(f.attrs["mode_colors_json"])
        labs = json.loads(f.attrs["mode_labels_json"])
        fm = f.attrs.get("fill_mode", FILL_MODE)
        fc = json.loads(f.attrs["fill_colors_json"]) if "fill_colors_json" in f.attrs else fill_colors(fm)
        modes = json.loads(f.attrs["modes"])
        cmap, mi = ListedColormap(fc), modes.index(fm)
        lines = [m for m in modes if m != fm]
        D = len(keys)
        fig, axs = plt.subplots(D, D, figsize=(2 * D, 2 * D))
        for r in range(D):
            for c in range(D):
                ax = axs[r, c]
                ax.set_xticks([]); ax.set_yticks([])
                if c > r:
                    ax.axis("off")
                    continue
                if r == c:
                    g = f["panels"][keys[c]]
                    for m in modes:
                        ax.plot(g[f"{m}_x"][()], g[f"{m}_pdf"][()], color=cols[m]["s1"])
                    continue
                g = f["panels"][f"{keys[c]}__{keys[r]}"]
                ax.pcolormesh(g["x"][()], g["y"][()], (g["cat"][()] // 3 ** mi) % 3, cmap=cmap, vmin=-0.5, vmax=2.5,
                              shading="nearest")
                for m in lines:
                    for s, ls, lw in (("sigma1", "-", 1.0), ("sigma2", "--", 0.7)):
                        p = g[f"{m}_{s}"][()]
                        if len(p):
                            ax.plot(p[:, 0], p[:, 1], color=cols[m]["s1"], ls=ls, lw=lw)
        for i, k in enumerate(keys):
            axs[D - 1, i].set_xlabel(f["params"][k].attrs["label_origin"].replace("\\", ""), fontsize=6)
            if i:
                axs[i, 0].set_ylabel(f["params"][k].attrs["label_origin"].replace("\\", ""), fontsize=6)
        handles = [Patch(color=fc[2], label=f"{labs[fm]} 68 %"), Patch(color=fc[1], label=f"{labs[fm]} 95 %")]
        for m in lines:
            handles += [Line2D([], [], color=cols[m]["s1"], label=f"{labs[m]} 68 %"),
                        Line2D([], [], color=cols[m]["s1"], ls="--", label=f"{labs[m]} 95 %")]
        handles += [Line2D([], [], color="k", marker="o", mfc="none", ls="", label="data"),
                    Line2D([], [], color="#E41A1C", label="median (joint)")]
        for box, grp in (([0.58, 0.66, 0.38, 0.28], "ppc"), ([0.68, 0.42, 0.28, 0.17], "sspl_ppc")):
            if grp not in f:
                continue
            ax = fig.add_axes(box)
            g = f[grp]
            lab = lambda a, log: ("log10 " if log else "") + g.attrs[a].replace("\\", "")
            xl, yl = ("QFLS (eV)", "log10 tau_diff (s)") if grp == "ppc" else \
                (lab("x_label_origin", g.attrs["x_log10"]), lab("y_label_origin", g.attrs.get("y_log10", 0)))
            ax.pcolormesh(g["x"][()], g["y"][()], (g["cat"][()] // 3 ** mi) % 3, cmap=cmap, vmin=-0.5, vmax=2.5,
                          shading="nearest")
            xb = g["data_qfls"][()] if grp == "ppc" else g["x"][()]
            for m in lines:
                for q, ls, lw in (("16", "-", 1.0), ("84", "-", 1.0), ("2.5", "--", 0.7), ("97.5", "--", 0.7)):
                    ax.plot(xb, g[f"{m}_q{q}"][()], color=cols[m]["s1"], ls=ls, lw=lw)
            ax.plot(g["median_x"][()], g["median_y"][()], color="#E41A1C", lw=1.3)
            xd = g["data_qfls"][()] if grp == "ppc" else g["data_x"][()]
            ax.plot(xd, g["data_y"][()], "o", mfc="none", mec="k", ms=2.5 if grp == "ppc" else 4, mew=0.6)
            ax.set_xlabel(xl); ax.set_ylabel(yl)
            ax.set_xlim(g["x"][0], g["x"][-1]); ax.set_ylim(g["y"][0], g["y"][-1])
            if grp == "ppc":
                ax.legend(handles=handles, fontsize=6, ncol=2, loc="lower left")
        fig.savefig(png, dpi=110)
    print("preview:", png)


if __name__ == "__main__":
    main()
