"""
nn_vs_ode_corner.py — NN SMC and ODE SMC posteriors of one run in one corner plot, for Origin
==============================================================================================
21/10/2026.  Input: <run>/ode_vs_nn/{nn,ode}_smc_weighted_samples.csv (ode_inference.ipynb: both samplers with the
same SMC settings), truth from the run config (true_params), best fit from 6_best_fit.json.

    python scripts/nn_vs_ode_corner.py results/1trap_synthetic/1trap_synthetic_trpl [--nn-source main]

--nn-source main takes the NN posterior of the main SMC run (<run>/smc/smc.h5) instead of the NN SMC of ode_vs_nn.

Writes <run>/ode_vs_nn/corner_nn_vs_ode/corner_nn_vs_ode.h5 (+ corner_nn_vs_ode_preview.png), read by
corner-plot/origin/build_corner_nn_vs_ode.py.  Same layout as corner_modes.h5 (scripts/modes_corner.py), without
the predictive-check groups:

  /params/<key>            attrs label_origin, lo, hi (decades), truth, best_fit, [log10_Nc, kT_ln10, label_origin_et]
  /layout                  lower triangle (diagonal = 1D marginals), same dtype as corner_export
  /panels/<kx>__<ky>/      nn_mass (G, G): HPD mass level of the NN posterior (filled in Origin), x, y (cell centres),
                           <method>_sigma1 / _sigma2 (N, 2) contour polylines
  /panels/<k>/             <method>_x, <method>_pdf (1D marginals, density per decade)
  attrs                    param_order, modes, mode_labels_json, mode_colors_json, fill_mode, fill_colors_json,
                           sigma_levels

NN (blue) is filled 1σ dark / 2σ light, ODE (wine red) drawn as 1σ solid / 2σ dashed lines.
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
sys.path.insert(0, str(ROOT / "scripts"))

from modes_corner import ALPHA, SIGMA, SMOOTH, hex2rgb, interp2   # noqa: E402

METHODS = ("nn", "ode")
METHOD_LABEL = {"nn": "NN SMC", "ode": "ODE SMC"}
COLORS = {"nn": {"s1": "#2166AC", "s2": "#92C5DE"},       # blue, as tr-PL in the modes corner
          "ode": {"s1": "#7B1E3A", "s2": "#C98A9B"}}      # wine red
FILL_MODE = "nn"


def fill_colors(colors=COLORS, fill_mode=FILL_MODE):
    """Outside, 2σ region, 1σ region of the filled method: its colours composited over white (as modes_corner)."""
    out = []
    for lv in (0, 1, 2):
        c = np.array([255.0, 255.0, 255.0])
        if lv:
            c = (1 - ALPHA[lv]) * c + ALPHA[lv] * hex2rgb(colors[fill_mode]["s1" if lv == 2 else "s2"])
        out.append("#%02x%02x%02x" % tuple(int(round(v)) for v in c))
    return out


def weighted_csv(path, names):
    import pandas as pd
    df = pd.read_csv(path)
    w = df["weight"].to_numpy(float)
    return df[names].to_numpy(float), w / w.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="run folder with ode_vs_nn/ (e.g. .../1trap_synthetic_trpl)")
    ap.add_argument("--sub", default="ode_vs_nn", help="subfolder of the NN vs ODE comparison")
    ap.add_argument("--nn-source", choices=("ode_vs_nn", "main"), default="ode_vs_nn")
    ap.add_argument("--grid", type=int, default=240, help="resolution of the filled regions")
    ap.add_argument("--gpu", default="1")
    a = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import h5py

    run = Path(a.run)
    cmp_dir = run / a.sub
    out = cmp_dir / "corner_nn_vs_ode"
    cfg, prior = run_prior(run)
    names = list(prior.names)

    theta, weights = {}, {}
    theta["ode"], weights["ode"] = weighted_csv(cmp_dir / "ode_smc_weighted_samples.csv", names)
    if a.nn_source == "main":
        with h5py.File(run / "smc" / "smc.h5", "r") as f:
            assert json.loads(f["inference"].attrs["names"]) == names
            theta["nn"] = f["inference/theta_equal_weight"][()]
        weights["nn"] = np.full(len(theta["nn"]), 1 / len(theta["nn"]))
    else:
        theta["nn"], weights["nn"] = weighted_csv(cmp_dir / "nn_smc_weighted_samples.csv", names)
    for m in METHODS:
        print(f"{METHOD_LABEL[m]}: {len(theta[m])} particles, Kish ESS {1 / np.sum(weights[m] ** 2):.0f}")
    write_corner(run, cfg, prior, theta, weights, out / "corner_nn_vs_ode.h5", a.grid, extra_attrs={"nn_source": a.nn_source})


def run_prior(run):
    """Config snapshot and prior of a main.ipynb run folder (network only for the prior bounds)."""
    from inference import surrogate
    from inference.prior import Prior
    cfg = json.loads((run / "settings.json").read_text())["config"]
    art = surrogate.load_network(Path(cfg["nn"]["dir"]), cfg["nn"].get("stem"))
    prior = Prior.from_artifacts(art, Eg=cfg["physics"]["Eg_eV"], order_traps=cfg["prior"]["order_traps"],
                                 photodoping_traps=cfg["prior"]["photodoping_traps"],
                                 bounds_override=cfg["prior"]["bounds_override"], fixed=cfg["prior"].get("fixed", {}),
                                 T=cfg["physics"]["T_K"])
    return cfg, prior


def write_corner(run, cfg, prior, theta, weights, h5, grid=240, methods=METHODS, labels=METHOD_LABEL, colors=COLORS,
                 fill_mode=FILL_MODE, extra_attrs=None):
    """Corner HDF5 of two posteriors (theta/weights: dict method -> (N, D) samples in prior coordinates, weights)
    read by corner-plot/origin/build_corner_nn_vs_ode.py; fill_mode is filled, the other drawn as lines."""
    import h5py
    from inference import corner_export as ce
    h5 = Path(h5)
    h5.parent.mkdir(parents=True, exist_ok=True)
    best = np.array(json.loads((run / "6_best_fit.json").read_text())["theta_best"], float)
    tp = cfg.get("true_params")
    truth = None if not tp else \
        np.atleast_2d(prior.from_physical(tp["Et_eV"], tp["Nt_cm3"], tp["tau_n_s"], tp["tau_p_s"], tp["krad_cm3s"]))[0]

    spec = ce.physical_corner_spec(prior, best_theta=best, truth_theta=truth)
    keys, perm, ranges, meta = spec["keys"], spec["perm"], spec["ranges"], spec["meta"]
    X = {m: spec["dmap"].to_x(theta[m])[:, perm] for m in methods}
    layout = ce.layout_triangle(keys, et_cols=[k for k in keys if k.startswith("log10_n1")])
    G = grid
    col = {k: i for i, k in enumerate(keys)}

    with h5py.File(h5, "w") as f:
        f.attrs.update({"param_order": json.dumps(keys), "modes": json.dumps(list(methods)),
                        "mode_labels_json": json.dumps(labels), "mode_colors_json": json.dumps(colors),
                        "sigma_levels": np.array(SIGMA), "fill_mode": fill_mode,
                        "fill_colors_json": json.dumps(fill_colors(colors, fill_mode)), "run": str(run),
                        **(extra_attrs or {})})
        f["layout"] = layout
        gp = f.create_group("params")
        for k in keys:
            g = gp.create_group(k)
            g.attrs.update({"lo": ranges[k][0], "hi": ranges[k][1], "label_origin": meta[k]["label_origin"]})
            for e in ("log10_Nc", "kT_ln10", "label_origin_et", "truth", "best_fit"):
                if meta[k].get(e) is not None:
                    g.attrs[e] = meta[k][e]
        pan = f.create_group("panels")
        for row in layout:
            kx, ky, kind = row["kx"].decode(), row["ky"].decode(), row["kind"].decode()
            if kind == "1d":
                g = pan.create_group(kx)
                for m in methods:
                    x, pdf = ce.density_1d(X[m][:, col[kx]], weights[m], *ranges[kx])
                    g[f"{m}_x"], g[f"{m}_pdf"] = x, pdf
                continue
            g = pan.create_group(f"{kx}__{ky}")
            xf, yf = np.linspace(*ranges[kx], G), np.linspace(*ranges[ky], G)
            for m in methods:
                x, y, dens = ce.density_2d(X[m][:, col[kx]], X[m][:, col[ky]], weights[m], ranges[kx], ranges[ky],
                                           smooth=SMOOTH)
                lev = ce.hpd_level_map(dens)
                if m == fill_mode:
                    g[f"{m}_mass"] = interp2(lev, x, y, xf, yf).astype(np.float32)
                g[f"{m}_sigma1"] = ce.contour_polyline(x, y, lev, SIGMA[0])
                g[f"{m}_sigma2"] = ce.contour_polyline(x, y, lev, SIGMA[1])
            g["x"], g["y"] = xf, yf
    print("written:", h5)
    preview(h5, h5.with_name(h5.stem + "_preview.png"))


def preview(h5, png):
    """Quick matplotlib check of the HDF5 content (not the Origin look)."""
    import h5py
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    with h5py.File(h5, "r") as f:
        keys = json.loads(f.attrs["param_order"])
        cols, labs = json.loads(f.attrs["mode_colors_json"]), json.loads(f.attrs["mode_labels_json"])
        fc, s1, s2 = json.loads(f.attrs["fill_colors_json"]), *f.attrs["sigma_levels"]
        modes, fm = json.loads(f.attrs["modes"]), f.attrs["fill_mode"]
        fm = fm.decode() if isinstance(fm, bytes) else fm
        lm = next(m for m in modes if m != fm)                  # the method drawn as lines
        P = {k: dict(f["params"][k].attrs) for k in keys}
        D = len(keys)
        fig, axs = plt.subplots(D, D, figsize=(2.2 * D, 2.2 * D))
        for r in range(D):
            for c in range(D):
                ax = axs[r, c]
                if c > r:
                    ax.axis("off")
                    continue
                kx = keys[c]
                ax.set_xlim(P[kx]["lo"], P[kx]["hi"])
                if "truth" in P[kx]:
                    ax.axvline(P[kx]["truth"], color="k", ls="--", lw=0.8)
                if r == c:
                    g = f["panels"][kx]
                    for m in modes:
                        ax.plot(g[f"{m}_x"][()], g[f"{m}_pdf"][()], color=cols[m]["s1"])
                    ax.set_yticks([])
                    continue
                ky = keys[r]
                g = f["panels"][f"{kx}__{ky}"]
                lv = np.where(g[f"{fm}_mass"][()] <= s1, 2, np.where(g[f"{fm}_mass"][()] <= s2, 1, 0))
                ax.pcolormesh(g["x"][()], g["y"][()], lv, cmap=ListedColormap(fc), vmin=-0.5, vmax=2.5, shading="nearest")
                for s, ls in (("sigma1", "-"), ("sigma2", "--")):
                    p = g[f"{lm}_{s}"][()]
                    if len(p):
                        ax.plot(p[:, 0], p[:, 1], color=cols[lm]["s1"], ls=ls, lw=1.0)
                if "truth" in P[ky]:
                    ax.axhline(P[ky]["truth"], color="k", ls="--", lw=0.8)
                if "best_fit" in P[kx] and "best_fit" in P[ky]:
                    ax.plot(P[kx]["best_fit"], P[ky]["best_fit"], "*", mfc="none", mec="#E41A1C", ms=8)
                ax.set_ylim(P[ky]["lo"], P[ky]["hi"])
        for i, k in enumerate(keys):
            axs[D - 1, i].set_xlabel(P[k]["label_origin"].replace("\\", ""), fontsize=7)
            if i:
                axs[i, 0].set_ylabel(P[k]["label_origin"].replace("\\", ""), fontsize=7)
        fig.legend(handles=[Patch(color=fc[2], label=f"{labs[fm]} 68 %"), Patch(color=fc[1], label=f"{labs[fm]} 95 %"),
                            Line2D([], [], color=cols[lm]["s1"], label=f"{labs[lm]} 68 %"),
                            Line2D([], [], color=cols[lm]["s1"], ls="--", label=f"{labs[lm]} 95 %"),
                            Line2D([], [], color="k", ls="--", label="true value"),
                            Line2D([], [], color="#E41A1C", marker="*", mfc="none", ls="", label="best fit")],
                   loc="upper right", fontsize=8)
        fig.tight_layout()
        fig.savefig(png, dpi=110)
    print("preview:", png)


if __name__ == "__main__":
    main()
