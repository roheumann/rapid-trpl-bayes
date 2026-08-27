"""
01_create_grid.py  —  Parameter grid creation
==============================================
Creates the parameter grid for the Bayesian pipeline.

Two modes (set in config or via --mode flag):
  full   Cartesian product over all parameter ranges.
         Streams row-by-row so very large grids (>60 M rows) can be written
         without holding everything in RAM.
  slice  2-D cross-sections through the best-fit point (one slice per
         parameter pair) + 1-D profiles (one per parameter). All other
         parameters are fixed at the best-fit value.

Outputs written to <output_dir>/
  full mode:
    grid.csv            all parameter combinations
    grid_metadata.json  axis values, bounds, spacing
  slice mode:
    grid_slices.csv     2-D slices (column slice_pair labels each panel)
    grid_1d.csv         1-D profiles (column varied_param labels each)
    grid_metadata.json

Usage
-----
  python 01_create_grid.py --config config.json
  python 01_create_grid.py --config config.json --mode slice
"""

import argparse, json, os, sys, time
import numpy as np
import pandas as pd
from itertools import combinations, product

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_SCRIPT_DIR, ".."))


def _p(msg): print(msg, flush=True)


# ── Parameter helpers ─────────────────────────────────────────────────────────

def _param_keys(num_traps):
    keys = []
    for k in range(1, num_traps + 1):
        keys.extend([f"Et_{k}", f"Nt_{k}", f"tau_n_{k}", f"tau_p_{k}"])
    keys.append("krad")
    return keys


def _csv_col(key):
    if key.startswith("Et_"):   return key + "_eV"
    if key.startswith("Nt_"):   return key + "_cm-3"
    if key.startswith("tau_"):  return key + "_s"
    if key == "krad":            return "krad_cm3s"
    return key


def _is_log(key):
    return not key.startswith("Et_")


def _axis(lo, hi, n, log_scale):
    if log_scale:
        return np.logspace(np.log10(lo), np.log10(hi), n)
    return np.linspace(lo, hi, n)


def make_anchored_grid(lo, hi, n, anchor_val, log_scale):
    """
    Return (grid, anchor_idx) where grid has n points spanning ≈[lo,hi]
    with anchor_val as an exact node.  Step size is derived from the bounds;
    the start is shifted so anchor_val lands exactly on index anchor_idx.
    """
    if log_scale:
        log_lo, log_hi = np.log10(lo), np.log10(hi)
        log_anc = np.log10(anchor_val)
        step = (log_hi - log_lo) / (n - 1)
        idx  = int(round((log_anc - log_lo) / step))
        idx  = max(0, min(n - 1, idx))
        log_start = log_anc - idx * step
        return np.logspace(log_start, log_start + (n - 1) * step, n), idx
    else:
        step = (hi - lo) / (n - 1)
        idx  = int(round((anchor_val - lo) / step))
        idx  = max(0, min(n - 1, idx))
        start = anchor_val - idx * step
        return np.linspace(start, start + (n - 1) * step, n), idx


# ── Config parsing ────────────────────────────────────────────────────────────

def _load_config(path):
    with open(path) as fh:
        return json.load(fh)


def _build_axes(cfg):
    """Return dict key → 1-D numpy array of grid values."""
    num_traps = cfg["physics"]["num_traps"]
    g = cfg["grid"]
    axes = {}
    for k in range(1, num_traps + 1):
        t = g["traps"][k - 1]
        axes[f"Et_{k}"]    = _axis(t["Et_bounds_eV"][0],   t["Et_bounds_eV"][1],
                                    t["Et_n_points"], False)
        axes[f"Nt_{k}"]    = _axis(t["Nt_bounds_cm3"][0],  t["Nt_bounds_cm3"][1],
                                    t["Nt_n_points"], True)
        axes[f"tau_n_{k}"] = _axis(t["taun_bounds_s"][0],  t["taun_bounds_s"][1],
                                    t["taun_n_points"], True)
        axes[f"tau_p_{k}"] = _axis(t["taup_bounds_s"][0],  t["taup_bounds_s"][1],
                                    t["taup_n_points"], True)
    axes["krad"] = _axis(g["krad_bounds_cm3s"][0], g["krad_bounds_cm3s"][1],
                          g["krad_n_points"], True)
    return axes


def _load_best_fit(cfg):
    """Return dict  key → value  (internal key names, e.g. 'Et_1', 'krad')."""
    bf_path = os.path.join(os.path.dirname(os.path.abspath(
        cfg.get("_config_path", "config.json"))), cfg["best_fit"]["json_path"])
    if not os.path.isabs(bf_path):
        bf_path = os.path.join(_SCRIPT_DIR, "..", cfg["best_fit"]["json_path"])
    with open(bf_path) as fh:
        js = json.load(fh)
    num_traps = cfg["physics"]["num_traps"]
    bf = {}
    traps = js.get("traps") or js.get("trap_parameters")
    for k in range(1, num_traps + 1):
        t = traps[k - 1]
        bf[f"Et_{k}"]    = float(t["Et_eV"])
        bf[f"Nt_{k}"]    = float(t.get("Nt_cm3") or t.get("Nt_1/cm3"))
        bf[f"tau_n_{k}"] = float(t["tau_n_s"])
        bf[f"tau_p_{k}"] = float(t["tau_p_s"])
    bf["krad"] = float(js["krad_cm3s"])
    return bf


# ── Full grid ─────────────────────────────────────────────────────────────────

def create_full_grid(cfg, out_dir):
    axes     = _build_axes(cfg)
    keys     = _param_keys(cfg["physics"]["num_traps"])
    col_names = [_csv_col(k) for k in keys]
    chunk_sz = cfg["grid"].get("chunk_size", 100_000)

    out_csv  = os.path.join(out_dir, "grid.csv")

    total = 1
    for k in keys:
        total *= len(axes[k])
    _p(f"Full grid: {total:,} rows × {len(keys)} params → {out_csv}")

    # Stream: outer loop over krad, inner Cartesian product over the rest
    # This keeps peak RAM to ≈ chunk_sz rows × n_params × 8 bytes
    krad_arr  = axes["krad"]
    inner_keys = [k for k in keys if k != "krad"]
    inner_axes = [axes[k] for k in inner_keys]
    inner_cols = [_csv_col(k) for k in inner_keys]

    t0 = time.time()
    rows_written = 0
    first = True

    with open(out_csv, "w") as fout:
        fout.write(",".join(col_names) + "\n")

    for krad_val in krad_arr:
        buf = []
        for combo in product(*inner_axes):
            row = list(combo) + [krad_val]
            buf.append(row)
            if len(buf) >= chunk_sz:
                _write_chunk(buf, col_names, inner_cols + [_csv_col("krad")], out_csv)
                rows_written += len(buf)
                buf = []
        if buf:
            _write_chunk(buf, col_names, inner_cols + [_csv_col("krad")], out_csv)
            rows_written += len(buf)
        _p(f"  {rows_written:>12,} rows  ({time.time()-t0:.1f}s)")

    _p(f"Done. {rows_written:,} rows in {time.time()-t0:.1f}s")

    # ── Metadata ──────────────────────────────────────────────────────────────
    meta = {
        "mode": "full",
        "total_rows": rows_written,
        "params": {
            _csv_col(k): {
                "values": axes[k].tolist(),
                "n": len(axes[k]),
                "log_scale": _is_log(k),
            }
            for k in keys
        },
    }
    _write_meta(meta, out_dir)
    return out_csv


def _write_chunk(buf, all_col_names, buf_col_names, out_csv):
    df = pd.DataFrame(buf, columns=buf_col_names)
    df = df.reindex(columns=all_col_names)
    df.to_csv(out_csv, mode="a", header=False, index=False, float_format="%.6e")


# ── Slice grid ────────────────────────────────────────────────────────────────

def create_slice_grid(cfg, out_dir):
    """
    2-D slices and 1-D profiles anchored at the optimizer best-fit.

    Each parameter's 1-D axis uses the n_points from config (per-parameter,
    so krad can have 10 while others have 50).  The best-fit value is an
    exact grid node on every axis (make_anchored_grid shifts the grid so
    the best-fit lands exactly on a node).  All parameters not being varied
    are fixed at their grid-anchored best-fit values.
    """
    keys      = _param_keys(cfg["physics"]["num_traps"])
    col_names = [_csv_col(k) for k in keys]
    g         = cfg["grid"]

    bf = _load_best_fit(cfg)   # internal key → optimizer value

    # ── Build anchored 1-D axes using per-parameter n_points ─────────────────
    slice_axes  = {}   # internal key → 1-D array
    anchor_idx  = {}   # internal key → int index of best-fit in that axis

    for k in range(1, cfg["physics"]["num_traps"] + 1):
        t = g["traps"][k - 1]
        specs = [
            (f"Et_{k}",    t["Et_bounds_eV"][0],   t["Et_bounds_eV"][1],
             t["Et_n_points"],   False),
            (f"Nt_{k}",    t["Nt_bounds_cm3"][0],  t["Nt_bounds_cm3"][1],
             t["Nt_n_points"],   True),
            (f"tau_n_{k}", t["taun_bounds_s"][0],   t["taun_bounds_s"][1],
             t["taun_n_points"], True),
            (f"tau_p_{k}", t["taup_bounds_s"][0],   t["taup_bounds_s"][1],
             t["taup_n_points"], True),
        ]
        for key, lo, hi, n, log_scale in specs:
            grid, idx = make_anchored_grid(lo, hi, n, bf[key], log_scale)
            slice_axes[key] = grid
            anchor_idx[key] = idx

    grid, idx = make_anchored_grid(
        g["krad_bounds_cm3s"][0], g["krad_bounds_cm3s"][1],
        g["krad_n_points"], bf["krad"], True)
    slice_axes["krad"] = grid
    anchor_idx["krad"] = idx

    # Exact grid-node values used to fix non-varied parameters
    # (may differ from bf by ≤ floating-point rounding, but are exact nodes)
    bf_grid = {_csv_col(k): float(slice_axes[k][anchor_idx[k]]) for k in keys}

    _p("Slice axes (anchored at optimizer best-fit):")
    for k in keys:
        col = _csv_col(k)
        node_val = slice_axes[k][anchor_idx[k]]
        _p(f"  {col:25s}: {len(slice_axes[k]):3d} pts  "
           f"anchor[{anchor_idx[k]:2d}] = {node_val:.6e}  "
           f"(optimizer: {bf[k]:.6e})")

    pairs = list(combinations(keys, 2))
    total_2d = sum(len(slice_axes[ki]) * len(slice_axes[kj]) for ki, kj in pairs)
    total_1d = sum(len(slice_axes[k]) for k in keys)
    _p(f"  {len(pairs)} 2-D panels  →  {total_2d:,} rows total")
    _p(f"  {len(keys)} 1-D profiles →  {total_1d:,} rows total")

    # ── 2-D slices ────────────────────────────────────────────────────────────
    slice_rows = []
    for ki, kj in pairs:
        ci, cj     = _csv_col(ki), _csv_col(kj)
        pair_label = f"{ci}|{cj}"
        for vi in slice_axes[ki]:
            for vj in slice_axes[kj]:
                row           = dict(bf_grid)   # fixed params at grid-anchored best-fit
                row[ci]       = vi
                row[cj]       = vj
                row["slice_pair"] = pair_label
                slice_rows.append(row)

    df_slices  = pd.DataFrame(slice_rows)
    out_slices = os.path.join(out_dir, "grid_slices.csv")
    df_slices[col_names + ["slice_pair"]].to_csv(
        out_slices, index=False, float_format="%.6e")
    _p(f"  Slices   → {out_slices}  ({len(df_slices):,} rows)")

    # ── 1-D profiles ──────────────────────────────────────────────────────────
    profile_rows = []
    for k in keys:
        c = _csv_col(k)
        for v in slice_axes[k]:
            row               = dict(bf_grid)
            row[c]            = v
            row["varied_param"] = c
            profile_rows.append(row)

    df_1d  = pd.DataFrame(profile_rows)
    out_1d = os.path.join(out_dir, "grid_1d.csv")
    df_1d[col_names + ["varied_param"]].to_csv(
        out_1d, index=False, float_format="%.6e")
    _p(f"  1-D      → {out_1d}  ({len(df_1d):,} rows)")

    # ── Metadata ──────────────────────────────────────────────────────────────
    meta = {
        "mode": "slice",
        "best_fit_optimizer": {_csv_col(k): float(bf[k]) for k in keys},
        "best_fit_grid":      bf_grid,
        "anchor_indices":     {_csv_col(k): anchor_idx[k] for k in keys},
        "n_pairs": len(pairs),
        "pairs":   [f"{_csv_col(ki)}|{_csv_col(kj)}" for ki, kj in pairs],
        "params": {
            _csv_col(k): {
                "values":     slice_axes[k].tolist(),
                "n":          len(slice_axes[k]),
                "anchor_idx": anchor_idx[k],
                "anchor_val": float(slice_axes[k][anchor_idx[k]]),
                "log_scale":  _is_log(k),
            }
            for k in keys
        },
    }
    _write_meta(meta, out_dir)
    return out_slices, out_1d


def _write_meta(meta, out_dir):
    path = os.path.join(out_dir, "grid_metadata.json")
    with open(path, "w") as fh:
        json.dump(meta, fh, indent=2)
    _p(f"  Metadata → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--mode",   choices=["full", "slice"], default=None,
                    help="Override grid.mode from config")
    args = ap.parse_args()

    cfg_path = args.config if os.path.isabs(args.config) else \
               os.path.join(os.getcwd(), args.config)
    cfg = _load_config(cfg_path)
    cfg["_config_path"] = cfg_path

    mode = args.mode or cfg["grid"]["mode"]
    cfg["grid"]["mode"] = mode

    out_dir = cfg["output_dir"]
    if not os.path.isabs(out_dir):
        out_dir = os.path.join(os.path.dirname(cfg_path), out_dir)
    os.makedirs(out_dir, exist_ok=True)

    _p("=" * 60)
    _p(f"01_create_grid  mode={mode}  num_traps={cfg['physics']['num_traps']}")
    _p(f"  Output dir: {out_dir}")
    _p("=" * 60)

    if mode == "full":
        create_full_grid(cfg, out_dir)
    else:
        create_slice_grid(cfg, out_dir)

    _p("Done.")


if __name__ == "__main__":
    main()
