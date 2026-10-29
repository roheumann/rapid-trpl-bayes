"""
raw_trpl_points.py — raw measured tau_diff vs QFLS points of a main.ipynb run (no spline), for plots
====================================================================================================
27/10/2026 (Robin).  The same chain as main.ipynb with trpl.points = "raw": load the raw transient of the run's config,
trim_and_normalize, t_pl_to_qfls_tau at the run's n_pulse and E_g, clean_trpl (finite, tau_diff > 0, sorted).

    python scripts/raw_trpl_points.py <run dir with settings.json> [--out <csv>]

Output (default): <run dir>/trpl_raw_points.csv with columns qfls, tau_diff (the format of trpl_used.csv).
"""
import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "preprocessing"))

import data_utils                                       # noqa: E402
from inference import data                              # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("run")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    run = Path(a.run)
    cfg = json.loads((run / "settings.json").read_text())["config"]
    tr = cfg["trpl"]
    path = Path(tr["path"]) if Path(tr["path"]).is_absolute() else REPO / tr["path"]
    raw = data.load_trpl_raw(path, skiprows=tr.get("skiprows", "auto"))
    with contextlib.redirect_stdout(io.StringIO()):
        trimmed = data_utils.trim_and_normalize(raw.copy())
        conv = data_utils.t_pl_to_qfls_tau(trimmed, cfg["physics"]["npulse_cm3"], cfg["physics"]["Eg_eV"])
    pts = data.clean_trpl(conv, qfls_range=tr.get("qfls_window"))
    out = Path(a.out) if a.out else run / "trpl_raw_points.csv"
    pts.to_csv(out, index=False)
    print(f"{len(pts)} raw points, QFLS {pts.qfls.min():.4f} … {pts.qfls.max():.4f} eV -> {out}")


if __name__ == "__main__":
    main()
