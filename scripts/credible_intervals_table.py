"""
credible_intervals_table.py — human-readable 68 % / 95 % credible intervals of the SMC posteriors of a synthetic case
===================================================================================================================
20/10/2026 (Robin).  For every mode run <base>_{trpl,sspl,joint}: smc/credible_intervals_smc.csv (central intervals of
the equal-weight SMC draws, theta coordinates) converted to physical units (log10 -> value, E_t/E_g -> E_t in eV), with
the true value and whether it lies inside.  Fixed parameters are listed as fixed.

    python scripts/credible_intervals_table.py results/1trap_synthetic/1trap_synthetic \
        --out results/1trap_synthetic/credible_intervals_1trap_synthetic
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

UNITS = {"krad": ("k_rad", "cm^3/s"), "Nt": ("N_t", "cm^-3"), "taun": ("tau_n", "s"), "taup": ("tau_p", "s"),
         "Et": ("E_t", "eV")}


def physical(name, v, Eg):
    if name.startswith("log10_"):
        return 10.0 ** v
    if name.startswith("Et_rel"):
        return v * Eg
    return v                                    # already physical (E_t in eV: display key Et_<k>_eV)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("--smc", default="smc")
    ap.add_argument("--out", required=True)
    ap.add_argument("--suffix", default="", help="run-folder suffix, e.g. _QUICK")
    a = ap.parse_args()
    rows, md = [], ["# 68 % and 95 % credible intervals (central, SMC posterior) — " + Path(a.base).name, ""]
    for mode in ("trpl", "sspl", "joint"):
        run = Path(f"{a.base}_{mode}{a.suffix}")
        f = run / a.smc / "credible_intervals_smc.csv"
        if not f.exists():
            continue
        cfg = json.loads((run / "settings.json").read_text())["config"]
        Eg = cfg["physics"]["Eg_eV"]
        tp = cfg.get("true_params") or {}
        et = (tp.get("Et_eV") or [None])[0]
        truth = {"log10_krad": tp.get("krad_cm3s"), "log10_Nt_1": (tp.get("Nt_cm3") or [None])[0],
                 "Et_rel_1": et, "Et_1_eV": et, "log10_taun_1": (tp.get("tau_n_s") or [None])[0],
                 "log10_taup_1": (tp.get("tau_p_s") or [None])[0]}
        ci = pd.read_csv(f)
        md += [f"## mode {mode}", "", "| parameter | unit | truth | median | 68 % interval | 95 % interval | truth in 68 % | truth in 95 % | note |",
               "|---|---|---|---|---|---|---|---|---|"]
        for _, r in ci.iterrows():
            p = r["parameter"]
            if p not in truth:
                continue
            key = p.replace("log10_", "").replace("_rel", "").split("_")[0]
            lab, unit = UNITS.get(key, (p, ""))
            conv = lambda v: physical(p, float(v), Eg)
            med, l68, h68, l95, h95 = (conv(r[c]) for c in ("median", "lo68", "hi68", "lo95", "hi95"))
            tv = truth[p]
            fixed = str(r.get("interval68", "")) == "fixed"
            in68 = "" if (tv is None or fixed) else bool(l68 <= tv <= h68)
            in95 = "" if (tv is None or fixed) else bool(l95 <= tv <= h95)
            lim = [s for s, c in (("lower", "prior_limited_low"), ("upper", "prior_limited_high")) if bool(r.get(c, False))]
            note = "fixed" if fixed else ("posterior extends to the " + " and ".join(lim) + " prior bound" if lim else "")
            rows.append({"mode": mode, "parameter": lab, "unit": unit, "truth": tv, "median": med, "lo68": l68, "hi68": h68,
                         "lo95": l95, "hi95": h95, "truth_in_68": in68, "truth_in_95": in95, "note": note})
            fmt = (lambda v: f"{v:.4f}") if key == "Et" else (lambda v: f"{v:.3e}")
            md.append(f"| {lab} | {unit} | {fmt(tv) if tv is not None else '—'} | {fmt(med)} | [{fmt(l68)}, {fmt(h68)}] | "
                      f"[{fmt(l95)}, {fmt(h95)}] | {in68} | {in95} | {note} |")
        md.append("")
    md.append("Intervals: central 68.27 % / 95.45 % of the equal-weight SMC draws (smc/credible_intervals_smc.csv), "
              "converted from the sampling coordinates (log10 values, E_t/E_g) to physical units.")
    out = Path(a.out)
    pd.DataFrame(rows).to_csv(out.with_suffix(".csv"), index=False)
    out.with_suffix(".md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
