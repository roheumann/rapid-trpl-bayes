"""Steady-state n, p, n_t, G, PLQY vs QFLS of the synthetic 1-trap truth (numpy twin of sspl_tf.SsplModel)."""
import json, sys, numpy as np, pandas as pd
sys.path.insert(0, ".")
from inference.sspl_tf import SsplSpecNumpy, KB_EV
D = sys.argv[1]
tp = json.load(open(f"{D}/true_parameters.json"))
data = pd.read_csv(f"{D}/synthetic_sspl_suns_qfls.csv")
g_sun, Eg, Nc = tp["G_per_sun"], tp["Eg_eV"], 2.21359e18
spec = SsplSpecNumpy(Eg=Eg, n_traps=1, Nc=Nc, Nv=Nc, kT=KB_EV * 300.0, n_sweep=np.logspace(8, np.log10(Nc), 160),
                     solver="auto", mode="qfls_vs_suns", x_model=np.log10(data.intensity_suns.values * g_sun))
s = {k: v[0] for k, v in spec.solve(np.atleast_2d(tp["theta"])).items()}
full = pd.DataFrame({"suns": s["G"] / g_sun, "qfls_eV": s["qfls"], "n_cm3": s["n"], "p_cm3": s["p"],
                     "nt_1_cm3": s["nt_1"], "G_cm3s": s["G"], "plqy": s["plqy"]})
lg, x = np.log10(s["G"]), np.log10(data.intensity_suns.values * g_sun)
at = pd.DataFrame({"suns": data.intensity_suns.values,
                   **{c: 10 ** np.interp(x, lg, np.log10(full[c])) for c in ["n_cm3", "p_cm3", "nt_1_cm3", "G_cm3s", "plqy"]}})
at.insert(1, "qfls_eV", np.interp(x, lg, s["qfls"]))
print("check vs synthetic data: max |dQFLS| [eV]", float(np.max(np.abs(at.qfls_eV - data.qfls_eV))),
      " max rel dPLQY", float(np.max(np.abs(at.plqy / data.plqy - 1))))
print("max |p - n - nt|/p", float(np.max(np.abs(full.p_cm3 - full.n_cm3 - full.nt_1_cm3) / full.p_cm3)))
head = "# synthetic 1-trap truth (85:15 fit, GaAs sigma): steady state, numpy twin of inference/sspl_tf.SsplModel; p = n + nt_1\n"
for name, df in [("synthetic_sspl_carriers_truth_curve.csv", full), ("synthetic_sspl_carriers_truth_at_data.csv", at)]:
    with open(f"{D}/{name}", "w") as f:
        f.write(head); df.to_csv(f, index=False, float_format="%.10g")
print(at.to_string())
