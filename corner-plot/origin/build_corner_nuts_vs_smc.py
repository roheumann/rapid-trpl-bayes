"""
build_corner_nuts_vs_smc.py — NUTS and SMC posteriors of one run in one corner plot (Origin)
=============================================================================================
Input : <run>/smc_B2nf/corner_nuts_vs_smc/corner_nuts_vs_smc.h5 (scripts/nuts_vs_smc_corner.py on rekenbeest),
        1-trap synthetic case (B2nf, k_rad fixed), modes trpl / sspl / joint
Output: corner_nuts_vs_smc.opju / .pdf / .png next to the .h5 file

The look of the NN vs ODE corner (build_corner_nn_vs_ode.py, loaded as a library): NUTS (reference, blue) filled 1σ
dark with a solid outline / 2σ light, SMC (wine red) as 1σ solid / 2σ dashed lines of equal width; true value as
dashed black lines, best fit as an open red star; legend "NUTS" / "SMC" (labels from the file).
build_corner.py, build_corner_modes.py and build_corner_nn_vs_ode.py must lie in the folder of BUILD_CORNER.
Run inside Origin:  run -pyf "C:\\...\\origin\\build_corner_nuts_vs_smc.py"
"""
import os

H5 = None                  # None or missing -> file dialog; e.g.
# r"R:\simulations\rapid-trpl-bayes\results\1trap_synthetic_paper_B2nf_kradfixed_joint\smc_B2nf\corner_nuts_vs_smc\corner_nuts_vs_smc.h5"
BUILD_CORNER = r"C:\Users\ro.heumann\Documents\PhDProject\NN_Paper\Data-Final\Second-Submission\origin\build_corner.py"


def load_nn_vs_ode():
    path = os.path.join(os.path.dirname(BUILD_CORNER), "build_corner_nn_vs_ode.py")
    if not os.path.isfile(path) and "__file__" in globals():
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "build_corner_nn_vs_ode.py")
    g = {"__name__": "build_corner_nn_vs_ode", "_NO_MAIN": True}
    with open(path, encoding="utf-8") as fh:
        exec(compile(fh.read(), path, "exec"), g)
    return g


def main():
    g = load_nn_vs_ode()
    g["BUILD_CORNER"] = BUILD_CORNER
    g["H5"] = H5 if H5 and os.path.isfile(H5) else None
    g["OUT_NAME"] = "corner_nuts_vs_smc"
    g["main"]()


if not globals().get("_NO_MAIN"):
    main()
