"""
build_corner_modes_1trap.py — the modes corner (build_corner_modes.py) of the 1-trap synthetic run, with fixed axis ranges
=========================================================================================================================
Input : results/1trap_synthetic/1trap_synthetic_joint/corner_modes_trpl_slice/corner_modes.h5
        (corner_modes/corner_modes.h5 with /slice replaced by the n1-tau_p panel of the tr-PL run's slices_paper.h5,
        both iso-tau_0 formulas over the whole panel: scripts/modes_corner.copy_slice(..., tau0_full=True))
Output: corner_modes.opju / .pdf / .png next to the .h5 file (as build_corner_modes.py)

Loads build_corner_modes.py as a library (its main() is not run on import), replaces its axis ranges by RANGES below,
lays the x tick labels flat with the decades of X_TICKS, puts the PPC row below the corner and runs its main().
Everything else (look, PPC panels, slice panel) comes from build_corner_modes.py, which must lie in the folder
of BUILD_CORNER together with build_corner.py.
Run inside Origin:  run -pyf "C:\\...\\origin\\build_corner_modes_1trap.py"
"""
import os

H5 = r"R:\simulations\rapid-trpl-bayes\results\1trap_synthetic\1trap_synthetic_joint\corner_modes_trpl_slice\corner_modes.h5"
                           # None or missing -> file dialog
BUILD_CORNER = r"C:\Users\ro.heumann\Documents\PhDProject\NN_Paper\Data-Final\Second-Submission\origin\build_corner.py"
RANGES = {                 # axis ranges in decades (None keeps that end as it is in the file); every panel with that
    "log10_n1_1": (14.0, None),        # parameter: corner, marginal, slice panel, E_t - E_g axis
    "log10_taup_1": (-10.0, -5.0),
    "log10_taun_1": (-10.0, -6.0),
}
SLICE_RANGES = {           # per slice panel <kx>__<ky>: {k: (lo, hi)}, applied after RANGES in that panel only
    "log10_n1_1__log10_taup_1": {"log10_taup_1": (-9.0, -6.0)},
}
X_LABEL_ROTATE = 0         # x tick labels of the corner and slice panels: flat
PPC_ROW_BELOW = True       # tau_diff and ss-PL PPC squares in a row below the corner (build_corner_modes: above)
PPC_UNDER_CM = 2.2         # ... starting this far below the corner's bottom panels (their x labels + titles)
X_TICKS = {                # the 10^n labels of each x axis (corner, marginals, slice panel); flat labels are centred on
    "log10_n1_1": [15, 16, 17],        # their tick, so none sits on an edge shared with the next panel
    "log10_taup_1": [-10, -8, -6],
    "log10_taun_1": [-9, -8, -7],
    "log10_Nt_1": [14, 16, 18],
}


def load_modes():
    path = os.path.join(os.path.dirname(BUILD_CORNER), "build_corner_modes.py")
    if not os.path.isfile(path) and "__file__" in globals():
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "build_corner_modes.py")
    g = {"__name__": "build_corner_modes", "_NO_MAIN": True}
    with open(path, encoding="utf-8") as fh:
        exec(compile(fh.read(), path, "exec"), g)
    return g


def with_ticks(read):
    """read() of build_corner_modes, plus 'ticks_x' (X_TICKS) on the parameters of the corner and the slice panels."""
    def read_ticks(path):
        D = read(path)
        for P in [D["params"]] + [sl["params"] for sl in D["slices"]]:
            for k, t in X_TICKS.items():
                if k in P:
                    P[k]["ticks_x"] = t
        return D
    return read_ticks


def main():
    g = load_modes()
    g["BUILD_CORNER"] = BUILD_CORNER
    g["H5"] = H5 if H5 and os.path.isfile(H5) else None
    g["RANGES"] = {1: RANGES}              # only the 1-trap entry: a 2-trap file is drawn without ranges
    g["SLICE_RANGES"] = {1: SLICE_RANGES}
    g["X_LABEL_ROTATE"] = X_LABEL_ROTATE
    g["PPC_ROW_BELOW"], g["PPC_UNDER_CM"] = PPC_ROW_BELOW, PPC_UNDER_CM
    g["read"] = with_ticks(g["read"])
    g["main"]()


if not globals().get("_NO_MAIN"):
    main()
