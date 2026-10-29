"""
build_corner_traps.py — per-trap corner plot in Origin (as Figure 9 of the paper), from the same HDF5 file
=========================================================================================================
Only the relations of the parameters WITHIN each trap, no marginal posteriors (no diagonal), no simulation panels:
  trap 1: lower-left triangle, x labels at the bottom, y labels on the left;
  trap 2: the same panels mirrored into the upper-right triangle, x labels on top, y labels on the right.
E_t - E_g is an extra axis on the n1 column of each trap.  Everything else (colour map, colour bar, white log ticks,
10^n labels, merge into one page) comes from build_corner.py, which must lie in the same folder (BUILD_CORNER).

Run inside Origin:  run -pyf "C:\\...\\origin\\build_corner_traps.py"   (h5py as for build_corner.py)
Output next to the .h5 file: corner_traps.opju / .pdf / .png
"""
import os

import numpy as np

H5 = None                  # None -> file dialog
BUILD_CORNER = r"C:\Users\ro.heumann\Documents\PhDProject\NN_Paper\Data-Final\Second-Submission\origin\build_corner.py"
STYLE_OVERRIDES = {"cbar_gap_cm": 2.8}     # room for the trap-2 labels and titles on the right of the grid


def load_build_corner():
    """build_corner.py as a library (its main() is not run)."""
    path = BUILD_CORNER
    if not os.path.isfile(path) and "__file__" in globals():
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "build_corner.py")
    bc = {"__name__": "build_corner", "_NO_MAIN": True}
    with open(path, encoding="utf-8") as fh:
        exec(compile(fh.read(), path, "exec"), bc)
    return bc


def trap_layout(G):
    """Layout table (same dtype as the file's) with the within-trap pairs only.  Trap t's parameters are the keys
    ending in _<t> (n1, tau_p, tau_n, N_t in file order).  Pair (i < j): trap 1 at (row j, col i), trap 2 mirrored at
    (row m-1-j, col m-1-i)."""
    dt = G["layout"].dtype
    rows = []
    traps = sorted({k.rsplit("_", 1)[1] for k in G["keys"] if k.rsplit("_", 1)[-1].isdigit()})
    for t in traps[:2]:
        keys = [k for k in G["keys"] if k.endswith(f"_{t}")]
        m = len(keys)
        for j in range(1, m):
            for i in range(j):
                kx, ky = keys[i], keys[j]
                if f"{kx}__{ky}" not in G["panels"]:
                    continue
                n1 = "log10_Nc" in G["params"][kx] and j == 1
                if t == traps[0]:
                    r, c = j, i
                    xs, ys = ("bottom" if j == m - 1 else "none"), ("left" if i == 0 else "none")
                    extra = "top:Et-Eg" if n1 else ""
                else:
                    r, c = m - 1 - j, m - 1 - i
                    xs, ys = ("top" if j == m - 1 else "none"), ("right" if i == 0 else "none")
                    extra = "bottom:Et-Eg" if n1 else ""
                rows.append((r, c, b"2d", kx.encode(), ky.encode(), xs.encode(), ys.encode(), extra.encode()))
    return np.array(rows, dtype=dt)


def main():
    bc = load_build_corner()
    bc["STYLE"].update(STYLE_OVERRIDES)
    if H5:
        bc["H5"] = H5
    path = bc["pick_file"]()
    G = bc["read_group"](path, "physical")
    G["layout"] = trap_layout(G)
    G["curves"] = {}                                    # no simulation panels
    print(f"per-trap corner: {len(G['layout'])} panels")
    gp = bc["build_corner"](G, path, "traps")
    if gp is None:
        return
    base = os.path.join(os.path.dirname(path), "corner_traps")
    bc["step"]("project")(bc["save_project"], base + ".opju")
    bc["step"]("export pdf")(gp.save_fig, base + ".pdf")
    bc["step"]("export png")(gp.save_fig, base + ".png", width=2400)
    print(f"done: {base}.opju / .pdf / .png")
    if bc["LOG"]:
        print("\nsteps that failed (the rest of the figure was built):")
        for m in bc["LOG"]:
            print("  ", m)
    if bc["LT_FAILED"]:
        print("\nLabTalk statements that failed (count, first example):")
        for n, ex in bc["LT_FAILED"].values():
            print(f"   {n:4d} x  {ex}")


main()
