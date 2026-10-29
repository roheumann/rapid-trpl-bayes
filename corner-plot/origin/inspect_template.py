"""
inspect_template.py — run ONCE inside Origin (2021+) on Windows
==============================================================
Dumps page size, layer geometry, axis scales/ranges, fonts and tick settings of the old corner templates
(corner-Plot-1Trap.otpu, corner-Plot-2Traps.otpu) to old_template_style.json, so that build_corner.py can
reproduce the old look exactly (copy the numbers into STYLE in build_corner.py, or send the JSON back).

Run: Origin -> Window -> Script Window:
    run -pyf "C:\\path\\to\\corner-plot\\origin\\inspect_template.py"
(or open it in Code Builder and press F5).  Needs no extra packages.
"""
import json
import os

import originpro as op

HERE = None                 # folder of corner-Plot-*.otpu; None -> folder dialog (__file__ is not always defined
                            # when Origin runs a script)
NAMES = ("corner-Plot-1Trap.otpu", "corner-Plot-2Traps.otpu")


def pick_dir():
    if HERE:
        return HERE
    op.lt_exec('dlgpath title:="folder with corner-Plot-1Trap.otpu / corner-Plot-2Traps.otpu"; string __d$ = path$;')
    return op.get_lt_str("__d$")

# LabTalk properties of a layer that define the look (missing ones are skipped)
LAYER_PROPS = ["unit", "left", "top", "width", "height", "x.from", "x.to", "x.type", "y.from", "y.to", "y.type",
               "x.inc", "y.inc", "x.ticks", "y.ticks", "x.showAxes", "y.showAxes", "x.label.pt", "y.label.pt",
               "x.majorTickLength", "y.majorTickLength", "x.thickness", "y.thickness", "plotcolor"]


def lt_get(expr, kind="f"):
    try:
        return op.lt_float(expr) if kind == "f" else op.lt_string(expr) if hasattr(op, "lt_string") else None
    except Exception:
        return None


def main():
    out = {}
    here = pick_dir()
    for path in [os.path.join(here, f) for f in NAMES]:
        path = os.path.abspath(path)
        if not os.path.exists(path):
            print("missing:", path)
            continue
        gp = op.new_graph(template=path)
        page = {"template": os.path.basename(path), "layers": []}
        op.lt_exec(f"win -a {gp.name}")
        page["page_width"] = lt_get("page.width")
        page["page_height"] = lt_get("page.height")
        page["page_resolution"] = lt_get("page.resx")
        n = int(lt_get("page.nlayers") or 0)
        for i in range(1, n + 1):
            op.lt_exec(f"page.active = {i}")
            lay = {"index": i}
            for p in LAYER_PROPS:
                lay[p] = lt_get(f"layer.{p}")
            try:
                gl = gp[i - 1]
                lay["name"] = gl.name
                lay["n_plots"] = len(gl.plot_list())
            except Exception as e:                      # empty layers in a template have no plots
                lay["error"] = str(e)
            # axis titles (text objects xb / yl / yr / xt) — strings, may be empty
            for obj in ("xb", "xt", "yl", "yr"):
                try:
                    op.lt_exec(f"string __s$ = {obj}.text$;")
                    lay[f"title_{obj}"] = op.get_lt_str("__s$")
                    lay[f"title_{obj}_fsize"] = lt_get(f"{obj}.fsize")
                except Exception:
                    pass
            page["layers"].append(lay)
        out[os.path.basename(path)] = page
        print(f"{os.path.basename(path)}: {n} layers")
    dst = os.path.join(here, "old_template_style.json")
    with open(dst, "w") as fh:
        json.dump(out, fh, indent=1)
    print("written:", dst)


main()
