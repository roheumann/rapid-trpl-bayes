"""
run_notebook.py — execute main.ipynb / validation.ipynb headless with settings
=============================================================================

Replaces values in the settings cell (the first code cell that defines
CONFIG_PATH) and executes the notebook with nbconvert; the executed copy
(with all outputs and figures) is written to OUT.

Usage (from the repository root):
    python scripts/run_notebook.py NOTEBOOK OUT [KEY=VALUE ...]

Example:
    python scripts/run_notebook.py main.ipynb results/executed_2trap_85-15_joint.ipynb \\
        'CONFIG_PATH="configs/2trap_85-15_experimental.json"' 'MODE="joint"' 'RUN_PROFILE=True'
"""
import re
import sys
import time

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor


def main(notebook, out, assignments):
    nb = nbformat.read(notebook, as_version=4)
    cell = next(c for c in nb.cells if c.cell_type == "code" and re.search(r"^CONFIG_PATH\s*=", c.source, re.M))
    src = cell.source
    for kv in assignments:
        k, v = kv.split("=", 1)
        src, n = re.subn(rf"^{k}\s*=.*$", f"{k} = {v}", src, flags=re.M)
        if n != 1:
            raise KeyError(f"setting {k} not found in the settings cell")
    cell.source = src
    t0 = time.time()
    try:
        ExecutePreprocessor(timeout=-1, kernel_name="python3").preprocess(nb, {"metadata": {"path": "."}})
    finally:
        nbformat.write(nb, out)
        print(f"executed in {time.time() - t0:.0f} s -> {out}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
