"""
merge_training_data.py — training HDF5 for run B2 (3609 data + extra Sobol samples)
===================================================================================
14/10/2026  (revision, plan ~/.claude/plans/floofy-purring-fiddle.md, Part 1)

    python merge_training_data.py <base_training.hdf5> <extra_training.hdf5> <out.hdf5>

* base  = the 3609 training HDF5 (split seed 42, scalers fitted on its train split)
* extra = prepare_training_data.py output of the new samples (same bounds, other Sobol scramble)

Result:
    train = base train + ALL extra samples (its own train/val/test)
    val   = base val      (so the validation loss is comparable with 3609)
    test  = base test     (never trained on; the comparison set of every run)
All splits are scaled with the BASE scalers (copied next to the output), so the network output scale
is identical to 3609.  Only the arrays the training and inference.surrogate read are written
(no pl/time/tau_diff copies).
"""
import shutil
import sys
from pathlib import Path

import h5py
import joblib
import numpy as np


def _s(v):
    return v.decode() if isinstance(v, bytes) else str(v)


def main(base_path, extra_path, out_path):
    base_path, extra_path, out_path = Path(base_path), Path(extra_path), Path(out_path)
    with h5py.File(base_path, "r") as hb, h5py.File(extra_path, "r") as he:
        assert np.allclose(hb["lb"][()], he["lb"][()]) and np.allclose(hb["ub"][()], he["ub"][()]), "bounds differ"
        assert list(hb["param_names"][()]) == list(he["param_names"][()]), "parameter names differ"
        p_scaler = joblib.load(_s(hb["param_scaler_path"][()]))
        o_scaler = joblib.load(_s(hb["output_scaler_path"][()]))

        Xe = np.concatenate([he[f"X_{s}_raw"][()] for s in ("train", "val", "test")])
        ye = np.concatenate([he[f"y_{s}_raw"][()] for s in ("train", "val", "test")])
        ok = np.all(np.isfinite(Xe), axis=1) & np.all(np.isfinite(ye), axis=1)
        Xe, ye = Xe[ok], ye[ok]
        print(f"extra samples: {len(Xe)} ({(~ok).sum()} non-finite dropped)")
        lo, hi = o_scaler.data_min_.min(), o_scaler.data_max_.max()
        print(f"extra y range [{ye.min():.3f}, {ye.max():.3f}] vs base scaler range [{lo:.3f}, {hi:.3f}] "
              f"({np.mean((ye < lo) | (ye > hi)):.2e} of the values outside)")

        qe = np.concatenate([he[f"qfls_{s}"][()] for s in ("train", "val", "test")])[ok]
        q_tr = np.concatenate([hb["qfls_train"][()], qe])           # per-curve QFLS axes (read by the
        X_tr_raw = np.concatenate([hb["X_train_raw"][()], Xe]).astype(np.float32)   # QFLS-axis check of main.ipynb)
        y_tr_raw = np.concatenate([hb["y_train_raw"][()], ye]).astype(np.float32)
        rng = np.random.default_rng(42)
        perm = rng.permutation(len(X_tr_raw))
        X_tr_raw, y_tr_raw, q_tr = X_tr_raw[perm], y_tr_raw[perm], q_tr[perm]

        out_path.parent.mkdir(parents=True, exist_ok=True)
        stem = out_path.with_suffix("")
        ps_path, os_path = Path(f"{stem}_param_scaler.joblib"), Path(f"{stem}_output_scaler.joblib")
        shutil.copy(_s(hb["param_scaler_path"][()]), ps_path)
        shutil.copy(_s(hb["output_scaler_path"][()]), os_path)

        with h5py.File(out_path, "w") as ho:
            ho["X_train_raw"], ho["y_train_raw"] = X_tr_raw, y_tr_raw
            ho["X_train"] = p_scaler.transform(X_tr_raw).astype(np.float32)
            ho["y_train"] = o_scaler.transform(y_tr_raw).astype(np.float32)
            ho["qfls_train"] = q_tr
            for s in ("val", "test"):
                for k in (f"X_{s}", f"y_{s}", f"X_{s}_raw", f"y_{s}_raw"):
                    ho[k] = hb[k][()]
            for k in ("qfls_val", "qfls_test", "lb", "ub", "constants", "N_TRAPS", "param_names"):
                ho[k] = hb[k][()]
            ho["param_scaler_path"] = str(ps_path.resolve())
            ho["output_scaler_path"] = str(os_path.resolve())
            ho["source_file"] = f"{_s(hb['source_file'][()])} + {_s(he['source_file'][()])}"
            ho.attrs["merged_from"] = [str(base_path), str(extra_path)]
        print(f"wrote {out_path}: train {len(X_tr_raw)}, val {len(hb['X_val'])}, test {len(hb['X_test'])}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    main(*sys.argv[1:])
