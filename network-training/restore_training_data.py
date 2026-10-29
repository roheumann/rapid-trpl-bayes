"""
restore_training_data.py — rebuild a training HDF5 from its compact archive version
====================================================================================

    python restore_training_data.py <compact.h5> <training.hdf5> [--no-verify]

Inverse of compact_training_data.py.  Writes

    <training.hdf5>                        all datasets of the original training file, in the original order
    <stem>_param_scaler.joblib             StandardScaler of the inputs
    <stem>_output_scaler.joblib            MinMaxScaler of the outputs

param_scaler_path / output_scaler_path inside the HDF5 point to these two files (the training scripts load the
scalers from there).  Every other dataset is checked against the SHA-256 checksum of the original (switch off with
--no-verify).  The result is the input of train_conv_v2.py / train_nn.py / train_axis_head.py, e.g.

    python restore_training_data.py 1trap_training_compact.h5 data/1Trap_TRPL_3609plus2e20_training.hdf5
    python train_conv_v2.py data/1Trap_TRPL_3609plus2e20_training.hdf5 --output-dir <net> ...
"""
import argparse
import hashlib
import io
from pathlib import Path

import h5py
import joblib
import numpy as np

BLOCK = 131072
PATH_KEYS = ("param_scaler_path", "output_scaler_path")


def qfls_from_ends(lo, hi, m):
    return np.linspace(lo.astype(np.float64), hi.astype(np.float64), m, axis=1).astype(np.float32)


def undo_bit_delta(d):
    return np.cumsum(d, axis=1, dtype=np.uint32).view(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("compact")
    ap.add_argument("out")
    ap.add_argument("--no-verify", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    stem = out.with_suffix("")
    sp, so = Path(f"{stem}_param_scaler.joblib"), Path(f"{stem}_output_scaler.joblib")

    with h5py.File(a.compact, "r") as hc:
        if hc.attrs.get("compact_format") != 1:
            raise SystemExit(f"{a.compact} is not a compact training file (compact_format 1)")
        meta = hc["_compact"]
        sp.write_bytes(meta["param_scaler.joblib"][()].tobytes())
        so.write_bytes(meta["output_scaler.joblib"][()].tobytes())
        p_scaler = joblib.load(io.BytesIO(meta["param_scaler.joblib"][()].tobytes()))
        o_scaler = joblib.load(io.BytesIO(meta["output_scaler.joblib"][()].tobytes()))
        sums, info = meta["sha256"].attrs, meta["datasets"].attrs

        def shape_of(k):
            return tuple(int(x) for x in info[k].split("|")[1].split(",") if x)

        bad = []
        with h5py.File(out, "w") as ho:
            for k, v in hc.attrs.items():
                if k not in ("compact_format", "compact_source", "dataset_order") and "__nonzero" not in k:
                    ho.attrs[k] = v
            order = list(hc.attrs["dataset_order"])
            raw_first = sorted(order, key=lambda k: 0 if (k.endswith("_raw") or k in hc) else 1)
            for k in raw_first:                  # raw data first: X_<s> / y_<s> are recomputed from them
                if k in PATH_KEYS:
                    continue
                h = hashlib.sha256()
                if k in hc:                      # stored as is
                    ds = hc[k]
                    if ds.ndim and ds.shape[0] > BLOCK:
                        o = ho.create_dataset(k, shape=ds.shape, dtype=ds.dtype)
                        for i in range(0, ds.shape[0], BLOCK):
                            blk = ds[i:i + BLOCK]
                            o[i:i + BLOCK] = blk
                            h.update(np.ascontiguousarray(blk).tobytes())
                    else:
                        hc.copy(k, ho)               # keeps dtype (e.g. variable-length strings)
                        v = ds[()]
                        h.update(v if isinstance(v, bytes) else np.asarray(v).tobytes())
                else:                            # rebuilt
                    n, m = shape_of(k)
                    o = ho.create_dataset(k, shape=(n, m), dtype=np.float32)
                    corr = hc.get(f"{k}__correction")
                    for i in range(0, n, BLOCK):
                        if f"{k}__bitdelta" in hc:
                            blk = undo_bit_delta(hc[f"{k}__bitdelta"][i:i + BLOCK])
                        elif k.startswith("qfls_"):
                            blk = qfls_from_ends(hc[f"{k}__first"][i:i + BLOCK], hc[f"{k}__last"][i:i + BLOCK], m)
                        else:
                            sc = p_scaler if k.startswith("X_") else o_scaler
                            blk = sc.transform(ho[f"{k}_raw"][i:i + BLOCK]).astype(np.float32)
                        if corr is not None:
                            bits = blk.view(np.int32).astype(np.int64) + corr[i:i + BLOCK].astype(np.int64)
                            blk = bits.astype(np.int32).view(np.float32)
                        o[i:i + BLOCK] = blk
                        h.update(np.ascontiguousarray(blk).tobytes())
                if not a.no_verify and h.hexdigest() != sums[k]:
                    bad.append(k)
                print(f"  {k}: {'checksum OK' if k not in bad else 'CHECKSUM MISMATCH'}", flush=True)
            ho["param_scaler_path"] = str(sp.resolve())
            ho["output_scaler_path"] = str(so.resolve())
    if bad and not a.no_verify:
        raise SystemExit(f"checksum mismatch: {bad}")
    print(f"wrote {out} ({out.stat().st_size / 1e9:.2f} GB), {sp.name}, {so.name}")


if __name__ == "__main__":
    main()
