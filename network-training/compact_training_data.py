"""
compact_training_data.py — lossless compact version of a training HDF5 (for archiving)
======================================================================================

    python compact_training_data.py <training.hdf5> <compact.h5> [--scalers <param.joblib> <output.joblib>]

A training HDF5 (prepare_training_data.py / merge_training_data.py) stores every curve three times:
y_<s>_raw (log10 tau_diff), y_<s> (the same curve scaled with the output scaler) and qfls_<s> (its QFLS axis,
linspace(q_lo, q_hi, 256)), and the inputs twice (X_<s>_raw and the scaled X_<s>).  The compact file keeps

* every dataset that cannot be recomputed, gzip-compressed; the curves y_<s>_raw as bit deltas along the QFLS axis
  (uint32 differences of the float32 bit patterns of neighbouring points; much smaller after gzip, exact);
* for X_<s> and y_<s>: nothing but a correction (normally all zero), because they are scaler.transform(raw);
* for qfls_<s>: the first and last point of every axis and the bit correction of the float32 linspace;
* both scaler files (joblib bytes) and a SHA-256 checksum of every original dataset.

restore_training_data.py rebuilds the training HDF5 with all datasets bit-identical to the original (except the two
scaler-path strings, which point to the restored scaler files).
"""
import argparse
import hashlib
from pathlib import Path

import h5py
import joblib
import numpy as np

SPLITS = ("train", "val", "test")
BLOCK = 131072                         # rows per processing block (memory)
GZ = dict(compression="gzip", compression_opts=9, shuffle=True)


def _s(v):
    return v.decode() if isinstance(v, bytes) else str(v)


def checksum(ds) -> str:
    """SHA-256 of the C-order bytes of a dataset (blockwise for large arrays)."""
    h = hashlib.sha256()
    if ds.shape == () or ds.ndim == 0:
        v = ds[()]
        h.update(v if isinstance(v, bytes) else np.asarray(v).tobytes())
    else:
        for i in range(0, ds.shape[0], BLOCK):
            h.update(np.ascontiguousarray(ds[i:i + BLOCK]).tobytes())
    return h.hexdigest()


def bit_delta(y):
    """float32 (n, m) -> uint32 differences of the bit patterns along axis 1 (first column = bit pattern)."""
    u = np.ascontiguousarray(y, dtype=np.float32).view(np.uint32)
    d = u.copy()
    d[:, 1:] = u[:, 1:] - u[:, :-1]           # modular uint32 arithmetic: exactly invertible by cumsum
    return d


def qfls_from_ends(lo, hi, m):
    return np.linspace(lo.astype(np.float64), hi.astype(np.float64), m, axis=1).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--scalers", nargs=2, metavar=("PARAM", "OUTPUT"),
                    help="scaler files (default: <src stem>_param_scaler.joblib / _output_scaler.joblib, "
                         "else the paths stored in the HDF5)")
    a = ap.parse_args()
    src, out = Path(a.src), Path(a.out)
    with h5py.File(src, "r") as hs:
        if a.scalers:
            sp, so = map(Path, a.scalers)
        else:
            sp = src.with_name(src.stem + "_param_scaler.joblib")
            so = src.with_name(src.stem + "_output_scaler.joblib")
            if not (sp.exists() and so.exists()):
                sp, so = Path(_s(hs["param_scaler_path"][()])), Path(_s(hs["output_scaler_path"][()]))
        p_scaler, o_scaler = joblib.load(sp), joblib.load(so)
        derived = {f"X_{s}" for s in SPLITS if f"X_{s}" in hs and f"X_{s}_raw" in hs} | \
                  {f"y_{s}" for s in SPLITS if f"y_{s}" in hs and f"y_{s}_raw" in hs} | \
                  {f"qfls_{s}" for s in SPLITS if f"qfls_{s}" in hs}
        delta = {f"y_{s}_raw" for s in SPLITS if f"y_{s}_raw" in hs}

        with h5py.File(out, "w") as ho:
            for k, v in hs.attrs.items():
                ho.attrs[k] = v
            ho.attrs["compact_format"] = 1
            ho.attrs["compact_source"] = str(src.name)
            ho.attrs["dataset_order"] = list(hs.keys())
            meta = ho.create_group("_compact")
            sums = meta.create_group("sha256")
            info = meta.create_group("datasets")
            meta["param_scaler.joblib"] = np.frombuffer(sp.read_bytes(), dtype=np.uint8)
            meta["output_scaler.joblib"] = np.frombuffer(so.read_bytes(), dtype=np.uint8)
            for k in hs:
                ds = hs[k]
                sums.attrs[k] = checksum(ds)
                info.attrs[k] = f"{ds.dtype.str}|{','.join(map(str, ds.shape))}"
                if k in derived or k in delta:
                    continue
                if ds.ndim and ds.size * ds.dtype.itemsize > 1 << 20:
                    ho.create_dataset(k, data=ds[()], chunks=True, **GZ)
                else:
                    hs.copy(k, ho)

            for k in sorted(delta):              # curves: bit deltas along the QFLS axis
                ds = hs[k]
                o = ho.create_dataset(f"{k}__bitdelta", shape=ds.shape, dtype=np.uint32,
                                      chunks=(min(4096, ds.shape[0]), ds.shape[1]), **GZ)
                for i in range(0, ds.shape[0], BLOCK):
                    o[i:i + BLOCK] = bit_delta(ds[i:i + BLOCK])

            for k in sorted(derived):            # recomputable arrays: only the correction to the recomputation
                ds = hs[k]
                n, m = ds.shape
                if k.startswith("qfls_"):
                    lo = ho.create_dataset(f"{k}__first", data=ds[:, 0], **GZ)
                    hi = ho.create_dataset(f"{k}__last", data=ds[:, -1], **GZ)
                corr = np.zeros((n, m), np.int64)
                for i in range(0, n, BLOCK):
                    orig = ds[i:i + BLOCK]
                    if k.startswith("qfls_"):
                        rec = qfls_from_ends(lo[i:i + BLOCK], hi[i:i + BLOCK], m)
                    else:
                        raw = hs[f"{k}_raw"][i:i + BLOCK]
                        sc = p_scaler if k.startswith("X_") else o_scaler
                        rec = sc.transform(raw).astype(np.float32)
                    corr[i:i + BLOCK] = (orig.view(np.int32).astype(np.int64) - rec.view(np.int32).astype(np.int64))
                nz = int(np.count_nonzero(corr))
                if nz:
                    dt = np.int8 if np.abs(corr).max() < 128 else np.int64
                    ho.create_dataset(f"{k}__correction", data=corr.astype(dt), chunks=(min(4096, n), m), **GZ)
                ho.attrs[f"{k}__nonzero_corrections"] = nz
                print(f"  {k}: recomputed, {nz} of {n * m} values corrected")
    print(f"wrote {out}: {out.stat().st_size / 1e9:.2f} GB (source {src.stat().st_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
