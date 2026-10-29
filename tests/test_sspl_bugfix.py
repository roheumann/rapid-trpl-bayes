"""
Bug 1.1 (REVISION_PLAN.md): the ss-PL interpolation of the original grid
pipeline (03_sspl_errors.py) used the bracket of row 0 for the whole chunk.
The fixed version is kept in tests/legacy_reference.py (the regression tests
compare against it); interpolating a chunk at once must give exactly the same
result as interpolating one row at a time (and the same as np.interp).
"""
import numpy as np


def _three_param_sets(m):
    Eg, NC, NV, kT = 1.625, 2.21359e18, 2.21359e18, 8.617e-5 * 300
    ni2 = m._ni2(NC, NV, Eg, kT)
    n_sw = m._n_sweep(1e8, NV, 80)
    taup1 = np.array([1e-7, 1e-9, 1e-5])
    M = len(taup1)
    tp = [dict(Et=np.full(M, 1.45), Nt=np.full(M, 1e16), taun=np.full(M, 1e-7), taup=taup1),
          dict(Et=np.full(M, 1.0), Nt=np.full(M, 1e14), taun=np.full(M, 1e-6), taup=np.full(M, 1e-6))]
    q, plqy, G = m._sspl_numeric(tp, np.full(M, 1e-11), n_sw, NC, NV, Eg, kT, ni2)
    return q, plqy, G


def test_interp_on_logG_rowwise(legacy_03):
    m = legacy_03
    q, plqy, G = _three_param_sets(m)
    G_exp = np.logspace(19, 23, 9)
    batched = m._interp_on_logG(G, q, G_exp)
    rowwise = np.vstack([m._interp_on_logG(G[i:i + 1], q[i:i + 1], G_exp) for i in range(len(q))])
    assert np.array_equal(np.isnan(batched), np.isnan(rowwise))
    assert np.nanmax(np.abs(batched - rowwise)) == 0.0
    ref = np.vstack([np.interp(np.log10(G_exp), np.log10(G[i]), q[i], left=np.nan, right=np.nan)
                     for i in range(len(q))])
    assert np.nanmax(np.abs(batched - ref)) < 1e-12


def test_plqy_vs_qfls_rowwise(legacy_03):
    m = legacy_03
    q, plqy, _ = _three_param_sets(m)
    qe = np.linspace(1.0, 1.3, 7)
    lp = np.log10(np.full(7, 0.05))
    batched = m._sspl_error(q, plqy, qe, lp)
    rowwise = np.array([m._sspl_error(q[i:i + 1], plqy[i:i + 1], qe, lp)[0] for i in range(len(q))])
    assert np.allclose(batched, rowwise, rtol=0, atol=0, equal_nan=True)
