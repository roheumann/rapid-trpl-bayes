"""
Frozen reference implementations from the ORIGINAL SUBMISSION's grid pipeline
(uncertainty_analysis/02_trpl_errors.py and 03_sspl_errors.py, commit be54a74,
with the bug fix 1.1 of REVISION_PLAN.md applied to the ss-PL interpolation).

The legacy pipeline itself has been removed from the repository; these
functions are kept only so that the regression tests can show that the new
TensorFlow code reproduces the old error definitions.  Do not use them for
analyses.
"""
import numpy as np
import tensorflow as tf
from scipy.integrate import trapezoid


def _build_nn_inputs(chunk_df, num_traps, Eg, npulse):
    N = len(chunk_df)
    n_feat = 3 + 4 * num_traps
    x = np.empty((N, n_feat), dtype=np.float64)
    x[:, 0] = np.log10(npulse)
    x[:, 1] = Eg
    x[:, 2] = np.log10(chunk_df["krad_cm3s"].values)
    for k in range(1, num_traps + 1):
        col = 3 + 4 * (k - 1)
        x[:, col    ] = np.log10(chunk_df[f"Nt_{k}_cm-3"].values)
        x[:, col + 1] = chunk_df[f"Et_{k}_eV"].values / Eg
        x[:, col + 2] = np.log10(chunk_df[f"tau_n_{k}_s"].values)
        x[:, col + 3] = np.log10(chunk_df[f"tau_p_{k}_s"].values)
    return x


def _predict_gpu(x_nn, model, input_scaler, output_scaler, batch_size):
    x_sc = input_scaler.transform(x_nn).astype(np.float32)
    chunks = []
    for s in range(0, len(x_sc), batch_size):
        y = model(tf.constant(x_sc[s: s + batch_size]), training=False)
        chunks.append(y.numpy())
    log_tau   = output_scaler.inverse_transform(np.concatenate(chunks, axis=0))
    tau_batch = 10.0 ** log_tau
    valid     = np.all(np.isfinite(tau_batch) & (tau_batch > 0), axis=1)
    return tau_batch, valid


def _compute_errors(tau_batch, valid_mask, qfls_nn, qfls_exp, tau_exp):
    N = len(tau_batch)
    nan_row = np.full(N, np.nan)
    err_mse  = nan_row.copy()
    err_imse = nan_row.copy()

    if not valid_mask.any():
        return err_mse, err_imse

    tau_v  = tau_batch[valid_mask]
    idx_v  = np.where(valid_mask)[0]

    sort_idx    = np.argsort(qfls_nn)
    qfls_sorted = qfls_nn[sort_idx]
    tau_sorted  = tau_v[:, sort_idx]        # (M, n_qfls)

    # Interpolate simulated tau to experimental QFLS points
    idx_b   = np.searchsorted(qfls_sorted, qfls_exp) - 1
    in_rng  = (idx_b >= 0) & (idx_b < len(qfls_sorted) - 1)
    idx_s   = np.clip(idx_b, 0, len(qfls_sorted) - 2)

    x0 = qfls_sorted[idx_s];  x1 = qfls_sorted[idx_s + 1]
    w  = np.where(in_rng, (qfls_exp - x0) / (x1 - x0 + 1e-30), 0.0)

    #linear interpolation
    tau_fit  = tau_sorted[:, idx_s] * (1.0 - w) + tau_sorted[:, idx_s + 1] * w
    tau_fit[:, ~in_rng] = np.nan

    log_res = np.log10(tau_fit) - np.log10(tau_exp)   # (M, n_exp)

    valid_pts = np.isfinite(log_res)
    n_valid   = valid_pts.sum(axis=1)                  # (M,)
    good      = n_valid > 0

    # MSE in log10 space
    res2 = np.where(valid_pts, log_res ** 2, 0.0)
    mse  = np.where(good, res2.sum(axis=1) / np.maximum(n_valid, 1), np.nan)
    err_mse[idx_v] = mse

    # Integral MSE: trapezoid / QFLS range  (weights by QFLS spacing)
    qfls_exp_v = np.where(in_rng, qfls_exp, np.nan)
    abs_res    = np.where(valid_pts, np.abs(log_res), np.nan)
    q_range    = np.nanmax(qfls_exp_v) - np.nanmin(qfls_exp_v)
    if q_range > 0:
        imse_vals = -trapezoid(abs_res, qfls_exp, axis=1) / q_range
        err_imse[idx_v] = np.where(good, imse_vals, np.nan)

    return err_mse, err_imse


def _ni2(NC, NV, Eg, kT):
    return NC * NV * np.exp(-Eg / kT)


def _n_sweep(lo, hi, n):
    return np.logspace(np.log10(lo), np.log10(hi), n)


def _sspl_analytic(Et, Nt, taun, taup, krad, n_sw, NC, NV, Eg, kT, ni2_val):
    """
    Vectorised for M grid points × N_N sweep points.
    Et, Nt, taun, taup, krad: (M,)
    n_sw:  (N_N,)
    Returns qfls_sim (M, N_N), plqy_sim (M, N_N).
    """
    Et_c   = Et  [:, None];  Nt_c   = Nt  [:, None]
    taun_c = taun[:, None];  taup_c = taup[:, None];  krad_c = krad[:, None]
    n = n_sw[None, :]

    n1 = NC * np.exp((Et_c - Eg) / kT)
    p1 = NV * np.exp(-Et_c / kT)

    aa  =  taun_c
    bb  =  taup_c * n1  + (taun_c + taup_c) * n + taun_c * p1
    cc  = -Nt_c   * (taup_c * n + taun_c * p1)
    disc = np.maximum(bb**2 - 4.0 * aa * cc, 0.0)
    nt  = (-bb + np.sqrt(disc)) / (2.0 * aa)
    nt  = np.clip(nt, 0.0, Nt_c)

    p = n + nt

    R_rad = krad_c * (n * p - ni2_val)
    R_srh = (n * p - ni2_val) / ((n + n1) * taup_c + (p + p1) * taun_c)
    R_tot = R_rad + R_srh

    plqy_sim = R_rad / R_tot
    qfls_sim = kT * np.log(n * p / ni2_val)
    return qfls_sim, plqy_sim, R_tot


def _sspl_numeric(trap_params, krad, n_sw, NC, NV, Eg, kT, ni2_val,
                  max_iter=60, rtol=1e-8):
    """
    trap_params: list of dicts  {Et:(M,), Nt:(M,), taun:(M,), taup:(M,)}
    krad: (M,)
    n_sw: (N_N,)
    Returns qfls_sim (M, N_N), plqy_sim (M, N_N).
    """
    n = n_sw[None, :]   # (1, N_N)
    p = n.copy() * np.ones((len(krad), 1))   # (M, N_N)  initial guess

    n1_list = []
    p1_list = []
    for t in trap_params:
        Et_c = t["Et"][:, None]
        n1_list.append(NC * np.exp((Et_c - Eg) / kT))
        p1_list.append(NV * np.exp(-Et_c / kT))

    for _ in range(max_iter):
        p_old   = p
        nt_sum  = np.zeros_like(p)
        for t, n1, p1 in zip(trap_params, n1_list, p1_list):
            Nt_c   = t["Nt"][:, None]
            taun_c = t["taun"][:, None]
            taup_c = t["taup"][:, None]
            denom = taup_c * (n + n1) + taun_c * (p + p1)
            denom = np.where(denom > 0, denom, 1e-300)
            nt_k  = np.clip(Nt_c * (taup_c * n + taun_c * p1) / denom, 0.0, Nt_c)
            nt_sum += nt_k
        p = n + nt_sum
        if np.max(np.abs(p - p_old) / (p_old + 1e-300)) < rtol:
            break

    krad_c = krad[:, None]
    R_rad  = krad_c * (n * p - ni2_val)
    R_srh  = np.zeros_like(p)
    for t, n1, p1 in zip(trap_params, n1_list, p1_list):
        taun_c = t["taun"][:, None];  taup_c = t["taup"][:, None]
        R_srh += (n * p -ni2_val) / ((n + n1) * taup_c + (p + p1) * taun_c)

    R_tot    = R_rad + R_srh
    plqy_sim = R_rad / R_tot
    qfls_sim = kT * np.log(n * p / ni2_val)
    return qfls_sim, plqy_sim, R_tot


def _interp_rows(x_sim, y_sim, x_exp):
    """
    Linearly interpolate every row of y_sim(x_sim) at the experimental x_exp.

    x_sim, y_sim : (M, N_N)  one simulated curve per grid point (row)
    x_exp        : (N_exp,)  experimental abscissae
    Returns y_fit (M, N_exp) with NaN where x_exp lies outside a row's range.

    BUG FIX (REVISION_PLAN.md, finding 1.1): the sort order and the
    interpolation bracket used to be taken from row 0 of the chunk and applied
    to every row.  G(n) and QFLS(n) shift by orders of magnitude between
    parameter sets, so all rows except the first were interpolated in the
    wrong interval (up to ~65 meV error in QFLS).  Sorting and bracketing are
    now done for each row separately.
    """
    n_n = x_sim.shape[1]

    # Per-row sort (a no-op for monotonic curves, but cheap insurance)
    order    = np.argsort(x_sim, axis=1)
    x_sorted = np.take_along_axis(x_sim, order, axis=1)
    y_sorted = np.take_along_axis(y_sim, order, axis=1)

    # Per-row bracket: idx[m, i] = number of x_sorted[m, :] < x_exp[i], minus 1
    # (identical to np.searchsorted(x_sorted[m], x_exp) - 1 for every row m)
    idx    = np.sum(x_sorted[:, :, None] < x_exp[None, None, :], axis=1) - 1
    in_rng = (idx >= 0) & (idx < n_n - 1)          # (M, N_exp), row-specific
    idx_s  = np.clip(idx, 0, n_n - 2)

    x0 = np.take_along_axis(x_sorted, idx_s, axis=1)
    x1 = np.take_along_axis(x_sorted, idx_s + 1, axis=1)
    dx = np.where(np.abs(x1 - x0) > 1e-30, x1 - x0, 1e-30)
    w  = np.clip((x_exp[None, :] - x0) / dx, 0.0, 1.0)

    y0 = np.take_along_axis(y_sorted, idx_s, axis=1)
    y1 = np.take_along_axis(y_sorted, idx_s + 1, axis=1)
    y_fit = y0 + w * (y1 - y0)
    y_fit[~in_rng] = np.nan
    return y_fit


def _sspl_error(qfls_sim, plqy_sim, qfls_exp, log10_plqy_exp):
    """
    qfls_sim, plqy_sim: (M, N_N)
    Returns err_mse: (M,)
    """
    log10_p_sim = np.log10(np.maximum(plqy_sim, 1e-300))
    log10_fit   = _interp_rows(qfls_sim, log10_p_sim, np.asarray(qfls_exp, float))

    res2 = (log10_fit - log10_plqy_exp) ** 2
    with np.errstate(all="ignore"):
        return np.nanmean(res2, axis=1)


def _interp_on_logG(G_sim, y_sim, G_exp):
    """
    Interpolate y_sim (M, N_N) at experimental G_exp points using a log10(G)
    x-axis.  Returns y_fit (M, N_exp) with NaN where G_exp is out of range.
    Sorting and bracketing are done per row (see _interp_rows, bug fix 1.1).
    """
    logG_sim = np.log10(np.maximum(G_sim, 1e-300))
    logG_exp = np.log10(np.maximum(np.asarray(G_exp, float), 1e-300))
    return _interp_rows(logG_sim, y_sim, logG_exp)
