"Author Robin Heumann 22/01/2026"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import time
import configparser
import os
import corner
import joblib
from scipy.interpolate import PchipInterpolator
from pymoo.algorithms.soo.nonconvex.cmaes import CMAES
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from trpl_module import solve_transient
from scipy.integrate import trapezoid
from constants import NC, NV, VT, DECAY_MAGNITUDE, QFLS_STEPS


from pymoo.core.termination import Termination


def fit_trpl_multitrap(df_trpl, Eg, npulse, num_traps=1, **kwargs):
    """
    CMA-ES fitting of a multi-trap SRH model to a single TRPL curve (ODE solver only).

    This is a TRPL-only predecessor to :func:`sstrpl_fitting.fit_multitrap`.
    It fits only the TRPL differential-lifetime data using the ODE transient
    solver.  For simultaneous SSPL+TRPL fitting, or for neural-network
    surrogate fitting, use :func:`sstrpl_fitting.fit_multitrap` instead.

    The objective function minimises the area-normalised integral of
    |log10(τ_sim) − log10(τ_data)| over the QFLS axis.  The optimiser uses
    a unit-hypercube normalisation: Et is linear; Nt, τ_n, τ_p, and krad are
    log10-scaled.

    Parameters
    ----------
    df_trpl : pd.DataFrame
        Experimental TRPL data.  Required columns:
        ``'qfls'`` [eV] — quasi-Fermi level splitting, and
        ``'tau_diff'`` [s] — differential carrier lifetime.
        Rows with non-finite or non-positive values are dropped automatically.
    Eg : float
        Bandgap [eV].
    npulse : float
        Photoexcited carrier density at t = 0 [cm⁻³].  Used as the initial
        condition for the transient ODE.
    num_traps : int, optional
        Number of SRH trap levels to include (default 1).
    **kwargs : optional
        ``Nc``, ``Nv`` : float — effective DOS [cm⁻³] (default 2.2e18).
        ``T`` : float — temperature [K] (default 300).
        ``sigma`` : float — CMA-ES initial step size in normalised space (default 0.25).
        ``maxfevals`` : int — maximum objective evaluations per trial (default 50 000).
        ``trials`` : int — number of independent CMA-ES restarts; best is kept (default 3).
        ``Et_bounds`` : (lo, hi) — trap energy search range [eV].
        ``Nt_bounds`` : (lo, hi) — trap density search range [cm⁻³].
        ``tau_n_bounds`` : (lo, hi) — electron lifetime search range [s].
        ``tau_p_bounds`` : (lo, hi) — hole lifetime search range [s].
        ``krad_bounds`` : (lo, hi) — radiative rate search range [cm³/s].
        ``x0_multitrap`` : list — manual initial guess, length 4·num_traps + 1,
            layout [Et_0, Nt_0, tau_n_0, tau_p_0, …, krad].
        ``tspan`` : [t_start, t_end] — ODE integration span [s] (default [1e-12, 1e5]).
        ``photodoping`` : list of int — 0-based trap indices for which
            Nt > n1 = Nc·exp(−(Eg−Et)/kT) is enforced as a penalty.

    Returns
    -------
    dict
        Fitting results with the following keys:
        ``'params'`` (DataFrame — fitted trap parameters),
        ``'krad_cm3s'`` (float), ``'trpl_sim'`` (DataFrame — full ODE simulation),
        ``'trpl_fit'`` (DataFrame — data vs fit on experimental QFLS grid),
        ``'error_total'``, ``'error_trpl'``, ``'n_evals'``, ``'n_gens'``,
        ``'solve_time'``, ``'history'``, ``'x_opt'``, ``'x0'``,
        ``'x0_unnormalized'``, ``'bounds'``.
    """
    
    # Default parameters
    Nc = kwargs.get('Nc', 2.2e18)
    Nv = kwargs.get('Nv', 2.2e18)
    T = kwargs.get('T', 300)
    x0_multitrap = kwargs.get('x0_multitrap', None)
    absorb_eff = kwargs.get('absorb_eff', 1.0)
    sigma = kwargs.get('sigma', 0.25)
    maxfevals = kwargs.get('maxfevals', 50000)
    
    # Bounds
    Et_bounds = kwargs.get('Et_bounds', (0.5*Eg, Eg))
    Nt_bounds = kwargs.get('Nt_bounds', (1e12, 1e20))
    tau_n_bounds = kwargs.get('tau_n_bounds', (1e-11, 1e-4))
    tau_p_bounds = kwargs.get('tau_p_bounds', (1e-11, 1e-4))
    krad_bounds = kwargs.get('krad_bounds', (1e-12, 1e-9))
    
    w_trpl = kwargs.get('w_trpl', 1.0)

    # List of 0-based trap indices for which Nt > n1 = Nc*exp(-Et/kT) is enforced
    # as a pymoo inequality constraint (G[j] = n1[j] - Nt[j] <= 0).
    # Pass a list, e.g. photodoping=[0] or photodoping=[0, 1].  [] disables.
    _pd_raw = kwargs.get('photodoping', [])
    if isinstance(_pd_raw, (int, np.integer)):
        _pd_raw = [int(_pd_raw)]
    photodoping_traps = [int(t) for t in _pd_raw]
    for _ti in photodoping_traps:
        if not (0 <= _ti < num_traps):
            raise ValueError(
                f"photodoping trap index {_ti} out of range [0, {num_traps - 1}]"
            )

    n_fits = kwargs.get('trials', 3)
    
    qfls_data = df_trpl['qfls'].values
    tau_data = df_trpl['tau_diff'].values
    
    
    mask_trpl = np.isfinite(qfls_data) & np.isfinite(tau_data) & (tau_data > 0)
    qfls_data = qfls_data[mask_trpl]
    tau_data = tau_data[mask_trpl]
    
    n_trpl = len(qfls_data)
    
    print("="*70)
    print("MULTITRAP MODEL FITTING - CMA-ES OPTIMIZATION")
    print("="*70)
    print(f"Number of traps: {num_traps}")
    print(f"TRPL data points: {n_trpl}")

    print("="*70)
    
    # Parameter conversion functions
    def normalize_params(Et, Nt, tau_n, tau_p, krad_val):
        """Convert physical parameters to [0,1] space"""
        K = len(Et)
        n_var = 4*K + 1  # Always include krad
        x = np.zeros(n_var)
        
        for i in range(K):
            idx = i*4
            x[idx]   = (Et[i] - Et_bounds[0]) / (Et_bounds[1] - Et_bounds[0])
            x[idx+1] = (np.log10(Nt[i]) - np.log10(Nt_bounds[0])) / (np.log10(Nt_bounds[1]) - np.log10(Nt_bounds[0]))
            x[idx+2] = (np.log10(tau_n[i]) - np.log10(tau_n_bounds[0])) / (np.log10(tau_n_bounds[1]) - np.log10(tau_n_bounds[0]))
            x[idx+3] = (np.log10(tau_p[i]) - np.log10(tau_p_bounds[0])) / (np.log10(tau_p_bounds[1]) - np.log10(tau_p_bounds[0]))
        
        # krad is always the last parameter
        x[-1] = (np.log10(krad_val) - np.log10(krad_bounds[0])) / (np.log10(krad_bounds[1]) - np.log10(krad_bounds[0]))
        
        return np.clip(x, 0, 1)
    
    def denormalize_params(x):
        """Convert [0,1] parameters to physical space"""
        K = num_traps
        Et = np.zeros(K)
        Nt = np.zeros(K)
        tau_n = np.zeros(K)
        tau_p = np.zeros(K)
        
        for i in range(K):
            idx = i*4
            Et[i] = Et_bounds[0] + np.clip(x[idx], 0, 1) * (Et_bounds[1] - Et_bounds[0])
            Nt[i] = 10**(np.log10(Nt_bounds[0]) + np.clip(x[idx+1], 0, 1) * (np.log10(Nt_bounds[1]) - np.log10(Nt_bounds[0])))
            tau_n[i] = 10**(np.log10(tau_n_bounds[0]) + np.clip(x[idx+2], 0, 1) * (np.log10(tau_n_bounds[1]) - np.log10(tau_n_bounds[0])))
            tau_p[i] = 10**(np.log10(tau_p_bounds[0]) + np.clip(x[idx+3], 0, 1) * (np.log10(tau_p_bounds[1]) - np.log10(tau_p_bounds[0])))
        
        # krad is always the last parameter
        krad_val = 10**(np.log10(krad_bounds[0]) + np.clip(x[-1], 0, 1) * (np.log10(krad_bounds[1]) - np.log10(krad_bounds[0])))
        
        return Et, Nt, tau_n, tau_p, krad_val
    
    # Objective function
    def objective(x):
        """Calculate RMSE for both experiments"""
        try:
            Et, Nt, tau_n, tau_p, krad_val = denormalize_params(x)

            if photodoping_traps:
                _kT_eV = 8.617333e-5 * T
                constraint_penalty = 0.0
                for _ti in photodoping_traps:
                    n1_i = Nc * np.exp(-(Eg - Et[_ti]) / _kT_eV)
                    if Nt[_ti] < n1_i:
                        constraint_penalty += np.log10(n1_i / Nt[_ti]) ** 2
                if constraint_penalty > 0.0:
                    return 1e4 * constraint_penalty

            # Solve transient for TRPL
            tspan = kwargs.get('tspan', [1e-12, 1e5])
            t_arr = np.logspace(np.log10(max(tspan[0], 1e-12)), np.log10(tspan[-1]), 256)
            
            result_tr = solve_transient(t_arr, npulse, Eg, Nc, Nv, krad_val, Nt, Et, tau_n, tau_p, T, 0)
            if not isinstance(result_tr, tuple) or len(result_tr) != 14:
                return 1e8
            
            _, _, _, _, QFLS_tr, tau_tr = result_tr[:6]
            
            # Build QFLS-tau map
            mask = np.isfinite(QFLS_tr) & np.isfinite(tau_tr) & (tau_tr >= 0)
            if not np.any(mask): #all of the simulation values are invalid
                return 1e6
            
            Q_sort = np.sort(QFLS_tr[mask])
            tau_sort = tau_tr[mask][np.argsort(QFLS_tr[mask])]
            Q_unique, idx_unique = np.unique(Q_sort, return_index=True)
            tau_unique = tau_sort[idx_unique]
            
            if len(Q_unique) < 2:
                return 1e4
            
            # Interpolate TRPL
            tau_fit = PchipInterpolator(Q_unique, tau_unique, extrapolate=False)(qfls_data)
            if not np.all(np.isfinite(tau_fit)) or np.any(tau_fit < 0): #covers the case if qfls data is outside the range of the simulation or if interpolation fails
                print(f"Range of QFLS_tr: {Q_unique.min():.3f} - {Q_unique.max():.3f}")
                print(f"QFLS data range: {qfls_data.min():.3f} - {qfls_data.max():.3f}")
                print(f"Invalid tau_fit values {tau_fit}")
                return 1e2
            
            # Residuals for TRPL (log space)
            #r_trpl = (np.log10(tau_fit) - np.log10(tau_data)) * w_trpl / np.sqrt(n_trpl) 
            #rmse_trpl = np.sqrt(np.mean((np.log10(tau_fit) - np.log10(tau_data))**2))
            error_trpl =  (-1.0*trapezoid(np.abs(np.log10(tau_fit) - np.log10(tau_data)), qfls_data) / (qfls_data.max() - qfls_data.min()))
            
            # Combined RMSE
            #residuals = np.concatenate([r_sspl, r_trpl])
            #rmse = np.sqrt(np.sum(residuals**2))# / n_total)
            total_error = w_trpl * error_trpl
            #rmse = w_sspl * rmse_sspl + w_trpl * rmse_trpl
            
            return total_error
            
        except Exception as e:
            return 1e6
    
    # Define optimization problem
    class FitProblem(Problem):
        def __init__(self, n_var):
            super().__init__(
                n_var=n_var,
                n_obj=1,
                xl=np.zeros(n_var),
                xu=np.ones(n_var),
                elementwise_evaluation=False,
            )

        def _evaluate(self, X, out, *args, **kwargs):
            out["F"] = np.array([objective(x) for x in X]).reshape(-1, 1)

    from pymoo.termination.default import DefaultSingleObjectiveTermination

    termination = DefaultSingleObjectiveTermination(
        xtol=1e-6,#1e-6
        cvtol=1e-6,
        #ftol=1e-6, #1e-6
        #ftarget=5e-2,
        period=100,
        #n_max_gen=1000,
        #n_max_evals=100000
    )


    
    # Initial guess
    #Et0 = np.linspace(0.6*Eg, 0.9*Eg, num_traps)
    #Nt0 = 1e15 * np.ones(num_traps)
    #tau_n0 = 1e-7 * np.ones(num_traps)
    #tau_p0 = 1e-7 * np.ones(num_traps)
    #krad0 = krad_init
    if x0_multitrap is not None:    #if guess is provided
        if len(x0_multitrap) != num_traps * 4 + 1:
            raise ValueError(f"Initial guess x0_multitrap must have length {num_traps * 4 + 1}")
        # x0_multitrap layout: [Et_0, Nt_0, tau_n_0, tau_p_0,  Et_1, ...,  krad]
        x0_multitrap_norm = normalize_params(
            np.array([x0_multitrap[i * 4]     for i in range(num_traps)]),  # Et
            np.array([x0_multitrap[i * 4 + 1] for i in range(num_traps)]),  # Nt
            np.array([x0_multitrap[i * 4 + 2] for i in range(num_traps)]),  # tau_n
            np.array([x0_multitrap[i * 4 + 3] for i in range(num_traps)]),  # tau_p
            x0_multitrap[num_traps * 4],                                     # krad
        )

    def random_start(problem):
        return np.random.uniform(problem.xl, problem.xu)
    
    # Run optimization
    n_var = num_traps * 4 + 1  # Always include krad
    problem = FitProblem(n_var)

    res = None
    x0 = None

    for i in range(n_fits):
        print(f"\n===== Fit {i+1}/{n_fits} =====")
        if i == 0 and x0_multitrap is not None: #use user provided guess for first fit
            x0_rand = x0_multitrap_norm #user provided initial guess
        else:
            x0_rand = random_start(problem)
    
        print("\nStarting values:")
        Et_start, Nt_start, tau_n_start, tau_p_start, krad_start = denormalize_params(x0_rand)
        for i in range(num_traps):
            print(f"  Trap {i+1}: Et={Et_start[i]:.3f} eV, Nt={Nt_start[i]:.2e} cm⁻³, "
                f"τn={tau_n_start[i]:.2e} s, τp={tau_p_start[i]:.2e} s")
        print(f"  krad={krad_start:.2e} cm³/s")
        print(f"\nStarting RMSE: {objective(x0_rand):.6f}")
        print("\nOptimizing...")
        print("-"*70)
    
        algorithm = CMAES(
            x0=x0_rand,
            sigma=sigma, 
            maxfevals=maxfevals, 
            parallelize=True,
            restarts=1,
            restart_from_best=True,
            bipop=True,
            #ftarget=5e-2,
        )

        start_time = time.time()
        trial_res = minimize(
            problem, 
            algorithm, 
            termination=termination, 
            seed=43, 
            verbose=True, 
            save_history=True)
        solve_time = time.time() - start_time
        print(f"Final Error: {trial_res.F[0]:.6f}")

        if res is None or trial_res.F < res.F:
            res = trial_res #save best result
            x0 = x0_rand

    # Extract results
    x_opt = res.X
    rmse_final = res.F[0] if isinstance(res.F, np.ndarray) else res.F
    n_evals = res.algorithm.evaluator.n_eval
    n_gens = len(res.history) if hasattr(res, 'history') else 0
    
    Et_opt, Nt_opt, tau_n_opt, tau_p_opt, krad_opt = denormalize_params(x_opt)
    
    
    tspan = kwargs.get('tspan', [0, 5e-5])
    t_arr = np.logspace(np.log10(max(tspan[0], 1e-12)), np.log10(tspan[-1]), 256)
    result_tr = solve_transient(t_arr, npulse, Eg, Nc, Nv, krad_opt, Nt_opt, Et_opt, tau_n_opt, tau_p_opt, T, 0)
    (t, n_tr, p_tr, nt_tr, QFLS_tr, tau_tr, tau_tr_n, tau_tr_p, PL_tr,
            e_trapping, e_detrapping, h_trapping, h_detrapping, krad_rate) = result_tr
    
    mask = np.isfinite(QFLS_tr) & np.isfinite(tau_tr) & (tau_tr > 0)
    Q_sort = np.sort(QFLS_tr[mask])
    tau_sort = tau_tr[mask][np.argsort(QFLS_tr[mask])]
    Q_unique, idx_unique = np.unique(Q_sort, return_index=True)
    tau_unique = tau_sort[idx_unique]
    
    tau_fit = PchipInterpolator(Q_unique, tau_unique, extrapolate=False)(qfls_data)
    r_trpl = np.log10(tau_fit) - np.log10(tau_data)
    rmse_trpl = np.sqrt(np.mean(r_trpl**2))
    error_trpl =  (-1.0*trapezoid(np.abs(np.log10(tau_fit) - np.log10(tau_data)), qfls_data) / (qfls_data.max() - qfls_data.min()))
            
    error_total = w_trpl * error_trpl
    
    # Print results
    print("\n" + "="*70)
    print("OPTIMIZATION RESULTS")
    print("="*70)
    print(f"Function evaluations: {n_evals}")
    print(f"Generations: {n_gens}")
    print(f"Solve time: {solve_time:.2f} s")
    print(f"\nTotal error: {error_total:.6f}")
    print(f"TRPL error: {error_trpl:.6f}")
    print("\nFitted parameters:")
    for i in range(num_traps):
        print(f"  Trap {i+1}:")
        print(f"    Et = {Et_opt[i]:.4f} eV")
        print(f"    Nt = {Nt_opt[i]:.3e} cm⁻³")
        print(f"    τn = {tau_n_opt[i]:.3e} s")
        print(f"    τp = {tau_p_opt[i]:.3e} s")
    print(f"  krad = {krad_opt:.3e} cm³/s")
    print("="*70)
    
    # Prepare output
    results = {
        'params': pd.DataFrame({
            'trap': np.arange(1, num_traps+1),
            'Et_eV': Et_opt,
            'Nt_1/cm3': Nt_opt,
            'tau_n_s': tau_n_opt,
            'tau_p_s': tau_p_opt
        }),
        'krad_cm3s': krad_opt,
        'trpl_sim': pd.DataFrame({
            'time_tr_ns': t.flatten(),
            'pl_tr': PL_tr.flatten(),
            'qfls_tr_eV': QFLS_tr.flatten(),
            'tau_tr_s': tau_tr.flatten(),
            'tau_n_tr_s': tau_tr_n.flatten(),
            'tau_p_tr_s': tau_tr_p.flatten(),
            'n_tr_1/cm3': n_tr.flatten(),
            'p_tr_1/cm3': p_tr.flatten(),
            **{f'nt_tr_{i+1}_1/cm3': nt_tr[:, i] for i in range(num_traps)},
            **{f'e_trapping_tr_trap{i+1}_1/cm3/s': e_trapping[:, i] for i in range(num_traps)},
            **{f'e_detrapping_tr_trap{i+1}_1/cm3/s': e_detrapping[:, i] for i in range(num_traps)},
            **{f'h_trapping_tr_trap{i+1}_1/cm3/s': h_trapping[:, i] for i in range(num_traps)},
            **{f'h_detrapping_tr_trap{i+1}_1/cm3/s': h_detrapping[:, i] for i in range(num_traps)},
        }),
        
        'error_total': error_total,
        'error_trpl': error_trpl,
        'n_evals': n_evals,
        'n_gens': n_gens,
        'solve_time': solve_time,
        'history': res.history if hasattr(res, 'history') else None,
        'x_opt': res.X,  # ADD THIS - optimal normalized parameters
        'x0': x0,        # ADD THIS - initial normalized parameters (you need to save this at the start)
        'x0_unnormalized': denormalize_params(x0),  # ADD THIS - initial unnormalized parameters
        'bounds': {
            'Et_bounds': Et_bounds,
            'Nt_bounds': Nt_bounds,
            'tau_n_bounds': tau_n_bounds,
            'tau_p_bounds': tau_p_bounds,
            'krad_bounds': krad_bounds
        },

        'trpl_fit': pd.DataFrame({
            'qfls': qfls_data,
            'tau_data': tau_data,
            'tau_fit': tau_fit
        })
    }
    
    return results


def compute_qfls_network(npulse, Eg):
    """
    Compute the QFLS axis for NN output from physical parameters.

    The axis spans from the initial QFLS right after the laser pulse
    (qfls_max) down by DECAY_MAGNITUDE orders of magnitude in PL
    (qfls_min), with QFLS_STEPS uniformly-spaced points.

    Parameters
    ----------
    npulse : float   Excited carrier density [cm-3].
    Eg     : float   Bandgap [eV].

    Returns
    -------
    qfls_network : np.ndarray (QFLS_STEPS,)  QFLS axis [eV], ascending.
    """
    ni       = np.sqrt(NC * NV) * np.exp(-Eg / (2.0 * VT))
    qfls_max = 2.0 * VT * np.log(npulse / ni)
    qfls_min = qfls_max + VT * np.log(1.0 / 10**DECAY_MAGNITUDE)
    return np.linspace(qfls_min, qfls_max, QFLS_STEPS)


def load_nn_artifacts(model_path, param_scaler_path, output_scaler_path,
                      hdf5_path=None):
    """
    Load the neural network model, its input/output scalers, and — optionally
    — the canonical QFLS axis from the training HDF5 file.

    Parameters
    ----------
    model_path : str
        Path to the .keras model file.
    param_scaler_path : str
        Path to the input (parameter) scaler .joblib file (StandardScaler).
    output_scaler_path : str
        Path to the output scaler .joblib file (MinMaxScaler).
    hdf5_path : str or None
        Path to the training HDF5 file.  When provided the function reads
        ``constants`` (Nc, Nv, Vt, QFLS_STEPS, PL_DECAY_MAGNITUDE) and the
        fixed training npulse / Eg from ``lb`` to compute the canonical
        256-point QFLS axis that the network was trained on.  Pass this axis
        to ``sstrpl_fitting.fit_multitrap`` via the ``nn_qfls_axis`` kwarg so
        the NN output is always mapped to the correct QFLS grid.

    Returns
    -------
    model : tf.keras.Model
    input_scaler : sklearn scaler
    output_scaler : sklearn scaler
    qfls_axis : np.ndarray or None
        QFLS axis [eV] of length QFLS_STEPS computed from the HDF5 training
        constants, or None if hdf5_path was not supplied.
    """
    import tensorflow as tf

    model         = tf.keras.models.load_model(model_path)
    input_scaler  = joblib.load(param_scaler_path)
    output_scaler = joblib.load(output_scaler_path)

    qfls_axis = None
    if hdf5_path is not None:
        import h5py
        with h5py.File(hdf5_path, 'r') as f:
            c    = f['constants'][()]
            Vt   = float(c['Vt'])
            Nc   = float(c['Nc'])
            Nv   = float(c['Nv'])
            steps = int(c['QFLS_STEPS'])
            decay = float(c['PL_DECAY_MAGNITUDE'])
            lb   = f['lb'][:]          # lower bounds of training parameters
        # lb[0] = log10(npulse),  lb[1] = Eg_eV  (both fixed in training)
        npulse_train = 10 ** lb[0]
        Eg_train     = lb[1]
        ni       = np.sqrt(Nc * Nv) * np.exp(-Eg_train / (2.0 * Vt))
        qfls_max = 2.0 * Vt * np.log(npulse_train / ni)
        qfls_min = qfls_max + Vt * np.log(1.0 / 10 ** decay)
        qfls_axis = np.linspace(qfls_min, qfls_max, steps)
        print(f"Loaded QFLS axis from HDF5: [{qfls_min:.4f}, {qfls_max:.4f}] eV, {steps} points")

    return model, input_scaler, output_scaler, qfls_axis


def plot_fit(results, figsize=(7.5, 3.75), save_path=None):
    """
    Plot SSPL and TRPL fits side by side (legacy two-panel version).

    Produces a figure with two panels: PLQY vs generation rate (log-log) on
    the left, and τ_diff vs QFLS (semi-log y) on the right.

    .. note::
        This function expects a results dict that contains **both** ``'sspl_fit'``
        and ``'trpl_fit'`` keys — as produced by combined SSPL+TRPL fits.  For
        TRPL-only results from :func:`fit_trpl_multitrap`, or for the full
        multi-mode plot that respects ``fit_mode``, use
        :func:`sstrpl_fitting.plot_fit` instead.

    Parameters
    ----------
    results : dict
        Results dictionary containing:
        ``'sspl_fit'`` — DataFrame with columns ``'intensity'``, ``'plqy_data'``,
        ``'plqy_fit'``; and
        ``'trpl_fit'`` — DataFrame with columns ``'qfls'``, ``'tau_data'``, ``'tau_fit'``.
        Also reads ``'error_sspl'`` and ``'error_trpl'`` for annotation text.
    figsize : tuple, optional
        Figure size ``(width, height)`` in inches (default ``(7.5, 3.75)``).
    save_path : str or None, optional
        If given, saves the figure to this path at 300 dpi.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : tuple of matplotlib.axes.Axes
        ``(ax_sspl, ax_trpl)``
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # SSPL plot
    df_sspl = results['sspl_fit']
    ax1.loglog(df_sspl['intensity'], df_sspl['plqy_data'], 'o', 
               color='#2E86AB', markersize=8, label='Data', 
               markeredgecolor='white', markeredgewidth=1.5)
    ax1.loglog(df_sspl['intensity'], df_sspl['plqy_fit'], '-', 
               color='#A23B72', linewidth=2.5, label='Fit')
    ax1.set_xlabel('Generation Rate [cm⁻³ s⁻¹]', fontsize=12, fontweight='bold')
    ax1.set_ylabel('PLQY', fontsize=12, fontweight='bold')
    ax1.set_title('Steady-State PL', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Add SSPL error
    ax1.text(0.05, 0.95, f"error: {results['error_sspl']:.4f}", 
             transform=ax1.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # TRPL plot
    df_trpl = results['trpl_fit']
    ax2.semilogy(df_trpl['qfls'], df_trpl['tau_data'], 'o',
                 color='#2E86AB', markersize=8, label='Data',
                 markeredgecolor='white', markeredgewidth=1.5)
    ax2.semilogy(df_trpl['qfls'], df_trpl['tau_fit'], '-',
                 color='#A23B72', linewidth=2.5, label='Fit')
    ax2.set_xlabel('QFLS [eV]', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Differential Lifetime [s]', fontsize=12, fontweight='bold')
    ax2.set_title('Time-Resolved PL', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Add TRPL error
    ax2.text(0.05, 0.95, f"error: {results['error_trpl']:.4f}",
             transform=ax2.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    
    return fig, (ax1, ax2)


def plot_corner(results, max_points=5000, figsize=(12, 12), save_path=None, error_threshold=100):
    """
    Create corner plot of optimization history with inferno_r colormap.
    Points with errors above threshold are marked in a separate color.
    
    Parameters
    ----------
    results : dict
        Results dictionary from fit_multitrap_simple containing:
        - 'history': optimization history
        - 'params': fitted parameters DataFrame
        - 'x_opt': optimal normalized parameters
        - 'x0': initial normalized parameters
        - 'bounds': dictionary of parameter bounds
    max_points : int
        Maximum number of points to plot (subsample if needed)
    figsize : tuple
        Figure size (width, height)
    save_path : str, optional
        Path to save figure
    error_threshold : float
        Error threshold above which points are marked as outliers (default: 100)
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        Corner plot figure
    """
    if results['history'] is None:
        raise ValueError("No optimization history available")
    
    # Extract history
    history = results['history']
    X_hist = [entry.pop.get("X") for entry in history]
    F_hist = [entry.pop.get("F") for entry in history]
    
    # Flatten
    X_all = np.vstack(X_hist)
    F_all = np.vstack(F_hist).flatten()
    
    # Subsample if needed
    n_points = len(X_all)
    if n_points > max_points:
        indices = np.random.choice(n_points, max_points, replace=False)
        X_all = X_all[indices]
        F_all = F_all[indices]
        print(f"Subsampled {max_points} from {n_points} evaluations")
    
    # Separate good and bad points
    good_mask = F_all <= error_threshold
    bad_mask = ~good_mask
    
    X_good = X_all[good_mask]
    F_good = F_all[good_mask]
    X_bad = X_all[bad_mask]
    F_bad = F_all[bad_mask]
    
    n_good = np.sum(good_mask)
    n_bad = np.sum(bad_mask)
    print(f"Good points (error ≤ {error_threshold}): {n_good}")
    print(f"Bad points (error > {error_threshold}): {n_bad}")
    
    # Get parameters
    num_traps = len(results['params'])
    bounds = results['bounds']
    x_opt = results['x_opt']
    x0 = results['x0']
    
    # Convert to physical space
    def convert_to_physical(x_norm_array, bounds, num_traps):
        """Convert normalized [0,1] parameters to physical space"""
        X_physical = np.zeros_like(x_norm_array)
        
        for i in range(len(x_norm_array)):
            x_norm = x_norm_array[i]
            
            for j in range(num_traps):
                idx = j * 4
                
                # Et (linear)
                X_physical[i, idx] = bounds['Et_bounds'][0] + np.clip(x_norm[idx], 0, 1) * \
                                     (bounds['Et_bounds'][1] - bounds['Et_bounds'][0])
                
                # Nt (log)
                X_physical[i, idx+1] = 10**(np.log10(bounds['Nt_bounds'][0]) + 
                                            np.clip(x_norm[idx+1], 0, 1) * 
                                            (np.log10(bounds['Nt_bounds'][1]) - np.log10(bounds['Nt_bounds'][0])))
                
                # tau_n (log)
                X_physical[i, idx+2] = 10**(np.log10(bounds['tau_n_bounds'][0]) + 
                                            np.clip(x_norm[idx+2], 0, 1) * 
                                            (np.log10(bounds['tau_n_bounds'][1]) - np.log10(bounds['tau_n_bounds'][0])))
                
                # tau_p (log)
                X_physical[i, idx+3] = 10**(np.log10(bounds['tau_p_bounds'][0]) + 
                                            np.clip(x_norm[idx+3], 0, 1) * 
                                            (np.log10(bounds['tau_p_bounds'][1]) - np.log10(bounds['tau_p_bounds'][0])))
            
            # krad (always present)
            X_physical[i, -1] = 10**(np.log10(bounds['krad_bounds'][0]) + 
                                     np.clip(x_norm[-1], 0, 1) * 
                                     (np.log10(bounds['krad_bounds'][1]) - np.log10(bounds['krad_bounds'][0])))
        
        return X_physical
    
    X_physical_all = convert_to_physical(X_all, bounds, num_traps)
    X_physical_good = X_physical_all[good_mask]
    X_physical_bad = X_physical_all[bad_mask]
    
    # Convert best and initial points
    x_opt_physical = convert_to_physical(x_opt.reshape(1, -1), bounds, num_traps)[0]
    x0_physical = convert_to_physical(x0.reshape(1, -1), bounds, num_traps)[0]
    
    # Generate parameter names
    param_names = []
    for j in range(num_traps):
        trap_label = f"{j+1}" if num_traps > 1 else ""
        param_names.extend([
            f'$E_{{t{trap_label}}}$ [eV]',
            f'$N_{{t{trap_label}}}$ [cm$^{{-3}}$]',
            f'$\\tau_{{n{trap_label}}}$ [s]',
            f'$\\tau_{{p{trap_label}}}$ [s]'
        ])
    param_names.append('$k_{rad}$ [cm$^3$/s]')
    
    n_params = len(param_names)
    
    # Normalize objective for colormap (only good points)
    if n_good > 0:
        F_min = np.min(F_good)
        F_max = np.max(F_good)
        F_norm = (F_good - F_min) / (F_max - F_min) if F_max > F_min else np.zeros_like(F_good)
    else:
        F_min = F_max = 0
        F_norm = np.array([])
    
    # Create corner plot (using all data for range)
    fig = corner.corner(
        X_physical_all,
        labels=param_names,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_kwargs={"fontsize": 10},
        label_kwargs={"fontsize": 11},
        use_math_text=True,
        plot_datapoints=False,
        plot_density=False,
        plot_contours=False,
        bins=30,
        color='black',
        hist_kwargs={'alpha': 0.6, 'edgecolor': 'black', 'linewidth': 1.2}
    )
    
    # Get axes
    axes = np.array(fig.axes).reshape((n_params, n_params))
    
    # Add scatter plots with color
    for i in range(n_params):
        for j in range(i):
            ax = axes[i, j]
            ax.clear()
            
            # Plot bad points first (gray)
            if n_bad > 0:
                ax.scatter(X_physical_bad[:, j], X_physical_bad[:, i], 
                          c='lightgray', s=5, alpha=0.3, rasterized=True, 
                          label='Error')
            
            # Plot good points with inferno_r colormap
            if n_good > 0:
                ax.scatter(X_physical_good[:, j], X_physical_good[:, i], 
                          c=F_norm, cmap='inferno_r', s=3, alpha=0.5, rasterized=True)
            
            # Mark starting point (square)
            ax.plot(x0_physical[j], x0_physical[i],
                   's', color='lime', markersize=12, markeredgecolor='white',
                   markeredgewidth=2, zorder=1001, label='Start')
            
            # Mark best point (star)
            ax.plot(x_opt_physical[j], x_opt_physical[i],
                   '*', color='cyan', markersize=18, markeredgecolor='white',
                   markeredgewidth=2, zorder=1000, label='Best')
            
            # Labels
            if i == n_params - 1:
                ax.set_xlabel(param_names[j], fontsize=11)
            if j == 0:
                ax.set_ylabel(param_names[i], fontsize=11)
            
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Update diagonal histograms
    for i in range(n_params):
        ax = axes[i, i]
        ax.clear()
        
        # Combined histogram for range
        hist_all, bins = np.histogram(X_physical_all[:, i], bins=30)
        
        # Separate histograms for good and bad
        hist_good, _ = np.histogram(X_physical_good[:, i], bins=bins)
        hist_bad, _ = np.histogram(X_physical_bad[:, i], bins=bins)
        
        # Mean objective per bin (only for good points)
        bin_colors = np.zeros(len(bins) - 1)
        for k in range(len(bins) - 1):
            if n_good > 0:
                mask = (X_physical_good[:, i] >= bins[k]) & (X_physical_good[:, i] < bins[k+1])
                if np.any(mask):
                    bin_colors[k] = np.mean(F_norm[mask])
        
        # Plot bars
        norm = plt.Normalize(vmin=0, vmax=1)
        cmap = plt.cm.inferno_r
        
        for k in range(len(bins) - 1):
            # Bad points (gray, bottom layer)
            if hist_bad[k] > 0:
                ax.bar(bins[k], hist_bad[k], width=bins[k+1] - bins[k],
                      color='lightgray', alpha=0.5,
                      edgecolor='black', linewidth=0.5)
            
            # Good points (colored, on top)
            if hist_good[k] > 0:
                ax.bar(bins[k], hist_good[k], width=bins[k+1] - bins[k],
                      color=cmap(norm(bin_colors[k])), alpha=0.7,
                      edgecolor='black', linewidth=0.8)
        
        # Mark starting point (dashed line, lime)
        ax.axvline(x0_physical[i], color='lime', linestyle='--',
                  linewidth=2.5, zorder=1001)
        
        # Mark best point (dashed line, cyan)
        ax.axvline(x_opt_physical[i], color='cyan', linestyle='--',
                  linewidth=2.5, zorder=1000)
        
        ax.set_ylabel('Count', fontsize=10)
        if i == n_params - 1:
            ax.set_xlabel(param_names[i], fontsize=11)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Add colorbar (only for good points)
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.87, 0.15, 0.02, 0.7])
    if n_good > 0:
        cbar = plt.colorbar(
            plt.cm.ScalarMappable(norm=plt.Normalize(vmin=F_min, vmax=F_max), cmap='inferno_r'),
            cax=cbar_ax
        )
        cbar.set_label('RMSE', fontsize=12, fontweight='bold')
        cbar.ax.tick_params(labelsize=10)
    
    # Add legend in top-left corner plot
    if n_params > 1:
        legend_ax = axes[1, 0]
        
        legend_elements = [
            Line2D([0], [0], marker='s', color='w', markerfacecolor='lime', 
                   markeredgecolor='white', markeredgewidth=2, markersize=10, label='Start'),
            Line2D([0], [0], marker='*', color='w', markerfacecolor='cyan', 
                   markeredgecolor='white', markeredgewidth=2, markersize=14, label='Best Fit')
        ]
        
        if n_bad > 0:
            legend_elements.append(
                Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgray',
                       markersize=8, alpha=0.5, label=f'Error (>{error_threshold})')
            )
        
        legend_ax.legend(handles=legend_elements, loc='upper right', fontsize=9, 
                        framealpha=0.9, edgecolor='gray')
    
    # Title
    title_str = f'Optimization History\n{len(X_all)} Evaluations'
    if n_good > 0:
        title_str += f' | Best RMSE: {F_min:.6f}'
    if n_bad > 0:
        title_str += f' | {n_bad} outliers'
    
    fig.suptitle(title_str, fontsize=14, fontweight='bold', y=0.995)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Corner plot saved to {save_path}")
    
    return fig