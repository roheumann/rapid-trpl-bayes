"Author Robin Heumann 22/01/2026"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import time
import configparser
import os
import corner
import h5py
import joblib
from scipy.interpolate import PchipInterpolator
from pymoo.algorithms.soo.nonconvex.cmaes import CMAES
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from trpl_module import solve_transient
from scipy.integrate import trapezoid, cumulative_trapezoid
from constants import NC, NV, VT, DECAY_MAGNITUDE, QFLS_STEPS


from pymoo.core.termination import Termination


def fit_trpl_multitrap(df_trpl, Eg, npulse, num_traps=1, **kwargs):
    """
    CMA-ES fitting for multitrap model.
    
    Parameters
    ----------
    df_sspl : pd.DataFrame
        Steady-state data with columns ['intensity', 'plqy']
    df_trpl : pd.DataFrame
        Transient data with columns ['qfls', 'tau_diff']
    Eg : float
        Bandgap [eV]
    npulse: excited carrier density from laser pulse [photons/cm^3]
    num_traps : int
        Number of trap levels
    **kwargs : optional parameters
        Nc, Nv, T, krad_init, absorb_eff, etc.
    
    Returns
    -------
    results : dict
        Fitting results and fitted parameters
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
            super().__init__(n_var=n_var, n_obj=1, xl=np.zeros(n_var), xu=np.ones(n_var), elementwise_evaluation=False)

        def _evaluate(self, X, out, *args, **kwargs):
            out["F"] = np.array([objective(x) for x in X])

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
        x0_multitrap_norm = normalize_params(
            x0_multitrap[:num_traps], #Et
            x0_multitrap[num_traps:2*num_traps], # Nt
            x0_multitrap[2*num_traps:3*num_traps], #taun
            x0_multitrap[3*num_traps:4*num_traps], #taup
            x0_multitrap[4*num_traps] #krad
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





def compute_qfls_network(npulse, Eg, pl_decay_magnitude=None, n_steps=None):
    """
    Compute the QFLS axis for NN output from physical parameters.

    The axis spans from the initial QFLS right after the laser pulse
    (qfls_max) down to qfls_min where PL has dropped by pl_decay_magnitude
    orders of magnitude, with n_steps uniformly-spaced points.

        ni       = sqrt(NC * NV) * exp(-Eg / (2*VT))
        qfls_max = 2 * VT * log(npulse / ni)
        qfls_min = qfls_max - pl_decay_magnitude * VT * ln(10)

    Parameters
    ----------
    npulse             : float  Excited carrier density [cm-3].
    Eg                 : float  Bandgap [eV].
    pl_decay_magnitude : float  Orders of magnitude of PL decay (default: DECAY_MAGNITUDE).
    n_steps            : int    Number of QFLS grid points (default: QFLS_STEPS).

    Returns
    -------
    qfls_network : np.ndarray (n_steps,)  QFLS axis [eV], ascending.
    """
    dm    = pl_decay_magnitude if pl_decay_magnitude is not None else DECAY_MAGNITUDE
    steps = n_steps            if n_steps            is not None else QFLS_STEPS
    ni       = np.sqrt(NC * NV) * np.exp(-Eg / (2.0 * VT))
    qfls_max = 2.0 * VT * np.log(npulse / ni)
    qfls_min = qfls_max - dm * VT * np.log(10.0)
    return np.linspace(qfls_min, qfls_max, steps)


def load_nn_artifacts(model_path, param_scaler_path, output_scaler_path,
                      training_hdf5_path=None):
    """
    Load the neural network model and its input/output scalers.

    Parameters
    ----------
    model_path : str
        Path to the .keras model file.
    param_scaler_path : str
        Path to the input (parameter) scaler .joblib file (StandardScaler).
    output_scaler_path : str
        Path to the output scaler .joblib file (MinMaxScaler).
    training_hdf5_path : str, optional
        Path to the training HDF5 (the *_training.hdf5 produced by
        prepare_training_data.py).  When given, the QFLS axis is read
        directly from qfls_train[0] and stored as model._qfls_axis so
        that fit_trpl_multitrap_network uses the exact axis the NN was
        trained on instead of the analytic compute_qfls_network estimate.

    Returns
    -------
    model : tf.keras.Model
    input_scaler : sklearn scaler
    output_scaler : sklearn scaler
    """
    import tensorflow as tf

    # Limit GPU memory to 50% of available, growing only as needed.
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                # Query total memory via nvidia-smi, fall back to 24 GB if unavailable
                try:
                    import subprocess, re
                    smi = subprocess.check_output(
                        ['nvidia-smi', '--query-gpu=memory.total',
                         '--format=csv,noheader,nounits'],
                        text=True
                    ).strip().split('\n')
                    # CUDA_VISIBLE_DEVICES remaps physical indices — take first visible
                    total_mb = int(smi[0].strip())
                except Exception:
                    total_mb = 24576  # 24 GB fallback
                limit_mb = total_mb // 2
                tf.config.set_logical_device_configuration(
                    gpu,
                    [tf.config.LogicalDeviceConfiguration(memory_limit=limit_mb)]
                )
                print(f"GPU memory capped at {limit_mb} MiB / {total_mb} MiB (50%)")
        except RuntimeError:
            # Logical devices must be set before GPUs are initialised — already done
            pass

    model         = tf.keras.models.load_model(model_path)
    input_scaler  = joblib.load(param_scaler_path)
    output_scaler = joblib.load(output_scaler_path)

    # Compile a tf.function so inference runs on-device (GPU) without
    # Python overhead per call.  The first call triggers tracing/JIT.
    @tf.function(reduce_retracing=True)
    def _infer(x):
        return model(x, training=False)

    model._infer_fn = _infer

    # Load PL_DECAY_MAGNITUDE and QFLS_STEPS from the training HDF5 so that
    # compute_qfls_network reproduces the exact QFLS axis used during training.
    if training_hdf5_path is not None:
        import h5py
        with h5py.File(training_hdf5_path, 'r') as hf:
            c = hf['constants']
            model._pl_decay_magnitude = float(c['PL_DECAY_MAGNITUDE'][()])
            model._qfls_steps         = int(c['QFLS_STEPS'][()])
        print(f"Training constants loaded: PL_DECAY_MAGNITUDE={model._pl_decay_magnitude}, "
              f"QFLS_STEPS={model._qfls_steps}")
    else:
        model._pl_decay_magnitude = None
        model._qfls_steps         = None

    return model, input_scaler, output_scaler


def fit_trpl_multitrap_network(df_trpl, Eg, npulse,
                               model, input_scaler, output_scaler,
                               num_traps=1, **kwargs):
    """
    CMA-ES fitting for the 1-trap multitrap model using a neural-network surrogate.

    The NN takes physical parameters as input and directly predicts the
    differential lifetime tau_diff as a function of QFLS, replacing the ODE
    solver used in fit_trpl_multitrap.

    Parameters
    ----------
    df_trpl : pd.DataFrame
        Transient data with columns ['qfls', 'tau_diff'].
    Eg : float
        Bandgap [eV].  Must match the value used to train the NN.
    npulse : float
        Excited carrier density from laser pulse [cm^-3].
        Must match the value used to train the NN.
    model : tf.keras.Model
        Trained Keras model (loaded with load_nn_artifacts).
    input_scaler : sklearn scaler
        StandardScaler for the 7 NN input features.
    output_scaler : sklearn scaler
        MinMaxScaler for the 256 NN output values (log10 tau_diff).
    num_traps : int
        Number of trap levels.
    **kwargs : optional
        Et_bounds, Nt_bounds, tau_n_bounds, tau_p_bounds, krad_bounds,
        x0_multitrap, sigma, maxfevals, trials, w_trpl,
        error_metric ('integral' | 'mse' | 'mae', default 'integral').

    Returns
    -------
    results : dict
        Fitting results and fitted parameters. Keys:
        'params', 'krad_cm3s', 'trpl_sim', 'trpl_fit',
        'error_total', 'error_integral_mse', 'error_mse', 'error_mae',
        'error_metric', 'n_evals', 'n_gens', 'solve_time',
        'history', 'x_opt', 'x0', 'x0_unnormalized', 'bounds'.

    Notes
    -----
    NN input order: [log10(n_pulse), Eg_eV, log10(krad),
                     log10(Nt_1), Et_1/Eg, log10(tau_n_1), log10(tau_p_1)]
    NN output: 256 values of log10(tau_diff) [s] on the qfls_network axis,
               after inverse-transforming the MinMaxScaler.
    """

    # ------------------------------------------------------------------ #
    # Default parameters
    # ------------------------------------------------------------------ #
    x0_multitrap      = kwargs.get('x0_multitrap', None)
    sigma             = kwargs.get('sigma', 0.25)
    maxfevals         = kwargs.get('maxfevals', 50000)
    restarts          = kwargs.get('restarts', 1)
    error_metric      = kwargs.get('error_metric', 'integral')   # 'mse' | 'mae' | 'integral'
    w_trpl            = kwargs.get('w_trpl', 1.0)
    n_fits            = kwargs.get('trials', 3)

    # Bounds (kept identical to fit_trpl_multitrap for compatibility)
    Et_bounds    = kwargs.get('Et_bounds',    (0.5 * Eg, Eg))
    Nt_bounds    = kwargs.get('Nt_bounds',    (1e12, 1e20))
    tau_n_bounds = kwargs.get('tau_n_bounds', (1e-12, 1e-4))
    tau_p_bounds = kwargs.get('tau_p_bounds', (1e-12, 1e-4))
    krad_bounds  = kwargs.get('krad_bounds',  (1e-12, 1e-9))

    if error_metric not in ('mse', 'mae', 'integral'):
        raise ValueError(
            f"Unknown error_metric '{error_metric}'. "
            "Choose from 'mse', 'mae', or 'integral'."
        )

    # ------------------------------------------------------------------ #
    # Prepare experimental data
    # ------------------------------------------------------------------ #
    qfls_data = df_trpl['qfls'].values
    tau_data  = df_trpl['tau_diff'].values

    mask_trpl = np.isfinite(qfls_data) & np.isfinite(tau_data) & (tau_data > 0)
    qfls_data = qfls_data[mask_trpl]
    tau_data  = tau_data[mask_trpl]
    n_trpl    = len(qfls_data)

    # Compute the QFLS axis using constants from the training HDF5 if available.
    qfls_nn = compute_qfls_network(
        npulse, Eg,
        pl_decay_magnitude=getattr(model, '_pl_decay_magnitude', None),
        n_steps=getattr(model, '_qfls_steps', None),
    )
    sort_idx       = np.argsort(qfls_nn)
    qfls_nn_sorted = qfls_nn[sort_idx]

    print("=" * 70)
    print("MULTITRAP MODEL NN FITTING - CMA-ES OPTIMIZATION")
    print("=" * 70)
    print(f"Number of traps : {num_traps}")
    print(f"TRPL data points: {n_trpl}")
    print(f"Error metric    : {error_metric}")
    print("=" * 70)

    # ------------------------------------------------------------------ #
    # Precomputed bound constants (avoids recomputing log-bounds each call)
    # ------------------------------------------------------------------ #
    K = num_traps
    _Et_idx   = np.arange(0, 4 * K, 4)
    _Nt_idx   = np.arange(1, 4 * K, 4)
    _taun_idx = np.arange(2, 4 * K, 4)
    _taup_idx = np.arange(3, 4 * K, 4)
    # krad always at index 4*K (last element)

    _dEt       = Et_bounds[1]    - Et_bounds[0]
    _log_Nt0   = np.log10(Nt_bounds[0]);    _dlog_Nt   = np.log10(Nt_bounds[1])    - _log_Nt0
    _log_taun0 = np.log10(tau_n_bounds[0]); _dlog_taun = np.log10(tau_n_bounds[1]) - _log_taun0
    _log_taup0 = np.log10(tau_p_bounds[0]); _dlog_taup = np.log10(tau_p_bounds[1]) - _log_taup0
    _log_krad0 = np.log10(krad_bounds[0]);  _dlog_krad = np.log10(krad_bounds[1])  - _log_krad0

    # ------------------------------------------------------------------ #
    # Fixed parameters — reduce optimization dimension
    # ------------------------------------------------------------------ #
    # fixed_params: dict mapping param name → physical value.
    # Supported names: 'Et','Nt','taun','taup' (apply to all traps),
    #                  'Et_1','Nt_2', ... (per-trap), 'krad'.
    fixed_params = kwargs.get('fixed_params', {})

    # Build name → (list of full-vector indices, normalize-fn)
    def _norm_Et(v):   return (v - Et_bounds[0]) / _dEt
    def _norm_Nt(v):   return (np.log10(v) - _log_Nt0)   / _dlog_Nt
    def _norm_tn(v):   return (np.log10(v) - _log_taun0)  / _dlog_taun
    def _norm_tp(v):   return (np.log10(v) - _log_taup0)  / _dlog_taup
    def _norm_kr(v):   return (np.log10(v) - _log_krad0)  / _dlog_krad

    _name_map = {'krad': ([4 * K], _norm_kr)}
    for _k in range(K):
        _name_map[f'Et_{_k+1}']   = ([int(_Et_idx[_k])],   _norm_Et)
        _name_map[f'Nt_{_k+1}']   = ([int(_Nt_idx[_k])],   _norm_Nt)
        _name_map[f'taun_{_k+1}'] = ([int(_taun_idx[_k])], _norm_tn)
        _name_map[f'taup_{_k+1}'] = ([int(_taup_idx[_k])], _norm_tp)
    _name_map['Et']   = ([int(i) for i in _Et_idx],   _norm_Et)
    _name_map['Nt']   = ([int(i) for i in _Nt_idx],   _norm_Nt)
    _name_map['taun'] = ([int(i) for i in _taun_idx], _norm_tn)
    _name_map['taup'] = ([int(i) for i in _taup_idx], _norm_tp)

    _fixed_norm = {}  # {full_vector_index: normalized_value}
    for _name, _val in fixed_params.items():
        if _name not in _name_map:
            raise ValueError(f"Unknown fixed_params key '{_name}'. "
                             f"Valid: {list(_name_map)}")
        _idxs, _nfn = _name_map[_name]
        for _idx in _idxs:
            _fixed_norm[_idx] = float(np.clip(_nfn(_val), 0.0, 1.0))

    _n_var_full  = 4 * K + 1
    _free_indices = np.array([i for i in range(_n_var_full) if i not in _fixed_norm])
    _n_var_free   = len(_free_indices)

    if _fixed_norm:
        fixed_names = [n for n in fixed_params]
        print(f"Fixed parameters : {fixed_names}  "
              f"({_n_var_full - _n_var_free} fixed, {_n_var_free} free)")

    def _expand(X_free):
        """Expand free-dim vector(s) (n_var_free,) or (N, n_var_free) to full dim."""
        batch = X_free.ndim == 2
        if not batch:
            X_free = X_free[np.newaxis]
        N = len(X_free)
        X_full = np.empty((N, _n_var_full))
        X_full[:, _free_indices] = X_free
        for _idx, _v in _fixed_norm.items():
            X_full[:, _idx] = _v
        return X_full[0] if not batch else X_full

    # ------------------------------------------------------------------ #
    # Parameter normalization / denormalization  (fully vectorized)
    # X can be (n_var,) for a single sample or (N, n_var) for a batch;
    # trailing dimensions are handled by numpy's ... indexing.
    # ------------------------------------------------------------------ #
    def normalize_params(Et, Nt, tau_n, tau_p, krad_val):
        """Et, Nt, tau_n, tau_p: 1-D arrays of length K.  krad_val: scalar."""
        x = np.empty(4 * K + 1)
        x[_Et_idx]   = (Et             - Et_bounds[0]) / _dEt
        x[_Nt_idx]   = (np.log10(Nt)   - _log_Nt0)    / _dlog_Nt
        x[_taun_idx] = (np.log10(tau_n) - _log_taun0)  / _dlog_taun
        x[_taup_idx] = (np.log10(tau_p) - _log_taup0)  / _dlog_taup
        x[-1]        = (np.log10(krad_val) - _log_krad0) / _dlog_krad
        return np.clip(x, 0, 1)

    def denormalize_params(X):
        """
        X: (n_var,) single sample  or  (N, n_var) batch.
        Returns Et, Nt, tau_n, tau_p, krad — last axis is traps (length K),
        krad has no trap axis.
        """
        X = np.clip(X, 0, 1)
        Et    = Et_bounds[0] + X[..., _Et_idx]   * _dEt
        Nt    = 10**(_log_Nt0   + X[..., _Nt_idx]   * _dlog_Nt)
        tau_n = 10**(_log_taun0 + X[..., _taun_idx] * _dlog_taun)
        tau_p = 10**(_log_taup0 + X[..., _taup_idx] * _dlog_taup)
        krad  = 10**(_log_krad0 + X[..., -1]        * _dlog_krad)
        return Et, Nt, tau_n, tau_p, krad

    # ------------------------------------------------------------------ #
    # Single-sample NN prediction (used for starting error print & final result)
    # ------------------------------------------------------------------ #
    def _predict_nn_single(x):
        """x: (n_var_free,) normalized.  Returns tau_diff (256,) [s]."""
        Et, Nt, tau_n, tau_p, krad_val = denormalize_params(_expand(x))
        nn_in = np.empty((1, 3 + 4 * num_traps), dtype=np.float32)
        nn_in[0, 0] = np.log10(npulse)
        nn_in[0, 1] = Eg                  # stored as Eg in eV directly (NOT log10!)
        nn_in[0, 2] = np.log10(krad_val)
        for i in range(num_traps):
            nn_in[0, 3 + 4*i]   = np.log10(Nt[i])
            nn_in[0, 3 + 4*i+1] = Et[i] / Eg
            nn_in[0, 3 + 4*i+2] = np.log10(tau_n[i])
            nn_in[0, 3 + 4*i+3] = np.log10(tau_p[i])
        import tensorflow as tf
        x_sc = input_scaler.transform(nn_in).astype(np.float32)
        y_sc = model._infer_fn(tf.constant(x_sc)).numpy()
        return 10**output_scaler.inverse_transform(y_sc)[0]

    def _interpolate_nn_to_exp(tau_nn_vals):
        """Interpolate single NN prediction (256,) to experimental QFLS points."""
        tau_sorted = tau_nn_vals[sort_idx]
        return PchipInterpolator(qfls_nn_sorted, tau_sorted, extrapolate=False)(qfls_data)

    # ------------------------------------------------------------------ #
    # Objective function (single sample, used only for printing start error)
    # ------------------------------------------------------------------ #
    def objective(x):
        try:
            tau_nn  = _predict_nn_single(x)
            if not np.all(np.isfinite(tau_nn)) or np.any(tau_nn <= 0):
                return 1e6
            tau_fit = _interpolate_nn_to_exp(tau_nn)
            if not np.all(np.isfinite(tau_fit)) or np.any(tau_fit <= 0):
                return 1e2
            log_res = np.abs(np.log10(tau_fit) - np.log10(tau_data))
            if error_metric == 'mse':
                return w_trpl * np.mean(log_res**2)
            elif error_metric == 'mae':
                return w_trpl * np.mean(log_res)
            else:  # integral
                return w_trpl * (-trapezoid(log_res, qfls_data) / (qfls_data.max() - qfls_data.min()))
        except Exception:
            return 1e6

    # ------------------------------------------------------------------ #
    # Timing accumulators (lists are mutable — closure can append to them)
    # Each entry: (n_samples, t_nn_inference, t_evaluate_total)
    # ------------------------------------------------------------------ #
    _eval_log = []   # one row per _evaluate call

    # ------------------------------------------------------------------ #
    # pymoo problem — fully vectorized batch evaluation
    # ------------------------------------------------------------------ #
    class FitProblem(Problem):
        def __init__(self, n_var):
            super().__init__(n_var=n_var, n_obj=1,
                             xl=np.zeros(n_var), xu=np.ones(n_var),
                             elementwise_evaluation=False)


        def _evaluate(self, X, out, *args, **kwargs):
            import tensorflow as tf
            t0_eval = time.perf_counter()
            N = len(X)

            # Expand free dims → full param vector, then denormalize
            Et, Nt, tau_n, tau_p, krad = denormalize_params(_expand(X))
            # Et, Nt, tau_n, tau_p: (N, K);  krad: (N,)

            # Build NN input batch (N, 3 + 4*K) in one shot
            nn_in = np.empty((N, 3 + 4 * num_traps), dtype=np.float32)
            nn_in[:, 0] = np.log10(npulse)   # stored as log10(n_pulse)
            nn_in[:, 1] = Eg                  # stored as Eg in eV directly (NOT log10!)
            nn_in[:, 2] = np.log10(krad)      # stored as log10(krad)
            for i in range(num_traps):
                nn_in[:, 3 + 4*i]   = np.log10(Nt[:, i])
                nn_in[:, 3 + 4*i+1] = Et[:, i] / Eg
                nn_in[:, 3 + 4*i+2] = np.log10(tau_n[:, i])
                nn_in[:, 3 + 4*i+3] = np.log10(tau_p[:, i])

            # Single GPU call for the whole generation
            x_sc = input_scaler.transform(nn_in).astype(np.float32)
            t0_nn = time.perf_counter()
            tau_nn = 10**output_scaler.inverse_transform(
                model._infer_fn(tf.constant(x_sc)).numpy()
            )  # (N, 256)
            t_nn = time.perf_counter() - t0_nn

            # Batch interpolation: PchipInterpolator accepts (n_x, N) y
            tau_fit = PchipInterpolator(
                qfls_nn_sorted, tau_nn[:, sort_idx].T, extrapolate=False
            )(qfls_data)  # (n_exp, N)

            # Validity: all experimental points finite and positive
            valid = np.all(np.isfinite(tau_fit) & (tau_fit > 0), axis=0)  # (N,)

            # Vectorized error across all candidates
            safe_fit = np.where(valid, tau_fit, 1.0)  # avoid log10(0) for invalid
            log_res = np.abs(np.log10(safe_fit) - np.log10(tau_data[:, None]))  # (n_exp, N)
            if error_metric == 'mse':
                raw = w_trpl * np.mean(log_res**2, axis=0)
            elif error_metric == 'mae':
                raw = w_trpl * np.mean(log_res, axis=0)
            else:  # integral
                raw = w_trpl * (-trapezoid(log_res, qfls_data, axis=0)
                                / (qfls_data.max() - qfls_data.min()))

            out["F"] = np.where(valid, raw, 1e6).reshape(-1, 1)

            _eval_log.append((N, t_nn, time.perf_counter() - t0_eval))

    from pymoo.termination.default import DefaultSingleObjectiveTermination

    termination = DefaultSingleObjectiveTermination(
        xtol=1e-6,
        cvtol=1e-6,
        period=100,
    )

    # ------------------------------------------------------------------ #
    # Initial guess
    # ------------------------------------------------------------------ #
    x0_multitrap_norm = None
    if x0_multitrap is not None:
        if len(x0_multitrap) != _n_var_full:
            raise ValueError(f"x0_multitrap must have length {_n_var_full}.")
        x0_full_norm = normalize_params(
            x0_multitrap[:K],
            x0_multitrap[K:2*K],
            x0_multitrap[2*K:3*K],
            x0_multitrap[3*K:4*K],
            x0_multitrap[4*K],
        )
        x0_multitrap_norm = x0_full_norm[_free_indices]  # keep only free dims

    def random_start(problem):
        return np.random.uniform(problem.xl, problem.xu)

    n_var   = _n_var_free   # reduced when fixed_params are present
    problem = FitProblem(n_var)
    res     = None
    x0      = None

    # ------------------------------------------------------------------ #
    # Optimisation loop
    # ------------------------------------------------------------------ #
    for trial in range(n_fits):
        print(f"\n===== Fit {trial + 1}/{n_fits} =====")
        if trial == 0 and x0_multitrap_norm is not None:
            x0_rand = x0_multitrap_norm
        else:
            x0_rand = random_start(problem)

        Et_s, Nt_s, tn_s, tp_s, kr_s = denormalize_params(_expand(x0_rand))
        print("\nStarting values:")
        for j in range(num_traps):
            print(f"  Trap {j+1}: Et={Et_s[j]:.3f} eV, Nt={Nt_s[j]:.2e} cm\u207b\u00b3, "
                  f"\u03c4n={tn_s[j]:.2e} s, \u03c4p={tp_s[j]:.2e} s")
        print(f"  krad={kr_s:.2e} cm\u00b3/s")
        print(f"\nStarting error ({error_metric}): {objective(x0_rand):.6f}")
        print("\nOptimizing...")
        print("-" * 70)

        # ── IPOP restart loop ──────────────────────────────────────────────
        # pymoo's CMAES breaks the internal cma restart generator at the
        # StopIteration boundary, so restarts never fire inside minimize().
        # We implement IPOP manually: each restart doubles the population
        # and starts from the best point found so far.
        _default_pop = 4 + int(3 * np.log(max(n_var, 1)))
        _pop_size    = None          # use cma default for first run
        _x0_run      = x0_rand
        _sigma_run   = sigma
        trial_res    = None

        start_time = time.time()
        for irun in range(restarts + 1):
            if irun > 0:
                _pop_size = (_pop_size or _default_pop) * 2
                _x0_run   = trial_res.X   # warm-start from best so far
                _sigma_run = sigma        # reset step-size each restart
                print(f"\n  -- IPOP restart {irun}/{restarts}  "
                      f"(pop_size={_pop_size}) --")

            algorithm = CMAES(
                x0=_x0_run,
                sigma=_sigma_run,
                maxfevals=maxfevals,
                parallelize=True,
                pop_size=_pop_size,
            )
            run_res = minimize(
                problem, algorithm,
                seed=43 + irun, verbose=True, save_history=True,
            )
            print(f"  Run {irun} error: {run_res.F[0]:.6f}")

            if trial_res is None or run_res.F < trial_res.F:
                trial_res = run_res

        solve_time = time.time() - start_time
        print(f"Final Error: {trial_res.F[0]:.6f}")

        if res is None or trial_res.F < res.F:
            res = trial_res
            x0  = x0_rand

    # ------------------------------------------------------------------ #
    # Extract best-fit results
    # ------------------------------------------------------------------ #
    x_opt  = res.X   # free-dim normalized optimum
    n_evals = res.algorithm.evaluator.n_eval
    n_gens  = len(res.history) if hasattr(res, 'history') else 0

    Et_opt, Nt_opt, tau_n_opt, tau_p_opt, krad_opt = denormalize_params(_expand(x_opt))

    # Final NN prediction at best-fit parameters
    tau_nn_opt = _predict_nn_single(x_opt)
    tau_fit    = _interpolate_nn_to_exp(tau_nn_opt)

    # Compute all three error metrics regardless of which was optimised
    log_res            = np.abs(np.log10(tau_fit) - np.log10(tau_data))
    error_integral_mse = -trapezoid(log_res, qfls_data) / (qfls_data.max() - qfls_data.min())
    error_mse          = float(np.mean(log_res**2))
    error_mae          = float(np.mean(log_res))
    error_total        = float(objective(x_opt))

    # ------------------------------------------------------------------ #
    # Timing breakdown
    # ------------------------------------------------------------------ #
    timing_df = pd.DataFrame(_eval_log, columns=['n_samples', 't_nn_s', 't_eval_s'])
    timing_df.index.name = 'generation'
    timing_df['t_nn_ms']   = timing_df['t_nn_s']   * 1e3
    timing_df['t_eval_ms'] = timing_df['t_eval_s'] * 1e3
    timing_df['t_overhead_ms'] = (timing_df['t_eval_s'] - timing_df['t_nn_s']) * 1e3
    timing_df['t_nn_cumul_s']   = timing_df['t_nn_s'].cumsum()
    timing_df['t_eval_cumul_s'] = timing_df['t_eval_s'].cumsum()
    timing_df['t_cmaes_overhead_cumul_s'] = timing_df['t_eval_cumul_s'] - timing_df['t_nn_cumul_s']

    t_nn_total   = timing_df['t_nn_s'].sum()
    t_eval_total = timing_df['t_eval_s'].sum()
    t_cmaes_overhead = solve_time - t_eval_total   # bookkeeping outside _evaluate
    n_generations = len(timing_df)

    

    # ------------------------------------------------------------------ #
    # Print summary
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 70)
    print("OPTIMIZATION RESULTS")
    print("=" * 70)
    print(f"Function evaluations : {n_evals}")
    print(f"Generations          : {n_gens}")
    print(f"Solve time           : {solve_time:.2f} s")
    print(f"\nTiming breakdown:")
    print(f"  NN inference (cumul.) : {t_nn_total:.3f} s  ({100*t_nn_total/solve_time:.1f}%)")
    print(f"  _evaluate total       : {t_eval_total:.3f} s  ({100*t_eval_total/solve_time:.1f}%)")
    print(f"  CMA-ES overhead       : {t_cmaes_overhead:.3f} s  ({100*t_cmaes_overhead/solve_time:.1f}%)")
    print(f"  Avg NN / generation   : {1e3*t_nn_total/max(n_generations,1):.2f} ms")
    print(f"\nTotal error ({error_metric:8s}): {error_total:.6f}")
    print(f"Error (integral)     : {error_integral_mse:.6f}")
    print(f"Error (MSE)          : {error_mse:.6f}")
    print(f"Error (MAE)          : {error_mae:.6f}")
    print("\nFitted parameters:")
    for i in range(num_traps):
        print(f"  Trap {i+1}:")
        print(f"    Et   = {Et_opt[i]:.4f} eV")
        print(f"    Nt   = {Nt_opt[i]:.3e} cm\u207b\u00b3")
        print(f"    \u03c4n   = {tau_n_opt[i]:.3e} s")
        print(f"    \u03c4p   = {tau_p_opt[i]:.3e} s")
    print(f"  krad = {krad_opt:.3e} cm\u00b3/s")
    print("=" * 70)

    # ------------------------------------------------------------------ #
    # Convert NN tau_diff(QFLS) → time, PL via qfls_tau_to_t_pl
    # ------------------------------------------------------------------ #
    _qfls_desc = qfls_nn[::-1]          # descending (high → low, i.e. t=0 → t=end)
    _tau_desc  = tau_nn_opt[::-1]
    _pl_base   = np.exp(_qfls_desc / VT)
    _pl_base  /= np.max(_pl_base)       # normalize to peak = 1
    _t_nn = np.concatenate([[0.0],
                             cumulative_trapezoid(-(_tau_desc / 2.0), np.log(_pl_base))])
    df_nn = pd.DataFrame({
        'pl':       _pl_base,
        'qfls':     _qfls_desc,
        'tau_diff': _tau_desc,
    }, index=_t_nn * 1e9)              # index = time [ns]
    df_nn.index.name = 'time_ns'

    # ------------------------------------------------------------------ #
    # Output dictionary
    # ------------------------------------------------------------------ #
    results = {
        'params': pd.DataFrame({
            'trap':      np.arange(1, num_traps + 1),
            'Et_eV':     Et_opt,
            'Nt_1/cm3':  Nt_opt,
            'tau_n_s':   tau_n_opt,
            'tau_p_s':   tau_p_opt,
        }),
        'krad_cm3s': krad_opt,
        # NN curve on its native QFLS axis plus reconstructed time/PL
        'trpl_nn': df_nn,     # time [s] index, cols: pl, qfls, tau_diff
        'error_total':    error_total,
        'error_integral_mse': error_integral_mse,
        'error_mse':      error_mse,
        'error_mae':      error_mae,
        'error_metric':   error_metric,
        'n_evals':    n_evals,
        'n_gens':     n_gens,
        'solve_time': solve_time,
        't_nn_total':        t_nn_total,
        't_eval_total':      t_eval_total,
        't_cmaes_overhead':  t_cmaes_overhead,
        'timing_per_gen':    timing_df,
        'history':    res.history, #if hasattr(res, 'history') else None,
        'x_opt':      res.X,
        'x0':         x0,
        'x0_unnormalized': denormalize_params(_expand(x0)),
        'fixed_params':    fixed_params,
        'bounds': {
            'Et_bounds':    Et_bounds,
            'Nt_bounds':    Nt_bounds,
            'tau_n_bounds': tau_n_bounds,
            'tau_p_bounds': tau_p_bounds,
            'krad_bounds':  krad_bounds,
        },
        'trpl_fit': pd.DataFrame({
            'qfls':      qfls_data,
            'tau_data':  tau_data,
            'tau_fit':   tau_fit,
        }),
    }

    # ------------------------------------------------------------------ #
    # Full history: all samples across all generations (denormalized)
    # Columns: generation, F, Et_k_eV, log10_Nt_k, log10_taun_k,
    #          log10_taup_k  (per trap k), log10_krad
    # ------------------------------------------------------------------ #
    history_rows = []
    if res.history is not None:
        for gen_idx, gen in enumerate(res.history):
            # gen.off = all lambda offspring (full population); gen.pop = selected best (mu)
            pop_src = gen.off if (hasattr(gen, 'off') and gen.off is not None) else gen.pop
            X_gen = pop_src.get('X')           # (pop_size, n_var_free)
            F_raw = pop_src.get('F')
            if X_gen is None or F_raw is None:
                continue
            F_gen = F_raw.flatten()            # (pop_size,)
            Et_g, Nt_g, tn_g, tp_g, kr_g = denormalize_params(_expand(X_gen))
            # Et_g, Nt_g, tn_g, tp_g: (pop_size, K); kr_g: (pop_size,)
            for s in range(len(F_gen)):
                row = {'generation': gen_idx, 'F': float(F_gen[s])}
                for k in range(num_traps):
                    row[f'Et_{k+1}_eV']      = float(Et_g[s, k])
                    row[f'log10_Nt_{k+1}']   = float(np.log10(max(Nt_g[s, k], 1e-300)))
                    row[f'log10_taun_{k+1}'] = float(np.log10(max(tn_g[s, k], 1e-300)))
                    row[f'log10_taup_{k+1}'] = float(np.log10(max(tp_g[s, k], 1e-300)))
                row['log10_krad'] = float(np.log10(max(kr_g[s], 1e-300)))
                history_rows.append(row)
    results['history_samples'] = pd.DataFrame(history_rows)

    # Release transient GPU tensors from this fitting run.
    # (Model weights stay allocated — call del model + gc.collect() outside
    #  this function if you want to free those too.)
    import gc
    gc.collect()

    return results


def export_fit_summary(results, basename, trpl_df=None, fit_results_dir="fit_results",
                       Eg=None, npulse=None, NC=2.2e18, NV=2.2e18, T=300,
                       sigma=1.0):
    """
    Write a formatted text summary of fitting results and export all artefacts.

    Creates:  fit_results_dir / {timestamp}_{basename}_fit_results /

    Artefacts
    ---------
    - summary.txt          : human-readable text report
    - parameters.json      : best-fit parameters
    - trpl.csv             : experimental TRPL (if trpl_df given)
    - trpl_sim.csv         : tau_diff(QFLS) comparison (exp vs NN fit)
    - history.npz          : best_f per generation (pymoo history)
    - history_samples.csv  : ALL CMA-ES samples (denormalized) across all generations
    - corner_plot.png      : pairwise scatter of history, coloured by error
    - rem_simulation.csv   : REM ODE output at best-fit params (if Eg & npulse given)
    - fit_plot.png         : 4-panel PL(t) / tau(QFLS) / QFLS(t) / tau(t) (if Eg & npulse given)

    Parameters
    ----------
    results         : dict returned by fit_trpl_multitrap_network
    basename        : str
    trpl_df         : pd.DataFrame, optional  experimental TRPL (qfls, tau_diff columns)
    fit_results_dir : str   root folder (default: "fit_results")
    Eg              : float, optional  bandgap [eV]   — needed for REM run + 4-panel plot
    npulse          : float, optional  carrier density [cm^-3] — needed for REM run
    NC, NV, T       : floats   effective DOS and temperature (defaults match training)

    Returns
    -------
    output_dir : str  path of the created run folder
    """
    import os, json
    from datetime import datetime
    timestamp  = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join(fit_results_dir,
                              f"{timestamp}_{basename}_fit_results")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving results to: {output_dir}")

    params  = results['params']
    n_traps = len(params)
    x0_Et, x0_Nt, x0_tn, x0_tp, x0_kr = results['x0_unnormalized']

    Et_opt   = params['Et_eV'].values
    Nt_opt   = params['Nt_1/cm3'].values
    tn_opt   = params['tau_n_s'].values
    tp_opt   = params['tau_p_s'].values
    krad_opt = float(results['krad_cm3s'])

    # ------------------------------------------------------------------ #
    # File name stems
    # ------------------------------------------------------------------ #
    trpl_csv         = f"{basename}_trpl_preprocessed.csv"
    trpl_sim_csv     = f"{basename}_trpl_sim.csv"
    nn_best_fit_csv      = f"{basename}_nn_best_fit.csv"
    param_json       = f"{basename}_parameters.json"
    history_npz      = f"{basename}_history.npz"
    best_f_history_npz      = f"{basename}_best_f_history.npz"
    timing_per_gen_csv = f"{basename}_timing_per_gen.csv"
    history_csv      = f"{basename}_history_samples.csv"
    corner_png       = f"{basename}_corner_plot.png"
    rem_csv          = f"{basename}_rem_simulation.csv"
    fit_plot_png     = f"{basename}_fit_plot.png"
    summary_txt      = f"{basename}_summary.txt"

    # ------------------------------------------------------------------ #
    # CSV exports
    # ------------------------------------------------------------------ #
    if trpl_df is not None:
        trpl_df.to_csv(os.path.join(output_dir, trpl_csv), float_format='%.6e')
    results['trpl_fit'].to_csv(
        os.path.join(output_dir, trpl_sim_csv), float_format='%.6e')

    nn_df = results.get('trpl_nn')
    if nn_df is not None:
        nn_df.to_csv(os.path.join(output_dir, nn_best_fit_csv),
                          index=True, float_format='%.6e')
        print(f"NN time/PL saved: {nn_best_fit_csv}")

    timing_df = results.get('timing_per_gen')
    timing_df.to_csv(
        os.path.join(output_dir, timing_per_gen_csv), float_format='%.6e'
    )
    if nn_df is not None:
        nn_df.to_csv(os.path.join(output_dir, nn_best_fit_csv),
                          index=True, float_format='%.6e')
        print(f"NN time/PL saved: {nn_best_fit_csv}")


    # ------------------------------------------------------------------ #
    # Parameters JSON
    # ------------------------------------------------------------------ #
    param_dict = {
        'traps': [
            {'Et_eV':   float(params['Et_eV'].iloc[i]),
             'Nt_cm3':  float(params['Nt_1/cm3'].iloc[i]),
             'tau_n_s': float(params['tau_n_s'].iloc[i]),
             'tau_p_s': float(params['tau_p_s'].iloc[i])}
            for i in range(n_traps)
        ],
        'krad_cm3s':   krad_opt,
        'error_total': float(results['error_total']),
        'fixed_params': {k: float(v) for k, v in results.get('fixed_params', {}).items()},
    }
    with open(os.path.join(output_dir, param_json), 'w') as f:
        json.dump(param_dict, f, indent=2)

    # ------------------------------------------------------------------ #
    # Laplace likelihood helper  L = 1/(2σ) · exp(−F/σ)
    # ------------------------------------------------------------------ #
    def _laplace(f_col):
        return (1.0 / (2.0 * sigma)) * np.exp(-f_col / sigma)

    # ------------------------------------------------------------------ #
    # Full history samples CSV  (+Laplace column)
    # ------------------------------------------------------------------ #
    if 'history_samples' in results and not results['history_samples'].empty:
        hs = results['history_samples'].copy()
        hs['laplace_likelihood'] = _laplace(hs['F'])
        hs.to_csv(os.path.join(output_dir, history_csv),
                  index=False, float_format='%.6e')
        print(f"History samples saved: {len(hs)} rows → {history_csv}")

    # ------------------------------------------------------------------ #
    # Best sample per generation: min-F row  (+Laplace column)
    # Columns: generation, best_f, laplace_likelihood, all denormalized params
    # ------------------------------------------------------------------ #
    if 'history_samples' in results and not results['history_samples'].empty:
        hs_all = results['history_samples']
        best_per_gen = (hs_all
                        .loc[hs_all.groupby('generation')['F'].idxmin()]
                        .reset_index(drop=True)
                        .rename(columns={'F': 'best_f'}))
        best_per_gen['laplace_likelihood'] = _laplace(best_per_gen['best_f'])
        best_per_gen.to_csv(
            os.path.join(output_dir, 'best_f_per_generation.csv'),
            index=False, float_format='%.6e',
        )
        np.savez(os.path.join(output_dir, best_f_history_npz),
                 best_f=best_per_gen['best_f'].values)
        print(f"Best-per-generation saved: best_f_per_generation.csv "
              f"({len(best_per_gen)} gens)")

        # -------------------------------------------------------------- #
        # Corner plot from history_samples
        # -------------------------------------------------------------- #
        try:
            _export_corner_plot(
                history_samples=hs,
                x0_unnorm=(x0_Et, x0_Nt, x0_tn, x0_tp, x0_kr),
                Et_opt=Et_opt, Nt_opt=Nt_opt, tn_opt=tn_opt,
                tp_opt=tp_opt, krad_opt=krad_opt,
                num_traps=n_traps,
                save_path=os.path.join(output_dir, corner_png),
            )
        except Exception as exc:
            print(f"Corner plot skipped: {exc}")

    # ------------------------------------------------------------------ #
    # REM simulation + 4-panel plot
    # ------------------------------------------------------------------ #
    rem_ok = False
    if Eg is not None and npulse is not None:
        try:
            tspan_log = np.logspace(-12, np.log10(5e-5), 512)
            rem_result = solve_transient(
                tspan_log, npulse, Eg, NC, NV, krad_opt,
                Nt_opt, Et_opt, tn_opt, tp_opt, T, 0,
            )
            (t_r, n_r, p_r, nt_r, qfls_r, tau_r,
             tau_r_n, tau_r_p, pl_r,
             e_trap, e_detrap, h_trap, h_detrap, krad_rate) = rem_result

            rem_df = pd.DataFrame({
                'pl':              pl_r.flatten(),
                'qfls_eV':         qfls_r.flatten(),
                'tau_diff_s':      tau_r.flatten(),
                'n_cm3':           n_r.flatten(),
                'p_cm3':           p_r.flatten(),
                **{f'nt_{k+1}_cm-3': nt_r[:, k] for k in range(n_traps)},
                'tau_n_eff_s':           tau_r_n.flatten(),
                'tau_p_eff_s':           tau_r_p.flatten(),
                'krad_rate_cm-3s-1':     krad_rate.flatten(),
                **{f'e_trap_{k+1}_cm-3s-1': e_trap[:, k] for k in range(n_traps)},
                **{f'e_detrap_{k+1}_cm-3s-1': e_detrap[:, k] for k in range(n_traps)},
                **{f'net_e_trapping_{k+1}_cm-3s-1': np.abs(e_trap[:, k] - e_detrap[:, k]) for k in range(n_traps)},
                **{f'h_trap_{k+1}_cm-3s-1': h_trap[:, k] for k in range(n_traps)},
                **{f'h_detrap_{k+1}_cm-3s-1': h_detrap[:, k] for k in range(n_traps)},
                **{f'net_h_trapping_{k+1}_cm-3s-1': np.abs(h_trap[:, k] - h_detrap[:, k]) for k in range(n_traps)},
            }, index=t_r.flatten() * 1e9)           # index = time [ns]
            rem_df.index.name = 'time_ns'
            rem_df.to_csv(os.path.join(output_dir, rem_csv),
                          index=True, float_format='%.6e')
            print(f"REM simulation saved: {rem_csv}")
            rem_ok = True

            # 4-panel figure
            try:
                _export_fit_plot(
                    rem_df=rem_df,
                    nn_time_df=results.get('trpl_nn'),
                    trpl_fit_df=results.get('trpl_fit'),
                    trpl_df=trpl_df,
                    n_traps=n_traps,
                    save_path=os.path.join(output_dir, fit_plot_png),
                )
            except Exception as exc:
                print(f"4-panel plot skipped: {exc}")
        except Exception as exc:
            print(f"REM simulation skipped: {exc}")

    # ------------------------------------------------------------------ #
    # Text summary
    # ------------------------------------------------------------------ #
    sep  = '=' * 70
    dash = '-' * 70
    lines = [sep, 'FIT RESULTS SUMMARY', sep, '',
             f'Export Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', '',
             'TRAP PARAMETERS:', dash]
    for i in range(n_traps):
        lines += [f'Trap {i+1}:',
                  f'  Et    = {params["Et_eV"].iloc[i]:.4f} eV',
                  f'  Nt    = {params["Nt_1/cm3"].iloc[i]:.4e} cm\u207b\u00b3',
                  f'  \u03c4\u2099    = {params["tau_n_s"].iloc[i]:.4e} s',
                  f'  \u03c4\u209a    = {params["tau_p_s"].iloc[i]:.4e} s', '']
    lines += ['RADIATIVE RECOMBINATION:', dash,
              f'krad = {krad_opt:.4e} cm\u00b3/s', '',
              'FIT QUALITY:', dash,
              f'Total error  = {results["error_total"]:.6f}',
              f'TRPL error   = {results["error_total"]:.6f}', '',
              'OPTIMIZATION STATISTICS:', dash,
              f'Evaluations  = {results["n_evals"]}',
              f'Generations  = {results["n_gens"]}',
              f'Solve Time   = {results["solve_time"]:.2f} s', '',
              'Start point values:', dash]
    for i in range(n_traps):
        lines += [f'Et_{i+1}_eV     : {x0_Et[i]:.4e}',
                  f'Nt_{i+1}_1/cm3  : {x0_Nt[i]:.4e}',
                  f'taun_{i+1}_s    : {x0_tn[i]:.4e}',
                  f'taup_{i+1}_s    : {x0_tp[i]:.4e}']
    lines += [f'krad_cm3s    : {float(x0_kr):.4e}', '']
    if results.get('fixed_params'):
        lines += ['Fixed parameters:', dash]
        for k, v in results['fixed_params'].items():
            lines.append(f'  {k} = {v:.4e}')
        lines.append('')
    error_metric = results.get('error_metric', 'unknown')
    error_val    = results.get('error_total', float('nan'))
    error_mse    = results.get('error_mse',   float('nan'))
    error_mae    = results.get('error_mae',   float('nan'))
    lines += ['ERROR METRICS:', dash,
              f'Optimised metric : {error_metric}',
              f'Error ({error_metric:8s}): {error_val:.6f}',
              f'Error (mse      ): {error_mse:.6f}',
              f'Error (mae      ): {error_mae:.6f}', '']

    exported = [
        f'summary          : {summary_txt}',
        f'parameters       : {param_json}',
        f'trpl_sim         : {trpl_sim_csv}',
    ]
    if trpl_df is not None:
        exported.append(f'trpl_preprocessed: {trpl_csv}')
    if nn_df is not None:
        exported.append(f'nn_best_fit      : {nn_best_fit_csv}')
    exported += [
        f'timing_per_gen   : {timing_per_gen_csv}',
        f'history_samples  : {history_csv}',
        f'best_f_per_gen   : best_f_per_generation.csv',
        f'corner_plot      : {corner_png}',
    ]
    if rem_ok:
        exported += [f'rem_simulation   : {rem_csv}',
                     f'fit_plot         : {fit_plot_png}']
    lines += ['EXPORTED FILES:', dash] + exported + ['', sep]

    txt_path = os.path.join(output_dir, summary_txt)
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"Summary written to {txt_path}")
    print('\n'.join(lines))
    return output_dir


# --------------------------------------------------------------------------- #
# Private helpers for export_fit_summary
# --------------------------------------------------------------------------- #

def _export_corner_plot(history_samples, x0_unnorm, Et_opt, Nt_opt, tn_opt,
                        tp_opt, krad_opt, num_traps, save_path):
    """
    Build parameter_config / best_fit / start_params from history_samples
    and delegate to cornerplot_module.plot_corner_with_bestfit (300 dpi PNG).
    """
    from cornerplot_module import plot_corner_with_bestfit

    df = history_samples.rename(columns={'F': 'MSE'}).copy()

    # Build parameter_config from column names
    parameter_config = {}
    for col in df.columns:
        if col in ('generation', 'MSE'):
            continue
        if col.startswith('Et_') and col.endswith('_eV'):
            k = col.split('_')[1]
            sfx = k if num_traps > 1 else ''
            parameter_config[col] = {'label': f'$E_{{t{sfx}}}$ [eV]'}
        elif col.startswith('log10_Nt_'):
            k = col.split('_')[-1]
            sfx = k if num_traps > 1 else ''
            parameter_config[col] = {'label': f'$N_{{t{sfx}}}$ [cm$^{{-3}}$]',
                                     'log_range': True}
        elif col.startswith('log10_taun_'):
            k = col.split('_')[-1]
            sfx = k if num_traps > 1 else ''
            parameter_config[col] = {'label': f'$\\tau_{{n{sfx}}}$ [s]',
                                     'log_range': True}
        elif col.startswith('log10_taup_'):
            k = col.split('_')[-1]
            sfx = k if num_traps > 1 else ''
            parameter_config[col] = {'label': f'$\\tau_{{p{sfx}}}$ [s]',
                                     'log_range': True}
        elif col == 'log10_krad':
            parameter_config[col] = {'label': '$k_{rad}$ [cm$^3$/s]',
                                     'log_range': True}

    param_cols = list(parameter_config.keys())

    # Best-fit single-row DataFrame (log10 for log params, linear for Et)
    best_row = {}
    for k in range(num_traps):
        best_row[f'Et_{k+1}_eV']      = float(Et_opt[k])
        best_row[f'log10_Nt_{k+1}']   = float(np.log10(max(float(Nt_opt[k]),  1e-300)))
        best_row[f'log10_taun_{k+1}'] = float(np.log10(max(float(tn_opt[k]),  1e-300)))
        best_row[f'log10_taup_{k+1}'] = float(np.log10(max(float(tp_opt[k]),  1e-300)))
    best_row['log10_krad'] = float(np.log10(max(float(krad_opt), 1e-300)))
    best_fit_params = pd.DataFrame([best_row])

    # Start-point single-row DataFrame
    x0_Et, x0_Nt, x0_tn, x0_tp, x0_kr = x0_unnorm
    start_row = {}
    for k in range(num_traps):
        start_row[f'Et_{k+1}_eV']      = float(x0_Et[k])
        start_row[f'log10_Nt_{k+1}']   = float(np.log10(max(float(x0_Nt[k]), 1e-300)))
        start_row[f'log10_taun_{k+1}'] = float(np.log10(max(float(x0_tn[k]), 1e-300)))
        start_row[f'log10_taup_{k+1}'] = float(np.log10(max(float(x0_tp[k]), 1e-300)))
    start_row['log10_krad'] = float(np.log10(max(float(x0_kr), 1e-300)))
    start_params = pd.DataFrame([start_row])

    g = plot_corner_with_bestfit(
        df, parameter_config, best_fit_params,
        params_to_plot=param_cols,
        start_params=start_params,
        save_path=save_path,
    )
    plt.close(g.fig)


def _export_fit_plot(rem_df, nn_time_df, trpl_fit_df, trpl_df,
                     n_traps, save_path):
    """
    Save a 4-panel figure (matplotlib, 150 dpi):
      [0,0] PL vs time (ns)  — linear x, log y
      [0,1] tau_diff vs time (s) — log-log
      [1,0] PL vs time (s)   — log-log
      [1,1] tau_diff vs QFLS — linear x, log y
    REM = orange solid, NN = blue dashed, Exp = dark-grey circles.
    Axis limits follow experimental data where available.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    ax_pl_ns, ax_tau_t, ax_pl_s, ax_tau_q = (
        axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1])

    c_rem = '#E07B39'
    c_nn  = '#3A86C8'
    c_exp = '#444444'

    # ── Common data prep ─────────────────────────────────────────────────
    t_rem_ns  = rem_df.index.to_numpy()          # ns  (index)
    t_rem_s   = t_rem_ns * 1e-9
    pl_rem    = rem_df['pl'].values
    mask_r    = pl_rem > 0
    pl_rem_n  = pl_rem / np.nanmax(pl_rem[mask_r]) if mask_r.any() else pl_rem
    qfls_rem  = rem_df['qfls_eV'].values
    tau_rem   = rem_df['tau_diff_s'].values
    mask_rem  = np.isfinite(tau_rem) & (tau_rem > 0)

    has_nn = nn_time_df is not None
    if has_nn:
        t_nn_ns  = nn_time_df.index.to_numpy()   # ns  (index)
        t_nn_s   = t_nn_ns * 1e-9
        pl_nn    = nn_time_df['pl'].values
        tau_nn   = nn_time_df['tau_diff'].values
        qfls_nn  = nn_time_df['qfls'].values

    # Experimental reference (from trpl_df — time in ns index)
    has_exp = trpl_df is not None
    if has_exp:
        t_exp_ns = trpl_df.index.to_numpy()
        t_exp_s  = t_exp_ns * 1e-9
        pl_exp   = trpl_df['pl'].values
        pl_exp_n = pl_exp / np.nanmax(pl_exp[pl_exp > 0])
        tau_exp  = trpl_df['tau_diff'].values if 'tau_diff' in trpl_df.columns else None

    # ── Panel [0,0]: PL vs time (ns), semilogy ───────────────────────────
    ax_pl_ns.semilogy(t_rem_ns, pl_rem_n, color=c_rem, lw=2, label='REM')
    if has_nn:
        ax_pl_ns.semilogy(t_nn_ns, pl_nn, color=c_nn, lw=2, ls='--', label='NN')
    if has_exp:
        ax_pl_ns.semilogy(t_exp_ns, pl_exp_n, 'o', color=c_exp, ms=3, alpha=0.6, label='Exp')
        ax_pl_ns.set_xlim(0, t_exp_ns.max())
    ax_pl_ns.set_xlabel('Time (ns)')
    ax_pl_ns.set_ylabel('PL (norm.)')
    ax_pl_ns.legend(fontsize=9)
    ax_pl_ns.grid(True, alpha=0.3)

    # ── Panel [0,1]: tau_diff vs time (s), loglog ────────────────────────
    if mask_rem.any():
        ax_tau_t.loglog(t_rem_s[mask_rem], tau_rem[mask_rem],
                        color=c_rem, lw=2, label='REM')
    if has_nn:
        mask_nn = tau_nn > 0
        ax_tau_t.loglog(t_nn_s[mask_nn], tau_nn[mask_nn],
                        color=c_nn, lw=2, ls='--', label='NN')
    if has_exp and tau_exp is not None:
        mask_te = (t_exp_s > 0) & np.isfinite(tau_exp) & (tau_exp > 0)
        ax_tau_t.loglog(t_exp_s[mask_te], tau_exp[mask_te],
                        'o', color=c_exp, ms=3, alpha=0.6, label='Exp')
        ax_tau_t.set_xlim(t_exp_s[mask_te].min(), t_exp_s[mask_te].max())
    ax_tau_t.set_xlabel('Time (s)')
    ax_tau_t.set_ylabel('τ_diff (s)')
    ax_tau_t.legend(fontsize=9)
    ax_tau_t.grid(True, alpha=0.3)

    # ── Panel [1,0]: PL vs time (s), loglog ─────────────────────────────
    mask_rs = mask_r & (t_rem_s > 0)
    ax_pl_s.loglog(t_rem_s[mask_rs], pl_rem_n[mask_rs], color=c_rem, lw=2, label='REM')
    if has_nn:
        mask_nns = (t_nn_s > 0) & (pl_nn > 0)
        ax_pl_s.loglog(t_nn_s[mask_nns], pl_nn[mask_nns],
                       color=c_nn, lw=2, ls='--', label='NN')
    if has_exp:
        mask_es = (t_exp_s > 0) & (pl_exp_n > 0)
        ax_pl_s.loglog(t_exp_s[mask_es], pl_exp_n[mask_es],
                       'o', color=c_exp, ms=3, alpha=0.6, label='Exp')
        ax_pl_s.set_xlim(t_exp_s[mask_es].min(), t_exp_s[mask_es].max())
    ax_pl_s.set_xlabel('Time (s)')
    ax_pl_s.set_ylabel('PL (norm.)')
    ax_pl_s.legend(fontsize=9)
    ax_pl_s.grid(True, alpha=0.3)

    # ── Panel [1,1]: tau_diff vs QFLS, semilogy ──────────────────────────
    if mask_rem.any():
        ax_tau_q.semilogy(qfls_rem[mask_rem], tau_rem[mask_rem],
                          color=c_rem, lw=2, label='REM')
    if has_nn:
        ax_tau_q.semilogy(qfls_nn, tau_nn, color=c_nn, lw=2, ls='--', label='NN')
    if trpl_fit_df is not None:
        ax_tau_q.semilogy(trpl_fit_df['qfls'].values, trpl_fit_df['tau_data'].values,
                          'o', color=c_exp, ms=4, alpha=0.7, label='Exp')
        q_exp = trpl_fit_df['qfls'].values
        ax_tau_q.set_xlim(q_exp.min(), q_exp.max())
    ax_tau_q.set_xlabel('QFLS (eV)')
    ax_tau_q.set_ylabel('τ_diff (s)')
    ax_tau_q.legend(fontsize=9)
    ax_tau_q.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"4-panel fit plot saved: {save_path}")





def plot_fit(results, figsize=(7.5, 3.75), save_path=None):
    """Plot SSPL and TRPL fits"""
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