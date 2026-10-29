"""
benchmark.py — timing (timeit) and evaluation counting
======================================================

Two small tools that every step of the workflow uses so that the timing table
(REVISION_PLAN.md, task T8) can be filled directly from the notebook output:

* :func:`timeit_call` runs a function with :class:`timeit.Timer` and returns
  both the function's result and the measured wall times.
* :class:`EvalCounter` counts log-likelihood evaluations.  It is built from
  ``tf.Variable`` objects, so the count is also correct inside
  ``tf.function(jit_compile=True)`` and inside TFP's MCMC while-loops, where a
  plain Python counter would only be incremented once at trace time.

Definitions used throughout
---------------------------
* ``n_calls``  : number of batched calls of the log-likelihood.
* ``n_points`` : number of parameter sets (rows) evaluated, summed over all
  calls.  This is "the number of likelihood evaluations" in the usual
  (unbatched) sense and is the number to compare with an ODE-based workflow.
"""
from __future__ import annotations

import gc
import timeit
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# timeit wrapper
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Timing:
    """Wall-clock times of repeated runs of one step (seconds)."""
    label: str
    times_s: list = field(default_factory=list)
    warmup: int = 0          # untimed calls made before the timed repeats

    @property
    def best_s(self) -> float:
        """Fastest run — the usual timeit convention for 'the' run time."""
        return float(min(self.times_s))

    @property
    def mean_s(self) -> float:
        return float(np.mean(self.times_s))

    @property
    def std_s(self) -> float:
        """Sample standard deviation (ddof = 1) over the repeats; 0 for one repeat."""
        return float(np.std(self.times_s, ddof=1)) if len(self.times_s) > 1 else 0.0

    def as_dict(self) -> dict:
        return {"label": self.label, "repeats": len(self.times_s), "warmup": self.warmup,
                "best_s": self.best_s, "mean_s": self.mean_s,
                "std_s": self.std_s, "times_s": [float(t) for t in self.times_s]}

    def __str__(self) -> str:
        if len(self.times_s) == 1:
            return f"{self.label}: {self.best_s:.3f} s"
        return (f"{self.label}: mean {self.mean_s:.3f} ± {self.std_s:.3f} s "
                f"(best {self.best_s:.3f} s) over {len(self.times_s)} runs")


def timeit_call(fn, *args, repeats: int = 1, warmup: int = 0, label: str = "", **kwargs):
    """
    Time ``fn(*args, **kwargs)`` with :class:`timeit.Timer` and keep its result.

    Each repeat runs the function exactly once (``number=1``); the result of
    the LAST repeat is returned.  Garbage collection is re-enabled during the
    timed runs (timeit disables it by default), because long TensorFlow runs
    would otherwise accumulate memory.

    Parameters
    ----------
    fn : callable
    repeats : int
        Number of timed runs (>= 2 for a standard deviation).
    warmup : int
        Untimed calls before the timed repeats.  They absorb one-time costs
        inside ``fn`` (XLA compilation of shapes not compiled beforehand,
        start of worker pools, numba caches) that would otherwise make the
        first repeat slower and inflate mean and SD.
    label : str
        Name of the step for the summary table.

    Returns
    -------
    result : whatever ``fn`` returns (last repeat)
    timing : Timing
    """
    for _ in range(int(warmup)):
        fn(*args, **kwargs)
    holder = {}

    def _stmt():
        holder["result"] = fn(*args, **kwargs)

    timer = timeit.Timer(_stmt, setup=gc.enable)
    times = timer.repeat(repeat=int(repeats), number=1)
    return holder["result"], Timing(label or getattr(fn, "__name__", "step"), times, int(warmup))


def format_mean_sd(mean_s: float, sd_s: float = 0.0, n: int | None = None, unit: str = "s",
                   prefix: str = "") -> str:
    """
    'mean ± sd unit (n = k)' for the timing table (input in seconds; unit
    'ms', 's', 'min' or 'h').  With ``n`` (number of timed repeats) the SD is
    shown for n > 1 and '(n = k)' is appended; without ``n`` (derived values,
    e.g. estimates) the SD is shown when it is > 0.  ``prefix``: e.g. '≈ '.
    """
    f = {"ms": 1e3, "s": 1.0, "min": 1 / 60, "h": 1 / 3600}[unit]
    m, s = mean_s * f, sd_s * f
    # decimals: 3 significant digits, at most 2 decimals
    dec = int(min(2, max(0, 2 - np.floor(np.log10(abs(m)))))) if m else 0
    show_sd = (n > 1) if n is not None else s > 0
    txt = f"{prefix}{m:.{dec}f}" + (f" ± {s:.{dec}f}" if show_sd else "") + f" {unit}"
    return txt + (f" (n = {n})" if n is not None else "")


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation counter
# ─────────────────────────────────────────────────────────────────────────────

class EvalCounter:
    """
    Counts batched log-likelihood calls and evaluated parameter sets.

    ``record(batch_size)`` must be called from inside the (possibly compiled)
    TensorFlow function that evaluates the likelihood; it uses
    ``tf.Variable.assign_add`` and therefore counts every execution, also
    inside XLA-compiled while-loops (NUTS, L-BFGS).
    """

    def __init__(self, name: str = "log_lik"):
        import tensorflow as tf
        self.name = name
        self.n_calls = tf.Variable(0, dtype=tf.int64, trainable=False,
                                   name=f"{name}_n_calls")
        self.n_points = tf.Variable(0, dtype=tf.int64, trainable=False,
                                    name=f"{name}_n_points")

    def record(self, batch_size) -> None:
        """Add one batched call with ``batch_size`` parameter sets."""
        import tensorflow as tf
        self.n_calls.assign_add(tf.constant(1, tf.int64))
        self.n_points.assign_add(tf.cast(batch_size, tf.int64))

    def reset(self) -> None:
        self.n_calls.assign(0)
        self.n_points.assign(0)

    def snapshot(self) -> dict:
        return {"n_calls": int(self.n_calls.numpy()),
                "n_points": int(self.n_points.numpy())}

    def delta_since(self, snap: dict) -> dict:
        """Counts accumulated since an earlier :meth:`snapshot`."""
        now = self.snapshot()
        return {k: now[k] - snap[k] for k in now}


class PyCounter:
    """Same interface as EvalCounter for pure-Python objectives (e.g. the ODE)."""

    def __init__(self, name: str = "py"):
        self.name = name
        self._calls = 0
        self._points = 0

    def record(self, batch_size) -> None:
        self._calls += 1
        self._points += int(batch_size)

    def reset(self) -> None:
        self._calls = self._points = 0

    def snapshot(self) -> dict:
        return {"n_calls": self._calls, "n_points": self._points}

    def delta_since(self, snap: dict) -> dict:
        now = self.snapshot()
        return {k: now[k] - snap[k] for k in now}


class count_evals:
    """
    Context manager that measures how many evaluations a block performed.

    Example
    -------
    >>> with count_evals(log_post.counter) as c:
    ...     result = run_something()
    >>> c.counts   # {'n_calls': ..., 'n_points': ...}
    """

    def __init__(self, counter: EvalCounter):
        self.counter = counter
        self.counts = None

    def __enter__(self):
        self._start = self.counter.snapshot()
        return self

    def __exit__(self, *exc):
        self.counts = self.counter.delta_since(self._start)
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Throughput benchmark (unit test 5 / task T8)
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_log_post(log_post, batch_sizes=(1, 16, 64, 256, 1024, 4096),
                       repeats: int = 5, with_gradient: bool = True,
                       seed: int = 0) -> pd.DataFrame:
    """
    Measure the time per batched log-posterior (+ gradient) evaluation.

    The first call for each batch size triggers tracing/XLA compilation and is
    excluded from the timing (reported separately as ``compile_s``).

    Returns
    -------
    DataFrame with columns batch_size, compile_s, time_per_call_ms,
    time_per_point_us, points_per_s.
    """
    import tensorflow as tf
    rng = np.random.default_rng(seed)
    rows = []
    for B in batch_sizes:
        u = tf.constant(rng.uniform(-2, 2, size=(int(B), log_post.dim)), tf.float64)
        fn = log_post.tf_log_prob_and_grad_u if with_gradient else log_post.tf_log_prob_u

        def _run():
            out = fn(u)
            # force execution (results live on the GPU otherwise)
            (out[0] if isinstance(out, (tuple, list)) else out).numpy()

        _, t_compile = timeit_call(_run, repeats=1)
        _, t = timeit_call(_run, repeats=repeats)
        rows.append({"batch_size": int(B),
                     "compile_s": t_compile.best_s,
                     "time_per_call_ms": 1e3 * t.best_s,
                     "time_per_point_us": 1e6 * t.best_s / B,
                     "points_per_s": B / t.best_s})
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Offline costs of the surrogate (timing table, task T8)
# ─────────────────────────────────────────────────────────────────────────────

def offline_costs(nn_dir, training_hdf5) -> dict:
    """
    Cost of generating the training simulations and of training the network,
    read from the files the training pipeline left behind:

    * simulations: per-sample ODE solve times (``solve_time``) and completion
      time stamps of the raw simulation HDF5 (path stored in the training
      HDF5 as ``source_file``); CPU time = sum of solve times, wall time =
      last completion time stamp;
    * training: wall time between the start time encoded in the run folder
      name (YYYYmmdd-HHMMSS) and the time stamp in training_details.json,
      minus the simulation wall time (the pipeline runs simulation,
      preparation and training in sequence) — an upper bound for training.
      Networks trained outside train_pipeline.sh: ``offline_provenance.json`` in
      the network folder (times read from the run logs) or ``wall_time_s`` in
      training_details.json (train_conv_v2.py / train_conv_steps.py).
    * merged training sets ("A + B" in ``source_file``): solve times of all
      simulation files; wall time = sum (simulated one after another).
    Missing information is returned as None.
    """
    import json
    from datetime import datetime
    from pathlib import Path
    import h5py
    nn_dir, out = Path(nn_dir), {}
    try:
        with h5py.File(training_hdf5, "r") as hf:
            src = hf["source_file"][()]
            src = src.decode() if isinstance(src, bytes) else str(src)
            out["n_training_samples"] = int(sum(hf[f"X_{s}_raw"].shape[0] for s in ("train", "val", "test")))
        # merged training sets (nn-training/merge_training_data.py) list several simulation files: "A + B"
        srcs = [Path(x.strip()) for x in src.split(" + ")]
        out["simulation_file"] = " + ".join(str(x) for x in srcs)
        st_all, wall = [], []
        for f in srcs:
            if not f.exists():
                raise FileNotFoundError(f)
            with h5py.File(f, "r") as hf:
                st_all.append(hf["solve_time"][:])
                if "completion_timestamps" in hf:          # the files were simulated one after another
                    ts = hf["completion_timestamps"][:]
                    wall.append(float(np.nanmax(ts) / 3600) if np.isfinite(ts).any() else np.nan)
                else:
                    wall.append(np.nan)
        st = np.concatenate(st_all)
        out["n_simulations"] = int(len(st))
        out["simulation_cpu_h"] = float(np.nansum(st[st > 0]) / 3600)
        ok = st[np.isfinite(st) & (st > 0)]
        out["simulation_s_per_solve_mean"] = float(ok.mean()) if len(ok) else None
        out["simulation_s_per_solve_sd"] = float(ok.std(ddof=1)) if len(ok) > 1 else None
        out["simulation_wall_h"] = float(np.sum(wall)) if np.all(np.isfinite(wall)) else None
        if len(srcs) > 1:
            out["simulation_files_n"] = [int(len(x)) for x in st_all]
    except Exception as e:
        out["simulation_error"] = f"{type(e).__name__}: {e}"
    try:
        det = json.loads((nn_dir / "training_details.json").read_text())
        out["epochs_trained"] = det.get("training", {}).get("epochs_trained")
        prov = nn_dir / "offline_provenance.json"
        if prov.exists():           # networks trained outside train_pipeline.sh: times taken from the run logs
            pv = json.loads(prov.read_text())
            out["training_wall_h_upper"] = pv["training_wall_h_upper"]
            out["training_source"] = pv.get("source")
            return out
        if det.get("training", {}).get("wall_time_s") is not None:      # train_conv_v2 / train_conv_steps
            out["training_wall_h_upper"] = det["training"]["wall_time_s"] / 3600
            out["training_source"] = "training_details.json (wall_time_s)"
            return out
        start = datetime.strptime("-".join(nn_dir.name.split("-")[:2]), "%Y%m%d-%H%M%S")
        end = datetime.fromisoformat(det["timestamp"])
        out["pipeline_wall_h"] = (end - start).total_seconds() / 3600
        log = nn_dir / "pipeline.log"
        sims_in_pipeline = not (log.exists() and "Skipping simulations" in log.read_text(errors="ignore"))
        if not sims_in_pipeline:          # simulated (and prepared) outside this pipeline run
            out["training_wall_h_upper"] = out["pipeline_wall_h"]
        elif out.get("simulation_wall_h") is not None:
            out["training_wall_h_upper"] = max(0.0, out["pipeline_wall_h"] - out["simulation_wall_h"])
    except Exception as e:
        out["training_error"] = f"{type(e).__name__}: {e}"
    return out


# ─────────────────────────────────────────────────────────────────────────────
# sampler efficiency (T8 table and 11_sampler_efficiency.csv)
# ─────────────────────────────────────────────────────────────────────────────
EFFICIENCY_COLUMNS = ["sampler", "timed_mean_s", "n_evals", "min_ess_bulk", "min_ess_tail", "max_r_hat",
                      "ess_per_s", "ess_per_s_sampling", "ess_per_1k_evals",
                      "step_size", "mean_leapfrogs", "n_divergent", "compile_s", "time_warmup_s",
                      "calls_per_iteration", "max_tau_int", "acceptance_fraction",
                      "ess_kish", "log_z", "log_z_se", "sigma_trpl", "note"]


def sampler_efficiency(sampler: str, diag: pd.DataFrame, info: dict, timed_mean_s: float | None,
                       free_keys=None) -> dict:
    """
    Efficiency metrics of one sampler run ('nuts', 'emcee' or 'smc'); every key of EFFICIENCY_COLUMNS is present
    (None where it does not apply), so the rows of all samplers share the same columns.

    diag          : samplers.diagnostics() table (index = display keys; columns ess_bulk, ess_tail, r_hat);
                    SMC: diagnostics of the final particles over the islands (diagnostics_smc.csv)
    info          : nuts_res.info / emcee_res.info (also sampling_info.json["nuts" | "emcee"]);
                    SMC: summary.json["smc"] of smc.ipynb (evals, ess_kish, log_z, log_z_se)
    timed_mean_s  : timeit mean of the timed run (NUTS: warm-up + sampling incl. the XLA recompilation inside the run,
                    info["compile_s"]; the first compilation is an untimed run beforehand; emcee: burn-in + production)
    free_keys     : display keys of the free parameters — fixed parameters are left out of min ESS / max R-hat
    ess_per_1k_evals uses evals_total_points (likelihood evaluations; for NUTS each includes the gradient).
    """
    d = diag if free_keys is None else diag.loc[[k for k in diag.index if k in set(free_keys)]]
    bulk, tail, rhat = float(d["ess_bulk"].min()), float(d["ess_tail"].min()), float(d["r_hat"].max())
    n_evals = info.get("evals_total_points", info.get("evals"))
    out = dict.fromkeys(EFFICIENCY_COLUMNS)
    out.update(sampler=sampler, timed_mean_s=timed_mean_s, n_evals=n_evals, min_ess_bulk=bulk, min_ess_tail=tail,
               max_r_hat=rhat, ess_per_s=bulk / timed_mean_s if timed_mean_s else None,
               ess_per_1k_evals=1000.0 * bulk / n_evals if n_evals else None)
    if sampler == "nuts":
        n_iter = (info.get("n_warmup") or 0) + (info.get("n_draws") or 0)
        out.update(ess_per_s_sampling=bulk / info["time_sampling_s"] if info.get("time_sampling_s") else None,
                   step_size=info.get("step_size"), mean_leapfrogs=info.get("mean_leapfrogs"),
                   n_divergent=info.get("n_divergent"), compile_s=info.get("compile_s"),
                   time_warmup_s=info.get("time_warmup_s"),
                   calls_per_iteration=info["evals_total_calls"] / n_iter if info.get("evals_total_calls") and n_iter else None)
    elif sampler == "emcee":
        tau = info.get("tau_int")
        out.update(max_tau_int=float(np.max(tau)) if tau is not None else None,
                   acceptance_fraction=info.get("acceptance_fraction"))
    elif sampler == "smc":
        out.update(ess_kish=info.get("ess_kish"), log_z=info.get("log_z"), log_z_se=info.get("log_z_se"),
                   note="ESS over the islands (final particles); Kish ESS of the beta = 1 weights in ess_kish")
    else:
        raise ValueError("sampler must be 'nuts', 'emcee' or 'smc'")
    return out


def importance_efficiency(name: str, ess_kish: float, n_evals: int, timed_mean_s: float | None,
                          sigma_trpl: float | None = None, note: str = "") -> dict:
    """EFFICIENCY_COLUMNS row of an importance sampler (cylinder IS): only the Kish ESS of the weights exists."""
    out = dict.fromkeys(EFFICIENCY_COLUMNS)
    out.update(sampler=name, timed_mean_s=timed_mean_s, n_evals=n_evals, ess_kish=ess_kish, sigma_trpl=sigma_trpl,
               ess_per_s=ess_kish / timed_mean_s if timed_mean_s else None,
               ess_per_1k_evals=1000.0 * ess_kish / n_evals if n_evals else None, note=note or None)
    return out


def efficiency_text(eff: dict) -> str:
    """'min ESS 8,205 (18 ESS/s); R-hat <= 1.004' for the nn_workflow column of the T8 table
    (importance samplers without chains: 'Kish ESS 3,275 (230 ESS/s)')."""
    if eff.get("min_ess_bulk") is None:
        s = f"Kish ESS {eff['ess_kish']:,.0f}"
    else:
        s = f"min ESS {eff['min_ess_bulk']:,.0f}"
    if eff.get("ess_per_s") is not None:
        v = eff["ess_per_s"]
        s += f" ({v:.0f} ESS/s)" if v >= 10 else f" ({v:.2g} ESS/s)"
    if eff.get("max_r_hat") is not None:
        s += f"; R-hat <= {eff['max_r_hat']:.3f}"
    return s


# ─────────────────────────────────────────────────────────────────────────────
# T8 rows of the samplers that run after main.ipynb (smc.ipynb, cylinder.ipynb)
# ─────────────────────────────────────────────────────────────────────────────
EXTERNAL_STEPS = ("posterior sampling: tempered SMC", "eigenvector cylinders", "sliced grid at the best fit")
EXTERNAL_SAMPLERS = ("smc", "cylinder IS")


def free_display_keys(run_dir):
    """Display keys of the free parameters of a main.ipynb run (prior_bounds.csv: fixed = degenerate interval)."""
    from pathlib import Path
    p = Path(run_dir) / "prior_bounds.csv"
    if not p.exists():
        return None
    b = pd.read_csv(p)
    free = b[(b["theta_hi"] - b["theta_lo"]) >= 1e-6]["parameter"]
    return [n.replace("Et_rel_", "Et_") + "_eV" if n.startswith("Et_rel_") else n for n in free]


def _timing_row(csv_path, prefix: str):
    t = pd.read_csv(csv_path)
    t = t[t["label"].astype(str).str.startswith(prefix)]
    if t.empty:
        return None
    r = t.iloc[0]
    return float(r.mean_s), float(r.std_s), int(r.repeats), (None if pd.isna(r.n_evals) else int(r.n_evals))


def external_t8_rows(run_dir, smc_dir=None, cylinder_dir=None, slices_dir=None):
    """
    T8 rows (and 11_sampler_efficiency rows) of tempered SMC (smc.ipynb) and of the eigenvector cylinders
    (cylinder.ipynb) of a finished main.ipynb run, in the format of main.ipynb Section 11.

    NN workflow: timeit mean ± SD of the sampler itself (XLA compilation excluded, as for NUTS).  ODE workflow:
    evaluations x (mean ± SD of one timed ODE solve of the run), as for emcee and the grid; mode 'sspl': the same
    as the NN workflow (the ss-PL model is exact, no surrogate).  Returns (t8_rows, efficiency_rows).
    """
    import json
    from pathlib import Path
    run_dir = Path(run_dir)
    mode = json.loads((run_dir / "settings.json").read_text())["mode"]
    tj = json.loads((run_dir / "11_timing.json").read_text()) if (run_dir / "11_timing.json").exists() else {}
    ode_s = tj.get("ode_seconds_per_solve")
    ode_row = _timing_row(run_dir / "11_timing.csv", "one ODE solve") if (run_dir / "11_timing.csv").exists() else None
    ode_sd = ode_row[1] if ode_row else 0.0
    fmt = format_mean_sd

    def unit(t):
        return "s" if t < 120 else "min"

    def ode_cols(n_evals, nn_mean, nn_sd):
        if mode == "sspl":
            return "same as NN workflow (exact ss-PL model, no surrogate)", nn_mean, nn_sd
        if not ode_s or not n_evals:
            return "—", None, None
        return (f"{fmt(n_evals * ode_s, n_evals * ode_sd, None, 'h', prefix='≈ ')} estimated (serial)",
                n_evals * ode_s, n_evals * ode_sd)

    t8, eff = [], []
    if smc_dir is not None:
        smc_dir = Path(smc_dir)
        sm = json.loads((smc_dir / "summary.json").read_text())
        tr = _timing_row(smc_dir / "8_timing.csv", "tempered SMC")
        comp = _timing_row(smc_dir / "8_timing.csv", "XLA compile")
        m, sd, n, evals = tr
        diag = pd.read_csv(smc_dir / "diagnostics_smc.csv", index_col=0)
        e = sampler_efficiency("smc", diag, sm["smc"], m, free_display_keys(run_dir))
        e["sigma_trpl"] = sm.get("sigma_trpl")
        o_txt, o_m, o_sd = ode_cols(evals, m, sd)
        nn_txt = (f"{fmt(m, sd, n, unit(m))}; {evals:,} evaluations; {efficiency_text(e)}; "
                  f"log Z = {sm['smc']['log_z']:.3f} ± {sm['smc']['log_z_se']:.3f}")
        if comp:
            nn_txt += f"; XLA compile {comp[0]:.0f} s excluded"
        t8.append({"stage": "per sample", "step": f"{EXTERNAL_STEPS[0]} (smc.ipynb, {smc_dir.name})",
                   "nn_workflow": nn_txt, "ode_workflow": o_txt, "nn_mean_s": m, "nn_sd_s": sd, "n_repeats": n,
                   "ode_mean_s": o_m, "ode_sd_s": o_sd,
                   **{k: v for k, v in e.items() if k not in ("sampler", "timed_mean_s", "n_evals", "note")}})
        eff.append(e)
    if cylinder_dir is not None:
        cylinder_dir = Path(cylinder_dir)
        cs = json.loads((cylinder_dir / "summary.json").read_text())
        tr = _timing_row(cylinder_dir / "9_timing.csv", "cylinder points")
        m, sd, n, evals = tr
        sig_run = json.loads((run_dir / "6_best_fit.json").read_text())["sigmas"]
        notes = []
        for term, name in (("trpl", "trPL"), ("sspl", "ssPL")):       # the sigma of every term of this mode
            if (term == "trpl" and mode == "sspl") or (term == "sspl" and mode == "trpl"):
                continue
            sig, sig_m = cs.get(f"sigma_{term}"), sig_run.get(term)
            if sig is not None:
                notes.append(f"sigma_{name} = {sig:.4f}" + ("" if sig_m is None or abs(sig - sig_m) < 1e-6 * sig_m
                                                            else f" (cylinder sigma; main run: {sig_m:.4f})"))
        sig_note = "; ".join(notes)
        e = importance_efficiency("cylinder IS", cs["ess"], evals, m, cs.get("sigma_trpl") if mode != "sspl" else None,
                                  sig_note)
        o_txt, o_m, o_sd = ode_cols(evals, m, sd)
        nn_txt = f"{fmt(m, sd, n, unit(m))}; {evals:,} evaluations; {efficiency_text(e)}"
        if sig_note:
            nn_txt += f"; {sig_note}"
        t8.append({"stage": "per sample", "step": f"{EXTERNAL_STEPS[1]} (importance sampling, cylinder.ipynb, {cylinder_dir.name})",
                   "nn_workflow": nn_txt, "ode_workflow": o_txt, "nn_mean_s": m, "nn_sd_s": sd, "n_repeats": n,
                   "ode_mean_s": o_m, "ode_sd_s": o_sd,
                   **{k: v for k, v in e.items() if k not in ("sampler", "timed_mean_s", "n_evals", "note")}})
        eff.append(e)
    if slices_dir is not None:                    # scripts/paper_slices.py: 2-D slices of the original submission
        slices_dir = Path(slices_dir)
        ss = json.loads((slices_dir / "summary.json").read_text())
        m, sd, n = ss["time"]["mean_s"], ss["time"]["std_s"], len(ss["time"]["times_s"])
        evals = int(ss["n_evals"])
        o_txt, o_m, o_sd = ode_cols(evals if mode != "sspl" else None, m, sd)
        if mode == "sspl":
            o_txt, o_m, o_sd = "same as NN workflow (exact ss-PL model, no surrogate)", m, sd
        t8.append({"stage": "per sample",
                   "step": f"{EXTERNAL_STEPS[2]} (paper method: 2-D slices, {ss['n_per_axis']}^2 points per pair; "
                           f"scripts/paper_slices.py)",
                   "nn_workflow": f"{fmt(m, sd, n, unit(m))}; {evals:,} evaluations; no samples (slices only)",
                   "ode_workflow": o_txt, "nn_mean_s": m, "nn_sd_s": sd, "n_repeats": n, "ode_mean_s": o_m,
                   "ode_sd_s": o_sd})
    return t8, eff


def update_t8(run_dir, smc_dir=None, cylinder_dir=None, slices_dir=None) -> pd.DataFrame:
    """
    Add (or replace) the SMC and cylinder rows in 11_T8_timing_table.csv and 11_sampler_efficiency.csv of a
    main.ipynb run.  Called at the end of smc.ipynb / cylinder.ipynb; idempotent.  Returns the new T8 table.
    """
    from pathlib import Path
    run_dir = Path(run_dir)
    rows, eff = external_t8_rows(run_dir, smc_dir, cylinder_dir, slices_dir)
    p8, pe = run_dir / "11_T8_timing_table.csv", run_dir / "11_sampler_efficiency.csv"
    t8 = pd.read_csv(p8) if p8.exists() else pd.DataFrame()
    for r in rows:
        if not t8.empty:
            t8 = t8[~t8["step"].astype(str).str.startswith(r["step"].split(" (")[0])]
        t8 = pd.concat([t8, pd.DataFrame([r])], ignore_index=True)
    t8.to_csv(p8, index=False)
    se = pd.read_csv(pe) if pe.exists() else pd.DataFrame(columns=EFFICIENCY_COLUMNS)
    se = se.reindex(columns=EFFICIENCY_COLUMNS)
    for e in eff:
        se = pd.concat([se[se["sampler"] != e["sampler"]], pd.DataFrame([e], columns=EFFICIENCY_COLUMNS)],
                       ignore_index=True)
    se.to_csv(pe, index=False)
    return t8
