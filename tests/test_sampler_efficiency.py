"""Tests of inference.benchmark.sampler_efficiency (T8 sampler-efficiency columns)."""
import pandas as pd
import pytest

from inference.benchmark import EFFICIENCY_COLUMNS, efficiency_text, sampler_efficiency

DIAG = pd.DataFrame({"ess_bulk": [27000.0, 5000.0, 8000.0], "ess_tail": [26000.0, 6000.0, 4000.0],
                     "r_hat": [1.0001, 1.004, 1.002]}, index=["log10_krad", "log10_Nt_1", "Et_1_eV"])


def test_nuts_row_free_parameters_only():
    info = {"evals_total_points": 1_000_000, "evals_total_calls": 30_000, "n_warmup": 1000, "n_draws": 2000,
            "time_sampling_s": 100.0, "step_size": 0.05, "mean_leapfrogs": 7.0, "n_divergent": 3,
            "compile_s": 20.0, "time_warmup_s": 80.0}
    e = sampler_efficiency("nuts", DIAG, info, 250.0, free_keys=["log10_Nt_1", "Et_1_eV"])   # k_rad fixed
    assert list(e) == EFFICIENCY_COLUMNS
    assert e["min_ess_bulk"] == 5000.0 and e["min_ess_tail"] == 4000.0 and e["max_r_hat"] == 1.004
    assert e["ess_per_s"] == pytest.approx(20.0) and e["ess_per_s_sampling"] == pytest.approx(50.0)
    assert e["ess_per_1k_evals"] == pytest.approx(5.0) and e["calls_per_iteration"] == pytest.approx(10.0)
    assert e["max_tau_int"] is None and efficiency_text(e) == "min ESS 5,000 (20 ESS/s); R-hat <= 1.004"


def test_emcee_row():
    info = {"evals_total_points": 2_000_000, "tau_int": [100.0, 499.0], "acceptance_fraction": 0.38}
    e = sampler_efficiency("emcee", DIAG, info, 200.0)
    assert e["ess_per_s_sampling"] is None and e["step_size"] is None
    assert e["max_tau_int"] == 499.0 and e["acceptance_fraction"] == 0.38 and e["min_ess_bulk"] == 5000.0
