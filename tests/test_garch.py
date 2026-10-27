import numpy as np
import pytest
from arch import arch_model

from tests.conftest import simulate_garch
from volengine.garch import fit_garch, garch_filter


def test_matches_arch_package(garch_returns):
    ours = fit_garch(garch_returns)
    ref = arch_model(garch_returns * 100, mean="Zero", vol="GARCH", p=1, q=1, rescale=False).fit(
        disp="off"
    )
    assert ours.params["alpha"] == pytest.approx(ref.params["alpha[1]"], abs=2e-3)
    assert ours.params["beta"] == pytest.approx(ref.params["beta[1]"], abs=2e-3)
    assert ours.params["omega"] * 1e4 == pytest.approx(ref.params["omega"], rel=0.02)
    # arch reports robust (sandwich) standard errors by default.
    assert ours.std_err["alpha"] == pytest.approx(ref.std_err["alpha[1]"], rel=0.05)
    assert ours.std_err["beta"] == pytest.approx(ref.std_err["beta[1]"], rel=0.05)
    loglik_ref = ref.loglikelihood + garch_returns.size * np.log(100)
    assert ours.loglik == pytest.approx(loglik_ref, abs=0.5)


def test_parameter_recovery_with_mean():
    r = simulate_garch(5000, 1e-5, 0.10, 0.85, seed=4) + 5e-4
    res = fit_garch(r, mean=True)
    assert res.converged
    assert res.params["alpha"] == pytest.approx(0.10, abs=0.03)
    assert res.params["beta"] == pytest.approx(0.85, abs=0.04)
    assert abs(res.tvalues["alpha"]) > 3


def test_residual_diagnostics_clean_on_correct_model(garch_returns):
    res = fit_garch(garch_returns)
    assert res.diagnostics["ljung_box_z2"].p_value > 0.01
    assert res.diagnostics["arch_lm_z"].p_value > 0.01
    assert res.diagnostics["jarque_bera_z"].p_value < 0.01  # simulated with t(6) shocks


def test_forecast_reverts_to_unconditional(garch_returns):
    res = fit_garch(garch_returns)
    f = res.forecast(2000)
    assert f[0] == pytest.approx(res.next_variance)
    assert f[-1] == pytest.approx(res.unconditional_variance, rel=1e-3)
    assert res.half_life == pytest.approx(np.log(0.5) / np.log(res.persistence))


def test_filter_reproduces_fitted_variance(garch_returns):
    res = fit_garch(garch_returns)
    p = res.params
    s0 = res.conditional_variance.iloc[0]
    path = garch_filter(garch_returns, p["omega"], p["alpha"], p["beta"], s0=s0)
    np.testing.assert_allclose(path[:-1], res.conditional_variance.to_numpy(), rtol=1e-10)
    assert path[-1] == pytest.approx(res.next_variance)


def test_rejects_nan():
    with pytest.raises(ValueError):
        fit_garch(np.array([0.01, np.nan, 0.02] * 100))
