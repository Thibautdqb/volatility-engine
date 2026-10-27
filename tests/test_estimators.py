import numpy as np
import pandas as pd
import pytest

from tests.conftest import simulate_gbm_ohlc
from volengine import estimators as est


@pytest.fixture(scope="module")
def gbm():
    return simulate_gbm_ohlc(3000, 0.60, seed=11)


@pytest.mark.parametrize(
    "name", ["close_to_close", "parkinson", "garman_klass", "rogers_satchell", "yang_zhang"]
)
def test_estimators_are_consistent_under_gbm(gbm, name):
    df, sigma_bar = gbm
    var = est.ESTIMATORS[name](df)
    # Range estimators are biased low by discrete monitoring (500 steps/bar -> ~3%).
    assert np.sqrt(var) == pytest.approx(sigma_bar, rel=0.06)


def test_range_estimators_are_more_efficient():
    """Across samples, Garman-Klass dispersion should be well below close-to-close."""
    cc, gk = [], []
    for seed in range(60):
        df, _ = simulate_gbm_ohlc(60, 0.5, steps_per_bar=200, seed=seed)
        cc.append(est.ESTIMATORS["close_to_close"](df))
        gk.append(est.garman_klass(df))
    assert np.var(gk) < 0.4 * np.var(cc)


def test_rolling_matches_point_estimate(gbm):
    df, _ = gbm
    periods = 365.0
    for name in ["parkinson", "garman_klass", "rogers_satchell", "yang_zhang"]:
        roll = est.rolling_volatility(df, 50, periods, name)
        point = np.sqrt(
            est.ESTIMATORS[name](df.iloc[-51:] if name == "yang_zhang" else df.iloc[-50:]) * periods
        )
        assert roll.iloc[-1] == pytest.approx(point, rel=1e-10)


def test_chi2_interval_coverage_under_iid_normal():
    rng = np.random.default_rng(0)
    cover = 0
    for _ in range(400):
        r = rng.normal(0, 0.02, 100)
        ci = est.volatility_ci(r, periods=1.0, method="chi2")
        cover += ci.lower <= 0.02 <= ci.upper
    assert 0.92 < cover / 400 < 0.98


def test_bootstrap_interval_contains_estimate(garch_returns):
    ci = est.volatility_ci(garch_returns, periods=365.0)
    assert ci.lower < ci.vol < ci.upper


def test_ewma_forecast_uses_only_past_information():
    r = np.array([0.0, 0.0, 0.1, 0.0])
    f = est.ewma_variance(r, 0.9, init=0.0)
    assert f[2] == 0.0  # the shock at t=2 is not yet known
    assert f[3] == pytest.approx(0.1 * 0.01)


def test_fit_ewma_lambda_is_in_bounds_and_beats_extremes(garch_returns):
    lam = est.fit_ewma_lambda(garch_returns)
    assert 0.8 < lam < 0.999
    r2 = garch_returns**2

    def loss(x):
        return est.qlike(r2[30:], est.ewma_variance(garch_returns, x)[30:]).mean()

    assert loss(lam) <= min(loss(0.8), loss(0.999))


def test_signature_plot_is_flat_without_noise():
    rng = np.random.default_rng(1)
    close = pd.Series(np.exp(np.cumsum(rng.normal(0, 1e-3, 20000))))
    sig = est.signature_plot(close, 60, steps=(1, 5, 15))
    assert sig.max() / sig.min() < 1.1


def test_signature_plot_detects_bid_ask_bounce():
    rng = np.random.default_rng(2)
    efficient = np.cumsum(rng.normal(0, 1e-4, 20000))
    noisy = efficient + rng.choice([-1, 1], efficient.size) * 3e-4
    sig = est.signature_plot(pd.Series(np.exp(noisy)), 60, steps=(1, 30))
    assert sig.iloc[0] > 2 * sig.iloc[1]


def test_volatility_cone_shape(gbm):
    df, _ = gbm
    cone = est.volatility_cone(df["close"], windows=(5, 21))
    assert list(cone.index) == [5, 21]
    assert (cone["q05"] <= cone["q50"]).all() and (cone["q50"] <= cone["q95"]).all()


def test_periods_per_year():
    assert est.periods_per_year(86400) == pytest.approx(365.0)
    assert est.periods_per_year(60) == pytest.approx(525600.0)
