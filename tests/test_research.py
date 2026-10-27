import numpy as np
import pandas as pd
import pytest

from volengine.research import DailyData, daily_from_hourly, variance_risk_premium


def _hourly(n_hours, start="2024-01-01 00:00", seed=0, sigma=0.005):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n_hours, freq="h", tz="UTC")
    close = 100 * np.exp(np.cumsum(rng.normal(0, sigma, n_hours)))
    return pd.DataFrame({"open": close, "high": close, "low": close, "close": close}, index=idx)


def test_sessions_start_at_0800_and_are_complete():
    h = _hourly(24 * 10)
    ohlc, ret, rv = daily_from_hourly(h)
    assert (ret.index.hour == 8).all()
    # First hourly return is at 01:00 on day 1; the first complete session starts day 1 08:00
    assert ret.index[0] == pd.Timestamp("2024-01-01 08:00", tz="UTC")
    # Last session (day 10 08:00 -> day 11 08:00) is incomplete and must be dropped.
    assert ret.index[-1] == pd.Timestamp("2024-01-09 08:00", tz="UTC")
    assert ret.index.equals(rv.index) and ret.index.equals(ohlc.index)


def test_session_return_matches_close_to_close():
    h = _hourly(24 * 5)
    _, ret, rv = daily_from_hourly(h)
    s = ret.index[1]
    end = s + pd.Timedelta(hours=23)
    prev = s - pd.Timedelta(hours=1)
    expected = np.log(h.loc[end, "close"] / h.loc[prev, "close"])
    assert ret.loc[s] == pytest.approx(expected)
    assert (rv > 0).all()


def test_vrp_detects_known_premium():
    n = 900
    idx = pd.date_range("2022-01-01 08:00", periods=n, freq="D", tz="UTC")
    rng = np.random.default_rng(1)
    true_vol = 0.5 + 0.2 * np.sin(np.arange(n) / 60)
    rv = pd.Series((true_vol**2 / 365) * rng.chisquare(24, n) / 24, index=idx)
    implied = pd.Series(true_vol + 0.05, index=idx)
    data = DailyData(pd.DataFrame(index=idx), pd.Series(0.0, index=idx), rv, implied)
    res = variance_risk_premium(data)
    assert res.vol_spread.mean == pytest.approx(0.05, abs=0.01)
    assert res.vol_spread.p_value < 0.01
    assert res.vol_spread.method == "hansen-hodrick"
    assert res.vol_spread_independent.p_value < 0.05


def test_vrp_test_size_without_premium():
    """With implied variance equal to expected realized variance, the test should
    reject about 5% of the time, not more."""
    n, sims = 900, 150
    idx = pd.date_range("2022-01-01 08:00", periods=n, freq="D", tz="UTC")
    rejections = 0
    for seed in range(sims):
        rng = np.random.default_rng(1000 + seed)
        rv = pd.Series((0.36 / 365) * rng.chisquare(24, n) / 24, index=idx)
        implied = pd.Series(0.6 + rng.normal(0, 0.01, n), index=idx)
        data = DailyData(pd.DataFrame(index=idx), pd.Series(0.0, index=idx), rv, implied)
        rejections += variance_risk_premium(data).variance_spread.p_value < 0.05
    assert rejections / sims < 0.10
