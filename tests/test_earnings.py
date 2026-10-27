import numpy as np
import pandas as pd
import pytest

pytest.importorskip("yfinance")

from volengine.earnings.data import classify_timing
from volengine.earnings.implied import (
    SQRT_2_OVER_PI,
    atm_straddle,
    bs_price,
    event_variance,
    straddle_implied_vol,
    trading_days_between,
)
from volengine.earnings.moves import event_windows, historical_moves
from volengine.earnings.screener import edge_statistics

NY = "America/New_York"


@pytest.mark.parametrize(
    ("hour", "minute", "expected"),
    [
        (7, 0, "BMO"),
        (9, 0, "BMO"),
        (16, 0, "AMC"),
        (16, 30, "AMC"),
        (12, 0, "unknown"),
        (0, 0, "unknown"),
    ],
)
def test_classify_timing(hour, minute, expected):
    assert classify_timing(pd.Timestamp(2024, 5, 2, hour, minute, tz=NY)) == expected


def test_event_windows():
    sessions = pd.bdate_range("2024-04-29", "2024-05-10")
    ev = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-05-02", "2024-05-02", "2024-05-02", "2024-05-04"]),
            "timing": ["AMC", "BMO", "unknown", "BMO"],  # 05-04 is a Saturday
        }
    )
    w = event_windows(ev, sessions)
    d = pd.Timestamp
    assert (w.loc[0, "entry"], w.loc[0, "reaction"]) == (d("2024-05-02"), d("2024-05-03"))
    assert (w.loc[1, "entry"], w.loc[1, "reaction"]) == (d("2024-05-01"), d("2024-05-02"))
    assert (w.loc[2, "entry"], w.loc[2, "reaction"]) == (d("2024-05-01"), d("2024-05-03"))
    assert (w.loc[3, "entry"], w.loc[3, "reaction"]) == (d("2024-05-03"), d("2024-05-06"))


def test_historical_moves_measures_the_jump():
    idx = pd.bdate_range("2023-01-02", periods=300)
    rng = np.random.default_rng(0)
    r = rng.normal(0, 0.01, idx.size)
    jump_day = idx[200]
    r[200] += 0.10
    prices = pd.DataFrame({"close": 100 * np.exp(np.cumsum(r))}, index=idx)
    events = pd.DataFrame({"date": [jump_day], "timing": ["BMO"]})
    m = historical_moves(prices, events, holding=3, lead=1)
    row = m.iloc[0]
    assert row["entry"] == idx[199] and row["reaction"] == jump_day
    assert np.log1p(row["reaction_move"]) == pytest.approx(r[200])
    expected_holding = np.exp(r[199:202].sum()) - 1  # from idx[198] close to idx[201] close
    assert row["holding_move"] == pytest.approx(expected_holding)
    assert row["z_reaction"] > 5
    # pre_vol must only use returns up to the start of the holding window
    assert row["pre_vol"] == pytest.approx(np.sqrt(np.mean(r[178:199] ** 2)))


def test_holding_window_must_cover_reaction():
    idx = pd.bdate_range("2023-01-02", periods=100)
    prices = pd.DataFrame({"close": np.linspace(100, 110, idx.size)}, index=idx)
    events = pd.DataFrame({"date": [idx[50]], "timing": ["AMC"]})
    assert historical_moves(prices, events, holding=1, lead=2).empty


def test_put_call_parity():
    S, K, T, s, r, q = 100, 95, 0.3, 0.4, 0.03, 0.01
    c = bs_price(S, K, T, s, r, q, "call")
    p = bs_price(S, K, T, s, r, q, "put")
    assert c - p == pytest.approx(S * np.exp(-q * T) - K * np.exp(-r * T))


def test_straddle_iv_round_trip():
    S, K, T, s = 100, 100, 10 / 365, 0.65
    price = bs_price(S, K, T, s, 0.04, 0, "call") + bs_price(S, K, T, s, 0.04, 0, "put")
    assert straddle_implied_vol(price, S, K, T, 0.04) == pytest.approx(s, abs=1e-6)
    # ATM straddle / spot ~ sqrt(2/pi) sigma sqrt(T), the expected absolute move
    assert price / S == pytest.approx(SQRT_2_OVER_PI * s * np.sqrt(T), rel=0.02)


def test_atm_straddle_interpolates_between_strikes():
    strikes = np.array([95.0, 100.0, 105.0])
    calls = pd.DataFrame(
        {"strike": strikes, "bid": [6.0, 3.0, 1.0], "ask": [6.2, 3.2, 1.2], "lastPrice": 0}
    )
    puts = pd.DataFrame(
        {"strike": strikes, "bid": [1.0, 3.0, 6.0], "ask": [1.2, 3.2, 6.2], "lastPrice": 0}
    )
    s = atm_straddle(calls, puts, 101.0)
    assert s.price == pytest.approx(0.8 * 6.2 + 0.2 * 7.2)
    assert not s.stale and 0 < s.spread_pct < 0.1


def test_atm_straddle_flags_stale_quotes():
    calls = pd.DataFrame({"strike": [100.0], "bid": [0.0], "ask": [0.0], "lastPrice": [3.0]})
    puts = pd.DataFrame({"strike": [100.0], "bid": [0.0], "ask": [0.0], "lastPrice": [2.5]})
    s = atm_straddle(calls, puts, 100.0)
    assert s.stale and s.price == 5.5


def test_event_variance_recovers_jump():
    daily, e = 0.015**2, 0.06
    n1, n2 = 3, 8
    ev, d = event_variance(daily * n1 + e**2, n1, daily * n2 + e**2, n2)
    assert np.sqrt(ev) == pytest.approx(e)
    assert d == pytest.approx(daily)


def test_event_variance_rejects_inconsistent_term_structure():
    ev, _ = event_variance(0.001, 3, 0.0005, 8)
    assert np.isnan(ev)


def test_trading_days_between_skips_weekend_and_holiday():
    assert trading_days_between(pd.Timestamp("2024-07-03"), pd.Timestamp("2024-07-08")) == 2
    assert trading_days_between(pd.Timestamp("2024-05-03"), pd.Timestamp("2024-05-06")) == 1


def _moves(z, pre_vol=0.01):
    z = np.asarray(z)
    return pd.DataFrame(
        {"holding_move": z * pre_vol, "z_holding": z, "z_reaction": z, "pre_vol": pre_vol}
    )


def test_edge_statistics_fair_pricing_not_significant():
    rng = np.random.default_rng(3)
    z = np.abs(rng.normal(0, 3, 40))
    implied = 3 * SQRT_2_OVER_PI * 0.01  # fair: E|N(0, 3 sigma)|
    st = edge_statistics(_moves(z), implied, daily_vol=0.01)
    assert st.adj_edge_ci[0] < 0 < st.adj_edge_ci[1]
    assert st.adj_edge_p > 0.05
    assert not st.small_sample


def test_edge_statistics_detects_overpriced_straddle():
    rng = np.random.default_rng(4)
    z = np.abs(rng.normal(0, 3, 50))
    implied = 2.0 * 3 * SQRT_2_OVER_PI * 0.01  # market prices twice the typical move
    st = edge_statistics(_moves(z), implied, daily_vol=0.01)
    assert st.adj_edge < 0 and st.adj_edge_ci[1] < 0 and st.adj_edge_p < 0.01
    assert st.hit_rate < st.hit_rate_ci[1]


def test_edge_statistics_rescales_by_current_vol():
    z = np.full(30, 2.0)
    st = edge_statistics(_moves(z), 0.04, daily_vol=0.02)
    assert st.predicted_move == pytest.approx(0.04)


def test_small_sample_flag():
    st = edge_statistics(_moves([1.0, 2.0, 3.0, 2.5, 1.5]), 0.02, 0.01)
    assert st.small_sample
