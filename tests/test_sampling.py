import numpy as np
import pandas as pd
import pytest

from tests.conftest import simulate_trades
from volengine.sampling import GridSampler, bar_closes, log_returns


def stream(trades, bar_seconds):
    s = GridSampler(bar_seconds)
    out = []
    for ts, px in zip(trades["timestamp"], trades["price"], strict=True):
        out += s.update(int(ts), float(px))
    return out, s


def test_streaming_matches_batch():
    # Sparse trading: many 1-second bars have no trade and must carry the last price.
    trades = simulate_trades(3600, 0.5, rate=0.3, seed=3)
    batch = bar_closes(trades, 1.0)
    done, s = stream(trades, 1.0)
    live = pd.Series(
        [c for _, c, _ in done] + [s.close],
        index=pd.to_datetime([t for t, _, _ in done] + [s.bucket * s.step], unit="ms", utc=True),
    )
    pd.testing.assert_series_equal(batch, live, check_names=False, check_freq=False)
    assert sum(n for *_, n in done) + s.trades == len(trades)


def test_empty_bars_late_trades_and_advance():
    s = GridSampler(10)
    assert s.update(1_000, 1.0) == []
    s.update(3_000, 2.0)
    done = s.update(35_000, 4.0)  # bars [10s, 20s) and [20s, 30s) had no trade
    assert done == [(0, 2.0, 2), (10_000, 2.0, 0), (20_000, 2.0, 0)]
    assert s.update(5_000, 9.0) == []  # late print for a closed bar
    assert s.advance(39_999) == []
    assert s.advance(41_000) == [(30_000, 4.0, 1)]
    assert s.update(42_000, 5.0) == []


def test_resume_from_state():
    s = GridSampler(10, bucket=5, close=100.0)
    assert s.update(45_000, 99.0) == []  # bucket 4 is already in the history
    assert s.update(61_000, 101.0) == [(50_000, 100.0, 0)]


def test_bar_closes_grid_bounds():
    trades = pd.DataFrame({"timestamp": [5_000, 25_000], "price": [1.0, np.e]})
    closes = bar_closes(
        trades,
        10,
        start=pd.Timestamp(0, unit="ms", tz="UTC"),
        end=pd.Timestamp(40_000, unit="ms", tz="UTC"),
    )
    assert closes.tolist() == [1.0, 1.0, np.e, np.e]
    assert log_returns(closes).tolist() == pytest.approx([0.0, 1.0, 0.0])
