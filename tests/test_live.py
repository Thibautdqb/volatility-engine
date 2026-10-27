import math

import numpy as np
import pandas as pd
import pytest

from volengine.deribit.stream import DvolUpdate, Trade, parse_message
from volengine.estimators import ewma_variance
from volengine.live import BarAggregator, LiveVolMonitor, StreamingEWMA

TRADES_MSG = {
    "jsonrpc": "2.0",
    "method": "subscription",
    "params": {
        "channel": "trades.BTC-PERPETUAL.100ms",
        "data": [
            {
                "timestamp": 1793130317626,
                "price": 80737.0,
                "direction": "buy",
                "index_price": 80715.5,
                "instrument_name": "BTC-PERPETUAL",
                "amount": 10000.0,
                "mark_price": 80739.72,
                "trade_id": "456928653",
            },
            {
                "timestamp": 1793130317700,
                "price": 80736.5,
                "direction": "sell",
                "index_price": 80715.5,
                "instrument_name": "BTC-PERPETUAL",
                "amount": 20.0,
                "mark_price": 80739.70,
                "trade_id": "456928654",
            },
        ],
    },
}
DVOL_MSG = {
    "jsonrpc": "2.0",
    "method": "subscription",
    "params": {
        "channel": "deribit_volatility_index.btc_usd",
        "data": {"timestamp": 1793124671248, "index_name": "btc_usd", "volatility": 37.36},
    },
}


def test_parse_trades():
    events = parse_message(TRADES_MSG)
    assert [type(e) for e in events] == [Trade, Trade]
    assert events[0].instrument == "BTC-PERPETUAL"
    assert events[0].price == 80737.0
    assert events[1].timestamp == pytest.approx(1793130317.7)


def test_parse_dvol_and_ignore_others():
    (ev,) = parse_message(DVOL_MSG)
    assert isinstance(ev, DvolUpdate) and ev.value == 37.36
    assert parse_message({"jsonrpc": "2.0", "id": 1, "result": []}) == []


def test_bar_aggregator_ohlc_and_gaps():
    agg = BarAggregator(10)
    assert agg.update(100.0, 1.0) == []
    agg.update(103.0, 3.0)
    agg.update(105.0, 0.5)
    agg.update(109.9, 2.0)
    done = agg.update(135.0, 4.0)  # skips the [120, 130) bucket
    assert [b.start for b in done] == [100.0, 110.0, 120.0]
    first = done[0]
    assert (first.open, first.high, first.low, first.close) == (1.0, 3.0, 0.5, 2.0)
    assert all(b.open == b.close == 2.0 for b in done[1:])  # flat filler bars
    assert agg.update(99.0, 100.0) == []  # late trade ignored


def test_bar_aggregator_skips_bars_already_seeded():
    agg = BarAggregator(60, after=120.0)  # bar [120, 180) is known from history
    assert agg.update(150.0, 1.0) == []
    agg.update(185.0, 2.0)
    (bar,) = agg.update(245.0, 3.0)
    assert bar.start == 180.0 and bar.open == 2.0


def test_streaming_ewma_equals_batch():
    rng = np.random.default_rng(0)
    r = rng.normal(0, 0.01, 500)
    s = StreamingEWMA(0.97, variance=1e-4)
    for x in r[:-1]:
        s.update(x)
    batch = ewma_variance(r, 0.97, init=1e-4)
    assert s.variance == pytest.approx(batch[-1], rel=1e-12)


def test_half_life():
    s = StreamingEWMA.from_half_life(10)
    assert s.lam**10 == pytest.approx(0.5)


def test_monitor_end_to_end_without_network():
    m = LiveVolMonitor(["BTC-PERPETUAL"], bar_seconds=1.0, window=200, half_life_seconds=30)
    rng = np.random.default_rng(1)
    sigma_bar = 1e-3
    price, t = 100.0, 1_000_000.0
    for _ in range(600):
        for k in range(5):
            m.on_event(Trade("BTC-PERPETUAL", t + 0.2 * k, price, 1.0))
        price *= math.exp(rng.normal(0, sigma_bar))
        t += 1.0
    m.on_event(DvolUpdate("btc_usd", t, 50.0))
    snap = m.snapshot("BTC-PERPETUAL")
    true_annual = sigma_bar * math.sqrt(m.periods)
    assert snap.bars == 200
    assert snap.dvol == pytest.approx(0.5)
    assert snap.cc_lower < true_annual < snap.cc_upper
    assert snap.ewma == pytest.approx(true_annual, rel=0.4)
    assert len(m.history("BTC-PERPETUAL")) > 500


def test_dvol_indices_mapping():
    m = LiveVolMonitor(["BTC-PERPETUAL", "ETH-PERPETUAL", "SOL_USDC-PERPETUAL"])
    assert m.dvol_indices == ["btc_usd", "eth_usd"]


class _FakeREST:
    """Serves 1-minute bars ending at the current minute, which is still in progress."""

    def __init__(self, now: pd.Timestamp):
        self.now = now

    def ohlc(self, instrument, resolution, start, end):
        idx = pd.date_range(end=self.now.floor("min"), periods=300, freq="min")
        rng = np.random.default_rng(0)
        close = 100 * np.exp(np.cumsum(rng.normal(0, 1e-3, idx.size)))
        return pd.DataFrame(
            {"open": close, "high": close * 1.0005, "low": close * 0.9995, "close": close},
            index=idx,
        )


def test_seed_excludes_bar_in_progress_and_live_continues():
    now = pd.Timestamp.now(tz="UTC")
    m = LiveVolMonitor(["BTC-PERPETUAL"], bar_seconds=60, window=120)
    m.seed(_FakeREST(now))
    st = m._state["BTC-PERPETUAL"]
    last_seeded = st.bars[-1].start
    assert last_seeded == (now.floor("min") - pd.Timedelta(minutes=1)).timestamp()
    # Live trades inside the current minute and the next one: exactly one new bar,
    # starting right after the last seeded bar, no duplicate timestamps.
    t0 = now.floor("min").timestamp()
    m.on_event(Trade("BTC-PERPETUAL", t0 + 5, 100.0, 1.0))
    m.on_event(Trade("BTC-PERPETUAL", t0 + 65, 100.1, 1.0))
    starts = [b.start for b in st.bars]
    assert starts[-1] == t0 and len(starts) == len(set(starts))
