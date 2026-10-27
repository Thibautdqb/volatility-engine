import math

import numpy as np
import pytest

from volengine.deribit.stream import DvolUpdate, Tick, parse_message
from volengine.estimators import ewma_variance
from volengine.live import BarAggregator, LiveVolMonitor, StreamingEWMA

TICKER_MSG = {
    "jsonrpc": "2.0",
    "method": "subscription",
    "params": {
        "channel": "ticker.BTC-PERPETUAL.100ms",
        "data": {
            "timestamp": 1793124672157,
            "instrument_name": "BTC-PERPETUAL",
            "mark_price": 80471.12,
            "index_price": 80440.5,
            "best_bid_price": 80471.0,
            "best_ask_price": 80471.5,
            "state": "open",
        },
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


def test_parse_ticker():
    ev = parse_message(TICKER_MSG)
    assert isinstance(ev, Tick)
    assert ev.instrument == "BTC-PERPETUAL"
    assert ev.mark_price == 80471.12
    assert ev.timestamp == pytest.approx(1793124672.157)


def test_parse_dvol_and_ignore_others():
    ev = parse_message(DVOL_MSG)
    assert isinstance(ev, DvolUpdate) and ev.value == 37.36
    assert parse_message({"jsonrpc": "2.0", "id": 1, "result": []}) is None


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
    assert agg.update(99.0, 100.0) == []  # late tick ignored


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
            m.on_event(Tick("BTC-PERPETUAL", t + 0.2 * k, price, price, None, None))
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
