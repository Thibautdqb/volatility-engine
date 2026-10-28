import asyncio
import json

import numpy as np
import pandas as pd
import pytest

from tests.conftest import simulate_trades
from volengine.deribit.stream import DvolUpdate, Reconnected, Trade, parse_message
from volengine.detector import DetectorConfig, Level, run_detector
from volengine.forecast import Regime, har_inputs, prepare
from volengine.live import LiveMonitor
from volengine.sampling import bar_closes, bar_counts, log_returns

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
INSTRUMENT = "BTC-PERPETUAL"
CONFIG = DetectorConfig(bar_seconds=10, fast_half_life=60, slow_half_life=600)


def test_parse_messages():
    trades = parse_message(TRADES_MSG)
    assert [type(e) for e in trades] == [Trade, Trade]
    assert trades[0] == Trade(INSTRUMENT, 1793130317626, 80737.0, 10000.0, "456928653")
    (dvol,) = parse_message(DVOL_MSG)
    assert isinstance(dvol, DvolUpdate) and dvol.value == 37.36
    assert parse_message({"jsonrpc": "2.0", "id": 1, "result": []}) == []


def as_events(df):
    return [
        Trade(INSTRUMENT, int(r.timestamp), float(r.price), float(r.amount), str(r.trade_id))
        for r in df.itertuples()
    ]


def tape(hours=3, seed=0, vol=0.5):
    return simulate_trades(hours * 3600, vol, rate=2.0, seed=seed)


def test_seed_then_live_equals_batch_replay():
    trades = tape()
    cut = trades["timestamp"].iloc[0] + 2 * 3600 * 1000 + 4_321  # mid-bar
    hist = trades[trades["timestamp"] < cut]
    # The live stream starts a little before the cut: those prints must not count twice.
    live = trades[trades["timestamp"] >= cut - 3_000]

    m = LiveMonitor([INSTRUMENT], CONFIG)
    m.apply_seed({INSTRUMENT: hist})
    for ev in as_events(live):
        m.on_event(ev)

    got = m.history(INSTRUMENT)
    assert got.index.is_unique and got.index.is_monotonic_increasing
    assert (got.index.to_series().diff().iloc[1:] == pd.Timedelta(seconds=10)).all()  # no gap

    batch = run_detector(log_returns(bar_closes(trades, 10)), CONFIG)
    batch.index = batch.index + pd.Timedelta(seconds=10)  # live rows are stamped at bar end
    common = got.index.intersection(batch.index)
    assert len(common) == len(got)  # both drop the warm-up
    for col in ("fast_vol", "slow_vol", "ratio"):
        np.testing.assert_allclose(got.loc[common, col], batch.loc[common, col], rtol=1e-9)

    # The streaming forecast equals the batch features run through the same model.
    closes = bar_closes(trades, 10)
    feats = prepare(log_returns(closes), bar_counts(trades, closes, 10), CONFIG)
    feats.index = feats.index + pd.Timedelta(seconds=10)
    model = m.models[INSTRUMENT]
    expected = np.exp(model.predict(har_inputs(feats, model))) * feats["slow_vol"]
    expected = expected.loc[common].iloc[200:]  # after the 30-minute look-back fills
    np.testing.assert_allclose(got.loc[expected.index, "forecast_vol"], expected, rtol=1e-9)


def test_alert_on_volatility_burst():
    vol = np.r_[np.full(3 * 3600, 0.4), np.full(600, 2.0)]
    trades = simulate_trades(vol.size, vol, rate=2.0, seed=1)
    cut = trades["timestamp"].iloc[0] + 3 * 3600 * 1000
    received = []
    m = LiveMonitor([INSTRUMENT], DetectorConfig(), on_alert=received.append)
    m.apply_seed({INSTRUMENT: trades[trades["timestamp"] < cut]})
    assert m.snapshot(INSTRUMENT).level == Level.NORMAL
    for ev in as_events(trades[trades["timestamp"] >= cut]):
        m.on_event(ev)
    assert received and received == m.alerts()
    spikes = [a for a in received if a.kind == "spike"]
    assert spikes[-1].level == "spike" or any(a.level == "spike" for a in spikes)
    assert any(a.kind == "regime" and a.level == "stressed" for a in received)
    first = spikes[0]
    assert (first.time - pd.Timestamp(cut, unit="ms", tz="UTC")).total_seconds() < 60
    json.dumps(first.as_dict())
    snap = m.snapshot(INSTRUMENT)
    assert snap.fast_vol > snap.slow_vol and snap.sigma_5m > 0
    assert snap.regime >= Regime.ACTIVE


def test_bars_close_on_time_without_trades():
    trades = tape(hours=1)
    m = LiveMonitor([INSTRUMENT], DetectorConfig(bar_seconds=10, slow_half_life=300))
    m.apply_seed({INSTRUMENT: trades})
    last = m.snapshot(INSTRUMENT).time
    later = int(last.timestamp() * 1000) + 35_000
    m.on_event(DvolUpdate("btc_usd", later, 40.0))
    snap = m.snapshot(INSTRUMENT)
    assert (snap.time - last).total_seconds() == 30  # three empty bars closed by the clock
    assert snap.dvol == pytest.approx(0.40)


def test_dvol_indices():
    m = LiveMonitor(["BTC-PERPETUAL", "ETH-PERPETUAL", "SOL_USDC-PERPETUAL"])
    assert m.dvol_indices == ["btc_usd", "eth_usd"]


class FakeREST:
    def __init__(self, trades):
        self.data = trades
        self.calls = 0

    def trades(self, instrument, start, end):
        self.calls += 1
        return self.data


class FakeStream:
    def __init__(self, events):
        self._events = events

    async def events(self):
        for ev in self._events:
            await asyncio.sleep(0.001)
            yield ev


async def test_run_buffers_during_seed_and_reseeds_after_reconnect():
    trades = tape(hours=2)
    cut = trades["timestamp"].iloc[0] + 7000 * 1000
    rest = FakeREST(trades[trades["timestamp"] < cut])
    live = as_events(trades[trades["timestamp"] >= cut - 2_000])
    events = [*live[:300], Reconnected(), *live[300:]]
    m = LiveMonitor([INSTRUMENT], CONFIG)
    await m.run(stream=FakeStream(events), rest=rest)
    assert rest.calls == 2
    hist = m.history(INSTRUMENT)
    assert hist.index.is_unique
    assert hist.index[-1] >= pd.Timestamp(int(live[-1].timestamp_ms) - 20_000, unit="ms", tz="UTC")


def test_alerts_found_in_startup_history_are_listed_but_not_sent():
    vol = np.r_[np.full(3 * 3600, 0.4), np.full(300, 2.0), np.full(1200, 0.4)]
    trades = simulate_trades(vol.size, vol, rate=2.0, seed=5)
    received = []
    m = LiveMonitor([INSTRUMENT], DetectorConfig(), on_alert=received.append)
    m.apply_seed({INSTRUMENT: trades})
    alerts = m.alerts()
    assert alerts and not received
    assert all(not a.live and a.dvol is None for a in alerts)
    m.apply_seed({INSTRUMENT: trades})  # re-seeding after a reconnection does not duplicate
    assert m.alerts() == alerts
