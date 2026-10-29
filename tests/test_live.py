import asyncio
import json

import numpy as np
import pandas as pd
import pytest

from tests.conftest import DAY, simulate_trades
from tests.test_forecast import grid
from volengine.data.bars import BarStore
from volengine.data.store import TradeStore
from volengine.deribit.stream import DvolUpdate, Reconnected, Trade, parse_message
from volengine.detector import DetectorConfig
from volengine.features import compute
from volengine.forecast import Regime, default_model
from volengine.live import LiveMonitor

INST = "BTC-PERPETUAL"
CONFIG = DetectorConfig()
TRADES_MSG = {
    "jsonrpc": "2.0",
    "method": "subscription",
    "params": {
        "channel": "trades.BTC-PERPETUAL.100ms",
        "data": [
            {
                "trade_seq": 311,
                "timestamp": 1793130317700,
                "price": 80736.5,
                "direction": "sell",
                "instrument_name": "BTC-PERPETUAL",
                "amount": 20.0,
                "trade_id": "2",
                "block_trade_id": "B1",
            },
            {
                "trade_seq": 310,
                "timestamp": 1793130317626,
                "price": 80737.0,
                "direction": "buy",
                "instrument_name": "BTC-PERPETUAL",
                "amount": 10000.0,
                "trade_id": "1",
                "liquidation": "T",
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


def test_parse_messages_in_sequence_order():
    a, b = parse_message(TRADES_MSG)
    assert (a.trade_seq, b.trade_seq) == (310, 311)
    assert a.kind == "book" and a.is_buy and a.liquidation
    assert b.kind == "block" and not b.is_buy
    (dvol,) = parse_message(DVOL_MSG)
    assert isinstance(dvol, DvolUpdate) and dvol.value == 37.36
    assert parse_message({"jsonrpc": "2.0", "id": 1, "result": []}) == []


def events(df):
    return [
        Trade(
            INST,
            int(t.timestamp),
            float(t.price),
            float(t.amount),
            int(t.trade_seq),
            t.kind,
            bool(t.is_buy),
            t.liquidation != "",
        )
        for t in df.itertuples()
    ]


@pytest.fixture(scope="module")
def tape():
    vol = np.r_[np.full(DAY + 3 * 3600, 0.4), np.full(600, 3.2), np.full(3600, 0.5)]
    return simulate_trades(vol.size, vol, rate=1.0, seed=1, block_share=0.01)


def no_store() -> BarStore:
    return BarStore(TradeStore("/nonexistent"))


def seeded(tape, cut_ms, **kw):
    m = LiveMonitor([INST], CONFIG, bar_store=no_store(), **kw)
    m.apply_seed({INST: ([], tape[tape["timestamp"] < cut_ms])})
    return m


def test_seed_then_live_equals_batch(tape):
    t0 = int(tape["timestamp"].iloc[0])
    cut = t0 + (DAY + 2 * 3600) * 1000 + 4_321  # mid-bar
    received = []
    m = seeded(tape, cut, on_alert=received.append)
    live = tape[tape["timestamp"] >= cut - 3_000]  # overlap: the stream starts early
    for ev in events(live):
        m.on_event(ev, received_ms=ev.timestamp_ms + 150)

    got = m.history(INST)
    assert got.index.is_unique
    assert (got.index.to_series().diff().dropna() == pd.Timedelta(seconds=10)).all()
    f = compute(grid(tape), CONFIG)
    model = m.models[INST]
    want = f.assign(forecast_vol=np.exp(model.predict(f)) * f["slow_vol"])
    want.index = want.index + pd.Timedelta(seconds=10)  # live rows are stamped at bar end
    common = got.index.intersection(want.index)
    assert len(common) > 0.9 * len(got)
    for col in ("slow_vol", "ratio", "forecast_vol"):
        np.testing.assert_allclose(got.loc[common, col], want.loc[common, col], rtol=1e-9)

    # The burst raised exactly one alert, live, as it started, with a readable message.
    burst = t0 + (DAY + 3 * 3600) * 1000
    assert len(received) == 1 and received[0].live
    alert = received[0]
    assert 0 <= (alert.time.timestamp() * 1000 - burst) < 120_000
    assert alert.stress >= 99 or alert.reason == "spike"
    assert alert.instrument in alert.title and "P(violent)" in alert.body
    json.dumps(alert.as_dict())
    snap = m.snapshot(INST)
    assert snap.latency_ms == 150 and snap.seq_gaps == 0
    assert 0 < snap.p_violent < 1 and snap.sigma_5m > 0 and 0 <= snap.stress <= 100


class TapeREST:
    def __init__(self, tape):
        self.tape = tape

    def trades(self, instrument, start, end):
        return self.tape


def test_sequence_holes_are_checked_against_rest(tape):
    t0 = int(tape["timestamp"].iloc[0])
    cut = t0 + (DAY + 2 * 3600) * 1000
    after = tape[tape["timestamp"] >= cut].iloc[:200]
    book = after.index[after["kind"] == "book"]

    # A hole holding a book trade: the bars are wrong, so the monitor re-seeds.
    m = seeded(tape, cut)
    for ev in events(after.drop(book[50])):
        m.on_event(ev)
    m.check_holes(TapeREST(tape))
    assert m.seq_gaps[INST] == 1 and m._resync

    # A hole holding only a block trade: noted, no re-seed.
    block = tape.copy()
    block.loc[book[60], "kind"] = "block"
    m = seeded(block, cut)
    for ev in events(block.loc[after.index].drop(book[60])):
        m.on_event(ev)
    m.check_holes(TapeREST(block))
    assert m.seq_gaps[INST] == 0 and m.seq_skipped[INST] == 1 and not m._resync


def test_duplicates_are_ignored_and_bars_close_on_the_clock(tape):
    t0 = int(tape["timestamp"].iloc[0])
    cut = t0 + (DAY + 2 * 3600) * 1000
    m = seeded(tape, cut)
    before = m.snapshot(INST).time
    for ev in events(tape[tape["timestamp"] < cut].iloc[-50:]):  # already loaded
        m.on_event(ev)
    assert m.snapshot(INST).time == before
    later = int(before.timestamp() * 1000) + 35_000
    m.on_event(DvolUpdate("btc_usd", later, 40.0))
    snap = m.snapshot(INST)
    assert (snap.time - before).total_seconds() == 30  # three empty bars closed by the clock
    assert snap.dvol == pytest.approx(0.40)


def test_seed_reads_stored_days_and_fetches_the_rest(tmp_path, tape):
    from tests.test_data import FakeAPI
    from volengine.data.sync import sync

    store = TradeStore(tmp_path)
    sync(INST, "2026-01-01", "2026-01-02", store=store, api=FakeAPI(tape))
    m = LiveMonitor([INST], CONFIG, bar_store=BarStore(store))
    end = pd.Timestamp("2026-01-02 02:30", tz="UTC")
    calls = []

    class REST:
        def trades(self, instrument, start, end):
            calls.append((start, end))
            lo, hi = start.timestamp() * 1000, end.timestamp() * 1000
            return tape[(tape["timestamp"] >= lo) & (tape["timestamp"] < hi)]

    seed = m.fetch_seed(REST(), end)
    frames, _ = seed[INST]
    assert len(frames) == 1  # 2026-01-01 comes from disk
    assert calls == [(pd.Timestamp("2026-01-02", tz="UTC"), end)]
    m.apply_seed(seed)
    assert m.snapshot(INST).regime in Regime


def test_dvol_indices_and_pooled_model_for_other_instruments():
    m = LiveMonitor(["BTC-PERPETUAL", "ETH-PERPETUAL", "SOL_USDC-PERPETUAL"])
    assert m.dvol_indices == ["btc_usd", "eth_usd"]
    assert m.models["SOL_USDC-PERPETUAL"] == default_model("pooled")


class FakeStream:
    """Replays events at a steady pace, leaving the seeding thread time to finish."""

    def __init__(self, evs, pace=0.002):
        self._evs = evs
        self.pace = pace
        self.clock = 0  # exchange time of the last trade delivered

    async def events(self):
        for ev in self._evs:
            await asyncio.sleep(self.pace)
            if isinstance(ev, Trade):
                self.clock = ev.timestamp_ms + 1
            yield ev


async def test_run_buffers_while_seeding_and_reseeds_after_reconnect(tape):
    t0 = int(tape["timestamp"].iloc[0])
    cut = t0 + (DAY + 2 * 3600) * 1000
    live = events(tape[tape["timestamp"] >= cut - 2_000].iloc[:3000])
    stream = FakeStream([*live[:500], Reconnected(), *live[500:]])
    calls = []

    class REST:
        """History up to the last trade the stream has delivered, like the real server."""

        def trades(self, instrument, start, end):
            calls.append(1)
            now = max(cut, stream.clock)
            return tape[tape["timestamp"] < now]

    m = LiveMonitor([INST], CONFIG, bar_store=no_store())
    m.fetch_seed = lambda rest, end=None: {INST: ([], rest.trades(INST, None, None))}
    await m.run(stream=stream, rest=REST())
    assert len(calls) == 2  # start and reconnection, no hole
    assert m.seq_gaps[INST] == 0
    h = m.history(INST)
    last = pd.Timestamp(live[-1].timestamp_ms - 20_000, unit="ms", tz="UTC")
    assert h.index.is_unique and h.index[-1] >= last
