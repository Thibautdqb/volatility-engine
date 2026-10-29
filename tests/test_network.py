"""Integration tests against Deribit's public API. Run with ``pytest -m network``."""

import asyncio

import pandas as pd
import pytest

from volengine.data.schema import validate
from volengine.deribit import DeribitREST, DeribitStream, Trade
from volengine.live import LiveMonitor

pytestmark = pytest.mark.network


def test_rest_pages_join_into_a_complete_sequence():
    end = pd.Timestamp.now(tz="UTC").floor("min") - pd.Timedelta(minutes=5)
    start = end - pd.Timedelta(minutes=30)
    with DeribitREST() as api:
        small = api.trades("BTC-PERPETUAL", start, end, page=300)
        big = api.trades("BTC-PERPETUAL", start, end)
    assert len(small) > 300  # several pages
    assert small["trade_seq"].tolist() == big["trade_seq"].tolist()
    report = validate(small)
    assert report.ok and report.n_missing == 0
    ms = small["timestamp"]
    assert ms.min() >= start.timestamp() * 1000 and ms.max() < end.timestamp() * 1000


async def test_stream_delivers_sequenced_trades():
    stream = DeribitStream(["BTC-PERPETUAL"], ["btc_usd"])
    seqs = []

    async def collect():
        async for ev in stream.events():
            if isinstance(ev, Trade):
                seqs.append(ev.trade_seq)
                if len(seqs) >= 20:
                    return

    await asyncio.wait_for(collect(), timeout=60)
    assert seqs == sorted(seqs) and seqs[0] > 0


async def test_monitor_seeds_and_runs_live():
    m = LiveMonitor(["BTC-PERPETUAL"])
    task = asyncio.ensure_future(m.run())
    try:
        for _ in range(240):
            await asyncio.sleep(1)
            if m.snapshot("BTC-PERPETUAL") is not None:
                break
        first = m.snapshot("BTC-PERPETUAL")
        await asyncio.sleep(25)
        snap = m.snapshot("BTC-PERPETUAL")
    finally:
        task.cancel()
    assert first is not None and snap.time > first.time  # live bars keep arriving
    assert 0.02 < snap.slow_vol < 5.0 and 0 < snap.p_violent < 1
    assert snap.latency_ms is not None and snap.latency_ms < 5_000
    assert m.history("BTC-PERPETUAL").index.is_unique
