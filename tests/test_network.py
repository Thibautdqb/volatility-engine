"""Integration tests against Deribit's public API. Run with ``pytest -m network``."""

import asyncio

import pandas as pd
import pytest

from volengine.deribit import DeribitREST, DeribitStream, Trade
from volengine.live import LiveSpikeMonitor

pytestmark = pytest.mark.network


def test_rest_trades_paginate_without_gaps_or_duplicates():
    end = pd.Timestamp.now(tz="UTC").floor("min")
    start = end - pd.Timedelta(minutes=20)
    with DeribitREST() as api:
        small = api.trades("BTC-PERPETUAL", start, end, page=200)
        big = api.trades("BTC-PERPETUAL", start, end)
    assert len(small) > 200  # several pages
    assert small["trade_id"].tolist() == big["trade_id"].tolist()
    assert small["timestamp"].is_monotonic_increasing
    ms = small["timestamp"]
    assert ms.min() >= start.timestamp() * 1000 and ms.max() < end.timestamp() * 1000


async def test_stream_delivers_trades():
    stream = DeribitStream(["BTC-PERPETUAL"], ["btc_usd"])

    async def first_trade():
        async for ev in stream.events():
            if isinstance(ev, Trade):
                return ev

    trade = await asyncio.wait_for(first_trade(), timeout=20)
    assert trade.instrument == "BTC-PERPETUAL" and trade.price > 0 and trade.trade_id


async def test_monitor_seeds_and_runs_live():
    m = LiveSpikeMonitor(["BTC-PERPETUAL"])
    task = asyncio.ensure_future(m.run())
    try:
        for _ in range(120):
            await asyncio.sleep(1)
            if len(m.history("BTC-PERPETUAL")) > m.config.warmup_bars:
                break
        start = m.snapshot("BTC-PERPETUAL").time
        await asyncio.sleep(25)
        snap = m.snapshot("BTC-PERPETUAL")
    finally:
        task.cancel()
    assert snap.time > start  # live bars keep arriving
    assert 0.02 < snap.slow_vol < 5.0
    h = m.history("BTC-PERPETUAL")
    assert h.index.is_unique and h.index.is_monotonic_increasing
