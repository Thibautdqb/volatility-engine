"""Integration tests against live public APIs. Run with ``pytest -m network``."""

import asyncio

import pandas as pd
import pytest

from volengine.deribit import DeribitREST, DeribitStream, Tick

pytestmark = pytest.mark.network


def test_rest_ohlc_paginates():
    end = pd.Timestamp.now(tz="UTC").floor("min")
    start = end - pd.Timedelta(days=5)  # 7200 one-minute bars, two pages
    with DeribitREST() as api:
        df = api.ohlc("BTC-PERPETUAL", "1", start, end)
    assert len(df) > 6000
    assert df.index.is_monotonic_increasing and not df.index.duplicated().any()
    assert (df["high"] >= df["low"]).all()


def test_rest_dvol():
    with DeribitREST() as api:
        df = api.dvol("BTC", "1D", start=pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=30))
    assert 25 <= len(df) <= 31
    assert df["close"].between(5, 300).all()


async def test_stream_delivers_ticks():
    stream = DeribitStream(["BTC-PERPETUAL"], ["btc_usd"])

    async def first_tick():
        async for ev in stream.events():
            if isinstance(ev, Tick):
                return ev

    tick = await asyncio.wait_for(first_tick(), timeout=20)
    assert tick.instrument == "BTC-PERPETUAL" and tick.mark_price > 0
