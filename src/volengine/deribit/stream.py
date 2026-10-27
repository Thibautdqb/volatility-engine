"""Deribit WebSocket market-data stream.

Only public channels are used, so no API key is required:

* ``ticker.{instrument}.100ms``: mark price, index price, best bid/ask
* ``deribit_volatility_index.{index}``: DVOL, the 30-day implied volatility index

The connection is kept alive with Deribit's heartbeat mechanism and re-established
with exponential backoff on any failure.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import logging
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass

import websockets

WS_URL = "wss://www.deribit.com/ws/api/v2"
log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Tick:
    instrument: str
    timestamp: float  # seconds since epoch, exchange time
    mark_price: float
    index_price: float
    best_bid: float | None
    best_ask: float | None


@dataclass(frozen=True)
class DvolUpdate:
    index_name: str
    timestamp: float
    value: float  # annualized vol in percent


Event = Tick | DvolUpdate


def parse_message(message: dict) -> Event | None:
    """Turn a raw subscription notification into a typed event (None otherwise)."""
    if message.get("method") != "subscription":
        return None
    params = message.get("params", {})
    channel: str = params.get("channel", "")
    data = params.get("data", {})
    if channel.startswith("ticker."):
        return Tick(
            instrument=data["instrument_name"],
            timestamp=data["timestamp"] / 1000.0,
            mark_price=float(data["mark_price"]),
            index_price=float(data["index_price"]),
            best_bid=data.get("best_bid_price"),
            best_ask=data.get("best_ask_price"),
        )
    if channel.startswith("deribit_volatility_index."):
        return DvolUpdate(
            index_name=data["index_name"],
            timestamp=data["timestamp"] / 1000.0,
            value=float(data["volatility"]),
        )
    return None


class DeribitStream:
    def __init__(
        self,
        instruments: Sequence[str],
        dvol_indices: Sequence[str] = (),
        url: str = WS_URL,
        heartbeat: int = 30,
        max_backoff: float = 60.0,
    ):
        self.channels = [f"ticker.{i}.100ms" for i in instruments] + [
            f"deribit_volatility_index.{i}" for i in dvol_indices
        ]
        self.url = url
        self.heartbeat = heartbeat
        self.max_backoff = max_backoff
        self._ids = itertools.count(1)

    def _request(self, method: str, **params) -> str:
        return json.dumps(
            {"jsonrpc": "2.0", "id": next(self._ids), "method": method, "params": params}
        )

    async def events(self) -> AsyncIterator[Event]:
        attempt = 0
        while True:
            try:
                async with websockets.connect(self.url, open_timeout=10, ping_interval=None) as ws:
                    await ws.send(self._request("public/set_heartbeat", interval=self.heartbeat))
                    await ws.send(self._request("public/subscribe", channels=self.channels))
                    log.info("subscribed to %s", self.channels)
                    attempt = 0
                    async for raw in ws:
                        msg = json.loads(raw)
                        if "error" in msg:
                            log.error("deribit error: %s", msg["error"])
                            continue
                        if msg.get("method") == "heartbeat":
                            if msg["params"].get("type") == "test_request":
                                await ws.send(self._request("public/test"))
                            continue
                        event = parse_message(msg)
                        if event is not None:
                            yield event
            except asyncio.CancelledError:
                raise
            except (OSError, websockets.WebSocketException, asyncio.TimeoutError) as exc:
                delay = min(2.0**attempt, self.max_backoff)
                attempt += 1
                log.warning("stream disconnected (%s), reconnecting in %.0fs", exc, delay)
                await asyncio.sleep(delay)
