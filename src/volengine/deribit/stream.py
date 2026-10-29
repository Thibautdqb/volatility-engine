"""Deribit WebSocket market-data stream.

Only public channels are used, so no API key is required:

* ``trades.{instrument}.100ms``: every trade print, the same tape the history
  server returns, so live and historical bars have one definition.
* ``deribit_volatility_index.{index}``: DVOL, the 30-day implied volatility index.

The connection is kept alive with Deribit's heartbeat mechanism and re-established
with exponential backoff on any failure. After a reconnection a :class:`Reconnected`
event is emitted so that consumers can backfill the gap.
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
class Trade:
    instrument: str
    timestamp_ms: int  # exchange time
    price: float
    amount: float
    trade_seq: int = 0  # increases by one per trade of the instrument
    kind: str = "book"  # "book", "block" or "combo", as in data.schema
    is_buy: bool = True
    liquidation: bool = False


@dataclass(frozen=True)
class DvolUpdate:
    index_name: str
    timestamp_ms: int
    value: float  # annualized vol in percent


@dataclass(frozen=True)
class Reconnected:
    pass


Event = Trade | DvolUpdate | Reconnected


def parse_message(message: dict) -> list[Event]:
    """Turn a raw subscription notification into typed events."""
    if message.get("method") != "subscription":
        return []
    params = message.get("params", {})
    channel: str = params.get("channel", "")
    data = params.get("data")
    if channel.startswith("trades."):
        # A notification batches up to 100 ms of trades; process them in matching order.
        return [
            Trade(
                instrument=t["instrument_name"],
                timestamp_ms=int(t["timestamp"]),
                price=float(t["price"]),
                amount=float(t["amount"]),
                trade_seq=int(t["trade_seq"]),
                kind="block"
                if t.get("block_trade_id")
                else "combo"
                if t.get("combo_id")
                else "book",
                is_buy=t.get("direction") == "buy",
                liquidation=bool(t.get("liquidation")),
            )
            for t in sorted(data, key=lambda t: t["trade_seq"])
        ]
    if channel.startswith("deribit_volatility_index."):
        return [
            DvolUpdate(
                index_name=data["index_name"],
                timestamp_ms=int(data["timestamp"]),
                value=float(data["volatility"]),
            )
        ]
    return []


class DeribitStream:
    def __init__(
        self,
        instruments: Sequence[str],
        dvol_indices: Sequence[str] = (),
        url: str = WS_URL,
        heartbeat: int = 30,
        max_backoff: float = 60.0,
    ):
        self.channels = [f"trades.{i}.100ms" for i in instruments] + [
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
        connected_before = False
        while True:
            try:
                async with websockets.connect(self.url, open_timeout=10, ping_interval=None) as ws:
                    await ws.send(self._request("public/set_heartbeat", interval=self.heartbeat))
                    await ws.send(self._request("public/subscribe", channels=self.channels))
                    log.info("subscribed to %s", self.channels)
                    attempt = 0
                    if connected_before:
                        yield Reconnected()
                    connected_before = True
                    async for raw in ws:
                        msg = json.loads(raw)
                        if "error" in msg:
                            log.error("deribit error: %s", msg["error"])
                            continue
                        if msg.get("method") == "heartbeat":
                            if msg["params"].get("type") == "test_request":
                                await ws.send(self._request("public/test"))
                            continue
                        for event in parse_message(msg):
                            yield event
            except asyncio.CancelledError:
                raise
            except (OSError, websockets.WebSocketException, asyncio.TimeoutError) as exc:
                delay = min(2.0**attempt, self.max_backoff)
                attempt += 1
                log.warning("stream disconnected (%s), reconnecting in %.0fs", exc, delay)
                await asyncio.sleep(delay)
