"""Public WebSocket feeds of several venues, normalized to one event type.

Every venue is reduced to top-of-book quotes and trades, in base-currency units (BTC
or ETH) and in the venue's quote currency (USD or USDT):

==============  =====================================  ==========  ==================
market          channels                               quote ccy   exchange time
==============  =====================================  ==========  ==================
deribit:perp    quote.<X>-PERPETUAL, trades            USD         quote and trade
binance:perp    <x>usdt@bookTicker, @trade (USD-M)     USDT        quote and trade
binance:spot    <x>usdt@bookTicker, @aggTrade          USDT        trade only
coinbase:spot   ticker <X>-USD (+ USDT-USD rate)       USD         quote and trade
==============  =====================================  ==========  ==================

Coinbase's public ticker is sent on each trade, so its quote can be stale between
trades; Binance spot quotes carry no exchange timestamp, so their local receipt time
is used. Both facts are kept in the event (``exchange_ms`` may be None).

No API key is used anywhere. Connections are kept alive and re-opened with
exponential backoff; a reconnection is reported so that consumers can reset state.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass

import websockets

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Quote:
    market: str  # "deribit:perp", "binance:perp", "binance:spot", "coinbase:spot"
    asset: str  # "BTC", "ETH"
    local_ms: float  # receipt time
    exchange_ms: float | None
    bid: float
    bid_size: float  # base units
    ask: float
    ask_size: float


@dataclass(frozen=True)
class Print:
    market: str
    asset: str
    local_ms: float
    exchange_ms: float
    price: float
    size: float  # base units
    is_buy: bool


@dataclass(frozen=True)
class Rate:
    """A conversion rate, e.g. USDT in USD from Coinbase USDT-USD."""

    name: str
    local_ms: float
    value: float


@dataclass(frozen=True)
class Disconnected:
    market: str


Event = Quote | Print | Rate | Disconnected
QUOTE_CCY = {
    "deribit:perp": "USD",
    "coinbase:spot": "USD",
    "binance:perp": "USDT",
    "binance:spot": "USDT",
}
MARKETS = tuple(QUOTE_CCY)


def _now() -> float:
    return time.time() * 1000.0


# -- parsers (pure functions, tested on recorded messages) ----------------------


def parse_deribit(msg: dict, local_ms: float) -> list[Event]:
    if msg.get("method") != "subscription":
        return []
    ch = msg["params"]["channel"]
    data = msg["params"]["data"]
    asset = ch.split(".")[1].split("-")[0]
    if ch.startswith("quote."):
        mid = (data["best_bid_price"] + data["best_ask_price"]) / 2
        return [
            Quote(
                "deribit:perp",
                asset,
                local_ms,
                float(data["timestamp"]),
                float(data["best_bid_price"]),
                float(data["best_bid_amount"]) / mid,  # USD notional -> base units
                float(data["best_ask_price"]),
                float(data["best_ask_amount"]) / mid,
            )
        ]
    if ch.startswith("trades."):
        return [
            Print(
                "deribit:perp",
                asset,
                local_ms,
                float(t["timestamp"]),
                float(t["price"]),
                float(t["amount"]) / float(t["price"]),
                t["direction"] == "buy",
            )
            for t in sorted(data, key=lambda t: t["trade_seq"])
            if not t.get("block_trade_id") and not t.get("combo_id")
        ]
    return []


def parse_binance(market: str) -> Callable[[dict, float], list[Event]]:
    def parse(msg: dict, local_ms: float) -> list[Event]:
        data = msg.get("data", msg)
        stream = msg.get("stream", "")
        asset = data.get("s", "")[:-4]  # "BTCUSDT" -> "BTC"
        if stream.endswith("@bookTicker") or data.get("e") == "bookTicker":
            ts = data.get("T")
            return [
                Quote(
                    market,
                    asset,
                    local_ms,
                    float(ts) if ts else None,
                    float(data["b"]),
                    float(data["B"]),
                    float(data["a"]),
                    float(data["A"]),
                )
            ]
        if data.get("e") == "trade" and data.get("X", "MARKET") != "MARKET":
            return []  # insurance-fund and auto-deleveraging fills are not market trades
        if data.get("e") in ("aggTrade", "trade"):
            return [
                Print(
                    market,
                    asset,
                    local_ms,
                    float(data["T"]),
                    float(data["p"]),
                    float(data["q"]),
                    not data["m"],  # buyer is maker -> the aggressor sold
                )
            ]
        return []

    return parse


def parse_coinbase(msg: dict, local_ms: float) -> list[Event]:
    if msg.get("type") != "ticker":
        return []
    product = msg["product_id"]
    if product == "USDT-USD":
        return [Rate("USDT/USD", local_ms, (float(msg["best_bid"]) + float(msg["best_ask"])) / 2)]
    asset = product.split("-")[0]
    exch = _iso_ms(msg.get("time"))
    out: list[Event] = [
        Quote(
            "coinbase:spot",
            asset,
            local_ms,
            exch,
            float(msg["best_bid"]),
            float(msg.get("best_bid_size") or 0.0),
            float(msg["best_ask"]),
            float(msg.get("best_ask_size") or 0.0),
        )
    ]
    if msg.get("last_size") and exch is not None:
        out.append(
            Print(
                "coinbase:spot",
                asset,
                local_ms,
                exch,
                float(msg["price"]),
                float(msg["last_size"]),
                msg.get("side") == "buy",  # Coinbase ticker side is the taker side
            )
        )
    return out


def _iso_ms(s: str | None) -> float | None:
    if not s:
        return None
    from datetime import datetime

    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp() * 1000.0


# -- connections ---------------------------------------------------------------


@dataclass(frozen=True)
class Feed:
    market: str
    url: str
    subscribe: dict | None
    parse: Callable[[dict, float], list[Event]]


def feeds(assets=("BTC",), markets=MARKETS) -> list[Feed]:
    lower = [a.lower() for a in assets]
    out = []
    if "deribit:perp" in markets:
        chans = [f"quote.{a}-PERPETUAL" for a in assets] + [
            f"trades.{a}-PERPETUAL.100ms" for a in assets
        ]
        sub = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "public/subscribe",
            "params": {"channels": chans},
        }
        out.append(Feed("deribit:perp", "wss://www.deribit.com/ws/api/v2", sub, parse_deribit))
    for market, host in (
        ("binance:perp", "wss://fstream.binance.com"),
        ("binance:spot", "wss://stream.binance.com:9443"),
    ):
        if market in markets:
            # USD-M futures serve aggTrade on another path; @trade comes on this one.
            kind = "trade" if market == "binance:perp" else "aggTrade"
            streams = "/".join(f"{a}usdt@bookTicker/{a}usdt@{kind}" for a in lower)
            out.append(
                Feed(market, f"{host}/stream?streams={streams}", None, parse_binance(market))
            )
    if "coinbase:spot" in markets:
        products = [f"{a}-USD" for a in assets] + ["USDT-USD"]
        sub = {"type": "subscribe", "product_ids": products, "channels": ["ticker", "heartbeat"]}
        out.append(
            Feed("coinbase:spot", "wss://ws-feed.exchange.coinbase.com", sub, parse_coinbase)
        )
    return out


async def _run_feed(feed: Feed, queue: asyncio.Queue, max_backoff: float = 30.0) -> None:
    attempt = 0
    while True:
        try:
            async with websockets.connect(feed.url, open_timeout=10, max_size=2**22) as ws:
                if feed.subscribe:
                    await ws.send(json.dumps(feed.subscribe))
                if feed.market == "deribit:perp":
                    hb = {
                        "jsonrpc": "2.0",
                        "id": 2,
                        "method": "public/set_heartbeat",
                        "params": {"interval": 30},
                    }
                    await ws.send(json.dumps(hb))
                attempt = 0
                log.info("%s connected", feed.market)
                async for raw in ws:
                    local = _now()
                    msg = json.loads(raw)
                    if msg.get("method") == "heartbeat" and feed.market == "deribit:perp":
                        if msg["params"].get("type") == "test_request":
                            await ws.send(
                                json.dumps({"jsonrpc": "2.0", "id": 3, "method": "public/test"})
                            )
                        continue
                    for ev in feed.parse(msg, local):
                        await queue.put(ev)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # network errors of every library, malformed messages
            delay = min(2.0**attempt, max_backoff)
            attempt += 1
            log.warning("%s disconnected (%s), retry in %.0fs", feed.market, exc, delay)
            await queue.put(Disconnected(feed.market))
            await asyncio.sleep(delay)


async def stream(assets=("BTC",), markets=MARKETS) -> AsyncIterator[Event]:
    """All venues merged into one stream, in order of receipt."""
    queue: asyncio.Queue = asyncio.Queue(maxsize=100_000)
    tasks = [asyncio.ensure_future(_run_feed(f, queue)) for f in feeds(assets, markets)]
    try:
        while True:
            yield await queue.get()
    finally:
        for t in tasks:
            t.cancel()
