import numpy as np
import pandas as pd
import pytest

from volengine.xvenue.engine import Divergence, Engine, EngineConfig, Opportunity
from volengine.xvenue.feeds import (
    Print,
    Quote,
    Rate,
    parse_binance,
    parse_coinbase,
    parse_deribit,
)
from volengine.xvenue.record import load, to_row, write

A, B = "deribit:perp", "binance:perp"


# -- parsers -----------------------------------------------------------------------


def test_parsers_normalize_units_and_sides():
    (q,) = parse_deribit(
        {
            "method": "subscription",
            "params": {
                "channel": "quote.BTC-PERPETUAL",
                "data": {
                    "timestamp": 1000,
                    "best_bid_price": 99.0,
                    "best_ask_price": 101.0,
                    "best_bid_amount": 1000.0,
                    "best_ask_amount": 500.0,
                },
            },
        },
        1030.0,
    )
    assert (q.market, q.asset, q.bid_size, q.ask_size) == ("deribit:perp", "BTC", 10.0, 5.0)
    trades = parse_deribit(
        {
            "method": "subscription",
            "params": {
                "channel": "trades.BTC-PERPETUAL.100ms",
                "data": [
                    {"trade_seq": 2, "timestamp": 5, "price": 100.0, "amount": 200.0, "direction": "sell"},
                    {"trade_seq": 1, "timestamp": 4, "price": 100.0, "amount": 100.0, "direction": "buy"},
                    {"trade_seq": 3, "timestamp": 6, "price": 90.0, "amount": 1.0, "direction": "buy", "block_trade_id": "x"},
                ],
            },
        },
        10.0,
    )  # fmt: skip
    assert [(t.size, t.is_buy) for t in trades] == [(1.0, True), (2.0, False)]

    perp = parse_binance("binance:perp")
    (q,) = perp({"stream": "btcusdt@bookTicker", "data": {"e": "bookTicker", "s": "BTCUSDT", "b": "1", "B": "2", "a": "3", "A": "4", "T": 7}}, 9.0)  # fmt: skip
    assert (q.asset, q.exchange_ms, q.bid, q.ask_size) == ("BTC", 7.0, 1.0, 4.0)
    (t,) = perp({"stream": "btcusdt@trade", "data": {"e": "trade", "s": "BTCUSDT", "p": "5", "q": "0.1", "T": 8, "m": True, "X": "MARKET"}}, 9.0)  # fmt: skip
    assert not t.is_buy  # buyer was the maker: a sell
    assert perp({"data": {"e": "trade", "s": "BTCUSDT", "X": "INSURANCE_FUND"}}, 9.0) == []
    (q,) = parse_binance("binance:spot")({"stream": "btcusdt@bookTicker", "data": {"s": "BTCUSDT", "b": "1", "B": "2", "a": "3", "A": "4"}}, 9.0)  # fmt: skip
    assert q.exchange_ms is None  # Binance spot quotes carry no exchange time

    (r,) = parse_coinbase({"type": "ticker", "product_id": "USDT-USD", "best_bid": "0.998", "best_ask": "1.000"}, 1.0)  # fmt: skip
    assert r.value == pytest.approx(0.999)
    q, t = parse_coinbase(
        {"type": "ticker", "product_id": "BTC-USD", "price": "100", "best_bid": "99", "best_bid_size": "1",
         "best_ask": "101", "best_ask_size": "2", "last_size": "0.5", "side": "buy",
         "time": "2026-10-29T12:00:00.123456Z"},
        1.0,
    )  # fmt: skip
    assert q.exchange_ms == pytest.approx(
        pd.Timestamp("2026-10-29T12:00:00.123456Z").timestamp() * 1000
    )
    assert t.size == 0.5 and t.is_buy


# -- engine --------------------------------------------------------------------------


def run(engine, events):
    out = []
    for ev in events:
        out += engine.on_event(ev)
    return out


def calm(seconds=300, wiggle_a=1e-5, wiggle_b=1e-5, seed=0, price=100.0, t0=0.0):
    """Two markets quoting the same price every 100 ms with small noise; both trading."""
    rng = np.random.default_rng(seed)
    events = [Rate("USDT/USD", t0, 1.0)]  # B quotes in USDT
    for k in range(seconds * 10):
        t = t0 + k * 100.0
        for m, w in ((A, wiggle_a), (B, wiggle_b)):
            mid = price * (1 + w * rng.standard_normal())
            events.append(Quote(m, "BTC", t, t, mid - 0.005, 1.0, mid + 0.005, 1.0))
            if k % 10 == 0:
                events.append(Print(m, "BTC", t, t, mid, 0.1, True))
    return events


CFG = EngineConfig(fees_bp={A: 0.0, B: 0.0}, warmup_s=60, latency_ms=300, z=2.0)


def test_a_dislocation_beyond_the_buffer_is_flagged_with_its_size():
    e = Engine(("BTC",), (A, B), CFG)
    run(e, calm())
    t = 300_000.0
    # B's ask drops 10 bp below normal for one second, with 0.4 BTC on offer.
    shock = []
    for k in range(10):
        shock.append(Quote(B, "BTC", t + 100 * k, t, 99.895, 1.0, 99.905, 0.4))
        shock.append(Quote(A, "BTC", t + 100 * k + 1, t, 99.995, 1.0, 100.005, 1.0))
    shock.append(Quote(B, "BTC", t + 1500, t, 99.995, 1.0, 100.005, 1.0))
    shock.append(Quote(A, "BTC", t + 1501, t, 99.995, 1.0, 100.005, 1.0))
    out = [o for o in run(e, shock) if isinstance(o, Opportunity)]
    assert len(out) == 1
    o = out[0]
    assert (o.sell_on, o.buy_on) == (A, B)
    assert o.peak_net_bp == pytest.approx(9.0, abs=0.5)  # 10 bp minus the 1 bp quoted spread
    assert o.size == pytest.approx(0.4) and 800 <= o.duration_ms <= 1600


def test_fees_and_volatility_raise_the_bar():
    shocked = []
    t = 300_000.0
    for k in range(10):
        shocked.append(Quote(B, "BTC", t + 100 * k, t, 99.945, 1.0, 99.955, 1.0))  # 5 bp off
        shocked.append(Quote(A, "BTC", t + 100 * k + 1, t, 99.995, 1.0, 100.005, 1.0))
    shocked.append(Quote(B, "BTC", t + 1500, t, 99.995, 1.0, 100.005, 1.0))  # back to normal
    # Fees larger than the edge: nothing.
    e = Engine(("BTC",), (A, B), EngineConfig(fees_bp={A: 3.0, B: 3.0}, warmup_s=60))
    run(e, calm())
    assert not [o for o in run(e, shocked) if isinstance(o, Opportunity)]
    # Same edge, no fees, but markets moving 5 bp a tick: the buffer swallows it.
    e = Engine(("BTC",), (A, B), CFG)
    run(e, calm(wiggle_a=5e-4, wiggle_b=5e-4))
    assert not [o for o in run(e, shocked) if isinstance(o, Opportunity)]
    # Quiet markets, no fees: flagged.
    e = Engine(("BTC",), (A, B), CFG)
    run(e, calm())
    assert [o for o in run(e, shocked) if isinstance(o, Opportunity)]


def test_a_halted_or_stale_market_is_never_an_opportunity():
    e = Engine(("BTC",), (A, B), CFG)
    events = calm()
    run(e, events)
    # A stops trading for two minutes while B trades and moves 50 bp: A's quote is stale.
    t = 300_000.0
    stuck = []
    for k in range(1200):
        tt = t + 100 * k
        stuck.append(Quote(A, "BTC", tt, tt, 99.995, 1.0, 100.005, 1.0))  # frozen book
        mid = 100.5
        stuck.append(Quote(B, "BTC", tt, tt, mid - 0.005, 1.0, mid + 0.005, 1.0))
        if k % 10 == 0:
            stuck.append(Print(B, "BTC", tt, tt, mid, 0.1, True))
    out = run(e, stuck)
    late = [o for o in out if isinstance(o, Opportunity) and o.start_ms > t + 61_000]
    assert not late


def test_usdt_quotes_are_converted():
    e = Engine(("BTC",), ("deribit:perp", "binance:spot"), CFG)
    run(e, [Rate("USDT/USD", 0.0, 0.999)])
    assert e._usd("binance:spot", 100.0) == pytest.approx(99.9)
    assert e._usd("deribit:perp", 100.0) == 100.0


def test_volatility_divergence_names_the_agitated_market():
    e = Engine(("BTC",), (A, B), CFG)
    run(e, calm(seconds=200))
    out = run(e, calm(seconds=200, wiggle_a=1e-5, wiggle_b=3e-4, seed=1, t0=200_000.0))
    d = [o for o in out if isinstance(o, Divergence)]
    assert d and d[0].leader == B and d[0].sigma_leader_bp > 2 * d[0].sigma_other_bp


def test_recording_round_trip(tmp_path):
    events = [*calm(seconds=5)[:50], Rate("USDT/USD", 10.0, 0.999)]
    t0 = pd.Timestamp("2026-10-29 12:00", tz="UTC").timestamp() * 1000
    rows = [to_row(ev) | {"local_ms": ev.local_ms + t0} for ev in events]
    path = write(rows, tmp_path)
    assert path.exists() and not list(tmp_path.rglob("*.tmp"))
    back = load("2026-10-29 11:00", "2026-10-29 13:00", tmp_path)
    assert len(back) == len(rows)
    assert set(back["kind"]) == {"quote", "print", "rate"}
