import numpy as np
import pandas as pd
import pytest

from tests.conftest import DAY, simulate_trades
from volengine.data.bars import BarStore, build_bars, load_bars
from volengine.data.schema import normalize, validate
from volengine.data.store import TradeStore
from volengine.data.sync import sync
from volengine.sampling import GridSampler

INST = "BTC-PERPETUAL"


def raw(seq, ts, price=100.0, **extra):
    return {
        "trade_seq": seq,
        "trade_id": str(seq),
        "timestamp": ts,
        "price": price,
        "amount": 10.0,
        "direction": "buy",
        "index_price": price,
        "mark_price": price,
        **extra,
    }


# -- schema -------------------------------------------------------------------


def test_normalize_kinds_order_and_duplicates():
    rows = [
        raw(3, 30),
        raw(1, 10),
        raw(2, 20, block_trade_id="B1"),
        raw(4, 40, combo_id="BTC-FS"),
        raw(1, 10),  # the same trade on two pages
        raw(5, 50, liquidation="T"),
    ]
    df = normalize(rows)
    assert df["trade_seq"].tolist() == [1, 2, 3, 4, 5]
    assert df["kind"].tolist() == ["book", "block", "book", "combo", "book"]
    assert df["liquidation"].tolist() == ["", "", "", "", "T"]
    assert df["is_buy"].all()


def test_validate_finds_holes_and_bad_prices():
    df = normalize([raw(s, s) for s in (1, 2, 3, 7, 8, 10)] + [raw(11, 11, price=0.0)])
    r = validate(df)
    assert r.missing == [(4, 6), (9, 9)] and r.n_missing == 4
    assert r.bad_prices == 1 and not r.ok
    assert validate(normalize([raw(s, s) for s in (1, 2, 3)])).ok


# -- store and sync -----------------------------------------------------------


class FakeAPI:
    """Serves a simulated tape by time window; can drop trades on the first request."""

    def __init__(self, tape: pd.DataFrame, drop: set[int] = frozenset(), always_drop=()):
        self.tape = tape
        self.drop = set(drop)
        self.always_drop = set(always_drop)
        self.calls = 0

    def trades(self, instrument, start, end, page=None):
        self.calls += 1
        lo, hi = pd.Timestamp(start).timestamp() * 1000, pd.Timestamp(end).timestamp() * 1000
        t = self.tape[(self.tape["timestamp"] >= lo) & (self.tape["timestamp"] < hi)]
        gone = self.always_drop | self.drop
        self.drop = set()  # dropped trades come back on the next request
        return t[~t["trade_seq"].isin(gone)].reset_index(drop=True)

    def close(self):
        pass


def test_sync_stores_complete_verified_days(tmp_path):
    tape = simulate_trades(2 * DAY, 0.5, rate=0.2)
    store = TradeStore(tmp_path)
    r = sync(INST, "2026-01-01", "2026-01-03", store=store, api=FakeAPI(tape))
    assert len(r.fetched) == 2 and not r.incomplete
    status = store.status(INST)
    assert status["trades"].sum() == len(tape)
    assert status["gap_before"].dropna().eq(0).all()
    back = store.read(INST, "2026-01-01", "2026-01-03")
    pd.testing.assert_frame_equal(back.reset_index(drop=True), tape.reset_index(drop=True))
    assert not list(tmp_path.rglob("*.tmp"))  # writes are atomic
    # A second run downloads nothing.
    api = FakeAPI(tape)
    assert sync(INST, "2026-01-01", "2026-01-03", store=store, api=api).already == 2
    assert api.calls == 0


def test_sync_repairs_holes_and_flags_what_it_cannot(tmp_path):
    tape = simulate_trades(DAY, 0.5, rate=0.2)
    seqs = tape["trade_seq"].to_numpy()
    store = TradeStore(tmp_path)
    r = sync(
        INST, "2026-01-01", "2026-01-02", store=store, api=FakeAPI(tape, drop=set(seqs[100:105]))
    )
    assert r.repaired and not r.incomplete
    assert store.info(INST, "2026-01-01").complete

    store2 = TradeStore(tmp_path / "other")
    api = FakeAPI(tape, always_drop={int(seqs[500])})
    r = sync(INST, "2026-01-01", "2026-01-02", store=store2, api=api)
    assert r.incomplete == {pd.Timestamp("2026-01-01", tz="UTC"): 1}
    assert not store2.info(INST, "2026-01-01").complete
    # An incomplete day is retried by the next sync.
    assert sync(INST, "2026-01-01", "2026-01-02", store=store2, api=FakeAPI(tape)).fetched


# -- bars ---------------------------------------------------------------------


def test_bars_ignore_block_trades_and_late_prints():
    trades = normalize(
        [
            raw(1, 1_000, 100.0),
            raw(2, 2_000, 101.0),
            raw(3, 12_000, 102.0),
            raw(4, 9_000, 50.0),  # late: stamped in a bar already passed
            raw(5, 13_000, 150.0, block_trade_id="B"),  # off-book
            raw(6, 14_000, 103.0, liquidation="M"),
        ]
    )
    bars = build_bars(trades, 10)
    assert bars["close"].tolist() == [101.0, 103.0]
    assert bars["n"].tolist() == [2, 2]
    assert bars["liq_volume"].tolist() == [0.0, 10.0]


def test_streaming_bars_equal_batch_bars():
    tape = simulate_trades(6 * 3600, 0.6, rate=0.3, block_share=0.02, seed=4)
    # Shuffle timestamps a little so that some prints arrive late.
    rng = np.random.default_rng(0)
    tape["timestamp"] = tape["timestamp"] + rng.integers(-300, 300, len(tape))
    batch = build_bars(tape, 10)
    s = GridSampler(10)
    done = []
    for t in tape[tape["kind"] == "book"].itertuples():
        done += s.update(t.timestamp, t.price, t.amount, t.is_buy, t.liquidation != "")
    done.append(s.current)
    live = pd.DataFrame([b.__dict__ for b in done if b.n > 0])
    live.index = pd.to_datetime(live.pop("start_ms"), unit="ms", utc=True)
    cols = ["close", "n", "volume", "buy_volume", "liq_volume"]
    pd.testing.assert_frame_equal(
        live[cols], batch[cols], check_names=False, check_dtype=False, check_freq=False
    )


def test_load_bars_fills_the_grid_and_refuses_missing_days(tmp_path):
    tape = simulate_trades(2 * DAY, 0.5, rate=0.05)
    store = TradeStore(tmp_path)
    sync(INST, "2026-01-01", "2026-01-03", store=store, api=FakeAPI(tape))
    bars = load_bars(INST, "2026-01-01", "2026-01-03", store=BarStore(store))
    assert len(bars) <= 2 * 8640 and bars.index.is_monotonic_increasing
    assert (bars.index.to_series().diff().dropna() == pd.Timedelta(seconds=10)).all()
    assert bars["n"].sum() == len(tape)
    with pytest.raises(FileNotFoundError):
        load_bars(INST, "2026-01-01", "2026-01-04", store=BarStore(store))


def test_bar_cache_is_rebuilt_when_trades_change(tmp_path):
    tape = simulate_trades(DAY, 0.5, rate=0.05)
    store = TradeStore(tmp_path)
    sync(INST, "2026-01-01", "2026-01-02", store=store, api=FakeAPI(tape))
    bs = BarStore(store)
    first = bs.day(INST, "2026-01-01", 10)
    sync(INST, "2026-01-01", "2026-01-02", store=store, api=FakeAPI(tape.iloc[::2]), refresh=True)
    assert bs.day(INST, "2026-01-01", 10)["n"].sum() < first["n"].sum()
