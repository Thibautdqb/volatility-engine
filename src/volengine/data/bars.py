"""Regular time bars from the trade tape.

Rules, identical to the streaming :class:`volengine.sampling.GridSampler`:

* only order-book trades (``kind == "book"``) are used;
* trades are taken in ``trade_seq`` order, and a trade stamped in a bar earlier than
  one already seen is a late print and is ignored;
* a bar's ``close`` is its last trade; a bar without trades has ``close`` NaN here and
  is forward-filled by :func:`load_bars`.

Bars are cached per day next to the trades and rebuilt when the trade file changes.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from volengine.data.store import TradeStore, day_range, utc

BAR_COLUMNS = ["close", "n", "volume", "buy_volume", "liq_volume", "index"]
_VERSION = 1


def bar_ms(bar_seconds: float) -> int:
    ms = round(bar_seconds * 1000)
    if ms <= 0 or 86_400_000 % ms:
        raise ValueError("bar_seconds must divide a day")
    return ms


def build_bars(trades: pd.DataFrame, bar_seconds: float) -> pd.DataFrame:
    """Bars that contain at least one book trade, indexed by bar start (UTC)."""
    step = bar_ms(bar_seconds)
    t = trades[trades["kind"] == "book"].sort_values("trade_seq", kind="stable")
    bucket = t["timestamp"].to_numpy(dtype=np.int64) // step
    on_time = bucket >= np.maximum.accumulate(bucket) if bucket.size else bucket.astype(bool)
    t, bucket = t[on_time], bucket[on_time]
    amount = t["amount"].to_numpy(float)
    g = pd.DataFrame(
        {
            "bucket": bucket,
            "close": t["price"].to_numpy(float),
            "n": 1,
            "volume": amount,
            "buy_volume": np.where(t["is_buy"].to_numpy(), amount, 0.0),
            "liq_volume": np.where(t["liquidation"].to_numpy() != "", amount, 0.0),
            "index": t["index_price"].to_numpy(float),
        }
    ).groupby("bucket", sort=True)
    out = g.agg(
        close=("close", "last"),
        n=("n", "sum"),
        volume=("volume", "sum"),
        buy_volume=("buy_volume", "sum"),
        liq_volume=("liq_volume", "sum"),
        index=("index", "last"),
    )
    out.index = pd.to_datetime(out.index.to_numpy() * step, unit="ms", utc=True)
    out.index.name = "time"
    return out


class BarStore:
    def __init__(self, trades: TradeStore | None = None):
        self.trades = trades or TradeStore()
        self.root = self.trades.root.parent.parent / "bars" / f"v{_VERSION}"

    def day(self, instrument: str, day, bar_seconds: float) -> pd.DataFrame:
        info = self.trades.info(instrument, day)
        if info is None:
            raise FileNotFoundError(f"{instrument} {utc(day):%Y-%m-%d} is not in the store")
        base = self.root / str(bar_ms(bar_seconds)) / instrument / f"{utc(day):%Y-%m-%d}"
        meta = base.with_suffix(".json")
        if meta.exists() and json.loads(meta.read_text()).get("source") == info.fetched_at:
            return pq.read_table(base.with_suffix(".parquet")).to_pandas()
        cols = ["timestamp", "trade_seq", "price", "amount", "is_buy", "index_price"]
        trades = self.trades.read_day(instrument, day, [*cols, "liquidation", "kind"])
        bars = build_bars(trades, bar_seconds)
        base.parent.mkdir(parents=True, exist_ok=True)
        tmp = base.with_suffix(".parquet.tmp")
        bars.to_parquet(tmp)
        tmp.replace(base.with_suffix(".parquet"))
        meta.write_text(json.dumps({"source": info.fetched_at}))
        return bars


def load_bars(
    instrument: str,
    start,
    end,
    bar_seconds: float = 10.0,
    store: BarStore | None = None,
) -> pd.DataFrame:
    """A complete grid of bars over the stored days in [start, end).

    Raises if a day in the range is not stored: a hole in the grid would create a
    spurious return across it. Empty bars carry the previous close and zero counts.
    """
    store = store or BarStore()
    days = day_range(start, end)
    absent = [d for d in days if not store.trades.has(instrument, d)]
    if absent:
        raise FileNotFoundError(
            f"{instrument}: {len(absent)} day(s) not stored, first {absent[0]:%Y-%m-%d}; "
            "run 'volengine data sync' first"
        )
    frames = [store.day(instrument, d, bar_seconds) for d in days]
    bars = pd.concat(frames)
    step = pd.Timedelta(milliseconds=bar_ms(bar_seconds))
    grid = pd.date_range(days[0], days[-1] + pd.Timedelta(days=1), freq=step, inclusive="left")
    bars = bars.reindex(grid)
    bars.index.name = "time"
    bars[["close", "index"]] = bars[["close", "index"]].ffill()
    counts = ["n", "volume", "buy_volume", "liq_volume"]
    bars[counts] = bars[counts].fillna(0.0)
    bars["n"] = bars["n"].astype(int)
    return bars.dropna(subset=["close"])
