"""Sample a trade tape on a regular time grid.

A bar's price is the last trade printed inside it; a bar without trades keeps the
previous price (zero return). Bars are aligned on the epoch and identified by their
integer index ``timestamp_ms // bar_ms``. The streaming :class:`GridSampler` and the
batch :func:`bar_closes` produce identical series from the same trades, so the live
monitor and the historical study measure exactly the same thing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def bar_ms(bar_seconds: float) -> int:
    ms = round(bar_seconds * 1000)
    if ms <= 0:
        raise ValueError("bar_seconds must be positive")
    return ms


def bar_closes(
    trades: pd.DataFrame,
    bar_seconds: float,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> pd.Series:
    """Last trade price per bar over [start, end), forward-filled across empty bars.

    ``trades`` needs ``timestamp`` (ms) and ``price`` columns, in time order. Bars before
    the first trade are dropped.
    """
    step = bar_ms(bar_seconds)
    ts = trades["timestamp"].to_numpy(dtype=np.int64)
    bucket = ts // step
    last = pd.Series(trades["price"].to_numpy(dtype=float)).groupby(bucket).last()
    lo = bucket[0] if start is None else int(pd.Timestamp(start).timestamp() * 1000) // step
    hi = bucket[-1] + 1 if end is None else -(-int(pd.Timestamp(end).timestamp() * 1000) // step)
    closes = last.reindex(np.arange(lo, hi)).ffill().dropna()
    closes.index = pd.to_datetime(closes.index.to_numpy() * step, unit="ms", utc=True)
    closes.index.name = "time"
    return closes


def log_returns(closes: pd.Series) -> pd.Series:
    return np.log(closes).diff().dropna()


class GridSampler:
    """Streaming equivalent of :func:`bar_closes`.

    ``bucket`` and ``close`` resume from a known state (e.g. after seeding from
    history): trades in buckets before ``bucket`` are ignored, and if no trade arrives
    in ``bucket`` the bar closes at ``close``.
    """

    def __init__(self, bar_seconds: float, bucket: int | None = None, close: float | None = None):
        self.step = bar_ms(bar_seconds)
        self.bucket = bucket
        self.close = close
        self.trades = 0  # trades in the bar in progress

    def _roll(self, new_bucket: int) -> list[tuple[int, float, int]]:
        done = [(self.bucket * self.step, self.close, self.trades)]
        # Empty bars between the last one and the new one keep the last price.
        done.extend((b * self.step, self.close, 0) for b in range(self.bucket + 1, new_bucket))
        self.bucket, self.trades = new_bucket, 0
        return done

    def update(self, timestamp_ms: int, price: float) -> list[tuple[int, float, int]]:
        """Add a trade; returns completed bars as ``(start_ms, close, n_trades)``."""
        b = timestamp_ms // self.step
        if self.bucket is None or self.close is None:
            self.bucket, self.close, self.trades = b, price, 1
            return []
        if b < self.bucket:  # late print or already covered by history
            return []
        done = self._roll(b) if b > self.bucket else []
        self.close = price
        self.trades += 1
        return done

    def advance(self, now_ms: int) -> list[tuple[int, float, int]]:
        """Close every bar that ended before ``now_ms``, even if no trade arrived since."""
        if self.bucket is None or self.close is None:
            return []
        b = now_ms // self.step
        return self._roll(b) if b > self.bucket else []


def bar_counts(trades: pd.DataFrame, closes: pd.Series, bar_seconds: float) -> pd.Series:
    """Number of trades in each bar of ``closes`` (zero for empty bars)."""
    step = bar_ms(bar_seconds)
    n = pd.Series(1, index=trades["timestamp"].to_numpy(dtype=np.int64) // step).groupby(level=0)
    n = n.sum()
    n.index = pd.to_datetime(n.index.to_numpy() * step, unit="ms", utc=True)
    return n.reindex(closes.index).fillna(0).astype(int)
