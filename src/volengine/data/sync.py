"""Download the trade tape into the local store, verify it and repair holes.

For each missing UTC day: download every trade by time, check that the sequence
numbers are contiguous, and if they are not, download again the short window around
each hole. Days stay flagged in their manifest if a hole cannot be filled. Day
boundaries are checked too: the first trade of a day must follow the last one of the
previous day.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import pandas as pd

from volengine.data.schema import validate
from volengine.data.store import TradeStore, day_range, utc
from volengine.deribit.rest import DeribitREST

log = logging.getLogger(__name__)
ONE_DAY = pd.Timedelta(days=1)


@dataclass
class SyncResult:
    instrument: str
    fetched: list[pd.Timestamp] = field(default_factory=list)
    already: int = 0
    repaired: list[pd.Timestamp] = field(default_factory=list)
    incomplete: dict[pd.Timestamp, int] = field(default_factory=dict)  # day -> trades missing

    def summary(self) -> str:
        bad = ", ".join(f"{d:%Y-%m-%d} ({n})" for d, n in self.incomplete.items()) or "none"
        return (
            f"{self.instrument}: {len(self.fetched)} days downloaded, {self.already} already "
            f"stored, {len(self.repaired)} repaired, incomplete: {bad}"
        )


def _merge(*frames: pd.DataFrame) -> pd.DataFrame:
    df = pd.concat([f for f in frames if len(f)], ignore_index=True)
    return df.drop_duplicates("trade_seq").sort_values("trade_seq").reset_index(drop=True)


def _fill_holes(api: DeribitREST, instrument: str, df: pd.DataFrame, lo, hi) -> pd.DataFrame:
    """Download again the window around each hole in the sequence numbers."""
    report = validate(df)
    if not report.missing:
        return df
    ts = df.set_index("trade_seq")["timestamp"]
    extra = []
    for a, b in report.missing:
        t0 = ts.get(a - 1, ts.iloc[0])
        t1 = ts.get(b + 1, ts.iloc[-1])
        window = api.trades(
            instrument,
            pd.Timestamp(t0, unit="ms", tz="UTC"),
            pd.Timestamp(t1 + 1, unit="ms", tz="UTC"),
            page=1000,
        )
        extra.append(window)
    merged = _merge(df, *extra)
    keep = (merged["timestamp"] >= lo) & (merged["timestamp"] < hi)
    return merged[keep].reset_index(drop=True)


def fetch_day(api: DeribitREST, instrument: str, day: pd.Timestamp, attempts: int = 2):
    lo = int(day.timestamp() * 1000)
    hi = int((day + ONE_DAY).timestamp() * 1000)
    fetch = getattr(api, "trades_parallel", api.trades)  # time slices downloaded concurrently
    df = fetch(instrument, day, day + ONE_DAY)
    repaired = False
    for _ in range(attempts):
        if not validate(df).missing:
            break
        df = _fill_holes(api, instrument, df, lo, hi)
        repaired = True
    return df, validate(df), repaired


def sync(
    instrument: str,
    start,
    end,
    store: TradeStore | None = None,
    api: DeribitREST | None = None,
    workers: int = 4,
    refresh: bool = False,
    on_day: Callable[[pd.Timestamp, int], None] | None = None,
) -> SyncResult:
    """Make the store hold every complete UTC day in [start, end)."""
    store = store or TradeStore()
    own_api = api is None
    api = api or DeribitREST()
    today = pd.Timestamp.now(tz="UTC").floor("D")
    days = [d for d in day_range(start, end) if d < today]
    result = SyncResult(instrument)

    def needed(d: pd.Timestamp) -> bool:
        info = store.info(instrument, d)
        return refresh or info is None or not info.complete

    todo = [d for d in days if needed(d)]
    result.already = len(days) - len(todo)

    def one(d: pd.Timestamp):
        df, report, repaired = fetch_day(api, instrument, d)
        store.write(instrument, d, df, report)
        if on_day is not None:
            on_day(d, report.n)
        return d, report, repaired

    try:
        with ThreadPoolExecutor(workers) as pool:
            for d, report, repaired in pool.map(one, todo):
                result.fetched.append(d)
                if repaired:
                    result.repaired.append(d)
                if report.missing:
                    result.incomplete[d] = report.n_missing
        _check_boundaries(api, instrument, store, days, result)
    finally:
        if own_api:
            api.close()
    for d in days:  # days stored earlier that are still flagged
        info = store.info(instrument, d)
        if info is not None and not info.complete and d not in result.incomplete:
            result.incomplete[d] = sum(b - a + 1 for a, b in info.report["missing"])
    return result


def _check_boundaries(api, instrument, store, days, result) -> None:
    """Repair holes in the sequence numbers across midnight."""
    for prev, nxt in zip(days, days[1:], strict=False):
        a, b = store.info(instrument, prev), store.info(instrument, nxt)
        if a is None or b is None or a.report["last_seq"] is None or b.report["first_seq"] is None:
            continue
        if b.report["first_seq"] == a.report["last_seq"] + 1:
            continue
        log.warning("%s: sequence hole across %s, re-downloading both days", instrument, nxt)
        for d in (prev, nxt):
            df, report, _ = fetch_day(api, instrument, d)
            store.write(instrument, d, df, report)
            result.repaired.append(d)


def boundary_gaps(store: TradeStore, instrument: str) -> pd.Series:
    """Missing sequence numbers between consecutive stored days (0 when continuous)."""
    s = store.status(instrument)
    return s["gap_before"].dropna() if len(s) else pd.Series(dtype=float)


def coverage(store: TradeStore, instrument: str, start, end) -> pd.DataFrame:
    """Status of every day in [start, end), including days not stored."""
    s = store.status(instrument)
    idx = day_range(start, end)
    out = s.reindex(idx) if len(s) else pd.DataFrame(index=idx)
    out.index.name = "day"
    return out


__all__ = ["SyncResult", "boundary_gaps", "coverage", "fetch_day", "sync", "utc"]
