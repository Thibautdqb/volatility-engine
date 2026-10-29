"""Record the merged feeds to Parquet, so that every study replays exactly what live saw.

Files: ``<root>/xvenue/<YYYY-MM-DD>/<HH>-<n>.parquet``, one per flush (every minute by
default), written to a temporary name and renamed. One row per event; columns that do
not apply to an event type are null.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from volengine.data.store import DATA_DIR
from volengine.xvenue.feeds import MARKETS, Disconnected, Print, Quote, Rate, stream

log = logging.getLogger(__name__)
ROOT = DATA_DIR / "xvenue"
COLUMNS = [
    "kind", "market", "asset", "local_ms", "exchange_ms",
    "bid", "bid_size", "ask", "ask_size", "price", "size", "is_buy", "name", "value",
]  # fmt: skip
KIND = {Quote: "quote", Print: "print", Rate: "rate", Disconnected: "disconnected"}


def to_row(ev) -> dict:
    row = asdict(ev)
    row["kind"] = KIND[type(ev)]
    if isinstance(ev, Rate):
        row["market"] = ""
    return row


def write(rows: list[dict], root: Path = ROOT) -> Path | None:
    if not rows:
        return None
    df = pd.DataFrame(rows).reindex(columns=COLUMNS)
    t = pd.Timestamp(rows[0]["local_ms"], unit="ms", tz="UTC")
    folder = root / f"{t:%Y-%m-%d}"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{t:%H}-{int(rows[0]['local_ms'])}.parquet"
    tmp = path.with_suffix(".tmp")
    df.to_parquet(tmp, compression="zstd")
    tmp.replace(path)
    return path


def load(start, end, root: Path = ROOT) -> pd.DataFrame:
    """Recorded events with local time in [start, end), in order of receipt."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    start = start.tz_localize("UTC") if start.tzinfo is None else start
    end = end.tz_localize("UTC") if end.tzinfo is None else end
    files = []
    for day in pd.date_range(start.floor("D"), end, freq="D"):
        files += sorted((root / f"{day:%Y-%m-%d}").glob("*.parquet"))
    if not files:
        return pd.DataFrame(columns=COLUMNS)
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    lo, hi = start.timestamp() * 1000, end.timestamp() * 1000
    df = df[(df["local_ms"] >= lo) & (df["local_ms"] < hi)]
    return df.sort_values("local_ms", kind="stable").reset_index(drop=True)


async def record(
    assets=("BTC",),
    markets=MARKETS,
    flush_seconds: float = 60.0,
    root: Path = ROOT,
    duration: float | None = None,
) -> None:
    rows: list[dict] = []
    last = time.monotonic()
    t0 = last
    async for ev in stream(assets, markets):
        rows.append(to_row(ev))
        now = time.monotonic()
        if now - last >= flush_seconds:
            path = await asyncio.to_thread(write, rows, root)
            log.info("wrote %d events to %s", len(rows), path)
            rows, last = [], now
        if duration is not None and now - t0 >= duration:
            break
    write(rows, root)
