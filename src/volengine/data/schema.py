"""Trade schema, normalization of Deribit payloads, and integrity checks.

Deribit numbers the trades of each instrument with a ``trade_seq`` that increases by
exactly one per trade. A set of trades is therefore complete if and only if its
sequence numbers are contiguous, which is what :func:`validate` checks.

Three kinds of trades share the tape:

* ``book``: matched on the order book. Only these define the price used for
  volatility, live and historical.
* ``block``: negotiated off-book and reported afterwards; the price can sit several
  basis points away from the market.
* ``combo``: legs of a spread (e.g. future against perpetual) whose price is set by
  the spread, not by the perpetual's own book.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pyarrow as pa

SCHEMA_VERSION = 1
SCHEMA = pa.schema(
    [
        ("timestamp", pa.int64()),  # ms since epoch, exchange time
        ("trade_seq", pa.int64()),
        ("trade_id", pa.string()),
        ("price", pa.float64()),
        ("amount", pa.float64()),  # USD notional for inverse perpetuals
        ("is_buy", pa.bool_()),  # aggressor side
        ("index_price", pa.float64()),
        ("mark_price", pa.float64()),
        ("liquidation", pa.string()),  # "", "M", "T" or "MT"
        ("kind", pa.string()),  # "book", "block" or "combo"
    ]
)
COLUMNS = SCHEMA.names
KINDS = ("book", "block", "combo")


def empty() -> pd.DataFrame:
    return SCHEMA.empty_table().to_pandas()


def normalize(rows: list[dict]) -> pd.DataFrame:
    """Deribit trade dicts (REST or WebSocket) -> typed frame, sorted by ``trade_seq``."""
    if not rows:
        return empty()
    raw = pd.DataFrame(rows)

    def col(name, default):
        return raw[name] if name in raw else pd.Series(default, index=raw.index)

    kind = np.where(
        col("block_trade_id", None).notna(),
        "block",
        np.where(col("combo_id", None).notna(), "combo", "book"),
    )
    df = pd.DataFrame(
        {
            "timestamp": raw["timestamp"].astype("int64"),
            "trade_seq": raw["trade_seq"].astype("int64"),
            "trade_id": raw["trade_id"].astype(str),
            "price": raw["price"].astype(float),
            "amount": raw["amount"].astype(float),
            "is_buy": raw["direction"].eq("buy"),
            "index_price": col("index_price", np.nan).astype(float),
            "mark_price": col("mark_price", np.nan).astype(float),
            "liquidation": col("liquidation", "").fillna("").astype(str),
            "kind": kind,
        }
    )
    df = df.drop_duplicates("trade_seq").sort_values("trade_seq", kind="stable")
    return df.reset_index(drop=True)


@dataclass(frozen=True)
class Report:
    """Integrity of a set of trades from one instrument."""

    n: int
    first_seq: int | None
    last_seq: int | None
    missing: list[tuple[int, int]] = field(default_factory=list)  # inclusive seq ranges
    duplicates: int = 0
    time_reversals: int = 0  # timestamp decreasing while trade_seq increases
    bad_prices: int = 0  # non-positive or non-finite
    kinds: dict[str, int] = field(default_factory=dict)

    @property
    def n_missing(self) -> int:
        return sum(b - a + 1 for a, b in self.missing)

    @property
    def ok(self) -> bool:
        return not self.missing and not self.duplicates and not self.bad_prices

    def as_dict(self) -> dict:
        return {
            "n": self.n,
            "first_seq": self.first_seq,
            "last_seq": self.last_seq,
            "missing": [list(r) for r in self.missing],
            "duplicates": self.duplicates,
            "time_reversals": self.time_reversals,
            "bad_prices": self.bad_prices,
            "kinds": self.kinds,
        }


def missing_ranges(seq: np.ndarray) -> list[tuple[int, int]]:
    """Gaps in a sorted array of sequence numbers, as inclusive ranges."""
    if seq.size < 2:
        return []
    d = np.diff(seq)
    at = np.flatnonzero(d > 1)
    return [(int(seq[i] + 1), int(seq[i + 1] - 1)) for i in at]


def validate(df: pd.DataFrame) -> Report:
    if df.empty:
        return Report(0, None, None)
    seq = df["trade_seq"].to_numpy()
    order = np.argsort(seq, kind="stable")
    seq, ts = seq[order], df["timestamp"].to_numpy()[order]
    price = df["price"].to_numpy()
    return Report(
        n=len(df),
        first_seq=int(seq[0]),
        last_seq=int(seq[-1]),
        missing=missing_ranges(np.unique(seq)),
        duplicates=int(len(seq) - np.unique(seq).size),
        time_reversals=int((np.diff(ts) < 0).sum()),
        bad_prices=int((~np.isfinite(price) | (price <= 0)).sum()),
        kinds={k: int(v) for k, v in df["kind"].value_counts().items()},
    )
