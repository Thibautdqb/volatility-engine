"""Local store of the Deribit trade tape: one Parquet file per instrument and UTC day.

Layout under the root (``$VOLENGINE_DATA``, default ``~/.cache/volengine``)::

    trades/v1/BTC-PERPETUAL/2026-10-14.parquet   trades, schema in data.schema
    trades/v1/BTC-PERPETUAL/2026-10-14.json      manifest: integrity report, timestamps

Files are written to a temporary name and renamed, so an interrupted run never
leaves a truncated file. Only complete UTC days are stored.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from volengine.data.schema import COLUMNS, SCHEMA, SCHEMA_VERSION, Report

DATA_DIR = Path(os.environ.get("VOLENGINE_DATA", Path.home() / ".cache" / "volengine"))


def utc(ts) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def day_range(start, end) -> pd.DatetimeIndex:
    """UTC days d with start <= d < end."""
    return pd.date_range(utc(start).floor("D"), utc(end).floor("D"), freq="D", inclusive="left")


@dataclass(frozen=True)
class DayInfo:
    instrument: str
    day: pd.Timestamp
    report: dict
    fetched_at: str

    @property
    def complete(self) -> bool:
        return not self.report["missing"] and not self.report["duplicates"]


class TradeStore:
    def __init__(self, root: Path | str = DATA_DIR):
        self.root = Path(root) / "trades" / f"v{SCHEMA_VERSION}"

    def _base(self, instrument: str, day: pd.Timestamp) -> Path:
        return self.root / instrument / f"{utc(day):%Y-%m-%d}"

    def has(self, instrument: str, day) -> bool:
        base = self._base(instrument, day)
        return base.with_suffix(".parquet").exists() and base.with_suffix(".json").exists()

    def write(self, instrument: str, day, df: pd.DataFrame, report: Report) -> None:
        base = self._base(instrument, day)
        base.parent.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pandas(df[COLUMNS], schema=SCHEMA, preserve_index=False)
        tmp = base.with_suffix(".parquet.tmp")
        pq.write_table(table, tmp, compression="zstd")
        tmp.replace(base.with_suffix(".parquet"))
        info = {
            "instrument": instrument,
            "day": f"{utc(day):%Y-%m-%d}",
            "schema_version": SCHEMA_VERSION,
            "fetched_at": pd.Timestamp.now(tz="UTC").isoformat(),
            "report": report.as_dict(),
        }
        tmp = base.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(info, indent=1))
        tmp.replace(base.with_suffix(".json"))

    def info(self, instrument: str, day) -> DayInfo | None:
        path = self._base(instrument, day).with_suffix(".json")
        if not path.exists():
            return None
        d = json.loads(path.read_text())
        return DayInfo(instrument, utc(d["day"]), d["report"], d["fetched_at"])

    def days(self, instrument: str) -> list[pd.Timestamp]:
        folder = self.root / instrument
        if not folder.exists():
            return []
        return sorted(utc(p.stem) for p in folder.glob("*.json"))

    def read_day(self, instrument: str, day, columns: list[str] | None = None) -> pd.DataFrame:
        path = self._base(instrument, day).with_suffix(".parquet")
        return pq.read_table(path, columns=columns).to_pandas()

    def read(
        self, instrument: str, start, end, columns: list[str] | None = None
    ) -> pd.DataFrame:
        """Trades of the stored days in [start, end), sorted by trade_seq."""
        frames = [
            self.read_day(instrument, d, columns)
            for d in day_range(start, end)
            if self.has(instrument, d)
        ]
        if not frames:
            return SCHEMA.empty_table().to_pandas()[columns or COLUMNS]
        return pd.concat(frames, ignore_index=True)

    def status(self, instrument: str) -> pd.DataFrame:
        """One row per stored day: counts, sequence range and integrity flags."""
        rows = []
        for d in self.days(instrument):
            r = self.info(instrument, d).report
            rows.append(
                {
                    "day": d,
                    "trades": r["n"],
                    "first_seq": r["first_seq"],
                    "last_seq": r["last_seq"],
                    "missing": sum(b - a + 1 for a, b in r["missing"]),
                    "duplicates": r["duplicates"],
                    "book": r["kinds"].get("book", 0),
                    "block": r["kinds"].get("block", 0),
                    "combo": r["kinds"].get("combo", 0),
                }
            )
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        df = df.set_index("day")
        # Across midnight the sequence must continue without a hole.
        df["gap_before"] = (df["first_seq"] - df["last_seq"].shift() - 1).where(
            df.index.to_series().diff() == pd.Timedelta(days=1)
        )
        return df
