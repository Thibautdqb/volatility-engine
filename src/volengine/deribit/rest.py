"""Deribit public REST API (no credentials required).

Trades come from ``history.deribit.com``, which serves the full trade tape with up to
10,000 trades per request and is updated within seconds of the live exchange.
"""

from __future__ import annotations

import time

import httpx
import pandas as pd

HISTORY_URL = "https://history.deribit.com/api/v2"
_PAGE = 10_000
TRADE_COLUMNS = [
    "timestamp",
    "trade_id",
    "price",
    "amount",
    "direction",  # aggressor side: "buy" or "sell"
    "index_price",  # Deribit spot index at the time of the trade
    "liquidation",  # empty, or which side(s) were liquidated: "M", "T" or "MT"
]


class DeribitError(RuntimeError):
    pass


class DeribitREST:
    def __init__(self, base_url: str = HISTORY_URL, timeout: float = 60.0, retries: int = 4):
        self._client = httpx.Client(base_url=base_url, timeout=timeout)
        self._retries = retries

    def close(self) -> None:
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _get(self, method: str, **params):
        last = None
        for attempt in range(self._retries):
            try:
                resp = self._client.get(f"/public/{method}", params=params)
                if resp.status_code == 429:
                    raise httpx.HTTPStatusError("rate limited", request=resp.request, response=resp)
                resp.raise_for_status()
                payload = resp.json()
                if "error" in payload:
                    raise DeribitError(payload["error"])
                return payload["result"]
            except (httpx.TransportError, httpx.HTTPStatusError) as exc:
                last = exc
                time.sleep(2**attempt)
        raise DeribitError(f"{method} failed after {self._retries} attempts: {last}")

    def trades(
        self, instrument: str, start: pd.Timestamp, end: pd.Timestamp, page: int = _PAGE
    ) -> pd.DataFrame:
        """Every trade in [start, end), oldest first.

        Columns: see ``TRADE_COLUMNS``; ``timestamp`` is in ms since epoch, exchange time.
        Pages are chained on the last timestamp received; trades sharing that millisecond
        are de-duplicated by id.
        """
        start_ms = int(pd.Timestamp(start).timestamp() * 1000)
        end_ms = int(pd.Timestamp(end).timestamp() * 1000) - 1  # the API bound is inclusive
        rows: list[dict] = []
        cursor = start_ms
        while cursor <= end_ms:
            res = self._get(
                "get_last_trades_by_instrument_and_time",
                instrument_name=instrument,
                start_timestamp=cursor,
                end_timestamp=end_ms,
                count=page,
                sorting="asc",
            )
            batch = res["trades"]
            rows.extend(batch)
            if not res.get("has_more") or not batch:
                break
            last = batch[-1]["timestamp"]
            cursor = last if last > cursor else cursor + 1
        if not rows:
            return pd.DataFrame(columns=TRADE_COLUMNS)
        df = pd.DataFrame(rows).reindex(columns=TRADE_COLUMNS)
        df["liquidation"] = df["liquidation"].fillna("")
        df["trade_id"] = df["trade_id"].astype(str)
        df = df.drop_duplicates("trade_id").sort_values("timestamp", kind="stable")
        return df.reset_index(drop=True)
