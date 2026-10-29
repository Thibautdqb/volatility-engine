"""Deribit public REST API (no credentials required).

Trades come from ``history.deribit.com``, which serves the full trade tape with up to
10,000 trades per request and is updated within seconds of the live exchange.
"""

from __future__ import annotations

import itertools
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import pandas as pd

from volengine.data.schema import normalize

HISTORY_URL = "https://history.deribit.com/api/v2"
PAGE = 10_000
log = logging.getLogger(__name__)


class DeribitError(RuntimeError):
    pass


class DeribitREST:
    """Thread-safe client with retries, exponential backoff and a request-rate cap."""

    def __init__(
        self,
        base_url: str = HISTORY_URL,
        timeout: float = 60.0,
        retries: int = 6,
        max_rate: float = 8.0,  # requests per second, shared by all threads
    ):
        self._client = httpx.Client(base_url=base_url, timeout=timeout)
        self._retries = retries
        self._interval = 1.0 / max_rate
        self._next = 0.0
        self._lock = threading.Lock()

    def close(self) -> None:
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _throttle(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait = self._next - now
            self._next = max(now, self._next) + self._interval
        if wait > 0:
            time.sleep(wait)

    def _get(self, method: str, **params):
        last: Exception | None = None
        for attempt in range(self._retries):
            self._throttle()
            try:
                resp = self._client.get(f"/public/{method}", params=params)
                if resp.status_code == 429 or resp.status_code >= 500:
                    raise httpx.HTTPStatusError(
                        f"HTTP {resp.status_code}", request=resp.request, response=resp
                    )
                resp.raise_for_status()
                payload = resp.json()
                if "error" in payload:
                    raise DeribitError(payload["error"])
                return payload["result"]
            except (httpx.TransportError, httpx.HTTPStatusError, ValueError) as exc:
                last = exc
                delay = min(2.0**attempt, 30.0)
                log.warning("%s failed (%s), retry in %.0fs", method, exc, delay)
                time.sleep(delay)
        raise DeribitError(f"{method} failed after {self._retries} attempts: {last}")

    def trades(
        self, instrument: str, start: pd.Timestamp, end: pd.Timestamp, page: int = PAGE
    ) -> pd.DataFrame:
        """Every trade with a timestamp in [start, end), normalized and sorted by trade_seq.

        Pages are chained on the last timestamp received, so trades sharing that
        millisecond come back twice and are de-duplicated by sequence number.
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
        return normalize(rows)

    def trades_parallel(
        self,
        instrument: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
        chunk_hours: float = 2.0,
        workers: int = 6,
    ) -> pd.DataFrame:
        """Same as :meth:`trades`, downloading consecutive time slices concurrently."""
        step = pd.Timedelta(hours=chunk_hours)
        edges = list(pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq=step))
        if not edges or edges[-1] < pd.Timestamp(end):
            edges.append(pd.Timestamp(end))
        if len(edges) < 2:
            return self.trades(instrument, start, end)
        with ThreadPoolExecutor(workers) as pool:
            slices = itertools.pairwise(edges)
            parts = list(pool.map(lambda ab: self.trades(instrument, *ab), slices))
        df = pd.concat([p for p in parts if len(p)], ignore_index=True)
        return df.drop_duplicates("trade_seq").sort_values("trade_seq").reset_index(drop=True)
