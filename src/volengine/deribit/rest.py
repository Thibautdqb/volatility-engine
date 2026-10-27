"""Deribit public REST API (no credentials required)."""

from __future__ import annotations

import time

import httpx
import pandas as pd

BASE_URL = "https://www.deribit.com/api/v2"

# Deribit resolution codes -> bar length in seconds
RESOLUTIONS = {
    "1": 60,
    "3": 180,
    "5": 300,
    "10": 600,
    "15": 900,
    "30": 1800,
    "60": 3600,
    "120": 7200,
    "180": 10800,
    "360": 21600,
    "720": 43200,
    "1D": 86400,
}
_MAX_POINTS = 5000


class DeribitError(RuntimeError):
    pass


class DeribitREST:
    def __init__(self, base_url: str = BASE_URL, timeout: float = 15.0, retries: int = 3):
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

    def ohlc(
        self,
        instrument: str,
        resolution: str = "1",
        start: pd.Timestamp | None = None,
        end: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Trade-based OHLCV bars, paginated over the 5000-points-per-call limit."""
        if resolution not in RESOLUTIONS:
            raise ValueError(f"resolution must be one of {list(RESOLUTIONS)}")
        step_ms = RESOLUTIONS[resolution] * 1000
        end_ms = int((end or pd.Timestamp.now(tz="UTC")).timestamp() * 1000)
        start_ms = int(
            (start or pd.Timestamp(end_ms, unit="ms") - pd.Timedelta(days=1)).timestamp() * 1000
        )

        frames = []
        cursor = start_ms
        while cursor < end_ms:
            chunk_end = min(end_ms, cursor + step_ms * (_MAX_POINTS - 1))
            res = self._get(
                "get_tradingview_chart_data",
                instrument_name=instrument,
                resolution=resolution,
                start_timestamp=cursor,
                end_timestamp=chunk_end,
            )
            if res.get("status") == "ok" and res["ticks"]:
                frames.append(
                    pd.DataFrame(
                        {k: res[k] for k in ("open", "high", "low", "close", "volume")},
                        index=pd.to_datetime(res["ticks"], unit="ms", utc=True),
                    )
                )
            cursor = chunk_end + step_ms
        if not frames:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        df = pd.concat(frames)
        df = df[~df.index.duplicated(keep="last")].sort_index()
        df.index.name = "time"
        return df

    def dvol(
        self,
        currency: str = "BTC",
        resolution: str = "1D",
        start: pd.Timestamp | None = None,
        end: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Deribit Volatility Index (30-day implied vol, in vol points) as OHLC."""
        if resolution not in {"1", "60", "3600", "43200", "1D"}:
            raise ValueError("DVOL resolution is in seconds: 1, 60, 3600, 43200 or 1D")
        end_ms = int((end or pd.Timestamp.now(tz="UTC")).timestamp() * 1000)
        start_ms = int((start or pd.Timestamp("2021-03-24", tz="UTC")).timestamp() * 1000)
        rows: list[list[float]] = []
        cursor_end = end_ms
        while True:
            res = self._get(
                "get_volatility_index_data",
                currency=currency,
                resolution=resolution,
                start_timestamp=start_ms,
                end_timestamp=cursor_end,
            )
            rows.extend(res["data"])
            cont = res.get("continuation")
            if not cont or not res["data"] or cont <= start_ms:
                break
            cursor_end = cont
        df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close"])
        df.index = pd.to_datetime(df.pop("ts"), unit="ms", utc=True)
        df = df[~df.index.duplicated()].sort_index()
        df.index.name = "time"
        return df

    def index_price(self, index_name: str = "btc_usd") -> float:
        return float(self._get("get_index_price", index_name=index_name)["index_price"])
