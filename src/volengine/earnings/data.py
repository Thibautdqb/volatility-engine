"""Equity data access through yfinance (earnings calendar, prices, option chains).

yfinance scrapes Yahoo Finance: it is free and keyless but has no SLA, so every
function here validates what it receives and fails loudly rather than silently
returning partial data.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import pandas as pd

try:
    import yfinance as yf
except ImportError as exc:  # pragma: no cover
    raise ImportError("the earnings module needs `pip install volengine[earnings]`") from exc

# yfinance re-enables deprecation warnings for itself on import; the pandas ones it
# triggers internally are not actionable for users of this package.
warnings.filterwarnings("ignore", message=".*inplace keyword.*", module="^yfinance")

NY = "America/New_York"


def classify_timing(ts: pd.Timestamp) -> str:
    """BMO (before market open), AMC (after market close) or unknown."""
    if ts.hour == 0 and ts.minute == 0:
        return "unknown"
    t = ts.hour + ts.minute / 60.0
    if t < 9.5:
        return "BMO"
    if t >= 16.0:
        return "AMC"
    return "unknown"


def earnings_calendar(ticker: str, limit: int = 60) -> pd.DataFrame:
    """Past and upcoming earnings announcements with their timing.

    Columns: ``datetime`` (New York time), ``date``, ``timing``, ``eps_estimate``,
    ``eps_reported``, ``reported`` (bool).
    """
    raw = yf.Ticker(ticker).get_earnings_dates(limit=limit)
    if raw is None or raw.empty:
        raise ValueError(f"no earnings dates for {ticker}")
    idx = raw.index.tz_convert(NY) if raw.index.tz is not None else raw.index.tz_localize(NY)
    df = pd.DataFrame(
        {
            "datetime": idx,
            "eps_estimate": raw["EPS Estimate"].to_numpy(),
            "eps_reported": raw["Reported EPS"].to_numpy(),
        }
    )
    df["date"] = df["datetime"].dt.tz_localize(None).dt.normalize()
    df["timing"] = df["datetime"].map(classify_timing)
    df["reported"] = df["eps_reported"].notna()
    return df.sort_values("datetime").drop_duplicates("date").reset_index(drop=True)


def price_history(ticker: str, years: int = 15) -> pd.DataFrame:
    """Split- and dividend-adjusted daily OHLC (tz-naive New York dates)."""
    df = yf.Ticker(ticker).history(period=f"{years}y", auto_adjust=True)
    if df.empty:
        raise ValueError(f"no price history for {ticker}")
    df.index = df.index.tz_localize(None).normalize()
    return df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]


@dataclass
class OptionChain:
    ticker: str
    spot: float
    asof: pd.Timestamp
    expiries: list[pd.Timestamp]
    chains: dict[pd.Timestamp, tuple[pd.DataFrame, pd.DataFrame]]  # expiry -> (calls, puts)


def option_chain(ticker: str, after: pd.Timestamp | None = None, count: int = 2) -> OptionChain:
    """Chains for the first ``count`` expiries on or after ``after`` (default: today)."""
    tk = yf.Ticker(ticker)
    listed = [pd.Timestamp(e) for e in tk.options]
    if not listed:
        raise ValueError(f"no listed options for {ticker}")
    cutoff = (after or pd.Timestamp.now(tz=NY).tz_localize(None)).normalize()
    expiries = [e for e in listed if e >= cutoff][:count]
    if not expiries:
        raise ValueError(f"{ticker}: no listed expiry on or after {cutoff.date()}")
    spot = float(tk.history(period="5d")["Close"].iloc[-1])
    chains = {}
    for e in expiries:
        ch = tk.option_chain(e.strftime("%Y-%m-%d"))
        chains[e] = (ch.calls.copy(), ch.puts.copy())
    return OptionChain(ticker, spot, pd.Timestamp.now(tz=NY), expiries, chains)
