"""Earnings straddle screener: implied move today vs. the historical distribution.

Hypothesis tested per ticker
----------------------------
H0: the expected absolute move to expiry equals the straddle-implied move, i.e.
a long ATM straddle held from the close before the announcement to expiry has
zero expected payoff net of premium (before costs).

* Primary statistic (decided before looking at results, to avoid picking the best
  of several): the **vol-adjusted edge**. Historical horizon-matched moves are
  expressed as multiples z_i of the pre-event daily vol, then rescaled by today's
  daily vol. This corrects for the fact that a 2020 move is not comparable to a
  2017 move in raw percent terms.
* Secondary: raw edge (mean |move| - implied), hit rate with a Wilson interval.
* Across a universe, p-values are adjusted with Benjamini-Hochberg; a ticker is
  flagged only if its q-value is below the threshold.

Known limitations (also in the README)
--------------------------------------
* Historical option prices are not used (they are not freely available): the
  historical implied move is unknown, only today's.
* Payoff at expiry ignores bid-ask, early exit and the residual time value of
  longer-dated expiries.
* With ~20-50 events per ticker, power is low: a 1-2 point edge is usually not
  distinguishable from zero for a single name.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy import stats

from volengine.earnings import data as edata
from volengine.earnings.implied import ImpliedEvent, implied_event, trading_days_between
from volengine.earnings.moves import current_daily_vol, event_windows, historical_moves
from volengine.stats import benjamini_hochberg, bootstrap_ci, wilson_interval

log = logging.getLogger(__name__)
MIN_EVENTS = 12


@dataclass
class EdgeStats:
    n: int
    implied_move: float
    mean_abs_move: float
    mean_abs_ci: tuple[float, float]
    median_abs_move: float
    implied_percentile: float  # share of past |moves| below today's implied move
    hit_rate: float  # share of past |moves| above today's implied move
    hit_rate_ci: tuple[float, float]
    raw_edge: float
    raw_edge_p: float
    predicted_move: float  # mean(z) * current daily vol
    adj_edge: float  # predicted_move - implied_move
    adj_edge_ci: tuple[float, float]
    adj_edge_p: float
    mean_z_reaction: float  # earnings-day move in units of a normal-day sigma
    small_sample: bool


def edge_statistics(moves: pd.DataFrame, implied_move: float, daily_vol: float) -> EdgeStats:
    abs_move = moves["holding_move"].abs().to_numpy()
    z = moves["z_holding"].to_numpy()
    n = abs_move.size
    hits = int(np.sum(abs_move > implied_move))

    raw_p = stats.ttest_1samp(abs_move, implied_move).pvalue if n > 2 else np.nan
    # Edge in return units; |log| and |simple| moves differ by < 1% of the move at these sizes.
    adj = z * daily_vol
    adj_p = stats.ttest_1samp(adj, implied_move).pvalue if n > 2 else np.nan
    lo, hi = bootstrap_ci(adj - implied_move) if n > 2 else (np.nan, np.nan)

    return EdgeStats(
        n=n,
        implied_move=implied_move,
        mean_abs_move=float(abs_move.mean()),
        mean_abs_ci=bootstrap_ci(abs_move) if n > 2 else (np.nan, np.nan),
        median_abs_move=float(np.median(abs_move)),
        implied_percentile=float(np.mean(abs_move < implied_move)),
        hit_rate=hits / n if n else np.nan,
        hit_rate_ci=wilson_interval(hits, n),
        raw_edge=float(abs_move.mean() - implied_move),
        raw_edge_p=float(raw_p),
        predicted_move=float(adj.mean()),
        adj_edge=float(adj.mean() - implied_move),
        adj_edge_ci=(lo, hi),
        adj_edge_p=float(adj_p),
        mean_z_reaction=float(moves["z_reaction"].mean()),
        small_sample=n < MIN_EVENTS,
    )


@dataclass
class TickerAnalysis:
    ticker: str
    event_date: pd.Timestamp
    timing: str
    entry: pd.Timestamp
    spot: float
    daily_vol: float
    implied: ImpliedEvent
    stats: EdgeStats
    moves: pd.DataFrame

    def row(self) -> dict:
        s = asdict(self.stats)
        return {
            "ticker": self.ticker,
            "event": self.event_date.date(),
            "timing": self.timing,
            "expiry": self.implied.expiry.date(),
            "days": self.implied.trading_days,
            "spot": self.spot,
            "atm_vol": self.implied.atm_vol,
            "event_move": self.implied.event_move,
            "stale_quotes": self.implied.straddle.stale,
            "spread_pct": self.implied.straddle.spread_pct,
            **{k: v for k, v in s.items() if not isinstance(v, tuple)},
            "adj_edge_lo": s["adj_edge_ci"][0],
            "adj_edge_hi": s["adj_edge_ci"][1],
            "hit_lo": s["hit_rate_ci"][0],
            "hit_hi": s["hit_rate_ci"][1],
        }


def analyze_ticker(
    ticker: str, horizon_days: int = 21, years: int = 15, r: float = 0.04
) -> TickerAnalysis:
    """Analyze the next announcement of ``ticker`` if it falls within ``horizon_days``."""
    cal = edata.earnings_calendar(ticker)
    today = pd.Timestamp.now(tz=edata.NY).tz_localize(None).normalize()
    upcoming = cal[(~cal["reported"]) & (cal["date"] >= today)]
    if upcoming.empty:
        raise ValueError(f"{ticker}: no upcoming earnings date")
    nxt = upcoming.iloc[0].copy()
    past_timing = cal.loc[cal["reported"], "timing"]
    if nxt["timing"] == "unknown" and len(past_timing) >= 8:
        # Yahoo often lacks the hour for future dates; companies rarely switch BMO/AMC.
        mode = past_timing.value_counts(normalize=True)
        if mode.iloc[0] >= 0.8 and mode.index[0] != "unknown":
            nxt["timing"] = mode.index[0]
    if (nxt["date"] - today).days > horizon_days:
        raise ValueError(f"{ticker}: next earnings on {nxt['date'].date()}, outside horizon")

    prices = edata.price_history(ticker, years)
    # Future sessions are needed to locate entry/reaction for the upcoming event.
    future = pd.bdate_range(prices.index[-1] + pd.Timedelta(days=1), periods=30)
    win = event_windows(pd.DataFrame([nxt]), prices.index.append(future)).iloc[0]

    chain = edata.option_chain(ticker, after=win["reaction"])
    imp = implied_event(chain, win["reaction"], r=r)
    # Horizon-match the history to a straddle bought at the last close and held to
    # expiry: same number of sessions before the announcement, same holding length.
    last_close = prices.index[-1]
    lead = trading_days_between(last_close, win["entry"])
    holding = max(1, trading_days_between(last_close, imp.expiry))

    past = cal[cal["reported"]]
    moves = historical_moves(prices, past, holding=holding, lead=lead)
    if moves.empty:
        raise ValueError(f"{ticker}: no usable past announcements")
    daily_vol = current_daily_vol(prices)
    st = edge_statistics(moves, imp.implied_move, daily_vol)
    return TickerAnalysis(
        ticker, nxt["date"], nxt["timing"], win["entry"], chain.spot, daily_vol, imp, st, moves
    )


def scan(tickers, horizon_days: int = 21, fdr: float = 0.10) -> pd.DataFrame:
    """Analyze each ticker; adds Benjamini-Hochberg q-values on the primary test."""
    rows, errors = [], {}
    for t in tickers:
        try:
            rows.append(analyze_ticker(t, horizon_days).row())
        except Exception as exc:  # one bad ticker must not stop the scan
            errors[t] = str(exc)
            log.warning("%s skipped: %s", t, exc)
    df = pd.DataFrame(rows)
    if not df.empty:
        df["q_value"] = benjamini_hochberg(df["adj_edge_p"].to_numpy())
        df["significant"] = (df["q_value"] < fdr) & ~df["small_sample"]
        df["signal"] = np.where(
            df["significant"], np.where(df["adj_edge"] > 0, "implied cheap", "implied rich"), "-"
        )
        df = df.sort_values("adj_edge_p").reset_index(drop=True)
    df.attrs["errors"] = errors
    df.attrs["fdr"] = fdr
    return df
