"""What the option market prices for an earnings announcement.

Two complementary measures:

* **Straddle implied move**: ATM straddle price / spot for the first expiry after
  the announcement. At expiry the straddle pays |S_T - K|, so this is the
  market's (risk-neutral) expected absolute move to expiry, including the normal
  diffusion on the non-event days.
* **Event volatility from the term structure**: with two expiries after the event,
  total implied variance w_i = sigma_i^2 T_i decomposes as
  w_i = sigma_d^2 n_i + e^2 (n_i trading days of diffusion, one event jump of
  variance e^2), which identifies the event-only standard deviation e. Its
  expected absolute move under normality is e * sqrt(2/pi).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from scipy import optimize
from scipy.stats import norm

SQRT_2_OVER_PI = np.sqrt(2.0 / np.pi)
_HOLIDAYS = (
    USFederalHolidayCalendar()
    .holidays("2000-01-01", "2035-12-31")
    .to_numpy()
    .astype("datetime64[D]")
)


def trading_days_between(start: pd.Timestamp, end: pd.Timestamp) -> int:
    """Sessions in (start, end], approximating the NYSE calendar with US federal holidays."""
    s = np.datetime64(pd.Timestamp(start).date(), "D") + 1
    e = np.datetime64(pd.Timestamp(end).date(), "D") + 1
    return int(np.busday_count(s, e, holidays=_HOLIDAYS))


# ---------------------------------------------------------------------------
# Black-Scholes
# ---------------------------------------------------------------------------


def bs_price(S, K, T, sigma, r=0.0, q=0.0, kind="call"):
    sqrt_t = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    if kind == "call":
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


def straddle_implied_vol(price: float, S: float, K: float, T: float, r=0.0, q=0.0) -> float:
    """Volatility that reprices the straddle (call + put at the same strike)."""

    def f(sigma):
        return (
            bs_price(S, K, T, sigma, r, q, "call") + bs_price(S, K, T, sigma, r, q, "put") - price
        )

    lo, hi = 1e-4, 10.0
    if f(lo) > 0 or f(hi) < 0:
        return float("nan")
    return float(optimize.brentq(f, lo, hi, xtol=1e-8))


# ---------------------------------------------------------------------------
# Chain handling
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Straddle:
    strike: float
    price: float
    spread_pct: float  # bid-ask width as a fraction of mid
    stale: bool  # True when last-trade prices had to be used (no live quotes)


def _quotes(df: pd.DataFrame) -> pd.DataFrame:
    df = df[["strike", "bid", "ask", "lastPrice"]].copy()
    live = (df["bid"] > 0) & (df["ask"] > 0) & (df["ask"] >= df["bid"])
    df["mid"] = np.where(live, (df["bid"] + df["ask"]) / 2, df["lastPrice"])
    df["width"] = np.where(live, df["ask"] - df["bid"], np.nan)
    df["live"] = live
    return df.set_index("strike")


def atm_straddle(calls: pd.DataFrame, puts: pd.DataFrame, spot: float) -> Straddle:
    """ATM straddle, linearly interpolated in strike between the two strikes that
    bracket spot (removes the intrinsic-value bias of picking the nearest strike)."""
    c, p = _quotes(calls), _quotes(puts)
    both = c.join(p, lsuffix="_c", rsuffix="_p", how="inner")
    both = both[(both["mid_c"] > 0) & (both["mid_p"] > 0)]
    if both.empty:
        raise ValueError("no strike with both a call and a put price")
    both["straddle"] = both["mid_c"] + both["mid_p"]
    both["width"] = both["width_c"] + both["width_p"]
    both["live"] = both["live_c"] & both["live_p"]
    strikes = both.index.to_numpy()
    below, above = strikes[strikes <= spot], strikes[strikes >= spot]
    if below.size and above.size:
        k0, k1 = below.max(), above.min()
        w = 0.0 if k1 == k0 else (spot - k0) / (k1 - k0)
        rows = both.loc[[k0, k1]]
        price = (1 - w) * rows["straddle"].iloc[0] + w * rows["straddle"].iloc[1]
        width = (1 - w) * rows["width"].iloc[0] + w * rows["width"].iloc[1]
        live = bool(rows["live"].all())
        return Straddle(float(spot), float(price), float(width / price), not live)
    k = strikes[np.argmin(np.abs(strikes - spot))]
    row = both.loc[k]
    return Straddle(
        float(k), float(row["straddle"]), float(row["width"] / row["straddle"]), not row["live"]
    )


def event_variance(w1: float, n1: int, w2: float, n2: int) -> tuple[float, float]:
    """Split total implied variances of two post-event expiries into
    (event variance e^2, diffusive variance per trading day).

    Returns NaN event variance when the term structure is inconsistent with a
    single positive jump (e.g. upward-sloping beyond what diffusion explains).
    """
    if n2 <= n1:
        raise ValueError("second expiry must be later than the first")
    daily = (w2 - w1) / (n2 - n1)
    e2 = w1 - daily * n1
    if daily <= 0 or e2 <= 0:
        return float("nan"), float(daily)
    return float(e2), float(daily)


@dataclass(frozen=True)
class ImpliedEvent:
    expiry: pd.Timestamp
    trading_days: int  # sessions from now to expiry
    straddle: Straddle
    implied_move: float  # straddle / spot
    atm_vol: float  # annualized, from inverting the straddle
    event_move: float  # expected |move| attributable to the event only (term structure)
    event_vol_daily: float  # e, standard deviation of the event-day return
    diffusive_vol: float  # annualized, ex-event


def _total_variance(
    straddle: Straddle, spot: float, expiry: pd.Timestamp, now: pd.Timestamp, r: float
):
    T = max((expiry.tz_localize(now.tz) + pd.Timedelta(hours=16) - now).total_seconds(), 3600.0)
    T /= 365.25 * 86400
    iv = straddle_implied_vol(straddle.price, spot, straddle.strike, T, r)
    return iv, iv**2 * T


def implied_event(chain, reaction_date: pd.Timestamp, r: float = 0.04) -> ImpliedEvent:
    """Price the earnings event from the chain. ``reaction_date`` is the first
    session whose close reflects the announcement."""
    now = chain.asof
    post = [e for e in chain.expiries if e.normalize() >= reaction_date.normalize()]
    if not post:
        raise ValueError("no listed expiry after the announcement")
    e1 = post[0]
    s1 = atm_straddle(*chain.chains[e1], chain.spot)
    iv1, w1 = _total_variance(s1, chain.spot, e1, now, r)
    n1 = trading_days_between(now.tz_localize(None), e1) + (1 if now.hour < 16 else 0)

    e_move = e_daily = d_vol = float("nan")
    if len(post) > 1:
        e2 = post[1]
        s2 = atm_straddle(*chain.chains[e2], chain.spot)
        _, w2 = _total_variance(s2, chain.spot, e2, now, r)
        n2 = trading_days_between(now.tz_localize(None), e2) + (1 if now.hour < 16 else 0)
        ev, daily = event_variance(w1, n1, w2, n2)
        if np.isfinite(ev):
            e_daily = float(np.sqrt(ev))
            e_move = float(SQRT_2_OVER_PI * e_daily)
            d_vol = float(np.sqrt(daily * 252))
    return ImpliedEvent(
        expiry=e1,
        trading_days=n1,
        straddle=s1,
        implied_move=s1.price / chain.spot,
        atm_vol=iv1,
        event_move=e_move,
        event_vol_daily=e_daily,
        diffusive_vol=d_vol,
    )
