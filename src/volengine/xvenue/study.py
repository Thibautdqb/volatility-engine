"""Replay a recording through the engine and measure what the flags were worth.

* opportunities per hour, size and duration, by pair, for several fee schedules;
* survival: share of episodes lasting longer than the latency, i.e. that could still
  have been hit when the orders arrived;
* the volatility dimension: opportunities by volatility tercile at their start;
* lead-lag between venues on the receipt clock, the only clock that matters for
  someone trading from this machine (and on exchange time, for comparison).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from volengine.xvenue.engine import TAKER_FEES_BP, Engine, EngineConfig
from volengine.xvenue.feeds import Disconnected, Print, Quote, Rate

FEE_SCHEDULES = {
    "taker, public base tier": TAKER_FEES_BP,
    "taker, high-volume tier": {
        "deribit:perp": 2.5,
        "binance:perp": 2.0,
        "binance:spot": 2.0,
        "coinbase:spot": 5.0,
    },
    "no fees (maker rebates)": dict.fromkeys(TAKER_FEES_BP, 0.0),
}


def events(df: pd.DataFrame):
    """Recorded rows back to feed events, in order of receipt."""
    for r in df.itertuples(index=False):
        if r.kind == "quote":
            ex = None if pd.isna(r.exchange_ms) else r.exchange_ms
            yield Quote(r.market, r.asset, r.local_ms, ex, r.bid, r.bid_size, r.ask, r.ask_size)
        elif r.kind == "print":
            yield Print(
                r.market, r.asset, r.local_ms, r.exchange_ms, r.price, r.size, bool(r.is_buy)
            )
        elif r.kind == "rate":
            yield Rate(r.name, r.local_ms, r.value)
        else:
            yield Disconnected(r.market)


def replay(df: pd.DataFrame, config: EngineConfig, assets=("BTC", "ETH")) -> Engine:
    markets = tuple(m for m in df["market"].dropna().unique() if m)
    engine = Engine(assets, tuple(sorted(markets)), config)
    for ev in events(df):
        engine.on_event(ev)
    return engine


def opportunities_table(engine: Engine, hours: float, latency_ms: float) -> pd.DataFrame:
    if not engine.opportunities:
        return pd.DataFrame()
    o = pd.DataFrame([x.__dict__ | {"duration_ms": x.duration_ms} for x in engine.opportunities])
    o["sigma"] = np.hypot(o["sigma_sell_bp"], o["sigma_buy_bp"])
    o["pair"] = o["asset"] + " sell " + o["sell_on"] + " / buy " + o["buy_on"]
    g = o.groupby("pair")
    return pd.DataFrame(
        {
            "per hour": g.size() / hours,
            "net bp (median)": g["peak_net_bp"].median(),
            "net bp (p90)": g["peak_net_bp"].quantile(0.9),
            "duration ms (median)": g["duration_ms"].median(),
            "lasts > latency": g["duration_ms"].apply(lambda d: (d > latency_ms).mean()),
            "notional USD (median)": g["notional_usd"].median(),
        }
    ).sort_values("per hour", ascending=False)


def by_volatility(engine: Engine) -> pd.DataFrame:
    """Opportunities by tercile of the combined volatility of the two legs at the peak."""
    if not engine.opportunities:
        return pd.DataFrame()
    o = pd.DataFrame([x.__dict__ | {"duration_ms": x.duration_ms} for x in engine.opportunities])
    o["sigma"] = np.hypot(o["sigma_sell_bp"], o["sigma_buy_bp"])
    o["tercile"] = pd.qcut(o["sigma"], 3, labels=["low vol", "mid vol", "high vol"])
    g = o.groupby("tercile", observed=True)
    return pd.DataFrame(
        {
            "episodes": g.size(),
            "sigma bp/s (median)": g["sigma"].median(),
            "net bp (median)": g["peak_net_bp"].median(),
            "buffer bp (median)": g["buffer_bp"].median(),
            "duration ms (median)": g["duration_ms"].median(),
        }
    )


def mids(df: pd.DataFrame, market: str, asset: str, clock: str, step_ms: int, t0, n) -> np.ndarray:
    q = df[(df["kind"] == "quote") & (df["market"] == market) & (df["asset"] == asset)]
    q = q.dropna(subset=[clock])
    k = ((q[clock].to_numpy() - t0) // step_ms).astype(np.int64)
    ok = (k >= 0) & (k < n)
    out = np.full(n, np.nan)
    out[k[ok]] = ((q["bid"] + q["ask"]) / 2).to_numpy()[ok]
    return pd.Series(out).ffill().to_numpy()


def lead_lag(df: pd.DataFrame, asset: str, a: str, b: str, clock: str, step_ms: int = 50):
    """corr(r_a(t), r_b(t - lag)) for lags in ms; positive peak = b moves first."""
    t0 = df["local_ms"].min()
    n = int((df["local_ms"].max() - t0) // step_ms)
    ra = np.diff(np.log(mids(df, a, asset, clock, step_ms, t0, n)))
    rb = np.diff(np.log(mids(df, b, asset, clock, step_ms, t0, n)))
    lags = np.arange(-20, 21)
    out = []
    for lag in lags:
        x, y = (ra[lag:], rb[: len(rb) - lag]) if lag >= 0 else (ra[:lag], rb[-lag:])
        m = np.isfinite(x) & np.isfinite(y) & ((x != 0) | (y != 0))
        out.append(np.corrcoef(x[m], y[m])[0, 1] if m.sum() > 100 else np.nan)
    s = pd.Series(out, index=lags * step_ms)
    return s


@dataclass
class XStudy:
    hours: float
    tables: dict[str, pd.DataFrame]
    vol: pd.DataFrame
    lead: pd.DataFrame
    divergences: int

    def summary(self) -> str:
        lines = [f"recording: {self.hours:.1f} hours", ""]
        for name, t in self.tables.items():
            lines.append(f"opportunities, fees = {name}")
            lines.append(t.to_string(float_format=lambda x: f"{x:.2f}") if len(t) else "  none")
            lines.append("")
        lines += ["by volatility at the peak (no-fee schedule)", self.vol.to_string(), ""]
        lines += ["lead-lag of mid returns (positive = the second market moves first)"]
        lead = self.lead.copy()
        lead["peak lag ms"] = lead["peak lag ms"].map("{:+.0f}".format)
        for col in ("corr at peak", "corr at 0"):
            lead[col] = lead[col].map("{:.3f}".format)
        lines.append(lead.to_string())
        lines.append(f"\nvolatility divergences flagged: {self.divergences}")
        return "\n".join(lines)


def study(df: pd.DataFrame, latency_ms: float = 300.0) -> XStudy:
    hours = (df["local_ms"].max() - df["local_ms"].min()) / 3.6e6
    tables, last = {}, None
    for name, fees in FEE_SCHEDULES.items():
        last = replay(df, EngineConfig(latency_ms=latency_ms, fees_bp=dict(fees)))
        tables[name] = opportunities_table(last, hours, latency_ms)
    rows = {}
    for asset in ("BTC", "ETH"):
        for other in ("binance:perp", "binance:spot", "coinbase:spot"):
            for clock in ("local_ms", "exchange_ms"):
                if other == "binance:spot" and clock == "exchange_ms":
                    continue  # no exchange time on Binance spot quotes
                s = lead_lag(df, asset, "deribit:perp", other, clock)
                rows[(asset, other, clock)] = {
                    "peak lag ms": s.idxmax(),
                    "corr at peak": s.max(),
                    "corr at 0": s.loc[0],
                }
    lead = pd.DataFrame(rows).T
    lead.index.names = ["asset", "vs deribit:perp", "clock"]
    return XStudy(hours, tables, by_volatility(last), lead, len(last.divergences))
