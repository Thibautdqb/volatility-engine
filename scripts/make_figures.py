"""Regenerate the figures in docs/img from live data.

python scripts/make_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

from volengine import estimators as est
from volengine.deribit.rest import DeribitREST
from volengine.garch import fit_garch
from volengine.research import DAYS_PER_YEAR, load_daily, variance_risk_premium

OUT = Path(__file__).resolve().parents[1] / "docs" / "img"
plt.rcParams.update(
    {
        "figure.dpi": 130,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "font.size": 9,
        "legend.frameon": False,
    }
)


def implied_vs_realized(instrument: str = "BTC-PERPETUAL") -> None:
    data = load_daily(instrument, "2019-01-01")
    vrp = variance_risk_premium(data).series
    g = fit_garch(data.returns)
    cond = np.sqrt(g.conditional_variance * DAYS_PER_YEAR).loc[vrp.index[0] :]

    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(vrp.index, vrp["implied"], lw=1, label="DVOL (30d implied, observed at t)")
    ax.plot(vrp.index, vrp["realized"], lw=1, label="Realized vol over (t, t+30d]")
    ax.plot(cond.index, cond, lw=0.6, alpha=0.6, color="grey", label="GARCH(1,1) conditional vol")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_title(f"{instrument}: implied vs subsequently realized volatility")
    ax.legend(loc="upper right", ncol=3)
    fig.tight_layout()
    fig.savefig(OUT / "implied_vs_realized.png")


def signature(instrument: str = "BTC-PERPETUAL", days: int = 14) -> None:
    end = pd.Timestamp.now(tz="UTC").floor("min")
    with DeribitREST() as api:
        bars = api.ohlc(instrument, "1", end - pd.Timedelta(days=days), end)
    steps = (1, 2, 3, 5, 10, 15, 30, 60, 120)
    sig = est.signature_plot(bars["close"], 60, steps=steps)
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.plot(sig.index / 60, sig.to_numpy(), marker="o", ms=3)
    ax.set_xscale("log")
    ax.set_xticks([1, 5, 15, 60, 120], labels=["1m", "5m", "15m", "1h", "2h"])
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("sampling interval")
    ax.set_title(f"{instrument}: volatility signature, last {days} days")
    fig.tight_layout()
    fig.savefig(OUT / "signature.png")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    implied_vs_realized()
    signature()
