"""Regenerate the figures in docs/img from the Deribit trade tape.

python scripts/make_figures.py

Every forecast shown is out of sample: models are fitted on one period
(2026-09-14 to 2026-10-13, or the reverse) and drawn on the other.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from volengine.detector import DetectorConfig, Level
from volengine.forecast import Regime, fit, har_inputs, track_regimes
from volengine.sampling import bar_closes
from volengine.study import forecast_study, load_trades, prepare_trades, run_study

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
INSTRUMENTS = ["BTC-PERPETUAL", "ETH-PERPETUAL"]
PERIODS = {"Sep 14-Oct 13": ("2026-09-14", "2026-10-14"), "Oct 14-27": ("2026-10-14", "2026-10-28")}
CONFIG = DetectorConfig()
REGIME_FILL = {Regime.ACTIVE: ("tab:orange", 0.12), Regime.STRESSED: ("tab:red", 0.22)}


def regimes() -> None:
    """P(violent next 5 minutes) by forecast regime, in the four out-of-sample tests."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    names = [r.name.lower() for r in Regime]
    a, b = list(PERIODS)
    for ax, inst in zip(axes, INSTRUMENTS, strict=True):
        trades = {p: load_trades(inst, *span) for p, span in PERIODS.items()}
        for k, (train, test) in enumerate(((a, b), (b, a))):
            t = forecast_study(trades[train], trades[test], inst).regimes.loc[names]
            x = np.arange(len(names)) + (k - 0.5) * 0.36
            err = [100 * (t["P(violent)"] - t["ci low"]), 100 * (t["ci high"] - t["P(violent)"])]
            ax.bar(x, 100 * t["P(violent)"], width=0.34, label=f"tested on {test}")
            ax.errorbar(x, 100 * t["P(violent)"], yerr=err, fmt="none", color="black", lw=0.8)
        ax.set_xticks(range(len(names)), [n.upper() for n in names])
        ax.set_title(inst)
    axes[0].set_ylabel("P(next 5 min vol >= 2x normal), %")
    axes[0].legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "regimes.png")


def example() -> None:
    """2026-10-14 around 12:30 UTC, with a model fitted on September only."""
    inst = "BTC-PERPETUAL"
    model = fit(prepare_trades(load_trades(inst, *PERIODS["Sep 14-Oct 13"]), CONFIG), CONFIG, inst)
    day = load_trades(inst, "2026-10-14", "2026-10-15")
    p = prepare_trades(day, CONFIG)
    p["forecast"] = np.exp(model.predict(har_inputs(p, model)))
    p["regime"] = track_regimes(p["forecast"].to_numpy(), model.cuts)
    p["price"] = bar_closes(day, CONFIG.bar_seconds).reindex(p.index)
    w = p.loc["2026-10-14 11:45":"2026-10-14 13:00"].copy()
    w.index = w.index + pd.Timedelta(seconds=CONFIG.bar_seconds)  # stamp at the bar's close

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 4.6), sharex=True)
    a1.plot(w.index, w["price"], lw=0.8)
    a1.set_ylabel("BTC price")
    spikes = w[w["level"] == Level.SPIKE]
    a1.plot(spikes.index, spikes["price"], "o", ms=2.5, color="tab:red", label="spike detector")
    a1.legend(loc="lower left")
    a2.plot(w.index, w["forecast"], lw=1.4, label="forecast, next 5 min")
    a2.plot(w.index, np.exp(w["y"]), lw=0.8, color="grey", label="realized, next 5 min")
    a2.set_yscale("log")
    a2.set_yticks([0.5, 1, 2, 5, 10], ["0.5x", "1x", "2x", "5x", "10x"])
    a2.set_ylabel("vol / normal")
    a2.legend(loc="upper left")
    run = (w["regime"] != w["regime"].shift()).cumsum()
    for _, seg in w.groupby(run):
        rg = Regime(int(seg["regime"].iloc[0]))
        if rg in REGIME_FILL:
            color, alpha = REGIME_FILL[rg]
            for ax in (a1, a2):
                ax.axvspan(seg.index[0], seg.index[-1], color=color, alpha=alpha, lw=0)
    a1.set_title("BTC-PERPETUAL, 2026-10-14 (UTC), model fitted on September only")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(OUT / "example.png")


def after_spike() -> None:
    """Spike detector: realized vol after a new alert, both periods."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    for ax, inst in zip(axes, INSTRUMENTS, strict=True):
        for label, (start, end) in PERIODS.items():
            days = (pd.Timestamp(end) - pd.Timestamp(start)).days
            p = run_study(inst, days=days, end=end).profile
            ax.fill_between(p.index, p["ci low"], p["ci high"], alpha=0.2, lw=0)
            ax.plot(
                p.index,
                p["after episode start"],
                marker="o",
                ms=3,
                label=f"after an alert, {label}",
            )
        ax.plot(p.index, p["any time"], marker="o", ms=3, color="grey", label="at any time")
        ax.axhline(1, color="black", lw=0.6)
        ax.set_xscale("log")
        ax.set_xticks(p.index, [str(m) for m in p.index])
        ax.set_xlabel("next ... minutes")
        ax.set_title(inst)
        ax.legend(loc="upper right", fontsize=7)
    axes[0].set_ylabel("realized vol / normal vol (median)")
    fig.tight_layout()
    fig.savefig(OUT / "after_alert.png")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    regimes()
    example()
    after_spike()


if __name__ == "__main__":
    main()
