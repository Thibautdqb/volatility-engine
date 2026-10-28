"""Regenerate the figures in docs/img from the Deribit trade tape.

python scripts/make_figures.py --end 2026-10-28 --days 14
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from volengine.detector import Level
from volengine.study import run_study

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


def after_episodes(studies) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    for ax, s in zip(axes, studies, strict=True):
        p = s.profile
        ax.fill_between(p.index, p["ci low"], p["ci high"], alpha=0.25, lw=0)
        ax.plot(p.index, p["after episode start"], marker="o", ms=3, label="after a new alert")
        ax.plot(p.index, p["any time"], marker="o", ms=3, color="grey", label="at any time")
        ax.axhline(1, color="black", lw=0.6)
        ax.set_xscale("log")
        ax.set_xticks(p.index, [str(m) for m in p.index])
        ax.set_xlabel("next ... minutes")
        ax.set_title(f"{s.instrument}, {len(s.episodes)} alerts")
    axes[0].set_ylabel("realized vol / normal vol (median)")
    axes[0].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "after_alert.png")


def example(study, hours: float = 1.5) -> None:
    sig = study.signals
    t0 = sig.index[(sig["alert"] & (sig["level"] == Level.SPIKE)).to_numpy()]
    # The spike alert with the largest ratio, shown with the hour before it.
    peak = sig.loc[t0, "ratio"].idxmax()
    win = sig.loc[peak - pd.Timedelta(hours=hours) : peak + pd.Timedelta(minutes=30)]
    price = win["return"].cumsum()
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 4.2), sharex=True, height_ratios=[1, 1])
    a1.plot(win.index, 100 * price, lw=0.8)
    a1.set_ylabel("price change (%)")
    a2.plot(win.index, win["ratio"], lw=0.8)
    c = study.config
    for y, name in [(c.elevated, "elevated"), (c.spike, "spike")]:
        a2.axhline(y, ls=":", lw=0.8, color="black")
        a2.text(win.index[0], y, f" {name}", va="bottom", fontsize=8)
    a2.set_ylabel("vol 1m / normal")
    run = (win["level"] != win["level"].shift()).cumsum()
    for _, seg in win.groupby(run):
        lv = int(seg["level"].iloc[0])
        if lv:
            for ax in (a1, a2):
                ax.axvspan(
                    seg.index[0],
                    seg.index[-1],
                    color="tab:orange" if lv == 1 else "tab:red",
                    alpha=0.15 if lv == 1 else 0.25,
                    lw=0,
                )
    a1.set_title(f"{study.instrument}, {peak:%Y-%m-%d} (UTC); orange = elevated, red = spike")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(OUT / "example.png")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--end", default="2026-10-28")
    p.add_argument("--days", type=int, default=14)
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    studies = [run_study(i, days=args.days, end=args.end) for i in INSTRUMENTS]
    after_episodes(studies)
    example(studies[0])


if __name__ == "__main__":
    main()
