"""Regenerate the figures in docs/img from the Deribit trade tape.

python scripts/make_figures.py

Samples: 2026-10-14 to 2026-10-27 (the period the README was first written on) and
2026-09-14 to 2026-10-13 (held out, analyzed afterwards with the same parameters).
"""

from __future__ import annotations

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


def after_episodes(samples: dict[str, list]) -> None:
    """One panel per instrument, one line per sample, with the unconditional median."""
    fig, axes = plt.subplots(1, len(INSTRUMENTS), figsize=(9, 3.4), sharey=True)
    for ax, inst in zip(axes, INSTRUMENTS, strict=True):
        for label, studies in samples.items():
            s = next(x for x in studies if x.instrument == inst)
            p = s.profile
            ax.fill_between(p.index, p["ci low"], p["ci high"], alpha=0.2, lw=0)
            ax.plot(
                p.index,
                p["after episode start"],
                marker="o",
                ms=3,
                label=f"after an alert, {label} ({len(s.episodes)})",
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
    OUT.mkdir(parents=True, exist_ok=True)
    samples = {
        "Oct 14-27": [run_study(i, days=14, end="2026-10-28") for i in INSTRUMENTS],
        "Sep 14-Oct 13": [run_study(i, days=30, end="2026-10-14") for i in INSTRUMENTS],
    }
    after_episodes(samples)
    example(samples["Oct 14-27"][0])


if __name__ == "__main__":
    main()
