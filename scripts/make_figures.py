"""Run the walk-forward backtest on BTC and ETH; write the figures and the full reports.

python scripts/make_figures.py        (after: volengine data sync --start 2026-04-28)

Outputs: docs/img/*.png and docs/results/<instrument>.txt (the text of
``volengine backtest <instrument> --end 2026-10-28``). Everything shown is out of
sample.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from volengine.data.bars import load_bars
from volengine.detector import DetectorConfig, Level
from volengine.evaluation import Report, evaluate
from volengine.features import compute
from volengine.forecast import Regime, StressAlarm

ROOT = Path(__file__).resolve().parents[1]
IMG, RESULTS = ROOT / "docs" / "img", ROOT / "docs" / "results"
START, END = "2026-04-28", "2026-10-28"
ASSETS = ("BTC-PERPETUAL", "ETH-PERPETUAL")
CONFIG = DetectorConfig()
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


def weekly(reports: dict[str, Report]) -> None:
    fig, ax = plt.subplots(figsize=(9, 3))
    width = 2.6
    for k, (inst, rep) in enumerate(reports.items()):
        f = rep.wf.folds
        x = f.index + pd.Timedelta(days=(k - 0.5) * width)
        ax.bar(x, f["R2"], width=width, label=f"{inst.split('-')[0]}: model")
        ax.plot(x, f["R2 last 5m"], "_", ms=8, mew=1.5, color=f"C{k}", alpha=0.7)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_ylabel("out-of-sample R2, one bar per week")
    ax.set_title("Weekly R2 of log(vol next 5m / normal); dashes: 'last 5 minutes' as forecast")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(IMG / "weekly.png")


def calibration(reports: dict[str, Report]) -> None:
    fig, ax = plt.subplots(figsize=(4.4, 4))
    for k, (inst, rep) in enumerate(reports.items()):
        t = rep.reliability
        err = [t["observed"] - t["ci low"], t["ci high"] - t["observed"]]
        ax.errorbar(
            t["mean forecast"], t["observed"], yerr=err, fmt="o-", ms=4, color=f"C{k}",
            label=inst.split("-")[0],
        )  # fmt: skip
    ax.plot([0.005, 0.4], [0.005, 0.4], color="grey", lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("forecast P(violent next 5 min)")
    ax.set_ylabel("observed frequency")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(IMG / "calibration.png")


def anticipation(reports: dict[str, Report]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    for ax, (inst, rep) in zip(axes, reports.items(), strict=True):
        a = rep.anticipation
        ax.plot(a.index, 100 * a["forecast STRESSED"], "o-", ms=4, label="forecast: STRESSED")
        ax.plot(a.index, 100 * a["spike detector on"], "o-", ms=4, label="spike detector on")
        ax.axhline(100 * a.attrs["time on: forecast STRESSED"], color="C0", ls=":", lw=0.9)
        ax.axhline(100 * a.attrs["time on: spike detector"], color="C1", ls=":", lw=0.9)
        ax.set_xlabel("minutes before the violent window starts")
        ax.set_title(f"{inst}, {int(a['episodes'].iloc[0])} episodes")
        ax.invert_xaxis()
    axes[0].set_ylabel("% of episodes already signalled")
    axes[0].legend(loc="upper left")
    fig.text(0.5, -0.02, "dotted: share of all time the signal is on", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(IMG / "anticipation.png", bbox_inches="tight")


def regimes(reports: dict[str, Report]) -> None:
    fig, ax = plt.subplots(figsize=(6, 3.2))
    names = [r.name for r in Regime]
    for k, (inst, rep) in enumerate(reports.items()):
        t = rep.regimes.loc[names]
        x = np.arange(len(names)) + (k - 0.5) * 0.36
        err = [100 * (t["P(violent)"] - t["ci low"]), 100 * (t["ci high"] - t["P(violent)"])]
        ax.bar(x, 100 * t["P(violent)"], width=0.34, label=inst.split("-")[0])
        ax.errorbar(x, 100 * t["P(violent)"], yerr=err, fmt="none", color="black", lw=0.8)
    ax.set_xticks(range(len(names)), names)
    ax.set_ylabel("P(next 5 min vol >= 2x normal), %")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(IMG / "regimes.png")


def example(rep: Report, start: str, end: str) -> None:
    """Price with the alerts the alarm sent, and the stress score, out of sample."""
    bars = rep.wf.bars
    alarm = StressAlarm()
    epoch = pd.Timestamp(0, tz="UTC")
    t = np.asarray((bars.index - epoch) / pd.Timedelta(seconds=1))
    spike = (bars["level"] >= Level.SPIKE).to_numpy()
    fired = [alarm.update(s, x, k) for s, x, k in zip(bars["stress"], t, spike, strict=True)]
    bars = bars.assign(alert=fired)
    b = bars.loc[start:end].copy()
    b.index = b.index + pd.Timedelta(seconds=CONFIG.bar_seconds)  # stamp at the bar's close

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 4.6), sharex=True, height_ratios=[1.2, 1])
    a1.plot(b.index, b["close"], lw=0.8)
    a1.set_ylabel("price")
    a2.plot(b.index, b["stress"], lw=1.0, color="tab:purple")
    a2.axhline(99, color="tab:red", ls=":", lw=0.9)
    a2.text(b.index[0], 99, " alert threshold 99", va="bottom", fontsize=8, color="tab:red")
    a2.axhline(75, color="grey", ls=":", lw=0.7)
    a2.text(b.index[0], 75, " re-arm below 75", va="bottom", fontsize=8, color="grey")
    a2.set_ylim(0, 102)
    a2.set_ylabel("stress (0-100)")
    for when, reason in b["alert"].dropna().items():
        for ax in (a1, a2):
            ax.axvline(when, color="tab:red", lw=1.4)
        a1.annotate(
            f"alert ({reason}) {when:%H:%M:%S}",
            (when, b["close"].max()),
            xytext=(4, -10),
            textcoords="offset points",
            fontsize=8,
            color="tab:red",
        )
    a1.set_title(f"{rep.wf.instrument}, {pd.Timestamp(start):%Y-%m-%d} UTC, out of sample")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(IMG / "example.png")


def main() -> None:
    IMG.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    reports = {}
    for inst in ASSETS:
        feats = compute(load_bars(inst, START, END), CONFIG)
        reports[inst] = rep = evaluate(feats, CONFIG, instrument=inst)
        (RESULTS / f"{inst}.txt").write_text(rep.summary() + "\n")
        print(rep.summary(), "\n")
    weekly(reports)
    calibration(reports)
    anticipation(reports)
    regimes(reports)
    example(reports["BTC-PERPETUAL"], "2026-10-14 11:45", "2026-10-14 13:00")


if __name__ == "__main__":
    main()
