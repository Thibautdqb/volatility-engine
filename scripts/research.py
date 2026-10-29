"""Reproduce every modelling decision in the README.

python scripts/research.py            (after: volengine data sync --start 2026-04-28)

1. Input selection. Candidate inputs are compared by walk-forward R2 on May-July only;
   an input group is kept if it raises that R2 by at least 0.002 on average over BTC
   and ETH. August-October is then used once, to confirm.
2. Training window: longest window of 28, 56 or 90 days, on common test weeks.
3. Transfer: the model of one asset applied to the other, and one model for both.
4. Direction: rank correlation of candidate signals with the next 1, 5 and 15 minutes'
   return, in each half of the sample.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from volengine.data.bars import load_bars
from volengine.detector import DetectorConfig
from volengine.evaluation import WalkForwardConfig, walk_forward
from volengine.features import compute, nbars

START, END = "2026-04-28", "2026-10-28"
SPLIT = pd.Timestamp("2026-08-01", tz="UTC")
ASSETS = ("BTC-PERPETUAL", "ETH-PERPETUAL")
CONFIG = DetectorConfig()
BASE = ["x1", "x5", "x30", "season", "trades", "absmove"]
CANDIDATES = {
    "HAR (x1, x5, x30)": ["x1", "x5", "x30"],
    "+ season": ["x1", "x5", "x30", "season"],
    "+ trades, absmove (base)": BASE,
    "base + x4h": [*BASE, "x4h"],
    "base + x1d": [*BASE, "x1d"],
    "base + weekend": [*BASE, "weekend"],
    "base + imbalance": [*BASE, "imbalance"],
    "base + liq": [*BASE, "liq"],
    "base + x4h, x1d, weekend (chosen)": [*BASE, "x4h", "x1d", "weekend"],
    "all": [*BASE, "x4h", "x1d", "weekend", "imbalance", "liq"],
}


def r2(w: pd.DataFrame) -> float:
    y = w["y"].to_numpy()
    return 1 - np.sum((y - w["forecast"]) ** 2) / np.sum((y - w["clim_y"]) ** 2)


def show(title: str, df: pd.DataFrame) -> None:
    print(f"\n{title}\n")
    print(df.to_string(float_format=lambda x: f"{x:.4f}"))


def main() -> None:
    pd.set_option("display.width", 200)
    bars = {a: load_bars(a, START, END) for a in ASSETS}
    feats = {a: compute(b, CONFIG) for a, b in bars.items()}

    rows = {}
    for name, cols in CANDIDATES.items():
        for a in ASSETS:
            w = walk_forward(feats[a], CONFIG, cols).windows
            rows[(name, a[:3], "select May-Jul")] = r2(w[w.index < SPLIT])
            rows[(name, a[:3], "confirm Aug-Oct")] = r2(w[w.index >= SPLIT])
    sel = pd.Series(rows).unstack([1, 2])
    sel[("select, mean", "")] = sel.xs("select May-Jul", axis=1, level=1).mean(axis=1)
    show("1. Input selection: walk-forward R2 of log(vol next 5m / normal)", sel)

    common = pd.Timestamp("2026-07-29", tz="UTC")  # first test week of the 90-day window
    rows = {}
    for days in (28, 56, 90):
        wf = WalkForwardConfig(max_train_days=days, min_train_days=min(days, 28))
        for a in ASSETS:
            w = walk_forward(feats[a], CONFIG, wf=wf).windows
            rows[(f"up to {days} days", a[:3])] = r2(w[w.index >= common])
    show("2. Training window (test weeks from 2026-07-29)", pd.Series(rows).unstack())

    rows = {}
    for a in ASSETS:
        other = feats[ASSETS[1] if a == ASSETS[0] else ASSETS[0]]
        rows[(a[:3], "own history")] = r2(walk_forward(feats[a], CONFIG).windows)
        rows[(a[:3], "other asset only")] = r2(
            walk_forward(feats[a], CONFIG, train_with=(other,), own_history=False).windows
        )
        rows[(a[:3], "both assets")] = r2(
            walk_forward(feats[a], CONFIG, train_with=(other,)).windows
        )
    show("3. Transfer across assets (walk-forward R2)", pd.Series(rows).unstack())

    rows = []
    for a in ASSETS:
        b = bars[a]
        r = np.log(b["close"]).diff()
        k1 = nbars(60, CONFIG.bar_seconds)
        vol, buy = b["volume"].rolling(k1).sum(), b["buy_volume"].rolling(k1).sum()
        signals = {
            "order-flow imbalance, 1m": ((2 * buy - vol) / vol.where(vol > 0)).fillna(0),
            "last move, 1m": r.rolling(k1).sum(),
            "perp minus index, change 1m": np.log(b["close"] / b["index"]).diff(k1),
            "index lead, 30s": np.log(b["index"]).diff(3) - r.rolling(3).sum(),
        }
        step = nbars(300, CONFIG.bar_seconds)
        for minutes in (1, 5, 15):
            k = nbars(60 * minutes, CONFIG.bar_seconds)
            fwd = r[::-1].rolling(k).sum()[::-1].shift(-1)
            for name, s in signals.items():
                frame = pd.DataFrame({"s": s, "f": fwd}).iloc[::step].dropna()
                for half, m in (
                    ("May-Jul", frame.index < SPLIT),
                    ("Aug-Oct", frame.index >= SPLIT),
                ):
                    rows.append(
                        {
                            "asset": a[:3],
                            "signal": name,
                            "horizon": f"{minutes}m",
                            "half": half,
                            "IC": stats.spearmanr(frame["s"][m], frame["f"][m]).statistic,
                        }
                    )
    ic = pd.DataFrame(rows).pivot_table(
        index=["asset", "signal", "horizon"], columns="half", values="IC"
    )
    ic["same sign"] = np.sign(ic["May-Jul"]) == np.sign(ic["Aug-Oct"])
    show("4. Direction: Spearman IC with the forward return, non-overlapping 5m grid", ic)
    print(f"\nlargest |IC|: {ic[['May-Jul', 'Aug-Oct']].abs().max().max():.4f}")


if __name__ == "__main__":
    main()
