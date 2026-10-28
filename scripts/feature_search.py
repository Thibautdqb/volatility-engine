"""Which past-only quantities predict the next five minutes? Reproduces the README table.

python scripts/feature_search.py

Two questions, each answered out of sample in both directions (fit on 2026-09-14 to
2026-10-13 and test on 2026-10-14 to 2026-10-27, then the reverse):

1. Volatility: out-of-sample R2 of log(vol next 5m / normal) for nested OLS feature sets.
2. Direction: Spearman correlation between candidate signals and the return over the
   next 1, 5 and 15 minutes. A usable signal needs the same sign in both periods.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from volengine.detector import DetectorConfig, SpikeDetector
from volengine.study import load_trades

CONFIG = DetectorConfig()
BAR_MS = round(CONFIG.bar_seconds * 1000)
H = 30  # five minutes of 10-second bars
PERIODS = {"Sep": ("2026-09-14", "2026-10-14"), "Oct": ("2026-10-14", "2026-10-28")}
FEATURE_SETS = {
    "HAR (1m, 5m, 30m vol)": ["x1", "x5", "x30"],
    "+ time of day": ["x1", "x5", "x30", "season"],
    "+ trade count": ["x1", "x5", "x30", "season", "trades"],
    "+ size of the last move": ["x1", "x5", "x30", "season", "trades", "absmove"],
    "+ liquidations": ["x1", "x5", "x30", "season", "trades", "absmove", "liq"],
    "+ order-flow imbalance, basis": [
        "x1",
        "x5",
        "x30",
        "season",
        "trades",
        "absmove",
        "liq",
        "absofi",
        "dbasis",
    ],
}
SIGNALS = ["ofi", "move", "idxlead", "dbasis", "basis"]


def _sum_past(x: np.ndarray, k: int) -> np.ndarray:
    c = np.concatenate([[0.0], np.cumsum(x)])
    out = np.full(x.size, np.nan)
    out[k - 1 :] = c[k:] - c[:-k]
    return out


def _sum_next(x: np.ndarray, k: int) -> np.ndarray:
    c = np.concatenate([[0.0], np.cumsum(x)])
    out = np.full(x.size, np.nan)
    t = np.arange(x.size - k)
    out[t] = c[t + 1 + k] - c[t + 1]
    return out


def bars(instrument: str, start: str, end: str) -> pd.DataFrame:
    t = load_trades(instrument, start, end)
    amount = t["amount"].to_numpy(float)
    g = (
        pd.DataFrame(
            {
                "b": t["timestamp"].to_numpy() // BAR_MS,
                "close": t["price"].to_numpy(float),
                "idx": t["index_price"].to_numpy(float),
                "n": 1,
                "vol": amount,
                "buy": np.where(t["direction"].to_numpy() == "buy", amount, 0.0),
                "liq": np.where(t["liquidation"].astype(str).str.len() > 0, amount, 0.0),
            }
        )
        .groupby("b")
        .agg(
            close=("close", "last"),
            idx=("idx", "last"),
            n=("n", "sum"),
            vol=("vol", "sum"),
            buy=("buy", "sum"),
            liq=("liq", "sum"),
        )
    )
    g = g.reindex(np.arange(g.index[0], g.index[-1] + 1))
    g[["close", "idx"]] = g[["close", "idx"]].ffill()
    g = g.fillna(0.0)
    g.index = pd.to_datetime(g.index * BAR_MS, unit="ms", utc=True)
    return g


def features(g: pd.DataFrame) -> pd.DataFrame:
    r = np.r_[0.0, np.diff(np.log(g["close"].to_numpy()))]
    ri = np.r_[0.0, np.diff(np.log(g["idx"].to_numpy()))]
    warm = CONFIG.warmup_bars
    det = SpikeDetector(CONFIG, float(np.mean(r[1:warm] ** 2)))
    slow = np.empty(r.size)
    for i, x in enumerate(r):
        slow[i] = det.update(x).slow_var
    r2 = r * r
    f = pd.DataFrame(index=g.index)
    for name, k in (("x1", 6), ("x5", 30), ("x30", 180)):
        f[name] = 0.5 * np.log(np.maximum(_sum_past(r2, k) / (k * slow), 1e-4))
    n, vol, buy = g["n"].to_numpy(), g["vol"].to_numpy(), g["buy"].to_numpy()
    f["trades"] = np.log((_sum_past(n, 6) + 1) / (_sum_past(n, 180) / 30 + 1))
    tot = _sum_past(vol, 6)
    f["ofi"] = np.where(tot > 0, (2 * _sum_past(buy, 6) - tot) / np.maximum(tot, 1), 0.0)
    f["absofi"] = f["ofi"].abs()
    f["liq"] = np.log1p(_sum_past(g["liq"].to_numpy(), 30))
    f["basis"] = np.log(g["close"] / g["idx"]).to_numpy() * 1e4
    f["dbasis"] = f["basis"] - f["basis"].shift(6)
    f["idxlead"] = (_sum_past(ri, 3) - _sum_past(r, 3)) * 1e4  # index moved, perp not yet
    f["move"] = _sum_past(r, 6) / np.sqrt(6 * slow)
    f["absmove"] = f["move"].abs()
    f["r2"] = r2 / slow
    fwd = _sum_next(r2, H) / H
    f["y"] = 0.5 * np.log(np.maximum(fwd / slow, 1e-4))
    for name, k in (("ret1", 6), ("ret5", 30), ("ret15", 90)):
        f[name] = _sum_next(r, k) * 1e4
    f = f.iloc[warm:]
    flat = 0.5 * np.log(1e-4) + 1e-9
    return f[(f["y"] > flat) & (f["x5"] > flat)]  # drop market closures, as in the package


def add_season(f: pd.DataFrame, train: pd.DataFrame) -> pd.DataFrame:
    """Mean squared return per 5-minute slot of the day, estimated on training data only."""
    slot = (train.index.hour * 60 + train.index.minute) // 5
    s = pd.Series(train["r2"].to_numpy()).groupby(slot).mean()
    s = np.log((s / s.mean()).rolling(3, center=True, min_periods=1).mean())
    nxt = f.index + pd.Timedelta(seconds=150)  # middle of the next five minutes
    f = f.copy()
    f["season"] = 0.5 * s.reindex((nxt.hour * 60 + nxt.minute) // 5).to_numpy()
    return f


def r2_oos(train: pd.DataFrame, test: pd.DataFrame, cols: list[str]) -> float:
    A = np.column_stack([np.ones(len(train)), train[cols]])
    beta = np.linalg.lstsq(A, train["y"], rcond=None)[0]
    pred = np.column_stack([np.ones(len(test)), test[cols]]) @ beta
    y = test["y"].to_numpy()
    return 1 - np.sum((y - pred) ** 2) / np.sum((y - train["y"].mean()) ** 2)


def main() -> None:
    pd.set_option("display.width", 160)
    vol_rows, dir_rows = [], []
    for inst in ("BTC-PERPETUAL", "ETH-PERPETUAL"):
        f = {p: features(bars(inst, *span)) for p, span in PERIODS.items()}
        for train_p, test_p in (("Sep", "Oct"), ("Oct", "Sep")):
            tr = add_season(f[train_p], f[train_p]).iloc[::H].dropna()
            te = add_season(f[test_p], f[train_p]).iloc[::H].dropna()
            label = f"{inst.split('-')[0]} test {test_p}"
            row = {"": label}
            yte = te["y"].to_numpy()
            row["last 5 minutes"] = 1 - np.sum((yte - te["x5"]) ** 2) / np.sum(
                (yte - tr["y"].mean()) ** 2
            )
            row.update({name: r2_oos(tr, te, cols) for name, cols in FEATURE_SETS.items()})
            vol_rows.append(row)
        for p, data in f.items():
            g = data.iloc[::H].dropna()
            for sig in SIGNALS:
                for tgt in ("ret1", "ret5", "ret15"):
                    dir_rows.append(
                        {
                            "instrument": inst.split("-")[0],
                            "signal": sig,
                            "horizon": tgt,
                            "period": p,
                            "IC": stats.spearmanr(g[sig], g[tgt]).statistic,
                        }
                    )
    print("Out-of-sample R2 of log(vol next 5m / normal)\n")
    print(pd.DataFrame(vol_rows).set_index("").T.to_string(float_format=lambda x: f"{x:+.3f}"))
    d = pd.DataFrame(dir_rows).pivot_table(
        index=["instrument", "signal", "horizon"], columns="period", values="IC"
    )
    d["same sign"] = np.sign(d["Sep"]) == np.sign(d["Oct"])
    print("\nSpearman correlation with the forward return (non-overlapping 5-minute grid)\n")
    print(d.to_string(float_format=lambda x: f"{x:+.3f}"))
    print(f"\nlargest |IC|: {d[['Sep', 'Oct']].abs().max().max():.3f}")


if __name__ == "__main__":
    main()
