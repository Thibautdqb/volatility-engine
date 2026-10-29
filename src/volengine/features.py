"""Forecast inputs, computed one way for history (batch) and live (streaming).

Every input is known at the close of a bar. Volatilities are expressed relative to the
*normal* level, the detector's 30-minute EWMA variance at that bar, which makes the
inputs unit-free and comparable across assets and calendar periods.

=========  ================================================================
name       definition
=========  ================================================================
x1 .. x1d  0.5 log(realized variance over the last 1 m, 5 m, 30 m, 4 h, 1 d
           / normal variance), floored at 1e-4
trades     log((trades in the last minute + 1)
           / (average trades per minute over 30 minutes + 1))
absmove    |price change over the last minute| / normal standard deviation
imbalance  |buy volume - sell volume| / volume, last minute (0 if no volume)
liq        log(1 + liquidated notional over the last 5 minutes / 10,000 USD)
weekend    1 on Saturday and Sunday (UTC) for the forecast window, else 0
season     time-of-day term, looked up in the model (see ``forecast``)
=========  ================================================================

The target ``y`` is 0.5 log(realized variance over the next five minutes / normal
variance), also floored. :func:`compute` and :class:`StreamingFeatures` are tested to
agree to floating-point precision.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np
import pandas as pd

from volengine.detector import DetectorConfig, run_detector

HORIZON_SECONDS = 300
VOL_WINDOWS = {"x1": 60, "x5": 300, "x30": 1800, "x4h": 14_400, "x1d": 86_400}
FLOOR = 1e-4
FLAT = 0.5 * math.log(FLOOR)  # value of a log vol ratio when nothing traded
ALL = (*VOL_WINDOWS, "trades", "absmove", "imbalance", "liq", "weekend")


def nbars(seconds: float, bar_seconds: float) -> int:
    return max(1, round(seconds / bar_seconds))


def lookback_seconds(features) -> float:
    """History a feature set needs before its first valid value."""
    vols = [VOL_WINDOWS[f] for f in features if f in VOL_WINDOWS]
    return max([1800.0, *vols])


def _log_ratio(var_ratio):
    return 0.5 * np.log(np.maximum(var_ratio, FLOOR))


def _rolling(x: np.ndarray, k: int) -> np.ndarray:
    c = np.concatenate([[0.0], np.cumsum(x)])
    out = np.full(x.size, np.nan)
    out[k - 1 :] = c[k:] - c[:-k]
    return out


def _weekend(t_seconds: np.ndarray) -> np.ndarray:
    day = (np.asarray(t_seconds) // 86400 + 3) % 7  # Monday = 0; 1970-01-01 was a Thursday
    return (day >= 5).astype(float)


def compute(bars: pd.DataFrame, config: DetectorConfig) -> pd.DataFrame:
    """Inputs and target for every bar after the detector's warm-up.

    ``bars`` comes from :func:`volengine.data.bars.load_bars`. Besides the inputs, the
    output keeps ``y`` (NaN at the end), ``t_mid`` (middle of the forecast window, epoch
    seconds), ``r2n`` (squared return in normal units, for the time-of-day profile),
    and the detector's ``slow_vol``, ``ratio`` and ``level``.
    """
    r = np.log(bars["close"]).diff().iloc[1:]
    signals = run_detector(r, config)
    idx = signals.index
    # Look-backs run over every bar, including the detector's warm-up, as they do live.
    full = bars.iloc[1:]
    keep = full.index.isin(idx)
    allr = r.to_numpy()

    def past(x: np.ndarray, k: int) -> np.ndarray:
        return _rolling(x, k)[keep]

    ret = signals["return"].to_numpy()
    r2 = ret * ret
    s = signals["slow_vol"].to_numpy() ** 2 / config.bars_per_year
    k = {f: nbars(sec, config.bar_seconds) for f, sec in VOL_WINDOWS.items()}
    k1, k5, k30 = k["x1"], k["x5"], k["x30"]

    out = pd.DataFrame(index=idx)
    for name, kb in k.items():
        out[name] = _log_ratio(past(allr * allr, kb) / (kb * s))
    n = full["n"].to_numpy(float)
    out["trades"] = np.log((past(n, k1) + 1) / (past(n, k30) / (k30 // k1) + 1))
    out["absmove"] = np.abs(past(allr, k1)) / np.sqrt(k1 * s)
    vol = past(full["volume"].to_numpy(float), k1)
    buy = past(full["buy_volume"].to_numpy(float), k1)
    out["imbalance"] = np.where(vol > 0, np.abs(2 * buy - vol) / np.where(vol > 0, vol, 1), 0.0)
    out["liq"] = np.log1p(past(full["liq_volume"].to_numpy(float), k5) / 1e4)
    b = bars.reindex(idx)
    epoch = pd.Timestamp(0, tz="UTC")
    bar_end = np.asarray((idx - epoch) / pd.Timedelta(seconds=1)) + config.bar_seconds
    out["t_mid"] = bar_end + HORIZON_SECONDS / 2
    out["weekend"] = _weekend(out["t_mid"].to_numpy())
    out["r2n"] = r2 / s

    h = nbars(HORIZON_SECONDS, config.bar_seconds)
    c = np.concatenate([[0.0], np.cumsum(r2)])
    fwd = np.full(r2.size, np.nan)
    t = np.arange(r2.size - h)
    fwd[t] = (c[t + 1 + h] - c[t + 1]) / h
    out["y"] = np.where(np.isnan(fwd), np.nan, _log_ratio(fwd / s))
    out["slow_vol"] = signals["slow_vol"].to_numpy()
    out["ratio"] = signals["ratio"].to_numpy()
    out["level"] = signals["level"].to_numpy()
    out["close"] = b["close"].to_numpy()
    return out


def windows(features: pd.DataFrame, config: DetectorConfig) -> pd.DataFrame:
    """One row every five minutes, so that targets do not overlap.

    Rows are dropped when an input is not yet defined, or when the last or the next five
    minutes had no price change at all: those are market closures (exchange
    maintenance), not a forecastable state.
    """
    h = nbars(HORIZON_SECONDS, config.bar_seconds)
    w = features.iloc[::h].dropna(subset=["y", "x5"])
    return w[(w["y"] > FLAT + 1e-9) & (w["x5"] > FLAT + 1e-9)]


class StreamingFeatures:
    """Running sums over the look-backs, O(1) per bar; same numbers as :func:`compute`."""

    def __init__(self, config: DetectorConfig, features=ALL):
        self.config = config
        bs = config.bar_seconds
        self.k = {f: nbars(sec, bs) for f, sec in VOL_WINDOWS.items()}
        self.vol_names = [f for f in VOL_WINDOWS if f in features or f in ("x1", "x5", "x30")]
        self.size = max(self.k[f] for f in self.vol_names)
        self._r: deque[float] = deque(maxlen=self.size)
        self._n: deque[float] = deque(maxlen=self.k["x30"])
        self._vol: deque[float] = deque(maxlen=self.k["x1"])
        self._buy: deque[float] = deque(maxlen=self.k["x1"])
        self._liq: deque[float] = deque(maxlen=self.k["x5"])
        self._count = 0
        self._resum()

    def _resum(self) -> None:
        r = list(self._r)
        self._r2 = {f: sum(x * x for x in r[-self.k[f] :]) for f in self.vol_names}
        self._r1 = sum(r[-self.k["x1"] :])
        n = list(self._n)
        self._n1, self._n30 = sum(n[-self.k["x1"] :]), sum(n)
        self._v1, self._b1, self._l5 = sum(self._vol), sum(self._buy), sum(self._liq)

    @staticmethod
    def _push(dq: deque, x: float) -> float:
        """Append; return the element that left the window (0 if none)."""
        old = dq[0] if len(dq) == dq.maxlen else 0.0
        dq.append(x)
        return old

    def update(self, r: float, n: int, volume: float, buy: float, liq: float) -> None:
        size = len(self._r)
        for f in self.vol_names:
            kb = self.k[f]
            self._r2[f] += r * r - (self._r[-kb] ** 2 if size >= kb else 0.0)
        k1 = self.k["x1"]
        self._r1 += r - (self._r[-k1] if size >= k1 else 0.0)
        self._r.append(r)
        nsize = len(self._n)
        self._n1 += n - (self._n[-k1] if nsize >= k1 else 0.0)
        self._n30 += n - self._push(self._n, float(n))
        self._v1 += volume - self._push(self._vol, volume)
        self._b1 += buy - self._push(self._buy, buy)
        self._l5 += liq - self._push(self._liq, liq)
        self._count += 1
        if self._count % 1000 == 0:  # clear floating-point drift of the running sums
            self._resum()

    def ready(self, features) -> bool:
        need = max([self.k["x30"], *(self.k[f] for f in features if f in self.k)])
        return len(self._r) >= need

    def values(self, slow_var: float, bar_end_seconds: float) -> dict[str, float]:
        k1, k30 = self.k["x1"], self.k["x30"]
        out = {f: float(_log_ratio(self._r2[f] / (self.k[f] * slow_var))) for f in self.vol_names}
        out["trades"] = math.log((self._n1 + 1) / (self._n30 / (k30 // k1) + 1))
        out["absmove"] = abs(self._r1) / math.sqrt(k1 * slow_var)
        out["imbalance"] = abs(2 * self._b1 - self._v1) / self._v1 if self._v1 > 0 else 0.0
        out["liq"] = math.log1p(self._l5 / 1e4)
        t_mid = bar_end_seconds + HORIZON_SECONDS / 2
        out["weekend"] = float(_weekend(np.array([t_mid]))[0])
        out["t_mid"] = t_mid
        return out
