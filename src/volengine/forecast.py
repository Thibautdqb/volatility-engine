"""Five-minute volatility forecast and market-regime classification.

The forecast is a linear model of the log volatility over the next five minutes,
relative to the normal (30-minute EWMA) level. Its terms, all known at the time of
the forecast:

* ``x1, x5, x30``: log realized volatility over the last 1, 5 and 30 minutes, relative
  to normal. This is the heterogeneous autoregression (HAR) of Corsi (2009), moved to
  intraday horizons.
* ``season``: the usual volatility of the coming five minutes of the day (US equity
  open, scheduled data releases), estimated on the training days.
* ``trades``: number of trades over the last minute against its 30-minute average.
* ``absmove``: size of the price change over the last minute, in normal standard
  deviations.

Order-flow imbalance, liquidations and the perpetual-index basis were tested and add
nothing out of sample (``scripts/feature_search.py``).

The forecast is turned into a regime by its rank among the forecasts made on the
training data: the lowest 25% are CALM, the middle 50% NORMAL, the next 20% ACTIVE and
the top 5% STRESSED. The regime rises as soon as a cut point is crossed and falls back
only after one minute below it. Coefficients, time-of-day profile and cut points are
stored together (``volengine fit``).
"""

from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import asdict, dataclass, field
from enum import IntEnum
from importlib import resources

import numpy as np
import pandas as pd

from volengine.detector import DetectorConfig, run_detector

FEATURES = ("x1", "x5", "x30", "season", "trades", "absmove")
WINDOWS_SECONDS = (60, 300, 1800)  # realized-volatility look-backs
HORIZON_SECONDS = 300
SLOT_SECONDS = 300  # time-of-day profile resolution
SLOTS = 86400 // SLOT_SECONDS
REGIME_QUANTILES = (0.25, 0.75, 0.95)
_FLOOR = 1e-4  # variance ratios below this are flat windows (no trade); keeps logs finite


class Regime(IntEnum):
    CALM = 0
    NORMAL = 1
    ACTIVE = 2
    STRESSED = 3


def classify(ratio: float, cuts: tuple[float, float, float]) -> Regime:
    return Regime(int(np.searchsorted(cuts, ratio, side="right")))


class RegimeTracker:
    """Regime with hysteresis: it rises at once, but falls only after ``hold`` bars below.

    Without it, a forecast sitting on a cut point flips the regime every bar.
    """

    def __init__(self, cuts: tuple[float, float, float], hold: int = 6):
        self.cuts = cuts
        self.hold = hold
        self.regime: Regime | None = None
        self._below = 0

    def update(self, ratio: float) -> Regime:
        target = classify(ratio, self.cuts)
        if self.regime is None or target >= self.regime:
            self.regime, self._below = target, 0
        else:
            self._below += 1
            if self._below >= self.hold:
                self.regime, self._below = target, 0
        return self.regime


def track_regimes(ratios: np.ndarray, cuts: tuple[float, float, float]) -> np.ndarray:
    tracker = RegimeTracker(cuts)
    return np.array([int(tracker.update(float(x))) for x in ratios])


@dataclass(frozen=True)
class HARModel:
    intercept: float
    coefs: tuple[float, ...]  # one per FEATURES entry
    season: tuple[float, ...] = field(default=(0.0,) * SLOTS, repr=False)  # per slot of day
    cuts: tuple[float, float, float] = (0.8, 0.95, 1.07)  # forecast ratio between regimes
    instrument: str = ""
    trained_on: str = ""  # "YYYY-MM-DD to YYYY-MM-DD"
    n: int = 0

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Log forward ratio from features of shape (n, len(FEATURES))."""
        return self.intercept + np.atleast_2d(x) @ np.asarray(self.coefs)

    def season_at(self, t_seconds: float | np.ndarray) -> np.ndarray:
        """Time-of-day term for the five minutes centred on ``t_seconds`` (epoch)."""
        slot = (np.asarray(t_seconds) % 86400 // SLOT_SECONDS).astype(int)
        return np.asarray(self.season)[slot]

    def regime(self, ratio: float) -> Regime:
        return classify(ratio, self.cuts)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["coefs"] = dict(zip(FEATURES, self.coefs, strict=True))
        return d

    @classmethod
    def from_dict(cls, d: dict) -> HARModel:
        d = dict(d)
        coefs = d["coefs"]
        d["coefs"] = tuple(coefs[f] for f in FEATURES) if isinstance(coefs, dict) else coefs
        d["season"] = tuple(d["season"])
        d["cuts"] = tuple(d["cuts"])
        return cls(**d)


def default_model(instrument: str) -> HARModel:
    """Coefficients shipped with the package, produced by ``volengine fit``."""
    data = json.loads(resources.files("volengine").joinpath("models.json").read_text())
    # Other instruments get the BTC coefficients: they are close across assets.
    return HARModel.from_dict(data.get(instrument) or data["BTC-PERPETUAL"])


def _bars(seconds: float, bar_seconds: float) -> int:
    return max(1, round(seconds / bar_seconds))


def _log_vol_ratio(var_ratio):
    return 0.5 * np.log(np.maximum(var_ratio, _FLOOR))


# ---------------------------------------------------------------------------
# batch
# ---------------------------------------------------------------------------


def _rolling_sum(x: np.ndarray, k: int) -> np.ndarray:
    c = np.concatenate([[0.0], np.cumsum(x)])
    out = np.full(x.size, np.nan)
    out[k - 1 :] = c[k:] - c[:-k]
    return out


def prepare(returns: pd.Series, counts: pd.Series, config: DetectorConfig) -> pd.DataFrame:
    """Every bar after the detector's warm-up, with forecast inputs and the target.

    ``counts`` is the number of trades per bar, on the same index as ``returns``. All
    inputs are known at the close of the bar; ``y`` (log vol over the next five minutes
    relative to normal) is not, and is NaN at the end of the sample. ``t_mid`` is the
    middle of the forecast window, in epoch seconds, for the time-of-day term.
    """
    signals = run_detector(returns, config)
    r = signals["return"].to_numpy()
    r2 = r * r
    slow_var = signals["slow_vol"].to_numpy() ** 2 / config.bars_per_year
    n = counts.reindex(signals.index).fillna(0).to_numpy(dtype=float)
    k1, k5, k30 = (_bars(s, config.bar_seconds) for s in WINDOWS_SECONDS)
    out = pd.DataFrame(index=signals.index)
    for name, k in (("x1", k1), ("x5", k5), ("x30", k30)):
        out[name] = _log_vol_ratio(_rolling_sum(r2, k) / (k * slow_var))
    out["trades"] = np.log((_rolling_sum(n, k1) + 1) / (_rolling_sum(n, k30) / (k30 // k1) + 1))
    out["absmove"] = np.abs(_rolling_sum(r, k1)) / np.sqrt(k1 * slow_var)
    epoch = pd.Timestamp(0, tz="UTC")
    bar_end = (signals.index - epoch) / pd.Timedelta(seconds=1) + config.bar_seconds
    out["t_mid"] = np.asarray(bar_end) + HORIZON_SECONDS / 2
    out["r2n"] = r2 / slow_var  # squared return in normal units, for the time-of-day profile
    h = _bars(HORIZON_SECONDS, config.bar_seconds)
    c = np.concatenate([[0.0], np.cumsum(r2)])
    fwd = np.full(r.size, np.nan)
    t = np.arange(r.size - h)
    fwd[t] = (c[t + 1 + h] - c[t + 1]) / h
    out["y"] = _log_vol_ratio(fwd / slow_var)
    out.loc[np.isnan(fwd), "y"] = np.nan
    out["level"] = signals["level"].to_numpy()
    out["ratio"] = signals["ratio"].to_numpy()
    out["slow_vol"] = signals["slow_vol"].to_numpy()
    return out


def season_profile(prepared: pd.DataFrame) -> np.ndarray:
    """Log volatility of each 5-minute slot of the day relative to the daily average.

    Smoothed over three neighbouring slots (wrapping around midnight) to limit noise.
    """
    epoch = pd.Timestamp(0, tz="UTC")
    t = np.asarray((prepared.index - epoch) / pd.Timedelta(seconds=1))
    slot = (t % 86400 // SLOT_SECONDS).astype(int)
    mean = pd.Series(prepared["r2n"].to_numpy()).groupby(slot).mean()
    mean = mean.reindex(range(SLOTS)).fillna(mean.mean()).to_numpy()
    smooth = (np.roll(mean, 1) + mean + np.roll(mean, -1)) / 3
    return 0.5 * np.log(smooth / smooth.mean())


def sample_windows(
    prepared: pd.DataFrame, profile: np.ndarray, config: DetectorConfig
) -> pd.DataFrame:
    """One row every five minutes (targets do not overlap), with the time-of-day term.

    Windows with no price change at all, before or after the cut, are dropped: they are
    market closures (exchange maintenance), not a forecastable state.
    """
    h = _bars(HORIZON_SECONDS, config.bar_seconds)
    w = prepared.iloc[::h].dropna().copy()
    slot = (w["t_mid"].to_numpy() % 86400 // SLOT_SECONDS).astype(int)
    w["season"] = np.asarray(profile)[slot]
    flat = 0.5 * np.log(_FLOOR) + 1e-9
    return w[(w["y"] > flat) & (w["x5"] > flat)]


def fit(prepared: pd.DataFrame, config: DetectorConfig, instrument: str = "") -> HARModel:
    profile = season_profile(prepared)
    w = sample_windows(prepared, profile, config)
    X = np.column_stack([np.ones(len(w)), w[list(FEATURES)].to_numpy()])
    beta = np.linalg.lstsq(X, w["y"].to_numpy(), rcond=None)[0]
    cuts = np.exp(np.quantile(X @ beta, REGIME_QUANTILES))
    return HARModel(
        intercept=round(float(beta[0]), 6),
        coefs=tuple(round(float(b), 6) for b in beta[1:]),
        season=tuple(round(float(s), 5) for s in profile),
        cuts=tuple(round(float(c), 4) for c in cuts),
        instrument=instrument,
        trained_on=f"{w.index[0]:%Y-%m-%d} to {w.index[-1]:%Y-%m-%d}",
        n=len(w),
    )


def har_inputs(frame: pd.DataFrame, model: HARModel) -> np.ndarray:
    """Feature matrix in FEATURES order, with the model's own time-of-day profile."""
    season = model.season_at(frame["t_mid"].to_numpy())
    return np.column_stack([season if c == "season" else frame[c].to_numpy() for c in FEATURES])


# ---------------------------------------------------------------------------
# streaming
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Forecast:
    ratio: float  # predicted vol over the next 5 minutes / normal vol
    vol: float  # the same, annualized
    regime: Regime


class StreamingHAR:
    """Rolling sums over the look-backs, O(1) per bar; same numbers as ``prepare``."""

    def __init__(self, model: HARModel, config: DetectorConfig):
        self.model = model
        self.config = config
        self.k = [_bars(s, config.bar_seconds) for s in WINDOWS_SECONDS]
        size = max(self.k)
        self._r: deque[float] = deque(maxlen=size)
        self._n: deque[float] = deque(maxlen=size)
        self._regime = RegimeTracker(model.cuts)
        self._count = 0
        self._recompute()

    def _recompute(self) -> None:
        r, n = list(self._r), list(self._n)
        k1, _, k30 = self.k
        self._r2_sums = [sum(x * x for x in r[-k:]) for k in self.k]
        self._r_1m = sum(r[-k1:])
        self._n_1m, self._n_30m = sum(n[-k1:]), sum(n[-k30:])

    def update(self, r: float, n_trades: int) -> None:
        k1, _, k30 = self.k
        size = len(self._r)
        for i, k in enumerate(self.k):
            self._r2_sums[i] += r * r
            if size >= k:
                self._r2_sums[i] -= self._r[-k] ** 2  # leaves the window
        self._r_1m += r - (self._r[-k1] if size >= k1 else 0.0)
        self._n_1m += n_trades - (self._n[-k1] if size >= k1 else 0.0)
        self._n_30m += n_trades - (self._n[-k30] if size >= k30 else 0.0)
        self._r.append(r)
        self._n.append(float(n_trades))
        self._count += 1
        if self._count % 1000 == 0:  # clear floating-point drift of the running sums
            self._recompute()

    @property
    def ready(self) -> bool:
        return len(self._r) == self._r.maxlen

    def forecast(self, slow_var: float, bar_end_seconds: float) -> Forecast | None:
        if not self.ready or slow_var <= 0:
            return None
        k1, _, k30 = self.k
        x = [
            float(_log_vol_ratio(s / (k * slow_var)))
            for s, k in zip(self._r2_sums, self.k, strict=True)
        ]
        x.append(float(self.model.season_at(bar_end_seconds + HORIZON_SECONDS / 2)))
        x.append(math.log((self._n_1m + 1.0) / (self._n_30m / (k30 // k1) + 1.0)))
        x.append(abs(self._r_1m) / math.sqrt(k1 * slow_var))
        ratio = math.exp(float(self.model.predict(np.array(x))[0]))
        vol = ratio * math.sqrt(slow_var * self.config.bars_per_year)
        return Forecast(ratio, vol, self._regime.update(ratio))
