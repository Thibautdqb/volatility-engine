"""Five-minute volatility forecast, probability of a violent move, and regimes.

The model is linear in the inputs of :mod:`volengine.features`, fitted by least squares
on non-overlapping five-minute windows:

    y = log(vol next 5m / normal) = c + sum_i b_i x_i + b_s season(time of day) + e

* ``season`` is the usual log volatility of each 5-minute slot of the day relative to
  the daily average (US equity open, scheduled releases), estimated on the training
  data and stored with the coefficients.
* A logistic regression of the event "vol next 5m >= 2 x normal" on the forecast,
  fitted on the same training windows, turns the forecast into a calibrated
  probability. (Reading it off the residual distribution is miscalibrated: residuals
  spread more when the forecast is high.)
* Regimes are the rank of the forecast among the training forecasts: the lowest 25%
  are CALM, the middle 50% NORMAL, the next 20% ACTIVE and the top 5% STRESSED. A
  regime rises as soon as a cut point is crossed and falls back after one minute below.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from enum import IntEnum
from importlib import resources

import numpy as np
import pandas as pd
from scipy import optimize

from volengine.detector import DetectorConfig
from volengine.features import StreamingFeatures, windows

SLOT_SECONDS = 300
SLOTS = 86_400 // SLOT_SECONDS
REGIME_QUANTILES = (0.25, 0.75, 0.95)
VIOLENT = 2.0  # "violent": realized vol over the next five minutes >= 2 x normal
DEFAULT_FEATURES = ("x1", "x5", "x30", "x4h", "x1d", "season", "trades", "absmove", "weekend")


class Regime(IntEnum):
    CALM = 0
    NORMAL = 1
    ACTIVE = 2
    STRESSED = 3


def classify(ratio: float, cuts) -> Regime:
    return Regime(int(np.searchsorted(cuts, ratio, side="right")))


class RegimeTracker:
    """Regime with hysteresis: it rises at once, but falls only after ``hold`` bars below.

    Without it, a forecast sitting on a cut point flips the regime every bar.
    """

    def __init__(self, cuts, hold: int = 6):
        self.cuts = tuple(cuts)
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


def track_regimes(ratios: np.ndarray, cuts, hold: int = 6) -> np.ndarray:
    tracker = RegimeTracker(cuts, hold)
    return np.array([int(tracker.update(float(x))) for x in ratios], dtype=int)


def season_profile(features: pd.DataFrame) -> np.ndarray:
    """Log volatility of each 5-minute slot of the day relative to the daily average.

    Smoothed over three neighbouring slots (wrapping around midnight) to limit noise.
    """
    epoch = pd.Timestamp(0, tz="UTC")
    t = np.asarray((features.index - epoch) / pd.Timedelta(seconds=1))
    slot = (t % 86_400 // SLOT_SECONDS).astype(int)
    mean = pd.Series(features["r2n"].to_numpy()).groupby(slot).mean()
    mean = mean.reindex(range(SLOTS)).fillna(mean.mean()).to_numpy()
    smooth = (np.roll(mean, 1) + mean + np.roll(mean, -1)) / 3
    return 0.5 * np.log(smooth / smooth.mean())


@dataclass(frozen=True)
class Model:
    features: tuple[str, ...]
    intercept: float
    coefs: tuple[float, ...]
    season: tuple[float, ...] = field(default=(0.0,) * SLOTS, repr=False)
    logit: tuple[float, float, float] = (-3.5, 3.0, 0.0)  # P(violent) on 1, f, f^2
    cuts: tuple[float, float, float] = (0.8, 0.95, 1.1)  # forecast ratios between regimes
    instrument: str = ""
    trained_on: str = ""
    n: int = 0

    # -- inputs ------------------------------------------------------------

    def season_at(self, t_mid) -> np.ndarray:
        slot = (np.asarray(t_mid) % 86_400 // SLOT_SECONDS).astype(int)
        return np.asarray(self.season)[slot]

    def design(self, frame: pd.DataFrame) -> np.ndarray:
        cols = [
            self.season_at(frame["t_mid"].to_numpy()) if f == "season" else frame[f].to_numpy()
            for f in self.features
        ]
        return np.column_stack(cols)

    # -- outputs -----------------------------------------------------------

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Forecast of log(vol next 5m / normal) for each row."""
        return self.intercept + self.design(frame) @ np.asarray(self.coefs)

    def predict_one(self, values: dict[str, float]) -> float:
        x = [
            float(self.season_at(values["t_mid"])) if f == "season" else values[f]
            for f in self.features
        ]
        return self.intercept + float(np.dot(self.coefs, x))

    def p_violent(self, log_forecast) -> np.ndarray:
        """P(vol next 5m >= 2 x normal) given the forecast of log(vol next 5m / normal)."""
        f = np.asarray(log_forecast, dtype=float)
        a, b, c = self.logit
        if c > 0:  # flatten the parabola below its minimum: P never falls as f rises
            f = np.maximum(f, -b / (2 * c))
        return 1.0 / (1.0 + np.exp(-np.clip(a + b * f + c * f * f, -50, 50)))

    def regime(self, ratio: float) -> Regime:
        return classify(ratio, self.cuts)

    # -- persistence -------------------------------------------------------

    def to_dict(self) -> dict:
        d = asdict(self)
        d["coefs"] = dict(zip(self.features, self.coefs, strict=True))
        del d["features"]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> Model:
        d = dict(d)
        coefs = d.pop("coefs")
        return cls(
            features=tuple(coefs),
            coefs=tuple(coefs.values()),
            season=tuple(d.pop("season")),
            logit=tuple(d.pop("logit")),
            cuts=tuple(d.pop("cuts")),
            **d,
        )


def fit(
    features: pd.DataFrame | list[pd.DataFrame],
    config: DetectorConfig,
    names=DEFAULT_FEATURES,
    instrument: str = "",
) -> Model:
    """Least squares on the non-overlapping five-minute windows of ``features``.

    A list of frames (one per instrument) fits a single pooled model; windows are cut
    within each frame, never across them.
    """
    names = tuple(names)
    frames = features if isinstance(features, list) else [features]
    profile = season_profile(pd.concat(frames, ignore_index=False))
    draft = Model(names, 0.0, (0.0,) * len(names), season=tuple(profile))
    w = pd.concat([windows(f, config) for f in frames]).sort_index()
    w = w.dropna(subset=[f for f in names if f != "season"])
    X = np.column_stack([np.ones(len(w)), draft.design(w)])
    y = w["y"].to_numpy()
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    fitted = X @ beta
    logit = _fit_logit(fitted, y >= math.log(VIOLENT))
    return Model(
        features=names,
        intercept=round(float(beta[0]), 6),
        coefs=tuple(round(float(b), 6) for b in beta[1:]),
        season=tuple(round(float(s), 5) for s in draft.season),
        logit=tuple(round(float(v), 5) for v in logit),
        cuts=tuple(round(float(c), 4) for c in np.exp(np.quantile(fitted, REGIME_QUANTILES))),
        instrument=instrument,
        trained_on=f"{w.index[0]:%Y-%m-%d} to {w.index[-1]:%Y-%m-%d}",
        n=len(w),
    )


def _fit_logit(f: np.ndarray, hit: np.ndarray, ridge: float = 1.0) -> np.ndarray:
    """P(hit) = sigmoid(a + b f + c f^2) by penalized log-loss.

    The quadratic term lets the upper tail bend. A convex fit is flattened below its
    minimum when used (see ``Model.p_violent``), so it never decreases; a concave fit
    that turns down inside the range of training forecasts is replaced by a linear one.
    """
    y = hit.astype(float)

    def solve(cols: np.ndarray) -> np.ndarray:
        A = np.column_stack([np.ones(len(cols)), cols])
        rate = min(max(y.mean(), 1e-4), 1 - 1e-4)
        x0 = np.zeros(A.shape[1])
        x0[0] = math.log(rate / (1 - rate))

        def loss(beta):
            z = A @ beta
            pen = 0.5 * ridge * beta[1:] @ beta[1:]
            return np.sum(np.logaddexp(0.0, z) - y * z) + pen

        def grad(beta):
            p = 1.0 / (1.0 + np.exp(-np.clip(A @ beta, -50, 50)))
            g = A.T @ (p - y)
            g[1:] += ridge * beta[1:]
            return g

        return optimize.minimize(loss, x0, jac=grad, method="L-BFGS-B").x

    a, b, c = solve(np.column_stack([f, f * f]))
    hi = np.quantile(f, 0.999)
    if c < 0 and b + 2 * c * hi <= 0:
        a, b = solve(f[:, None])
        c = 0.0
    return np.array([a, b, c])


def default_model(instrument: str) -> Model:
    """Coefficients shipped with the package, produced by ``volengine fit``."""
    data = json.loads(resources.files("volengine").joinpath("models.json").read_text())
    # Instruments without their own model get the one fitted on BTC and ETH together:
    # coefficients transfer across assets with little loss (see the README).
    return Model.from_dict(data.get(instrument) or data["pooled"])


@dataclass(frozen=True)
class Forecast:
    ratio: float  # forecast vol over the next 5 minutes / normal vol
    vol: float  # the same, annualized
    p_violent: float  # P(vol next 5 minutes >= 2 x normal)
    regime: Regime


class StreamingForecaster:
    """Live counterpart of ``Model.predict`` and ``track_regimes``."""

    def __init__(self, model: Model, config: DetectorConfig):
        self.model = model
        self.config = config
        self.features = StreamingFeatures(config, model.features)
        self.tracker = RegimeTracker(model.cuts)

    def update(self, r: float, n: int, volume: float, buy: float, liq: float) -> None:
        self.features.update(r, n, volume, buy, liq)

    def forecast(self, slow_var: float, bar_end_seconds: float) -> Forecast | None:
        if slow_var <= 0 or not self.features.ready(self.model.features):
            return None
        log_f = self.model.predict_one(self.features.values(slow_var, bar_end_seconds))
        ratio = math.exp(log_f)
        vol = ratio * math.sqrt(slow_var * self.config.bars_per_year)
        p = float(self.model.p_violent(log_f))
        return Forecast(ratio, vol, p, self.tracker.update(ratio))
