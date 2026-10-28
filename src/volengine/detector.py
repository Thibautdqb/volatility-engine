"""Short-term volatility spike detector.

Two exponentially weighted variances run on the same bar returns:

* a **fast** one (half-life 1 minute by default): the volatility right now;
* a **slow** one (half-life 30 minutes): the normal level it is compared with.

The signal is the ratio of the two volatilities, ``fast / slow``. A ratio of 2 means
the market is currently moving twice as much as it has over the last half hour. The
ratio is unit-free, so the same thresholds work for any instrument or bar size.

Alert levels use hysteresis so that a ratio hovering around a threshold does not
flicker: the level rises as soon as a threshold is crossed and only returns to normal
once the ratio has fallen below ``reset``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum

import numpy as np
import pandas as pd

SECONDS_PER_YEAR = 365 * 24 * 3600  # crypto trades every day


class Level(IntEnum):
    NORMAL = 0
    ELEVATED = 1
    SPIKE = 2


@dataclass(frozen=True)
class DetectorConfig:
    bar_seconds: float = 10.0
    fast_half_life: float = 60.0  # seconds
    slow_half_life: float = 1800.0  # seconds
    elevated: float = 2.0  # fast/slow volatility ratio
    spike: float = 3.0
    reset: float = 1.5

    def __post_init__(self):
        if not 0 < self.fast_half_life < self.slow_half_life:
            raise ValueError("need 0 < fast_half_life < slow_half_life")
        if not 1 <= self.reset < self.elevated < self.spike:
            raise ValueError("need 1 <= reset < elevated < spike")

    @property
    def lam_fast(self) -> float:
        return 0.5 ** (self.bar_seconds / self.fast_half_life)

    @property
    def lam_slow(self) -> float:
        return 0.5 ** (self.bar_seconds / self.slow_half_life)

    @property
    def bars_per_year(self) -> float:
        return SECONDS_PER_YEAR / self.bar_seconds

    @property
    def warmup_bars(self) -> int:
        """Bars needed before the slow variance has forgotten its starting value (4 half-lives)."""
        return math.ceil(4 * self.slow_half_life / self.bar_seconds)


@dataclass(frozen=True)
class Reading:
    fast_var: float  # per-bar variance
    slow_var: float
    ratio: float  # fast / slow volatility
    level: Level
    escalated: bool  # level went up on this bar: this is an alert


class SpikeDetector:
    def __init__(self, config: DetectorConfig, variance: float):
        """``variance`` is the per-bar variance both averages start from."""
        if not variance > 0:
            raise ValueError("initial variance must be positive")
        self.config = config
        self.fast_var = variance
        self.slow_var = variance
        self.level = Level.NORMAL

    @property
    def ratio(self) -> float:
        return math.sqrt(self.fast_var / self.slow_var)

    def update(self, r: float) -> Reading:
        c = self.config
        r2 = r * r
        self.fast_var = c.lam_fast * self.fast_var + (1.0 - c.lam_fast) * r2
        self.slow_var = c.lam_slow * self.slow_var + (1.0 - c.lam_slow) * r2
        ratio = self.ratio

        previous = self.level
        self.level = next_level(previous, ratio, c)
        return Reading(self.fast_var, self.slow_var, ratio, self.level, self.level > previous)


def next_level(level: Level, ratio: float, config: DetectorConfig) -> Level:
    """Escalate as soon as a threshold is crossed; step down only with a margin."""
    if ratio >= config.spike:
        return Level.SPIKE
    if ratio >= config.elevated:
        return max(level, Level.ELEVATED)
    if ratio < config.reset:
        return Level.NORMAL
    return min(level, Level.ELEVATED)  # between reset and elevated: hold, but leave spike


def run_detector(returns: pd.Series, config: DetectorConfig) -> pd.DataFrame:
    """Run the detector over a historical return series, bar by bar.

    The first ``config.warmup_bars`` bars only initialize the averages: their rows are
    dropped and no alert can fire during them. Volatilities in the output are annualized.
    Columns: ``return``, ``fast_vol``, ``slow_vol``, ``ratio``, ``level``, ``alert``.
    """
    r = returns.to_numpy(dtype=float)
    warm = config.warmup_bars
    if r.size <= warm:
        raise ValueError(f"need more than {warm} bars, got {r.size}")
    det = SpikeDetector(config, variance=float(np.mean(r[:warm] ** 2)))
    out = np.empty((r.size, 4))
    alert = np.zeros(r.size, dtype=bool)
    for i, x in enumerate(r):
        rd = det.update(x)
        out[i] = rd.fast_var, rd.slow_var, rd.ratio, rd.level
        alert[i] = rd.escalated
    df = pd.DataFrame(
        {
            "return": r,
            "fast_vol": np.sqrt(out[:, 0] * config.bars_per_year),
            "slow_vol": np.sqrt(out[:, 1] * config.bars_per_year),
            "ratio": out[:, 2],
            "level": out[:, 3].astype(int),
            "alert": alert,
        },
        index=returns.index,
    )
    return df.iloc[warm:]
