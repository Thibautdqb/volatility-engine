"""Confidence intervals used by the study."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy import stats


def stationary_bootstrap_indices(
    n: int, block_size: float, n_boot: int, rng: np.random.Generator
) -> np.ndarray:
    """Resampling indices of the stationary bootstrap (Politis & Romano, 1994).

    Blocks have geometric lengths with mean ``block_size`` and wrap around the end of
    the sample. With ``block_size <= 1`` this is the ordinary iid bootstrap.
    """
    if block_size <= 1:
        return rng.integers(0, n, size=(n_boot, n))
    idx = np.empty((n_boot, n), dtype=np.int64)
    idx[:, 0] = rng.integers(0, n, size=n_boot)
    restart = rng.random((n_boot, n)) < 1.0 / block_size
    fresh = rng.integers(0, n, size=(n_boot, n))
    for t in range(1, n):
        idx[:, t] = np.where(restart[:, t], fresh[:, t], (idx[:, t - 1] + 1) % n)
    return idx


def bootstrap_ci(
    data: np.ndarray,
    statistic: Callable[[np.ndarray], float] = np.mean,
    level: float = 0.95,
    n_boot: int = 2000,
    block_size: float = 1.0,
    seed: int | None = 0,
) -> tuple[float, float]:
    """Percentile bootstrap interval for ``statistic(data)``.

    ``data`` is resampled along its first axis, so rows of a 2-D array (e.g. a flag and
    an outcome observed at the same time) stay together. Set ``block_size`` above 1 for
    serially dependent data.
    """
    data = np.asarray(data, dtype=float)
    rng = np.random.default_rng(seed)
    idx = stationary_bootstrap_indices(len(data), block_size, n_boot, rng)
    boots = np.array([statistic(data[i]) for i in idx])
    boots = boots[np.isfinite(boots)]
    if boots.size == 0:
        return float("nan"), float("nan")
    alpha = 1.0 - level
    lo, hi = np.quantile(boots, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lo), float(hi)


def wilson_interval(successes: int, n: int, level: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (well-behaved for small n)."""
    if n == 0:
        return float("nan"), float("nan")
    z = stats.norm.ppf(0.5 + level / 2.0)
    p = successes / n
    denom = 1.0 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return float(centre - half), float(centre + half)
