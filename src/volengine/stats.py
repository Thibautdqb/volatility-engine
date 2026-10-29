"""Intervals and tests used by the evaluation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

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


def newey_west_variance(x: np.ndarray, lags: int) -> float:
    """Long-run variance of a series with Bartlett weights (Newey & West, 1987)."""
    x = np.asarray(x, dtype=float) - np.mean(x)
    n = x.size
    v = x @ x / n
    for k in range(1, min(lags, n - 1) + 1):
        v += 2 * (1 - k / (lags + 1)) * (x[k:] @ x[:-k]) / n
    return float(v)


@dataclass(frozen=True)
class MeanTest:
    mean: float
    se: float
    t: float
    p_value: float


def hac_mean_test(x: np.ndarray, lags: int) -> MeanTest:
    """t-test of mean(x) = 0 with a HAC standard error."""
    x = np.asarray(x, dtype=float)
    se = float(np.sqrt(newey_west_variance(x, lags) / x.size))
    t = float(np.mean(x) / se) if se > 0 else float("nan")
    return MeanTest(float(np.mean(x)), se, t, float(2 * stats.norm.sf(abs(t))))


def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray, lags: int) -> MeanTest:
    """Test of equal expected loss; a negative mean means A has the lower loss."""
    return hac_mean_test(np.asarray(loss_a) - np.asarray(loss_b), lags)


def ols_hac(y: np.ndarray, x: np.ndarray, lags: int) -> tuple[np.ndarray, np.ndarray]:
    """Coefficients of y on [1, x] and their HAC standard errors."""
    X = np.column_stack([np.ones(len(y)), x])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    u = y - X @ beta
    xtx_inv = np.linalg.inv(X.T @ X)
    g = X * u[:, None]
    s = g.T @ g
    for k in range(1, lags + 1):
        w = 1 - k / (lags + 1)
        c = g[k:].T @ g[:-k]
        s += w * (c + c.T)
    cov = xtx_inv @ s @ xtx_inv
    return beta, np.sqrt(np.diag(cov))


def auc(score: np.ndarray, label: np.ndarray) -> float:
    """Area under the ROC curve: P(score of a positive > score of a negative)."""
    label = np.asarray(label).astype(bool)
    n1, n0 = label.sum(), (~label).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    rank = stats.rankdata(score)
    return float((rank[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
