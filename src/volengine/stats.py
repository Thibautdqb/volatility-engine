"""Statistical inference utilities.

Everything here is deliberately small and dependency-free (numpy/scipy only) so that
the inference behind each reported number can be audited in one place. The test
suite cross-checks these implementations against statsmodels.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy import stats

# ---------------------------------------------------------------------------
# HAC (Newey-West) variance
# ---------------------------------------------------------------------------


def newey_west_lag(n: int) -> int:
    """Automatic bandwidth floor(4 (n/100)^(2/9)) (Newey & West, 1994)."""
    return int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))


def long_run_variance(x: np.ndarray, lags: int | None = None, kernel: str = "bartlett") -> float:
    """Long-run variance of a (demeaned) series.

    ``bartlett`` is Newey-West (always positive). ``uniform`` is Hansen-Hodrick, the
    right choice for overlapping h-period sums, whose autocorrelation is exactly
    zero beyond lag h-1; it can turn negative in small samples.
    """
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    n = x.size
    if lags is None:
        lags = newey_west_lag(n)
    lrv = x @ x / n
    for k in range(1, lags + 1):
        w = 1.0 if kernel == "uniform" else 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * (x[k:] @ x[:-k]) / n
    return float(lrv)


@dataclass(frozen=True)
class MeanTest:
    """t-test on the mean of a (possibly autocorrelated) series."""

    mean: float
    se: float
    t_stat: float
    p_value: float
    n: int
    lags: int
    df: float | None = None  # Student-t reference; None means normal
    method: str = "newey-west"

    def ci(self, level: float = 0.95) -> tuple[float, float]:
        q = 0.5 + level / 2.0
        z = stats.norm.ppf(q) if self.df is None else stats.t.ppf(q, self.df)
        return self.mean - z * self.se, self.mean + z * self.se


def hac_mean_test(x: np.ndarray, lags: int | None = None) -> MeanTest:
    """Two-sided test of H0: E[x] = 0 with a Newey-West standard error.

    Use this whenever the observations overlap (e.g. 30-day realized vol sampled
    daily), where the naive iid standard error is badly understated.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    lags = newey_west_lag(n) if lags is None else lags
    se = np.sqrt(long_run_variance(x, lags) / n)
    t = x.mean() / se
    p = 2.0 * stats.norm.sf(abs(t))
    return MeanTest(float(x.mean()), float(se), float(t), float(p), n, lags)


def overlapping_mean_test(x: np.ndarray, horizon: int) -> MeanTest:
    """Test H0: E[x] = 0 when x_t are overlapping ``horizon``-period averages.

    Newey-West with lag = horizon over-rejects badly here (about 12% at a nominal
    5% in simulations with n = 900, h = 30). Hansen-Hodrick (uniform kernel,
    h - 1 lags) with a Student-t reference on n/h - 1 degrees of freedom, the
    number of independent windows, is close to nominal size (5.4%). If the
    Hansen-Hodrick variance is not positive, fall back to Bartlett with 2h lags.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    df = max(n / horizon - 1.0, 1.0)
    lags, method = horizon - 1, "hansen-hodrick"
    lrv = long_run_variance(x, lags, kernel="uniform")
    if lrv <= 0:
        lags, method = 2 * horizon, "newey-west"
        lrv = long_run_variance(x, lags)
    se = np.sqrt(lrv / n)
    t = x.mean() / se
    p = 2.0 * stats.t.sf(abs(t), df)
    return MeanTest(float(x.mean()), float(se), float(t), float(p), n, lags, df, method)


def non_overlapping_mean_test(x: np.ndarray, horizon: int, offset: int = 0) -> MeanTest:
    """Plain t-test on every ``horizon``-th observation (independent windows).
    Exact size, lower power: used as a robustness check."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)][offset::horizon]
    n = x.size
    se = x.std(ddof=1) / np.sqrt(n)
    t = x.mean() / se
    p = 2.0 * stats.t.sf(abs(t), n - 1)
    return MeanTest(
        float(x.mean()), float(se), float(t), float(p), n, 0, n - 1.0, "non-overlapping"
    )


def ols_hac(y: np.ndarray, X: np.ndarray, lags: int | None = None):
    """OLS with Newey-West covariance. Returns (beta, cov)."""
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    n = y.size
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    u = y - X @ beta
    g = X * u[:, None]
    lags = newey_west_lag(n) if lags is None else lags
    S = g.T @ g / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        G = g[k:].T @ g[:-k] / n
        S += w * (G + G.T)
    cov = n * XtX_inv @ S @ XtX_inv
    return beta, cov


# ---------------------------------------------------------------------------
# Forecast comparison
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DMResult:
    mean_loss_diff: float
    statistic: float
    p_value: float
    n: int

    @property
    def significant(self) -> bool:
        return self.p_value < 0.05


def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray, horizon: int = 1) -> DMResult:
    """Diebold-Mariano test of equal predictive accuracy.

    H0: E[loss_a - loss_b] = 0. A negative statistic means model A has lower loss.
    Uses the Harvey, Leybourne & Newbold (1997) small-sample correction and a
    Student-t reference distribution with n-1 degrees of freedom.
    """
    d = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    d = d[np.isfinite(d)]
    n = d.size
    lags = max(horizon - 1, newey_west_lag(n))
    var = long_run_variance(d, lags) / n
    dm = d.mean() / np.sqrt(var)
    h = horizon
    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    stat = dm * hln
    p = 2.0 * stats.t.sf(abs(stat), df=n - 1)
    return DMResult(float(d.mean()), float(stat), float(p), n)


@dataclass(frozen=True)
class MZResult:
    """Mincer-Zarnowitz regression proxy = a + b * forecast + u."""

    intercept: float
    slope: float
    se_intercept: float
    se_slope: float
    r2: float
    wald_stat: float
    wald_p_value: float  # H0: a = 0, b = 1 (forecast is unbiased)
    n: int


def mincer_zarnowitz(proxy: np.ndarray, forecast: np.ndarray, lags: int | None = None) -> MZResult:
    y = np.asarray(proxy, dtype=float)
    f = np.asarray(forecast, dtype=float)
    mask = np.isfinite(y) & np.isfinite(f)
    y, f = y[mask], f[mask]
    X = np.column_stack([np.ones_like(f), f])
    beta, cov = ols_hac(y, X, lags)
    r = beta - np.array([0.0, 1.0])
    wald = float(r @ np.linalg.solve(cov, r))
    p = float(stats.chi2.sf(wald, df=2))
    resid = y - X @ beta
    r2 = 1.0 - resid.var() / y.var()
    se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    return MZResult(beta[0], beta[1], se[0], se[1], float(r2), wald, p, int(y.size))


# ---------------------------------------------------------------------------
# Conditional heteroskedasticity diagnostics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TestResult:
    statistic: float
    p_value: float
    df: int


def ljung_box(x: np.ndarray, lags: int = 10, fitted_params: int = 0) -> TestResult:
    """Ljung-Box Q statistic. Apply to squared standardized residuals to check
    that a volatility model has absorbed the ARCH effects."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)] - np.nanmean(x)
    n = x.size
    denom = x @ x
    q = 0.0
    for k in range(1, lags + 1):
        rho = (x[k:] @ x[:-k]) / denom
        q += rho**2 / (n - k)
    q *= n * (n + 2)
    df = lags - fitted_params
    return TestResult(float(q), float(stats.chi2.sf(q, df)), df)


def arch_lm(returns: np.ndarray, lags: int = 5) -> TestResult:
    """Engle (1982) ARCH-LM test: regress e_t^2 on its own lags, LM = n * R^2."""
    e2 = np.asarray(returns, dtype=float)
    e2 = (e2[np.isfinite(e2)] - np.nanmean(e2)) ** 2
    y = e2[lags:]
    X = np.column_stack([np.ones(y.size)] + [e2[lags - k : -k] for k in range(1, lags + 1)])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    r2 = 1.0 - resid.var() / y.var()
    lm = y.size * r2
    return TestResult(float(lm), float(stats.chi2.sf(lm, lags)), lags)


# ---------------------------------------------------------------------------
# Resampling and small-sample inference
# ---------------------------------------------------------------------------


def bootstrap_ci(
    x: np.ndarray,
    statistic: Callable[[np.ndarray], float] = np.mean,
    level: float = 0.95,
    n_boot: int = 5000,
    block_size: float | None = None,
    seed: int | None = 0,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval.

    With ``block_size`` set, uses the stationary bootstrap (Politis & Romano, 1994)
    with geometric block lengths of that mean, which preserves serial dependence.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    rng = np.random.default_rng(seed)
    if block_size is None or block_size <= 1:
        idx = rng.integers(0, n, size=(n_boot, n))
    else:
        p = 1.0 / block_size
        idx = np.empty((n_boot, n), dtype=np.int64)
        idx[:, 0] = rng.integers(0, n, size=n_boot)
        restart = rng.random((n_boot, n)) < p
        fresh = rng.integers(0, n, size=(n_boot, n))
        for t in range(1, n):
            idx[:, t] = np.where(restart[:, t], fresh[:, t], (idx[:, t - 1] + 1) % n)
    boots = np.apply_along_axis(statistic, 1, x[idx])
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


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (q-values) controlling the FDR.

    Needed whenever a screener tests many tickers at once: at a 5% threshold,
    one in twenty null tickers would otherwise look 'significant' by chance.
    """
    p = np.asarray(p_values, dtype=float)
    out = np.full_like(p, np.nan)
    mask = np.isfinite(p)
    pv = p[mask]
    m = pv.size
    if m == 0:
        return out
    order = np.argsort(pv)
    ranked = pv[order] * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)
    res = np.empty(m)
    res[order] = q
    out[mask] = res
    return out
