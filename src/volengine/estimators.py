"""Realized volatility estimators.

All estimators return *per-period* variances unless their name says otherwise; use
:func:`annualize` with the right number of periods per year to convert. Crypto
markets trade continuously, so a year is 365 days; equities use 252 sessions.

References
----------
Parkinson (1980), Garman & Klass (1980), Rogers & Satchell (1991),
Yang & Zhang (2000), Andersen, Bollerslev, Diebold & Labys (2000) for the
volatility signature plot.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize_scalar

from volengine.stats import bootstrap_ci

SECONDS_PER_YEAR_CRYPTO = 365.0 * 24 * 3600
TRADING_DAYS_EQUITY = 252


def periods_per_year(bar_seconds: float, calendar: str = "crypto") -> float:
    """Number of bars of length ``bar_seconds`` in a year."""
    if calendar == "crypto":
        return SECONDS_PER_YEAR_CRYPTO / bar_seconds
    if calendar == "equity":
        # 6.5 trading hours per session
        return TRADING_DAYS_EQUITY * 6.5 * 3600 / bar_seconds
    raise ValueError(f"unknown calendar {calendar!r}")


def annualize(variance, periods: float):
    """Per-period variance -> annualized volatility."""
    out = np.sqrt(np.asarray(variance, dtype=float) * periods)
    return float(out) if out.ndim == 0 else out


def log_returns(close: pd.Series) -> pd.Series:
    return np.log(close).diff().dropna()


# ---------------------------------------------------------------------------
# Point estimators on a window of OHLC bars
# ---------------------------------------------------------------------------


def close_to_close(returns: np.ndarray | pd.Series, demean: bool = False) -> float:
    """Per-period variance from log returns.

    The zero-mean version is the default: at intraday and daily horizons the drift
    is negligible relative to its estimation error, and dropping it lowers the MSE.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if demean:
        return float(r.var(ddof=1))
    return float(r @ r / r.size)


def _hloc(df: pd.DataFrame):
    o, h, l, c = (np.log(df[k].to_numpy(dtype=float)) for k in ("open", "high", "low", "close"))
    return o, h, l, c


def parkinson(df: pd.DataFrame) -> float:
    """High-low estimator, ~5x more efficient than close-to-close under GBM.
    Biased downward when the true range is not observed (discrete sampling)."""
    _, h, l, _ = _hloc(df)
    return float(np.mean((h - l) ** 2) / (4.0 * np.log(2.0)))


def garman_klass(df: pd.DataFrame) -> float:
    """OHLC estimator for zero-drift GBM without opening jumps (efficiency ~7.4)."""
    o, h, l, c = _hloc(df)
    return float(np.mean(0.5 * (h - l) ** 2 - (2.0 * np.log(2.0) - 1.0) * (c - o) ** 2))


def rogers_satchell(df: pd.DataFrame) -> float:
    """Drift-independent OHLC estimator."""
    o, h, l, c = _hloc(df)
    return float(np.mean((h - c) * (h - o) + (l - c) * (l - o)))


def yang_zhang(df: pd.DataFrame) -> float:
    """Minimum-variance combination of overnight, open-to-close and Rogers-Satchell
    variances; robust to both drift and opening jumps."""
    o, h, l, c = _hloc(df)
    n = len(df) - 1
    if n < 2:
        return float("nan")
    overnight = o[1:] - c[:-1]
    open_close = c[1:] - o[1:]
    rs = (h[1:] - c[1:]) * (h[1:] - o[1:]) + (l[1:] - c[1:]) * (l[1:] - o[1:])
    k = 0.34 / (1.34 + (n + 1) / (n - 1))
    return float(overnight.var(ddof=1) + k * open_close.var(ddof=1) + (1 - k) * rs.mean())


ESTIMATORS = {
    "close_to_close": lambda df: close_to_close(log_returns(df["close"])),
    "parkinson": parkinson,
    "garman_klass": garman_klass,
    "rogers_satchell": rogers_satchell,
    "yang_zhang": yang_zhang,
}


def rolling_volatility(
    df: pd.DataFrame, window: int, periods: float, estimator: str = "yang_zhang"
) -> pd.Series:
    """Annualized rolling volatility using one of :data:`ESTIMATORS`."""
    if estimator == "close_to_close":
        r = np.log(df["close"]).diff()
        var = (r**2).rolling(window).mean()
    elif estimator == "yang_zhang":
        values = [
            yang_zhang(df.iloc[i - window : i + 1]) if i >= window else np.nan
            for i in range(len(df))
        ]
        var = pd.Series(values, index=df.index)
    else:
        o, h, l, c = _hloc(df)
        per_bar = {
            "parkinson": (h - l) ** 2 / (4.0 * np.log(2.0)),
            "garman_klass": 0.5 * (h - l) ** 2 - (2.0 * np.log(2.0) - 1.0) * (c - o) ** 2,
            "rogers_satchell": (h - c) * (h - o) + (l - c) * (l - o),
        }[estimator]
        var = pd.Series(per_bar, index=df.index).rolling(window).mean()
    return np.sqrt(var * periods).rename(estimator)


# ---------------------------------------------------------------------------
# Confidence intervals
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VolEstimate:
    vol: float
    lower: float
    upper: float
    n: int
    method: str


def volatility_ci(
    returns: np.ndarray | pd.Series,
    periods: float,
    level: float = 0.95,
    method: str = "bootstrap",
    block_size: float | None = None,
) -> VolEstimate:
    """Annualized close-to-close volatility with a confidence interval.

    ``chi2`` is exact under iid Gaussian returns: n * s^2 / sigma^2 ~ chi2(n). It is
    too narrow with fat tails and volatility clustering, which is why the default
    is a stationary block bootstrap (block length defaults to n^(1/3)).
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    var = r @ r / n
    vol = float(np.sqrt(var * periods))
    alpha = 1.0 - level
    if method == "chi2":
        lo = np.sqrt(n * var / stats.chi2.ppf(1 - alpha / 2, n) * periods)
        hi = np.sqrt(n * var / stats.chi2.ppf(alpha / 2, n) * periods)
    elif method == "bootstrap":
        bs = block_size if block_size is not None else max(1.0, n ** (1.0 / 3.0))
        lo_v, hi_v = bootstrap_ci(r, lambda x: x @ x / x.size, level, block_size=bs)
        lo, hi = np.sqrt(lo_v * periods), np.sqrt(hi_v * periods)
    else:
        raise ValueError(f"unknown method {method!r}")
    return VolEstimate(vol, float(lo), float(hi), n, method)


# ---------------------------------------------------------------------------
# EWMA
# ---------------------------------------------------------------------------


def ewma_variance(returns: np.ndarray | pd.Series, lam: float, init: float | None = None):
    """One-step-ahead EWMA variance forecasts.

    ``out[t]`` is the forecast of Var(r_t) made with information up to t-1:
    sigma2_t = lam * sigma2_{t-1} + (1 - lam) * r_{t-1}^2.
    """
    r = np.asarray(returns, dtype=float)
    out = np.empty(r.size)
    s2 = float(np.mean(r[: min(r.size, 30)] ** 2)) if init is None else init
    for t in range(r.size):
        out[t] = s2
        s2 = lam * s2 + (1.0 - lam) * r[t] ** 2
    if isinstance(returns, pd.Series):
        return pd.Series(out, index=returns.index, name="ewma")
    return out


def qlike(proxy: np.ndarray, forecast: np.ndarray) -> np.ndarray:
    """QLIKE loss, robust to noise in the variance proxy (Patton, 2011)."""
    ratio = np.asarray(proxy) / np.asarray(forecast)
    return ratio - np.log(ratio) - 1.0


def fit_ewma_lambda(
    returns: np.ndarray | pd.Series, burn_in: int = 30, bounds=(0.80, 0.999)
) -> float:
    """Estimate the EWMA decay by minimizing the Gaussian QLIKE on r_t^2.

    The RiskMetrics value 0.94 was calibrated on daily equity and FX returns in
    the 1990s; it has no reason to be optimal for crypto or for intraday bars.
    """
    r = np.asarray(returns, dtype=float)
    r2 = r**2
    floor = max(np.mean(r2) * 1e-6, 1e-20)

    def loss(lam):
        f = ewma_variance(r, lam)[burn_in:]
        p = np.maximum(r2[burn_in:], floor)
        return float(np.mean(qlike(p, f)))

    res = minimize_scalar(loss, bounds=bounds, method="bounded")
    return float(res.x)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def volatility_cone(
    close: pd.Series,
    windows=(5, 10, 21, 63, 126),
    periods: float = 365.0,
    quantiles=(0.05, 0.25, 0.5, 0.75, 0.95),
) -> pd.DataFrame:
    """Distribution of rolling realized vol across horizons (Burghardt & Lane, 1990).

    Note that overlapping windows make the quantiles of long horizons rest on few
    independent observations.
    """
    r = log_returns(close)
    rows = {}
    for w in windows:
        rv = np.sqrt((r**2).rolling(w).mean() * periods).dropna()
        row = rv.quantile(list(quantiles))
        row.index = [f"q{int(q * 100):02d}" for q in quantiles]
        row["current"] = rv.iloc[-1] if len(rv) else np.nan
        row["independent_obs"] = len(r) // w
        rows[w] = row
    return pd.DataFrame(rows).T.rename_axis("window")


def signature_plot(
    close: pd.Series, base_seconds: float, steps=(1, 2, 5, 10, 15, 30, 60), periods_fn=None
) -> pd.Series:
    """Annualized realized vol as a function of the sampling interval.

    An upward slope at the finest frequencies indicates microstructure noise
    (bid-ask bounce) and tells you which bar size the live monitor should use.
    """
    periods_fn = periods_fn or (lambda s: periods_per_year(s, "crypto"))
    out = {}
    for k in steps:
        sampled = close.iloc[::k]
        r = log_returns(sampled)
        out[int(k * base_seconds)] = float(np.sqrt(np.mean(r**2) * periods_fn(k * base_seconds)))
    return pd.Series(out, name="realized_vol").rename_axis("sampling_seconds")
