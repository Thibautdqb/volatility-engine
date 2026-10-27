import numpy as np
import pandas as pd
import pytest


def simulate_garch(n, omega, alpha, beta, seed=0, df=None):
    rng = np.random.default_rng(seed)
    r = np.empty(n)
    s2 = omega / (1 - alpha - beta)
    for t in range(n):
        z = rng.standard_normal() if df is None else rng.standard_t(df) / np.sqrt(df / (df - 2))
        r[t] = np.sqrt(s2) * z
        s2 = omega + alpha * r[t] ** 2 + beta * s2
    return r


def simulate_gbm_ohlc(n_bars, sigma_annual, steps_per_bar=500, periods=365.0, seed=0):
    """OHLC bars from a finely sampled driftless GBM; returns (df, per-bar sigma)."""
    rng = np.random.default_rng(seed)
    sigma_bar = sigma_annual / np.sqrt(periods)
    dz = rng.standard_normal((n_bars, steps_per_bar)) * sigma_bar / np.sqrt(steps_per_bar)
    logp = np.log(100.0) + np.cumsum(dz.ravel()).reshape(n_bars, steps_per_bar)
    prev_close = np.r_[np.log(100.0), logp[:-1, -1]]
    paths = np.column_stack([prev_close, logp])
    df = pd.DataFrame(
        {
            "open": np.exp(paths[:, 0]),
            "high": np.exp(paths.max(axis=1)),
            "low": np.exp(paths.min(axis=1)),
            "close": np.exp(paths[:, -1]),
        },
        index=pd.date_range("2024-01-01", periods=n_bars, freq="D", tz="UTC"),
    )
    return df, sigma_bar


@pytest.fixture(scope="session")
def garch_returns():
    return simulate_garch(4000, 2e-6, 0.08, 0.90, seed=1, df=6)
