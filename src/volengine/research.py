"""End-to-end volatility study on a Deribit instrument.

1. Is there volatility clustering to model at all?      (ARCH-LM on daily returns)
2. In-sample GARCH(1,1) fit with robust standard errors (and residual diagnostics)
3. Out-of-sample: does GARCH or a fitted EWMA beat RiskMetrics? (Diebold-Mariano)
4. Variance risk premium: is DVOL a biased predictor of realized vol? (HAC t-test)

The variance proxy for day t is the sum of squared hourly log returns (realized
variance), which is far less noisy than the squared daily return.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from volengine import estimators as est
from volengine.deribit.rest import DeribitREST
from volengine.evaluation import compare_forecasts, walk_forward_forecasts
from volengine.garch import GarchResult, fit_garch
from volengine.stats import (
    MeanTest,
    TestResult,
    arch_lm,
    mincer_zarnowitz,
    non_overlapping_mean_test,
    overlapping_mean_test,
)

DAYS_PER_YEAR = 365.0


@dataclass
class DailyData:
    ohlc: pd.DataFrame  # session OHLC
    returns: pd.Series  # session log returns
    realized_variance: pd.Series  # sum of squared hourly returns per session
    dvol: pd.Series | None  # DVOL close, as a decimal


SESSION_OFFSET = pd.Timedelta(hours=8)  # Deribit daily settlement at 08:00 UTC


def daily_from_hourly(hourly: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Aggregate hourly bars into 08:00-UTC sessions.

    Returns (ohlc, returns, realized_variance), all indexed by session start and
    restricted to complete sessions (24 hourly returns). The daily return is the
    sum of the hourly log returns, so it covers exactly the same window as the
    realized variance it is compared against.
    """
    hr = np.log(hourly["close"]).diff().dropna()
    session = (hourly.index - SESSION_OFFSET).floor("D") + SESSION_OFFSET
    hr_session = (hr.index - SESSION_OFFSET).floor("D") + SESSION_OFFSET
    count = hr.groupby(hr_session).count()
    complete = count[count == 24].index
    returns = hr.groupby(hr_session).sum().loc[complete].rename("return")
    rv = (hr**2).groupby(hr_session).sum().loc[complete].rename("rv")
    ohlc = (
        hourly.groupby(session)
        .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
        .loc[complete]
    )
    return ohlc, returns, rv


def load_daily(instrument: str = "BTC-PERPETUAL", start: str = "2019-01-01") -> DailyData:
    start_ts = pd.Timestamp(start, tz="UTC")
    with DeribitREST() as api:
        hourly = api.ohlc(instrument, "60", start=start_ts)
        currency = instrument.split("-")[0]
        dvol = None
        if currency in ("BTC", "ETH"):
            dvol = api.dvol(currency, "1D", start=start_ts)["close"] / 100.0
    ohlc, returns, rv = daily_from_hourly(hourly)
    if dvol is not None:
        # The close of DVOL's UTC day d is observed at 00:00 on d+1, i.e. before the
        # session starting 08:00 on d+1: label it with that session (no look-ahead).
        dvol.index = dvol.index.floor("D") + pd.Timedelta(days=1) + SESSION_OFFSET
    return DailyData(ohlc, returns, rv, dvol)


@dataclass
class VRPResult:
    """Implied (DVOL) minus subsequently realized 30-day volatility."""

    vol_spread: MeanTest  # in vol points (decimal), Hansen-Hodrick
    variance_spread: MeanTest
    vol_spread_independent: MeanTest  # robustness: non-overlapping windows only
    share_positive: float
    mz_slope: float  # realized_var = a + b * implied_var; b < 1 => implied overreacts
    mz_unbiased_p: float
    series: pd.DataFrame


def variance_risk_premium(data: DailyData, horizon: int = 30) -> VRPResult:
    if data.dvol is None:
        raise ValueError("no implied volatility index for this instrument")
    rv = data.realized_variance
    # Realized variance over the sessions [t, t+horizon), annualized. DVOL labelled t
    # was observed before session t opened. Requires `horizon` consecutive sessions.
    rv = rv.asfreq("D")
    fwd = rv[::-1].rolling(horizon).sum()[::-1] * DAYS_PER_YEAR / horizon
    df = pd.concat({"implied": data.dvol, "realized_var": fwd}, axis=1).dropna()
    df["realized"] = np.sqrt(df["realized_var"])
    df["implied_var"] = df["implied"] ** 2
    vol_diff = (df["implied"] - df["realized"]).to_numpy()
    var_diff = (df["implied_var"] - df["realized_var"]).to_numpy()
    # Overlapping windows: consecutive observations share 29 of 30 days.
    vol_t = overlapping_mean_test(vol_diff, horizon)
    var_t = overlapping_mean_test(var_diff, horizon)
    vol_ind = non_overlapping_mean_test(vol_diff, horizon)
    mz = mincer_zarnowitz(
        df["realized_var"].to_numpy(), df["implied_var"].to_numpy(), lags=2 * horizon
    )
    return VRPResult(
        vol_t, var_t, vol_ind, float(np.mean(vol_diff > 0)), mz.slope, mz.wald_p_value, df
    )


@dataclass
class Report:
    instrument: str
    n_days: int
    start: pd.Timestamp
    end: pd.Timestamp
    realized_vol: est.VolEstimate
    arch_effects: TestResult
    garch: GarchResult
    ewma_lambda: float
    forecast_table: pd.DataFrame
    vrp: VRPResult | None
    cone: pd.DataFrame


def run_report(
    instrument: str = "BTC-PERPETUAL",
    start: str = "2019-01-01",
    train: int = 750,
    refit_every: int = 21,
) -> Report:
    data = load_daily(instrument, start)
    r = data.returns
    garch = fit_garch(r)
    forecasts = walk_forward_forecasts(r, train=train, refit_every=refit_every)
    table = compare_forecasts(forecasts, data.realized_variance, benchmark="ewma_0.94")
    vrp = variance_risk_premium(data) if data.dvol is not None else None
    last_year = r.iloc[-365:]
    return Report(
        instrument=instrument,
        n_days=len(r),
        start=r.index[0],
        end=r.index[-1],
        realized_vol=est.volatility_ci(last_year, DAYS_PER_YEAR, block_size=10),
        arch_effects=arch_lm(r.to_numpy(), lags=5),
        garch=garch,
        ewma_lambda=est.fit_ewma_lambda(r),
        forecast_table=table,
        vrp=vrp,
        cone=est.volatility_cone(data.ohlc["close"], periods=DAYS_PER_YEAR),
    )
