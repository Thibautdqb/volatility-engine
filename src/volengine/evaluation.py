"""Out-of-sample evaluation of variance forecasts.

The question this module answers is not "which model fits best in-sample" but
"which forecast would have been more accurate, and is the difference larger than
what sampling noise alone could produce".

Protocol
--------
* Walk-forward: parameters are re-estimated every ``refit_every`` periods on data
  available at that date only; forecasts are strictly one step ahead.
* Losses: QLIKE (primary, robust to a noisy proxy) and MSE on variance.
* Significance: Diebold-Mariano with HAC variance against a benchmark, and a
  Mincer-Zarnowitz regression for forecast bias.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from volengine.estimators import ewma_variance, fit_ewma_lambda, qlike
from volengine.garch import fit_garch, garch_filter
from volengine.stats import diebold_mariano, mincer_zarnowitz


def walk_forward_forecasts(
    returns: pd.Series,
    train: int = 500,
    refit_every: int = 21,
    hist_window: int = 21,
    rolling_window: int | None = None,
) -> pd.DataFrame:
    """One-step-ahead variance forecasts for several models.

    Columns: ``historical`` (rolling mean of r^2), ``ewma_0.94`` (RiskMetrics),
    ``ewma_fit`` (decay re-estimated on the training set), ``garch``.
    ``rolling_window`` limits the estimation sample; ``None`` means expanding.
    """
    r = returns.to_numpy(dtype=float)
    n = r.size
    if n <= train + 10:
        raise ValueError(f"need more than {train + 10} observations, got {n}")
    out = {k: np.full(n, np.nan) for k in ("historical", "ewma_0.94", "ewma_fit", "garch")}

    out["historical"] = pd.Series(r**2).rolling(hist_window).mean().shift(1).to_numpy()
    out["ewma_0.94"] = ewma_variance(r, 0.94)

    for k in range(train, n, refit_every):
        start = 0 if rolling_window is None else max(0, k - rolling_window)
        end = min(k + refit_every, n)
        sample = r[start:k]
        lam = fit_ewma_lambda(sample)
        out["ewma_fit"][k:end] = ewma_variance(r[start:end], lam)[k - start :]
        g = fit_garch(sample)
        p = g.params
        path = garch_filter(r[start:end], p["omega"], p["alpha"], p["beta"])
        out["garch"][k:end] = path[k - start : end - start]

    df = pd.DataFrame(out, index=returns.index)
    return df.iloc[train:]


def compare_forecasts(
    forecasts: pd.DataFrame, proxy: pd.Series, benchmark: str = "ewma_0.94"
) -> pd.DataFrame:
    """Loss table with Diebold-Mariano tests against ``benchmark``.

    A negative ``dm_stat`` means the model beats the benchmark; ``dm_p`` < 0.05
    means the difference is statistically significant.
    """
    proxy = proxy.reindex(forecasts.index)
    mask = forecasts.notna().all(axis=1) & proxy.notna() & (proxy > 0)
    f = forecasts[mask]
    p = proxy[mask].to_numpy()

    losses = {m: qlike(p, f[m].to_numpy()) for m in f.columns}
    rows = []
    for m in f.columns:
        mz = mincer_zarnowitz(p, f[m].to_numpy())
        row = {
            "model": m,
            "qlike": losses[m].mean(),
            "mse": np.mean((p - f[m].to_numpy()) ** 2),
            "mz_intercept": mz.intercept,
            "mz_slope": mz.slope,
            "mz_r2": mz.r2,
            "mz_unbiased_p": mz.wald_p_value,
        }
        if m == benchmark:
            row.update(dm_stat=np.nan, dm_p=np.nan)
        else:
            dm = diebold_mariano(losses[m], losses[benchmark])
            row.update(dm_stat=dm.statistic, dm_p=dm.p_value)
        rows.append(row)
    table = pd.DataFrame(rows).set_index("model")
    table.attrs["n"] = int(mask.sum())
    table.attrs["benchmark"] = benchmark
    return table
