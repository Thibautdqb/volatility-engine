"""GARCH(1,1) estimated by Gaussian quasi-maximum likelihood.

    r_t = mu + e_t,   e_t = sigma_t z_t
    sigma_t^2 = omega + alpha e_{t-1}^2 + beta sigma_{t-1}^2

Standard errors are the Bollerslev-Wooldridge (1992) sandwich estimator, which
stays valid when z_t is not Gaussian (crypto returns are strongly leptokurtic).
The variance recursion is a linear IIR filter in e^2 and is evaluated with
``scipy.signal.lfilter``, so a fit on several thousand observations takes
milliseconds and rolling re-estimation is cheap.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import optimize, signal, stats

from volengine.stats import TestResult, arch_lm, ljung_box

_LOG_2PI = np.log(2.0 * np.pi)


def _variance_path(eps: np.ndarray, omega: float, alpha: float, beta: float, s0: float):
    """sigma^2_t for t = 0..n (the last element is the one-step-ahead forecast)."""
    n = eps.size
    out = np.empty(n + 1)
    out[0] = s0
    u = omega + alpha * eps**2
    out[1:], _ = signal.lfilter([1.0], [1.0, -beta], u, zi=np.array([beta * s0]))
    return out


def _backcast(eps: np.ndarray) -> float:
    # Exponentially weighted start value, as in Engle & Mezrich / the arch package.
    m = min(75, eps.size)
    w = 0.94 ** np.arange(m)
    return float(w @ eps[:m] ** 2 / w.sum())


def _loglik_obs(theta: np.ndarray, r: np.ndarray, mean: bool, s0: float) -> np.ndarray:
    if mean:
        mu, omega, alpha, beta = theta
    else:
        mu, (omega, alpha, beta) = 0.0, theta
    eps = r - mu
    s2 = _variance_path(eps, omega, alpha, beta, s0)[:-1]
    s2 = np.maximum(s2, 1e-12)
    return -0.5 * (_LOG_2PI + np.log(s2) + eps**2 / s2)


@dataclass
class GarchResult:
    params: pd.Series
    std_err: pd.Series
    loglik: float
    n: int
    conditional_variance: pd.Series
    std_resid: pd.Series
    next_variance: float
    converged: bool
    diagnostics: dict[str, TestResult] = field(default_factory=dict)

    @property
    def tvalues(self) -> pd.Series:
        return self.params / self.std_err

    @property
    def pvalues(self) -> pd.Series:
        return pd.Series(2 * stats.norm.sf(np.abs(self.tvalues)), index=self.params.index)

    @property
    def persistence(self) -> float:
        return float(self.params["alpha"] + self.params["beta"])

    @property
    def half_life(self) -> float:
        """Periods for a variance shock to decay by half."""
        p = self.persistence
        return float(np.log(0.5) / np.log(p)) if 0 < p < 1 else float("inf")

    @property
    def unconditional_variance(self) -> float:
        p = self.persistence
        return float(self.params["omega"] / (1 - p)) if p < 1 else float("inf")

    @property
    def aic(self) -> float:
        return float(2 * len(self.params) - 2 * self.loglik)

    @property
    def bic(self) -> float:
        return float(np.log(self.n) * len(self.params) - 2 * self.loglik)

    def forecast(self, horizon: int) -> np.ndarray:
        """Per-period variance forecasts for steps 1..horizon."""
        k = np.arange(horizon)
        lr = self.unconditional_variance
        return lr + self.persistence**k * (self.next_variance - lr)

    def term_volatility(self, horizon: int, periods: float) -> float:
        """Annualized volatility over the next ``horizon`` periods."""
        return float(np.sqrt(self.forecast(horizon).mean() * periods))

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "estimate": self.params,
                "std_err": self.std_err,
                "t": self.tvalues,
                "p_value": self.pvalues,
            }
        )


def _numerical_scores(fn, theta: np.ndarray) -> np.ndarray:
    k = theta.size
    h = 1e-5 * np.maximum(np.abs(theta), 1e-3)
    cols = []
    for i in range(k):
        e = np.zeros(k)
        e[i] = h[i]
        cols.append((fn(theta + e) - fn(theta - e)) / (2 * h[i]))
    return np.column_stack(cols)


def _numerical_hessian(f, theta: np.ndarray) -> np.ndarray:
    k = theta.size
    h = 1e-4 * np.maximum(np.abs(theta), 1e-2)
    H = np.empty((k, k))
    for i in range(k):
        for j in range(i, k):
            ei = np.zeros(k)
            ej = np.zeros(k)
            ei[i] = h[i]
            ej[j] = h[j]
            H[i, j] = H[j, i] = (
                f(theta + ei + ej) - f(theta + ei - ej) - f(theta - ei + ej) + f(theta - ei - ej)
            ) / (4 * h[i] * h[j])
    return H


def fit_garch(returns: pd.Series | np.ndarray, mean: bool = False) -> GarchResult:
    """Fit a GARCH(1,1).

    Returns are rescaled to unit variance before optimization and parameters are
    mapped back, which keeps the problem well conditioned whatever the sampling
    frequency (1-minute crypto returns have variances around 1e-7).
    """
    index = returns.index if isinstance(returns, pd.Series) else pd.RangeIndex(len(returns))
    r_raw = np.asarray(returns, dtype=float)
    if not np.all(np.isfinite(r_raw)):
        raise ValueError("returns contain NaN or inf")
    scale = 1.0 / r_raw.std()
    r = r_raw * scale
    mu0 = r.mean() if mean else 0.0
    s0 = _backcast(r - mu0)

    def nll(theta):
        return -_loglik_obs(theta, r, mean, s0).sum()

    var = r.var()
    best = None
    for a, b in [(0.05, 0.90), (0.10, 0.85), (0.15, 0.80), (0.03, 0.96), (0.20, 0.70)]:
        x0 = np.array([var * (1 - a - b), a, b])
        if mean:
            x0 = np.r_[mu0, x0]
        bounds = ([(-10 * np.sqrt(var), 10 * np.sqrt(var))] if mean else []) + [
            (1e-8 * var, 10 * var),
            (0.0, 1.0),
            (0.0, 1.0),
        ]
        cons = {"type": "ineq", "fun": lambda t: 0.99999 - t[-2] - t[-1]}
        res = optimize.minimize(
            nll, x0, method="SLSQP", bounds=bounds, constraints=[cons], options={"ftol": 1e-10}
        )
        if best is None or res.fun < best.fun:
            best = res
    theta = best.x

    # Bollerslev-Wooldridge robust covariance: H^-1 J H^-1
    scores = _numerical_scores(lambda t: _loglik_obs(t, r, mean, s0), theta)
    J = scores.T @ scores
    H = -_numerical_hessian(lambda t: -nll(t), theta)
    try:
        H_inv = np.linalg.inv(H)
        cov = H_inv @ J @ H_inv
        se = np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:
        se = np.full(theta.size, np.nan)

    names = (["mu"] if mean else []) + ["omega", "alpha", "beta"]
    unscale = np.array(([1 / scale] if mean else []) + [1 / scale**2, 1.0, 1.0])
    params = pd.Series(theta * unscale, index=names)
    std_err = pd.Series(se * unscale, index=names)

    mu = params["mu"] if mean else 0.0
    eps = r_raw - mu
    s2 = _variance_path(eps, params["omega"], params["alpha"], params["beta"], s0 / scale**2)
    cond = pd.Series(s2[:-1], index=index, name="garch_variance")
    z = pd.Series(eps / np.sqrt(s2[:-1]), index=index, name="std_resid")

    # The log-likelihood is reported on the original scale (Jacobian of the rescaling).
    loglik = float(-best.fun + r.size * np.log(scale))
    result = GarchResult(
        params=params,
        std_err=std_err,
        loglik=loglik,
        n=r.size,
        conditional_variance=cond,
        std_resid=z,
        next_variance=float(s2[-1]),
        converged=bool(best.success),
    )
    result.diagnostics = {
        # Residual ARCH effects left after the fit; 2 dof absorbed by alpha, beta.
        "ljung_box_z2": ljung_box(z.to_numpy() ** 2, lags=10, fitted_params=2),
        "arch_lm_z": arch_lm(z.to_numpy(), lags=5),
        "jarque_bera_z": TestResult(*stats.jarque_bera(z.to_numpy())[:2], 2),
    }
    return result


def garch_filter(
    returns: np.ndarray, omega: float, alpha: float, beta: float, s0: float | None = None
) -> np.ndarray:
    """One-step-ahead variance forecasts with fixed parameters (for out-of-sample use).

    ``out[t]`` uses information up to t-1; ``out[-1]`` forecasts the period after
    the sample.
    """
    r = np.asarray(returns, dtype=float)
    s0 = _backcast(r) if s0 is None else s0
    return _variance_path(r, omega, alpha, beta, s0)
