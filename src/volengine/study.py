"""Historical evaluation on the Deribit trade tape: spike detector and 5-minute forecast.

The detector is replayed bar by bar over past trades, exactly as it runs live, and
two questions are answered:

1. **Does a flag say something about the next minutes?** The sample is cut into
   non-overlapping windows (5 minutes by default). At the start of each window we note
   the detector's level, using only past data, and then measure the realized volatility
   over the window relative to the slow (normal) volatility. A window is *violent* when
   that forward ratio reaches ``violent`` (2 by default). Comparing the share of violent
   windows with and without a flag gives the flag's lift.
2. **How long does a spike last?** For each new episode (a flag raised from the normal
   level), the forward ratio is measured over horizons from 1 to 60 minutes.

Windows next to each other are not independent (volatility clusters), so intervals
come from a stationary block bootstrap with one-hour blocks; episodes are at least
30 minutes apart and resampled as independent draws.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from volengine.deribit.rest import TRADE_COLUMNS, DeribitREST
from volengine.detector import DetectorConfig, Level, run_detector
from volengine.forecast import (
    FEATURES,
    HARModel,
    Regime,
    fit,
    har_inputs,
    prepare,
    sample_windows,
    track_regimes,
)
from volengine.sampling import bar_closes, bar_counts, log_returns
from volengine.stats import bootstrap_ci

log = logging.getLogger(__name__)

CACHE_DIR = Path(os.environ.get("VOLENGINE_CACHE", Path.home() / ".cache" / "volengine"))
PROFILE_MINUTES = (1, 2, 5, 10, 15, 30, 60)


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------


def _utc(ts) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def load_trades(
    instrument: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    cache_dir: Path = CACHE_DIR,
    workers: int = 4,
) -> pd.DataFrame:
    """Trades for the UTC days in [start, end), downloaded once and cached per day."""
    start, end = _utc(start).floor("D"), _utc(end).floor("D")
    days = pd.date_range(start, end, freq="D", inclusive="left")
    folder = Path(cache_dir) / "trades-v2" / instrument
    folder.mkdir(parents=True, exist_ok=True)
    now = pd.Timestamp.now(tz="UTC")

    with DeribitREST() as api:

        def one(day: pd.Timestamp) -> pd.DataFrame:
            path = folder / f"{day:%Y-%m-%d}.csv.gz"
            if path.exists():
                return pd.read_csv(
                    path, dtype={"trade_id": str, "liquidation": str}, keep_default_na=False
                )
            df = api.trades(instrument, day, day + pd.Timedelta(days=1))
            if day + pd.Timedelta(days=1) <= now:  # only complete days are cached
                tmp = path.with_suffix(".tmp")
                df.to_csv(tmp, index=False, compression="gzip")
                tmp.replace(path)  # atomic: an interrupted run never leaves a truncated file
            log.info("%s %s: %d trades", instrument, day.date(), len(df))
            return df

        with ThreadPoolExecutor(workers) as pool:
            frames = list(pool.map(one, days))
    frames = [f for f in frames if len(f)]
    if not frames:
        return pd.DataFrame(columns=TRADE_COLUMNS)
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------


def forward_ratio(signals: pd.DataFrame, horizon_bars: int, bars_per_year: float) -> np.ndarray:
    """Realized volatility over the next ``horizon_bars`` bars divided by the slow volatility.

    The value at bar t uses returns t+1..t+h only, so it is never known at t. NaN where
    the window runs past the end of the sample.
    """
    r2 = signals["return"].to_numpy() ** 2
    csum = np.concatenate([[0.0], np.cumsum(r2)])
    n = r2.size
    out = np.full(n, np.nan)
    t = np.arange(n - horizon_bars)
    future_var = (csum[t + 1 + horizon_bars] - csum[t + 1]) / horizon_bars * bars_per_year
    out[t] = np.sqrt(future_var) / signals["slow_vol"].to_numpy()[t]
    return out


def _share(cond: np.ndarray, outcome: np.ndarray) -> float:
    m = cond.astype(bool)
    return float(outcome[m].mean()) if m.any() else np.nan


@dataclass(frozen=True)
class Study:
    instrument: str
    config: DetectorConfig
    start: pd.Timestamp
    end: pd.Timestamp
    n_trades: int
    empty_share: float
    signals: pd.DataFrame
    episodes: pd.DataFrame
    by_level: pd.DataFrame
    lift: float
    lift_ci: tuple[float, float]
    recall: float
    recall_ci: tuple[float, float]
    profile: pd.DataFrame
    horizon_minutes: int
    violent: float

    @property
    def days(self) -> float:
        return (self.end - self.start) / pd.Timedelta(days=1)

    def summary(self) -> str:
        c = self.config
        alerts = self.signals[self.signals["alert"]]
        n_elev = int((alerts["level"] >= Level.ELEVATED).sum())
        n_spike = int((alerts["level"] == Level.SPIKE).sum())
        h = self.horizon_minutes
        lines = [
            f"{self.instrument}  {self.start:%Y-%m-%d} to {self.end:%Y-%m-%d} "
            f"({self.days:.0f} days, {self.n_trades:,} trades)",
            f"{c.bar_seconds:g}s bars ({self.empty_share:.0%} without a trade), "
            f"fast half-life {c.fast_half_life:g}s, slow half-life {c.slow_half_life:g}s",
            f"thresholds: elevated {c.elevated:g}x, spike {c.spike:g}x, reset {c.reset:g}x",
            "",
            f"alerts: {n_elev} escalations ({n_elev / self.days:.1f}/day), "
            f"of which {n_spike} to spike ({n_spike / self.days:.1f}/day)",
            "",
            f"next {h} minutes, by detector level at the start of the window "
            f"(violent = forward vol >= {self.violent:g}x normal):",
            self.by_level.to_string(float_format=lambda x: f"{x:.3f}"),
            "",
            f"lift P(violent | flag) / P(violent | no flag): {self.lift:.1f} "
            f"[{self.lift_ci[0]:.1f}, {self.lift_ci[1]:.1f}]",
            f"violent windows flagged at their start: {self.recall:.0%} "
            f"[{self.recall_ci[0]:.0%}, {self.recall_ci[1]:.0%}]",
            "",
            f"forward vol / normal after a new episode ({len(self.episodes)} episodes):",
            self.profile.to_string(float_format=lambda x: f"{x:.2f}"),
        ]
        return "\n".join(lines)


def analyze(
    trades: pd.DataFrame,
    instrument: str,
    config: DetectorConfig | None = None,
    horizon_minutes: int = 5,
    violent: float = 2.0,
    episode_gap_minutes: int = 30,
    n_boot: int = 2000,
) -> Study:
    config = config or DetectorConfig()
    closes = bar_closes(trades, config.bar_seconds)
    returns = log_returns(closes)
    signals = run_detector(returns, config)
    bars_per_min = 60.0 / config.bar_seconds

    # 1. Non-overlapping windows: level at the start, outcome over the window.
    h = round(horizon_minutes * bars_per_min)
    fwd = forward_ratio(signals, h, config.bars_per_year)
    grid = np.arange(0, len(signals), h)
    grid = grid[np.isfinite(fwd[grid])]
    level = signals["level"].to_numpy()[grid]
    hit = (fwd[grid] >= violent).astype(float)
    flag = (level >= Level.ELEVATED).astype(float)
    block = 60.0 / horizon_minutes  # one-hour blocks
    rows = []
    for lv in Level:
        m = level == lv
        data = np.column_stack([m, hit])
        ci = bootstrap_ci(data, lambda d: _share(d[:, 0], d[:, 1]), n_boot=n_boot, block_size=block)
        rows.append(
            {
                "level": lv.name.lower(),
                "windows": int(m.sum()),
                "share of time": m.mean(),
                "P(violent)": _share(m, hit),
                "ci low": ci[0],
                "ci high": ci[1],
                "median fwd ratio": float(np.median(fwd[grid][m])) if m.any() else np.nan,
            }
        )
    by_level = pd.DataFrame(rows).set_index("level")

    pair = np.column_stack([flag, hit])

    def lift_of(d: np.ndarray) -> float:
        off = _share(1 - d[:, 0], d[:, 1])
        return _share(d[:, 0], d[:, 1]) / off if off > 0 else np.nan

    def recall_of(d: np.ndarray) -> float:
        return _share(d[:, 1], d[:, 0])

    lift = lift_of(pair)
    lift_ci = bootstrap_ci(pair, lift_of, n_boot=n_boot, block_size=block)
    recall = recall_of(pair)
    recall_ci = bootstrap_ci(pair, recall_of, n_boot=n_boot, block_size=block)

    # 2. Episodes: raised from normal, at least `episode_gap_minutes` apart.
    lvl = signals["level"].to_numpy()
    onset = np.flatnonzero(signals["alert"].to_numpy() & (np.r_[0, lvl[:-1]] == Level.NORMAL))
    kept: list[int] = []
    gap = episode_gap_minutes * bars_per_min
    for i in onset:
        if not kept or i - kept[-1] >= gap:
            kept.append(int(i))
    episodes = signals.iloc[kept][["ratio", "fast_vol", "slow_vol"]].copy()
    prof = []
    for m in PROFILE_MINUTES:
        f = forward_ratio(signals, round(m * bars_per_min), config.bars_per_year)
        ev = f[kept] if kept else np.array([])
        ev = ev[np.isfinite(ev)]
        lo, hi = bootstrap_ci(ev, np.median, n_boot=n_boot) if ev.size > 5 else (np.nan, np.nan)
        prof.append(
            {
                "minutes": m,
                "after episode start": float(np.median(ev)) if ev.size else np.nan,
                "ci low": lo,
                "ci high": hi,
                "any time": float(np.nanmedian(f)),
            }
        )
    profile = pd.DataFrame(prof).set_index("minutes")

    ts = trades["timestamp"]
    return Study(
        instrument=instrument,
        config=config,
        start=pd.Timestamp(int(ts.iloc[0]), unit="ms", tz="UTC").floor("D"),
        end=pd.Timestamp(int(ts.iloc[-1]), unit="ms", tz="UTC").ceil("D"),
        n_trades=len(trades),
        empty_share=1.0 - ts.floordiv(round(config.bar_seconds * 1000)).nunique() / len(closes),
        signals=signals,
        episodes=episodes,
        by_level=by_level,
        lift=lift,
        lift_ci=lift_ci,
        recall=recall,
        recall_ci=recall_ci,
        profile=profile,
        horizon_minutes=horizon_minutes,
        violent=violent,
    )


def run_study(
    instrument: str,
    days: int = 7,
    end: pd.Timestamp | None = None,
    config: DetectorConfig | None = None,
    **kwargs,
) -> Study:
    """Download (or read from cache) the last ``days`` complete UTC days and analyze them."""
    end = (_utc(end) if end is not None else pd.Timestamp.now(tz="UTC")).floor("D")
    trades = load_trades(instrument, end - pd.Timedelta(days=days), end)
    if trades.empty:
        raise RuntimeError(f"no trades for {instrument}")
    return analyze(trades, instrument, config, **kwargs)


# ---------------------------------------------------------------------------
# five-minute forecast and regimes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ForecastStudy:
    instrument: str
    model: HARModel
    test_span: str
    n_test: int
    scores: pd.DataFrame
    gain_ci: tuple[float, float]
    mz_slope: float
    regimes: pd.DataFrame
    violent: float

    def summary(self) -> str:
        m = self.model
        coefs = "  ".join(f"{f} {c:+.3f}" for f, c in zip(FEATURES, m.coefs, strict=True))
        lines = [
            f"{self.instrument}  fitted on {m.trained_on} ({m.n} windows), "
            f"tested on {self.test_span} ({self.n_test} windows)",
            f"  log(vol next 5m / normal) = {m.intercept:+.3f}  {coefs}",
            "  regime cut points on the forecast ratio: " + " / ".join(f"{c:.2f}" for c in m.cuts),
            "",
            "out-of-sample accuracy of log(vol next 5m / normal):",
            self.scores.to_string(float_format=lambda x: f"{x:.3f}"),
            f"MSE reduction vs 'last 5 minutes': 95% CI [{self.gain_ci[0]:.4f}, "
            f"{self.gain_ci[1]:.4f}]   Mincer-Zarnowitz slope {self.mz_slope:.2f}",
            "",
            f"regimes (violent = next 5m vol >= {self.violent:g}x normal):",
            self.regimes.to_string(float_format=lambda x: f"{x:.3f}"),
        ]
        return "\n".join(lines)


def prepare_trades(trades: pd.DataFrame, config: DetectorConfig) -> pd.DataFrame:
    closes = bar_closes(trades, config.bar_seconds)
    return prepare(log_returns(closes), bar_counts(trades, closes, config.bar_seconds), config)


def forecast_study(
    train: pd.DataFrame,
    test: pd.DataFrame,
    instrument: str,
    config: DetectorConfig | None = None,
    violent: float = 2.0,
    n_boot: int = 2000,
) -> ForecastStudy:
    """Fit the HAR forecast on ``train`` trades and score it on later ``test`` trades."""
    config = config or DetectorConfig()
    model = fit(prepare_trades(train, config), config, instrument)
    bars = prepare_trades(test, config)
    # The regime is tracked bar by bar, exactly as live, then read at each window start.
    bars["regime"] = track_regimes(np.exp(model.predict(har_inputs(bars, model))), model.cuts)
    te = sample_windows(bars, model.season, config)
    y = te["y"].to_numpy()
    candidates = {
        "normal level (no change)": np.zeros(len(te)),
        "last 5 minutes": te["x5"].to_numpy(),
        "spike detector (vol 1m)": np.log(te["ratio"].to_numpy()),
        "forecast": model.predict(har_inputs(te, model)),
    }
    sst = np.sum((y - y.mean()) ** 2)
    scores = pd.DataFrame(
        {
            "R2": [1 - np.sum((y - f) ** 2) / sst for f in candidates.values()],
            "MSE": [np.mean((y - f) ** 2) for f in candidates.values()],
        },
        index=pd.Index(list(candidates), name="forecast"),
    )
    pred = candidates["forecast"]
    gain = (y - candidates["last 5 minutes"]) ** 2 - (y - pred) ** 2
    gain_ci = bootstrap_ci(gain, n_boot=n_boot, block_size=12)
    slope = float(np.polyfit(pred, y, 1)[0])

    regime = te["regime"].to_numpy()
    hit = (y >= np.log(violent)).astype(float)
    rows = []
    for rg in Regime:
        m = (regime == rg).astype(float)
        ci = bootstrap_ci(
            np.column_stack([m, hit]),
            lambda d: _share(d[:, 0], d[:, 1]),
            n_boot=n_boot,
            block_size=12,
        )
        sel = m.astype(bool)
        rows.append(
            {
                "regime": rg.name.lower(),
                "share": sel.mean(),
                "realized ratio (median)": float(np.exp(np.median(y[sel])))
                if sel.any()
                else np.nan,
                "P(violent)": _share(m, hit),
                "ci low": ci[0],
                "ci high": ci[1],
                "spike detector off": float((te["level"].to_numpy()[sel] == 0).mean())
                if sel.any()
                else np.nan,
            }
        )
    span = f"{te.index[0]:%Y-%m-%d} to {te.index[-1]:%Y-%m-%d}"
    return ForecastStudy(
        instrument=instrument,
        model=model,
        test_span=span,
        n_test=len(te),
        scores=scores,
        gain_ci=gain_ci,
        mz_slope=slope,
        regimes=pd.DataFrame(rows).set_index("regime"),
        violent=violent,
    )


def run_forecast_study(
    instrument: str,
    train_days: int = 30,
    test_days: int = 14,
    end: pd.Timestamp | None = None,
    config: DetectorConfig | None = None,
    **kwargs,
) -> ForecastStudy:
    """Train on ``train_days`` complete UTC days, test on the ``test_days`` that follow."""
    end = (_utc(end) if end is not None else pd.Timestamp.now(tz="UTC")).floor("D")
    split = end - pd.Timedelta(days=test_days)
    train = load_trades(instrument, split - pd.Timedelta(days=train_days), split)
    test = load_trades(instrument, split, end)
    return forecast_study(train, test, instrument, config, **kwargs)


def fit_model(
    instrument: str, days: int = 44, end: pd.Timestamp | None = None, config=None
) -> HARModel:
    end = (_utc(end) if end is not None else pd.Timestamp.now(tz="UTC")).floor("D")
    trades = load_trades(instrument, end - pd.Timedelta(days=days), end)
    config = config or DetectorConfig()
    return fit(prepare_trades(trades, config), config, instrument)
