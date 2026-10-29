"""Walk-forward evaluation of the five-minute forecast.

Protocol
--------
The sample is cut into weeks. For each week, a model is fitted on the preceding
28 to 90 days only and used to forecast every bar of the week, exactly as it would
have run live (regimes tracked bar by bar). Nothing from the test week, or later,
enters the fit: coefficients, time-of-day profile, residual distribution and regime
cut points all come from the training window.

All statistics are computed on the out-of-sample five-minute windows pooled across
weeks, and per week to show that the result holds every week, not just on average.
Consecutive windows are dependent (volatility clusters), so standard errors are HAC
with a one-hour lag window and intervals come from a stationary block bootstrap with
one-hour blocks.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from volengine.detector import DetectorConfig, Level
from volengine.features import nbars, windows
from volengine.forecast import (
    DEFAULT_FEATURES,
    VIOLENT,
    Model,
    Regime,
    StressAlarm,
    fit,
    track_regimes,
)
from volengine.stats import (
    auc,
    bootstrap_ci,
    diebold_mariano,
    ols_hac,
    wilson_interval,
)

LAGS = 12  # one hour of five-minute windows
BLOCK = 12.0
LEADS_MINUTES = (0, 1, 2, 5, 10, 15)


@dataclass(frozen=True)
class WalkForwardConfig:
    step_days: int = 7
    min_train_days: int = 28
    max_train_days: int = 90


@dataclass
class WalkForward:
    instrument: str
    features: tuple[str, ...]
    windows: pd.DataFrame  # out-of-sample five-minute windows
    bars: pd.DataFrame  # out-of-sample bars: forecast, regime, detector
    folds: pd.DataFrame
    models: list[Model] = field(repr=False, default_factory=list)


def walk_forward(
    features: pd.DataFrame,
    config: DetectorConfig,
    names=DEFAULT_FEATURES,
    wf: WalkForwardConfig | None = None,
    instrument: str = "",
    train_with: tuple[pd.DataFrame, ...] = (),
    own_history: bool = True,
) -> WalkForward:
    """Refit every ``step_days`` on the past only; forecast the next ``step_days``.

    ``train_with`` adds other instruments' features to each training window (same
    dates), and ``own_history=False`` leaves the tested instrument out of training:
    together they measure how well a model transfers across assets.
    """
    wf = wf or WalkForwardConfig()
    day0 = features.index[0].floor("D") + pd.Timedelta(days=1)  # first full day
    first = day0 + pd.Timedelta(days=wf.min_train_days)
    starts = pd.date_range(first, features.index[-1], freq=f"{wf.step_days}D")
    out_w, out_b, rows, models = [], [], [], []
    for k, a in enumerate(starts):
        b = a + pd.Timedelta(days=wf.step_days)
        lo = max(day0, a - pd.Timedelta(days=wf.max_train_days))

        def window(f: pd.DataFrame, lo=lo, a=a) -> pd.DataFrame:
            return f[(f.index >= lo) & (f.index < a)]

        train = window(features)
        test = features[(features.index >= a) & (features.index < b)].copy()
        if len(test) < 1000:
            continue
        frames = ([train] if own_history else []) + [window(f) for f in train_with]
        model = fit(frames, config, names, instrument)
        models.append(model)
        pred = model.predict(test)
        test["forecast"] = pred
        test["regime"] = track_regimes(np.exp(np.nan_to_num(pred)), model.cuts)
        test["stress"] = model.stress(np.nan_to_num(pred))
        test["fold"] = k
        w = windows(test, config).dropna(subset=["forecast"]).copy()
        w["p_violent"] = model.p_violent(w["forecast"].to_numpy())
        train_w = pd.concat([windows(f, config) for f in frames])
        w["clim_y"] = train_w["y"].mean()
        w["clim_p"] = float((train_w["y"] >= np.log(VIOLENT)).mean())
        out_w.append(w)
        out_b.append(test[["forecast", "regime", "stress", "ratio", "level", "y", "close"]])
        y = w["y"].to_numpy()
        sst = np.sum((y - w["clim_y"].to_numpy()) ** 2)
        rows.append(
            {
                "start": a,
                "train_from": lo,
                "n": len(w),
                "R2": 1 - np.sum((y - w["forecast"]) ** 2) / sst,
                "R2 last 5m": 1 - np.sum((y - w["x5"]) ** 2) / sst,
                **dict(zip(model.features, model.coefs, strict=True)),
            }
        )
    return WalkForward(
        instrument=instrument,
        features=tuple(names),
        windows=pd.concat(out_w),
        bars=pd.concat(out_b),
        folds=pd.DataFrame(rows).set_index("start"),
        models=models,
    )


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


def accuracy(w: pd.DataFrame) -> pd.DataFrame:
    """R2 and Diebold-Mariano tests of the forecast against simple alternatives."""
    y = w["y"].to_numpy()
    clim = w["clim_y"].to_numpy()
    cands = {
        "forecast": w["forecast"].to_numpy(),
        "training mean": clim,
        "last 5 minutes": w["x5"].to_numpy(),
        "last 30 minutes": w["x30"].to_numpy(),
        "spike detector (vol 1m)": np.log(w["ratio"].to_numpy()),
    }
    sst = np.sum((y - clim) ** 2)
    loss = {k: (y - v) ** 2 for k, v in cands.items()}
    rows = []
    for k, v in cands.items():
        dm = diebold_mariano(loss["forecast"], loss[k], LAGS) if k != "forecast" else None
        rows.append(
            {
                "forecast": k,
                "R2": 1 - np.sum((y - v) ** 2) / sst,
                "MSE": float(np.mean((y - v) ** 2)),
                "DM t (forecast vs this)": dm.t if dm else np.nan,
                "p": dm.p_value if dm else np.nan,
            }
        )
    return pd.DataFrame(rows).set_index("forecast")


def mincer_zarnowitz(w: pd.DataFrame) -> dict:
    beta, se = ols_hac(w["y"].to_numpy(), w["forecast"].to_numpy(), LAGS)
    return {"intercept": beta[0], "slope": beta[1], "slope se": se[1]}


def probability(w: pd.DataFrame, n_boot: int = 1000) -> dict:
    """Discrimination (AUC) and calibration (Brier) of P(violent next 5 minutes)."""
    hit = (w["y"] >= np.log(VIOLENT)).to_numpy()
    p = w["p_violent"].to_numpy()
    quiet = w["level"].to_numpy() == Level.NORMAL
    data = np.column_stack([p, hit, np.log(w["ratio"]), quiet])

    def auc_of(d, col=0, mask=None):
        m = np.ones(len(d), bool) if mask is None else d[:, 3].astype(bool)
        return auc(d[m, col], d[m, 1])

    brier = float(np.mean((p - hit) ** 2))
    brier_clim = float(np.mean((w["clim_p"].to_numpy() - hit) ** 2))
    return {
        "base rate": float(hit.mean()),
        "AUC": auc_of(data),
        "AUC ci": bootstrap_ci(data, auc_of, n_boot=n_boot, block_size=BLOCK),
        "AUC spike detector": auc_of(data, col=2),
        "AUC detector quiet": auc_of(data, mask=True),
        "AUC detector quiet ci": bootstrap_ci(
            data, lambda d: auc_of(d, mask=True), n_boot=n_boot, block_size=BLOCK
        ),
        "Brier": brier,
        "Brier skill": 1 - brier / brier_clim if brier_clim > 0 else float("nan"),
    }


def reliability(w: pd.DataFrame, bins=(0, 0.01, 0.02, 0.05, 0.1, 0.2, 1.0)) -> pd.DataFrame:
    """Forecast probability against observed frequency, by probability bin."""
    hit = (w["y"] >= np.log(VIOLENT)).to_numpy()
    p = w["p_violent"].to_numpy()
    cut = pd.cut(p, bins, include_lowest=True)
    rows = []
    for interval, idx in pd.Series(range(len(p))).groupby(cut, observed=True):
        i = idx.to_numpy()
        lo, hi = wilson_interval(int(hit[i].sum()), len(i))
        rows.append(
            {
                "bin": str(interval),
                "n": len(i),
                "mean forecast": p[i].mean(),
                "observed": hit[i].mean(),
                "ci low": lo,
                "ci high": hi,
            }
        )
    return pd.DataFrame(rows).set_index("bin")


def regimes(w: pd.DataFrame, n_boot: int = 1000) -> pd.DataFrame:
    hit = (w["y"] >= np.log(VIOLENT)).astype(float).to_numpy()
    reg = w["regime"].to_numpy()
    rows = []
    for r in Regime:
        m = (reg == r).astype(float)
        sel = m.astype(bool)

        def share(d):
            on = d[:, 0].astype(bool)
            return d[on, 1].mean() if on.any() else np.nan

        ci = bootstrap_ci(np.column_stack([m, hit]), share, n_boot=n_boot, block_size=BLOCK)
        rows.append(
            {
                "regime": r.name,
                "share of time": sel.mean(),
                "realized / normal (median)": float(np.exp(np.median(w["y"][sel]))),
                "P(violent)": hit[sel].mean(),
                "ci low": ci[0],
                "ci high": ci[1],
                "spike detector quiet": float((w["level"][sel] == Level.NORMAL).mean()),
            }
        )
    return pd.DataFrame(rows).set_index("regime")


def anticipation(wfr: WalkForward, config: DetectorConfig, gap_minutes: int = 30) -> pd.DataFrame:
    """How often each signal is already on, some minutes before a violent episode starts.

    An episode starts at a five-minute window that is violent while the previous one was
    not, at least ``gap_minutes`` after the previous episode. For each lead time L, the
    table gives the share of episodes for which the signal was on L minutes before the
    window began, using only information available then, and the share of all time the
    signal is on (its false-alarm cost). The forecast's signal is "regime STRESSED";
    the spike detector's is "level ELEVATED or above".
    """
    b = wfr.bars
    kept = _episodes(wfr.windows, gap_minutes)
    stressed = (b["regime"] == Regime.STRESSED).astype(float)
    detector = (b["level"] >= Level.ELEVATED).astype(float)
    rows = []
    for lead in LEADS_MINUTES:
        # The window starting at t is forecast at the close of the bar starting at t.
        at = kept - pd.Timedelta(minutes=lead)
        s = stressed.reindex(at).dropna()
        d = detector.reindex(at).dropna()
        rows.append(
            {
                "minutes before": lead,
                "episodes": len(s),
                "forecast STRESSED": s.mean(),
                "spike detector on": d.mean(),
            }
        )
    out = pd.DataFrame(rows).set_index("minutes before")
    out.attrs["time on: forecast STRESSED"] = float(stressed.mean())
    out.attrs["time on: spike detector"] = float(detector.mean())
    return out


ALERT_THRESHOLDS = (98.0, 99.0, 99.5)


def _episodes(w: pd.DataFrame, gap_minutes: int = 30) -> pd.DatetimeIndex:
    """Starts of violent episodes: a violent window after a calm one, 30 minutes apart."""
    violent = (w["y"] >= np.log(VIOLENT)).to_numpy()
    onset = w.index[violent & ~np.r_[False, violent[:-1]]]
    kept: list[pd.Timestamp] = []
    for t in onset:
        if not kept or t - kept[-1] >= pd.Timedelta(minutes=gap_minutes):
            kept.append(t)
    return pd.DatetimeIndex(kept)


def alert_study(
    wfr: WalkForward,
    config: DetectorConfig,
    thresholds=ALERT_THRESHOLDS,
    lookahead_minutes: int = 10,
    spikes: bool = True,
) -> pd.DataFrame:
    """What the stress alarm would have sent, run bar by bar over the out-of-sample bars.

    The alarm is the live one (:class:`StressAlarm`), with the spike detector included
    unless ``spikes`` is False.

    * precision: share of alerts followed by a violent five-minute window starting in
      the next ``lookahead_minutes``; ``base`` is the same share at a random bar;
    * caught: share of violent episodes with an alert between 10 minutes before and
      2 minutes after the episode starts;
    * quiet detector: share of alerts sent while the spike detector saw nothing yet.
    """
    b = wfr.bars
    epoch = pd.Timestamp(0, tz="UTC")
    t = np.asarray((b.index - epoch) / pd.Timedelta(seconds=1))
    violent = pd.Series((b["y"] >= np.log(VIOLENT)).astype(float).to_numpy())
    k = nbars(60 * lookahead_minutes, config.bar_seconds)
    ahead = violent[::-1].rolling(k, min_periods=1).max()[::-1].shift(-1).to_numpy()
    episodes = _episodes(wfr.windows)
    days = (b.index[-1] - b.index[0]) / pd.Timedelta(days=1)
    rows = []
    for th in thresholds:
        alarm = StressAlarm(th, spikes=spikes)
        spike = (b["level"] >= Level.SPIKE).to_numpy()
        fired = np.array(
            [
                alarm.update(s, x, k) is not None
                for s, x, k in zip(b["stress"], t, spike, strict=True)
            ]
        )
        at = b.index[fired]
        near = [
            ((at >= e - pd.Timedelta(minutes=10)) & (at <= e + pd.Timedelta(minutes=2))).any()
            for e in episodes
        ]
        rows.append(
            {
                "threshold": th,
                "alerts per day": fired.sum() / days,
                "precision": float(np.nanmean(ahead[fired])),
                "base": float(np.nanmean(ahead)),
                "next 5m vol / normal (median)": float(np.exp(np.nanmedian(b["y"][fired]))),
                "episodes caught": float(np.mean(near)),
                "quiet detector": float((b["level"][fired] == Level.NORMAL).mean()),
            }
        )
    return pd.DataFrame(rows).set_index("threshold")


@dataclass
class Report:
    wf: WalkForward
    accuracy: pd.DataFrame
    mz: dict
    probability: dict
    reliability: pd.DataFrame
    regimes: pd.DataFrame
    anticipation: pd.DataFrame
    alerts: pd.DataFrame

    def summary(self) -> str:
        w, f = self.wf.windows, self.wf.folds
        fmt = lambda x: f"{x:.3f}"  # noqa: E731
        p = self.probability
        a = self.anticipation
        lines = [
            f"{self.wf.instrument}: {len(f)} weekly folds, {len(w):,} out-of-sample windows, "
            f"{w.index[0]:%Y-%m-%d} to {w.index[-1]:%Y-%m-%d}",
            f"inputs: {', '.join(self.wf.features)}",
            "",
            "accuracy of log(vol next 5m / normal), pooled; DM < 0 favours the forecast",
            self.accuracy.to_string(float_format=fmt),
            f"weekly R2: min {f['R2'].min():.3f}, median {f['R2'].median():.3f}, "
            f"positive in {(f['R2'] > 0).sum()} of {len(f)} weeks; beats 'last 5 minutes' "
            f"in {(f['R2'] > f['R2 last 5m']).sum()} of {len(f)}",
            f"Mincer-Zarnowitz slope {self.mz['slope']:.3f} (HAC se {self.mz['slope se']:.3f})",
            "",
            f"P(violent next 5m), violent = vol >= {VIOLENT:g}x normal, base rate "
            f"{p['base rate']:.2%}",
            f"  AUC {p['AUC']:.3f} [{p['AUC ci'][0]:.3f}, {p['AUC ci'][1]:.3f}]   "
            f"spike detector alone {p['AUC spike detector']:.3f}",
            f"  AUC while the spike detector is quiet {p['AUC detector quiet']:.3f} "
            f"[{p['AUC detector quiet ci'][0]:.3f}, {p['AUC detector quiet ci'][1]:.3f}]",
            f"  Brier skill vs training base rate {p['Brier skill']:+.3f}",
            self.reliability.to_string(float_format=fmt),
            "",
            "regimes",
            self.regimes.to_string(float_format=fmt),
            "",
            "anticipation: share of violent episodes already signalled, by lead time",
            a.to_string(float_format=fmt),
            f"  share of all time signalled: forecast STRESSED "
            f"{a.attrs['time on: forecast STRESSED']:.1%}, spike detector "
            f"{a.attrs['time on: spike detector']:.1%}",
            "",
            "stress alarm: stress >= threshold or spike under way; one alert per episode,",
            "re-armed when stress falls below 75, at least 30 minutes apart;",
            "precision = a violent 5-minute window starts within 10 minutes of the alert",
            self.alerts.to_string(float_format=fmt),
        ]
        return "\n".join(lines)


def evaluate(
    features: pd.DataFrame,
    config: DetectorConfig,
    names=DEFAULT_FEATURES,
    wf: WalkForwardConfig | None = None,
    instrument: str = "",
    n_boot: int = 1000,
) -> Report:
    r = walk_forward(features, config, names, wf, instrument)
    w = r.windows
    return Report(
        wf=r,
        accuracy=accuracy(w),
        mz=mincer_zarnowitz(w),
        probability=probability(w, n_boot),
        reliability=reliability(w),
        regimes=regimes(w, n_boot),
        anticipation=anticipation(r, config),
        alerts=alert_study(r, config),
    )
