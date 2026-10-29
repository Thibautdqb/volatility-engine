import json
import math

import numpy as np
import pandas as pd
import pytest

from tests.conftest import DAY, bursty_vol, simulate_trades
from volengine.data.bars import build_bars
from volengine.detector import DetectorConfig
from volengine.evaluation import (
    WalkForwardConfig,
    anticipation,
    evaluate,
    walk_forward,
)
from volengine.features import ALL, StreamingFeatures, _weekend, compute, windows
from volengine.forecast import (
    DEFAULT_FEATURES,
    Model,
    Regime,
    RegimeTracker,
    StreamingForecaster,
    default_model,
    fit,
    season_profile,
)

CONFIG = DetectorConfig()


def grid(trades: pd.DataFrame) -> pd.DataFrame:
    """Bars on a full 10-second grid, as load_bars returns them."""
    b = build_bars(trades, 10)
    idx = pd.date_range(b.index[0], b.index[-1], freq="10s")
    b = b.reindex(idx)
    b[["close", "index"]] = b[["close", "index"]].ffill()
    return b.fillna(0.0)


@pytest.fixture(scope="module")
def two_days():
    return grid(simulate_trades(2 * DAY, bursty_vol(2 * DAY), rate=1.0, seed=3))


# -- features -----------------------------------------------------------------


def test_weekend_flag():
    days = pd.date_range("2026-10-19 12:00", periods=7, freq="D", tz="UTC")  # Mon..Sun
    t = (days - pd.Timestamp(0, tz="UTC")) / pd.Timedelta(seconds=1)
    assert _weekend(np.asarray(t)).tolist() == [0, 0, 0, 0, 0, 1, 1]


def test_inputs_use_the_past_and_the_target_the_future(two_days):
    f = compute(two_days, CONFIG)
    r = np.log(two_days["close"]).diff().reindex(f.index).to_numpy()
    s = f["slow_vol"].to_numpy() ** 2 / CONFIG.bars_per_year
    t = 5000
    assert f["x5"].iloc[t] == pytest.approx(0.5 * math.log(np.mean(r[t - 29 : t + 1] ** 2) / s[t]))
    assert f["y"].iloc[t] == pytest.approx(0.5 * math.log(np.mean(r[t + 1 : t + 31] ** 2) / s[t]))
    assert f["absmove"].iloc[t] == pytest.approx(abs(r[t - 5 : t + 1].sum()) / math.sqrt(6 * s[t]))
    assert f["y"].iloc[-30:].isna().all()
    # Changing the future does not change any input at t.
    shocked = two_days.copy()
    shocked.iloc[t + 2 + CONFIG.warmup_bars :, shocked.columns.get_loc("close")] *= 1.05
    g = compute(shocked, CONFIG)
    cols = ["x1", "x5", "x30", "x4h", "trades", "absmove", "imbalance", "liq", "slow_vol"]
    pd.testing.assert_series_equal(g[cols].iloc[t], f[cols].iloc[t])


def test_streaming_features_equal_batch(two_days):
    f = compute(two_days, CONFIG)
    sf = StreamingFeatures(CONFIG, ALL)
    b = two_days.reindex(f.index)
    r = np.log(two_days["close"]).diff().reindex(f.index).to_numpy()
    s = f["slow_vol"].to_numpy() ** 2 / CONFIG.bars_per_year
    ends = f["t_mid"].to_numpy() - 150
    cols = ["x1", "x5", "x30", "x4h", "trades", "absmove", "imbalance", "liq", "weekend"]
    checked = 0
    for i in range(len(f)):
        row = b.iloc[i]
        sf.update(r[i], int(row["n"]), row["volume"], row["buy_volume"], row["liq_volume"])
        if i >= 1440 and i % 97 == 0:
            got = sf.values(s[i], ends[i])
            for c in cols:
                assert got[c] == pytest.approx(f[c].iloc[i], rel=1e-9, abs=1e-12), c
            checked += 1
    assert checked > 50


# -- model --------------------------------------------------------------------


def test_fit_recovers_known_coefficients():
    rng = np.random.default_rng(0)
    n = 30 * 6000
    idx = pd.date_range("2026-01-01", periods=n, freq="10s", tz="UTC")
    f = pd.DataFrame(rng.normal(0, 0.3, (n, 3)), index=idx, columns=["x1", "x5", "x30"])
    f["y"] = -0.1 + f @ np.array([0.05, 0.3, 0.2]) + rng.normal(0, 0.05, n)
    f["t_mid"] = (idx - pd.Timestamp(0, tz="UTC")) / pd.Timedelta(seconds=1) + 160
    f["r2n"] = 1.0
    m = fit(f, CONFIG, ["x1", "x5", "x30"])
    assert [m.intercept, *m.coefs] == pytest.approx([-0.1, 0.05, 0.3, 0.2], abs=0.01)
    assert m.n == len(windows(f, CONFIG))


def test_pooled_fit_cuts_windows_within_each_instrument(two_days):
    f = compute(two_days, CONFIG)
    one = fit(f, CONFIG, ["x5"])
    two = fit([f, f], CONFIG, ["x5"])
    assert two.n == 2 * one.n
    assert two.coefs == pytest.approx(one.coefs, abs=1e-6)


def test_season_profile_finds_the_busy_slot():
    rng = np.random.default_rng(3)
    idx = pd.date_range("2026-01-01", periods=3 * 8640, freq="10s", tz="UTC")
    busy = (idx.hour == 13) & (idx.minute >= 30) & (idx.minute < 35)
    f = pd.DataFrame({"r2n": rng.chisquare(1, idx.size) * np.where(busy, 9.0, 1.0)}, index=idx)
    prof = season_profile(f)
    assert np.argmax(prof) in (161, 162, 163)  # 13:30 is slot 162
    assert prof[162] > 0.5 and abs(np.median(prof)) < 0.1


def test_regime_hysteresis():
    t = RegimeTracker((0.8, 1.0, 1.2), hold=3)
    seq = [t.update(x) for x in (0.9, 1.3, 1.1, 1.1, 1.1, 0.5, 0.5, 0.5)]
    N, A, S, C = Regime.NORMAL, Regime.ACTIVE, Regime.STRESSED, Regime.CALM
    assert seq == [N, S, S, S, A, A, A, C]


def test_probability_is_calibrated_in_sample():
    rng = np.random.default_rng(1)
    n = 60 * 8640
    idx = pd.date_range("2026-01-01", periods=n, freq="10s", tz="UTC")
    x = rng.normal(0, 0.4, n)
    f = pd.DataFrame({"x5": x, "y": x + rng.normal(0, 0.45, n)}, index=idx)
    f["t_mid"] = (idx - pd.Timestamp(0, tz="UTC")) / pd.Timedelta(seconds=1) + 160
    f["r2n"] = 1.0
    m = fit(f, CONFIG, ["x5"])
    w = windows(f, CONFIG)
    p = m.p_violent(m.predict(w))
    hit = w["y"] >= math.log(2)
    assert p.mean() == pytest.approx(hit.mean(), abs=0.003)
    top = p > np.quantile(p, 0.9)
    assert p[top].mean() == pytest.approx(hit[top].mean(), abs=0.02)


def test_model_round_trips_through_json_and_ships_a_pooled_default():
    m = default_model("BTC-PERPETUAL")
    assert m.features == DEFAULT_FEATURES and len(m.season) == 288
    assert Model.from_dict(json.loads(json.dumps(m.to_dict()))) == m
    assert default_model("SOL_USDC-PERPETUAL").instrument == "pooled"


def test_streaming_forecaster_equals_batch(two_days):
    f = compute(two_days, CONFIG)
    model = default_model("BTC-PERPETUAL")
    sf = StreamingForecaster(model, CONFIG)
    b = two_days.reindex(f.index)
    r = np.log(two_days["close"]).diff().reindex(f.index).to_numpy()
    s = f["slow_vol"].to_numpy() ** 2 / CONFIG.bars_per_year
    pred = model.predict(f)
    for i in range(len(f)):
        row = b.iloc[i]
        sf.update(r[i], int(row["n"]), row["volume"], row["buy_volume"], row["liq_volume"])
        fc = sf.forecast(s[i], f["t_mid"].iloc[i] - 150)
        if i >= 8640 and i % 50 == 0:
            assert math.log(fc.ratio) == pytest.approx(pred[i], rel=1e-9)
            assert fc.p_violent == pytest.approx(model.p_violent(pred[i]), rel=1e-9)


# -- walk-forward -------------------------------------------------------------

WF = WalkForwardConfig(step_days=1, min_train_days=3, max_train_days=5)
NAMES = ("x1", "x5", "x30", "season", "trades", "absmove")


def test_walk_forward_never_trains_on_the_future(monkeypatch):
    import volengine.evaluation as ev

    f = compute(grid(simulate_trades(7 * DAY, 0.5, rate=0.3, seed=5)), CONFIG)
    seen = []
    real_fit = ev.fit

    def spy(frames, *a, **k):
        seen.append(max(fr.index.max() for fr in frames))
        return real_fit(frames, *a, **k)

    monkeypatch.setattr(ev, "fit", spy)
    r = walk_forward(f, CONFIG, NAMES, WF)
    assert len(seen) == len(r.folds) > 0
    for last_train, start in zip(seen, r.folds.index, strict=True):
        assert last_train < start
    fold_start = r.folds.index[r.windows["fold"].to_numpy() - r.windows["fold"].min()]
    assert (r.windows.index >= fold_start).all()


def test_no_skill_without_volatility_clustering():
    # Fat tails from independent jumps, but no memory: nothing to forecast.
    tape = simulate_trades(8 * DAY, 0.5, rate=0.5, seed=6, jumps_per_day=150, jump_sd=0.003)
    rep = evaluate(compute(grid(tape), CONFIG), CONFIG, NAMES, WF, n_boot=100)
    assert rep.accuracy.loc["forecast", "R2"] < 0.02
    assert rep.probability["base rate"] > 0.01
    lo = rep.probability["AUC ci"][0]
    assert lo < 0.5 + 0.03 and abs(rep.probability["AUC"] - 0.5) < 0.06


def test_skill_with_volatility_clusters():
    vol = bursty_vol(8 * DAY, seed=2)
    f = compute(grid(simulate_trades(8 * DAY, vol, rate=0.5, seed=7)), CONFIG)
    rep = evaluate(f, CONFIG, NAMES, WF, n_boot=100)
    acc = rep.accuracy
    assert acc.loc["forecast", "R2"] > 0.1
    assert acc.loc["last 5 minutes", "DM t (forecast vs this)"] < -2
    assert rep.probability["AUC"] > 0.6
    reg = rep.regimes["P(violent)"]
    assert reg["STRESSED"] > reg["NORMAL"]
    assert "weekly folds" in rep.summary()
    a = anticipation(rep.wf, CONFIG)
    assert a.loc[0, "episodes"] > 0


# -- stress and alarm -----------------------------------------------------------


def test_stress_is_the_training_percentile():
    m = default_model("BTC-PERPETUAL")
    lv = np.asarray(m.levels)
    assert np.all(np.diff(lv) >= 0)
    assert m.stress(lv[0] - 1) == 0 and m.stress(lv[-1] + 1) == 100
    # The STRESSED cut point sits at stress 95.
    assert m.stress(math.log(m.cuts[2])) == pytest.approx(95, abs=0.5)
    assert m.stress(math.log(m.cuts[0])) == pytest.approx(25, abs=0.5)


def test_alarm_fires_once_per_episode():
    from volengine.forecast import StressAlarm

    a = StressAlarm(threshold=99, rearm=75, gap_seconds=1800)
    seq = [(50, 0), (99.5, 10), (99.9, 20), (90, 30), (99.5, 40)]  # never calmed down
    assert [a.update(s, t) for s, t in seq] == [None, "forecast", None, None, None]
    assert a.update(70, 50) is None  # calm again: re-armed, but too soon
    assert a.update(99.5, 1000) is None
    assert a.update(60, 1900) is None
    assert a.update(99.2, 1910) == "forecast"
    b = StressAlarm(threshold=99)
    assert b.update(60, 0, spike=True) == "spike"
    assert StressAlarm(threshold=99, spikes=False).update(60, 0, spike=True) is None
    with pytest.raises(ValueError):
        StressAlarm(threshold=50, rearm=75)
