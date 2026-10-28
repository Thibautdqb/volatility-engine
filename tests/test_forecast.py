import math

import numpy as np
import pandas as pd
import pytest

from tests.conftest import simulate_trades
from tests.test_study import DAY, bursty_vol
from volengine.detector import DetectorConfig
from volengine.forecast import (
    FEATURES,
    HARModel,
    Regime,
    StreamingHAR,
    classify,
    default_model,
    har_inputs,
    prepare,
    season_profile,
)
from volengine.study import forecast_study

CONFIG = DetectorConfig()


def test_classify_uses_cut_points():
    cuts = (0.8, 0.95, 1.07)
    assert [classify(x, cuts) for x in (0.5, 0.8, 0.9, 1.0, 1.07, 3.0)] == [
        Regime.CALM,
        Regime.NORMAL,
        Regime.NORMAL,
        Regime.ACTIVE,
        Regime.STRESSED,
        Regime.STRESSED,
    ]


def test_shipped_models_load():
    btc, eth = default_model("BTC-PERPETUAL"), default_model("ETH-PERPETUAL")
    assert btc.instrument == "BTC-PERPETUAL" and eth.instrument == "ETH-PERPETUAL"
    assert len(btc.coefs) == len(FEATURES) and len(btc.season) == 288
    assert btc.cuts[0] < btc.cuts[1] < btc.cuts[2]
    assert default_model("SOL_USDC-PERPETUAL") == btc
    assert HARModel.from_dict(btc.to_dict()) == btc


def _prepared(r, counts=None, start="2026-01-01"):
    idx = pd.date_range(start, periods=len(r), freq="10s", tz="UTC")
    counts = pd.Series(counts if counts is not None else 3, index=idx)
    return prepare(pd.Series(r, index=idx), counts, CONFIG)


def test_fit_recovers_coefficients():
    rng = np.random.default_rng(0)
    n = 6000
    X = rng.normal(0, 0.3, (n, len(FEATURES)))
    beta = np.array([0.05, 0.3, 0.2, 0.0, 0.1, 0.05])
    w = pd.DataFrame(X, columns=FEATURES).assign(y=-0.1 + X @ beta + rng.normal(0, 0.05, n))
    A = np.column_stack([np.ones(n), w[list(FEATURES)]])
    est = np.linalg.lstsq(A, w["y"], rcond=None)[0]
    assert est == pytest.approx(np.r_[-0.1, beta], abs=0.01)


def test_features_are_past_only_and_target_future_only():
    rng = np.random.default_rng(1)
    r = rng.standard_normal(CONFIG.warmup_bars + 600) * 1e-4
    counts = rng.poisson(3, r.size)
    f = _prepared(r, counts)
    rr, nn = np.asarray(r)[CONFIG.warmup_bars :], counts[CONFIG.warmup_bars :]
    slow = f["slow_vol"].to_numpy() ** 2 / CONFIG.bars_per_year
    t = 300
    x5 = 0.5 * math.log(np.mean(rr[t - 29 : t + 1] ** 2) / slow[t])
    y = 0.5 * math.log(np.mean(rr[t + 1 : t + 31] ** 2) / slow[t])
    assert f["x5"].iloc[t] == pytest.approx(x5)
    assert f["y"].iloc[t] == pytest.approx(y)
    assert f["absmove"].iloc[t] == pytest.approx(
        abs(rr[t - 5 : t + 1].sum()) / math.sqrt(6 * slow[t])
    )
    trades = math.log((nn[t - 5 : t + 1].sum() + 1) / (nn[t - 179 : t + 1].sum() / 30 + 1))
    assert f["trades"].iloc[t] == pytest.approx(trades)
    assert f["y"].iloc[-30:].isna().all()


def test_time_of_day_profile_finds_the_busy_slot():
    rng = np.random.default_rng(3)
    n = CONFIG.warmup_bars + 3 * 8640
    sd = np.full(n, 1e-4)
    idx = pd.date_range("2026-01-01", periods=n, freq="10s", tz="UTC")
    busy = (idx.hour == 13) & (idx.minute >= 30) & (idx.minute < 35)
    sd[busy] *= 3
    prof = season_profile(_prepared(rng.standard_normal(n) * sd))
    assert np.argmax(prof) in (161, 162, 163)  # 13:30 is slot 162
    assert prof[162] > 0.3 and abs(np.median(prof)) < 0.1


def test_streaming_matches_batch():
    rng = np.random.default_rng(2)
    n = CONFIG.warmup_bars + 400
    r = rng.standard_normal(n) * 1e-4
    counts = rng.poisson(3, n)
    f = _prepared(r, counts)
    model = default_model("BTC-PERPETUAL")
    expected = np.exp(model.predict(har_inputs(f, model)))
    s = StreamingHAR(model, CONFIG)
    slow = f["slow_vol"].to_numpy() ** 2 / CONFIG.bars_per_year
    ends = f["t_mid"].to_numpy() - 150
    rr, nn = r[CONFIG.warmup_bars :], counts[CONFIG.warmup_bars :]
    for i in range(len(f)):
        s.update(rr[i], int(nn[i]))
        if i >= 179:
            assert s.forecast(slow[i], ends[i]).ratio == pytest.approx(expected[i], rel=1e-9)


def test_no_forecast_skill_under_constant_volatility():
    train = simulate_trades(4 * DAY, 0.4, rate=1.0, seed=7)
    test = simulate_trades(2 * DAY, 0.4, rate=1.0, seed=8, start="2026-01-05")
    s = forecast_study(train, test, "SIM", n_boot=200)
    assert s.scores.loc["forecast", "R2"] < 0.02


def test_forecast_skill_with_volatility_clusters():
    train = simulate_trades(4 * DAY, bursty_vol(4 * DAY, seed=1), rate=1.0, seed=9)
    test = simulate_trades(
        2 * DAY, bursty_vol(2 * DAY, seed=2), rate=1.0, seed=10, start="2026-01-05"
    )
    s = forecast_study(train, test, "SIM", n_boot=200)
    sc = s.scores["R2"]
    assert sc["forecast"] > 0.1
    assert sc["forecast"] > sc["last 5 minutes"]
    reg = s.regimes["P(violent)"]
    assert reg["stressed"] > reg["calm"]
    assert "fitted on" in s.summary()
