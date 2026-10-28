import numpy as np
import pandas as pd
import pytest

from tests.conftest import simulate_trades
from volengine.detector import DetectorConfig
from volengine.study import analyze, forward_ratio

DAY = 86_400


def bursty_vol(seconds, base=0.4, mult=4.0, per_day=10, length=600, seed=0):
    rng = np.random.default_rng(seed)
    vol = np.full(seconds, base)
    for s in rng.integers(0, seconds - length, size=round(per_day * seconds / DAY)):
        vol[s : s + length] = base * mult
    return vol


def test_forward_ratio_uses_only_future_returns():
    rng = np.random.default_rng(0)
    sig = pd.DataFrame({"return": rng.standard_normal(50), "slow_vol": np.full(50, 2.0)})
    f = forward_ratio(sig, 5, bars_per_year=4.0)
    r = sig["return"].to_numpy()
    for t in (0, 10, 44):
        assert f[t] == pytest.approx(np.sqrt(np.mean(r[t + 1 : t + 6] ** 2) * 4.0) / 2.0)
    assert np.isnan(f[45:]).all()


def test_gaussian_constant_volatility_raises_no_alert():
    trades = simulate_trades(2 * DAY, 0.4, rate=1.0, seed=2)
    s = analyze(trades, "SIM", DetectorConfig(), n_boot=100)
    assert not s.signals["alert"].any()
    assert "0 escalations" in s.summary()


def test_isolated_jumps_raise_alerts_that_predict_nothing():
    # Fat tails but no clustering: jumps trip the detector, the next minutes are ordinary.
    trades = simulate_trades(4 * DAY, 0.4, rate=1.0, seed=4, jumps_per_day=30, jump_sd=0.004)
    s = analyze(trades, "SIM", DetectorConfig(), violent=1.3, n_boot=500)
    assert s.signals["alert"].sum() > 40
    assert s.lift_ci[0] < 1.0
    assert s.profile.loc[5, "ci low"] < s.profile.loc[5, "any time"]


def test_flags_predict_volatility_bursts():
    trades = simulate_trades(3 * DAY, bursty_vol(3 * DAY), rate=1.0, seed=3)
    s = analyze(trades, "SIM", DetectorConfig(), n_boot=300)
    assert s.lift_ci[0] > 2.0
    assert s.recall > 0.3
    prof = s.profile
    assert (prof["ci low"] > prof["any time"]).loc[[1, 5]].all()
    assert len(s.episodes) > 10
    assert "lift" in s.summary()
