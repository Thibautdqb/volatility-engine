import math

import numpy as np
import pandas as pd
import pytest

from volengine.detector import DetectorConfig, Level, SpikeDetector, next_level, run_detector


def test_config_validation():
    with pytest.raises(ValueError):
        DetectorConfig(fast_half_life=600, slow_half_life=60)
    with pytest.raises(ValueError):
        DetectorConfig(elevated=3.0, spike=2.0)
    c = DetectorConfig(bar_seconds=10, fast_half_life=60)
    assert c.lam_fast**6 == pytest.approx(0.5)


def test_ewma_recursion():
    c = DetectorConfig()
    d = SpikeDetector(c, variance=1e-8)
    rd = d.update(1e-3)
    assert rd.fast_var == pytest.approx(c.lam_fast * 1e-8 + (1 - c.lam_fast) * 1e-6)
    assert rd.slow_var == pytest.approx(c.lam_slow * 1e-8 + (1 - c.lam_slow) * 1e-6)
    assert rd.ratio == pytest.approx(math.sqrt(rd.fast_var / rd.slow_var))


def test_levels_and_hysteresis():
    c = DetectorConfig()
    N, E, S = Level.NORMAL, Level.ELEVATED, Level.SPIKE
    level, levels = N, []
    for ratio in [1.0, 1.8, 2.1, 1.8, 3.5, 2.5, 1.9, 1.6, 1.4, 1.8, 2.0, 1.0]:
        level = next_level(level, ratio, c)
        levels.append(level)
    assert levels == [N, N, E, E, S, S, E, E, N, N, E, N]


def test_quiet_market_rarely_alerts():
    # Constant volatility: flags are false alarms. Gaussian 10-second returns, 10 days.
    c = DetectorConfig()
    n = 10 * 8640
    r = np.random.default_rng(0).standard_normal(n) * 1e-4
    sig = run_detector(pd.Series(r), c)
    per_day = sig["alert"].sum() / (len(sig) / 8640)
    assert per_day < 1.0
    assert (sig["level"] == Level.SPIKE).sum() == 0


def test_detects_volatility_jump_within_a_minute():
    c = DetectorConfig()
    rng = np.random.default_rng(1)
    calm, n_calm = 1e-4, c.warmup_bars + 600
    r = np.r_[rng.standard_normal(n_calm) * calm, rng.standard_normal(60) * 4 * calm]
    sig = run_detector(pd.Series(r), c)
    after = sig.iloc[-60:]
    first = np.flatnonzero(after["level"] >= Level.ELEVATED)[0]
    assert first <= 6  # six 10-second bars
    assert after["level"].iloc[:18].max() == Level.SPIKE  # within three minutes


def test_run_detector_drops_warmup_and_annualizes():
    c = DetectorConfig()
    r = np.full(c.warmup_bars + 10, 1e-4)
    sig = run_detector(pd.Series(r), c)
    assert len(sig) == 10
    assert sig["slow_vol"].iloc[-1] == pytest.approx(1e-4 * math.sqrt(c.bars_per_year))
    assert sig["ratio"].to_numpy() == pytest.approx(1.0)
