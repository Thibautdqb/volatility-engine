import numpy as np
import pandas as pd

from tests.conftest import simulate_garch
from volengine.evaluation import compare_forecasts, walk_forward_forecasts


def test_walk_forward_has_no_lookahead():
    r = pd.Series(simulate_garch(800, 2e-6, 0.1, 0.85, seed=3))
    base = walk_forward_forecasts(r, train=500, refit_every=50)
    shocked = r.copy()
    shocked.iloc[650] = 0.5  # a huge shock at t=650
    after = walk_forward_forecasts(shocked, train=500, refit_every=50)
    cols = ["historical", "ewma_0.94", "ewma_fit", "garch"]
    # Forecasts up to and including t=650 must not change.
    pd.testing.assert_frame_equal(base.loc[:650, cols], after.loc[:650, cols])
    assert (after.loc[651, cols] > base.loc[651, cols]).all()


def test_garch_beats_historical_on_garch_data():
    n = 2500
    r = pd.Series(simulate_garch(n, 2e-6, 0.12, 0.85, seed=8))
    rng = np.random.default_rng(0)
    fc = walk_forward_forecasts(r, train=750, refit_every=50)
    # Squared return with extra multiplicative noise: an unbiased but noisy proxy.
    proxy = r**2 * rng.uniform(0.9, 1.1, n)
    table = compare_forecasts(fc, proxy, benchmark="historical")
    assert table.loc["garch", "qlike"] < table.loc["historical", "qlike"]
    assert table.loc["garch", "dm_stat"] < 0
    assert table.loc["garch", "dm_p"] < 0.05
    assert table.attrs["n"] > 1500
