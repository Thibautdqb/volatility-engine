import numpy as np

from volengine.stats import bootstrap_ci, stationary_bootstrap_indices, wilson_interval


def test_wilson_coverage():
    rng = np.random.default_rng(0)
    p, n = 0.05, 80
    intervals = (wilson_interval(rng.binomial(n, p), n) for _ in range(2000))
    hits = [lo <= p <= hi for lo, hi in intervals]
    assert 0.92 < np.mean(hits) < 0.98


def test_stationary_bootstrap_keeps_blocks():
    idx = stationary_bootstrap_indices(1000, 20, 50, np.random.default_rng(0))
    consecutive = np.mean(np.diff(idx, axis=1) == 1)
    assert 0.93 < consecutive < 0.97  # restart probability 1/20


def test_block_bootstrap_covers_mean_of_dependent_series():
    # AR(1) with phi = 0.8: the iid bootstrap is far too narrow, the block bootstrap is not.
    rng = np.random.default_rng(1)
    cover_iid, cover_block = [], []
    for _ in range(150):
        e = rng.standard_normal(400)
        x = np.empty(400)
        x[0] = e[0]
        for t in range(1, 400):
            x[t] = 0.8 * x[t - 1] + e[t]
        lo, hi = bootstrap_ci(x, n_boot=300, seed=None)
        cover_iid.append(lo <= 0 <= hi)
        lo, hi = bootstrap_ci(x, n_boot=300, block_size=25, seed=None)
        cover_block.append(lo <= 0 <= hi)
    assert np.mean(cover_iid) < 0.75
    assert np.mean(cover_block) > 0.8


def test_bootstrap_rows_stay_paired():
    x = np.column_stack([np.arange(100.0), np.arange(100.0)])
    lo, hi = bootstrap_ci(x, lambda d: float(np.all(d[:, 0] == d[:, 1])), n_boot=100)
    assert lo == hi == 1.0


def test_hac_regression_matches_statsmodels():
    import statsmodels.api as sm

    from volengine.stats import ols_hac

    rng = np.random.default_rng(0)
    x = rng.normal(size=800)
    e = np.convolve(rng.normal(size=820), np.ones(20) / 4, "valid")[:800]
    y = 1 + 0.5 * x + e
    beta, se = ols_hac(y, x, 10)
    ref = sm.OLS(y, sm.add_constant(x)).fit(
        cov_type="HAC", cov_kwds={"maxlags": 10, "use_correction": False}
    )
    np.testing.assert_allclose(beta, ref.params, rtol=1e-10)
    np.testing.assert_allclose(se, ref.bse, rtol=1e-10)


def test_diebold_mariano_size_with_dependent_losses():
    from volengine.stats import diebold_mariano

    rng = np.random.default_rng(1)
    rejections = 0
    for _ in range(400):
        common = np.convolve(rng.normal(size=1012), np.ones(12) / 12, "valid")
        a = common + rng.normal(0, 0.3, 1001)
        b = common + rng.normal(0, 0.3, 1001)
        rejections += diebold_mariano(a**2, b**2, lags=12).p_value < 0.05
    assert 0.02 < rejections / 400 < 0.09


def test_auc():
    from volengine.stats import auc

    assert auc(np.array([1, 2, 3, 4]), np.array([0, 0, 1, 1])) == 1.0
    assert auc(np.array([1, 2, 3, 4]), np.array([1, 0, 1, 0])) == 0.25
    rng = np.random.default_rng(2)
    assert abs(auc(rng.normal(size=5000), rng.random(5000) < 0.1) - 0.5) < 0.03
