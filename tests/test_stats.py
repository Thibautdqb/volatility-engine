import numpy as np
import pytest
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint

from volengine import stats as vs


@pytest.fixture
def ar1():
    rng = np.random.default_rng(3)
    x = np.zeros(1500)
    for t in range(1, x.size):
        x[t] = 0.5 * x[t - 1] + rng.standard_normal()
    return x


def test_ljung_box_matches_statsmodels(ar1):
    ours = vs.ljung_box(ar1, lags=10)
    ref = acorr_ljungbox(ar1, lags=[10])
    assert ours.statistic == pytest.approx(ref["lb_stat"].iloc[0], rel=1e-10)
    assert ours.p_value == pytest.approx(ref["lb_pvalue"].iloc[0], rel=1e-8, abs=1e-300)


def test_arch_lm_matches_statsmodels(garch_returns):
    ours = vs.arch_lm(garch_returns, lags=5)
    lm, p, *_ = het_arch(garch_returns - garch_returns.mean(), nlags=5)
    assert ours.statistic == pytest.approx(lm, rel=1e-8)
    assert ours.p_value == pytest.approx(p, rel=1e-6, abs=1e-300)


def test_ols_hac_matches_statsmodels(ar1):
    rng = np.random.default_rng(0)
    x = rng.standard_normal(ar1.size)
    y = 0.3 + 2.0 * x + ar1
    X = np.column_stack([np.ones_like(x), x])
    beta, cov = vs.ols_hac(y, X, lags=8)
    ref = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 8, "use_correction": False})
    np.testing.assert_allclose(beta, ref.params, rtol=1e-10)
    np.testing.assert_allclose(cov, ref.cov_params(), rtol=1e-8)


def test_hac_mean_test_widens_for_autocorrelated_data(ar1):
    naive_se = ar1.std() / np.sqrt(ar1.size)
    assert vs.hac_mean_test(ar1).se > 1.5 * naive_se


def test_diebold_mariano_identical_losses_not_significant():
    rng = np.random.default_rng(1)
    loss = rng.exponential(size=500)
    res = vs.diebold_mariano(loss, loss + rng.normal(0, 1e-3, 500))
    assert res.p_value > 0.05


def test_diebold_mariano_size_under_null():
    """Rejection rate at 5% should be close to 5% when models are equally good."""
    rng = np.random.default_rng(7)
    rejections = 0
    sims = 400
    for _ in range(sims):
        a, b = rng.exponential(size=(2, 300))
        rejections += vs.diebold_mariano(a, b).p_value < 0.05
    assert 0.02 < rejections / sims < 0.09


def test_diebold_mariano_detects_real_difference():
    rng = np.random.default_rng(2)
    a = rng.exponential(size=1000)
    res = vs.diebold_mariano(a, a + 0.2 + rng.normal(0, 0.5, 1000))
    assert res.statistic < 0 and res.p_value < 1e-3


def test_mincer_zarnowitz_unbiased_forecast():
    rng = np.random.default_rng(4)
    f = rng.uniform(1, 3, 2000)
    y = f + rng.normal(0, 0.5, f.size)
    res = vs.mincer_zarnowitz(y, f)
    assert res.slope == pytest.approx(1.0, abs=0.05)
    assert res.wald_p_value > 0.01


def test_wilson_matches_statsmodels():
    for k, n in [(0, 10), (3, 17), (25, 59)]:
        np.testing.assert_allclose(
            vs.wilson_interval(k, n), proportion_confint(k, n, method="wilson"), rtol=1e-10
        )


def test_benjamini_hochberg_matches_statsmodels():
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216])
    np.testing.assert_allclose(vs.benjamini_hochberg(p), multipletests(p, method="fdr_bh")[1])


def test_benjamini_hochberg_keeps_nan():
    q = vs.benjamini_hochberg(np.array([0.01, np.nan, 0.04]))
    assert np.isnan(q[1]) and np.isfinite(q[[0, 2]]).all()


def test_bootstrap_ci_coverage():
    """95% percentile bootstrap of a mean should cover the truth ~95% of the time."""
    rng = np.random.default_rng(5)
    cover = 0
    sims = 200
    for i in range(sims):
        x = rng.normal(1.0, 1.0, 80)
        lo, hi = vs.bootstrap_ci(x, n_boot=1000, seed=i)
        cover += lo <= 1.0 <= hi
    assert 0.88 < cover / sims < 0.99


def test_stationary_bootstrap_wider_than_iid_for_dependent_data(ar1):
    iid = vs.bootstrap_ci(ar1, n_boot=2000)
    block = vs.bootstrap_ci(ar1, n_boot=2000, block_size=20)
    assert (block[1] - block[0]) > 1.3 * (iid[1] - iid[0])


def test_overlapping_mean_test_has_correct_size():
    """Overlapping 30-day averages of iid noise: rejection rate must stay near 5%."""
    rng = np.random.default_rng(0)
    h, n, sims = 30, 900, 400
    rejections = 0
    for _ in range(sims):
        daily = rng.chisquare(24, n + h) / 24 - 1
        fwd = np.convolve(daily, np.ones(h), "valid")[:n] / h
        rejections += vs.overlapping_mean_test(fwd, h).p_value < 0.05
    assert rejections / sims < 0.085


def test_non_overlapping_mean_test_uses_independent_windows():
    x = np.arange(100, dtype=float)
    res = vs.non_overlapping_mean_test(x, 10)
    assert res.n == 10 and res.df == 9
