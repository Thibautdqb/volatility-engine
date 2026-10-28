import numpy as np
import pandas as pd

SECONDS_PER_YEAR = 365 * 24 * 3600


def simulate_trades(
    seconds, vol, rate=2.0, start="2026-01-01", seed=0, jumps_per_day=0.0, jump_sd=0.0
):
    """Trade tape from a driftless random walk observed at Poisson trade times.

    ``vol`` is the annualized volatility, either a constant or an array with one value
    per second. Independent jumps (compound Poisson, normal log sizes with standard
    deviation ``jump_sd``) can be added to get fat tails without volatility clustering.
    Returns columns ``timestamp`` (ms), ``trade_id``, ``price``, ``amount``.
    """
    rng = np.random.default_rng(seed)
    vol = np.broadcast_to(np.asarray(vol, dtype=float), (seconds,))
    # Log price on a 100 ms grid, then sampled at the trade times.
    steps = seconds * 10
    sd = np.repeat(vol, 10) * np.sqrt(0.1 / SECONDS_PER_YEAR)
    dlogp = rng.standard_normal(steps) * sd
    jumps = rng.random(steps) < jumps_per_day / 864_000
    dlogp[jumps] += rng.standard_normal(jumps.sum()) * jump_sd
    logp = np.log(50_000.0) + np.cumsum(dlogp)
    n = rng.poisson(rate * seconds)
    t_ms = np.sort(rng.integers(0, seconds * 1000, size=n))
    t0 = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    return pd.DataFrame(
        {
            "timestamp": t0 + t_ms,
            "trade_id": np.arange(n).astype(str),
            "price": np.exp(logp[t_ms // 100]),
            "amount": 1.0,
        }
    )
