import numpy as np
import pandas as pd

from volengine.data.schema import COLUMNS

SECONDS_PER_YEAR = 365 * 24 * 3600
DAY = 86_400


def simulate_trades(
    seconds,
    vol,
    rate=1.0,
    start="2026-01-01",
    seed=0,
    jumps_per_day=0.0,
    jump_sd=0.0,
    block_share=0.0,
):
    """Trade tape in the store's schema, from a driftless random walk.

    ``vol`` is the annualized volatility, a constant or one value per second. Trades
    arrive as a Poisson process with ``rate`` per second and are numbered by
    ``trade_seq``. A ``block_share`` of them are block trades priced 20 bp off the
    market, which the bars must ignore. Independent jumps give fat tails without
    volatility clustering.
    """
    rng = np.random.default_rng(seed)
    vol = np.broadcast_to(np.asarray(vol, dtype=float), (seconds,))
    steps = seconds * 10  # log price on a 100 ms grid
    dlogp = rng.standard_normal(steps) * np.repeat(vol, 10) * np.sqrt(0.1 / SECONDS_PER_YEAR)
    jumps = rng.random(steps) < jumps_per_day / (DAY * 10)
    dlogp[jumps] += rng.standard_normal(jumps.sum()) * jump_sd
    logp = np.log(50_000.0) + np.cumsum(dlogp)
    n = rng.poisson(rate * seconds)
    t_ms = np.sort(rng.integers(0, seconds * 1000, size=n))
    price = np.exp(logp[t_ms // 100])
    block = rng.random(n) < block_share
    price = np.where(block, price * 1.002, price)
    t0 = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    df = pd.DataFrame(
        {
            "timestamp": t0 + t_ms,
            "trade_seq": 1_000 + np.arange(n),
            "trade_id": (5_000 + np.arange(n)).astype(str),
            "price": price,
            "amount": rng.integers(1, 100, n) * 10.0,
            "is_buy": rng.random(n) < 0.5,
            "index_price": np.exp(logp[t_ms // 100]),
            "mark_price": np.exp(logp[t_ms // 100]),
            "liquidation": np.where(rng.random(n) < 0.01, "T", ""),
            "kind": np.where(block, "block", "book"),
        }
    )
    return df[COLUMNS]


def bursty_vol(seconds, base=0.4, mult=4.0, per_day=10, length=600, seed=0):
    """Volatility path with bursts: clustered, hence forecastable."""
    rng = np.random.default_rng(seed)
    vol = np.full(seconds, base)
    for s in rng.integers(0, seconds - length, size=round(per_day * seconds / DAY)):
        vol[s : s + length] = base * mult
    return vol
