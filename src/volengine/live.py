"""Streaming volatility monitor.

Ticks are aggregated into fixed-length bars aligned on the epoch. Each completed bar
updates an O(1) EWMA variance and a rolling window of OHLC bars used for the
range-based estimators. The monitor is seeded with real historical bars from the
REST API, never with interpolated prices: interpolation invents zero-variance
returns and biases the volatility estimate downward.
"""

from __future__ import annotations

import asyncio
import logging
import math
import threading
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from volengine import estimators as est
from volengine.deribit.rest import RESOLUTIONS, DeribitREST
from volengine.deribit.stream import DeribitStream, DvolUpdate, Event, Tick

log = logging.getLogger(__name__)

DVOL_INDEX = {"BTC": "btc_usd", "ETH": "eth_usd"}


@dataclass
class Bar:
    start: float
    open: float
    high: float
    low: float
    close: float


class BarAggregator:
    """Build epoch-aligned OHLC bars from a tick stream."""

    def __init__(self, bar_seconds: float):
        self.bar_seconds = bar_seconds
        self._bucket: int | None = None
        self._bar: Bar | None = None

    def update(self, timestamp: float, price: float) -> list[Bar]:
        bucket = int(timestamp // self.bar_seconds)
        done: list[Bar] = []
        if self._bar is None:
            self._bucket = bucket
            self._bar = Bar(bucket * self.bar_seconds, price, price, price, price)
            return done
        if bucket < self._bucket:  # late tick, ignore
            return done
        if bucket > self._bucket:
            done.append(self._bar)
            last = self._bar.close
            # Buckets without ticks: price did not move, emit flat bars to keep the grid regular.
            for b in range(self._bucket + 1, bucket):
                done.append(Bar(b * self.bar_seconds, last, last, last, last))
            self._bucket = bucket
            self._bar = Bar(bucket * self.bar_seconds, price, price, price, price)
            return done
        bar = self._bar
        bar.high = max(bar.high, price)
        bar.low = min(bar.low, price)
        bar.close = price
        return done


class StreamingEWMA:
    """sigma2 <- lam * sigma2 + (1 - lam) * r^2, holding the forecast for the next bar."""

    def __init__(self, lam: float, variance: float | None = None):
        if not 0 < lam < 1:
            raise ValueError("lambda must be in (0, 1)")
        self.lam = lam
        self.variance = variance

    def update(self, r: float) -> float:
        if self.variance is None:
            self.variance = r * r
        else:
            self.variance = self.lam * self.variance + (1.0 - self.lam) * r * r
        return self.variance

    @classmethod
    def from_half_life(cls, half_life_bars: float, **kw) -> StreamingEWMA:
        return cls(0.5 ** (1.0 / half_life_bars), **kw)


@dataclass(frozen=True)
class Snapshot:
    instrument: str
    time: pd.Timestamp
    price: float
    bars: int
    lam: float
    ewma: float
    close_to_close: float
    cc_lower: float
    cc_upper: float
    parkinson: float
    garman_klass: float
    yang_zhang: float
    dvol: float | None

    def as_dict(self) -> dict:
        return self.__dict__.copy()


class _InstrumentState:
    def __init__(self, bar_seconds: float, window: int, lam: float):
        self.aggregator = BarAggregator(bar_seconds)
        self.bars: deque[Bar] = deque(maxlen=window)
        self.ewma = StreamingEWMA(lam)
        self.history: deque[tuple[float, float]] = deque(maxlen=5000)  # (bar start, ewma var)
        self.last_price: float | None = None
        self.last_time: float | None = None


class LiveVolMonitor:
    def __init__(
        self,
        instruments: Iterable[str],
        bar_seconds: float = 60.0,
        window: int = 120,
        half_life_seconds: float | None = None,
        level: float = 0.95,
    ):
        self.bar_seconds = bar_seconds
        self.window = window
        self.level = level
        self.periods = est.periods_per_year(bar_seconds, "crypto")
        self._half_life = half_life_seconds
        default_lam = 0.5 ** (bar_seconds / (half_life_seconds or 3600.0))
        self._state = {i: _InstrumentState(bar_seconds, window, default_lam) for i in instruments}
        self._dvol: dict[str, float] = {}
        self._lock = threading.Lock()

    @property
    def instruments(self) -> list[str]:
        return list(self._state)

    @property
    def dvol_indices(self) -> list[str]:
        return sorted(
            {DVOL_INDEX[i.split("-")[0]] for i in self._state if i.split("-")[0] in DVOL_INDEX}
        )

    # -- seeding ------------------------------------------------------------

    def seed(self, rest: DeribitREST, lookback_bars: int = 2000) -> None:
        """Warm up every instrument from historical trade bars."""
        native = {v: k for k, v in RESOLUTIONS.items()}
        for instrument, st in self._state.items():
            res = native.get(int(self.bar_seconds), "1")
            res_seconds = RESOLUTIONS[res]
            end = pd.Timestamp.now(tz="UTC")
            start = end - pd.Timedelta(seconds=res_seconds * lookback_bars)
            hist = rest.ohlc(instrument, res, start, end)
            if len(hist) < 50:
                log.warning("not enough history to seed %s", instrument)
                continue
            r = est.log_returns(hist["close"])
            # Variance scales linearly with horizon under a random walk.
            scale = self.bar_seconds / res_seconds
            if self._half_life is None and len(r) >= 200:
                lam_native = est.fit_ewma_lambda(r)
                st.ewma.lam = lam_native**scale
            lam_native = st.ewma.lam ** (1.0 / scale)
            st.ewma.variance = float(est.ewma_variance(r.to_numpy(), lam_native)[-1]) * scale
            if res_seconds == self.bar_seconds:
                path = est.ewma_variance(r.to_numpy(), st.ewma.lam)
                st.history.extend(zip((t.timestamp() for t in r.index), path, strict=False))
                for ts, row in hist.tail(self.window).iterrows():
                    st.bars.append(Bar(ts.timestamp(), row.open, row.high, row.low, row.close))
            st.last_price = float(hist["close"].iloc[-1])
            log.info("seeded %s with %d bars, lambda=%.4f", instrument, len(hist), st.ewma.lam)

    # -- event handling -----------------------------------------------------

    def on_event(self, event: Event) -> bool:
        """Process an event; returns True when at least one bar completed."""
        with self._lock:
            if isinstance(event, DvolUpdate):
                self._dvol[event.index_name] = event.value / 100.0
                return False
            if not isinstance(event, Tick) or event.instrument not in self._state:
                return False
            st = self._state[event.instrument]
            st.last_price, st.last_time = event.mark_price, event.timestamp
            completed = st.aggregator.update(event.timestamp, event.mark_price)
            for bar in completed:
                prev = st.bars[-1].close if st.bars else None
                if prev is not None and prev > 0:
                    st.ewma.update(math.log(bar.close / prev))
                    st.history.append((bar.start, st.ewma.variance))
                st.bars.append(bar)
            return bool(completed)

    # -- output ------------------------------------------------------------

    def snapshot(self, instrument: str) -> Snapshot | None:
        with self._lock:
            st = self._state[instrument]
            if st.ewma.variance is None or st.last_price is None:
                return None
            bars = list(st.bars)
            lam, ewma_var = st.ewma.lam, st.ewma.variance
            dvol = self._dvol.get(DVOL_INDEX.get(instrument.split("-")[0], ""))
            last_time, price = st.last_time, st.last_price

        nan = float("nan")
        cc = lo = hi = pk = gk = yz = nan
        if len(bars) >= 10:
            df = pd.DataFrame([b.__dict__ for b in bars])
            r = np.diff(np.log(df["close"].to_numpy()))
            n = r.size
            v = r @ r / n
            alpha = 1 - self.level
            # chi2 interval: cheap enough to recompute on every bar; assumes iid normal returns.
            cc = math.sqrt(v * self.periods)
            lo = math.sqrt(n * v / stats.chi2.ppf(1 - alpha / 2, n) * self.periods)
            hi = math.sqrt(n * v / stats.chi2.ppf(alpha / 2, n) * self.periods)
            pk = math.sqrt(max(est.parkinson(df), 0) * self.periods)
            gk = math.sqrt(max(est.garman_klass(df), 0) * self.periods)
            yz = math.sqrt(max(est.yang_zhang(df), 0) * self.periods)
        ts = (
            pd.Timestamp(last_time, unit="s", tz="UTC") if last_time else pd.Timestamp.now(tz="UTC")
        )
        return Snapshot(
            instrument=instrument,
            time=ts,
            price=price,
            bars=len(bars),
            lam=lam,
            ewma=math.sqrt(ewma_var * self.periods),
            close_to_close=cc,
            cc_lower=lo,
            cc_upper=hi,
            parkinson=pk,
            garman_klass=gk,
            yang_zhang=yz,
            dvol=dvol,
        )

    def history(self, instrument: str) -> pd.Series:
        """Annualized EWMA volatility path (seed + live), indexed by bar time."""
        with self._lock:
            h = list(self._state[instrument].history)
        if not h:
            return pd.Series(dtype=float, name=instrument)
        t, v = zip(*h, strict=False)
        idx = pd.to_datetime(np.array(t), unit="s", utc=True)
        return pd.Series(np.sqrt(np.array(v) * self.periods), index=idx, name=instrument)

    def snapshots(self) -> list[Snapshot]:
        return [s for s in (self.snapshot(i) for i in self._state) if s is not None]

    # -- runners -----------------------------------------------------------

    async def run(self, stream: DeribitStream | None = None, on_bar=None) -> None:
        stream = stream or DeribitStream(self.instruments, self.dvol_indices)
        async for event in stream.events():
            if self.on_event(event) and on_bar is not None:
                on_bar(self)

    def start_background(self) -> threading.Thread:
        """Run the stream in a daemon thread (used by the dashboard)."""
        thread = threading.Thread(target=lambda: asyncio.run(self.run()), daemon=True)
        thread.start()
        return thread
