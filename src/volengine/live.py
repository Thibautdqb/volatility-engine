"""Live spike monitor on Deribit perpetuals.

Start-up sequence, also replayed after every reconnection:

1. subscribe to the trade stream and buffer what arrives;
2. download the last two hours of trades from the history server and run the
   detector over them, so the slow (normal) volatility is warm from the first bar;
3. replay the buffered trades that the download did not contain, then go live.

Bars close on exchange time: any message (a trade or a DVOL update, roughly once per
second) closes the bars that ended more than ``GRACE_MS`` earlier, so a quiet market
still produces bars on schedule.
"""

from __future__ import annotations

import asyncio
import logging
import math
import threading
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from volengine.deribit.rest import DeribitREST
from volengine.deribit.stream import DeribitStream, DvolUpdate, Event, Reconnected, Trade
from volengine.detector import SECONDS_PER_YEAR, DetectorConfig, Level, SpikeDetector
from volengine.sampling import GridSampler, bar_closes

log = logging.getLogger(__name__)

DVOL_INDEX = {"BTC": "btc_usd", "ETH": "eth_usd"}
GRACE_MS = 1000  # trades are pushed in 100 ms batches; wait before closing a bar
HISTORY_BARS = 2160  # bars kept for charts (6 hours of 10-second bars)


@dataclass(frozen=True)
class Snapshot:
    """What a trader needs at a glance, refreshed on every bar."""

    instrument: str
    time: pd.Timestamp  # end of the last completed bar
    price: float
    level: Level
    level_since: pd.Timestamp
    ratio: float  # fast / slow volatility
    fast_vol: float  # annualized, last ~minute
    slow_vol: float  # annualized, last ~30 minutes
    move_1m: float  # log return over the last minute
    sigma_1m: float  # one-standard-deviation price move over one minute at the current vol
    activity: float  # trade rate, fast / slow
    dvol: float | None  # 30-day implied volatility


@dataclass(frozen=True)
class Alert:
    time: pd.Timestamp
    instrument: str
    level: Level
    previous: Level
    ratio: float
    fast_vol: float
    slow_vol: float
    price: float
    move_1m: float
    dvol: float | None

    def as_dict(self) -> dict:
        d = {k: round(v, 6) if isinstance(v, float) else v for k, v in asdict(self).items()}
        d["time"] = self.time.isoformat()
        d["level"], d["previous"] = self.level.name.lower(), self.previous.name.lower()
        return d


class _Rate:
    """Fast and slow EWMA of the number of trades per bar."""

    def __init__(self, config: DetectorConfig, init: float):
        self.lf, self.ls = config.lam_fast, config.lam_slow
        self.fast = self.slow = init

    def update(self, n: int) -> None:
        self.fast = self.lf * self.fast + (1 - self.lf) * n
        self.slow = self.ls * self.slow + (1 - self.ls) * n

    @property
    def ratio(self) -> float:
        return self.fast / self.slow if self.slow > 0 else float("nan")


class _InstrumentState:
    def __init__(self, config: DetectorConfig):
        self.config = config
        self.sampler = GridSampler(config.bar_seconds)
        self.detector: SpikeDetector | None = None
        self.rate: _Rate | None = None
        per_minute = max(1, round(60 / config.bar_seconds))
        self.closes: deque[float] = deque(maxlen=per_minute + 1)
        self.history: deque[tuple] = deque(maxlen=HISTORY_BARS)
        self.level_since: float | None = None
        self.last_bar_ms: int | None = None
        self.seed_end: tuple[int, set[str]] | None = None  # last downloaded trade
        self.warm: list[tuple[int, float, float, int]] = []  # bars seen before the detector starts

    def add_bar(self, start_ms: int, close: float, n_trades: int) -> tuple[Level, Level] | None:
        """Feed one completed bar; returns (previous, new) level when it escalated."""
        prev_close = self.closes[-1] if self.closes else None
        self.closes.append(close)
        self.last_bar_ms = start_ms
        if prev_close is None:
            return None
        r = math.log(close / prev_close)
        if self.detector is not None:
            return self._step(start_ms, close, r, n_trades)
        # Same start-up as run_detector: initialize on the warm-up sample, then replay it.
        self.warm.append((start_ms, close, r, n_trades))
        if len(self.warm) < self.config.warmup_bars:
            return None
        var = float(np.mean([w[2] ** 2 for w in self.warm]))
        self.detector = SpikeDetector(self.config, variance=max(var, 1e-12))
        self.rate = _Rate(self.config, max(float(np.mean([w[3] for w in self.warm])), 1e-6))
        for w in self.warm:
            self._step(*w)
        self.warm = []
        return None

    def _step(self, start_ms: int, close: float, r: float, n_trades: int):
        before = self.detector.level
        rd = self.detector.update(r)
        self.rate.update(n_trades)
        end_ms = start_ms + self.sampler.step
        if rd.level != before or self.level_since is None:
            self.level_since = end_ms / 1000.0
        ppy = self.config.bars_per_year
        fast, slow = math.sqrt(rd.fast_var * ppy), math.sqrt(rd.slow_var * ppy)
        self.history.append((end_ms, close, fast, slow, rd.ratio, int(rd.level)))
        return (before, rd.level) if rd.escalated else None


class LiveSpikeMonitor:
    def __init__(
        self,
        instruments: Iterable[str],
        config: DetectorConfig | None = None,
        on_alert: Callable[[Alert], None] | None = None,
    ):
        self.config = config or DetectorConfig()
        self._state = {i: _InstrumentState(self.config) for i in instruments}
        self._dvol: dict[str, float] = {}
        self._alerts: deque[Alert] = deque(maxlen=500)
        self._on_alert = on_alert
        self._lock = threading.Lock()

    @property
    def instruments(self) -> list[str]:
        return list(self._state)

    @property
    def dvol_indices(self) -> list[str]:
        roots = {i.split("-")[0] for i in self._state}
        return sorted(DVOL_INDEX[r] for r in roots if r in DVOL_INDEX)

    @property
    def seed_span(self) -> pd.Timedelta:
        """History downloaded at start-up: the warm-up plus as much again for the charts."""
        return pd.Timedelta(seconds=1.5 * self.config.warmup_bars * self.config.bar_seconds)

    # -- seeding ------------------------------------------------------------

    def fetch_seed(self, rest: DeribitREST, end: pd.Timestamp | None = None) -> dict:
        end = end or pd.Timestamp.now(tz="UTC")
        return {i: rest.trades(i, end - self.seed_span, end) for i in self._state}

    def apply_seed(self, trades: dict[str, pd.DataFrame]) -> None:
        """Rebuild every instrument's state from a trade history, without raising alerts."""
        for instrument, df in trades.items():
            st = _InstrumentState(self.config)
            if len(df):
                closes = bar_closes(df, self.config.bar_seconds)
                step = st.sampler.step
                buckets = df["timestamp"].to_numpy(dtype=np.int64) // step
                counts = pd.Series(1, index=buckets).groupby(level=0).sum()
                epoch = pd.Timestamp(0, tz="UTC")
                starts = ((closes.index - epoch) // pd.Timedelta(milliseconds=1)).to_numpy()
                # Completed bars go through the detector; the last one is still in progress.
                for t, c in zip(starts[:-1], closes.to_numpy()[:-1], strict=True):
                    st.add_bar(int(t), float(c), int(counts.get(t // step, 0)))
                last = int(starts[-1] // step)
                st.sampler = GridSampler(self.config.bar_seconds, last, float(closes.iloc[-1]))
                st.sampler.trades = int(counts.get(last, 0))
                st.seed_end = (int(df["timestamp"].iloc[-1]), set(df["trade_id"].iloc[-200:]))
            with self._lock:
                self._state[instrument] = st
            log.info("seeded %s with %d trades", instrument, len(df))

    # -- event handling -----------------------------------------------------

    def on_event(self, event: Event) -> list[Alert]:
        """Process one stream event; returns the alerts it triggered."""
        alerts: list[Alert] = []
        with self._lock:
            if isinstance(event, DvolUpdate):
                self._dvol[event.index_name] = event.value / 100.0
            elif isinstance(event, Trade) and event.instrument in self._state:
                st = self._state[event.instrument]
                seed_end = st.seed_end
                if seed_end and (
                    event.timestamp_ms < seed_end[0]
                    or (event.timestamp_ms == seed_end[0] and event.trade_id in seed_end[1])
                ):
                    return alerts  # already in the downloaded history
                bars = st.sampler.update(event.timestamp_ms, event.price)
                alerts += self._add_bars(event.instrument, st, bars)
            now = getattr(event, "timestamp_ms", None)
            if now is not None:
                for name, st in self._state.items():
                    alerts += self._add_bars(name, st, st.sampler.advance(now - GRACE_MS))
        for a in alerts:
            if self._on_alert is not None:
                self._on_alert(a)
        return alerts

    def _add_bars(self, instrument: str, st: _InstrumentState, bars) -> list[Alert]:
        out = []
        for start_ms, close, n in bars:
            change = st.add_bar(start_ms, close, n)
            if change is not None:
                snap = self._snapshot(instrument, st)
                alert = Alert(
                    time=snap.time,
                    instrument=instrument,
                    level=change[1],
                    previous=change[0],
                    ratio=snap.ratio,
                    fast_vol=snap.fast_vol,
                    slow_vol=snap.slow_vol,
                    price=snap.price,
                    move_1m=snap.move_1m,
                    dvol=snap.dvol,
                )
                self._alerts.append(alert)
                out.append(alert)
        return out

    # -- output ------------------------------------------------------------

    def _snapshot(self, instrument: str, st: _InstrumentState) -> Snapshot | None:
        if st.detector is None or not st.history:
            return None
        det, ppy = st.detector, self.config.bars_per_year
        fast = math.sqrt(det.fast_var * ppy)
        price = st.closes[-1]
        full_minute = len(st.closes) == st.closes.maxlen
        return Snapshot(
            instrument=instrument,
            time=pd.Timestamp(st.history[-1][0], unit="ms", tz="UTC"),
            price=price,
            level=det.level,
            level_since=pd.Timestamp(st.level_since, unit="s", tz="UTC"),
            ratio=det.ratio,
            fast_vol=fast,
            slow_vol=math.sqrt(det.slow_var * ppy),
            move_1m=math.log(price / st.closes[0]) if full_minute else float("nan"),
            sigma_1m=price * fast * math.sqrt(60.0 / SECONDS_PER_YEAR),
            activity=st.rate.ratio,
            dvol=self._dvol.get(DVOL_INDEX.get(instrument.split("-")[0], "")),
        )

    def snapshot(self, instrument: str) -> Snapshot | None:
        with self._lock:
            return self._snapshot(instrument, self._state[instrument])

    def snapshots(self) -> list[Snapshot]:
        return [s for s in (self.snapshot(i) for i in self._state) if s is not None]

    def alerts(self) -> list[Alert]:
        with self._lock:
            return list(self._alerts)

    def history(self, instrument: str) -> pd.DataFrame:
        """Recent bars: price, fast and slow annualized vol, ratio and level."""
        with self._lock:
            rows = list(self._state[instrument].history)
        cols = ["time", "price", "fast_vol", "slow_vol", "ratio", "level"]
        df = pd.DataFrame(rows, columns=cols)
        df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
        return df.set_index("time")

    # -- runners -----------------------------------------------------------

    async def run(
        self,
        stream: DeribitStream | None = None,
        rest: DeribitREST | None = None,
        on_bar: Callable[[LiveSpikeMonitor], None] | None = None,
    ) -> None:
        """Consume the stream forever, seeding at start and after every reconnection."""
        stream = stream or DeribitStream(self.instruments, self.dvol_indices)
        rest = rest or DeribitREST()
        buffer: list[Event] = []
        seeding: asyncio.Future | None = None
        async for event in stream.events():
            if isinstance(event, Reconnected) or seeding is None:
                log.info("downloading recent trades to seed the detector")
                buffer = []
                seeding = asyncio.ensure_future(asyncio.to_thread(self.fetch_seed, rest))
                if isinstance(event, Reconnected):
                    continue
            if not seeding.done():
                buffer.append(event)
                continue
            if buffer is not None:
                try:
                    self.apply_seed(seeding.result())
                except Exception:
                    log.exception("seeding failed, retrying")
                    seeding = None
                    continue
                pending, buffer = [*buffer, event], None
                for ev in pending:
                    self.on_event(ev)
                if on_bar is not None:
                    on_bar(self)
                continue
            before = self._last_bars()
            self.on_event(event)
            if on_bar is not None and self._last_bars() != before:
                on_bar(self)

    def _last_bars(self) -> tuple:
        return tuple(st.last_bar_ms for st in self._state.values())

    def start_background(self) -> threading.Thread:
        """Run the stream in a daemon thread (used by the dashboard)."""
        thread = threading.Thread(target=lambda: asyncio.run(self.run()), daemon=True)
        thread.start()
        return thread
