"""Live monitor: five-minute forecast, P(violent), regime, and the spike detector.

Start-up, replayed after a reconnection or a hole in the trade sequence:

1. subscribe to the trade stream and buffer what arrives;
2. load the recent past as 10-second bars: complete days from the local store when
   they are there, the rest from the REST history server;
3. run the models over those bars, then replay the buffered trades whose sequence
   number is past the last one loaded.

Every trade carries a ``trade_seq`` that increases by one. When the stream skips
numbers, the missing trades are looked up on the history server: if any of them was an
order-book trade, the bars are wrong and the monitor re-seeds; if they were all block or
combo trades, which do not enter the bars, it carries on. Bars close on exchange time:
any message (a trade, or DVOL about twice a second) closes the bars that ended more than
``GRACE_MS`` earlier.
"""

from __future__ import annotations

import asyncio
import logging
import math
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from volengine.data.bars import BarStore, build_bars
from volengine.data.store import utc
from volengine.deribit.rest import DeribitREST
from volengine.deribit.stream import DeribitStream, DvolUpdate, Event, Reconnected, Trade
from volengine.detector import SECONDS_PER_YEAR, DetectorConfig, Level, SpikeDetector
from volengine.features import HORIZON_SECONDS, VOL_WINDOWS
from volengine.forecast import Forecast, Model, Regime, StreamingForecaster, default_model
from volengine.sampling import Bar, GridSampler

log = logging.getLogger(__name__)

DVOL_INDEX = {"BTC": "btc_usd", "ETH": "eth_usd"}
GRACE_MS = 1000  # trades are pushed in 100 ms batches; wait before closing a bar
HISTORY_HOURS = 6  # bars kept for charts
BAR_FIELDS = ["close", "n", "volume", "buy_volume", "liq_volume"]


@dataclass(frozen=True)
class Snapshot:
    instrument: str
    time: pd.Timestamp  # end of the last completed bar
    price: float  # last trade
    regime: Regime
    regime_since: pd.Timestamp
    forecast_ratio: float  # forecast vol over the next 5 minutes / normal vol
    forecast_vol: float  # the same, annualized
    p_violent: float  # P(vol next 5 minutes >= 2 x normal)
    sigma_5m: float  # one-standard-deviation price move over the next 5 minutes
    level: Level  # spike detector: what is happening now
    ratio: float  # vol over the last minute / normal
    fast_vol: float
    slow_vol: float  # "normal"
    move_1m: float  # log return over the last minute
    dvol: float | None
    latency_ms: float | None  # receipt time minus exchange time of the last trade
    seq_gaps: int  # holes in the trade sequence since start


@dataclass(frozen=True)
class Alert:
    """``kind`` is "regime" (forecast enters STRESSED) or "spike" (detector escalates)."""

    time: pd.Timestamp
    instrument: str
    kind: str
    previous: str
    level: str
    forecast_ratio: float
    forecast_vol: float
    p_violent: float
    ratio: float
    fast_vol: float
    slow_vol: float
    price: float
    move_1m: float
    dvol: float | None
    live: bool = True  # False when found while replaying the start-up history

    def as_dict(self) -> dict:
        d = {k: round(v, 6) if isinstance(v, float) else v for k, v in asdict(self).items()}
        d["time"] = self.time.isoformat()
        return d


def _bar(start_ms: int, row) -> Bar:
    return Bar(
        int(start_ms), float(row[0]), int(row[1]), float(row[2]), float(row[3]), float(row[4])
    )


class _State:
    def __init__(self, config: DetectorConfig, model: Model):
        self.config = config
        self.sampler = GridSampler(config.bar_seconds)
        self.detector: SpikeDetector | None = None
        self.forecaster = StreamingForecaster(model, config)
        self.forecast: Forecast | None = None
        per_minute = max(1, round(60 / config.bar_seconds))
        self.minute: deque[float] = deque(maxlen=per_minute + 1)
        self.history: deque[tuple] = deque(maxlen=round(HISTORY_HOURS * 3600 / config.bar_seconds))
        self.warm: list[tuple[Bar, float]] = []
        self.level_since: float | None = None
        self.regime_since: float | None = None
        self.regime_armed = True  # a regime alert re-arms once back to NORMAL or CALM
        self.last_close: float | None = None
        self.last_bar_ms: int | None = None
        self.last_seq: int | None = None
        self.last_trade_ms: int | None = None
        self.last_latency_ms: float | None = None

    def add_bar(self, bar: Bar) -> list[dict]:
        """Feed one completed bar; returns the alerts it raised."""
        prev, self.last_close = self.last_close, bar.close
        self.last_bar_ms = bar.start_ms
        if prev is None:
            return []
        r = math.log(bar.close / prev)
        if self.detector is not None:
            return self._step(bar, r)
        # Same start-up as run_detector: initialize on the warm-up sample, then replay it.
        self.warm.append((bar, r))
        if len(self.warm) < self.config.warmup_bars:
            return []
        var = float(np.mean([x * x for _, x in self.warm]))
        self.detector = SpikeDetector(self.config, variance=max(var, 1e-12))
        for b, x in self.warm:
            self._step(b, x, report=False)
        self.warm = []
        return []

    def _step(self, bar: Bar, r: float, report: bool = True) -> list[dict]:
        changes = []
        end_s = (bar.start_ms + self.sampler.step) / 1000.0
        self.minute.append(bar.close)
        before = self.detector.level
        rd = self.detector.update(r)
        if rd.level != before or self.level_since is None:
            self.level_since = end_s
        if rd.escalated:
            changes.append(("spike", before.name.lower(), rd.level.name.lower()))

        self.forecaster.update(r, bar.n, bar.volume, bar.buy_volume, bar.liq_volume)
        old = self.forecast.regime if self.forecast is not None else None
        self.forecast = fc = self.forecaster.forecast(rd.slow_var, end_s)
        if fc is not None:
            if fc.regime != old or self.regime_since is None:
                self.regime_since = end_s
            if fc.regime <= Regime.NORMAL:
                self.regime_armed = True
            elif fc.regime == Regime.STRESSED and self.regime_armed and old is not None:
                changes.append(("regime", old.name.lower(), "stressed"))
                self.regime_armed = False
        if not report or fc is None:
            return []

        ppy = self.config.bars_per_year
        fast, slow = math.sqrt(rd.fast_var * ppy), math.sqrt(rd.slow_var * ppy)
        row = (int(end_s * 1000), bar.close, fast, slow, rd.ratio, int(rd.level))
        self.history.append((*row, fc.vol, fc.p_violent, int(fc.regime)))
        full = len(self.minute) == self.minute.maxlen
        move = math.log(bar.close / self.minute[0]) if full else float("nan")
        return [
            {
                "time": pd.Timestamp(end_s, unit="s", tz="UTC"),
                "kind": kind,
                "previous": prev,
                "level": new,
                "forecast_ratio": fc.ratio,
                "forecast_vol": fc.vol,
                "p_violent": fc.p_violent,
                "ratio": rd.ratio,
                "fast_vol": fast,
                "slow_vol": slow,
                "price": bar.close,
                "move_1m": move,
            }
            for kind, prev, new in changes
        ]


class LiveMonitor:
    def __init__(
        self,
        instruments: Iterable[str],
        config: DetectorConfig | None = None,
        on_alert: Callable[[Alert], None] | None = None,
        models: dict[str, Model] | None = None,
        bar_store: BarStore | None = None,
    ):
        """``models`` maps instruments to fitted models; defaults ship with the package."""
        self.config = config or DetectorConfig()
        instruments = list(instruments)
        models = models or {}
        self.models = {i: models.get(i) or default_model(i) for i in instruments}
        self.bar_store = bar_store or BarStore()
        self._state = {i: _State(self.config, self.models[i]) for i in instruments}
        self._dvol: dict[str, float] = {}
        self._alerts: deque[Alert] = deque(maxlen=500)
        self._on_alert = on_alert
        self._lock = threading.Lock()
        self.seq_gaps = dict.fromkeys(instruments, 0)  # holes that contained book trades
        self.seq_skipped = dict.fromkeys(instruments, 0)  # holes of block/combo trades only
        self._holes: list[tuple[str, int, int, int, int]] = []  # to check against REST
        self._resync = False

    @property
    def instruments(self) -> list[str]:
        return list(self._state)

    @property
    def dvol_indices(self) -> list[str]:
        roots = {i.split("-")[0] for i in self._state}
        return sorted(DVOL_INDEX[r] for r in roots if r in DVOL_INDEX)

    @property
    def seed_span(self) -> pd.Timedelta:
        """Warm-up of the detector plus the longest look-back of the forecast."""
        looks = [
            VOL_WINDOWS[f] for m in self.models.values() for f in m.features if f in VOL_WINDOWS
        ]
        warm = self.config.warmup_bars * self.config.bar_seconds
        return pd.Timedelta(seconds=warm + max([1800, *looks]) + 600)

    # -- seeding ------------------------------------------------------------

    def fetch_seed(self, rest: DeribitREST, end: pd.Timestamp | None = None) -> dict:
        """Recent bars per instrument: stored days from disk, the rest from REST trades."""
        end = utc(end) if end is not None else pd.Timestamp.now(tz="UTC")
        start = end - self.seed_span
        out = {}
        for inst in self._state:
            frames, cursor = [], start.floor("D")
            while cursor + pd.Timedelta(days=1) <= end and self.bar_store.trades.has(inst, cursor):
                frames.append(self.bar_store.day(inst, cursor, self.config.bar_seconds))
                cursor += pd.Timedelta(days=1)
            fetch = getattr(rest, "trades_parallel", rest.trades)
            trades = fetch(inst, max(cursor, start), end)
            out[inst] = (frames, trades)
        return out

    def apply_seed(self, seed: dict) -> None:
        """Rebuild each instrument from loaded bars; escalations are kept with live=False."""
        for inst, (frames, trades) in seed.items():
            st = _State(self.config, self.models[inst])
            parts = [f for f in frames if len(f)]
            if len(trades):
                parts.append(build_bars(trades, self.config.bar_seconds))
            replayed = []
            if parts:
                bars = pd.concat(parts)
                bars = bars[~bars.index.duplicated(keep="last")].sort_index()
                step = pd.Timedelta(seconds=self.config.bar_seconds)
                bars = bars.reindex(pd.date_range(bars.index[0], bars.index[-1], freq=step))
                bars["close"] = bars["close"].ffill()
                bars = bars.fillna(0.0)
                epoch = pd.Timestamp(0, tz="UTC")
                starts = ((bars.index - epoch) // pd.Timedelta(milliseconds=1)).to_numpy()
                rows = bars[BAR_FIELDS].to_numpy()
                # Completed bars go through the models; the last one is still in progress.
                for t, row in zip(starts[:-1], rows[:-1], strict=True):
                    for draft in st.add_bar(_bar(t, row)):
                        replayed.append(self._make_alert(inst, draft, live=False))
                st.sampler.current = _bar(starts[-1], rows[-1])
            if len(trades):
                st.last_seq = int(trades["trade_seq"].max())
            with self._lock:
                self._state[inst] = st
                known = {(a.instrument, a.time, a.kind) for a in self._alerts}
                fresh = [a for a in replayed if (a.instrument, a.time, a.kind) not in known]
                merged = sorted([*self._alerts, *fresh], key=lambda a: a.time)
                self._alerts = deque(merged, maxlen=self._alerts.maxlen)
            log.info("seeded %s, last trade_seq %s", inst, st.last_seq)

    # -- event handling -----------------------------------------------------

    def on_event(self, event: Event, received_ms: float | None = None) -> list[Alert]:
        """Process one stream event; returns the alerts it triggered."""
        alerts: list[Alert] = []
        with self._lock:
            if isinstance(event, DvolUpdate):
                self._dvol[event.index_name] = event.value / 100.0
            elif isinstance(event, Trade) and event.instrument in self._state:
                st = self._state[event.instrument]
                if st.last_seq is not None:
                    if event.trade_seq <= st.last_seq:
                        return alerts  # already loaded
                    if event.trade_seq > st.last_seq + 1:
                        t0 = (
                            st.last_trade_ms if st.last_trade_ms is not None else event.timestamp_ms
                        )
                        self._holes.append(
                            (
                                event.instrument,
                                st.last_seq + 1,
                                event.trade_seq - 1,
                                min(t0, event.timestamp_ms),
                                event.timestamp_ms,
                            )
                        )
                st.last_seq = event.trade_seq
                st.last_trade_ms = event.timestamp_ms
                if received_ms is not None:
                    st.last_latency_ms = received_ms - event.timestamp_ms
                if event.kind == "book":
                    bars = st.sampler.update(
                        event.timestamp_ms,
                        event.price,
                        event.amount,
                        event.is_buy,
                        event.liquidation,
                    )
                    alerts += self._add_bars(event.instrument, st, bars)
            now = getattr(event, "timestamp_ms", None)
            if now is not None:
                for name, st in self._state.items():
                    alerts += self._add_bars(name, st, st.sampler.advance(now - GRACE_MS))
        for a in alerts:
            if self._on_alert is not None:
                self._on_alert(a)
        return alerts

    def _add_bars(self, instrument: str, st: _State, bars: list[Bar]) -> list[Alert]:
        out = []
        for bar in bars:
            for draft in st.add_bar(bar):
                alert = self._make_alert(instrument, draft)
                self._alerts.append(alert)
                out.append(alert)
        return out

    def check_holes(self, rest: DeribitREST) -> None:
        """Look up the trades the stream skipped; re-seed if any was a book trade."""
        with self._lock:
            holes, self._holes = self._holes, []
        for inst, first, last, t0, t1 in holes:
            lo = pd.Timestamp(t0 - 5_000, unit="ms", tz="UTC")
            hi = pd.Timestamp(t1 + 5_000, unit="ms", tz="UTC")
            missed = rest.trades(inst, lo, hi)
            missed = missed[missed["trade_seq"].between(first, last)]
            book = int((missed["kind"] == "book").sum())
            if book or len(missed) < last - first + 1:
                log.warning(
                    "%s: %d book trades missed (seq %d-%d), re-seeding", inst, book, first, last
                )
                self.seq_gaps[inst] += 1
                self._resync = True
            else:
                self.seq_skipped[inst] += 1
                log.info("%s: stream skipped %d block/combo trades", inst, len(missed))

    def _make_alert(self, instrument: str, draft: dict, live: bool = True) -> Alert:
        dvol = self._dvol.get(DVOL_INDEX.get(instrument.split("-")[0], "")) if live else None
        return Alert(instrument=instrument, dvol=dvol, live=live, **draft)

    # -- output ------------------------------------------------------------

    def _snapshot(self, instrument: str, st: _State) -> Snapshot | None:
        if st.detector is None or st.forecast is None or not st.history:
            return None
        det, fc, ppy = st.detector, st.forecast, self.config.bars_per_year
        price = st.sampler.close if st.sampler.close is not None else st.last_close
        full = len(st.minute) == st.minute.maxlen
        return Snapshot(
            instrument=instrument,
            time=pd.Timestamp(st.history[-1][0], unit="ms", tz="UTC"),
            price=price,
            regime=fc.regime,
            regime_since=pd.Timestamp(st.regime_since, unit="s", tz="UTC"),
            forecast_ratio=fc.ratio,
            forecast_vol=fc.vol,
            p_violent=fc.p_violent,
            sigma_5m=price * fc.vol * math.sqrt(HORIZON_SECONDS / SECONDS_PER_YEAR),
            level=det.level,
            ratio=det.ratio,
            fast_vol=math.sqrt(det.fast_var * ppy),
            slow_vol=math.sqrt(det.slow_var * ppy),
            move_1m=math.log(st.minute[-1] / st.minute[0]) if full else float("nan"),
            dvol=self._dvol.get(DVOL_INDEX.get(instrument.split("-")[0], "")),
            latency_ms=st.last_latency_ms,
            seq_gaps=self.seq_gaps[instrument],
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
        """Recent bars: price, vols, spike ratio and level, forecast, P(violent), regime."""
        with self._lock:
            rows = list(self._state[instrument].history)
        cols = ["time", "price", "fast_vol", "slow_vol", "ratio", "level"]
        cols += ["forecast_vol", "p_violent", "regime"]
        df = pd.DataFrame(rows, columns=cols)
        df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
        return df.set_index("time")

    # -- runners -----------------------------------------------------------

    async def run(
        self,
        stream: DeribitStream | None = None,
        rest: DeribitREST | None = None,
        on_bar: Callable[[LiveMonitor], None] | None = None,
    ) -> None:
        """Consume the stream forever; seed at start, after reconnections and sequence holes."""
        stream = stream or DeribitStream(self.instruments, self.dvol_indices)
        rest = rest or DeribitREST()
        buffer: list[tuple[Event, float]] | None = []
        seeding: asyncio.Future | None = None
        checking: asyncio.Future | None = None
        async for event in stream.events():
            received = time.time() * 1000
            if self._holes and (checking is None or checking.done()):
                checking = asyncio.ensure_future(asyncio.to_thread(self.check_holes, rest))
            if isinstance(event, Reconnected) or seeding is None or self._resync:
                log.info("loading recent history to seed the models")
                self._resync = False
                buffer = []
                seeding = asyncio.ensure_future(asyncio.to_thread(self.fetch_seed, rest))
                if isinstance(event, Reconnected):
                    continue
            if not seeding.done():
                buffer.append((event, received))
                continue
            if buffer is not None:
                try:
                    self.apply_seed(seeding.result())
                except Exception:
                    log.exception("seeding failed, retrying")
                    seeding = None
                    continue
                pending, buffer = [*buffer, (event, received)], None
                for ev, rcv in pending:
                    self.on_event(ev, rcv)
                if on_bar is not None:
                    on_bar(self)
                continue
            before = self._last_bars()
            self.on_event(event, received)
            if on_bar is not None and self._last_bars() != before:
                on_bar(self)

    def _last_bars(self) -> tuple:
        return tuple(st.last_bar_ms for st in self._state.values())

    def start_background(self) -> threading.Thread:
        """Run the stream in a daemon thread (used by the dashboard)."""
        thread = threading.Thread(target=lambda: asyncio.run(self.run()), daemon=True)
        thread.start()
        return thread
