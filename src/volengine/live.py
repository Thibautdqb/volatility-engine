"""Live monitor on Deribit perpetuals: five-minute volatility forecast and regime, plus
the spike detector.

Start-up sequence, also replayed after every reconnection:

1. subscribe to the trade stream and buffer what arrives;
2. download the last four hours of trades from the history server and run the
   models over them, so the normal volatility and the forecast are warm from the start;
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
from volengine.forecast import (
    HORIZON_SECONDS,
    Forecast,
    HARModel,
    Regime,
    StreamingHAR,
    default_model,
)
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
    price: float  # last trade, updated on every print
    regime: Regime  # forecast for the next 5 minutes
    regime_since: pd.Timestamp
    forecast_ratio: float  # forecast vol over the next 5 minutes / normal vol
    forecast_vol: float  # the same, annualized
    sigma_5m: float  # one-standard-deviation price move over the next 5 minutes
    level: Level  # spike detector: what is happening right now
    ratio: float  # vol over the last minute / normal vol
    fast_vol: float  # annualized, last ~minute
    slow_vol: float  # annualized, last ~30 minutes ("normal")
    move_1m: float  # log return over the last minute
    dvol: float | None  # 30-day implied volatility


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


class _InstrumentState:
    def __init__(self, config: DetectorConfig, model: HARModel):
        self.config = config
        self.sampler = GridSampler(config.bar_seconds)
        self.detector: SpikeDetector | None = None
        self.har = StreamingHAR(model, config)
        self.forecast: Forecast | None = None
        per_minute = max(1, round(60 / config.bar_seconds))
        self.closes: deque[float] = deque(maxlen=per_minute + 1)  # bars received
        self.minute: deque[float] = deque(maxlen=per_minute + 1)  # bars through the models
        self.history: deque[tuple] = deque(maxlen=HISTORY_BARS)
        self.level_since: float | None = None
        self.regime_since: float | None = None
        self.regime_armed = True  # a regime alert re-arms once back to NORMAL or CALM
        self.last_bar_ms: int | None = None
        self.seed_end: tuple[int, set[str]] | None = None  # last downloaded trade
        self.warm: list[tuple] = []  # bars seen before the detector starts

    def add_bar(self, start_ms: int, close: float, n_trades: int) -> list[dict]:
        """Feed one completed bar; returns the alerts it raised (see ``_step``)."""
        prev_close = self.closes[-1] if self.closes else None
        self.closes.append(close)
        self.last_bar_ms = start_ms
        if prev_close is None:
            return []
        r = math.log(close / prev_close)
        if self.detector is not None:
            return self._step(start_ms, close, r, n_trades)
        # Same start-up as run_detector: initialize on the warm-up sample, then replay it.
        self.warm.append((start_ms, close, r, n_trades))
        if len(self.warm) < self.config.warmup_bars:
            return []
        var = float(np.mean([w[2] ** 2 for w in self.warm]))
        self.detector = SpikeDetector(self.config, variance=max(var, 1e-12))
        # The warm-up only initializes the models: like run_detector, it is not reported.
        for w in self.warm:
            self._step(*w, report=False)
        self.warm = []
        return []

    def _step(
        self, start_ms: int, close: float, r: float, n_trades: int, report: bool = True
    ) -> list[dict]:
        """Update the models with one bar. Alerts carry the values at that bar."""
        changes = []
        end_s = (start_ms + self.sampler.step) / 1000.0
        self.minute.append(close)
        before = self.detector.level
        rd = self.detector.update(r)
        if rd.level != before or self.level_since is None:
            self.level_since = end_s
        if rd.escalated:
            changes.append(("spike", before.name.lower(), rd.level.name.lower()))

        self.har.update(r, n_trades)
        old = self.forecast.regime if self.forecast is not None else None
        self.forecast = self.har.forecast(rd.slow_var, end_s)
        if self.forecast is not None:
            new = self.forecast.regime
            if new != old or self.regime_since is None:
                self.regime_since = end_s
            if new <= Regime.NORMAL:
                self.regime_armed = True
            elif new == Regime.STRESSED and self.regime_armed and old is not None:
                changes.append(("regime", old.name.lower(), new.name.lower()))
                self.regime_armed = False
        if not report:
            return []

        ppy = self.config.bars_per_year
        fc = self.forecast
        fast, slow = math.sqrt(rd.fast_var * ppy), math.sqrt(rd.slow_var * ppy)
        self.history.append(
            (
                int(end_s * 1000),
                close,
                fast,
                slow,
                rd.ratio,
                int(rd.level),
                fc.vol if fc else float("nan"),
                int(fc.regime) if fc else -1,
            )
        )
        full = len(self.minute) == self.minute.maxlen
        move = math.log(close / self.minute[0]) if full else float("nan")
        nan = float("nan")
        return [
            dict(
                time=pd.Timestamp(end_s, unit="s", tz="UTC"),
                kind=kind,
                previous=prev,
                level=new,
                forecast_ratio=fc.ratio if fc else nan,
                forecast_vol=fc.vol if fc else nan,
                ratio=rd.ratio,
                fast_vol=fast,
                slow_vol=slow,
                price=close,
                move_1m=move,
            )
            for kind, prev, new in changes
        ]


class LiveMonitor:
    def __init__(
        self,
        instruments: Iterable[str],
        config: DetectorConfig | None = None,
        on_alert: Callable[[Alert], None] | None = None,
        models: dict[str, HARModel] | None = None,
    ):
        """``models`` maps instruments to HAR coefficients; defaults ship with the package."""
        self.config = config or DetectorConfig()
        instruments = list(instruments)
        models = models or {}
        self.models = {i: models.get(i) or default_model(i) for i in instruments}
        self._state = {i: self._new_state(i) for i in instruments}
        self._dvol: dict[str, float] = {}
        self._alerts: deque[Alert] = deque(maxlen=500)
        self._on_alert = on_alert
        self._lock = threading.Lock()

    def _new_state(self, instrument: str) -> _InstrumentState:
        return _InstrumentState(self.config, self.models[instrument])

    @property
    def instruments(self) -> list[str]:
        return list(self._state)

    @property
    def dvol_indices(self) -> list[str]:
        roots = {i.split("-")[0] for i in self._state}
        return sorted(DVOL_INDEX[r] for r in roots if r in DVOL_INDEX)

    @property
    def seed_span(self) -> pd.Timedelta:
        """History downloaded at start-up: the warm-up plus two hours shown on the charts."""
        warm = self.config.warmup_bars * self.config.bar_seconds
        return pd.Timedelta(seconds=warm + 2 * 3600)

    # -- seeding ------------------------------------------------------------

    def fetch_seed(self, rest: DeribitREST, end: pd.Timestamp | None = None) -> dict:
        end = end or pd.Timestamp.now(tz="UTC")
        return {i: rest.trades(i, end - self.seed_span, end) for i in self._state}

    def apply_seed(self, trades: dict[str, pd.DataFrame]) -> None:
        """Rebuild every instrument's state from a trade history.

        Escalations found in the history are recorded with ``live=False`` so that the alert
        list matches the charts, but ``on_alert`` is not called for them.
        """
        for instrument, df in trades.items():
            st = self._new_state(instrument)
            if len(df):
                closes = bar_closes(df, self.config.bar_seconds)
                step = st.sampler.step
                buckets = df["timestamp"].to_numpy(dtype=np.int64) // step
                counts = pd.Series(1, index=buckets).groupby(level=0).sum()
                epoch = pd.Timestamp(0, tz="UTC")
                starts = ((closes.index - epoch) // pd.Timedelta(milliseconds=1)).to_numpy()
                # Completed bars go through the detector; the last one is still in progress.
                replayed = []
                for t, c in zip(starts[:-1], closes.to_numpy()[:-1], strict=True):
                    for draft in st.add_bar(int(t), float(c), int(counts.get(t // step, 0))):
                        replayed.append(self._make_alert(instrument, draft, live=False))
                last = int(starts[-1] // step)
                st.sampler = GridSampler(self.config.bar_seconds, last, float(closes.iloc[-1]))
                st.sampler.trades = int(counts.get(last, 0))
                st.seed_end = (int(df["timestamp"].iloc[-1]), set(df["trade_id"].iloc[-200:]))
            with self._lock:
                self._state[instrument] = st
                if len(df):
                    known = {(a.instrument, a.time, a.kind) for a in self._alerts}
                    fresh = [a for a in replayed if (a.instrument, a.time, a.kind) not in known]
                    merged = sorted([*self._alerts, *fresh], key=lambda a: a.time)
                    self._alerts = deque(merged, maxlen=self._alerts.maxlen)
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
            for draft in st.add_bar(start_ms, close, n):
                alert = self._make_alert(instrument, draft)
                self._alerts.append(alert)
                out.append(alert)
        return out

    def _make_alert(self, instrument: str, draft: dict, live: bool = True) -> Alert:
        dvol = self._dvol.get(DVOL_INDEX.get(instrument.split("-")[0], "")) if live else None
        return Alert(instrument=instrument, dvol=dvol, live=live, **draft)

    # -- output ------------------------------------------------------------

    def _snapshot(self, instrument: str, st: _InstrumentState) -> Snapshot | None:
        if st.detector is None or st.forecast is None or not st.history:
            return None
        det, fc, ppy = st.detector, st.forecast, self.config.bars_per_year
        price = st.sampler.close if st.sampler.close is not None else st.closes[-1]
        full_minute = len(st.closes) == st.closes.maxlen
        return Snapshot(
            instrument=instrument,
            time=pd.Timestamp(st.history[-1][0], unit="ms", tz="UTC"),
            price=price,
            regime=fc.regime,
            regime_since=pd.Timestamp(st.regime_since, unit="s", tz="UTC"),
            forecast_ratio=fc.ratio,
            forecast_vol=fc.vol,
            sigma_5m=price * fc.vol * math.sqrt(HORIZON_SECONDS / SECONDS_PER_YEAR),
            level=det.level,
            ratio=det.ratio,
            fast_vol=math.sqrt(det.fast_var * ppy),
            slow_vol=math.sqrt(det.slow_var * ppy),
            move_1m=math.log(st.closes[-1] / st.closes[0]) if full_minute else float("nan"),
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
        """Recent bars: price, annualized vols, spike ratio and level, forecast and regime."""
        with self._lock:
            rows = list(self._state[instrument].history)
        cols = [
            "time",
            "price",
            "fast_vol",
            "slow_vol",
            "ratio",
            "level",
            "forecast_vol",
            "regime",
        ]
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
