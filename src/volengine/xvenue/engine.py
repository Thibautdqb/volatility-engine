"""Cross-venue opportunity engine: price dislocations and volatility divergence.

For a pair of markets A and B quoting the same asset, define in log terms (bp):

    spread  = log(mid_A) - log(mid_B), in USD (USDT quotes converted at USDT/USD)
    fair    = EWMA of the spread, half-life 10 minutes: the usual basis between the two
    edge    = log(bid_A) - log(ask_B) - fair      (sell A at its bid, buy B at its ask)
              and symmetrically log(bid_B) - log(ask_A) + fair
    net     = edge - taker fee A - taker fee B
    buffer  = z * sqrt(sigma_A^2 + sigma_B^2) * sqrt(latency)

``sigma`` is each market's volatility per square root of a second, an EWMA of 1-second
mid returns. The buffer is how far prices can move, at the current volatility, while
the two orders travel; an opportunity is flagged only when ``net > buffer``. Higher
volatility therefore demands a larger edge, which is the volatility dimension of the
liquidity signal.

A market is ignored while its quote is older than ``max_quote_age`` or while it has
printed no trade for ``max_silence`` although the other one has: that is a venue that
is halted or lagging (Deribit stopped trading for 24 minutes on 2026-10-21, which a
naive spread reads as a 50 bp arbitrage).

The engine also reports *volatility divergence*: one market's short-term volatility
reaching ``vol_ratio`` times the other's, i.e. agitation starting on one venue first.

Everything runs on the receipt clock (``local_ms``), so a replay of a recording gives
the same flags as live.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from volengine.xvenue.feeds import QUOTE_CCY, Disconnected, Event, Print, Quote, Rate

TAKER_FEES_BP = {  # public base-tier taker fees; override with what you actually pay
    "deribit:perp": 5.0,
    "binance:perp": 5.0,
    "binance:spot": 10.0,
    "coinbase:spot": 40.0,
}


@dataclass(frozen=True)
class EngineConfig:
    latency_ms: float = 300.0  # time for both legs to reach the venues
    z: float = 2.0  # buffer, in standard deviations of the move during the latency
    fair_half_life_s: float = 600.0
    vol_half_life_s: float = 60.0
    max_quote_age_ms: float = 2000.0
    max_silence_ms: float = 60_000.0
    vol_ratio: float = 2.0  # divergence when one market's vol is this many times the other's
    min_vol_bp: float = 0.5  # per sqrt(s); below this both markets are quiet, no divergence
    warmup_s: float = 120.0  # fair spread and volatilities need this long first
    fees_bp: dict = field(default_factory=lambda: dict(TAKER_FEES_BP))


@dataclass
class _Market:
    quote: Quote | None = None
    last_print_ms: float = -math.inf
    last_mid_usd: float | None = None
    last_tick_s: int | None = None
    var: float | None = None  # per-second variance of log mid, EWMA
    first_ms: float | None = None

    def mid(self) -> float | None:
        return None if self.quote is None else (self.quote.bid + self.quote.ask) / 2


@dataclass(frozen=True)
class Opportunity:
    """One episode of ``net > buffer`` between two markets (sell ``sell_on``, buy ``buy_on``)."""

    asset: str
    sell_on: str
    buy_on: str
    start_ms: float
    end_ms: float
    peak_net_bp: float
    peak_edge_bp: float
    buffer_bp: float  # at the peak
    size: float  # base units available at the peak, min of both sides
    notional_usd: float
    sigma_sell_bp: float
    sigma_buy_bp: float
    updates: int

    @property
    def duration_ms(self) -> float:
        return self.end_ms - self.start_ms


@dataclass(frozen=True)
class Divergence:
    asset: str
    leader: str  # the market whose volatility is higher
    other: str
    time_ms: float
    sigma_leader_bp: float
    sigma_other_bp: float


@dataclass
class _Pair:
    a: str
    b: str
    fair: float | None = None
    open: dict = field(default_factory=dict)  # direction -> running episode stats
    diverging: bool = False


class Engine:
    def __init__(
        self, assets=("BTC",), markets=tuple(QUOTE_CCY), config: EngineConfig | None = None
    ):
        self.config = config or EngineConfig()
        self.assets = tuple(assets)
        self.markets = tuple(markets)
        self.state = {(m, a): _Market() for m in self.markets for a in self.assets}
        self.pairs = {
            a: [_Pair(x, y) for i, x in enumerate(self.markets) for y in self.markets[i + 1 :]]
            for a in self.assets
        }
        self.usdt_usd: float | None = None
        self.opportunities: list[Opportunity] = []
        self.divergences: list[Divergence] = []
        self.now_ms = 0.0

    # -- helpers ------------------------------------------------------------

    def _usd(self, market: str, price: float) -> float | None:
        if QUOTE_CCY[market] == "USD":
            return price
        return None if self.usdt_usd is None else price * self.usdt_usd

    def sigma_bp(self, market: str, asset: str) -> float | None:
        st = self.state[(market, asset)]
        return None if st.var is None else 1e4 * math.sqrt(st.var)

    def _usable(self, market: str, asset: str, other: str) -> bool:
        st, ot = self.state[(market, asset)], self.state[(other, asset)]
        c = self.config
        if st.quote is None or st.first_ms is None or st.var is None:
            return False
        if self.now_ms - st.first_ms < c.warmup_s * 1000:
            return False
        if self.now_ms - st.quote.local_ms > c.max_quote_age_ms:
            return False
        # Silent while the other market trades: halted or lagging.
        if ot.last_print_ms - st.last_print_ms > c.max_silence_ms:
            return False
        return True

    # -- clock: volatility and fair spread on a 1-second grid -----------------

    def _tick(self, now_ms: float) -> None:
        sec = int(now_ms // 1000)
        c = self.config
        lam_v = 0.5 ** (1.0 / c.vol_half_life_s)
        for (market, _), st in self.state.items():
            mid = st.mid()
            if mid is None:
                continue
            usd = self._usd(market, mid)
            if usd is None:
                continue
            if st.last_tick_s is None:
                st.last_tick_s, st.last_mid_usd = sec, usd
                continue
            steps = sec - st.last_tick_s
            if steps <= 0:
                continue
            # Variance per second, spread over the seconds elapsed since the last tick.
            r2 = math.log(usd / st.last_mid_usd) ** 2 / steps
            lam = lam_v**steps
            st.var = r2 if st.var is None else lam * st.var + (1 - lam) * r2
            st.last_tick_s, st.last_mid_usd = sec, usd
        lam_f = 0.5 ** (1.0 / c.fair_half_life_s)
        for asset, pairs in self.pairs.items():
            for p in pairs:
                spread = self._spread(p, asset)
                if spread is not None:
                    p.fair = spread if p.fair is None else lam_f * p.fair + (1 - lam_f) * spread

    def _spread(self, p: _Pair, asset: str) -> float | None:
        ma, mb = self.state[(p.a, asset)].mid(), self.state[(p.b, asset)].mid()
        if ma is None or mb is None:
            return None
        ua, ub = self._usd(p.a, ma), self._usd(p.b, mb)
        if ua is None or ub is None:
            return None
        return math.log(ua / ub)

    # -- events ---------------------------------------------------------------

    def on_event(self, ev: Event) -> list[Opportunity | Divergence]:
        """Feed one event; returns the episodes that ended and divergences that started."""
        out: list[Opportunity | Divergence] = []
        if isinstance(ev, Rate):
            if ev.name == "USDT/USD":
                self.usdt_usd = ev.value
            return out
        if isinstance(ev, Disconnected):
            for (market, _), st in self.state.items():
                if market == ev.market:
                    st.quote = None
            return out
        prev_sec = int(self.now_ms // 1000)
        self.now_ms = max(self.now_ms, ev.local_ms)
        if int(self.now_ms // 1000) > prev_sec:
            self._tick(self.now_ms)
        if (ev.market, ev.asset) not in self.state:
            return out
        st = self.state[(ev.market, ev.asset)]
        if st.first_ms is None:
            st.first_ms = ev.local_ms
        if isinstance(ev, Print):
            st.last_print_ms = ev.local_ms
            return out
        st.quote = ev
        for p in self.pairs[ev.asset]:
            if ev.market in (p.a, p.b):
                out += self._evaluate(p, ev.asset)
        return out

    def _evaluate(self, p: _Pair, asset: str) -> list:
        c = self.config
        out: list = []
        ok = p.fair is not None and self._usable(p.a, asset, p.b) and self._usable(p.b, asset, p.a)
        if not ok:
            for direction in list(p.open):
                out.append(self._close(p, asset, direction))
            return out
        qa, qb = self.state[(p.a, asset)].quote, self.state[(p.b, asset)].quote
        sa, sb = self.sigma_bp(p.a, asset), self.sigma_bp(p.b, asset)
        buffer = c.z * math.hypot(sa, sb) * math.sqrt(c.latency_ms / 1000.0)
        fees = c.fees_bp[p.a] + c.fees_bp[p.b]
        ua = self._usd(p.a, 1.0)
        ub = self._usd(p.b, 1.0)
        # direction (sell on, buy on) -> edge, sizes, sell price in USD, sigmas
        sell_a = 1e4 * (math.log(qa.bid * ua / (qb.ask * ub)) - p.fair)
        sell_b = 1e4 * (math.log(qb.bid * ub / (qa.ask * ua)) + p.fair)
        legs = {
            (p.a, p.b): (sell_a, qa.bid_size, qb.ask_size, qa.bid * ua, sa, sb),
            (p.b, p.a): (sell_b, qb.bid_size, qa.ask_size, qb.bid * ub, sb, sa),
        }
        for direction, (edge, sell_size, buy_size, px, s_sell, s_buy) in legs.items():
            net = edge - fees
            if net > buffer:
                size = min(sell_size, buy_size)
                ep = p.open.get(direction)
                if ep is None:
                    p.open[direction] = ep = {"start": self.now_ms, "peak": -math.inf, "updates": 0}
                ep["updates"] += 1
                ep["end"] = self.now_ms
                if net > ep["peak"]:
                    ep.update(
                        peak=net,
                        edge=edge,
                        buffer=buffer,
                        size=size,
                        px=px,
                        s_sell=s_sell,
                        s_buy=s_buy,
                    )
            elif direction in p.open:
                out.append(self._close(p, asset, direction))
        # Volatility divergence between the two markets.
        hi, lo = (p.a, p.b) if sa >= sb else (p.b, p.a)
        s_hi, s_lo = max(sa, sb), min(sa, sb)
        diverging = s_hi >= c.min_vol_bp and s_hi >= c.vol_ratio * max(s_lo, 1e-9)
        if diverging and not p.diverging:
            d = Divergence(asset, hi, lo, self.now_ms, s_hi, s_lo)
            self.divergences.append(d)
            out.append(d)
        p.diverging = diverging
        return out

    def _close(self, p: _Pair, asset: str, direction) -> Opportunity:
        ep = p.open.pop(direction)
        opp = Opportunity(
            asset=asset,
            sell_on=direction[0],
            buy_on=direction[1],
            start_ms=ep["start"],
            end_ms=max(ep["end"], self.now_ms),
            peak_net_bp=ep["peak"],
            peak_edge_bp=ep["edge"],
            buffer_bp=ep["buffer"],
            size=ep["size"],
            notional_usd=ep["size"] * ep["px"],
            sigma_sell_bp=ep["s_sell"],
            sigma_buy_bp=ep["s_buy"],
            updates=ep["updates"],
        )
        self.opportunities.append(opp)
        return opp

    def snapshot(self, asset: str) -> list[dict]:
        """Current spread, fair, best net edge and buffer for every pair."""
        rows = []
        c = self.config
        for p in self.pairs[asset]:
            spread = self._spread(p, asset)
            sa, sb = self.sigma_bp(p.a, asset), self.sigma_bp(p.b, asset)
            usable = (
                p.fair is not None
                and self._usable(p.a, asset, p.b)
                and self._usable(p.b, asset, p.a)
            )
            row = {"pair": f"{p.a} / {p.b}", "usable": usable}
            if spread is not None and p.fair is not None:
                row["spread dev bp"] = 1e4 * (spread - p.fair)
            if sa is not None and sb is not None:
                row["sigma bp/s"] = (sa, sb)
                row["buffer bp"] = c.z * math.hypot(sa, sb) * math.sqrt(c.latency_ms / 1000.0)
            row["fees bp"] = c.fees_bp[p.a] + c.fees_bp[p.b]
            rows.append(row)
        return rows
