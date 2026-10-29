"""Streaming counterpart of :func:`volengine.data.bars.build_bars`.

Same rules: order-book trades only, taken in ``trade_seq`` order, late prints (stamped
in a bar that has already closed) ignored, and a bar without trades keeps the previous
close. Bars are identified by their start time in milliseconds.
"""

from __future__ import annotations

from dataclasses import dataclass

from volengine.data.bars import bar_ms


@dataclass
class Bar:
    start_ms: int
    close: float
    n: int = 0
    volume: float = 0.0
    buy_volume: float = 0.0
    liq_volume: float = 0.0


class GridSampler:
    """Build bars from trades; ``bucket``/``close`` resume from a known state."""

    def __init__(self, bar_seconds: float, bucket: int | None = None, close: float | None = None):
        self.step = bar_ms(bar_seconds)
        self.current: Bar | None = None
        if bucket is not None and close is not None:
            self.current = Bar(bucket * self.step, close)

    @property
    def close(self) -> float | None:
        return self.current.close if self.current else None

    def _roll(self, bucket: int) -> list[Bar]:
        cur = self.current
        done = [cur]
        # Empty bars between the last one and the new one keep the last price.
        start = cur.start_ms // self.step
        done.extend(Bar(b * self.step, cur.close) for b in range(start + 1, bucket))
        self.current = Bar(bucket * self.step, cur.close)
        return done

    def update(
        self, timestamp_ms: int, price: float, amount: float, is_buy: bool, liquidation: bool
    ) -> list[Bar]:
        """Add a book trade (in trade_seq order); returns the bars it completed."""
        bucket = timestamp_ms // self.step
        if self.current is None:
            self.current = Bar(bucket * self.step, price)
        elif bucket < self.current.start_ms // self.step:
            return []  # late print
        done = self._roll(bucket) if bucket > self.current.start_ms // self.step else []
        bar = self.current
        bar.close = price
        bar.n += 1
        bar.volume += amount
        bar.buy_volume += amount if is_buy else 0.0
        bar.liq_volume += amount if liquidation else 0.0
        return done

    def advance(self, now_ms: int) -> list[Bar]:
        """Close every bar that ended before ``now_ms``, even if no trade arrived since."""
        if self.current is None:
            return []
        bucket = now_ms // self.step
        return self._roll(bucket) if bucket > self.current.start_ms // self.step else []
