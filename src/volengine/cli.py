"""Command-line interface.

volengine live BTC-PERPETUAL ETH-PERPETUAL --log alerts.jsonl
volengine study BTC-PERPETUAL --days 14
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys

import pandas as pd

from volengine.detector import DetectorConfig, Level

_COLORS = {Level.NORMAL: "", Level.ELEVATED: "\x1b[33m", Level.SPIKE: "\x1b[1;31m"}
_RESET = "\x1b[0m"


def _pct(x: float | None, digits: int = 1, sign: bool = False) -> str:
    if x is None or pd.isna(x):
        return "n/a"
    return f"{100 * x:{'+' if sign else ''}.{digits}f}%"


def _bp(x: float | None) -> str:
    return "n/a" if x is None or pd.isna(x) else f"{1e4 * x:+.1f}bp"


def _duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds // 3600}h{seconds % 3600 // 60:02d}m"


def _price(x: float) -> str:
    return f"{x:,.2f}" if x < 1000 else f"{x:,.1f}"


def _config(args) -> DetectorConfig:
    return DetectorConfig(
        bar_seconds=args.bar,
        fast_half_life=args.fast,
        slow_half_life=args.slow,
        elevated=args.elevated,
        spike=args.spike,
        reset=args.reset,
    )


# ---------------------------------------------------------------------------
# live
# ---------------------------------------------------------------------------


def cmd_live(args) -> None:
    from volengine.live import Alert, LiveSpikeMonitor

    color = sys.stdout.isatty() and not args.no_color
    log_file = open(args.log, "a", buffering=1) if args.log else None

    def on_alert(a: Alert) -> None:
        if log_file is not None:
            log_file.write(json.dumps(a.as_dict()) + "\n")
        if args.bell and a.level == Level.SPIKE:
            sys.stdout.write("\a")

    config = _config(args)
    monitor = LiveSpikeMonitor(args.instruments, config, on_alert=on_alert)

    def paint(text: str, level: Level) -> str:
        return f"{_COLORS[level]}{text}{_RESET}" if color and _COLORS[level] else text

    def render(m: LiveSpikeMonitor) -> None:
        snaps = m.snapshots()
        if not snaps:
            return
        rows = []
        for s in snaps:
            rows.append(
                {
                    "": s.instrument,
                    "price": _price(s.price),
                    "state": s.level.name,
                    "for": _duration((s.time - s.level_since).total_seconds()),
                    "ratio": f"{s.ratio:.2f}x",
                    "vol 1m": _pct(s.fast_vol),
                    "normal": _pct(s.slow_vol),
                    "move 1m": _bp(s.move_1m),
                    "1sd 1m": _price(s.sigma_1m),
                    "trades": f"{s.activity:.1f}x",
                    "dvol": _pct(s.dvol),
                }
            )
        table = pd.DataFrame(rows).to_string(index=False).splitlines()
        lines = [table[0]] + [paint(t, s.level) for t, s in zip(table[1:], snaps, strict=True)]
        stamp = max(s.time for s in snaps).strftime("%H:%M:%S")
        out = [
            f"{config.bar_seconds:g}s bars | vol 1m = EWMA half-life {config.fast_half_life:g}s, "
            f"normal = {config.slow_half_life / 60:g}m | elevated >= {config.elevated:g}x, "
            f"spike >= {config.spike:g}x, reset < {config.reset:g}x | {stamp} UTC",
            "",
            *lines,
        ]
        alerts = m.alerts()[-args.show_alerts :]
        if alerts:
            out += ["", "alerts"]
            for a in reversed(alerts):
                out.append(
                    paint(
                        f"{a.time:%H:%M:%S}  {a.instrument:<14} "
                        f"{a.previous.name.lower():>8} -> {a.level.name.lower():<8}  "
                        f"ratio {a.ratio:4.2f}x  vol 1m {_pct(a.fast_vol):>6}  "
                        f"normal {_pct(a.slow_vol):>6}  move 1m {_bp(a.move_1m):>7}  "
                        f"@ {_price(a.price)}" + ("" if a.live else "  (history)"),
                        a.level,
                    )
                )
        sys.stdout.write("\x1b[2J\x1b[H" + "\n".join(out) + "\n")
        sys.stdout.flush()

    print(f"connecting and loading the last {monitor.seed_span} of trades...")
    try:
        asyncio.run(monitor.run(on_bar=render))
    except KeyboardInterrupt:
        pass
    finally:
        if log_file is not None:
            log_file.close()


# ---------------------------------------------------------------------------
# study
# ---------------------------------------------------------------------------


def cmd_study(args) -> None:
    from volengine.study import run_study

    study = run_study(
        args.instrument,
        days=args.days,
        end=args.end,
        config=_config(args),
        horizon_minutes=args.horizon,
        violent=args.violent,
    )
    print(study.summary())


# ---------------------------------------------------------------------------


def _add_detector_args(p: argparse.ArgumentParser) -> None:
    d = DetectorConfig()
    g = p.add_argument_group("detector")
    g.add_argument("--bar", type=float, default=d.bar_seconds, help="bar length, seconds")
    g.add_argument("--fast", type=float, default=d.fast_half_life, help="fast half-life, seconds")
    g.add_argument("--slow", type=float, default=d.slow_half_life, help="slow half-life, seconds")
    g.add_argument("--elevated", type=float, default=d.elevated, help="elevated ratio threshold")
    g.add_argument("--spike", type=float, default=d.spike, help="spike ratio threshold")
    g.add_argument("--reset", type=float, default=d.reset, help="ratio below which it resets")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="volengine", description="Short-term volatility spike monitor for Deribit perpetuals."
    )
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    live = sub.add_parser("live", help="monitor volatility in real time and flag spikes")
    live.add_argument("instruments", nargs="*", default=["BTC-PERPETUAL", "ETH-PERPETUAL"])
    live.add_argument("--log", help="append alerts to this file, one JSON object per line")
    live.add_argument("--bell", action="store_true", help="ring the terminal bell on spikes")
    live.add_argument("--show-alerts", type=int, default=10, help="alerts listed under the table")
    live.add_argument("--no-color", action="store_true")
    _add_detector_args(live)
    live.set_defaults(func=cmd_live)

    study = sub.add_parser("study", help="replay the detector on past trades and score it")
    study.add_argument("instrument", nargs="?", default="BTC-PERPETUAL")
    study.add_argument("--days", type=int, default=14, help="complete UTC days to replay")
    study.add_argument("--end", default=None, help="last day excluded (default: today)")
    study.add_argument("--horizon", type=int, default=5, help="evaluation window, minutes")
    study.add_argument("--violent", type=float, default=2.0, help="forward/normal vol ratio")
    _add_detector_args(study)
    study.set_defaults(func=cmd_study)

    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args.func(args)


if __name__ == "__main__":
    main()
