"""Command-line interface.

volengine live BTC-PERPETUAL ETH-PERPETUAL --log alerts.jsonl
volengine study BTC-PERPETUAL --end 2026-10-28
volengine fit BTC-PERPETUAL ETH-PERPETUAL --days 44 --out models.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys

import pandas as pd

from volengine.detector import DetectorConfig

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


def _severity(regime, level) -> int:
    """0 = plain, 1 = yellow, 2 = red: the worse of the forecast and the spike detector."""
    from volengine.forecast import Regime

    r = {Regime.ACTIVE: 1, Regime.STRESSED: 2}.get(regime, 0)
    return max(r, int(level))


def _load_models(path: str | None):
    if not path:
        return None
    from volengine.forecast import HARModel

    with open(path) as f:
        return {k: HARModel.from_dict(v) for k, v in json.load(f).items()}


def cmd_live(args) -> None:
    from volengine.live import Alert, LiveMonitor

    color = sys.stdout.isatty() and not args.no_color
    log_file = open(args.log, "a", buffering=1) if args.log else None
    colors = {0: "", 1: "\x1b[33m", 2: "\x1b[1;31m"}

    def on_alert(a: Alert) -> None:
        if log_file is not None:
            log_file.write(json.dumps(a.as_dict()) + "\n")
        if args.bell:
            sys.stdout.write("\a")

    config = _config(args)
    monitor = LiveMonitor(
        args.instruments, config, on_alert=on_alert, models=_load_models(args.models)
    )

    def paint(text: str, severity: int) -> str:
        return f"{colors[severity]}{text}{_RESET}" if color and severity else text

    def render(m: LiveMonitor) -> None:
        snaps = m.snapshots()
        if not snaps:
            return
        rows = [
            {
                "": s.instrument,
                "price": _price(s.price),
                "regime": s.regime.name,
                "for": _duration((s.time - s.regime_since).total_seconds()),
                "next 5m vol": _pct(s.forecast_vol),
                "vs normal": f"{s.forecast_ratio:.2f}x",
                "1sd 5m": _price(s.sigma_5m),
                "spike": s.level.name,
                "vol 1m": _pct(s.fast_vol),
                "normal": _pct(s.slow_vol),
                "move 1m": _bp(s.move_1m),
                "dvol": _pct(s.dvol),
            }
            for s in snaps
        ]
        table = pd.DataFrame(rows).to_string(index=False).splitlines()
        lines = [table[0]] + [
            paint(t, _severity(s.regime, s.level)) for t, s in zip(table[1:], snaps, strict=True)
        ]
        stamp = max(s.time for s in snaps).strftime("%H:%M:%S")
        out = [
            f"{stamp} UTC | regime = forecast of the next 5 minutes | "
            f"spike = vol 1m vs normal (30m): elevated >= {config.elevated:g}x, "
            f"spike >= {config.spike:g}x",
            "",
            *lines,
        ]
        alerts = m.alerts()[-args.show_alerts :]
        if alerts:
            out += ["", "alerts"]
            for a in reversed(alerts):
                sev = 2 if a.level in ("stressed", "spike") else 1
                out.append(
                    paint(
                        f"{a.time:%H:%M:%S}  {a.instrument:<14} {a.kind:<6} "
                        f"{a.previous:>8} -> {a.level:<8}  "
                        f"next 5m {_pct(a.forecast_vol):>6} ({a.forecast_ratio:.2f}x)  "
                        f"vol 1m {_pct(a.fast_vol):>6} ({a.ratio:.2f}x)  "
                        f"move 1m {_bp(a.move_1m):>7}  @ {_price(a.price)}"
                        + ("" if a.live else "  (history)"),
                        sev,
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
# study and fit
# ---------------------------------------------------------------------------


def cmd_study(args) -> None:
    from volengine.study import run_forecast_study, run_study

    config = _config(args)
    fc = run_forecast_study(
        args.instrument,
        train_days=args.train_days,
        test_days=args.days,
        end=args.end,
        config=config,
        violent=args.violent,
    )
    print("FIVE-MINUTE FORECAST AND REGIMES\n")
    print(fc.summary())
    spike = run_study(
        args.instrument,
        days=args.days,
        end=args.end,
        config=config,
        horizon_minutes=args.horizon,
        violent=args.violent,
    )
    print("\n\nSPIKE DETECTOR (same test period)\n")
    print(spike.summary())


def cmd_fit(args) -> None:
    from volengine.study import fit_model

    models = {
        i: fit_model(i, days=args.days, end=args.end, config=_config(args)).to_dict()
        for i in args.instruments
    }
    text = json.dumps(models, indent=2)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")
    print(text)


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
        prog="volengine",
        description="Five-minute volatility forecast, regimes and spike alerts for Deribit.",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    live = sub.add_parser("live", help="forecast, regime and spike alerts in real time")
    live.add_argument("instruments", nargs="*", default=["BTC-PERPETUAL", "ETH-PERPETUAL"])
    live.add_argument("--log", help="append alerts to this file, one JSON object per line")
    live.add_argument("--bell", action="store_true", help="ring the terminal bell on alerts")
    live.add_argument("--models", help="JSON file of HAR coefficients from 'volengine fit'")
    live.add_argument("--show-alerts", type=int, default=10, help="alerts listed under the table")
    live.add_argument("--no-color", action="store_true")
    _add_detector_args(live)
    live.set_defaults(func=cmd_live)

    study = sub.add_parser("study", help="fit on past trades, score on the days that follow")
    study.add_argument("instrument", nargs="?", default="BTC-PERPETUAL")
    study.add_argument("--days", type=int, default=14, help="test days (complete UTC days)")
    study.add_argument("--train-days", type=int, default=30, help="training days before them")
    study.add_argument("--end", default=None, help="last day excluded (default: today)")
    study.add_argument("--horizon", type=int, default=5, help="spike evaluation window, minutes")
    study.add_argument("--violent", type=float, default=2.0, help="forward/normal vol ratio")
    _add_detector_args(study)
    study.set_defaults(func=cmd_study)

    fit = sub.add_parser("fit", help="estimate the HAR coefficients on past trades")
    fit.add_argument("instruments", nargs="*", default=["BTC-PERPETUAL", "ETH-PERPETUAL"])
    fit.add_argument("--days", type=int, default=44, help="complete UTC days to fit on")
    fit.add_argument("--end", default=None, help="last day excluded (default: today)")
    fit.add_argument("--out", help="write the coefficients to this JSON file")
    _add_detector_args(fit)
    fit.set_defaults(func=cmd_fit)

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
