"""Command-line interface.

volengine data sync --start 2026-04-28          download and verify the trade tape
volengine data status                           what is stored, and is it complete
volengine backtest BTC-PERPETUAL                walk-forward evaluation
volengine fit --days 90 --out models.json       coefficients for the live monitor
volengine watch --notify ntfy:my-topic          alerts only, to a phone
volengine live                                  full real-time table
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time

import pandas as pd

from volengine.detector import DetectorConfig

INSTRUMENTS = ["BTC-PERPETUAL", "ETH-PERPETUAL"]
_COLORS = {0: "", 1: "\x1b[33m", 2: "\x1b[1;31m"}
_RESET = "\x1b[0m"


def _pct(x, digits: int = 1) -> str:
    return "n/a" if x is None or pd.isna(x) else f"{100 * x:.{digits}f}%"


def _bp(x) -> str:
    return "n/a" if x is None or pd.isna(x) else f"{1e4 * x:+.1f}bp"


def _duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds // 3600}h{seconds % 3600 // 60:02d}m"


def _price(x: float) -> str:
    return f"{x:,.2f}" if x < 1000 else f"{x:,.1f}"


def _today() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC").floor("D")


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------


def cmd_sync(args) -> None:
    from volengine.data.sync import sync

    end = args.end or _today()
    for inst in args.instruments:
        result = sync(
            inst,
            args.start,
            end,
            workers=args.workers,
            refresh=args.refresh,
            on_day=lambda d, n, i=inst: print(f"  {i} {d:%Y-%m-%d}: {n:,} trades", flush=True),
        )
        print(result.summary())


def cmd_status(args) -> None:
    from volengine.data.store import TradeStore

    store = TradeStore()
    print(f"store: {store.root}\n")
    for inst in args.instruments:
        s = store.status(inst)
        if s.empty:
            print(f"{inst}: nothing stored\n")
            continue
        holes = int((s["gap_before"].fillna(0) != 0).sum())
        span = pd.date_range(s.index[0], s.index[-1], freq="D")
        print(
            f"{inst}: {len(s)} days, {s.index[0]:%Y-%m-%d} to {s.index[-1]:%Y-%m-%d} "
            f"({len(span) - len(s)} days absent in between)\n"
            f"  {s['trades'].sum():,} trades: {s['book'].sum():,} book, "
            f"{s['block'].sum():,} block, {s['combo'].sum():,} combo\n"
            f"  missing sequence numbers inside days: {int(s['missing'].sum())}, "
            f"duplicates: {int(s['duplicates'].sum())}, holes across midnight: {holes}\n"
        )


# ---------------------------------------------------------------------------
# backtest and fit
# ---------------------------------------------------------------------------


def _features(instrument: str, start, end, config: DetectorConfig) -> pd.DataFrame:
    from volengine.data.bars import load_bars
    from volengine.features import compute

    return compute(load_bars(instrument, start, end, config.bar_seconds), config)


def cmd_backtest(args) -> None:
    from volengine.evaluation import WalkForwardConfig, evaluate

    config = DetectorConfig()
    feats = _features(args.instrument, args.start, args.end or _today(), config)
    wf = WalkForwardConfig(args.step, args.min_train, args.max_train)
    report = evaluate(feats, config, wf=wf, instrument=args.instrument, n_boot=args.boot)
    print(report.summary())
    if args.out:
        report.wf.windows.to_parquet(args.out)
        print(f"\nout-of-sample windows written to {args.out}")


def cmd_fit(args) -> None:
    from volengine.forecast import fit

    config = DetectorConfig()
    end = pd.Timestamp(args.end, tz="UTC") if args.end else _today()
    start = end - pd.Timedelta(days=args.days)
    feats = {i: _features(i, start, end, config) for i in args.instruments}
    models = {i: fit(f, config, instrument=i).to_dict() for i, f in feats.items()}
    if len(feats) > 1:  # one model on all instruments, for instruments without their own
        pooled = fit(list(feats.values()), config, instrument="pooled")
        models["pooled"] = pooled.to_dict()
    text = json.dumps(models, indent=1)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")
    for name, m in models.items():
        coefs = ", ".join(f"{k} {v:+.3f}" for k, v in m["coefs"].items())
        print(f"{name} ({m['trained_on']}, {m['n']} windows): {coefs}")


# ---------------------------------------------------------------------------
# live
# ---------------------------------------------------------------------------


def _load_models(path: str | None):
    if not path:
        return None
    from volengine.forecast import Model

    with open(path) as f:
        return {k: Model.from_dict(v) for k, v in json.load(f).items()}


def _monitor(args, on_alert):
    from volengine.live import LiveMonitor

    return LiveMonitor(
        args.instruments,
        DetectorConfig(),
        on_alert=on_alert,
        models=_load_models(args.models),
        threshold=args.threshold,
        spikes=not args.no_spike,
    )


def cmd_watch(args) -> None:
    """Quiet mode: one line of status now and then, and the alerts."""
    from volengine.live import Alert, LiveMonitor
    from volengine.notify import Notifier

    notifier = Notifier(args.notify or ["terminal"])
    if args.test:
        notifier.send({"test": True}, "volengine: test alert", "If you can read this, alerts work.")
        print(f"test alert sent to: {', '.join(notifier.targets)}")
        return

    def on_alert(a: Alert) -> None:
        notifier.send(a.as_dict(), a.title, a.body)

    monitor = _monitor(args, on_alert)
    last_status = [0.0]

    def status(m: LiveMonitor) -> None:
        snaps = m.snapshots()
        if not snaps or time.monotonic() - last_status[0] < 60 * args.status:
            return
        last_status[0] = time.monotonic()
        parts = [
            f"{s.instrument.split('-')[0]} stress {s.stress:4.1f} ({s.regime.name.lower()})"
            for s in snaps
        ]
        stamp = max(s.time for s in snaps).strftime("%H:%M")
        print(f"[{stamp} UTC] " + "  ".join(parts), flush=True)

    print(
        f"watching {', '.join(args.instruments)}: alert when stress >= {args.threshold:g}"
        f"{'' if args.no_spike else ' or a spike is under way'}; "
        f"notify: {', '.join(notifier.targets)}",
        flush=True,
    )
    try:
        asyncio.run(monitor.run(on_bar=status))
    except KeyboardInterrupt:
        pass


def cmd_live(args) -> None:
    """Full table, refreshed on every bar."""
    from volengine.forecast import Regime
    from volengine.live import Alert, LiveMonitor

    color = sys.stdout.isatty() and not args.no_color
    log_file = open(args.log, "a", buffering=1) if args.log else None

    def on_alert(a: Alert) -> None:
        if log_file is not None:
            log_file.write(json.dumps(a.as_dict()) + "\n")
        if args.bell:
            sys.stdout.write("\a")

    monitor = _monitor(args, on_alert)

    def paint(text: str, severity: int) -> str:
        return f"{_COLORS[severity]}{text}{_RESET}" if color and severity else text

    def severity(s) -> int:
        return {Regime.ACTIVE: 1, Regime.STRESSED: 2}.get(s.regime, 0)

    def render(m: LiveMonitor) -> None:
        snaps = m.snapshots()
        if not snaps:
            return
        rows = [
            {
                "": s.instrument,
                "price": _price(s.price),
                "stress": f"{s.stress:.1f}",
                "regime": s.regime.name,
                "for": _duration((s.time - s.regime_since).total_seconds()),
                "P(violent)": _pct(s.p_violent),
                "next 5m vol": _pct(s.forecast_vol),
                "vs normal": f"{s.forecast_ratio:.2f}x",
                "1sd 5m": _price(s.sigma_5m),
                "vol 1m": _pct(s.fast_vol),
                "normal": _pct(s.slow_vol),
                "move 1m": _bp(s.move_1m),
                "dvol": _pct(s.dvol),
            }
            for s in snaps
        ]
        table = pd.DataFrame(rows).to_string(index=False).splitlines()
        lines = [table[0]] + [paint(t, severity(s)) for t, s in zip(table[1:], snaps, strict=True)]
        stamp = max(s.time for s in snaps).strftime("%H:%M:%S")
        health = "  ".join(
            f"{s.instrument.split('-')[0]} latency "
            f"{'n/a' if s.latency_ms is None else f'{s.latency_ms:.0f}ms'}, seq holes {s.seq_gaps}"
            for s in snaps
        )
        out = [
            f"{stamp} UTC | stress 0-100 = rank of the 5-minute volatility forecast; "
            f"alert at {args.threshold:g} | {health}",
            "",
            *lines,
        ]
        alerts = m.alerts()[-args.show_alerts :]
        if alerts:
            out += ["", "alerts"]
            for a in reversed(alerts):
                tag = "" if a.live else "  (history)"
                out.append(paint(f"{a.title}{tag}\n  {a.body}", 2))
        sys.stdout.write("\x1b[2J\x1b[H" + "\n".join(out) + "\n")
        sys.stdout.flush()

    print(f"connecting and loading the last {monitor.seed_span} of bars...")
    try:
        asyncio.run(monitor.run(on_bar=render))
    except KeyboardInterrupt:
        pass
    finally:
        if log_file is not None:
            log_file.close()


# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="volengine",
        description="Volatility-stress indicator and alerts for Deribit perpetuals.",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    data = sub.add_parser("data", help="local store of the trade tape")
    dsub = data.add_subparsers(dest="data_command", required=True)
    s = dsub.add_parser("sync", help="download, verify and repair complete UTC days")
    s.add_argument("instruments", nargs="*", default=INSTRUMENTS)
    s.add_argument("--start", required=True, help="first day, e.g. 2026-04-28")
    s.add_argument("--end", default=None, help="day after the last one (default: today)")
    s.add_argument("--workers", type=int, default=4)
    s.add_argument("--refresh", action="store_true", help="download stored days again")
    s.set_defaults(func=cmd_sync)
    st = dsub.add_parser("status", help="stored days and their integrity")
    st.add_argument("instruments", nargs="*", default=INSTRUMENTS)
    st.set_defaults(func=cmd_status)

    bt = sub.add_parser("backtest", help="walk-forward out-of-sample evaluation")
    bt.add_argument("instrument", nargs="?", default="BTC-PERPETUAL")
    bt.add_argument("--start", default="2026-04-28")
    bt.add_argument("--end", default=None, help="day after the last one (default: today)")
    bt.add_argument("--step", type=int, default=7, help="days between refits")
    bt.add_argument("--min-train", type=int, default=28, help="days before the first test")
    bt.add_argument("--max-train", type=int, default=90, help="longest training window, days")
    bt.add_argument("--boot", type=int, default=1000, help="bootstrap replications")
    bt.add_argument("--out", help="write the out-of-sample windows to this Parquet file")
    bt.set_defaults(func=cmd_backtest)

    ft = sub.add_parser("fit", help="fit the forecast on the most recent stored days")
    ft.add_argument("instruments", nargs="*", default=INSTRUMENTS)
    ft.add_argument("--days", type=int, default=90)
    ft.add_argument("--end", default=None, help="day after the last one (default: today)")
    ft.add_argument("--out", help="write the models to this JSON file")
    ft.set_defaults(func=cmd_fit)

    def alarm_args(q: argparse.ArgumentParser) -> None:
        q.add_argument("instruments", nargs="*", default=INSTRUMENTS)
        q.add_argument(
            "--threshold",
            type=float,
            default=99.0,
            help="stress (0-100) that triggers an alert; 99 gives 4-6 alerts a day per asset",
        )
        q.add_argument("--no-spike", action="store_true", help="do not alert on spikes")
        q.add_argument("--models", help="JSON file written by 'volengine fit'")

    wt = sub.add_parser("watch", help="send an alert when volatility starts to pick up")
    alarm_args(wt)
    wt.add_argument(
        "--notify",
        action="append",
        metavar="TARGET",
        help="terminal, bell, desktop, ntfy:<topic>, webhook:<url>, file:<path>; repeatable",
    )
    wt.add_argument("--status", type=float, default=15, help="minutes between status lines")
    wt.add_argument("--test", action="store_true", help="send a test alert and exit")
    wt.set_defaults(func=cmd_watch)

    lv = sub.add_parser("live", help="full table of stress, forecast and alerts")
    alarm_args(lv)
    lv.add_argument("--log", help="append alerts to this file, one JSON object per line")
    lv.add_argument("--bell", action="store_true", help="ring the terminal bell on alerts")
    lv.add_argument("--show-alerts", type=int, default=5, help="alerts listed under the table")
    lv.add_argument("--no-color", action="store_true")
    lv.set_defaults(func=cmd_live)
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
