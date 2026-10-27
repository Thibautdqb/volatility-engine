"""Command-line interface.

volengine live BTC-PERPETUAL ETH-PERPETUAL --bar 60
volengine report BTC-PERPETUAL
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)


def _pct(x: float, digits: int = 1) -> str:
    return "n/a" if x is None or pd.isna(x) else f"{100 * x:.{digits}f}%"


def _ci(bounds, digits: int = 1) -> str:
    return f"[{_pct(bounds[0], digits)}, {_pct(bounds[1], digits)}]"


def _p(x: float) -> str:
    return "n/a" if pd.isna(x) else ("<0.001" if x < 1e-3 else f"{x:.3f}")


# ---------------------------------------------------------------------------
# live
# ---------------------------------------------------------------------------


def cmd_live(args) -> None:
    from volengine.deribit.rest import DeribitREST
    from volengine.live import LiveVolMonitor

    monitor = LiveVolMonitor(args.instruments, args.bar, args.window, args.half_life)
    with DeribitREST() as rest:
        monitor.seed(rest)

    def render(m: LiveVolMonitor) -> None:
        rows = [s.as_dict() for s in m.snapshots()]
        if not rows:
            return
        df = pd.DataFrame(rows).set_index("instrument")
        out = pd.DataFrame(
            {
                "price": df["price"].map("{:,.2f}".format),
                "ewma": df["ewma"].map(_pct),
                "cc": df["close_to_close"].map(_pct),
                "cc 95% ci": [
                    f"[{_pct(lo)}, {_pct(hi)}]"
                    for lo, hi in zip(df["cc_lower"], df["cc_upper"], strict=False)
                ],
                "parkinson": df["parkinson"].map(_pct),
                "garman-klass": df["garman_klass"].map(_pct),
                "yang-zhang": df["yang_zhang"].map(_pct),
                "dvol": df["dvol"].map(_pct),
                "bars": df["bars"],
                "lambda": df["lam"].map("{:.4f}".format),
            }
        )
        stamp = pd.Timestamp.now(tz="UTC").strftime("%H:%M:%S")
        sys.stdout.write("\x1b[2J\x1b[H")
        print(
            f"annualized volatility, {args.bar:g}s bars, window {args.window} bars   {stamp} UTC\n"
        )
        print(out.to_string())
        sys.stdout.flush()

    render(monitor)
    try:
        asyncio.run(monitor.run(on_bar=render))
    except KeyboardInterrupt:
        pass


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def cmd_report(args) -> None:
    from volengine.research import run_report

    rep = run_report(args.instrument, start=args.start, train=args.train)
    g = rep.garch
    print(f"{rep.instrument}  {rep.start.date()} -> {rep.end.date()}  ({rep.n_days} sessions)\n")

    rv = rep.realized_vol
    print("Realized volatility, last 365 sessions")
    print(f"  {_pct(rv.vol)}  95% CI {_ci((rv.lower, rv.upper))}  (stationary block bootstrap)\n")

    a = rep.arch_effects
    print(f"ARCH-LM(5) on daily returns: stat {a.statistic:.1f}, p {_p(a.p_value)}")
    print(
        "  -> volatility clustering " + ("present" if a.p_value < 0.05 else "not detected") + "\n"
    )

    print("GARCH(1,1), QMLE with Bollerslev-Wooldridge robust standard errors")
    print(g.summary().to_string(float_format=lambda x: f"{x:.4g}"))
    print(
        f"  persistence {g.persistence:.4f}, half-life {g.half_life:.1f} sessions, "
        f"long-run vol {_pct((g.unconditional_variance * 365) ** 0.5)}"
    )
    for name, t in g.diagnostics.items():
        print(f"  {name:<14} stat {t.statistic:9.2f}  p {_p(t.p_value)}")
    print(f"  EWMA decay fitted by QLIKE: lambda = {rep.ewma_lambda:.4f}\n")

    t = rep.forecast_table
    print(
        f"Out-of-sample one-day variance forecasts (n = {t.attrs['n']}, walk-forward, "
        f"proxy = hourly realized variance)"
    )
    print(
        f"Diebold-Mariano vs {t.attrs['benchmark']} on QLIKE; negative stat = better than benchmark"
    )
    print(t.to_string(float_format=lambda x: f"{x:.4g}"))
    print()

    if rep.vrp is not None:
        v = rep.vrp
        lo, hi = v.vol_spread.ci()
        print("Variance risk premium: DVOL vs subsequent 30-day realized vol")
        ind = v.vol_spread_independent
        print(
            f"  mean spread {100 * v.vol_spread.mean:.2f} vol pts, "
            f"95% CI [{100 * lo:.2f}, {100 * hi:.2f}], "
            f"Hansen-Hodrick t {v.vol_spread.t_stat:.2f} (df {v.vol_spread.df:.0f}), "
            f"p {_p(v.vol_spread.p_value)}"
        )
        print(
            f"  robustness, {ind.n} non-overlapping windows: mean {100 * ind.mean:.2f} pts, "
            f"t {ind.t_stat:.2f}, p {_p(ind.p_value)}"
        )
        print(
            f"  implied above realized on {_pct(v.share_positive, 0)} of days "
            f"(n = {v.vol_spread.n}, overlapping)"
        )
        print(f"  Mincer-Zarnowitz slope {v.mz_slope:.2f}, H0 (unbiased) p {_p(v.mz_unbiased_p)}\n")

    print("Volatility cone (annualized, close-to-close)")
    print(rep.cone.to_string(float_format=lambda x: f"{x:.3f}"))


# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="volengine", description="Realized and implied volatility toolkit."
    )
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    live = sub.add_parser("live", help="stream real-time volatility from Deribit")
    live.add_argument("instruments", nargs="*", default=["BTC-PERPETUAL", "ETH-PERPETUAL"])
    live.add_argument("--bar", type=float, default=60.0, help="bar length in seconds")
    live.add_argument("--window", type=int, default=120, help="bars in the rolling window")
    live.add_argument(
        "--half-life",
        type=float,
        default=None,
        help="EWMA half-life in seconds (default: decay fitted on history)",
    )
    live.set_defaults(func=cmd_live)

    rep = sub.add_parser("report", help="historical study: GARCH, forecast evaluation, VRP")
    rep.add_argument("instrument", nargs="?", default="BTC-PERPETUAL")
    rep.add_argument("--start", default="2019-01-01")
    rep.add_argument("--train", type=int, default=750, help="initial training sessions")
    rep.set_defaults(func=cmd_report)

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
