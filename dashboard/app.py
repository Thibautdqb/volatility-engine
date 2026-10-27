"""Streamlit front-end for volengine.

streamlit run dashboard/app.py
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from volengine.deribit.rest import DeribitREST
from volengine.live import LiveVolMonitor

st.set_page_config(page_title="volengine", layout="wide")

INSTRUMENTS = ["BTC-PERPETUAL", "ETH-PERPETUAL", "SOL_USDC-PERPETUAL"]


def pct(x, d=1):
    return "n/a" if x is None or pd.isna(x) else f"{100 * x:.{d}f}%"


def layout(fig: go.Figure, title: str, ytitle: str) -> go.Figure:
    fig.update_layout(
        title=title,
        yaxis_title=ytitle,
        template="plotly_white",
        height=420,
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", y=-0.15),
    )
    return fig


# ---------------------------------------------------------------------------
# Live
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Seeding from Deribit history...")
def get_monitor(instruments: tuple[str, ...], bar: float, window: int) -> LiveVolMonitor:
    monitor = LiveVolMonitor(instruments, bar_seconds=bar, window=window)
    with DeribitREST() as rest:
        monitor.seed(rest)
    monitor.start_background()
    return monitor


def live_page() -> None:
    c1, c2, c3 = st.columns([3, 1, 1])
    instruments = c1.multiselect("Instruments", INSTRUMENTS, default=INSTRUMENTS[:2])
    bar = c2.selectbox("Bar", [60.0, 300.0, 900.0], format_func=lambda s: f"{int(s // 60)} min")
    window = c3.number_input("Window (bars)", 30, 1000, 120, step=10)
    if not instruments:
        st.info("Select at least one instrument.")
        return
    monitor = get_monitor(tuple(instruments), bar, int(window))

    @st.fragment(run_every=2)
    def render() -> None:
        snaps = monitor.snapshots()
        cols = st.columns(len(snaps) or 1)
        for col, s in zip(cols, snaps, strict=False):
            col.metric(s.instrument, f"{s.price:,.2f}")
            col.metric("EWMA vol", pct(s.ewma), help=f"lambda = {s.lam:.4f} per bar")
            col.metric(
                "Close-to-close vol",
                pct(s.close_to_close),
                help=f"95% CI [{pct(s.cc_lower)}, {pct(s.cc_upper)}], {s.bars} bars, chi-square",
            )
            if s.dvol is not None:
                col.metric(
                    "DVOL (30d implied)",
                    pct(s.dvol),
                    delta=f"{100 * (s.dvol - s.ewma):+.1f} pts vs EWMA",
                    delta_color="off",
                )
        table = pd.DataFrame(
            [
                {
                    "instrument": s.instrument,
                    "close-to-close": pct(s.close_to_close),
                    "95% CI": f"[{pct(s.cc_lower)}, {pct(s.cc_upper)}]",
                    "Parkinson": pct(s.parkinson),
                    "Garman-Klass": pct(s.garman_klass),
                    "Yang-Zhang": pct(s.yang_zhang),
                    "bars": s.bars,
                }
                for s in snaps
            ]
        )
        st.dataframe(table, hide_index=True, width="stretch")
        fig = go.Figure()
        for inst in instruments:
            h = monitor.history(inst).iloc[-1500:]
            fig.add_trace(go.Scatter(x=h.index, y=h.values, mode="lines", name=inst))
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(layout(fig, "EWMA volatility (annualized)", "vol"), width="stretch")

    render()


# ---------------------------------------------------------------------------
# Research
# ---------------------------------------------------------------------------


@st.cache_data(ttl=6 * 3600, show_spinner="Downloading history and running walk-forward...")
def get_report(instrument: str, start: str):
    from volengine.research import run_report

    return run_report(instrument, start=start)


def research_page() -> None:
    c1, c2 = st.columns(2)
    instrument = c1.selectbox("Instrument", INSTRUMENTS[:2])
    start = c2.text_input("Start", "2019-01-01")
    rep = get_report(instrument, start)
    g = rep.garch

    m = st.columns(4)
    m[0].metric(
        "Realized vol (1y)",
        pct(rep.realized_vol.vol),
        help=f"95% CI [{pct(rep.realized_vol.lower)}, {pct(rep.realized_vol.upper)}]",
    )
    m[1].metric(
        "GARCH persistence", f"{g.persistence:.3f}", help=f"half-life {g.half_life:.1f} sessions"
    )
    m[2].metric("ARCH-LM p-value", f"{rep.arch_effects.p_value:.1e}")
    if rep.vrp is not None:
        m[3].metric(
            "Mean VRP",
            f"{100 * rep.vrp.vol_spread.mean:.1f} pts",
            help=f"Newey-West t = {rep.vrp.vol_spread.t_stat:.2f}",
        )

    st.subheader("GARCH(1,1), robust standard errors")
    st.dataframe(g.summary().style.format("{:.4g}"), width="stretch")

    st.subheader("Out-of-sample forecast comparison")
    st.caption(
        f"Walk-forward, n = {rep.forecast_table.attrs['n']}. Diebold-Mariano vs "
        f"{rep.forecast_table.attrs['benchmark']} on QLIKE; negative = better."
    )
    st.dataframe(rep.forecast_table.style.format("{:.4g}"), width="stretch")

    fig = go.Figure()
    cv = (g.conditional_variance * 365) ** 0.5
    fig.add_trace(go.Scatter(x=cv.index, y=cv, name="GARCH conditional vol", line=dict(width=1)))
    if rep.vrp is not None:
        s = rep.vrp.series
        fig.add_trace(go.Scatter(x=s.index, y=s["implied"], name="DVOL", line=dict(width=1)))
        fig.add_trace(
            go.Scatter(x=s.index, y=s["realized"], name="Realized next 30d", line=dict(width=1))
        )
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(
        layout(fig, "Conditional, implied and realized volatility", "vol"), width="stretch"
    )

    st.subheader("Volatility cone")
    st.dataframe(rep.cone.style.format("{:.3f}"), width="stretch")


# ---------------------------------------------------------------------------
# Earnings
# ---------------------------------------------------------------------------


@st.cache_data(ttl=900, show_spinner="Pulling calendars and option chains...")
def get_scan(tickers: tuple[str, ...], horizon: int, fdr: float) -> pd.DataFrame:
    from volengine.earnings.screener import scan

    return scan(list(tickers), horizon_days=horizon, fdr=fdr)


def earnings_page() -> None:
    st.caption(
        "H0: expected |move| to expiry equals the straddle-implied move. Edge is the vol-adjusted "
        "historical expectation minus the implied move, before transaction costs. "
        "Signals require a "
        "Benjamini-Hochberg q-value below the FDR level and at least 12 past announcements."
    )
    c1, c2, c3 = st.columns([4, 1, 1])
    raw = c1.text_input("Tickers", "AAPL MSFT AMZN GOOGL META NVDA AMD NFLX TSLA JPM XOM")
    horizon = c2.number_input("Horizon (days)", 1, 60, 21)
    fdr = c3.number_input("FDR", 0.01, 0.5, 0.10, step=0.01)
    tickers = tuple(t.strip().upper() for t in raw.replace(",", " ").split() if t.strip())
    if not st.button("Scan", type="primary"):
        return
    df = get_scan(tickers, int(horizon), float(fdr))
    if df.empty:
        st.warning("No ticker with an announcement in the horizon.")
    else:
        view = df[
            [
                "ticker",
                "event",
                "timing",
                "expiry",
                "implied_move",
                "event_move",
                "mean_abs_move",
                "predicted_move",
                "adj_edge",
                "adj_edge_lo",
                "adj_edge_hi",
                "adj_edge_p",
                "q_value",
                "hit_rate",
                "n",
                "signal",
            ]
        ]
        pct_cols = [
            "implied_move",
            "event_move",
            "mean_abs_move",
            "predicted_move",
            "adj_edge",
            "adj_edge_lo",
            "adj_edge_hi",
            "hit_rate",
        ]
        st.dataframe(
            view.style.format(
                {c: "{:.2%}" for c in pct_cols} | {"adj_edge_p": "{:.3f}", "q_value": "{:.3f}"}
            ),
            hide_index=True,
            width="stretch",
        )
        fig = go.Figure()
        fig.add_trace(go.Bar(x=df["ticker"], y=df["implied_move"], name="Straddle implied"))
        fig.add_trace(
            go.Bar(
                x=df["ticker"],
                y=df["predicted_move"],
                name="Vol-adjusted historical",
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=df["adj_edge_hi"] - df["adj_edge"],
                    arrayminus=df["adj_edge"] - df["adj_edge_lo"],
                ),
            )
        )
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(
            layout(fig, "Implied vs expected absolute move to expiry (95% CI)", "move"),
            width="stretch",
        )
    for t, err in df.attrs.get("errors", {}).items():
        st.caption(f"Skipped {t}: {err}")


pages = {"Live (Deribit)": live_page, "Research": research_page, "Earnings": earnings_page}
choice = st.sidebar.radio("View", list(pages))
pages[choice]()
