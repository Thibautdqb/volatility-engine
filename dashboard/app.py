"""Streamlit front-end for the live monitor.

streamlit run dashboard/app.py
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from volengine.detector import DetectorConfig, Level
from volengine.forecast import Regime
from volengine.live import LiveMonitor

st.set_page_config(page_title="volengine", layout="wide")

INSTRUMENTS = ["BTC-PERPETUAL", "ETH-PERPETUAL"]
REGIME_COLOR = {
    Regime.CALM: "#1565c0",
    Regime.NORMAL: "#2e7d32",
    Regime.ACTIVE: "#ef6c00",
    Regime.STRESSED: "#c62828",
}
REGIME_FILL = {Regime.ACTIVE: "rgba(239,108,0,0.10)", Regime.STRESSED: "rgba(198,40,40,0.18)"}
LEVEL_COLOR = {Level.NORMAL: "#757575", Level.ELEVATED: "#ef6c00", Level.SPIKE: "#c62828"}


def duration(td: pd.Timedelta) -> str:
    s = max(0, int(td.total_seconds()))
    return f"{s // 60}m {s % 60:02d}s" if s < 3600 else f"{s // 3600}h {s % 3600 // 60:02d}m"


def price(x: float) -> str:
    return f"{x:,.1f}" if x >= 1000 else f"{x:,.2f}"


def bp(x) -> str:
    return "n/a" if x is None or pd.isna(x) else f"{1e4 * x:+.1f} bp"


def pct(x, d=1):
    return "n/a" if x is None or pd.isna(x) else f"{100 * x:.{d}f}%"


def badge(text: str, color: str) -> str:
    return (
        f"<span style='background:{color};color:white;padding:2px 10px;border-radius:4px;"
        f"font-size:0.8em'>{text}</span>"
    )


@st.cache_resource(show_spinner=False)
def get_monitor(instruments: tuple[str, ...], config: DetectorConfig) -> LiveMonitor:
    monitor = LiveMonitor(instruments, config)
    monitor.start_background()
    return monitor


def regime_spans(regimes: pd.Series):
    """Contiguous runs of ACTIVE or STRESSED forecasts, as (start, end, regime)."""
    run_id = (regimes != regimes.shift()).cumsum()
    for _, run in regimes.groupby(run_id):
        rg = int(run.iloc[0])
        if rg in (Regime.ACTIVE, Regime.STRESSED):
            yield run.index[0], run.index[-1], Regime(rg)


def chart(h: pd.DataFrame) -> go.Figure:
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.5, 0.5], vertical_spacing=0.04
    )
    fig.add_trace(go.Scatter(x=h.index, y=h["price"], name="price", line=dict(width=1)), 1, 1)
    vols = [
        ("forecast_vol", "forecast, next 5m", dict(width=2)),
        ("fast_vol", "realized, last 1m", dict(width=1, dash="dot")),
        ("slow_vol", "normal (30m)", dict(width=1, color="grey")),
    ]
    for col, name, line in vols:
        fig.add_trace(go.Scatter(x=h.index, y=h[col], name=name, line=line), 2, 1)
    for x0, x1, rg in regime_spans(h["regime"]):
        fig.add_vrect(x0=x0, x1=x1, fillcolor=REGIME_FILL[rg], line_width=0)
    spikes = h[h["level"] == Level.SPIKE]
    fig.add_trace(
        go.Scatter(
            x=spikes.index,
            y=spikes["price"],
            mode="markers",
            name="spike",
            marker=dict(color="#c62828", size=5),
        ),
        1,
        1,
    )
    fig.update_yaxes(title_text="price", row=1, col=1)
    fig.update_yaxes(title_text="annualized vol", tickformat=".0%", row=2, col=1)
    fig.update_layout(
        template="plotly_white",
        height=480,
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", y=-0.08),
    )
    return fig


with st.sidebar:
    instruments = st.multiselect("Instruments", INSTRUMENTS, default=INSTRUMENTS)
    hours = st.slider("Chart window (h)", 0.5, 6.0, 2.0, 0.5)

st.title("Short-term volatility monitor")
st.caption(
    "Deribit perpetuals, 10-second bars, live over WebSocket. **P(violent)** = probability that "
    "the next 5 minutes move at least twice the normal (30-minute) volatility. **Regime** = "
    "rank of the 5-minute forecast among the training forecasts: CALM (lowest 25%), NORMAL, "
    "ACTIVE, STRESSED (top 5%). **Spike** = last minute vs normal: elevated 2x, spike 3x."
)

if not instruments:
    st.info("Select at least one instrument.")
    st.stop()
monitor = get_monitor(tuple(instruments), DetectorConfig())


@st.fragment(run_every=1)
def render() -> None:
    snaps = {s.instrument: s for s in monitor.snapshots()}
    if not snaps:
        st.info("Loading the last day of bars (store, then Deribit) to warm up the models...")
        return
    for inst in instruments:
        s = snaps.get(inst)
        if s is None:
            st.caption(f"{inst}: warming up")
            continue
        st.markdown(
            f"### {inst} &nbsp; {badge(s.regime.name, REGIME_COLOR[s.regime])} "
            f"<span style='font-size:0.7em;color:gray'>for {duration(s.time - s.regime_since)}"
            f"</span> &nbsp; {badge('spike: ' + s.level.name, LEVEL_COLOR[s.level])} "
            f"<span style='font-size:0.7em;color:gray'>· {s.time:%H:%M:%S} UTC · latency "
            f"{'n/a' if s.latency_ms is None else f'{s.latency_ms:.0f} ms'} · "
            f"sequence holes {s.seq_gaps}</span>",
            unsafe_allow_html=True,
        )
        c = st.columns(7)
        c[0].metric("Price", price(s.price), bp(s.move_1m), help="change over the last minute")
        c[6].metric(
            "P(violent)",
            pct(s.p_violent),
            help="P(vol over the next 5 minutes >= 2x normal); about 3% on average",
        )
        c[1].metric(
            "Next 5m vol",
            pct(s.forecast_vol),
            f"{s.forecast_ratio:.2f}x normal",
            delta_color="off",
            help="forecast, annualized",
        )
        c[2].metric(
            "1 sd move, 5m", f"±{price(s.sigma_5m)}", help="one standard deviation, forecast"
        )
        c[3].metric(
            "Vol 1m", pct(s.fast_vol), f"{s.ratio:.2f}x normal", delta_color="off", help="EWMA"
        )
        c[4].metric("Normal vol", pct(s.slow_vol), help="EWMA, half-life 30 minutes")
        c[5].metric("DVOL", pct(s.dvol), help="Deribit 30-day implied vol index")
        h = monitor.history(inst)
        h = h[(h.index >= h.index[-1] - pd.Timedelta(hours=hours)) & (h["regime"] >= 0)]
        st.plotly_chart(chart(h), width="stretch", key=f"chart-{inst}")

    alerts = monitor.alerts()
    st.subheader("Alerts")
    st.caption(
        "regime = the forecast enters STRESSED · spike = the detector escalates · "
        "history = found in the trades loaded at start-up, before the live feed"
    )
    if not alerts:
        st.caption("No alert since start.")
        return
    table = pd.DataFrame([a.as_dict() for a in reversed(alerts)])
    table["source"] = table.pop("live").map({True: "live", False: "history"})
    table["change"] = table.pop("previous") + " → " + table.pop("level")
    table["time"] = pd.to_datetime(table["time"]).dt.strftime("%H:%M:%S")
    for col in ("forecast_vol", "fast_vol", "slow_vol", "dvol"):
        table[col] = table[col].map(pct)
    for col in ("forecast_ratio", "ratio"):
        table[col] = table[col].map("{:.2f}x".format)
    table["move_1m"] = table["move_1m"].map(bp)
    table["price"] = table["price"].map(price)
    table = table.rename(
        columns={
            "forecast_vol": "next 5m vol",
            "forecast_ratio": "next 5m / normal",
            "fast_vol": "vol 1m",
            "ratio": "1m / normal",
            "slow_vol": "normal vol",
            "move_1m": "move 1m",
        }
    )
    table["p_violent"] = table["p_violent"].map(pct)
    table = table.rename(columns={"p_violent": "P(violent)"})
    cols = ["time", "instrument", "kind", "change", "P(violent)", "next 5m vol"]
    rest = [c for c in table.columns if c not in cols]
    st.dataframe(table[cols + rest], hide_index=True, width="stretch")


render()
