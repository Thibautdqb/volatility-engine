"""Streamlit front-end for the live spike monitor.

streamlit run dashboard/app.py
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from volengine.detector import DetectorConfig, Level
from volengine.live import LiveSpikeMonitor

st.set_page_config(page_title="volengine", layout="wide")

INSTRUMENTS = ["BTC-PERPETUAL", "ETH-PERPETUAL"]
LEVEL_COLOR = {Level.NORMAL: "#2e7d32", Level.ELEVATED: "#ef6c00", Level.SPIKE: "#c62828"}
LEVEL_FILL = {Level.ELEVATED: "rgba(239,108,0,0.12)", Level.SPIKE: "rgba(198,40,40,0.18)"}


def pct(x, d=1, sign=False):
    return "n/a" if x is None or pd.isna(x) else f"{100 * x:{'+' if sign else ''}.{d}f}%"


@st.cache_resource(show_spinner=False)
def get_monitor(instruments: tuple[str, ...], config: DetectorConfig) -> LiveSpikeMonitor:
    monitor = LiveSpikeMonitor(instruments, config)
    monitor.start_background()
    return monitor


def level_spans(levels: pd.Series):
    """Contiguous runs of elevated or spike bars, as (start, end, level)."""
    if levels.empty:
        return
    run_id = (levels != levels.shift()).cumsum()
    for _, run in levels.groupby(run_id):
        lv = Level(int(run.iloc[0]))
        if lv != Level.NORMAL:
            yield run.index[0], run.index[-1], lv


def chart(h: pd.DataFrame, config: DetectorConfig, title: str) -> go.Figure:
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.55, 0.45], vertical_spacing=0.04
    )
    fig.add_trace(go.Scatter(x=h.index, y=h["price"], name="price", line=dict(width=1)), 1, 1)
    fig.add_trace(
        go.Scatter(x=h.index, y=h["ratio"], name="vol 1m / normal", line=dict(width=1)), 2, 1
    )
    for y, name in [(config.elevated, "elevated"), (config.spike, "spike")]:
        fig.add_hline(y=y, line=dict(dash="dot", width=1), row=2, col=1, annotation_text=name)
    for x0, x1, lv in level_spans(h["level"]):
        fig.add_vrect(x0=x0, x1=x1, fillcolor=LEVEL_FILL[lv], line_width=0)
    fig.update_yaxes(title_text="price", row=1, col=1)
    fig.update_yaxes(title_text="ratio", row=2, col=1)
    fig.update_layout(
        title=title,
        template="plotly_white",
        height=460,
        margin=dict(l=10, r=10, t=40, b=10),
        showlegend=False,
    )
    return fig


with st.sidebar:
    instruments = st.multiselect("Instruments", INSTRUMENTS, default=INSTRUMENTS)
    d = DetectorConfig()
    bar = st.select_slider("Bar (s)", [5.0, 10.0, 15.0, 30.0], value=d.bar_seconds)
    elevated = st.number_input("Elevated at ratio", 1.2, 10.0, d.elevated, 0.1)
    spike = st.number_input("Spike at ratio", 1.5, 20.0, d.spike, 0.1)
    reset = st.number_input("Reset below ratio", 1.0, 5.0, d.reset, 0.1)
    hours = st.slider("Chart window (h)", 0.5, 6.0, 2.0, 0.5)

if not instruments:
    st.info("Select at least one instrument.")
    st.stop()
try:
    config = DetectorConfig(bar_seconds=bar, elevated=elevated, spike=spike, reset=reset)
except ValueError as exc:
    st.error(str(exc))
    st.stop()
monitor = get_monitor(tuple(instruments), config)


@st.fragment(run_every=2)
def render() -> None:
    snaps = {s.instrument: s for s in monitor.snapshots()}
    if not snaps:
        st.info("Loading the last three hours of trades from Deribit to warm up the detector...")
        return
    for inst in instruments:
        s = snaps.get(inst)
        if s is None:
            st.caption(f"{inst}: warming up")
            continue
        badge = LEVEL_COLOR[s.level]
        held = s.time - s.level_since
        st.markdown(
            f"### {inst} &nbsp; <span style='background:{badge};color:white;padding:2px 10px;"
            f"border-radius:4px;font-size:0.8em'>{s.level.name}</span> "
            f"<span style='font-size:0.7em;color:gray'>for {str(held).split('.')[0]} "
            f"· {s.time:%H:%M:%S} UTC</span>",
            unsafe_allow_html=True,
        )
        c = st.columns(7)
        c[0].metric("Price", f"{s.price:,.2f}", pct(s.move_1m, 2, sign=True), help="move over 1m")
        c[1].metric("Vol ratio", f"{s.ratio:.2f}x", help="vol over the last minute / normal")
        c[2].metric("Vol 1m", pct(s.fast_vol), help="EWMA, half-life 1 minute, annualized")
        c[3].metric("Normal vol", pct(s.slow_vol), help="EWMA, half-life 30 minutes")
        c[4].metric("1 sd move / 1m", f"{s.sigma_1m:,.2f}", help="at the current 1m vol")
        c[5].metric("Trade rate", f"{s.activity:.1f}x", help="trades per bar vs normal")
        c[6].metric("DVOL", pct(s.dvol), help="Deribit 30-day implied vol index")
        h = monitor.history(inst)
        h = h[h.index >= h.index[-1] - pd.Timedelta(hours=hours)]
        st.plotly_chart(chart(h, config, ""), width="stretch", key=f"chart-{inst}")

    alerts = monitor.alerts()
    st.subheader("Alerts")
    if not alerts:
        st.caption("No alert since start.")
        return
    table = pd.DataFrame([a.as_dict() for a in reversed(alerts)])
    table["time"] = pd.to_datetime(table["time"]).dt.strftime("%H:%M:%S")
    for col in ("fast_vol", "slow_vol", "dvol"):
        table[col] = table[col].map(pct)
    table["move_1m"] = table["move_1m"].map(lambda x: pct(x, 2, sign=True))
    table["ratio"] = table["ratio"].map("{:.2f}x".format)
    st.dataframe(table, hide_index=True, width="stretch")


render()
