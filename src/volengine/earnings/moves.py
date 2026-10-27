"""Historical price reaction to earnings announcements.

Timing conventions (closes are the only prices used, so results do not depend on
the noisy opening auction):

=========  =========================  =========================
timing     entry (last close before)  reaction (first close after)
=========  =========================  =========================
BMO        session before the date    session of the date
AMC        session of the date        next session
unknown    session before the date    next session (2-day window)
=========  =========================  =========================
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def event_windows(events: pd.DataFrame, sessions: pd.DatetimeIndex) -> pd.DataFrame:
    """Map announcement dates to (entry, reaction) sessions."""
    sessions = pd.DatetimeIndex(sessions).sort_values()
    rows = []
    for _, ev in events.iterrows():
        d, timing = pd.Timestamp(ev["date"]), ev["timing"]
        before = sessions[sessions <= d] if timing == "AMC" else sessions[sessions < d]
        after = sessions[sessions >= d] if timing == "BMO" else sessions[sessions > d]
        entry = before[-1] if len(before) else None
        reaction = after[0] if len(after) else None
        rows.append({"date": d, "timing": timing, "entry": entry, "reaction": reaction})
    return pd.DataFrame(rows)


def historical_moves(
    prices: pd.DataFrame,
    events: pd.DataFrame,
    holding: int | None = None,
    lead: int = 0,
    vol_window: int = 21,
) -> pd.DataFrame:
    """One row per past announcement.

    Columns
    -------
    reaction_move : simple return entry close -> reaction close
    holding_move  : simple return from the close ``lead`` sessions before entry to the
                    close ``holding`` sessions after that, if ``holding`` is given. This
                    replicates a straddle bought today (``lead`` sessions before the
                    entry) and held to an expiry ``holding`` sessions away.
    pre_vol       : daily RMS log return over the ``vol_window`` sessions before the
                    start of the holding window (information available at that time)
    z_reaction    : |log reaction move| / (pre_vol * sqrt(window sessions)), the move in
                    units of a normal-day standard deviation
    z_holding     : |log holding move| / pre_vol, scale-free so that it can be rescaled
                    by today's volatility
    """
    close = prices["close"].dropna()
    sessions = close.index
    logret = np.log(close).diff()
    pos = {ts: i for i, ts in enumerate(sessions)}
    win = event_windows(events, sessions)
    rows = []
    for _, w in win.iterrows():
        if pd.isna(w["entry"]) or pd.isna(w["reaction"]):
            continue
        i0, i1 = pos[w["entry"]], pos[w["reaction"]]
        start = i0 - lead
        if start < vol_window:
            continue
        pre = logret.iloc[start - vol_window + 1 : start + 1].to_numpy()
        pre_vol = float(np.sqrt(np.mean(pre**2)))
        c0 = close.iloc[i0]
        row = {
            "date": w["date"],
            "timing": w["timing"],
            "entry": w["entry"],
            "reaction": w["reaction"],
            "reaction_move": close.iloc[i1] / c0 - 1.0,
            "pre_vol": pre_vol,
        }
        row["z_reaction"] = abs(np.log(close.iloc[i1] / c0)) / (pre_vol * np.sqrt(i1 - i0))
        if holding is not None:
            if start + holding >= len(close) or start + holding < i1:
                continue
            cs, ch = close.iloc[start], close.iloc[start + holding]
            row["holding_move"] = ch / cs - 1.0
            row["z_holding"] = abs(np.log(ch / cs)) / pre_vol
        rows.append(row)
    return pd.DataFrame(rows)


def current_daily_vol(prices: pd.DataFrame, window: int = 21) -> float:
    """Same estimator as ``pre_vol``, on the latest window."""
    r = np.log(prices["close"]).diff().dropna().iloc[-window:]
    return float(np.sqrt(np.mean(r.to_numpy() ** 2)))
