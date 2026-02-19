"""Page 6 — Action Events: Shots, Passes, Headers, Footedness"""
import importlib.util
from pathlib import Path

import streamlit as st

_shared = importlib.util.module_from_spec(
    s := importlib.util.spec_from_file_location("shared", Path(__file__).resolve().parent / "shared.py")
)
s.loader.exec_module(_shared)
ROOT = _shared.ROOT
require_session, get_df, get_meta = _shared.require_session, _shared.get_df, _shared.get_meta
DARK_CSS = _shared.DARK_CSS

import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.imu_analytics import detect_action_events, estimate_footedness
from src.plots import plot_action_events

st.set_page_config(page_title="Action Events", page_icon="🦵", layout="wide")
st.markdown(DARK_CSS, unsafe_allow_html=True)
st.markdown("# 🦵 Action Events — Shots, Passes & Headers")
st.markdown(
    "Detects likely ball-contact events from IMU acceleration and rotation signatures. "
    "These are **signal-based heuristics** — manual video validation is essential before "
    "drawing any conclusions."
)
st.info(
    "**Methodology**: Shots = high acc peak (>2.5 g) + high rotation (>120 °/s). "
    "Passes = moderate acc peak (1.5–2.5 g). Headers = large vertical Z-acc (>2 g) + pitch change (>20°).",
    icon="🔬",
)
st.divider()


def show_actions(df, meta, label):
    st.markdown(f"## {label} — `{meta.get('file_name', '')}`")

    df = detect_action_events(df)
    foot = estimate_footedness(df)

    col1, col2, col3 = st.columns(3)
    col1.metric("Probable shots", int(df["event_shot"].sum()))
    col2.metric("Probable passes", int(df["event_pass"].sum()))
    col3.metric("Probable headers", int(df["event_header"].sum()))

    st.markdown("### Action Event Timeline")
    st.plotly_chart(plot_action_events(df), use_container_width=True)

    st.markdown("### Footedness Estimate")
    if foot and not (len(foot) == 1 and "note" in foot):
        fc1, fc2, fc3 = st.columns(3)
        fc1.metric("Left foot actions", foot.get("estimated_left_foot_actions", "—"))
        fc2.metric("Right foot actions", foot.get("estimated_right_foot_actions", "—"))
        fc3.metric("Likely dominant foot", foot.get("likely_dominant_foot", "—").title())
        if foot.get("left_pct") is not None:
            left_pct = float(foot["left_pct"])
            st.progress(left_pct / 100, text=f"Left: {left_pct:.1f}%  |  Right: {foot['right_pct']:.1f}%")
    st.caption(foot.get("note", ""))

    with st.expander("Show detected events table"):
        event_cols = [c for c in ["ts_cet", "acc_magnitude_g", "rot_magnitude_dps",
                                   "event_shot", "event_pass", "event_header"] if c in df.columns]
        events_only = df[df["event_shot"] | df["event_pass"] | df["event_header"]]
        if events_only.empty:
            st.info("No events detected.")
        else:
            st.dataframe(events_only[event_cols], use_container_width=True)

    st.divider()


if "df1" in st.session_state:
    show_actions(get_df("df1"), get_meta("meta1"), "File 1")
if "df2" in st.session_state:
    show_actions(get_df("df2"), get_meta("meta2"), "File 2")
if "df1" not in st.session_state and "df2" not in st.session_state:
    require_session("df1")
