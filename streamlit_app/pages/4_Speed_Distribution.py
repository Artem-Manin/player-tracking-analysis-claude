"""Page 4 — Speed Distribution"""
import importlib.util
from pathlib import Path

import plotly.express as px
import streamlit as st

_shared = importlib.util.module_from_spec(
    s := importlib.util.spec_from_file_location("shared", Path(__file__).resolve().parent / "shared.py")
)
s.loader.exec_module(_shared)
ROOT = _shared.ROOT
require_session, get_df, get_meta = _shared.require_session, _shared.get_df, _shared.get_meta
unavailable, DARK_CSS = _shared.unavailable, _shared.DARK_CSS

import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.gps_analytics import speed_zone_distribution
from src.plots import plot_speed_zone_bar

st.set_page_config(page_title="Speed Distribution", page_icon="⚡", layout="wide")
st.markdown(DARK_CSS, unsafe_allow_html=True)
st.markdown("# ⚡ Speed Distribution")
st.markdown("Time spent in each UEFA/EPTS speed zone and histogram of raw speed values.")
st.divider()


def show_speed(df, meta, label):
    st.markdown(f"## {label}")
    if "speed_distribution" not in meta.get("available_insights", []):
        unavailable("speed_distribution", meta)
        return

    zone_df = speed_zone_distribution(df)
    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("### Zone Breakdown")
        st.plotly_chart(plot_speed_zone_bar(zone_df), use_container_width=True)
    with col_right:
        st.markdown("### Speed Histogram")
        speed_kmh = df["speed_ms"].dropna() * 3.6
        fig = px.histogram(
            speed_kmh, nbins=80, template="plotly_dark",
            color_discrete_sequence=["#00e5ff"],
            labels={"value": "Speed (km/h)"},
            title="Raw speed distribution",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Zone table")
    st.dataframe(zone_df, use_container_width=True, hide_index=True)


if "df1" in st.session_state:
    show_speed(get_df("df1"), get_meta("meta1"), "File 1")
if "df2" in st.session_state:
    show_speed(get_df("df2"), get_meta("meta2"), "File 2")
if "df1" not in st.session_state and "df2" not in st.session_state:
    require_session("df1")
