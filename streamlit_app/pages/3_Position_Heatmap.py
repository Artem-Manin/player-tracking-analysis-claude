"""Page 3 — Position Heatmap & Goalkeeper Clustering"""
import importlib.util
from pathlib import Path

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

from src.gps_analytics import goalkeeper_clustering, haversine_distance, total_distance
from src.plots import plot_position_heatmap, plot_speed_trajectory

st.set_page_config(page_title="Position Heatmap", page_icon="🗺️", layout="wide")
st.markdown(DARK_CSS, unsafe_allow_html=True)
st.markdown("# 🗺️ Position Heatmap & Goalkeeper Clustering")
st.markdown(
    "GPS position density map with K-Means clustering. The cluster most likely to correspond "
    "to a goalkeeping role is identified by low mean speed + proximity to a goal-line."
)
st.divider()

if "df2" in st.session_state:
    meta2 = get_meta("meta2")
    if "position_heatmap" not in meta2.get("available_insights", []):
        st.info("ℹ️ **File 2** has no GPS data — heatmap is only available for File 1.", icon="ℹ️")

if not require_session("df1"):
    st.stop()

meta = get_meta("meta1")
if "position_heatmap" not in meta.get("available_insights", []):
    unavailable("position_heatmap", meta)
    st.stop()

df = get_df("df1")
df = haversine_distance(df)

n_clusters = st.slider("Number of position clusters", min_value=2, max_value=6, value=3)

with st.spinner("Clustering positions…"):
    df_cl, cluster_summary = goalkeeper_clustering(df, n_clusters=n_clusters)

st.markdown("### Cluster Summary")
display_summary = cluster_summary.drop(columns=["gk_score", "dist_to_goalline_deg"], errors="ignore").copy()
display_summary["likely_gk"] = display_summary.get("likely_gk", False).apply(lambda x: "★ Likely GK" if x else "")
st.dataframe(display_summary, use_container_width=True, hide_index=True)

total_dist = total_distance(df)
st.metric("Total distance (session)", f"{total_dist:.0f} m  ({total_dist/1000:.2f} km)")
st.caption(
    "⚠️ The GPS bounding box for this session is approximately **18 m × 35 m** — "
    "consistent with goalkeeper or set-piece drill data, not a full pitch. "
    "Clustering reflects micro-zones within that area."
)
st.divider()

col_left, col_right = st.columns(2)
with col_left:
    st.markdown("### Density Heatmap with Cluster Centres")
    st.plotly_chart(plot_position_heatmap(df_cl, cluster_summary), use_container_width=True)
with col_right:
    st.markdown("### Speed Trajectory")
    st.plotly_chart(plot_speed_trajectory(df_cl), use_container_width=True)
