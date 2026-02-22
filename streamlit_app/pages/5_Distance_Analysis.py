"""Page 5 — Distance Analysis (foundation ready, deeper work TBD)"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# ── shared loader ────────────────────────────────────────────────────────────
_shared = importlib.util.module_from_spec(
    s := importlib.util.spec_from_file_location(
        "shared", Path(__file__).resolve().parent / "shared.py"
    )
)
s.loader.exec_module(_shared)
ROOT = _shared.ROOT
require_session, get_df, get_meta = _shared.require_session, _shared.get_df, _shared.get_meta
unavailable, DARK_CSS = _shared.unavailable, _shared.DARK_CSS

import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.gk_analysis import (
    add_metric_coords, add_grid_cells, build_windows,
    rule_based_gk, cluster_gk, WINDOW_SECONDS,
)
from src.gps_analytics import haversine_distance

st.set_page_config(page_title="Distance Analysis", page_icon="📏", layout="wide")
st.markdown(DARK_CSS, unsafe_allow_html=True)
st.markdown("# 📏 Distance Analysis")
st.markdown(
    "Foundation distance metrics by role (GK vs Outfield). "
    "This page will be extended — deeper analysis coming."
)
st.info(
    "🚧 **Work in progress.** The core distance data is computed and ready below. "
    "Deeper per-drill segmentation, sprint distance, and comparisons will be added here.",
    icon="🚧",
)

if not require_session("df1"):
    st.stop()

meta = get_meta("meta1")
if not meta.get("has_gps"):
    unavailable("distance_total", meta)
    st.stop()

@st.cache_data(show_spinner="Computing distances…")
def compute(_df_json):
    df = pd.read_json(_df_json, orient="split")
    if "ts_utc" in df.columns:
        df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True, unit="ms", errors="coerce")
    df, x_max, y_max = add_metric_coords(df)
    df = add_grid_cells(df, x_max, y_max)
    df = haversine_distance(df)
    windows = build_windows(df)
    windows = rule_based_gk(windows, x_max, y_max)
    windows, _ = cluster_gk(windows, x_max)
    df["_window"] = (
        (df["ts_utc"] - df["ts_utc"].min()).dt.total_seconds() // WINDOW_SECONDS
    ).astype(int) if "ts_utc" in df.columns else 0
    df = df.merge(
        windows[["window_idx", "rule_gk"]],
        left_on="_window", right_on="window_idx", how="left",
    )
    return df, windows

raw_df = get_df("df1")
df, windows = compute(raw_df.to_json(orient="split", date_format="epoch", date_unit="ms"))

# ── total distance ────────────────────────────────────────────────────────────
total_m = df["dist_m"].sum()
gk_m    = df[df["rule_gk"] == True]["dist_m"].sum()
out_m   = df[df["rule_gk"] == False]["dist_m"].sum()

st.divider()
c1, c2, c3 = st.columns(3)
c1.metric("Total distance", f"{total_m:.0f} m", f"{total_m/1000:.2f} km")
c2.metric("Distance in GK windows", f"{gk_m:.0f} m", f"{gk_m/total_m*100:.1f}% of total")
c3.metric("Distance in Outfield windows", f"{out_m:.0f} m", f"{out_m/total_m*100:.1f}% of total")

st.divider()

# ── distance per window ───────────────────────────────────────────────────────
window_dist = df.groupby("_window")["dist_m"].sum().reset_index()
window_dist.columns = ["window_idx", "dist_m"]
window_dist = window_dist.merge(windows[["window_idx", "rule_gk"]], on="window_idx", how="left")
window_dist["role"] = window_dist["rule_gk"].map({True: "GK", False: "Outfield"})
window_dist["window_min"] = window_dist["window_idx"] * WINDOW_SECONDS / 60

st.markdown("## Distance per 3-minute window")
fig = px.bar(
    window_dist, x="window_min", y="dist_m",
    color="role",
    color_discrete_map={"GK": "#00e5ff", "Outfield": "#ff7f0e"},
    labels={"window_min": "Session time (min)", "dist_m": "Distance (m)", "role": "Role"},
    title="Distance covered per 3-minute window",
    template="plotly_dark",
)
st.plotly_chart(fig, use_container_width=True)

# ── cumulative ────────────────────────────────────────────────────────────────
st.markdown("## Cumulative distance over session")
df_sorted = df.sort_values("ts_utc") if "ts_utc" in df.columns else df
df_sorted["cum_dist_m"] = df_sorted["dist_m"].cumsum()
ts_col = "ts_cet" if "ts_cet" in df_sorted.columns else "ts_utc"

fig2 = go.Figure(go.Scatter(
    x=df_sorted[ts_col], y=df_sorted["cum_dist_m"],
    line=dict(color="#00e5ff", width=1.5),
    fill="tozeroy", fillcolor="rgba(0,229,255,0.07)",
    name="Cumulative distance",
))
fig2.update_layout(
    template="plotly_dark",
    title="Cumulative distance (m)",
    xaxis_title="Time (CET)", yaxis_title="Distance (m)",
)
st.plotly_chart(fig2, use_container_width=True)

st.caption(
    "**Note**: GPS bounding box for this session is ~18×35 m (goalkeeper/set-piece drill area). "
    "Total distance of 4.5 km over 84 min is realistic for intensive GK movement within a small zone."
)
