"""Page 2 — GPS Speed Validation"""
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

from src.gps_analytics import gps_speed_validation
from src.plots import plot_speed_timeseries, plot_speed_scatter, plot_speed_error_histogram

st.set_page_config(page_title="GPS Speed Validation", page_icon="📡", layout="wide")
st.markdown(DARK_CSS, unsafe_allow_html=True)
st.markdown("# 📡 GPS Speed Validation")
st.markdown(
    "Recompute speed from GPS coordinates (Haversine formula) and compare against "
    "the device-reported speed value. This is the foundation validation step."
)
st.divider()

if not require_session("df1"):
    st.stop()

meta = get_meta("meta1")
if "gps_speed_validation" not in meta.get("available_insights", []):
    unavailable("gps_speed_validation", meta)
    st.stop()

df = get_df("df1")
with st.spinner("Computing Haversine speeds…"):
    metrics, df_val = gps_speed_validation(df)

st.markdown("### Validation Metrics")
c1, c2, c3, c4 = st.columns(4)
c1.metric("MAE", f"{metrics['mae_ms']:.4f} m/s", help="Mean Absolute Error between device and Haversine speed")
c2.metric("RMSE", f"{metrics['rmse_ms']:.4f} m/s")
c3.metric("Correlation", f"{metrics['correlation']:.4f}")
c4.metric("Flagged rows", f"{metrics['n_flagged']} / {metrics['n_valid_rows']}")
st.caption(
    f"Rows flagged when |error| > {metrics['flag_threshold_ms']:.4f} m/s (2×RMSE). "
    "Speed unit is **m/s**. Max device speed ≈ 5.15 m/s ≈ 18.5 km/h, consistent with training."
)
st.divider()

st.markdown("### Time Series")
st.plotly_chart(plot_speed_timeseries(df_val), use_container_width=True)

col_left, col_right = st.columns(2)
with col_left:
    st.markdown("### Device vs Haversine Scatter")
    st.plotly_chart(plot_speed_scatter(df_val), use_container_width=True)
with col_right:
    st.markdown("### Error Distribution")
    st.plotly_chart(plot_speed_error_histogram(df_val), use_container_width=True)

st.markdown("### Flagged Rows")
flagged = df_val[df_val["speed_flagged"]] if "speed_flagged" in df_val.columns else df_val.head(0)
if flagged.empty:
    st.success("No rows flagged — device speed and Haversine speed agree well.")
else:
    show_cols = [c for c in ["ts_cet", "speed_ms", "speed_haversine_ms", "speed_error_ms", "speed_flagged"] if c in flagged.columns]
    st.dataframe(flagged[show_cols].head(50), use_container_width=True)
