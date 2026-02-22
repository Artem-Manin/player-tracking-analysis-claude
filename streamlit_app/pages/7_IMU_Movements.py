"""Page 5 — IMU Movement Detection"""
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

from src.imu_analytics import detect_special_movements, detect_outliers
from src.plots import plot_movement_events, plot_acc_trajectory

st.set_page_config(page_title="Movement Detection", page_icon="🔄", layout="wide")
st.markdown(DARK_CSS, unsafe_allow_html=True)
st.markdown("# 🔄 Special Movement Detection")
st.markdown(
    "Detects **twists**, **leans**, and **turns** from IMU gyroscope and accelerometer data. "
    "Results are heuristic — designed to flag candidates for **manual video validation**."
)
st.divider()


def show_movements(df, meta, label):
    st.markdown(f"## {label} — `{meta.get('file_name', '')}`")

    df = detect_outliers(df)
    df = detect_special_movements(df)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Twists detected", int(df["is_twist"].sum()))
    c2.metric("Leans detected", int(df["is_lean"].sum()) if "is_lean" in df.columns else "N/A")
    c3.metric("Turns detected", int(df["is_turn"].sum()) if "is_turn" in df.columns else "N/A")
    c4.metric("Acc outliers", int(df["outlier_acc"].sum()) if "outlier_acc" in df.columns else "N/A")

    st.caption(
        "Thresholds: twist = rotation > 100 °/s | lean = |roll| > 15° | turn = |yaw| > 80 °/s. "
        "Validate against video before drawing conclusions."
    )
    st.divider()

    st.markdown("### IMU Time Series with Events")
    st.plotly_chart(plot_movement_events(df), use_container_width=True)

    st.markdown("### Acceleration Magnitude")
    st.plotly_chart(plot_acc_trajectory(df), use_container_width=True)

    with st.expander("Show flagged rows"):
        flag_cols = [c for c in ["ts_cet", "rot_magnitude_dps", "roll_deg", "rot_z_dps",
                                  "is_twist", "is_lean", "is_turn", "outlier_acc"] if c in df.columns]
        event_mask = (
            df.get("is_twist", False) | df.get("is_lean", False) | df.get("is_turn", False)
        ) if any(c in df.columns for c in ["is_twist", "is_lean", "is_turn"]) else None
        flagged = df[event_mask] if event_mask is not None else df.head(0)
        st.dataframe(flagged[flag_cols].head(100), use_container_width=True)

    st.divider()


if "df1" in st.session_state:
    show_movements(get_df("df1"), get_meta("meta1"), "File 1")
if "df2" in st.session_state:
    show_movements(get_df("df2"), get_meta("meta2"), "File 2")
if "df1" not in st.session_state and "df2" not in st.session_state:
    require_session("df1")
