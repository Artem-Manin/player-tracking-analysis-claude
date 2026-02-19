"""Page 7 — Asymmetry & Fatigue Analysis"""
import importlib.util
from pathlib import Path

import plotly.graph_objects as go
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

from src.imu_analytics import asymmetry_analysis, fatigue_analysis
from src.plots import plot_fatigue

st.set_page_config(page_title="Asymmetry & Fatigue", page_icon="⚖️", layout="wide")
st.markdown(DARK_CSS, unsafe_allow_html=True)
st.markdown("# ⚖️ Asymmetry & Fatigue Analysis")
st.divider()

# ── Asymmetry ──────────────────────────────────────────────────────────────
st.markdown("## Left–Right Asymmetry")
st.markdown(
    "Compares the magnitude of lateral acceleration between left and right sides. "
    "A healthy player should show near-symmetry; large asymmetry can indicate injury risk or strong dominance."
)


def show_asymmetry(df, meta, label):
    st.markdown(f"### {label}")
    result = asymmetry_analysis(df)
    if not result:
        st.warning("IMU data not available for asymmetry analysis.")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Left acc mean (g)", result.get("acc_x_mean_left_g", "—"))
    c2.metric("Right acc mean (g)", result.get("acc_x_mean_right_g", "—"))
    c3.metric(
        "Asymmetry index",
        f"{result.get('asymmetry_index_pct', 0):.1f}%",
        help="(|Left| − |Right|) / (|Left| + |Right|) × 100. Positive = left dominant.",
    )
    dominant = result.get("dominant_side", "—")
    colour = {"left": "🔵", "right": "🔴", "symmetric": "🟢"}.get(dominant, "⚪")
    st.markdown(f"**Dominant side (heuristic):** {colour} {dominant.upper()}")

    if "rot_x_mean_left_dps" in result:
        st.caption(
            f"Rotation asymmetry — Left: {result['rot_x_mean_left_dps']:.1f} °/s  |  "
            f"Right: {result['rot_x_mean_right_dps']:.1f} °/s"
        )

    fig = go.Figure(go.Bar(
        x=["Left acc (g)", "Right acc (g)"],
        y=[result.get("acc_x_mean_left_g", 0), result.get("acc_x_mean_right_g", 0)],
        marker_color=["#1f77b4", "#d62728"],
    ))
    fig.update_layout(template="plotly_dark", title=f"Lateral Acceleration — {label}", height=300)
    st.plotly_chart(fig, use_container_width=True)


if "df1" in st.session_state:
    show_asymmetry(get_df("df1"), get_meta("meta1"), "File 1")
if "df2" in st.session_state:
    show_asymmetry(get_df("df2"), get_meta("meta2"), "File 2")

st.divider()

# ── Fatigue ────────────────────────────────────────────────────────────────
st.markdown("## Fatigue Analysis (Speed Drop Over Time)")
st.markdown(
    "Tracks peak and mean speed across rolling time windows. A downward trend in peak speed "
    "across the session is a commonly used proxy for physical fatigue."
)

window = st.slider("Window size (minutes)", min_value=5, max_value=20, value=10)


def show_fatigue(df, meta, label):
    st.markdown(f"### {label}")
    if "fatigue_analysis" not in meta.get("available_insights", []):
        unavailable("fatigue_analysis", meta)
        return
    fat_df = fatigue_analysis(df, window_minutes=window)
    if fat_df.empty:
        st.warning("Not enough data for fatigue analysis.")
        return
    st.plotly_chart(plot_fatigue(fat_df), use_container_width=True)
    st.dataframe(fat_df.drop(columns=["window_end"], errors="ignore"), use_container_width=True, hide_index=True)


if "df1" in st.session_state:
    show_fatigue(get_df("df1"), get_meta("meta1"), "File 1")
if "df2" in st.session_state:
    show_fatigue(get_df("df2"), get_meta("meta2"), "File 2")
if "df1" not in st.session_state and "df2" not in st.session_state:
    require_session("df1")
