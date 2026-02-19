"""
🏟️ Football Tracker — Main Page
================================
Upload one or two player tracking CSVs and get an instant overview.
Navigate to individual insight pages from the sidebar.
"""

import sys
from pathlib import Path

import streamlit as st
import pandas as pd

# Resolve repo root from absolute path (works regardless of CWD)
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.loader import load_file

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Football Tracker",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — dark tactical aesthetic
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Rajdhani', sans-serif;
        background-color: #0d1117;
        color: #e6edf3;
    }
    h1, h2, h3 { font-family: 'Rajdhani', sans-serif; letter-spacing: 0.04em; }
    .metric-card {
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 8px;
        padding: 1rem 1.4rem;
        margin-bottom: 0.6rem;
    }
    .metric-label { font-size: 0.75rem; color: #8b949e; text-transform: uppercase; letter-spacing: 0.1em; }
    .metric-value { font-size: 1.8rem; font-weight: 700; color: #00e5ff; font-family: 'IBM Plex Mono', monospace; }
    .tag-available { background: #1a472a; color: #3fb950; border-radius: 4px; padding: 2px 8px; font-size: 0.75rem; margin: 2px; display: inline-block; }
    .tag-unavailable { background: #2a1a1a; color: #f85149; border-radius: 4px; padding: 2px 8px; font-size: 0.75rem; margin: 2px; display: inline-block; }
    .file-header { border-left: 3px solid #00e5ff; padding-left: 0.8rem; margin-bottom: 0.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("# ⚽ Football Player Tracker")
st.markdown("*GPS + IMU data analysis for football performance analytics*")
st.divider()

# ---------------------------------------------------------------------------
# Sidebar — file upload
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 📂 Load Files")
    st.markdown("Upload one or two tracking CSV files.")

    uploaded_1 = st.file_uploader("File 1 (GPS + IMU)", type="csv", key="f1")
    imu_start_1 = st.text_input("File 1 start time (if no timestamp)", placeholder="e.g. 04.02.2026 17:30")

    st.divider()

    uploaded_2 = st.file_uploader("File 2 (IMU-only)", type="csv", key="f2")
    imu_start_2 = st.text_input(
        "File 2 start time (if no timestamp)",
        value="16.02.2026 19:30",
        placeholder="e.g. 16.02.2026 19:30",
    )

    st.divider()
    st.markdown("### 🗂 Navigation")
    st.page_link("Home.py", label="🏠 Overview", icon="🏠")
    st.page_link("pages/1_Exploratory_Analysis.py", label="🔍 Exploratory Analysis")
    st.page_link("pages/2_GPS_Speed_Validation.py", label="📡 GPS Speed Validation")
    st.page_link("pages/3_Position_Heatmap.py", label="🗺️ Position Heatmap")
    st.page_link("pages/4_Speed_Distribution.py", label="⚡ Speed Distribution")
    st.page_link("pages/5_IMU_Movements.py", label="🔄 Movement Detection")
    st.page_link("pages/6_Action_Events.py", label="🦵 Action Events")
    st.page_link("pages/7_Asymmetry_Fatigue.py", label="⚖️ Asymmetry & Fatigue")

# ---------------------------------------------------------------------------
# Load & cache
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading file…")
def cached_load(file_bytes: bytes, filename: str, imu_start: str):
    import tempfile, os
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        df, meta = load_file(tmp_path, imu_only_start=imu_start or None)
    finally:
        os.unlink(tmp_path)
    meta["file_name"] = filename
    return df, meta


def _display_file_summary(df: pd.DataFrame, meta: dict, label: str):
    st.markdown(f'<div class="file-header"><h3>{label} — {meta["file_name"]}</h3></div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    def card(col, label, value):
        col.markdown(
            f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div></div>',
            unsafe_allow_html=True,
        )

    card(col1, "Rows", f"{meta['n_rows']:,}")
    card(col2, "GPS", "✓" if meta["has_gps"] else "✗")

    dur = meta.get("duration_s")
    dur_str = f"{int(dur//60)}m {int(dur%60)}s" if dur else "—"
    card(col3, "Duration", dur_str)

    ts_label = "Reconstructed" if meta.get("timestamp_reconstructed") else ("Yes" if meta["has_timestamp"] else "No")
    card(col4, "Timestamp", ts_label)

    if meta.get("session_start_utc"):
        st.caption(f"📅 Session: **{meta['session_start_utc']}** → **{meta['session_end_utc']}** (UTC)")

    # Available / unavailable insights
    st.markdown("**Available insights:**")
    avail_html = " ".join(
        f'<span class="tag-available">{i.replace("_", " ")}</span>'
        for i in meta["available_insights"]
    )
    unavail_html = " ".join(
        f'<span class="tag-unavailable">{i.replace("_", " ")}</span>'
        for i in meta["unavailable_insights"]
    )
    st.markdown(avail_html + ("&nbsp;&nbsp;" + unavail_html if unavail_html else ""), unsafe_allow_html=True)

    with st.expander("Preview data (first 5 rows)"):
        display_cols = [c for c in df.columns if not c.endswith("_raw")]
        st.dataframe(df[display_cols].head(), use_container_width=True)


# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------
files_loaded = False

if uploaded_1 or uploaded_2:
    dfs = {}
    metas = {}

    if uploaded_1:
        df1, meta1 = cached_load(uploaded_1.read(), uploaded_1.name, imu_start_1)
        dfs["file1"] = df1
        metas["file1"] = meta1
        st.session_state["df1"] = df1
        st.session_state["meta1"] = meta1

    if uploaded_2:
        df2, meta2 = cached_load(uploaded_2.read(), uploaded_2.name, imu_start_2)
        dfs["file2"] = df2
        metas["file2"] = meta2
        st.session_state["df2"] = df2
        st.session_state["meta2"] = meta2

    files_loaded = True

    # Display summaries
    if "file1" in dfs and "file2" in dfs:
        col_left, col_right = st.columns(2)
        with col_left:
            _display_file_summary(dfs["file1"], metas["file1"], "File 1")
        with col_right:
            _display_file_summary(dfs["file2"], metas["file2"], "File 2")
    elif "file1" in dfs:
        _display_file_summary(dfs["file1"], metas["file1"], "File 1")
    elif "file2" in dfs:
        _display_file_summary(dfs["file2"], metas["file2"], "File 2")

    st.divider()
    st.success("✅ Files loaded. Use the sidebar to navigate to any insight page.")

else:
    # Demo mode with bundled data files
    st.info("👈 Upload files in the sidebar, or use the demo data below.")

    DATA_DIR = ROOT / "data"
    demo_f1 = DATA_DIR / "new_player_data_2026_02_06_174048.csv"
    demo_f2 = DATA_DIR / "player_activity_imu_2026_02_16.csv"

    if st.button("▶ Load bundled demo files", type="primary"):
        if demo_f1.exists():
            df1, meta1 = load_file(str(demo_f1))
            st.session_state["df1"] = df1
            st.session_state["meta1"] = meta1
        if demo_f2.exists():
            df2, meta2 = load_file(str(demo_f2), imu_only_start="16.02.2026 19:30")
            st.session_state["df2"] = df2
            st.session_state["meta2"] = meta2
        st.rerun()

    if "df1" in st.session_state:
        _display_file_summary(st.session_state["df1"], st.session_state["meta1"], "File 1 (demo)")
    if "df2" in st.session_state:
        _display_file_summary(st.session_state["df2"], st.session_state["meta2"], "File 2 (demo)")

    if "df1" in st.session_state or "df2" in st.session_state:
        st.success("✅ Demo data loaded. Navigate via the sidebar.")
