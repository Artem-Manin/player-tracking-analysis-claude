"""
shared.py — helpers used by all Streamlit pages.
Provides robust repo-root resolution regardless of CWD when streamlit is launched.
"""
import sys
import os
from pathlib import Path
import streamlit as st


def _find_repo_root() -> Path:
    """
    Walk up from this file's location until we find the repo root,
    identified by the presence of requirements.txt or src/.
    Works regardless of CWD or how streamlit was invoked.
    """
    candidate = Path(__file__).resolve()
    for _ in range(6):  # max 6 levels up
        candidate = candidate.parent
        if (candidate / "requirements.txt").exists() or (candidate / "src").exists():
            return candidate
    # Fallback: 3 levels up from this file (pages/ → streamlit_app/ → repo/)
    return Path(__file__).resolve().parent.parent.parent


ROOT = _find_repo_root()

# Ensure src/ is importable
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def require_session(key: str = "df1") -> bool:
    """Return True if data is loaded; otherwise show a warning and return False."""
    if key not in st.session_state:
        st.warning(
            "⚠️ No data loaded yet. Go to the **Home** page and upload a file or load the demo data.",
            icon="⚠️",
        )
        st.page_link("Home.py", label="← Back to Home")
        return False
    return True


def get_df(key: str = "df1"):
    return st.session_state.get(key)


def get_meta(key: str = "meta1"):
    return st.session_state.get(key, {})


def unavailable(insight: str, meta: dict):
    """Display a friendly 'not available' message for a given insight."""
    st.warning(
        f"**{insight.replace('_', ' ').title()}** is not available for this file.\n\n"
        + _reason(insight, meta),
        icon="ℹ️",
    )


def _reason(insight: str, meta: dict) -> str:
    gps_insights = {"gps_speed_validation", "position_heatmap", "goalkeeper_clustering",
                    "speed_distribution", "distance_total", "speed_trajectories", "fatigue_analysis"}
    ts_insights = {"fatigue_analysis", "imu_movement_detection", "shot_pass_header_detection"}
    reasons = []
    if insight in gps_insights and not meta.get("has_gps"):
        reasons.append("This file contains no GPS data (latitude/longitude are all zero or absent).")
    if insight in ts_insights and not meta.get("has_timestamp"):
        reasons.append("This file has no valid timestamp column.")
    return " ".join(reasons) if reasons else "Required data columns are missing."


DARK_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');
html, body, [class*="css"] {
    font-family: 'Rajdhani', sans-serif;
    background-color: #0d1117;
    color: #e6edf3;
}
h1, h2, h3 { font-family: 'Rajdhani', sans-serif; letter-spacing: 0.04em; }
</style>
"""
