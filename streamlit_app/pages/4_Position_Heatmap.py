"""Page 4 — Position Heatmap by Role"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
    rule_based_gk, cluster_gk,
    N_GRID_ROWS, N_GRID_COLS, WINDOW_SECONDS,
)

st.set_page_config(page_title="Position Heatmap", page_icon="🗺️", layout="wide")
st.markdown(DARK_CSS, unsafe_allow_html=True)
st.markdown("# 🗺️ Position Heatmap")
st.markdown(
    "**5 × 3 grid heatmap** — 5 zones along the pitch length, 3 across the width. "
    "Our goal is at the **bottom**. Heatmaps shown separately for GK and Outfield windows, "
    "plus a combined overall view."
)

if not require_session("df1"):
    st.stop()

meta = get_meta("meta1")
if not meta.get("has_gps"):
    unavailable("position_heatmap", meta)
    st.stop()

# ── compute (cached) ──────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Building heatmap data…")
def compute(_df_json):
    df = pd.read_json(_df_json, orient="split")
    if "ts_utc" in df.columns:
        df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True, unit="ms", errors="coerce")
    df, x_max, y_max = add_metric_coords(df)
    df = add_grid_cells(df, x_max, y_max)
    windows = build_windows(df)
    windows = rule_based_gk(windows, x_max, y_max)
    windows, _ = cluster_gk(windows, x_max)
    # Merge window labels back to raw records
    df["_window"] = (
        (df["ts_utc"] - df["ts_utc"].min()).dt.total_seconds() // WINDOW_SECONDS
    ).astype(int) if "ts_utc" in df.columns else 0
    df = df.merge(
        windows[["window_idx", "rule_gk", "cluster_gk"]],
        left_on="_window", right_on="window_idx", how="left",
    )
    return df, x_max, y_max

raw_df = get_df("df1")
df, x_max, y_max = compute(raw_df.to_json(orient="split", date_format="epoch", date_unit="ms"))

# ── sidebar controls ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🗺️ Heatmap options")
    method = st.radio("Classification method", ["Rule-based", "Clustering"], index=0)
    color_scale = st.selectbox("Colour scale", ["YlOrRd", "Blues", "Viridis", "Hot"], index=0)
    show_raw = st.checkbox("Overlay raw GPS points", value=False)
    st.divider()
    st.markdown("## 🏟️ Grid guide")
    st.markdown(
        "**5 rows** along pitch length\n"
        "- Row 0 = Our goal *(bottom)*\n"
        "- Row 4 = Opponent goal *(top)*\n\n"
        "**3 cols** across pitch width\n"
        "- Col 0 = South sideline\n"
        "- Col 1 = Centre\n"
        "- Col 2 = North sideline"
    )

label_col = "rule_gk" if method == "Rule-based" else "cluster_gk"

# ── pitch drawing helper ──────────────────────────────────────────────────────
ROW_LABELS = ["Our Goal", "Zone 2", "Zone 3", "Zone 4", "Opponent"]  # row 0 = our goal (bottom)
COL_LABELS = ["South", "Centre", "North"]

def build_grid_z(subset: pd.DataFrame) -> np.ndarray:
    """
    Build 5×3 count matrix.
    z[row, col] where row=0 is our goal (bottom of plot).
    Plotly heatmap: z[0] = bottom row when origin='lower' (we'll handle manually).
    """
    z = np.zeros((N_GRID_ROWS, N_GRID_COLS), dtype=int)
    grp = subset.groupby(["grid_row", "grid_col"]).size()
    for (r, c), cnt in grp.items():
        if 0 <= r < N_GRID_ROWS and 0 <= c < N_GRID_COLS:
            z[r, c] = cnt
    return z


def pitch_heatmap(z: np.ndarray, title: str, colorscale: str, show_pts: bool,
                  pts_df: pd.DataFrame | None = None) -> go.Figure:
    """
    Draw a vertical-orientation pitch heatmap.
    Row 0 (our goal) at the BOTTOM; row 4 (opponent) at the TOP.
    Plotly heatmap y-axis: y[0]=bottom → reverse the row order so row0=our goal=bottom.
    """
    # Flip z so that z_plot[0] = row 4 (opponent, top) and z_plot[4] = row 0 (our goal, bottom)
    z_plot = z[::-1, :]       # now z_plot row index 0 = plotly top = opponent

    # Normalise for display
    z_norm = z_plot / z_plot.max() if z_plot.max() > 0 else z_plot.astype(float)

    y_labels = ROW_LABELS[::-1]  # top→bottom in display: Opponent … Our Goal

    fig = go.Figure()

    # Heatmap
    fig.add_trace(go.Heatmap(
        z=z_plot,
        x=COL_LABELS,
        y=y_labels,
        colorscale=colorscale,
        showscale=True,
        colorbar=dict(title="Records"),
        hovertemplate="<b>%{y}</b> | %{x}<br>Records: %{z}<extra></extra>",
    ))

    # Cell annotations
    for ri in range(N_GRID_ROWS):
        for ci in range(N_GRID_COLS):
            val = z_plot[ri, ci]
            total = z_plot.sum()
            pct = val / total * 100 if total > 0 else 0
            fig.add_annotation(
                x=COL_LABELS[ci],
                y=y_labels[ri],
                text=f"<b>{val}</b><br><span style='font-size:10px'>{pct:.1f}%</span>",
                showarrow=False,
                font=dict(color="white" if z_norm[ri, ci] > 0.4 else "#333", size=11),
            )

    # Our goal marker at bottom
    fig.add_annotation(
        x=1.05, y=ROW_LABELS[0],   # Our Goal = bottom row label
        text="⚽ Our Goal", showarrow=False, xref="paper",
        font=dict(color="#00e5ff", size=11),
    )
    fig.add_annotation(
        x=1.05, y=ROW_LABELS[-1],  # Opponent = top row label
        text="🥅 Opponent", showarrow=False, xref="paper",
        font=dict(color="#ff7f0e", size=11),
    )

    if show_pts and pts_df is not None and not pts_df.empty:
        # Map x_m/y_m to grid column/row label
        pts_df = pts_df.copy()
        pts_df["_col_label"] = pd.Categorical(
            pts_df["grid_col"].map({0: "South", 1: "Centre", 2: "North"}),
            categories=COL_LABELS, ordered=True
        )
        pts_df["_row_label"] = pd.Categorical(
            pts_df["grid_row"].map(dict(enumerate(ROW_LABELS))),
            categories=y_labels, ordered=True
        )
        fig.add_trace(go.Scatter(
            x=pts_df["_col_label"].astype(str),
            y=pts_df["_row_label"].astype(str),
            mode="markers",
            marker=dict(color="white", size=2, opacity=0.15),
            name="GPS points",
            hoverinfo="skip",
        ))

    fig.update_layout(
        title=title,
        template="plotly_dark",
        height=480,
        yaxis=dict(title="← Opponent goal          Our goal →"),
        xaxis=dict(title="Across pitch width"),
        margin=dict(r=90),
    )
    return fig


# ── draw heatmaps ─────────────────────────────────────────────────────────────
st.divider()
st.markdown(f"## Heatmaps — {method} classification")

df_gk  = df[df[label_col] == True]
df_out = df[df[label_col] == False]
df_all = df

n_gk  = len(df_gk)
n_out = len(df_out)
n_all = len(df_all)

c1, c2, c3 = st.columns(3)
c1.metric("GK records", f"{n_gk:,}", f"{n_gk/n_all*100:.1f}%")
c2.metric("Outfield records", f"{n_out:,}", f"{n_out/n_all*100:.1f}%")
c3.metric("Total records", f"{n_all:,}")

tab1, tab2, tab3 = st.tabs(["🥅 GK windows", "🏃 Outfield windows", "📊 Combined"])

with tab1:
    if df_gk.empty:
        st.info("No GK windows detected.")
    else:
        z = build_grid_z(df_gk)
        st.plotly_chart(
            pitch_heatmap(z, "GK Position Density", color_scale, show_raw, df_gk),
            use_container_width=True,
        )
        st.caption(
            f"Showing {n_gk:,} records from **{df_gk['_window'].nunique()} GK windows**. "
            "As expected for a GK, density concentrates in the goal-side zones."
        )

with tab2:
    if df_out.empty:
        st.info("No Outfield windows detected.")
    else:
        z = build_grid_z(df_out)
        st.plotly_chart(
            pitch_heatmap(z, "Outfield Position Density", color_scale, show_raw, df_out),
            use_container_width=True,
        )
        st.caption(f"Showing {n_out:,} records from **{df_out['_window'].nunique()} Outfield windows**.")

with tab3:
    z = build_grid_z(df_all)
    st.plotly_chart(
        pitch_heatmap(z, "Overall Position Density (All Windows)", color_scale, show_raw, df_all),
        use_container_width=True,
    )

# ── side-by-side comparison ───────────────────────────────────────────────────
st.divider()
st.markdown("## Side-by-side: GK vs Outfield")

col_gk, col_out = st.columns(2)
with col_gk:
    if not df_gk.empty:
        st.plotly_chart(
            pitch_heatmap(build_grid_z(df_gk), "GK", color_scale, False),
            use_container_width=True,
        )
with col_out:
    if not df_out.empty:
        st.plotly_chart(
            pitch_heatmap(build_grid_z(df_out), "Outfield", color_scale, False),
            use_container_width=True,
        )

# ── grid counts table ─────────────────────────────────────────────────────────
with st.expander("📋 Raw grid counts"):
    for label, subset in [("GK", df_gk), ("Outfield", df_out), ("All", df_all)]:
        z = build_grid_z(subset)
        tdf = pd.DataFrame(
            z[::-1],  # top = opponent in display
            index=ROW_LABELS[::-1],
            columns=COL_LABELS,
        )
        st.markdown(f"**{label}**")
        st.dataframe(tdf, use_container_width=True)
