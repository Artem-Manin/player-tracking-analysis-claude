"""Page 4 — Position Heatmap by Role"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
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
    show_pitch = st.checkbox("Show pitch markings", value=True)
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

ROW_LABELS = ["Our Goal", "Zone 2", "Zone 3", "Zone 4", "Opponent"]
COL_LABELS = ["South", "Centre", "North"]
Y_VALS = list(range(N_GRID_ROWS))   # 0=Our Goal (bottom) … 4=Opponent (top)
X_VALS = list(range(N_GRID_COLS))   # 0=South … 2=North

# ── Real pitch dimensions (metres) ───────────────────────────────────────────
# Standard pitch: 105m long × 68m wide
# Our grid: 5 rows (length) × 3 cols (width)
# To get a real circle we must fix aspect ratio to 105:68 * (3/5) = 0.926
# i.e. for every 1 unit in x, y must span 105/68 * (3 cols / 5 rows) = 0.926 units
# We achieve this by setting the plot pixel dimensions: width/height = 3/5 * 68/105 = 0.389 * (px)
# Practically: height = width * (5/3) * (68/105)
# We'll use width=420px per heatmap in side-by-side, 560 standalone → height derived.

PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M  = 68.0

# Grid units: x ∈ [0,2] (cols), y ∈ [0,4] (rows)
# 1 grid-col = 68/3 m, 1 grid-row = 105/5 m = 21m
# For shapes: convert metres → grid units
def m_to_col(m): return m / (PITCH_WIDTH_M  / N_GRID_COLS)   # metres across → col units
def m_to_row(m): return m / (PITCH_LENGTH_M / N_GRID_ROWS)   # metres along  → row units

# Penalty area: 40.32m wide, 16.5m deep
PA_W  = m_to_col(40.32) / 2        # half-width in col units from centre (col=1)
PA_D  = m_to_row(16.5)             # depth in row units

# Goal area: 18.32m wide, 5.5m deep
GA_W  = m_to_col(18.32) / 2
GA_D  = m_to_row(5.5)

# Centre circle: r=9.15m
CC_RX = m_to_col(9.15)             # radius in col units
CC_RY = m_to_row(9.15)             # radius in row units  (will be equal if aspect correct)

# Penalty spots: 11m from goal line
PS_D_OUR = m_to_row(11.0)          # from y=−0.5 (goal line) = y = −0.5 + PS_D_OUR
PS_D_OPP = 4.5 - PS_D_OUR         # from top

# Goal width: 7.32m
GOAL_W = m_to_col(7.32) / 2

LINE = dict(color="white", width=2)
FILL = "rgba(0,0,0,0)"


def _pitch_shapes(show: bool) -> list:
    if not show:
        return []

    shapes = []

    # Pitch border — filled green
    shapes.append(dict(type="rect", x0=-0.5, x1=2.5, y0=-0.5, y1=4.5,
                       line=LINE, fillcolor="#2d6a2d", layer="below"))

    # Halfway line
    shapes.append(dict(type="line", x0=-0.5, x1=2.5, y0=2.0, y1=2.0, line=LINE))

    # Our Goal — penalty area (bottom)
    shapes.append(dict(type="rect",
                       x0=1 - PA_W, x1=1 + PA_W, y0=-0.5, y1=-0.5 + PA_D,
                       line=LINE, fillcolor=FILL))

    # Our Goal — goal area (bottom)
    shapes.append(dict(type="rect",
                       x0=1 - GA_W, x1=1 + GA_W, y0=-0.5, y1=-0.5 + GA_D,
                       line=LINE, fillcolor=FILL))

    # Our Goal — goal mouth on border
    shapes.append(dict(type="rect",
                       x0=1 - GOAL_W, x1=1 + GOAL_W, y0=-0.55, y1=-0.5,
                       line=dict(color="white", width=3), fillcolor="rgba(255,255,255,0.15)"))

    # Opponent — penalty area (top)
    shapes.append(dict(type="rect",
                       x0=1 - PA_W, x1=1 + PA_W, y0=4.5 - PA_D, y1=4.5,
                       line=LINE, fillcolor=FILL))

    # Opponent — goal area (top)
    shapes.append(dict(type="rect",
                       x0=1 - GA_W, x1=1 + GA_W, y0=4.5 - GA_D, y1=4.5,
                       line=LINE, fillcolor=FILL))

    # Opponent — goal mouth on border
    shapes.append(dict(type="rect",
                       x0=1 - GOAL_W, x1=1 + GOAL_W, y0=4.5, y1=4.55,
                       line=dict(color="white", width=3), fillcolor="rgba(255,255,255,0.15)"))

    # Centre circle — ellipse with correct rx/ry in grid units
    shapes.append(dict(type="circle",
                       x0=1 - CC_RX, x1=1 + CC_RX,
                       y0=2.0 - CC_RY, y1=2.0 + CC_RY,
                       line=LINE, fillcolor=FILL))

    # Centre spot
    shapes.append(dict(type="circle", x0=0.98, x1=1.02, y0=1.98, y1=2.02,
                       line=dict(color="white", width=1),
                       fillcolor="white"))

    # Our Goal penalty spot
    ps_our_y = -0.5 + PS_D_OUR
    shapes.append(dict(type="circle",
                       x0=0.98, x1=1.02, y0=ps_our_y - 0.02, y1=ps_our_y + 0.02,
                       line=dict(color="white", width=1), fillcolor="white"))

    # Opponent penalty spot
    ps_opp_y = 4.5 - PS_D_OUR
    shapes.append(dict(type="circle",
                       x0=0.98, x1=1.02, y0=ps_opp_y - 0.02, y1=ps_opp_y + 0.02,
                       line=dict(color="white", width=1), fillcolor="white"))

    return shapes


def build_grid_z(subset: pd.DataFrame) -> np.ndarray:
    z = np.zeros((N_GRID_ROWS, N_GRID_COLS), dtype=int)
    grp = subset.groupby(["grid_row", "grid_col"]).size()
    for (r, c), cnt in grp.items():
        if 0 <= r < N_GRID_ROWS and 0 <= c < N_GRID_COLS:
            z[r, c] = cnt
    return z


def pitch_heatmap(z: np.ndarray, title: str, colorscale: str,
                  show_pts: bool, show_pitch_lines: bool,
                  pts_df: pd.DataFrame | None = None,
                  standalone: bool = True) -> go.Figure:
    """
    Our Goal at y=0 (bottom), Opponent at y=4 (top).
    Aspect ratio locked to real pitch proportions so centre circle is round.
    """
    z_norm = z / z.max() if z.max() > 0 else z.astype(float)

    fig = go.Figure()

    # Heatmap — pass z directly, y=0 = bottom = Our Goal
    fig.add_trace(go.Heatmap(
        z=z,
        x=X_VALS,
        y=Y_VALS,
        colorscale=colorscale,
        showscale=True,
        colorbar=dict(
            title="Records",
            lenmode="fraction", len=0.75,
            yanchor="middle", y=0.5,
            thickness=15,
        ),
        zmin=0,
        opacity=0.72,          # let pitch green show through slightly
        hovertemplate="<b>%{customdata[0]}</b> | %{customdata[1]}<br>Records: %{z}<extra></extra>",
        customdata=[
            [ROW_LABELS[r], COL_LABELS[c]]
            for r in Y_VALS for c in X_VALS
        ],
    ))

    # Cell annotations
    total = z.sum()
    for r in Y_VALS:
        for c in X_VALS:
            val = z[r, c]
            pct = val / total * 100 if total > 0 else 0
            text_color = "white" if z_norm[r, c] > 0.45 else "#e0e0e0"
            fig.add_annotation(
                x=c, y=r,
                text=f"<b>{val}</b><br><span style='font-size:9px'>{pct:.1f}%</span>",
                showarrow=False,
                font=dict(color=text_color, size=11),
            )

    # Pitch shapes
    for shape in _pitch_shapes(show_pitch_lines):
        fig.add_shape(**shape)

    # Goal labels outside pitch border
    fig.add_annotation(x=2.7, y=0, text="⚽ Our Goal", showarrow=False,
                       font=dict(color="#00e5ff", size=11), xref="x", yref="y")
    fig.add_annotation(x=2.7, y=4, text="🥅 Opponent", showarrow=False,
                       font=dict(color="#ff7f0e", size=11), xref="x", yref="y")

    # Raw GPS overlay
    if show_pts and pts_df is not None and not pts_df.empty:
        fig.add_trace(go.Scatter(
            x=pts_df["grid_col"], y=pts_df["grid_row"],
            mode="markers",
            marker=dict(color="white", size=2, opacity=0.12),
            name="GPS points", hoverinfo="skip",
        ))

    # ── Aspect ratio: lock to real pitch proportions ──────────────────────────
    # grid is N_GRID_COLS wide × N_GRID_ROWS tall in grid units
    # real pitch is PITCH_WIDTH_M wide × PITCH_LENGTH_M tall
    # so 1 col-unit : 1 row-unit = (PITCH_WIDTH_M/N_GRID_COLS) : (PITCH_LENGTH_M/N_GRID_ROWS)
    # pixels_per_col / pixels_per_row = (PITCH_WIDTH_M/N_GRID_COLS) / (PITCH_LENGTH_M/N_GRID_ROWS)
    col_m = PITCH_WIDTH_M  / N_GRID_COLS   # metres per col unit = 22.67m
    row_m = PITCH_LENGTH_M / N_GRID_ROWS   # metres per row unit = 21.0m
    # ratio of col:row in metres — if 1:1 in pixels the circle will be round
    aspect_ratio = col_m / row_m           # ≈ 1.079

    plot_width  = 560 if standalone else 420
    plot_height = int(plot_width * (N_GRID_ROWS / N_GRID_COLS) / aspect_ratio)

    fig.update_layout(
        title=title,
        template="plotly_dark",
        width=plot_width,
        height=plot_height,
        xaxis=dict(
            title="Across pitch width",
            tickvals=X_VALS, ticktext=COL_LABELS,
            range=[-0.5, 3.1],
            showgrid=False, zeroline=False, scaleanchor="y", scaleratio=aspect_ratio,
        ),
        yaxis=dict(
            title="Our goal ↕ Opponent",
            tickvals=Y_VALS, ticktext=ROW_LABELS,
            range=[-0.6, 4.6],
            showgrid=False, zeroline=False,
        ),
        plot_bgcolor="#0d1117",   # dark — pitch border shape provides the green
        paper_bgcolor="#0d1117",
        margin=dict(r=110, t=50, l=80, b=60),
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
c1.metric("GK records",       f"{n_gk:,}",  f"{n_gk/n_all*100:.1f}%")
c2.metric("Outfield records",  f"{n_out:,}", f"{n_out/n_all*100:.1f}%")
c3.metric("Total records",    f"{n_all:,}")

tab1, tab2, tab3 = st.tabs(["🥅 GK windows", "🏃 Outfield windows", "📊 Combined"])

with tab1:
    if df_gk.empty:
        st.info("No GK windows detected.")
    else:
        z = build_grid_z(df_gk)
        st.plotly_chart(
            pitch_heatmap(z, "GK Position Density", color_scale, show_raw, show_pitch, df_gk, standalone=True),
            use_container_width=False,
        )
        st.caption(
            f"Showing {n_gk:,} records from **{df_gk['_window'].nunique()} GK windows**. "
            "Density concentrates near our goal — as expected for a GK."
        )

with tab2:
    if df_out.empty:
        st.info("No Outfield windows detected.")
    else:
        z = build_grid_z(df_out)
        st.plotly_chart(
            pitch_heatmap(z, "Outfield Position Density", color_scale, show_raw, show_pitch, df_out, standalone=True),
            use_container_width=False,
        )
        st.caption(f"Showing {n_out:,} records from **{df_out['_window'].nunique()} Outfield windows**.")

with tab3:
    z = build_grid_z(df_all)
    st.plotly_chart(
        pitch_heatmap(z, "Overall Position Density (All Windows)", color_scale, show_raw, show_pitch, df_all, standalone=True),
        use_container_width=False,
    )

# ── side-by-side comparison ───────────────────────────────────────────────────
st.divider()
st.markdown("## Side-by-side: GK vs Outfield")

col_gk, col_out = st.columns(2)
with col_gk:
    if not df_gk.empty:
        st.plotly_chart(
            pitch_heatmap(build_grid_z(df_gk), "GK", color_scale, False, show_pitch, standalone=False),
            use_container_width=False,
        )
with col_out:
    if not df_out.empty:
        st.plotly_chart(
            pitch_heatmap(build_grid_z(df_out), "Outfield", color_scale, False, show_pitch, standalone=False),
            use_container_width=False,
        )

# ── grid counts table ─────────────────────────────────────────────────────────
with st.expander("📋 Raw grid counts"):
    for label, subset in [("GK", df_gk), ("Outfield", df_out), ("All", df_all)]:
        z = build_grid_z(subset)
        tdf = pd.DataFrame(
            z[::-1],
            index=ROW_LABELS[::-1],
            columns=COL_LABELS,
        )
        st.markdown(f"**{label}**")
        st.dataframe(tdf, use_container_width=True)