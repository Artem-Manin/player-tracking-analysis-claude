"""Page 3 — Goalkeeper Role Clustering"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
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
    rule_based_gk, cluster_gk, compare_methods,
    GK_X_FRACTION, GK_Y_FRACTION_LOW, GK_Y_FRACTION_HIGH, GK_SPEED_THRESHOLD_MS,
    WINDOW_SECONDS, CLUSTER_FEATURES,
)

# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="GK Clustering", page_icon="🥅", layout="wide")
st.markdown(DARK_CSS, unsafe_allow_html=True)
st.markdown("# 🥅 Goalkeeper Role Detection")
st.markdown(
    "The player alternates between **goalkeeping** and **outfield** activity. "
    "We aggregate GPS records into **3-minute windows** and classify each window "
    "using two independent methods, then compare them."
)

if not require_session("df1"):
    st.stop()

meta = get_meta("meta1")
if not meta.get("has_gps"):
    unavailable("goalkeeper_clustering", meta)
    st.stop()

# ── compute ──────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Preparing windows…")
def compute(_df_json):
    df = pd.read_json(_df_json, orient="split")
    if "ts_utc" in df.columns:
        df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True, unit="ms", errors="coerce")
        if df["ts_utc"].isna().all():
            df["ts_utc"] = pd.to_datetime(
                pd.read_json(_df_json, orient="split")["ts_utc"], utc=True
            )
    df, x_max, y_max = add_metric_coords(df)
    df = add_grid_cells(df, x_max, y_max)
    windows = build_windows(df)
    windows = rule_based_gk(windows, x_max, y_max)
    windows, cluster_info = cluster_gk(windows, x_max)
    agreement = compare_methods(windows)
    return df, windows, x_max, y_max, cluster_info, agreement

raw_df = get_df("df1")
df, windows, x_max, y_max, cluster_info, agreement = compute(
    raw_df.to_json(orient="split", date_format="epoch", date_unit="ms")
)

# ── sidebar: methodology ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Rule-based thresholds")
    st.markdown(f"- **Our half** x > `{GK_X_FRACTION*100:.0f}%` of field length")
    st.markdown(f"- **Central band** y ∈ `{GK_Y_FRACTION_LOW*100:.0f}%–{GK_Y_FRACTION_HIGH*100:.0f}%` of width")
    st.markdown(f"- **Mean speed** < `{GK_SPEED_THRESHOLD_MS} m/s`")
    st.divider()
    st.markdown("## 🔬 Clustering features")
    for f in CLUSTER_FEATURES:
        st.markdown(f"- `{f}`")
    st.markdown("*k = 2, StandardScaler + KMeans*")
    st.divider()
    st.markdown("## 🏟️ Field orientation")
    st.markdown("East = **Our goal** (bottom of pitch)\nWest = Opponent goal (top)")

st.divider()

# ── agreement banner ─────────────────────────────────────────────────────────
col_a, col_b, col_c, col_d, col_e = st.columns(5)
col_a.metric("3-min windows", agreement["n_windows"])
col_b.metric("Agreement", f"{agreement['agreement_pct']}%")
col_c.metric("Both → GK", agreement["both_gk"])
col_d.metric("Rule only → GK", agreement["rule_only_gk"])
col_e.metric("Cluster only → GK", agreement["cluster_only_gk"])

if cluster_info.get("fallback"):
    st.warning(
        f"⚠️ **Clustering skipped** — {cluster_info['reason']} "
        "Showing rule-based labels only. Upload a longer session to enable clustering.",
        icon="⚠️",
    )

if agreement["agreement_pct"] >= 90:
    st.success(
        f"✅ Both methods agree on **{agreement['agreement_pct']}%** of windows. "
        f"Rule-based: {agreement['rule_total_gk']} GK windows. "
        f"Clustering: {agreement['cluster_total_gk']} GK windows.",
        icon="✅",
    )
else:
    st.warning(
        f"⚠️ Methods agree on {agreement['agreement_pct']}% of windows. "
        "Inspect the scatter plot and table below for discrepancies.",
        icon="⚠️",
    )

st.divider()

# ── shared colour maps ────────────────────────────────────────────────────────
COLOR_RULE    = {"GK": "#00e5ff", "Outfield": "#ff7f0e"}
COLOR_CLUSTER = {"GK": "#3fb950", "Outfield": "#f85149"}

windows["rule_label"]        = windows["rule_gk"].map({True: "GK", False: "Outfield"})
windows["cluster_label_str"] = windows["cluster_gk"].map({True: "GK", False: "Outfield"})
windows["window_min"]        = (windows["window_idx"] * WINDOW_SECONDS / 60).round(1)

# ── merge labels onto raw records ────────────────────────────────────────────
df_merged = df.copy()
if "ts_utc" in df_merged.columns:
    df_merged["_window"] = (
        (df_merged["ts_utc"] - df_merged["ts_utc"].min()).dt.total_seconds() // WINDOW_SECONDS
    ).astype(int)
else:
    df_merged["_window"] = 0
df_merged = df_merged.merge(
    windows[["window_idx", "rule_label", "cluster_label_str"]],
    left_on="_window", right_on="window_idx", how="left",
)
df_merged["speed_kmh"] = df_merged["speed_ms"] * 3.6

# ════════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Window-level 2-D scatter (position only)
# ════════════════════════════════════════════════════════════════════════════════
st.markdown("## 📍 Window-level view — position (x, y)")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📐 Rule-based")
    fig = px.scatter(
        windows, x="mean_x", y="mean_y",
        color="rule_label",
        color_discrete_map=COLOR_RULE,
        size="pct_slow",
        hover_data={"window_min": True, "mean_speed": ":.3f", "pct_slow": ":.2f",
                    "rule_in_our_half": True, "rule_central_y": True, "rule_slow": True},
        labels={"mean_x": "x (m, W→E, our goal=right)", "mean_y": "y (m, S→N)"},
        title="3-min windows coloured by rule-based label",
        template="plotly_dark",
    )
    fig.add_shape(type="rect",
        x0=GK_X_FRACTION * x_max, x1=x_max,
        y0=GK_Y_FRACTION_LOW * y_max, y1=GK_Y_FRACTION_HIGH * y_max,
        line=dict(color="#00e5ff", dash="dash", width=1.5),
        fillcolor="rgba(0,229,255,0.05)",
    )
    fig.add_annotation(
        x=(GK_X_FRACTION * x_max + x_max) / 2,
        y=GK_Y_FRACTION_HIGH * y_max,
        text="GK zone", showarrow=False,
        font=dict(color="#00e5ff", size=10),
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("### 🤖 KMeans clustering")
    fig2 = px.scatter(
        windows, x="mean_x", y="mean_y",
        color="cluster_label_str",
        color_discrete_map=COLOR_CLUSTER,
        size="pct_slow",
        hover_data={"window_min": True, "mean_speed": ":.3f", "pct_slow": ":.2f"},
        labels={"mean_x": "x (m, W→E, our goal=right)", "mean_y": "y (m, S→N)"},
        title="3-min windows coloured by cluster label",
        template="plotly_dark",
    )
    st.plotly_chart(fig2, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Speed distributions
# ════════════════════════════════════════════════════════════════════════════════
st.divider()
st.markdown("## ⚡ Speed distribution by role")
col3, col4 = st.columns(2)

with col3:
    fig3 = px.violin(
        df_merged, x="rule_label", y="speed_kmh",
        color="rule_label", color_discrete_map=COLOR_RULE,
        box=True, points=False,
        title="Speed distribution — Rule-based",
        labels={"rule_label": "Role", "speed_kmh": "Speed (km/h)"},
        template="plotly_dark",
    )
    st.plotly_chart(fig3, use_container_width=True)

with col4:
    fig4 = px.violin(
        df_merged, x="cluster_label_str", y="speed_kmh",
        color="cluster_label_str", color_discrete_map=COLOR_CLUSTER,
        box=True, points=False,
        title="Speed distribution — Clustering",
        labels={"cluster_label_str": "Role", "speed_kmh": "Speed (km/h)"},
        template="plotly_dark",
    )
    st.plotly_chart(fig4, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Role timeline
# ════════════════════════════════════════════════════════════════════════════════
st.divider()
st.markdown("## 🕐 Role timeline across session")
fig_tl = go.Figure()
fig_tl.add_trace(go.Scatter(
    x=windows["window_min"],
    y=windows["rule_gk"].astype(int),
    mode="lines+markers",
    name="Rule-based GK",
    line=dict(color="#00e5ff", width=2, shape="hv"),
    marker=dict(size=7),
))
fig_tl.add_trace(go.Scatter(
    x=windows["window_min"],
    y=windows["cluster_gk"].astype(int) * 0.9,
    mode="lines+markers",
    name="Cluster GK",
    line=dict(color="#3fb950", width=2, shape="hv", dash="dot"),
    marker=dict(size=7, symbol="diamond"),
))
fig_tl.update_layout(
    template="plotly_dark",
    title="GK windows over session (1=GK, 0=Outfield)",
    xaxis_title="Session time (minutes)",
    yaxis=dict(tickvals=[0, 1], ticktext=["Outfield", "GK"], range=[-0.2, 1.3]),
    height=250,
)
st.plotly_chart(fig_tl, use_container_width=True)

st.divider()

# ════════════════════════════════════════════════════════════════════════════════
# SECTION 4 — NEW: 3-D scatter  (x, y, speed)
# ════════════════════════════════════════════════════════════════════════════════
st.markdown("## 🧊 3-D feature space — position + speed")
st.caption(
    "Each point is one 3-minute window. Axes: x = position along pitch, "
    "y = position across pitch, z = mean speed. Bubble size = % time slow. "
    "Rotate to inspect cluster separation in 3-D."
)

method_3d = st.radio("Colour by", ["Rule-based", "Clustering"], horizontal=True, key="r3d")
color_col_3d = "rule_label"   if method_3d == "Rule-based" else "cluster_label_str"
color_map_3d = COLOR_RULE     if method_3d == "Rule-based" else COLOR_CLUSTER

windows["_size3d"] = (windows["pct_slow"] * 20 + 4).clip(4, 30)

fig_3d = px.scatter_3d(
    windows,
    x="mean_x", y="mean_y", z="mean_speed",
    color=color_col_3d,
    color_discrete_map=color_map_3d,
    size="_size3d",
    size_max=20,
    hover_data={"window_min": True, "mean_speed": ":.3f",
                "pct_slow": ":.2f", "std_x": ":.2f", "std_y": ":.2f"},
    labels={
        "mean_x": "x (m) → our goal",
        "mean_y": "y (m) → north",
        "mean_speed": "mean speed (m/s)",
    },
    title=f"3-D: position (x, y) + speed — {method_3d}",
    template="plotly_dark",
)
fig_3d.update_layout(height=600, margin=dict(l=0, r=0, t=40, b=0))
fig_3d.update_traces(marker=dict(opacity=0.85, line=dict(width=0)))
st.plotly_chart(fig_3d, use_container_width=True)

st.divider()

# ════════════════════════════════════════════════════════════════════════════════
# SECTION 5 — NEW: Combined-score 2-D  (position score vs mobility score)
# ════════════════════════════════════════════════════════════════════════════════
st.markdown("## 🎯 Combined-score view — position vs mobility")
st.caption(
    "**Position score** (x-axis): how deep in our half + how central. "
    "**Mobility score** (y-axis): how slow/stationary (inverted speed + % standing). "
    "GK windows should cluster top-right. Both axes normalised 0–1."
)

w = windows.copy()

x_frac    = w["mean_x"] / x_max
y_central = 1 - 2 * (w["mean_y"] / y_max - 0.5).abs()
pos_score = (x_frac + y_central) / 2

spd_norm  = (w["mean_speed"] - w["mean_speed"].min()) / (
    w["mean_speed"].max() - w["mean_speed"].min() + 1e-9
)
mob_score = (w["pct_slow"] + (1 - spd_norm)) / 2

w["position_score"] = pos_score.round(3)
w["mobility_score"] = mob_score.round(3)

method_cs = st.radio("Colour by", ["Rule-based", "Clustering"], horizontal=True, key="rcs")
color_col_cs = "rule_label"   if method_cs == "Rule-based" else "cluster_label_str"
color_map_cs = COLOR_RULE     if method_cs == "Rule-based" else COLOR_CLUSTER

fig_cs = px.scatter(
    w, x="position_score", y="mobility_score",
    color=color_col_cs,
    color_discrete_map=color_map_cs,
    size="n_records",
    size_max=18,
    hover_data={"window_min": True, "mean_speed": ":.3f",
                "mean_x": ":.1f", "mean_y": ":.1f", "pct_slow": ":.2f"},
    labels={
        "position_score": "Position score  (0 = opponent side/sideline → 1 = our goal/centre)",
        "mobility_score": "Mobility score  (0 = fast/active → 1 = slow/stationary)",
    },
    title=f"Combined scores — {method_cs}",
    template="plotly_dark",
)
fig_cs.add_shape(
    type="rect", x0=0.5, x1=1.02, y0=0.5, y1=1.02,
    fillcolor="rgba(0,229,255,0.05)",
    line=dict(color="#00e5ff", dash="dash", width=1),
)
fig_cs.add_annotation(
    x=0.76, y=1.01, text="Expected GK zone", showarrow=False,
    font=dict(color="#00e5ff", size=11),
)
fig_cs.update_layout(height=480)
st.plotly_chart(fig_cs, use_container_width=True)

st.divider()

# ════════════════════════════════════════════════════════════════════════════════
# SECTION 6 — NEW: PCA projection of all 6 clustering features
# ════════════════════════════════════════════════════════════════════════════════
st.markdown("## 🔬 PCA projection — what KMeans actually sees")
st.caption(
    "All 6 features (`mean_x`, `mean_y`, `std_x`, `std_y`, `mean_speed`, `pct_slow`) "
    "are standardised and projected onto their first two principal components — the exact "
    "space where KMeans draws the decision boundary. Arrows show each feature's direction "
    "and weight in this space."
)

X_raw    = windows[CLUSTER_FEATURES].fillna(0).values
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

pca    = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(X_scaled)

pca_df = windows[["window_min", "rule_label", "cluster_label_str",
                   "mean_speed", "pct_slow", "mean_x", "mean_y"]].copy()
pca_df["PC1"] = coords[:, 0]
pca_df["PC2"] = coords[:, 1]

ev1, ev2 = pca.explained_variance_ratio_ * 100

loadings = pd.DataFrame(
    pca.components_.T,
    index=CLUSTER_FEATURES,
    columns=["PC1", "PC2"],
).round(3)

method_pca = st.radio("Colour by", ["Rule-based", "Clustering"], horizontal=True, key="rpca")
color_col_pca = "rule_label"   if method_pca == "Rule-based" else "cluster_label_str"
color_map_pca = COLOR_RULE     if method_pca == "Rule-based" else COLOR_CLUSTER

fig_pca = px.scatter(
    pca_df, x="PC1", y="PC2",
    color=color_col_pca,
    color_discrete_map=color_map_pca,
    size="pct_slow",
    size_max=18,
    hover_data={"window_min": True, "mean_speed": ":.3f",
                "mean_x": ":.1f", "mean_y": ":.1f", "pct_slow": ":.2f"},
    labels={
        "PC1": f"PC1 ({ev1:.1f}% variance)",
        "PC2": f"PC2 ({ev2:.1f}% variance)",
    },
    title=f"PCA of 6 clustering features — {method_pca}  |  {ev1+ev2:.1f}% total variance explained",
    template="plotly_dark",
)

arrow_scale = 2.5
for feat in CLUSTER_FEATURES:
    lx = loadings.loc[feat, "PC1"] * arrow_scale
    ly = loadings.loc[feat, "PC2"] * arrow_scale
    fig_pca.add_annotation(
        ax=0, ay=0, x=lx, y=ly,
        axref="x", ayref="y", xref="x", yref="y",
        showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.5,
        arrowcolor="#8b949e",
    )
    fig_pca.add_annotation(
        x=lx * 1.18, y=ly * 1.18,
        text=feat.replace("_", " "),
        showarrow=False,
        font=dict(color="#8b949e", size=10),
    )

fig_pca.update_layout(height=520)
st.plotly_chart(fig_pca, use_container_width=True)

with st.expander("📊 PCA feature loadings"):
    st.caption("How much each original feature contributes to each principal component.")
    loadings_display = loadings.copy()
    loadings_display.index.name = "Feature"
    loadings_display.columns = [
        f"PC1 ({ev1:.1f}% var)",
        f"PC2 ({ev2:.1f}% var)",
    ]
    st.dataframe(loadings_display, use_container_width=True)

st.divider()

# ── Detailed window table ─────────────────────────────────────────────────────
with st.expander("📋 Full window table"):
    show_cols = ["window_idx", "window_min", "mean_x", "mean_y", "mean_speed",
                 "max_speed", "pct_slow", "rule_gk", "cluster_gk", "n_records"]
    disp = windows[show_cols].copy()
    disp["mean_speed"] = disp["mean_speed"].round(3)
    disp["max_speed"]  = disp["max_speed"].round(3)
    disp["pct_slow"]   = (disp["pct_slow"] * 100).round(1)
    disp.columns = [c.replace("_", " ").title() for c in disp.columns]
    disp = disp.rename(columns={
        "Pct Slow": "% Slow (%)",
        "Mean Speed": "Mean Speed (m/s)",
        "Max Speed": "Max Speed (m/s)",
    })
    st.dataframe(disp, use_container_width=True, hide_index=True)

st.caption(
    "**Methodology note**: Windows where the player's mean position falls in the eastern 45% of the "
    "tracked area, within the central corridor, and with mean speed < 0.9 m/s are classified as GK "
    "by the rule-based approach. The clustering approach uses KMeans(k=2) on position + mobility "
    "features; the GK cluster is identified as the one with higher mean x and lower mean speed."
)