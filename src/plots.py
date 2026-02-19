"""
plots.py
--------
Reusable Matplotlib/Plotly figure factories used by both notebooks and
the Streamlit app.

All functions return a Plotly Figure (go.Figure) for easy embedding.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Colour palette
PITCH_GREEN = "#2d5a27"
LINE_COLOR = "#00e5ff"
ACCENT = "#ff4136"
ZONE_COLORS = {
    "Standing":   "#1f77b4",
    "Walking":    "#aec7e8",
    "Jogging":    "#98df8a",
    "Running":    "#ffbb78",
    "High Speed": "#ff7f0e",
    "Sprinting":  "#d62728",
}


# ---------------------------------------------------------------------------
# GPS Speed Validation
# ---------------------------------------------------------------------------
def plot_speed_timeseries(df: pd.DataFrame) -> go.Figure:
    """Overlay device speed vs Haversine-recomputed speed over time."""
    ts_col = "ts_cet" if "ts_cet" in df.columns else "ts_utc"
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df[ts_col], y=df["speed_ms"] * 3.6,
        name="Device speed", line=dict(color="#00e5ff", width=1.2),
    ))
    if "speed_haversine_ms" in df.columns:
        fig.add_trace(go.Scatter(
            x=df[ts_col], y=df["speed_haversine_ms"] * 3.6,
            name="Haversine speed", line=dict(color="#ff7f0e", width=1.2, dash="dot"),
        ))
    if "speed_flagged" in df.columns:
        flagged = df[df["speed_flagged"]]
        if not flagged.empty:
            fig.add_trace(go.Scatter(
                x=flagged[ts_col], y=flagged["speed_ms"] * 3.6,
                mode="markers", name="Flagged",
                marker=dict(color="#ff4136", size=5),
            ))
    fig.update_layout(
        title="GPS Speed Validation — Device vs Haversine",
        xaxis_title="Time (CET)", yaxis_title="Speed (km/h)",
        template="plotly_dark", legend=dict(orientation="h"),
    )
    return fig


def plot_speed_scatter(df: pd.DataFrame) -> go.Figure:
    """Scatter: device speed vs haversine speed."""
    if "speed_haversine_ms" not in df.columns:
        return go.Figure()
    valid = df[["speed_ms", "speed_haversine_ms"]].dropna()
    max_v = max(valid["speed_ms"].max(), valid["speed_haversine_ms"].max()) * 3.6 * 1.05
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=valid["speed_ms"] * 3.6, y=valid["speed_haversine_ms"] * 3.6,
        mode="markers", marker=dict(color="#00e5ff", size=3, opacity=0.5),
        name="Samples",
    ))
    fig.add_trace(go.Scatter(
        x=[0, max_v], y=[0, max_v],
        mode="lines", line=dict(color="#ff4136", dash="dash"),
        name="Perfect agreement",
    ))
    fig.update_layout(
        title="Device Speed vs Haversine Speed",
        xaxis_title="Device speed (km/h)", yaxis_title="Haversine speed (km/h)",
        template="plotly_dark",
    )
    return fig


def plot_speed_error_histogram(df: pd.DataFrame) -> go.Figure:
    """Histogram of speed errors (device − haversine)."""
    if "speed_error_ms" not in df.columns:
        return go.Figure()
    err_kmh = df["speed_error_ms"].dropna() * 3.6
    fig = px.histogram(
        err_kmh, nbins=60, color_discrete_sequence=["#00e5ff"],
        title="Speed Error Distribution (Device − Haversine)",
        labels={"value": "Error (km/h)", "count": "Frequency"},
        template="plotly_dark",
    )
    fig.add_vline(x=0, line_color="#ff4136", line_dash="dash")
    return fig


# ---------------------------------------------------------------------------
# Heatmap
# ---------------------------------------------------------------------------
def plot_position_heatmap(df: pd.DataFrame, cluster_summary: pd.DataFrame | None = None) -> go.Figure:
    """
    2-D density heatmap of GPS positions (lat/lon converted to metres).
    Optionally overlays cluster centroids.
    """
    lat0 = df["latitude"].mean()
    lon0 = df["longitude"].mean()
    x_m = (df["longitude"] - lon0) * 111_319 * np.cos(np.radians(lat0))
    y_m = (df["latitude"] - lat0) * 111_319

    fig = go.Figure()
    fig.add_trace(go.Histogram2dContour(
        x=x_m, y=y_m,
        colorscale="YlOrRd",
        showscale=True,
        contours=dict(showlines=False),
        name="Density",
    ))
    fig.add_trace(go.Scatter(
        x=x_m, y=y_m,
        mode="markers",
        marker=dict(color="#00e5ff", size=2, opacity=0.3),
        name="GPS points",
    ))

    if cluster_summary is not None and not cluster_summary.empty:
        cx_m = (cluster_summary["mean_lon"] - lon0) * 111_319 * np.cos(np.radians(lat0))
        cy_m = (cluster_summary["mean_lat"] - lat0) * 111_319
        labels = [
            f"Cluster {row['cluster']}" + (" ★GK" if row.get("likely_gk") else "")
            for _, row in cluster_summary.iterrows()
        ]
        fig.add_trace(go.Scatter(
            x=cx_m, y=cy_m,
            mode="markers+text",
            marker=dict(color=ACCENT, size=14, symbol="star"),
            text=labels, textposition="top center",
            name="Cluster centres",
        ))

    fig.update_layout(
        title="Player Position Heatmap",
        xaxis_title="East–West (m, +East)", yaxis_title="North–South (m, +North)",
        template="plotly_dark",
        yaxis_scaleanchor="x",  # equal aspect ratio
    )
    return fig


# ---------------------------------------------------------------------------
# Speed distribution
# ---------------------------------------------------------------------------
def plot_speed_zone_bar(zone_df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart of time spent per speed zone."""
    if zone_df.empty:
        return go.Figure()
    colors = [ZONE_COLORS.get(z, "#888") for z in zone_df["zone"]]
    fig = go.Figure(go.Bar(
        y=zone_df["zone"].astype(str),
        x=zone_df["pct"],
        orientation="h",
        marker_color=colors,
        text=[f"{p}%" for p in zone_df["pct"]],
        textposition="outside",
    ))
    fig.update_layout(
        title="Speed Zone Distribution (% time)",
        xaxis_title="% of session", yaxis_title="Zone",
        template="plotly_dark",
        xaxis=dict(range=[0, zone_df["pct"].max() * 1.2]),
    )
    return fig


# ---------------------------------------------------------------------------
# Speed & acceleration trajectories
# ---------------------------------------------------------------------------
def plot_speed_trajectory(df: pd.DataFrame) -> go.Figure:
    """GPS track coloured by speed."""
    lat0 = df["latitude"].mean()
    lon0 = df["longitude"].mean()
    x_m = (df["longitude"] - lon0) * 111_319 * np.cos(np.radians(lat0))
    y_m = (df["latitude"] - lat0) * 111_319
    speed_kmh = df["speed_ms"] * 3.6 if "speed_ms" in df.columns else pd.Series(np.zeros(len(df)))

    fig = go.Figure(go.Scatter(
        x=x_m, y=y_m,
        mode="lines+markers",
        marker=dict(
            color=speed_kmh,
            colorscale="RdYlGn_r",
            size=4,
            colorbar=dict(title="km/h"),
        ),
        line=dict(color="rgba(150,150,150,0.3)", width=1),
    ))
    fig.update_layout(
        title="Speed Trajectory (coloured by speed)",
        xaxis_title="East–West (m)", yaxis_title="North–South (m)",
        template="plotly_dark",
        yaxis_scaleanchor="x",
    )
    return fig


def plot_acc_trajectory(df: pd.DataFrame) -> go.Figure:
    """IMU total acceleration over time."""
    ts_col = "ts_cet" if "ts_cet" in df.columns else "ts_utc"
    if "acc_magnitude_g" not in df.columns:
        return go.Figure()
    fig = go.Figure(go.Scatter(
        x=df[ts_col], y=df["acc_magnitude_g"],
        mode="lines",
        line=dict(color="#00e5ff", width=0.8),
        name="|Acc| (g)",
    ))
    fig.update_layout(
        title="Total Acceleration Magnitude Over Time",
        xaxis_title="Time", yaxis_title="Acceleration (g)",
        template="plotly_dark",
    )
    return fig


# ---------------------------------------------------------------------------
# IMU movement detection
# ---------------------------------------------------------------------------
def plot_movement_events(df: pd.DataFrame) -> go.Figure:
    """Plot IMU time-series with overlay of detected movement events."""
    ts_col = "ts_cet" if "ts_cet" in df.columns else "ts_utc"
    has_ts = ts_col in df.columns

    x_axis = df[ts_col] if has_ts else df.index

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=["Rotation magnitude (°/s)", "Roll angle (°)"])

    if "rot_magnitude_dps" in df.columns:
        fig.add_trace(go.Scatter(x=x_axis, y=df["rot_magnitude_dps"],
                                 line=dict(color="#00e5ff", width=0.8), name="Rotation"), row=1, col=1)

    for event_col, color, label in [
        ("is_twist", "#ff4136", "Twist"),
        ("is_turn", "#ff7f0e", "Turn"),
    ]:
        if event_col in df.columns:
            ev = df[df[event_col]]
            ev_x = ev[ts_col] if has_ts else ev.index
            if not ev.empty and "rot_magnitude_dps" in df.columns:
                fig.add_trace(go.Scatter(
                    x=ev_x, y=ev["rot_magnitude_dps"],
                    mode="markers", marker=dict(color=color, size=5),
                    name=label,
                ), row=1, col=1)

    if "roll_deg" in df.columns:
        fig.add_trace(go.Scatter(x=x_axis, y=df["roll_deg"],
                                 line=dict(color="#98df8a", width=0.8), name="Roll"), row=2, col=1)
        if "is_lean" in df.columns:
            leans = df[df["is_lean"]]
            lx = leans[ts_col] if has_ts else leans.index
            fig.add_trace(go.Scatter(
                x=lx, y=leans["roll_deg"],
                mode="markers", marker=dict(color="#ff4136", size=5), name="Lean",
            ), row=2, col=1)

    fig.update_layout(template="plotly_dark", title="Special Movement Detection",
                      height=500, showlegend=True)
    return fig


# ---------------------------------------------------------------------------
# Action events (shots, passes, headers)
# ---------------------------------------------------------------------------
def plot_action_events(df: pd.DataFrame) -> go.Figure:
    """Acceleration trace with shot / pass / header event markers."""
    ts_col = "ts_cet" if "ts_cet" in df.columns else "ts_utc"
    has_ts = ts_col in df.columns
    x_axis = df[ts_col] if has_ts else df.index

    fig = go.Figure()
    if "acc_magnitude_g" in df.columns:
        fig.add_trace(go.Scatter(
            x=x_axis, y=df["acc_magnitude_g"],
            line=dict(color="#00e5ff", width=0.8), name="|Acc| (g)",
        ))

    markers = [
        ("event_shot", "#ff4136", "Shot ⚽", "star"),
        ("event_pass", "#ff7f0e", "Pass", "circle"),
        ("event_header", "#98df8a", "Header", "diamond"),
    ]
    for col, color, label, symbol in markers:
        if col in df.columns:
            ev = df[df[col]]
            ev_x = ev[ts_col] if has_ts else ev.index
            if not ev.empty and "acc_magnitude_g" in ev.columns:
                fig.add_trace(go.Scatter(
                    x=ev_x, y=ev["acc_magnitude_g"],
                    mode="markers",
                    marker=dict(color=color, size=8, symbol=symbol),
                    name=label,
                ))

    fig.update_layout(
        title="Detected Action Events (Shots / Passes / Headers)",
        xaxis_title="Time", yaxis_title="Acceleration (g)",
        template="plotly_dark",
    )
    return fig


# ---------------------------------------------------------------------------
# Fatigue
# ---------------------------------------------------------------------------
def plot_fatigue(fatigue_df: pd.DataFrame) -> go.Figure:
    """Peak speed per time window to visualise fatigue."""
    if fatigue_df.empty:
        return go.Figure()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=fatigue_df["window_start"].astype(str),
        y=fatigue_df["peak_speed_kmh"],
        name="Peak speed (km/h)",
        marker_color="#ff7f0e",
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=fatigue_df["window_start"].astype(str),
        y=fatigue_df["mean_speed_kmh"],
        name="Mean speed (km/h)",
        line=dict(color="#00e5ff"),
    ), secondary_y=True)
    fig.update_layout(
        title="Speed Over Time (Fatigue Proxy)",
        xaxis_title="Window start",
        template="plotly_dark",
    )
    fig.update_yaxes(title_text="Peak speed (km/h)", secondary_y=False)
    fig.update_yaxes(title_text="Mean speed (km/h)", secondary_y=True)
    return fig
