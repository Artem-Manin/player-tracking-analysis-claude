import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

# python -m streamlit run heatmap.py

st.set_page_config(page_title="Field Coverage Heatmap", layout="wide")

st.markdown("""
<style>
  .section {
    font-size: 11px; font-weight: 600; color: #999;
    text-transform: uppercase; letter-spacing: .06em;
    margin: 1.4rem 0 0.5rem;
  }
  .insight {
    background: #f0f5ff; border-left: 3px solid #185FA5;
    border-radius: 6px; padding: 10px 14px;
    font-size: 13px; color: #1a1a2e; margin: 8px 0 4px;
  }
  .stat-box {
    background: #fafaf8; border: 1px solid #e5e5e0; border-radius: 8px;
    padding: 14px 18px; text-align: center;
  }
  .stat-value {
    font-size: 26px; font-weight: 700; color: #1a1a2e; line-height: 1.1;
  }
  .stat-label {
    font-size: 11px; font-weight: 600; color: #999;
    text-transform: uppercase; letter-spacing: .05em; margin-top: 4px;
  }
</style>
""", unsafe_allow_html=True)

# ── constants ─────────────────────────────────────────────────────────────────
FILE1 = Path(__file__).parent / "new_player_data_2026_04_20_152626.csv"
FILE2 = Path(__file__).parent / "player-activity-77686439338177232025102910010005-2026-04-20T22-26-36-698Z.csv"

BLUE, CORAL, GREEN = "#185FA5", "#D85A30", "#2D8659"
COLORS = {1: BLUE, 2: CORAL, 3: GREEN}

ZONE_DEFS = [
    ("Standing",  0.0, 0.5),
    ("Walking",   0.5, 2.0),
    ("Jogging",   2.0, 3.0),
    ("Running",   3.0, 5.0),
    ("Sprinting", 5.0, 99.0),
]

# ── field orientation ─────────────────────────────────────────────────────────
# PITCH_ANGLE_DEG: clockwise degrees from geographic north that the
# "top goal" (opponent end) faces.
# Fortuna-Platz (S1, S2): pitch long axis NW-SE, NW end = top → 315°
# S3 field: similar, slightly steeper tilt → 315° (adjust if needed after viewing)
PITCH_ANGLE_DEG = {1: 315, 2: 315, 3: 315}

# Pitch dimensions in metres (goal-to-goal × touchline-to-touchline)
PITCH_LEN = {1: 100, 2: 100, 3: 95}   # adjust to actual field size
PITCH_WID = {1: 65,  2: 65,  3: 63}

# ── helpers ───────────────────────────────────────────────────────────────────
def _section(text):
    st.markdown(f'<p class="section">{text}</p>', unsafe_allow_html=True)

def _insight(text):
    st.markdown(f'<div class="insight">💡 {text}</div>', unsafe_allow_html=True)

def _stat(value, label):
    return (f'<div class="stat-box">'
            f'<div class="stat-value">{value}</div>'
            f'<div class="stat-label">{label}</div>'
            f'</div>')

def _layout(height=400):
    return dict(
        height=height, margin=dict(l=10, r=10, t=40, b=36),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#1a472a",
        font=dict(family="sans-serif", size=12),
    )

# ── data loading (same pattern as overview.py) ────────────────────────────────
@st.cache_data
def load_sessions():
    df1 = pd.read_csv(FILE1)
    df2 = pd.read_csv(FILE2)
    df = pd.concat([df1, df2], ignore_index=True)

    player_name = df["playerName"].iloc[0] if "playerName" in df.columns else None
    player_name = player_name if pd.notna(player_name) and player_name else None

    df = df[(df["latitude"] != 0) & (df["longitude"] != 0)].copy()

    def parse_time(val):
        try:
            return pd.to_datetime(val, format="%d.%m.%Y %H:%M:%S.%f")
        except Exception:
            return pd.to_datetime(int(val), unit="ms")

    df["time"] = df["epoch_time"].apply(parse_time)

    out = {}
    for sess in sorted(df["session"].unique()):
        g = df[df["session"] == sess].sort_values("time").reset_index(drop=True)
        high_speed = g[g["speed"] >= 1.5]
        if len(high_speed) > 0:
            g = g.loc[high_speed.index[0]:high_speed.index[-1]].reset_index(drop=True)
        g["elapsed_min"] = (g["time"] - g["time"].iloc[0]).dt.total_seconds() / 60
        out[int(sess)] = g

    return out, player_name


# ── coordinate helpers ────────────────────────────────────────────────────────
def _to_metres(lats, lons, lat_c, lon_c):
    m_per_lat = 111_320
    m_per_lon = 111_320 * np.cos(np.radians(lat_c))
    return (lons - lon_c) * m_per_lon, (lats - lat_c) * m_per_lat


def _rotate(x, y, angle_deg):
    """Rotate coords so that `angle_deg` CW-from-north direction becomes +Y (up)."""
    theta = np.radians(-angle_deg)
    return (x * np.cos(theta) - y * np.sin(theta),
            x * np.sin(theta) + y * np.cos(theta))


def _rot_pts(pts, angle_deg):
    theta = np.radians(-angle_deg)
    c, s  = np.cos(theta), np.sin(theta)
    return [(x * c - y * s, x * s + y * c) for x, y in pts]


# ── pitch drawing ─────────────────────────────────────────────────────────────
def _pitch_shapes(half_L, half_W, xref="x", yref="y",
                  line_color="rgba(255,255,255,0.6)"):
    """
    Full standard pitch markings in a coordinate system where:
      +Y = toward opponent goal (top of plot)
      +X = right touchline
    half_L = half pitch length, half_W = half pitch width.
    No rotation applied here — data is already rotated.
    xref/yref must match the subplot axis (e.g. "x", "x2", "y", "y2").
    """
    lw      = dict(color=line_color, width=1.5)
    goal_lw = dict(color="rgba(255,255,255,0.9)", width=2.0)
    shapes  = []

    def seg(x0, y0, x1, y1, style=None):
        shapes.append(dict(
            type="line",
            x0=x0, y0=y0, x1=x1, y1=y1,
            xref=xref, yref=yref,
            line=style or lw,
        ))

    def arc_pts(cx, cy, r, t0, t1, n=48):
        ts = np.linspace(t0, t1, n)
        return [(cx + r * np.cos(t), cy + r * np.sin(t)) for t in ts]

    def add_arc(cx, cy, r, t0, t1, style=None):
        pts = arc_pts(cx, cy, r, t0, t1)
        for i in range(len(pts) - 1):
            shapes.append(dict(
                type="line",
                x0=pts[i][0], y0=pts[i][1],
                x1=pts[i+1][0], y1=pts[i+1][1],
                xref=xref, yref=yref,
                line=style or lw,
            ))

    L, W = half_L, half_W

    # Outer boundary
    seg(-W, -L,  W, -L)
    seg( W, -L,  W,  L)
    seg( W,  L, -W,  L)
    seg(-W,  L, -W, -L)

    # Halfway line
    seg(-W, 0, W, 0)

    # Centre circle + spot
    add_arc(0, 0, 9.15, 0, 2 * np.pi)
    add_arc(0, 0, 0.4,  0, 2 * np.pi)

    # ── Penalty areas ─────────────────────────────────────────────────────────
    pa_w, pa_d = 20.16, 16.5   # half-width, depth
    # Top (opponent)
    seg(-pa_w, L,        pa_w, L)
    seg(-pa_w, L,       -pa_w, L - pa_d)
    seg(-pa_w, L - pa_d, pa_w, L - pa_d)
    seg( pa_w, L - pa_d, pa_w, L)
    # Bottom (ours)
    seg(-pa_w, -L,        pa_w, -L)
    seg(-pa_w, -L,       -pa_w, -L + pa_d)
    seg(-pa_w, -L + pa_d, pa_w, -L + pa_d)
    seg( pa_w, -L + pa_d, pa_w, -L)

    # ── Goal areas (6-yard box) ───────────────────────────────────────────────
    ga_w, ga_d = 9.16, 5.5
    seg(-ga_w, L,        ga_w, L)
    seg(-ga_w, L,       -ga_w, L - ga_d)
    seg(-ga_w, L - ga_d, ga_w, L - ga_d)
    seg( ga_w, L - ga_d, ga_w, L)
    seg(-ga_w, -L,        ga_w, -L)
    seg(-ga_w, -L,       -ga_w, -L + ga_d)
    seg(-ga_w, -L + ga_d, ga_w, -L + ga_d)
    seg( ga_w, -L + ga_d, ga_w, -L)

    # ── Penalty spots ─────────────────────────────────────────────────────────
    add_arc(0,  L - 11, 0.4, 0, 2 * np.pi)
    add_arc(0, -L + 11, 0.4, 0, 2 * np.pi)

    # ── Penalty arcs (outside the box) ────────────────────────────────────────
    # Top arc: centre at penalty spot (0, L-11), arc from ~53° to ~127° (above box edge)
    add_arc(0,  L - 11, 9.15, np.radians(53),  np.radians(127))
    # Bottom arc: centre at (0, -(L-11)), arc from ~233° to ~307°
    add_arc(0, -(L - 11), 9.15, np.radians(233), np.radians(307))

    # ── Goals (2.44m deep, 7.32m wide) ────────────────────────────────────────
    gw, gd = 3.66, 2.44
    # Top goal
    seg(-gw,  L,      gw,  L,      goal_lw)
    seg(-gw,  L,     -gw,  L + gd, goal_lw)
    seg( gw,  L,      gw,  L + gd, goal_lw)
    seg(-gw,  L + gd, gw,  L + gd, goal_lw)
    # Bottom goal
    seg(-gw, -L,      gw, -L,      goal_lw)
    seg(-gw, -L,     -gw, -L - gd, goal_lw)
    seg( gw, -L,      gw, -L - gd, goal_lw)
    seg(-gw, -L - gd, gw, -L - gd, goal_lw)

    return shapes


def _pitch_annotations(half_L, half_W, ax_ref, ay_ref,
                        angle_for_compass=None, compass_r=None):
    """Goal labels + optional compass rose in rotated coordinates."""
    anns = [
        dict(x=0, y= half_L + 9,
             text="<b>⬆ Opponent</b>",
             showarrow=False, font=dict(size=11, color="rgba(255,255,255,0.75)"),
             xref=ax_ref, yref=ay_ref, xanchor="center", yanchor="bottom"),
        dict(x=0, y=-half_L - 9,
             text="<b>Our goal ⬇</b>",
             showarrow=False, font=dict(size=11, color="rgba(255,255,255,0.75)"),
             xref=ax_ref, yref=ay_ref, xanchor="center", yanchor="top"),
    ]
    if angle_for_compass is not None and compass_r is not None:
        for label, geo_x, geo_y in [("N", 0, 1), ("S", 0, -1),
                                     ("E", 1, 0), ("W", -1, 0)]:
            rx, ry = _rotate(np.array([geo_x * compass_r]),
                             np.array([geo_y * compass_r]), angle_for_compass)
            anns.append(dict(
                x=float(rx[0]), y=float(ry[0]),
                text=f"<b>{label}</b>",
                showarrow=False,
                font=dict(size=10, color="rgba(220,220,200,0.55)"),
                xref=ax_ref, yref=ay_ref,
                xanchor="center", yanchor="middle",
            ))
    return anns


# ── coverage stats helpers ────────────────────────────────────────────────────
def _coverage_pct(rx, ry, half_L, half_W, n=40, threshold=5):
    xe = np.linspace(-half_W, half_W, n + 1)
    ye = np.linspace(-half_L, half_L, n + 1)
    z, _, _ = np.histogram2d(ry, rx, bins=[ye, xe])
    return round((z > threshold).sum() / (n * n) * 100, 1)


def _third_time_pct_rotated(ry, half_L):
    edges = np.linspace(-half_L, half_L, 4)
    return [round(((ry >= edges[i]) & (ry < edges[i + 1])).mean() * 100, 1)
            for i in range(3)]


# ── heatmap figure ────────────────────────────────────────────────────────────
def _heatmap_fig(sessions, selected_sids):
    n_sess = len(selected_sids)
    if n_sess == 0:
        return go.Figure()

    fig = make_subplots(
        rows=1, cols=n_sess,
        subplot_titles=[f"Session {s}" for s in selected_sids],
        horizontal_spacing=0.06,
    )

    all_shapes      = []
    all_annotations = list(fig.layout.annotations)
    GRID            = 50

    for col_idx, sid in enumerate(selected_sids):
        g      = sessions[sid]
        lats   = g["latitude"].values
        lons   = g["longitude"].values
        lat_c  = lats.mean()
        lon_c  = lons.mean()
        angle  = PITCH_ANGLE_DEG.get(sid, 0)
        half_L = PITCH_LEN.get(sid, 100) / 2
        half_W = PITCH_WID.get(sid, 65)  / 2
        pad    = 12

        x_m, y_m = _to_metres(lats, lons, lat_c, lon_c)
        rx, ry   = _rotate(x_m, y_m, angle)

        xe = np.linspace(-half_W - pad, half_W + pad, GRID + 1)
        ye = np.linspace(-half_L - pad, half_L + pad, GRID + 1)
        z, _, _ = np.histogram2d(ry, rx, bins=[ye, xe])
        if z.max() > 0:
            z = z / z.max() * 100
        xc = (xe[:-1] + xe[1:]) / 2
        yc = (ye[:-1] + ye[1:]) / 2

        ax_ref = "x"  if col_idx == 0 else f"x{col_idx + 1}"
        ay_ref = "y"  if col_idx == 0 else f"y{col_idx + 1}"

        fig.add_trace(go.Heatmap(
            x=xc, y=yc, z=z,
            colorscale=[
                [0.00, "rgba(0,0,0,0)"],
                [0.01, "rgba(24,95,165,0.10)"],
                [0.25, "rgba(24,95,165,0.45)"],
                [0.55, "rgba(29,158,117,0.72)"],
                [0.80, "rgba(239,159,39,0.90)"],
                [1.00, "rgba(216,90,48,1.00)"],
            ],
            zmin=0, zmax=100,
            showscale=(col_idx == n_sess - 1),
            colorbar=dict(title="Density", ticksuffix="%",
                          len=0.6, thickness=12, tickfont=dict(size=10),
                          bgcolor="rgba(0,0,0,0)", borderwidth=0,
                          tickfont_color="white", title_font_color="white"),
            hovertemplate=(
                "Along pitch: %{y:.1f}m  Across: %{x:.1f}m"
                "<br>Density: %{z:.0f}%<extra>S" + str(sid) + "</extra>"
            ),
            name=f"S{sid}",
        ), row=1, col=col_idx + 1)

        all_shapes.extend(
            _pitch_shapes(half_L, half_W, xref=ax_ref, yref=ay_ref)
        )
        all_annotations += _pitch_annotations(
            half_L, half_W, ax_ref, ay_ref,
            angle_for_compass=angle, compass_r=half_W + pad + 5,
        )

        ax_key = "xaxis"  if col_idx == 0 else f"xaxis{col_idx + 1}"
        ay_key = "yaxis"  if col_idx == 0 else f"yaxis{col_idx + 1}"
        fig.update_layout(**{
            ax_key: dict(showgrid=False, zeroline=False,
                         showticklabels=False, title=None,
                         range=[-half_W - pad - 12, half_W + pad + 12]),
            ay_key: dict(showgrid=False, zeroline=False,
                         showticklabels=False, title=None,
                         scaleanchor=ax_ref, scaleratio=1,
                         range=[-half_L - pad - 16, half_L + pad + 16]),
        })

    fig.update_layout(
        **_layout(600),
        shapes=all_shapes,
        annotations=all_annotations,
    )
    return fig


# ── sprint pitch figure ───────────────────────────────────────────────────────
def _sprint_pitch_fig(sessions, selected_sids):
    n_sess = len(selected_sids)
    if n_sess == 0:
        return go.Figure()

    fig = make_subplots(
        rows=1, cols=n_sess,
        subplot_titles=[f"Session {s} — Sprints" for s in selected_sids],
        horizontal_spacing=0.06,
    )
    all_shapes      = []
    all_annotations = list(fig.layout.annotations)

    for col_idx, sid in enumerate(selected_sids):
        g      = sessions[sid]
        lats   = g["latitude"].values
        lons   = g["longitude"].values
        lat_c  = lats.mean()
        lon_c  = lons.mean()
        angle  = PITCH_ANGLE_DEG.get(sid, 0)
        half_L = PITCH_LEN.get(sid, 100) / 2
        half_W = PITCH_WID.get(sid, 65)  / 2
        pad    = 10

        ax_ref = "x"  if col_idx == 0 else f"x{col_idx + 1}"
        ay_ref = "y"  if col_idx == 0 else f"y{col_idx + 1}"

        sprints = g[g["speed"] >= 5.0]
        if len(sprints) > 0:
            sx, sy = _to_metres(sprints["latitude"].values,
                                sprints["longitude"].values, lat_c, lon_c)
            srx, sry = _rotate(sx, sy, angle)
            fig.add_trace(go.Scatter(
                x=srx, y=sry, mode="markers",
                marker=dict(color=COLORS.get(sid, "#888"), size=7, opacity=0.85,
                            line=dict(color="white", width=0.5)),
                name=f"S{sid}",
                hovertemplate=(
                    f"S{sid} sprint<br>"
                    "Along: %{y:.1f}m  Across: %{x:.1f}m<extra></extra>"
                ),
            ), row=1, col=col_idx + 1)
        else:
            fig.add_trace(go.Scatter(x=[], y=[], name=f"S{sid} (no sprints)"),
                          row=1, col=col_idx + 1)

        all_shapes.extend(
            _pitch_shapes(half_L, half_W, xref=ax_ref, yref=ay_ref)
        )
        all_annotations += _pitch_annotations(half_L, half_W, ax_ref, ay_ref)

        ax_key = "xaxis"  if col_idx == 0 else f"xaxis{col_idx + 1}"
        ay_key = "yaxis"  if col_idx == 0 else f"yaxis{col_idx + 1}"
        fig.update_layout(**{
            ax_key: dict(showgrid=False, zeroline=False,
                         showticklabels=False, title=None,
                         range=[-half_W - pad, half_W + pad]),
            ay_key: dict(showgrid=False, zeroline=False,
                         showticklabels=False, title=None,
                         scaleanchor=ax_ref, scaleratio=1,
                         range=[-half_L - pad - 8, half_L + pad + 8]),
        })

    fig.update_layout(
        **_layout(520),
        shapes=all_shapes,
        annotations=all_annotations,
    )
    return fig


# ── zone heatmap per session ──────────────────────────────────────────────────
def _zone_heatmap_fig(g, zone_name, lo, hi, angle, half_L, half_W):
    lats  = g["latitude"].values
    lons  = g["longitude"].values
    lat_c = lats.mean()
    lon_c = lons.mean()
    sub   = g[(g["speed"] >= lo) & (g["speed"] < hi)]
    pad   = 10
    GRID  = 40
    xe    = np.linspace(-half_W - pad, half_W + pad, GRID + 1)
    ye    = np.linspace(-half_L - pad, half_L + pad, GRID + 1)

    if len(sub) < 5:
        return None

    sx, sy   = _to_metres(sub["latitude"].values, sub["longitude"].values, lat_c, lon_c)
    rx, ry   = _rotate(sx, sy, angle)
    z, _, _  = np.histogram2d(ry, rx, bins=[ye, xe])
    if z.max() > 0:
        z = z / z.max() * 100

    ZONE_CS = {
        "Standing":  [[0, "rgba(0,0,0,0)"], [1, "rgba(180,178,169,0.85)"]],
        "Walking":   [[0, "rgba(0,0,0,0)"], [1, "rgba(133,183,235,0.85)"]],
        "Jogging":   [[0, "rgba(0,0,0,0)"], [1, "rgba(29,158,117,0.90)"]],
        "Running":   [[0, "rgba(0,0,0,0)"], [1, "rgba(239,159,39,0.95)"]],
        "Sprinting": [[0, "rgba(0,0,0,0)"], [1, "rgba(216,90,48,1.00)"]],
    }

    xc = (xe[:-1] + xe[1:]) / 2
    yc = (ye[:-1] + ye[1:]) / 2

    shapes = _pitch_shapes(half_L, half_W,
                           line_color="rgba(255,255,255,0.35)")
    clean  = shapes  # xref/yref already set correctly by default

    fig = go.Figure(go.Heatmap(
        x=xc, y=yc, z=z,
        colorscale=ZONE_CS.get(zone_name, [[0, "white"], [1, "blue"]]),
        zmin=0, zmax=100, showscale=False,
        hovertemplate=f"{zone_name}: %{{z:.0f}}%<extra></extra>",
    ))
    fig.update_layout(
        height=260,
        margin=dict(l=4, r=4, t=28, b=4),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#1a472a",
        title_text=zone_name,
        title_font=dict(size=12, color="#ccc"),
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False,
                   range=[-half_W - pad, half_W + pad]),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False,
                   scaleanchor="x", scaleratio=1,
                   range=[-half_L - pad - 8, half_L + pad + 8]),
        shapes=clean,
    )
    return fig


# ── load ──────────────────────────────────────────────────────────────────────
with st.spinner("Loading sessions…"):
    SESS, player_name = load_sessions()

# ── header ────────────────────────────────────────────────────────────────────
header_text = f"{player_name} · Field Coverage" if player_name else "Field Coverage"
st.markdown(f"## {header_text}")
st.caption(
    f"{len(SESS)} sessions loaded · GPS only · "
    "Pitch rotated to real field orientation · Top = opponent goal · Bottom = our goal"
)

# ── session selector ──────────────────────────────────────────────────────────
_section("Session selection")
all_sids = sorted(SESS.keys())
selected = st.multiselect(
    "Show sessions",
    options=all_sids,
    default=all_sids,
    format_func=lambda s: f"Session {s}  ({SESS[s]['time'].iloc[0].strftime('%d.%m.%Y')})",
)

if not selected:
    st.warning("Select at least one session to display.")
    st.stop()

# ── coverage stats ────────────────────────────────────────────────────────────
_section("Coverage stats per session")
stat_cols = st.columns(len(selected))
for i, sid in enumerate(selected):
    g      = SESS[sid]
    angle  = PITCH_ANGLE_DEG.get(sid, 0)
    half_L = PITCH_LEN.get(sid, 100) / 2
    half_W = PITCH_WID.get(sid, 65)  / 2
    lats, lons = g["latitude"].values, g["longitude"].values
    x_m, y_m  = _to_metres(lats, lons, lats.mean(), lons.mean())
    rx, ry     = _rotate(x_m, y_m, angle)
    cov        = _coverage_pct(rx, ry, half_L, half_W)
    thirds     = _third_time_pct_rotated(ry, half_L)
    with stat_cols[i]:
        st.markdown(
            _stat(f"{cov}%", f"S{sid} · Field coverage") +
            '<div style="margin-top:10px;">' +
            _stat(f"{thirds[0]}% / {thirds[1]}% / {thirds[2]}%",
                  "Thirds: our goal / mid / opponent") +
            '</div>',
            unsafe_allow_html=True,
        )

# ── main heatmap ──────────────────────────────────────────────────────────────
_section("Density heatmap — pitch view")
st.caption(
    "GPS points projected onto real pitch orientation. "
    "Top = opponent goal · Bottom = our goal · "
    "Faint letters = geographic compass direction."
)
fig_heat = _heatmap_fig(SESS, selected)
st.plotly_chart(fig_heat, use_container_width=True, config={"displayModeBar": False})

for sid in selected:
    g      = SESS[sid]
    angle  = PITCH_ANGLE_DEG.get(sid, 0)
    half_L = PITCH_LEN.get(sid, 100) / 2
    lats, lons = g["latitude"].values, g["longitude"].values
    x_m, y_m  = _to_metres(lats, lons, lats.mean(), lons.mean())
    _, ry      = _rotate(x_m, y_m, angle)
    thirds     = _third_time_pct_rotated(ry, half_L)
    dominant   = ["defensive", "midfield", "attacking"][int(np.argmax(thirds))]
    _insight(
        f"S{sid}: most time in the <strong>{dominant} third</strong> "
        f"({max(thirds):.0f}% of GPS points)."
    )

# ── sprint pitch ──────────────────────────────────────────────────────────────
_section("Sprint locations — pitch view")
st.caption("Each dot = one GPS point at sprint speed (≥5 m/s). No interpolation.")
total_sprints = {sid: (SESS[sid]["speed"] >= 5.0).sum() for sid in selected}
if all(v == 0 for v in total_sprints.values()):
    st.info("No sprint-speed GPS points found in selected sessions.")
else:
    fig_spr = _sprint_pitch_fig(SESS, selected)
    st.plotly_chart(fig_spr, use_container_width=True, config={"displayModeBar": False})
    for sid in selected:
        n = total_sprints[sid]
        _insight(f"S{sid}: <strong>{n} GPS points</strong> at sprint speed "
                 f"(each = 0.5 s of sprinting).")

# ── zone breakdown ────────────────────────────────────────────────────────────
_section("Coverage by speed zone — pitch view")
st.caption("Same pitch grid filtered per speed zone. Where does the player walk vs sprint?")

for sid in selected:
    g      = SESS[sid]
    angle  = PITCH_ANGLE_DEG.get(sid, 0)
    half_L = PITCH_LEN.get(sid, 100) / 2
    half_W = PITCH_WID.get(sid, 65)  / 2
    with st.expander(f"Session {sid} — by zone", expanded=(len(selected) == 1)):
        zone_cols = st.columns(len(ZONE_DEFS))
        for col_i, (name, lo, hi) in enumerate(ZONE_DEFS):
            with zone_cols[col_i]:
                fig_z = _zone_heatmap_fig(g, name, lo, hi, angle, half_L, half_W)
                if fig_z:
                    st.plotly_chart(fig_z, use_container_width=True,
                                    config={"displayModeBar": False})
                else:
                    st.caption(f"{name}: not enough data")

st.caption(
    "**GPS accuracy:** ±2–5 m outdoor. Pitch orientation from Google Maps satellite view. "
    "To adjust rotation: change PITCH_ANGLE_DEG at the top of this file "
    "(degrees clockwise from north that the opponent goal faces)."
)