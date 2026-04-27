import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="Intensity & Speed Zones", layout="wide")

# ── colors & constants ────────────────────────────────────────────────────────
BLUE   = "#185FA5"
CORAL  = "#D85A30"
GREEN  = "#2D8659"
COLORS = {1: BLUE, 2: CORAL, 3: GREEN}
LABELS = {1: "S1 · 12.04", 2: "S2 · 19.04", 3: "S3 · 20.04"}
DATES  = {1: "12.04.2026", 2: "19.04.2026", 3: "20.04.2026"}

ZONE_DEFS = [
    ("Standing",  0.0, 0.5,  "#B4B2A9"),
    ("Walking",   0.5, 2.0,  "#85B7EB"),
    ("Jogging",   2.0, 3.0,  "#1D9E75"),
    ("Running",   3.0, 5.0,  "#EF9F27"),
    ("Sprinting", 5.0, 99.0, "#D85A30"),
]
ZONE_NAMES = [name for name, _, _, _ in ZONE_DEFS]

st.markdown("""
<style>
  [data-testid="stMetricValue"] { font-size: 28px !important; font-weight: 600 !important; }
  [data-testid="stMetricLabel"] { font-size: 12px !important; color: #888 !important; }
  .section { font-size: 13px; font-weight: 600; color: #666; text-transform: uppercase;
             letter-spacing: .05em; margin: 1.2rem 0 0.4rem; }
  .insight { background:#f0f5ff; border-left:3px solid #185FA5; border-radius:6px;
             padding:10px 14px; font-size:13px; color:#1a1a2e; margin:6px 0 12px; }
</style>
""", unsafe_allow_html=True)

# ── data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_sessions():
    # Read both CSV files
    file1 = Path(__file__).parent / "new_player_data_2026_04_20_152626.csv"
    file2 = Path(__file__).parent / "player-activity-77686439338177232025102910010005-2026-04-20T22-26-36-698Z.csv"
    
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    df = pd.concat([df1, df2], ignore_index=True)
    
    df = df[(df["latitude"] != 0) & (df["longitude"] != 0)].copy()
    
    # Handle both timestamp formats: formatted strings and Unix milliseconds
    def parse_time(val):
        try:
            return pd.to_datetime(val, format="%d.%m.%Y %H:%M:%S.%f")
        except:
            return pd.to_datetime(int(val), unit='ms')
    
    df["time"] = df["epoch_time"].apply(parse_time)
    
    out = {}
    for sess in sorted(df["session"].unique()):
        g = df[df["session"] == sess].sort_values("time").reset_index(drop=True)
        
        # Speed-based trimming: find first and last point with speed ≥ 1.5 m/s
        high_speed = g[g["speed"] >= 1.5]
        if len(high_speed) > 0:
            trim_start_idx = high_speed.index[0]
            trim_end_idx = high_speed.index[-1]
            g = g.loc[trim_start_idx:trim_end_idx].reset_index(drop=True)
        
        g["elapsed_min"] = (g["time"] - g["time"].iloc[0]).dt.total_seconds() / 60
        g["bucket2"] = (g["elapsed_min"] / 2).astype(int)
        g["bucket5"] = (g["elapsed_min"] / 5).astype(int)
        g["half"] = (g["elapsed_min"] > g["elapsed_min"].max() / 2).map({False: "First half", True: "Second half"})
        out[sess] = g
    return out

SESS = load_sessions()

def _zones(spd):
    """Calculate speed zone distributions"""
    return [{
        "name": name, "color": color,
        "minutes": round((((spd >= lo) & (spd < hi)).sum() * 0.5) / 60, 1),
        "pct": round(((spd >= lo) & (spd < hi)).mean() * 100, 1),
    } for name, lo, hi, color in ZONE_DEFS]

def _layout(height=300):
    return dict(
        height=height, margin=dict(l=10, r=10, t=28, b=36),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="sans-serif", size=12),
    )

def _insight(text, kind="insight"):
    st.markdown(f'<div class="{kind}">💡 {text}</div>', unsafe_allow_html=True)

def _section(text):
    st.markdown(f'<p class="section">{text}</p>', unsafe_allow_html=True)

# precompute per-session zone stats
@st.cache_data
def compute_zone_stats():
    stats = {}
    for sid in sorted(SESS.keys()):
        g = SESS[sid]
        spd = g["speed"]
        dur = g["elapsed_min"].max()
        
        hi_first = g[g["half"] == "First half"]["speed"].apply(lambda x: x >= 3.0).mean() * 100
        hi_last  = g[g["half"] == "Second half"]["speed"].apply(lambda x: x >= 3.0).mean() * 100
        
        hi_pct = round(((spd >= 3.0).mean()) * 100, 1)
        
        stats[sid] = {
            "zones": _zones(spd),
            "hi_pct": hi_pct,
            "hi_first": round(hi_first, 1),
            "hi_last": round(hi_last, 1),
        }
    return stats

STATS = compute_zone_stats()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE — INTENSITY & SPEED ZONES
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("## Intensity & Speed Zones")
st.caption(f"Analyzing {len(SESS)} sessions · Speed zones and high-intensity time")

# zone distribution — % of session time (stacked bars)
_section("Zone distribution — % of session time")
fig = go.Figure()
for sid in sorted(SESS.keys()):
    zones = STATS[sid]["zones"]
    label = LABELS.get(sid, f"S{sid}")
    for z in zones:
        fig.add_trace(go.Bar(
            name=z["name"], y=[label], x=[z["pct"]],
            orientation="h", marker_color=z["color"],
            text=f"{z['pct']}%", textposition="inside",
            insidetextanchor="middle",
            textfont=dict(size=11, color="white"),
            hovertemplate=f"<b>{z['name']}</b><br>{z['pct']}% · {z['minutes']} min<extra></extra>",
            showlegend=bool(sid == min(SESS.keys())),
        ))
fig.update_layout(**_layout(160), barmode="stack",
    xaxis=dict(range=[0,100], showticklabels=False, showgrid=False),
    yaxis=dict(showgrid=False),
    legend=dict(orientation="h", y=1.2, x=0, traceorder="normal"),
)
st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# zone time — absolute minutes (grouped bars)
_section("Zone time — absolute minutes")
fig2 = go.Figure()
for sid in sorted(SESS.keys()):
    zones = STATS[sid]["zones"]
    fig2.add_trace(go.Bar(
        name=LABELS.get(sid, f"S{sid}"),
        x=[z["name"] for z in zones],
        y=[z["minutes"] for z in zones],
        marker_color=COLORS.get(sid, "#888"),
        text=[f"{z['minutes']}m" for z in zones],
        textposition="outside",
        hovertemplate="%{x}: %{y} min<extra>S{sid}</extra>",
    ))
fig2.update_layout(**_layout(280), barmode="group",
    yaxis=dict(title="Minutes", showgrid=True, gridcolor="#f0f0f0"),
    xaxis=dict(showgrid=False, categoryorder="array", categoryarray=ZONE_NAMES),
    legend=dict(orientation="h", y=1.1, x=0),
)
st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

# zone breakdown — donut view
_section("Zone breakdown — donut view")
cols = st.columns(len(SESS))
for col, sid in zip(cols, sorted(SESS.keys())):
    zones = STATS[sid]["zones"]
    fig3 = go.Figure(go.Pie(
        labels=[z["name"] for z in zones],
        values=[z["pct"] for z in zones],
        hole=0.55,
        marker_colors=[z["color"] for z in zones],
        textinfo="label+percent",
        hovertemplate="%{label}: %{value}%<extra></extra>",
        sort=False,
    ))
    fig3.update_layout(**_layout(260),
        annotations=[dict(text=LABELS.get(sid, f"S{sid}"), x=0.5, y=0.5, font_size=11, showarrow=False)],
        showlegend=False,
    )
    col.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})

# hi-intensity % by half
_section("High-intensity % — first half vs second half")
fig4 = go.Figure()
for sid in sorted(SESS.keys()):
    s = STATS[sid]
    fig4.add_trace(go.Bar(
        name=LABELS.get(sid, f"S{sid}"),
        x=["First half", "Second half"],
        y=[s["hi_first"], s["hi_last"]],
        marker_color=COLORS.get(sid, "#888"),
        marker_opacity=[1.0, 0.6],
        text=[f"{s['hi_first']}%", f"{s['hi_last']}%"],
        textposition="outside",
        hovertemplate="%{x}: %{y:.1f}%<extra>S{sid}</extra>",
    ))
fig4.update_layout(**_layout(260), barmode="group",
    yaxis=dict(title="% time at >3 m/s", showgrid=True, gridcolor="#f0f0f0"),
    xaxis=dict(showgrid=False),
    legend=dict(orientation="h", y=1.1, x=0),
)
st.plotly_chart(fig4, use_container_width=True, config={"displayModeBar": False})

# summary insight
st.divider()
_section("Summary")
insights = []
for sid in sorted(SESS.keys()):
    s = STATS[sid]
    insights.append(f"**S{sid}**: {s['hi_pct']}% high-intensity time (running + sprinting)")

_insight(" · ".join(insights))
