import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="Player Dashboard", layout="wide")

# ── styles ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .metric-card {
    background: #f7f7f5;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 4px;
  }
  .metric-label { font-size: 12px; color: #888; margin: 0; }
  .metric-value { font-size: 26px; font-weight: 600; margin: 2px 0 0; color: #111; }
  .metric-sub   { font-size: 12px; margin: 3px 0 0; }
  .s1 { color: #185FA5; }
  .s2 { color: #D85A30; }
  .section-title { font-size: 14px; font-weight: 600; color: #444; margin: 0 0 10px; letter-spacing: .02em; text-transform: uppercase; }
</style>
""", unsafe_allow_html=True)

# ── load & process data ───────────────────────────────────────────────────────
DATA_PATH = Path(__file__).parent / "new_player_data_2026_04_20_152626.csv"

@st.cache_data
def load_sessions(path: Path):
    df = pd.read_csv(path)
    df = df[(df["latitude"] != 0) & (df["longitude"] != 0)].copy()
    df["time"] = pd.to_datetime(df["epoch_time"], format="%d.%m.%Y %H:%M:%S.%f")

    ZONE_DEFS = [
        ("Standing",  0.0, 0.5,  "#B4B2A9"),
        ("Walking",   0.5, 2.0,  "#85B7EB"),
        ("Jogging",   2.0, 3.0,  "#1D9E75"),
        ("Running",   3.0, 5.0,  "#EF9F27"),
        ("Sprinting", 5.0, 99.0, "#D85A30"),
    ]

    sessions = {}
    for sess in [1, 2]:
        g = df[df["session"] == sess].sort_values("time").reset_index(drop=True)
        g["elapsed_min"] = (g["time"] - g["time"].iloc[0]).dt.total_seconds() / 60
        spd = g["speed"]
        duration = g["elapsed_min"].max()
        distance_km = round((spd * 0.5).sum() / 1000, 2)

        zones = []
        for name, lo, hi, color in ZONE_DEFS:
            mask = (spd >= lo) & (spd < hi)
            zones.append({
                "name": name,
                "minutes": round(mask.sum() * 0.5 / 60, 1),
                "pct": round(mask.mean() * 100, 1),
                "color": color,
            })

        # sprint count
        is_sprint = (spd > 5.0).astype(int).values
        sprints, in_s = 0, False
        for v in is_sprint:
            if v and not in_s: sprints += 1; in_s = True
            elif not v: in_s = False

        # 2-min timeline buckets
        g["bucket"] = (g["elapsed_min"] / 2).astype(int)
        tl = (g.groupby("bucket")["speed"]
                .mean()
                .reset_index()
                .rename(columns={"bucket": "min2", "speed": "avg_speed"}))
        tl["minute"] = tl["min2"] * 2

        sessions[sess] = {
            "date":        g["time"].iloc[0].strftime("%d.%m.%Y"),
            "start":       g["time"].iloc[0].strftime("%H:%M"),
            "end":         g["time"].iloc[-1].strftime("%H:%M"),
            "duration":    round(duration, 1),
            "distance_km": distance_km,
            "dist_per_min": round(distance_km / duration * 1000, 1),   # m/min
            "max_speed":   round(float(spd.max()), 2),
            "mean_speed":  round(float(spd.mean()), 2),
            "steps":       int(g["step"].max() - g["step"].min()),
            "sprints":     sprints,
            "spr_per_min": round(sprints / duration, 2),
            "zones":       zones,
            "timeline":    tl,
        }
    return sessions

sessions = load_sessions(DATA_PATH)
s1, s2 = sessions[1], sessions[2]

BLUE   = "#185FA5"
CORAL  = "#D85A30"

# ── header ────────────────────────────────────────────────────────────────────
st.markdown("## Engin Kaya · Field A · Session comparison")
st.markdown(
    f"<span class='s1'>● S1 — {s1['date']} &nbsp; {s1['start']}–{s1['end']} &nbsp; ({s1['duration']} min)</span>"
    f"&nbsp;&nbsp;&nbsp;"
    f"<span class='s2'>● S2 — {s2['date']} &nbsp; {s2['start']}–{s2['end']} &nbsp; ({s2['duration']} min)</span>",
    unsafe_allow_html=True,
)
st.divider()

# ── section 1: summary cards ──────────────────────────────────────────────────
def card(label, v1, v2, unit="", note=""):
    delta_raw = v1 - v2 if isinstance(v1, (int, float)) else None
    if delta_raw is not None:
        sign = "+" if delta_raw > 0 else ""
        delta_str = f"{sign}{delta_raw:.1f} {unit}"
        delta_color = "#185FA5" if delta_raw > 0 else "#D85A30"
    else:
        delta_str, delta_color = "", "#888"

    st.markdown(f"""
    <div class="metric-card">
      <p class="metric-label">{label} {f'<span style="color:#aaa">({note})</span>' if note else ''}</p>
      <p class="metric-value"><span class="s1">{v1}{unit}</span> <span style="color:#ccc;font-size:18px">vs</span> <span class="s2">{v2}{unit}</span></p>
      {'<p class="metric-sub" style="color:'+delta_color+'">S1 '+delta_str+' vs S2</p>' if delta_str else ''}
    </div>
    """, unsafe_allow_html=True)

st.markdown('<p class="section-title">Key metrics</p>', unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)
with c1: card("Duration", s1["duration"], s2["duration"], " min")
with c2: card("Distance", s1["distance_km"], s2["distance_km"], " km", "total")
with c3: card("Distance / min", s1["dist_per_min"], s2["dist_per_min"], " m/min", "normalized")
with c4: card("Max speed", s1["max_speed"], s2["max_speed"], " m/s")

c5, c6, c7, c8 = st.columns(4)
with c5: card("Sprints (>5 m/s)", s1["sprints"], s2["sprints"])
with c6: card("Sprints / min", s1["spr_per_min"], s2["spr_per_min"], "", "normalized")
with c7: card("Avg speed", s1["mean_speed"], s2["mean_speed"], " m/s")
with c8: card("Steps", s1["steps"], s2["steps"])

st.divider()

# ── section 2: speed zones ────────────────────────────────────────────────────
st.markdown('<p class="section-title">Speed zones (% of session time)</p>', unsafe_allow_html=True)

fig_zones = go.Figure()

for sess_id, s, label in [(1, s1, f"S1 · {s1['date']}"), (2, s2, f"S2 · {s2['date']}")]:
    for z in s["zones"]:
        fig_zones.add_trace(go.Bar(
            name=z["name"],
            y=[label],
            x=[z["pct"]],
            orientation="h",
            marker_color=z["color"],
            text=f"{z['pct']}%",
            textposition="inside",
            insidetextanchor="middle",
            textfont=dict(size=11, color="white"),
            hovertemplate=f"<b>{z['name']}</b><br>{z['pct']}% · {z['minutes']} min<extra></extra>",
            showlegend=(sess_id == 1),
        ))

fig_zones.update_layout(
    barmode="stack",
    height=160,
    margin=dict(l=0, r=0, t=0, b=0),
    xaxis=dict(range=[0, 100], showticklabels=False, showgrid=False),
    yaxis=dict(showgrid=False),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    legend=dict(orientation="h", yanchor="bottom", y=1.1, xanchor="left", x=0, font=dict(size=12)),
    font=dict(family="sans-serif"),
)
st.plotly_chart(fig_zones, use_container_width=True, config={"displayModeBar": False})

st.divider()

# ── section 3: activity timeline ──────────────────────────────────────────────
st.markdown('<p class="section-title">Activity over time — avg speed per 2-min window</p>', unsafe_allow_html=True)

col_check1, col_check2, _ = st.columns([1, 1, 6])
show_s1 = col_check1.checkbox(f"Show S1 ({s1['date']})", value=True)
show_s2 = col_check2.checkbox(f"Show S2 ({s2['date']})", value=True)

fig_tl = go.Figure()

if show_s1:
    t1 = s1["timeline"]
    fig_tl.add_trace(go.Scatter(
        x=t1["minute"], y=t1["avg_speed"].round(2),
        mode="lines",
        name=f"S1 · {s1['date']}",
        line=dict(color=BLUE, width=2),
        fill="tozeroy",
        fillcolor="rgba(24,95,165,0.08)",
        hovertemplate="min %{x} · %{y:.2f} m/s<extra>S1</extra>",
    ))

if show_s2:
    t2 = s2["timeline"]
    fig_tl.add_trace(go.Scatter(
        x=t2["minute"], y=t2["avg_speed"].round(2),
        mode="lines",
        name=f"S2 · {s2['date']}",
        line=dict(color=CORAL, width=2, dash="dot"),
        fill="tozeroy",
        fillcolor="rgba(216,90,48,0.07)",
        hovertemplate="min %{x} · %{y:.2f} m/s<extra>S2</extra>",
    ))

# zone reference lines
for speed, label in [(0.5, "walking"), (2.0, "jogging"), (3.0, "running"), (5.0, "sprint")]:
    fig_tl.add_hline(
        y=speed, line_dash="dash", line_color="#ddd", line_width=1,
        annotation_text=label, annotation_position="right",
        annotation_font_size=10, annotation_font_color="#aaa",
    )

fig_tl.update_layout(
    height=340,
    margin=dict(l=0, r=60, t=10, b=40),
    xaxis=dict(title="Elapsed minutes", showgrid=False, zeroline=False),
    yaxis=dict(title="Avg speed (m/s)", showgrid=True, gridcolor="#f0f0f0", zeroline=False, range=[0, 4.5]),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=12)),
    hovermode="x unified",
    font=dict(family="sans-serif"),
)
st.plotly_chart(fig_tl, use_container_width=True, config={"displayModeBar": False})

st.divider()
st.caption("Data: GPS tracker (2Hz) · speed from GPS · distance = speed × 0.5s · field A, Vienna")
