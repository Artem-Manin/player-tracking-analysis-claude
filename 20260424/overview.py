import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import requests
#python -m streamlit run overview.py

st.set_page_config(page_title="Session Overview", layout="wide")

st.markdown("""
<style>
  /* clean table styling */
  .stDataFrame { font-size: 14px; }
  [data-testid="stDataFrame"] table { border-collapse: collapse; width: 100%; }
  [data-testid="stDataFrame"] th {
    font-size: 11px !important;
    font-weight: 600 !important;
    color: #888 !important;
    text-transform: uppercase;
    letter-spacing: .05em;
    padding: 8px 12px !important;
    border-bottom: 1px solid #eee !important;
    background: #fafaf8 !important;
  }
  [data-testid="stDataFrame"] td {
    font-size: 14px !important;
    padding: 10px 12px !important;
    border-bottom: 1px solid #f3f3f1 !important;
    vertical-align: middle !important;
  }
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
  .legend {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px;
    background: #f9f9f7; border: 1px solid #e5e5e0; border-radius: 6px; padding: 12px;
    margin: 1.2rem 0; font-size: 13px;
  }
  .legend-item {
    display: flex; align-items: center; gap: 8px;
  }
  .legend-swatch {
    width: 16px; height: 16px; border-radius: 2px; flex-shrink: 0;
  }
  .info-tooltip {
    display: inline-block; width: 18px; height: 18px; line-height: 18px;
    text-align: center; background: #ddd; color: white; border-radius: 50%;
    font-size: 12px; font-weight: 600; cursor: help; margin-left: 4px;
    position: relative;
  }
</style>
""", unsafe_allow_html=True)

# ── constants ─────────────────────────────────────────────────────────────────
DATA_PATH = Path(__file__).parent / "player-activity-77686439338177232025102910010005-2026-04-20T22-26-36-698Z.csv"
BLUE, CORAL, GREEN = "#185FA5", "#D85A30", "#2D8659"
COLORS = {1: BLUE, 2: CORAL, 3: GREEN}

ZONE_DEFS = [
    ("Standing",  0.0, 0.5, "#B4B2A9"),
    ("Walking",   0.5, 2.0, "#85B7EB"),
    ("Jogging",   2.0, 3.0, "#1D9E75"),
    ("Running",   3.0, 5.0, "#EF9F27"),
    ("Sprinting", 5.0, 99.0, "#D85A30"),
]
ZONE_NAMES = [name for name, _, _, _ in ZONE_DEFS]

# ── helpers ───────────────────────────────────────────────────────────────────
def sprint_count(spd):
    count, in_s = 0, False
    for v in (spd > 5.0).values:
        if v and not in_s: count += 1; in_s = True
        elif not v: in_s = False
    return count

def zone_pct(spd, lo, hi):
    return round(((spd >= lo) & (spd < hi)).mean() * 100, 1)

def _zones(spd):
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


@st.cache_data
def compute_zone_stats(SESS):
    stats = {}
    for sid in sorted(SESS.keys()):
        g = SESS[sid]
        spd = g["speed"]
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


def reverse_geocode(lat, lon):
    """Return a short street + district label from OSM Nominatim."""
    try:
        r = requests.get(
            "https://nominatim.openstreetmap.org/reverse",
            params={"lat": lat, "lon": lon, "format": "json", "accept-language": "en"},
            headers={"User-Agent": "coach-dashboard/1.0"},
            timeout=4,
        )
        addr = r.json().get("address", {})
        street   = addr.get("road") or addr.get("pedestrian") or addr.get("path") or ""
        number   = addr.get("house_number") or ""
        district = addr.get("suburb") or addr.get("city_district") or addr.get("neighbourhood") or ""
        street_full = f"{street} {number}".strip() if number else street
        parts = [p for p in [street_full, district] if p]
        return ", ".join(parts) if parts else "Unknown location"
    except Exception:
        return "Unknown location"

# ── load & compute ────────────────────────────────────────────────────────────
@st.cache_data
def load_sessions():
    # Read both CSV files
    file1 = Path(__file__).parent / "new_player_data_2026_04_20_152626.csv"
    file2 = Path(__file__).parent / "player-activity-77686439338177232025102910010005-2026-04-20T22-26-36-698Z.csv"
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    df = pd.concat([df1, df2], ignore_index=True)
    
    # Get player name before filtering
    player_name = df["playerName"].iloc[0] if "playerName" in df.columns else None
    player_name = player_name if pd.notna(player_name) and player_name else None
    
    df = df[(df["latitude"] != 0) & (df["longitude"] != 0)].copy()
    
    # Handle both timestamp formats: formatted strings and Unix milliseconds
    def parse_time(val):
        try:
            return pd.to_datetime(val, format="%d.%m.%Y %H:%M:%S.%f")
        except:
            return pd.to_datetime(int(val), unit='ms')
    
    df["time"] = df["epoch_time"].apply(parse_time)
    
    rows = []
    out = {}
    for sess in sorted(df["session"].unique()):
        g = df[df["session"] == sess].sort_values("time").reset_index(drop=True)
        
        # Store gross (unfiltered) time range
        gross_start = g["time"].iloc[0]
        gross_end = g["time"].iloc[-1]
        gross_duration_min = (gross_end - gross_start).total_seconds() / 60
        
        # Speed-based trimming: find first and last point with speed ≥ 1.5 m/s
        high_speed = g[g["speed"] >= 1.5]
        if len(high_speed) > 0:
            trim_start_idx = high_speed.index[0]
            trim_end_idx = high_speed.index[-1]
            g = g.loc[trim_start_idx:trim_end_idx].reset_index(drop=True)
        
        # Store session data for zone analysis
        out[int(sess)] = g
        
        g["elapsed_min"] = (g["time"] - g["time"].iloc[0]).dt.total_seconds() / 60
        spd = g["speed"]
        dur = g["elapsed_min"].max()
        
        # Add half column for zone analysis
        g["half"] = g["elapsed_min"].apply(lambda x: "First half" if x <= dur/2 else "Second half")
        
        # Store session data for zone analysis
        out[int(sess)] = g
        
        dist = (spd * 0.5).sum() / 1000
        active_min = (spd > 0.5).sum() * 0.5 / 60
        sc = sprint_count(spd)
        hi = round(((spd >= 3.0).mean()) * 100, 1)

        # fatigue: first vs last 20 min
        first20 = g[g["elapsed_min"] <= 20]["speed"].mean()
        last20  = g[g["elapsed_min"] >= dur - 20]["speed"].mean()
        fade    = round((first20 - last20) / first20 * 100, 1)

        lat_c = g["latitude"].mean()
        lon_c = g["longitude"].mean()
        location = reverse_geocode(lat_c, lon_c)

        rows.append({
            "_sess":         int(sess),
            "_lat":          lat_c,
            "_lon":          lon_c,
            "_gross_start":  gross_start,
            "_gross_end":    gross_end,
            "_gross_dur":    gross_duration_min,
            "_net_start":    g["time"].iloc[0],
            "_net_end":      g["time"].iloc[-1],
            "_trimmed_min":  gross_duration_min - dur,
            "Session":       f"S{int(sess)}",
            "Gross time":    f"{gross_start.strftime('%H:%M:%S')} – {gross_end.strftime('%H:%M:%S')} ({gross_duration_min:.0f} min)",
            "Net time":      f"{g['time'].iloc[0].strftime('%H:%M:%S')} – {g['time'].iloc[-1].strftime('%H:%M:%S')} ({dur:.0f} min)",
            "Trimmed":       f"{round(gross_duration_min - dur, 1)} min",
            "Date":          g["time"].iloc[0].strftime("%d.%m.%Y"),
            "Time":          f"{g['time'].iloc[0].strftime('%H:%M')}–{g['time'].iloc[-1].strftime('%H:%M')}",
            "Location":      location,
            "Duration":      f"{round(dur, 0):.0f} min",
            "Distance":      f"{dist:.2f} km",
            "Dist/min":      f"{round(dist*1000/dur, 1)} m",
            "Active time":   f"{round(active_min, 0):.0f} min",
            "Avg speed":     f"{round(float(spd.mean()), 2)} m/s",
            "Max speed":     f"{round(float(spd.max()), 2)} m/s",
            "Hi-intensity":  f"{hi}%",
            "Sprints":       str(sc),
            "Sprints/min":   f"{round(sc/dur, 3):.3f}",
            "Fade index":    f"{'+' if fade > 0 else ''}{fade}%",
            # raw for sorting / radar
            "_dur":          dur,
            "_dist":         dist,
            "_dist_per_min": dist * 1000 / dur,
            "_max_spd":      float(spd.max()),
            "_mean_spd":     float(spd.mean()),
            "_hi":           hi,
            "_sprints":      sc,
            "_spr_per_min":  sc / dur,
            "_fade":         fade,
            "_active_pct":   active_min / dur * 100,
        })
    return rows, player_name, out

with st.spinner("Loading sessions..."):
    rows, player_name, SESS = load_sessions()

display_cols = [
    "Session", "Date", "Gross time", "Net time",
    "Location",
    "Duration", "Distance", "Dist/min", "Active time",
    "Avg speed", "Max speed", "Hi-intensity", "Sprints", "Sprints/min", "Fade index",
]
table_df = pd.DataFrame(rows)[display_cols]
raw_rows  = {r["_sess"]: r for r in rows}
STATS = compute_zone_stats(SESS)

# ── header ────────────────────────────────────────────────────────────────────
header_text = f"{player_name} · Session overview" if player_name else "Session overview"
st.markdown(f"## {header_text}")
st.caption(f"{len(rows)} sessions loaded · All metrics calculated using **net time** (after noise filtering)")

# ── speed zone definitions ────────────────────────────────────────────────────
st.markdown('<p class="section">Speed zones</p>', unsafe_allow_html=True)
st.markdown("""
<div class="legend">
  <div class="legend-item"><div class="legend-swatch" style="background: #B4B2A9;"></div> <strong>Standing</strong> 0.0–0.5 m/s</div>
  <div class="legend-item"><div class="legend-swatch" style="background: #85B7EB;"></div> <strong>Walking</strong> 0.5–2.0 m/s</div>
  <div class="legend-item"><div class="legend-swatch" style="background: #1D9E75;"></div> <strong>Jogging</strong> 2.0–3.0 m/s</div>
  <div class="legend-item"><div class="legend-swatch" style="background: #EF9F27;"></div> <strong>Running</strong> 3.0–5.0 m/s</div>
  <div class="legend-item"><div class="legend-swatch" style="background: #D85A30;"></div> <strong>Sprinting</strong> 5.0+ m/s</div>
</div>
""", unsafe_allow_html=True)

# ── session table ─────────────────────────────────────────────────────────────
st.markdown('<p class="section">All sessions</p>', unsafe_allow_html=True)

# Create HTML table with tooltips on headers
def create_table_with_hints(df):
    hints = {
        "Session": "Session identifier",
        "Gross time": "Full tracked time range (before noise filtering)",
        "Net time": "Actual session time (after removing pre/post-activity noise)",
        "Date": "Session date",
        "Location": "Geographic location (reverse geocoded)",
        "Duration": "Net session duration in minutes",
        "Distance": "Total distance covered in km",
        "Dist/min": "Average distance per minute",
        "Active time": "Time spent moving (speed > 0.5 m/s)",
        "Avg speed": "Average speed in m/s",
        "Max speed": "Peak speed reached",
        "Hi-intensity": "% time running or sprinting (≥3 m/s)",
        "Sprints": "Number of detected sprints",
        "Sprints/min": "Sprints per minute of activity",
        "Fade index": "Speed change from first to last 20 min (negative = improved)",
    }
    
    header_html = "<thead><tr>"
    for col in df.columns:
        hint = hints.get(col, "")
        header_html += f'<th title="{hint}" style="cursor: help; border-bottom: 2px solid #ddd;">{col}</th>'
    header_html += "</tr></thead>"
    
    rows_html = "<tbody>"
    for _, row in df.iterrows():
        rows_html += "<tr>"
        for val in row:
            rows_html += f"<td style=\"padding: 10px 12px; border-bottom: 1px solid #f3f3f1;\">{val}</td>"
        rows_html += "</tr>"
    rows_html += "</tbody>"
    
    table_html = f"""
    <table style="width:100%; border-collapse: collapse; font-size: 14px;">
      {header_html}
      {rows_html}
    </table>
    """
    return table_html

st.markdown(create_table_with_hints(table_df), unsafe_allow_html=True)

# ── radar comparison ──────────────────────────────────────────────────────────
st.markdown('<p class="section">Radar — 6 dimensions (normalized)</p>', unsafe_allow_html=True)

# Add radar dimensions legend
st.markdown("""
<style>
  .radar-legend { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px; margin: 12px 0 16px; }
  .radar-legend-item { font-size: 12px; color: #666; margin-bottom: 2px; }
  .radar-legend-item strong { color: #333; }
</style>
<div class="radar-legend">
  <div class="radar-legend-item"><strong>Dist/min</strong> <span class="info-tooltip" title="Average distance covered per minute of activity">ℹ</span><br><span style="color: #999; font-size: 11px;">Work intensity (meters/min)</span></div>
  <div class="radar-legend-item"><strong>Sprint density</strong> <span class="info-tooltip" title="Sprints per minute, scaled ×100 for visibility">ℹ</span><br><span style="color: #999; font-size: 11px;">Sprint frequency</span></div>
  <div class="radar-legend-item"><strong>Hi-intensity %</strong> <span class="info-tooltip" title="Percentage of time running or sprinting (speed ≥3 m/s)">ℹ</span><br><span style="color: #999; font-size: 11px;">% running + sprinting</span></div>
  <div class="radar-legend-item"><strong>Max speed</strong> <span class="info-tooltip" title="Peak speed reached during the session">ℹ</span><br><span style="color: #999; font-size: 11px;">m/s</span></div>
  <div class="radar-legend-item"><strong>Active time %</strong> <span class="info-tooltip" title="Percentage of session actively moving (speed > 0.5 m/s)">ℹ</span><br><span style="color: #999; font-size: 11px;">% moving</span></div>
  <div class="radar-legend-item"><strong>Consistency</strong> <span class="info-tooltip" title="Pace stability: high = steady throughout, low = variable or faded">ℹ</span><br><span style="color: #999; font-size: 11px;">Pace stability (inverted)</span></div>
</div>
""", unsafe_allow_html=True)

DIMS = ["Dist/min", "Sprint\ndensity", "Hi-intensity\n%", "Max\nspeed", "Active\ntime %", "Consistency\n(inv)"]

def radar_vals(r):
    std_proxy = 1 / (1 + abs(r["_fade"]) / 10)   # rough consistency proxy
    return [
        r["_dist_per_min"],
        r["_spr_per_min"] * 100,
        r["_hi"],
        r["_max_spd"],
        r["_active_pct"],
        std_proxy * 10,
    ]

all_vals = {sid: radar_vals(raw_rows[sid]) for sid in raw_rows}
maxv = [max(all_vals[s][i] for s in all_vals) for i in range(6)]
norm = {sid: [round(v/m*100, 1) if m else 0 for v, m in zip(all_vals[sid], maxv)] for sid in all_vals}

fig_radar = go.Figure()
fill_colors = {1: "rgba(24,95,165,0.1)", 2: "rgba(216,90,48,0.1)", 3: "rgba(45,134,89,0.1)"}
for sid in sorted(all_vals.keys()):
    vals = norm[sid]
    color = COLORS.get(sid, "#888")
    fill  = fill_colors.get(sid, "rgba(200,200,200,0.1)")
    fig_radar.add_trace(go.Scatterpolar(
        r=vals + [vals[0]], theta=DIMS + [DIMS[0]],
        fill="toself", name=f"S{sid} · {raw_rows[sid]['Date']}",
        line=dict(color=color, width=2),
        fillcolor=fill,
    ))
fig_radar.update_layout(
    polar=dict(radialaxis=dict(visible=True, range=[0, 110])),
    height=360, margin=dict(l=40, r=40, t=20, b=20),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="sans-serif", size=12),
    legend=dict(orientation="h", y=-0.08, x=0.2),
)
st.plotly_chart(fig_radar, use_container_width=True, config={"displayModeBar": False})

st.caption("💡 **Zoom tip:** Click and drag to zoom in. Double-click anywhere on the chart to reset zoom.", help="Plotly charts support interactive zooming. Double-click to reset to full view.")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION — INTENSITY & SPEED ZONES
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("## Intensity & Speed Zones")
st.caption(f"Analyzing {len(SESS)} sessions · Speed zones and high-intensity time")

# zone distribution — % of session time
_section("Zone distribution — % of session time")
fig = go.Figure()
for sid in sorted(SESS.keys()):
    zones = STATS[sid]["zones"]
    label = f"S{sid}"
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

# zone time — absolute minutes
_section("Zone time — absolute minutes")
fig2 = go.Figure()
for sid in sorted(SESS.keys()):
    zones = STATS[sid]["zones"]
    fig2.add_trace(go.Bar(
        name=f"S{sid}",
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
ncols = int(len(SESS)) if len(SESS) else 1
cols = st.columns(ncols)
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
        annotations=[dict(text=f"S{sid}", x=0.5, y=0.5, font_size=11, showarrow=False)],
        showlegend=False,
    )
    col.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})

# hi-intensity % by half
_section("High-intensity % — first half vs second half")
fig4 = go.Figure()
for sid in sorted(SESS.keys()):
    s = STATS[sid]
    fig4.add_trace(go.Bar(
        name=f"S{sid}",
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
