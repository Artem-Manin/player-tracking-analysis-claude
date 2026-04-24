import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import requests

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
    ("Standing",  0.0, 0.5),
    ("Walking",   0.5, 2.0),
    ("Jogging",   2.0, 3.0),
    ("Running",   3.0, 5.0),
    ("Sprinting", 5.0, 99.0),
]

# ── helpers ───────────────────────────────────────────────────────────────────
def sprint_count(spd):
    count, in_s = 0, False
    for v in (spd > 5.0).values:
        if v and not in_s: count += 1; in_s = True
        elif not v: in_s = False
    return count

def zone_pct(spd, lo, hi):
    return round(((spd >= lo) & (spd < hi)).mean() * 100, 1)

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
        
        g["elapsed_min"] = (g["time"] - g["time"].iloc[0]).dt.total_seconds() / 60
        spd = g["speed"]
        dur = g["elapsed_min"].max()
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
    return rows, player_name

with st.spinner("Loading sessions..."):
    rows, player_name = load_sessions()

display_cols = [
    "Session", "Date", "Gross time", "Net time",
    "Location",
    "Duration", "Distance", "Dist/min", "Active time",
    "Avg speed", "Max speed", "Hi-intensity", "Sprints", "Sprints/min", "Fade index",
]
table_df = pd.DataFrame(rows)[display_cols]
raw_rows  = {r["_sess"]: r for r in rows}

# ── header ────────────────────────────────────────────────────────────────────
header_text = f"{player_name} · Session overview" if player_name else "Session overview"
st.markdown(f"## {header_text}")
st.caption(f"{len(rows)} sessions loaded · All metrics calculated using **net time** (after noise filtering)")

# ── speed zone definitions ────────────────────────────────────────────────────
st.markdown('<p class="section">Speed zones</p>', unsafe_allow_html=True)
st.markdown("""
<div class="legend">
  <div class="legend-item"><div class="legend-swatch" style="background: #92a3a3;"></div> <strong>Standing</strong> 0.0–0.5 m/s</div>
  <div class="legend-item"><div class="legend-swatch" style="background: #99b3cc;"></div> <strong>Walking</strong> 0.5–2.0 m/s</div>
  <div class="legend-item"><div class="legend-swatch" style="background: #f4a460;"></div> <strong>Jogging</strong> 2.0–3.0 m/s</div>
  <div class="legend-item"><div class="legend-swatch" style="background: #ff8c42;"></div> <strong>Running</strong> 3.0–5.0 m/s</div>
  <div class="legend-item"><div class="legend-swatch" style="background: #e74c3c;"></div> <strong>Sprinting</strong> 5.0+ m/s</div>
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

# ── comparison bar — key normalized metrics ────────────────────────────────────
st.markdown('''<p class="section">Key metrics — side by side
  <span class="info-tooltip" title="Distance covered per minute of activity">ℹ</span>
</p>''', unsafe_allow_html=True)

metrics = [
    ("Dist/min (m)",    "_dist_per_min", "Average distance covered per minute"),
    ("Max speed (m/s)", "_max_spd",      "Peak speed reached during session"),
    ("Hi-intensity %",  "_hi",           "Percentage of time spent running or sprinting (≥3 m/s)"),
    ("Sprints/min",     "_spr_per_min",  "Number of sprints per minute of activity"),
]

cols = st.columns(len(metrics))
for col, (label, key, hint) in zip(cols, metrics):
    # Display metric label with tooltip
    col.markdown(f'<div style="font-size: 12px; color: #666; margin-bottom: 8px;"><strong>{label}</strong> <span class="info-tooltip" title="{hint}">ℹ</span></div>', unsafe_allow_html=True)
    
    fig_bar = go.Figure()
    for sid in sorted(raw_rows.keys()):
        r = raw_rows[sid]
        val = r[key]
        fig_bar.add_trace(go.Bar(
            name=f"S{sid}",
            x=[f"S{sid}"],
            y=[round(val, 3)],
            marker_color=COLORS.get(sid, "#888"),
            text=[f"{round(val, 2)}"],
            textposition="outside",
            showlegend=False,
            hovertemplate=f"S{sid}: {round(val,3)}<extra></extra>",
        ))
    fig_bar.update_layout(
        title=dict(text="", font=dict(size=12), x=0),
        height=200,
        margin=dict(l=4, r=4, t=8, b=4),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(showgrid=False, showticklabels=False, range=[0, max(raw_rows[s][key] for s in raw_rows) * 1.25]),
        xaxis=dict(showgrid=False),
        font=dict(size=11),
        bargap=0.35,
    )
    col.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

# ── insight ───────────────────────────────────────────────────────────────────
if len(raw_rows) >= 2:
    # Find sessions with highest and lowest work rate
    sessions_sorted = sorted(raw_rows.items(), key=lambda x: x[1]["_dist_per_min"], reverse=True)
    best_sid, best_row = sessions_sorted[0]
    worst_sid, worst_row = sessions_sorted[-1]
    
    st.markdown(
        f'<div class="insight">💡 '
        f'S{best_sid} had highest work rate ({best_row["_dist_per_min"]:.1f} m/min) '
        f'vs S{worst_sid} ({worst_row["_dist_per_min"]:.1f} m/min). '
        f'S{best_sid} had {best_row["_sprints"]} sprints in {best_row["_dur"]:.0f} min '
        f'({best_row["_spr_per_min"]:.3f}/min), '
        f'vs S{worst_sid} with {worst_row["_sprints"]} sprints in {worst_row["_dur"]:.0f} min '
        f'({worst_row["_spr_per_min"]:.3f}/min).'
        f'</div>',
        unsafe_allow_html=True,
    )
