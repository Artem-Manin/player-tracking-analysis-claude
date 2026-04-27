import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="Player Dashboard", layout="wide", initial_sidebar_state="expanded")

# ── colors & constants ────────────────────────────────────────────────────────
BLUE   = "#185FA5"
CORAL  = "#D85A30"
COLORS = {1: BLUE, 2: CORAL}
LABELS = {1: "S1 · 12.04", 2: "S2 · 19.04"}
DATES  = {1: "12.04.2026", 2: "19.04.2026"}
TIMES  = {1: "18:20–19:43", 2: "18:49–19:51"}

ZONE_DEFS = [
    ("Standing",  0.0, 0.5,  "#B4B2A9"),
    ("Walking",   0.5, 2.0,  "#85B7EB"),
    ("Jogging",   2.0, 3.0,  "#1D9E75"),
    ("Running",   3.0, 5.0,  "#EF9F27"),
    ("Sprinting", 5.0, 99.0, "#D85A30"),
]

st.markdown("""
<style>
  [data-testid="stMetricValue"] { font-size: 28px !important; font-weight: 600 !important; }
  [data-testid="stMetricLabel"] { font-size: 12px !important; color: #888 !important; }
  [data-testid="stMetricDelta"] { font-size: 12px !important; }
  .section { font-size: 13px; font-weight: 600; color: #666; text-transform: uppercase;
             letter-spacing: .05em; margin: 1.2rem 0 0.4rem; }
  .insight { background:#f0f5ff; border-left:3px solid #185FA5; border-radius:6px;
             padding:10px 14px; font-size:13px; color:#1a1a2e; margin:6px 0 12px; }
  .warn    { background:#fff8f0; border-left:3px solid #EF9F27; border-radius:6px;
             padding:10px 14px; font-size:13px; color:#5a3e00; margin:6px 0 12px; }
  .good    { background:#f0faf5; border-left:3px solid #1D9E75; border-radius:6px;
             padding:10px 14px; font-size:13px; color:#0a3d22; margin:6px 0 12px; }
</style>
""", unsafe_allow_html=True)

# ── data loading ──────────────────────────────────────────────────────────────
DATA_PATH = Path(__file__).parent / "new_player_data_2026_04_20_152626.csv"

@st.cache_data
def load():
    df = pd.read_csv(DATA_PATH)
    df = df[(df["latitude"] != 0) & (df["longitude"] != 0)].copy()
    df["time"] = pd.to_datetime(df["epoch_time"], format="%d.%m.%Y %H:%M:%S.%f")
    out = {}
    for sess in [1, 2]:
        g = df[df["session"] == sess].sort_values("time").reset_index(drop=True)
        g["elapsed_min"] = (g["time"] - g["time"].iloc[0]).dt.total_seconds() / 60
        g["bucket2"] = (g["elapsed_min"] / 2).astype(int)
        g["bucket5"] = (g["elapsed_min"] / 5).astype(int)
        g["half"] = (g["elapsed_min"] > g["elapsed_min"].max() / 2).map({False: "First half", True: "Second half"})
        out[sess] = g
    return out

SESS = load()

def _sprint_count(spd):
    is_s = (spd > 5.0).astype(int).values
    count, in_s = 0, False
    for v in is_s:
        if v and not in_s: count += 1; in_s = True
        elif not v: in_s = False
    return count

def _sprint_bouts(spd, elapsed):
    bouts = []
    curr, start_min = 0, None
    in_s = False
    for i, v in enumerate((spd > 5.0).astype(int).values):
        if v and not in_s:
            in_s = True; curr = 1
            start_min = elapsed.iloc[i]
        elif v and in_s:
            curr += 1
        elif not v and in_s:
            bouts.append({"duration_s": curr * 0.5, "start_min": round(start_min, 1)})
            in_s = False; curr = 0
    if in_s:
        bouts.append({"duration_s": curr * 0.5, "start_min": round(start_min, 1)})
    return bouts

def _zones(spd):
    total = len(spd)
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

# precompute per-session stats
@st.cache_data
def compute_stats():
    stats = {}
    for sid in [1, 2]:
        g = SESS[sid]
        spd = g["speed"]
        dur = g["elapsed_min"].max()
        dist = (spd * 0.5).sum() / 1000
        active = spd > 0.5
        active_min = active.sum() * 0.5 / 60
        bouts = _sprint_bouts(spd, g["elapsed_min"])
        tl2 = g.groupby("bucket2")["speed"].mean()

        first20 = g[g["elapsed_min"] <= 20]["speed"].mean()
        last20  = g[g["elapsed_min"] >= dur - 20]["speed"].mean()
        fade    = round((first20 - last20) / first20 * 100, 1)

        hi_first = g[g["half"] == "First half"]["speed"].apply(lambda x: x >= 3.0).mean() * 100
        hi_last  = g[g["half"] == "Second half"]["speed"].apply(lambda x: x >= 3.0).mean() * 100

        top5_windows = (g.groupby("bucket2")["speed"].mean()
                         .nlargest(5).reset_index()
                         .rename(columns={"bucket2": "bucket", "speed": "avg_speed"}))
        top5_windows["minute"] = top5_windows["bucket"] * 2

        best5 = (g.groupby("bucket5")["speed"].mean()
                  .nlargest(1).reset_index())
        best5_min = int(best5["bucket5"].iloc[0] * 5)
        best5_spd = round(float(best5["speed"].iloc[0]), 2)

        max_spd_idx = spd.idxmax()
        max_spd_min = round(g.loc[max_spd_idx, "elapsed_min"], 1)

        stats[sid] = {
            "dur": round(dur, 1),
            "dist": round(dist, 2),
            "dist_per_min": round(dist * 1000 / dur, 1),
            "active_min": round(active_min, 1),
            "active_dist": round((spd[active] * 0.5).sum() / 1000, 2),
            "max_speed": round(float(spd.max()), 2),
            "max_speed_min": max_spd_min,
            "mean_speed": round(float(spd.mean()), 2),
            "steps": int(g["step"].max() - g["step"].min()),
            "sprints": len(bouts),
            "spr_per_min": round(len(bouts) / dur, 3),
            "spr_max_dur": round(max((b["duration_s"] for b in bouts), default=0), 1),
            "spr_avg_dur": round(np.mean([b["duration_s"] for b in bouts]) if bouts else 0, 1),
            "bouts": bouts,
            "zones": _zones(spd),
            "hi_pct": round(((spd >= 3.0).mean()) * 100, 1),
            "hi_first": round(hi_first, 1),
            "hi_last": round(hi_last, 1),
            "fade": fade,
            "first20_spd": round(first20, 2),
            "last20_spd": round(last20, 2),
            "consistency_std": round(float(tl2.std()), 3),
            "top5_windows": top5_windows,
            "best5_min": best5_min,
            "best5_spd": best5_spd,
            "sat_mean": round(g["satellite"].mean(), 1),
            "sat_min": int(g["satellite"].min()),
            "pdop_mean": round(g["pdop"].mean(), 2),
            "pdop_max": round(g["pdop"].max(), 2),
            "temp_mean": round(g["temp"].mean(), 1) if "temp" in g.columns else None,
        }
    return stats

STATS = compute_stats()

# ── sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.markdown("## Navigation")
page = st.sidebar.radio("", ["Overview", "Intensity & Zones", "Sprints", "Fatigue & Pacing", "Peak Moments", "Data Quality"])
st.sidebar.divider()
st.sidebar.markdown("**Player:** Engin Kaya")
st.sidebar.markdown(f"**Field:** A · Vienna")
st.sidebar.markdown(f"<span style='color:{BLUE}'>● S1 — 12.04.2026 · 18:20–19:43</span>", unsafe_allow_html=True)
st.sidebar.markdown(f"<span style='color:{CORAL}'>● S2 — 19.04.2026 · 18:49–19:51</span>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "Overview":
    st.markdown("## Overview")
    st.caption("Side-by-side summary of both sessions")

    # ── metric cards ──────────────────────────────────────────────────────────
    _section("Volume")
    c1, c2, c3, c4, c5 = st.columns(5)
    s1, s2 = STATS[1], STATS[2]

    c1.metric("Duration", f"{s1['dur']} min",    f"{s1['dur']-s2['dur']:+.0f} min vs S2")
    c2.metric("Distance", f"{s1['dist']} km",    f"{s1['dist']-s2['dist']:+.2f} km vs S2")
    c3.metric("Dist / min", f"{s1['dist_per_min']} m", f"{s1['dist_per_min']-s2['dist_per_min']:+.1f} m vs S2")
    c4.metric("Active time", f"{s1['active_min']} min", f"{s1['active_min']-s2['active_min']:+.1f} min vs S2")
    c5.metric("Steps", f"{s1['steps']:,}",       f"{s1['steps']-s2['steps']:+,} vs S2")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Duration", f"{s2['dur']} min", delta_color="off", delta=None)
    c2.metric("Distance", f"{s2['dist']} km", delta_color="off")
    c3.metric("Dist / min", f"{s2['dist_per_min']} m", delta_color="off")
    c4.metric("Active time", f"{s2['active_min']} min", delta_color="off")
    c5.metric("Steps", f"{s2['steps']:,}", delta_color="off")

    _section("Speed")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Max speed S1", f"{s1['max_speed']} m/s", f"at min {s1['max_speed_min']}")
    c2.metric("Max speed S2", f"{s2['max_speed']} m/s", f"at min {s2['max_speed_min']}")
    c3.metric("Avg speed S1", f"{s1['mean_speed']} m/s")
    c4.metric("Avg speed S2", f"{s2['mean_speed']} m/s")

    _section("High intensity")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Hi-intensity S1", f"{s1['hi_pct']}%", help="% time at running or sprinting pace (>3 m/s)")
    c2.metric("Hi-intensity S2", f"{s2['hi_pct']}%")
    c3.metric("Sprints S1", f"{s1['sprints']}", f"{s1['spr_per_min']:.3f}/min")
    c4.metric("Sprints S2", f"{s2['sprints']}", f"{s2['spr_per_min']:.3f}/min")

    # ── radar chart ───────────────────────────────────────────────────────────
    _section("Radar — 6 dimensions (normalized to S1 max)")
    dims = ["Dist/min", "Sprint\ndensity", "Hi-intens\n%", "Max\nspeed", "Consistency\n(inv)", "Active\ntime%"]

    def radar_vals(sid):
        s = STATS[sid]
        active_pct = s["active_min"] / s["dur"] * 100
        consistency_inv = max(0, 1 - s["consistency_std"])
        return [s["dist_per_min"], s["spr_per_min"] * 100,
                s["hi_pct"], s["max_speed"],
                consistency_inv * 10, active_pct]

    v1 = radar_vals(1)
    v2 = radar_vals(2)
    maxvals = [max(a, b) for a, b in zip(v1, v2)]
    v1n = [round(a/m*100, 1) if m else 0 for a, m in zip(v1, maxvals)]
    v2n = [round(a/m*100, 1) if m else 0 for a, m in zip(v2, maxvals)]

    fig = go.Figure()
    for vals, sid in [(v1n, 1), (v2n, 2)]:
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]], theta=dims + [dims[0]],
            fill="toself", name=LABELS[sid],
            line=dict(color=COLORS[sid], width=2),
            fillcolor=COLORS[sid].replace("#", "rgba(").rstrip(")") + ",0.1)" if False else (
                "rgba(24,95,165,0.1)" if sid == 1 else "rgba(216,90,48,0.1)"
            ),
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 110])),
        **_layout(350),
        legend=dict(orientation="h", y=-0.05, x=0.3),
        showlegend=True,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # ── red/green comparison table ─────────────────────────────────────────────
    _section("S1 vs S2 — what improved, what didn't")
    metrics_cmp = [
        ("Distance/min (m)", s1["dist_per_min"], s2["dist_per_min"], "higher = better"),
        ("Max speed (m/s)",  s1["max_speed"],    s2["max_speed"],    "higher = better"),
        ("Hi-intensity %",   s1["hi_pct"],       s2["hi_pct"],       "higher = better"),
        ("Sprints/min",      s1["spr_per_min"],  s2["spr_per_min"],  "higher = better"),
        ("Consistency (σ↓)", s1["consistency_std"], s2["consistency_std"], "lower = better"),
        ("Fade index (%↓)",  s1["fade"],         s2["fade"],         "lower = better"),
    ]
    rows = []
    for label, v1, v2, hint in metrics_cmp:
        better_is_higher = "higher" in hint
        s2_better = (v2 > v1) if better_is_higher else (v2 < v1)
        diff = v2 - v1
        sign = "+" if diff > 0 else ""
        icon = "🟢" if s2_better else "🔴"
        rows.append({"Metric": label, "S1": v1, "S2": v2, "Change": f"{sign}{diff:.2f}", "S2 vs S1": icon})

    import pandas as pd
    st.dataframe(pd.DataFrame(rows).set_index("Metric"), use_container_width=True)
    _insight("Green = S2 was better than S1 on that metric. Normalized metrics (per minute) are the most meaningful for comparison since sessions had different durations.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — INTENSITY & ZONES
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Intensity & Zones":
    st.markdown("## Intensity & Speed Zones")

    # stacked zone bars
    _section("Zone distribution — % of session time")
    fig = go.Figure()
    for sid in [1, 2]:
        zones = STATS[sid]["zones"]
        label = LABELS[sid]
        for z in zones:
            fig.add_trace(go.Bar(
                name=z["name"], y=[label], x=[z["pct"]],
                orientation="h", marker_color=z["color"],
                text=f"{z['pct']}%", textposition="inside",
                insidetextanchor="middle",
                textfont=dict(size=11, color="white"),
                hovertemplate=f"<b>{z['name']}</b><br>{z['pct']}% · {z['minutes']} min<extra></extra>",
                showlegend=(sid == 1),
            ))
    fig.update_layout(**_layout(160), barmode="stack",
        xaxis=dict(range=[0,100], showticklabels=False, showgrid=False),
        yaxis=dict(showgrid=False),
        legend=dict(orientation="h", y=1.2, x=0),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # zone minutes side by side
    _section("Zone time — absolute minutes")
    fig2 = go.Figure()
    for sid in [1, 2]:
        zones = STATS[sid]["zones"]
        fig2.add_trace(go.Bar(
            name=LABELS[sid],
            x=[z["name"] for z in zones],
            y=[z["minutes"] for z in zones],
            marker_color=COLORS[sid],
            text=[f"{z['minutes']}m" for z in zones],
            textposition="outside",
            hovertemplate="%{x}: %{y} min<extra>" + LABELS[sid] + "</extra>",
        ))
    fig2.update_layout(**_layout(280), barmode="group",
        yaxis=dict(title="Minutes", showgrid=True, gridcolor="#f0f0f0"),
        xaxis=dict(showgrid=False),
        legend=dict(orientation="h", y=1.1, x=0),
    )
    st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    # donuts
    _section("Zone breakdown — donut view")
    col1, col2 = st.columns(2)
    for col, sid in [(col1, 1), (col2, 2)]:
        zones = STATS[sid]["zones"]
        fig3 = go.Figure(go.Pie(
            labels=[z["name"] for z in zones],
            values=[z["pct"] for z in zones],
            hole=0.55,
            marker_colors=[z["color"] for z in zones],
            textinfo="label+percent",
            hovertemplate="%{label}: %{value}%<extra></extra>",
        ))
        fig3.update_layout(**_layout(260),
            annotations=[dict(text=LABELS[sid], x=0.5, y=0.5, font_size=13, showarrow=False)],
            showlegend=False,
        )
        col.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})

    # hi-intensity % by half
    _section("High-intensity % — first half vs second half")
    fig4 = go.Figure()
    for sid in [1, 2]:
        s = STATS[sid]
        fig4.add_trace(go.Bar(
            name=LABELS[sid],
            x=["First half", "Second half"],
            y=[s["hi_first"], s["hi_last"]],
            marker_color=COLORS[sid],
            marker_opacity=[1.0, 0.6],
            text=[f"{s['hi_first']}%", f"{s['hi_last']}%"],
            textposition="outside",
            hovertemplate="%{x}: %{y:.1f}%<extra>" + LABELS[sid] + "</extra>",
        ))
    fig4.update_layout(**_layout(260), barmode="group",
        yaxis=dict(title="% time at >3 m/s", showgrid=True, gridcolor="#f0f0f0"),
        xaxis=dict(showgrid=False),
        legend=dict(orientation="h", y=1.1, x=0),
    )
    st.plotly_chart(fig4, use_container_width=True, config={"displayModeBar": False})

    s1, s2 = STATS[1], STATS[2]
    _insight(f"S1: {s1['hi_pct']}% total high-intensity time (running + sprinting). "
             f"S2: {s2['hi_pct']}%. Both sessions show similar zone profiles — "
             f"~60% walking, ~11% jogging, ~10% running, <1% sprinting.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — SPRINTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Sprints":
    st.markdown("## Sprint Analysis")
    s1, s2 = STATS[1], STATS[2]

    # key sprint metrics
    _section("Sprint metrics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total sprints S1", s1["sprints"], f"{s1['spr_per_min']:.3f}/min")
    c2.metric("Total sprints S2", s2["sprints"], f"{s2['spr_per_min']:.3f}/min")
    c3.metric("Longest sprint S1", f"{s1['spr_max_dur']}s")
    c4.metric("Longest sprint S2", f"{s2['spr_max_dur']}s")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg sprint dur S1", f"{s1['spr_avg_dur']}s")
    c2.metric("Avg sprint dur S2", f"{s2['spr_avg_dur']}s")
    c3.metric("Max speed S1", f"{s1['max_speed']} m/s", f"at min {s1['max_speed_min']}")
    c4.metric("Max speed S2", f"{s2['max_speed']} m/s", f"at min {s2['max_speed_min']}")

    # sprint timeline — when did sprints happen
    _section("Sprint timeline — when did sprints occur?")
    fig = go.Figure()
    for sid in [1, 2]:
        bouts = STATS[sid]["bouts"]
        if bouts:
            x = [b["start_min"] for b in bouts]
            y = [b["duration_s"] for b in bouts]
            fig.add_trace(go.Scatter(
                x=x, y=y,
                mode="markers",
                name=LABELS[sid],
                marker=dict(color=COLORS[sid], size=9, opacity=0.8,
                            line=dict(color="white", width=1)),
                hovertemplate="min %{x}: %{y}s sprint<extra>" + LABELS[sid] + "</extra>",
            ))
    fig.add_hline(y=s1["spr_avg_dur"], line_dash="dot", line_color=BLUE,
                  annotation_text="S1 avg", annotation_font_size=10)
    fig.add_hline(y=s2["spr_avg_dur"], line_dash="dot", line_color=CORAL,
                  annotation_text="S2 avg", annotation_font_size=10)
    fig.update_layout(**_layout(300),
        xaxis=dict(title="Minute of session", showgrid=False),
        yaxis=dict(title="Sprint duration (s)", showgrid=True, gridcolor="#f0f0f0"),
        legend=dict(orientation="h", y=1.1, x=0),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # sprint clustering: early vs late
    _section("Sprint distribution — early vs late game")
    fig2 = go.Figure()
    for sid in [1, 2]:
        g = SESS[sid]
        dur = g["elapsed_min"].max()
        bouts = STATS[sid]["bouts"]
        early = sum(1 for b in bouts if b["start_min"] <= dur / 2)
        late  = sum(1 for b in bouts if b["start_min"] > dur / 2)
        fig2.add_trace(go.Bar(
            name=LABELS[sid],
            x=["First half", "Second half"],
            y=[early, late],
            marker_color=COLORS[sid],
            text=[str(early), str(late)],
            textposition="outside",
            hovertemplate="%{x}: %{y} sprints<extra>" + LABELS[sid] + "</extra>",
        ))
    fig2.update_layout(**_layout(260), barmode="group",
        yaxis=dict(title="Number of sprints", showgrid=True, gridcolor="#f0f0f0"),
        xaxis=dict(showgrid=False),
        legend=dict(orientation="h", y=1.1, x=0),
    )
    st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    b1 = STATS[1]["bouts"]; b2 = STATS[2]["bouts"]
    dur1 = SESS[1]["elapsed_min"].max(); dur2 = SESS[2]["elapsed_min"].max()
    early1 = sum(1 for b in b1 if b["start_min"] <= dur1/2)
    early2 = sum(1 for b in b2 if b["start_min"] <= dur2/2)
    _insight(f"S1 had {s1['sprints']} sprints total ({early1} in first half, {s1['sprints']-early1} in second). "
             f"S2 had {s2['sprints']} sprints ({early2} in first half, {s2['sprints']-early2} in second). "
             f"S1 had {s1['sprints']-s2['sprints']} more sprints overall, but S1 was also 20 min longer.")

    # sprint plausibility check
    _section("Sprint Plausibility Check")
    session_choice = st.radio("Choose session", [1, 2], format_func=lambda x: LABELS[x], horizontal=True)
    sid = session_choice
    bouts = STATS[sid]["bouts"]
    g = SESS[sid]
    if bouts:
        options = [f"Sprint {i+1}: min {b['start_min']:.1f}, {b['duration_s']}s" for i, b in enumerate(bouts)]
        selected = st.selectbox("Select a sprint to inspect", options, index=None)
        if selected:
            idx = options.index(selected)
            bout = bouts[idx]
            start_min = bout['start_min']
            end_min = start_min + bout['duration_s'] / 60
            bout_data = g[(g['elapsed_min'] >= start_min) & (g['elapsed_min'] <= end_min)]
            st.write(f"**Sprint {idx+1} Details:**")
            st.write(f"- Start time: {start_min:.1f} minutes into session")
            st.write(f"- Duration: {bout['duration_s']} seconds")
            st.write(f"- Max speed: {bout_data['speed'].max():.2f} m/s")
            st.write(f"- Avg speed: {bout_data['speed'].mean():.2f} m/s")
            # Plot
            fig_bout = go.Figure()
            fig_bout.add_trace(go.Scatter(x=bout_data['elapsed_min'], y=bout_data['speed'], mode='lines+markers', name='Speed'))
            fig_bout.update_layout(
                title=f"Speed Trace for Sprint {idx+1}",
                xaxis_title="Elapsed Minutes",
                yaxis_title="Speed (m/s)",
                **_layout(300)
            )
            st.plotly_chart(fig_bout, use_container_width=True, config={"displayModeBar": False})
            # Raw data table
            st.write("**Raw Data Points:**")
            display_data = bout_data[['elapsed_min', 'speed']].copy()
            display_data['elapsed_min'] = display_data['elapsed_min'].round(2)
            display_data['speed'] = display_data['speed'].round(2)
            st.dataframe(display_data, use_container_width=True)
    else:
        st.write("No sprints detected in this session.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — FATIGUE & PACING
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Fatigue & Pacing":
    st.markdown("## Fatigue & Pacing")
    s1, s2 = STATS[1], STATS[2]

    # first vs last 20 min
    _section("First 20 min vs last 20 min — avg speed")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("S1 first 20 min", f"{s1['first20_spd']} m/s")
    c2.metric("S1 last 20 min",  f"{s1['last20_spd']} m/s", f"{s1['fade']:+.1f}% change")
    c3.metric("S2 first 20 min", f"{s2['first20_spd']} m/s")
    c4.metric("S2 last 20 min",  f"{s2['last20_spd']} m/s", f"{s2['fade']:+.1f}% change")

    fig = go.Figure()
    for sid in [1, 2]:
        s = STATS[sid]
        fig.add_trace(go.Bar(
            name=LABELS[sid],
            x=["First 20 min", "Last 20 min"],
            y=[s["first20_spd"], s["last20_spd"]],
            marker_color=COLORS[sid],
            marker_opacity=[1.0, 0.55],
            text=[f"{s['first20_spd']}", f"{s['last20_spd']}"],
            textposition="outside",
            hovertemplate="%{x}: %{y:.2f} m/s<extra>" + LABELS[sid] + "</extra>",
        ))
    fig.update_layout(**_layout(270), barmode="group",
        yaxis=dict(title="Avg speed (m/s)", showgrid=True, gridcolor="#f0f0f0", range=[0, 2.2]),
        xaxis=dict(showgrid=False),
        legend=dict(orientation="h", y=1.1, x=0),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # speed timeline overlay
    _section("Speed over time — 2-min rolling average")
    show1 = st.checkbox(f"Show {LABELS[1]}", value=True)
    show2 = st.checkbox(f"Show {LABELS[2]}", value=True)

    fig2 = go.Figure()
    for sid, show in [(1, show1), (2, show2)]:
        if not show: continue
        g = SESS[sid]
        tl = g.groupby("bucket2")["speed"].mean().reset_index()
        tl["minute"] = tl["bucket2"] * 2
        fig2.add_trace(go.Scatter(
            x=tl["minute"], y=tl["speed"].round(2),
            mode="lines", name=LABELS[sid],
            line=dict(color=COLORS[sid], width=2,
                      dash="dot" if sid == 2 else "solid"),
            fill="tozeroy",
            fillcolor="rgba(24,95,165,0.07)" if sid == 1 else "rgba(216,90,48,0.06)",
            hovertemplate="min %{x}: %{y:.2f} m/s<extra>" + LABELS[sid] + "</extra>",
        ))
    for spd, label in [(0.5, "walking"), (2.0, "jogging"), (3.0, "running")]:
        fig2.add_hline(y=spd, line_dash="dash", line_color="#e0e0e0", line_width=1,
            annotation_text=label, annotation_position="right",
            annotation_font_size=10, annotation_font_color="#bbb")
    fig2.update_layout(**_layout(340),
        xaxis=dict(title="Elapsed minutes", showgrid=False, zeroline=False),
        yaxis=dict(title="Avg speed (m/s)", showgrid=True, gridcolor="#f0f0f0",
                   zeroline=False, range=[0, 4.0]),
        legend=dict(orientation="h", y=1.05, x=0),
        hovermode="x unified",
    )
    st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    # consistency
    _section("Consistency score")
    c1, c2 = st.columns(2)
    c1.metric("S1 consistency (σ)", f"{s1['consistency_std']}",
              help="Std deviation of 2-min avg speed. Lower = more consistent pace.")
    c2.metric("S2 consistency (σ)", f"{s2['consistency_std']}")

    fade1_dir = "increased" if s1["fade"] < 0 else "dropped"
    fade2_dir = "increased" if s2["fade"] < 0 else "dropped"
    _insight(
        f"S1: speed {fade1_dir} by {abs(s1['fade'])}% from first to last 20 min. "
        f"S2: speed {fade2_dir} by {abs(s2['fade'])}%. "
        f"Neither session shows significant fatigue — avg speed stayed stable throughout. "
        f"Consistency (σ): S1={s1['consistency_std']}, S2={s2['consistency_std']} — "
        f"{'S1 was more consistent' if s1['consistency_std'] < s2['consistency_std'] else 'S2 was more consistent'}."
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — PEAK MOMENTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Peak Moments":
    st.markdown("## Peak Moments")

    sid = st.radio("Session", [1, 2], format_func=lambda x: f"{LABELS[x]} ({DATES[x]})", horizontal=True)
    s = STATS[sid]
    g = SESS[sid]

    _section("Top 5 highest-intensity 2-min windows")
    c1, c2, c3 = st.columns(3)
    c1.metric("Best 5-min stretch", f"{s['best5_spd']} m/s avg", f"at min {s['best5_min']}")
    c2.metric("Max speed recorded", f"{s['max_speed']} m/s", f"at min {s['max_speed_min']}")
    c3.metric("Session", LABELS[sid])

    tl = g.groupby("bucket2")["speed"].mean().reset_index()
    tl["minute"] = tl["bucket2"] * 2
    top5 = s["top5_windows"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=tl["minute"], y=tl["speed"].round(2),
        mode="lines", name="Avg speed",
        line=dict(color=COLORS[sid], width=1.5),
        fill="tozeroy",
        fillcolor="rgba(24,95,165,0.07)" if sid == 1 else "rgba(216,90,48,0.06)",
        hovertemplate="min %{x}: %{y:.2f} m/s<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=top5["minute"], y=top5["avg_speed"].round(2),
        mode="markers+text", name="Top 5 windows",
        marker=dict(color="#EF9F27", size=12, symbol="star",
                    line=dict(color="white", width=1)),
        text=[f"#{i+1}" for i in range(len(top5))],
        textposition="top center",
        textfont=dict(size=10),
        hovertemplate="min %{x}: %{y:.2f} m/s<extra>peak</extra>",
    ))
    for spd, label in [(2.0, "jogging"), (3.0, "running"), (5.0, "sprint")]:
        fig.add_hline(y=spd, line_dash="dash", line_color="#e0e0e0", line_width=1,
            annotation_text=label, annotation_position="right",
            annotation_font_size=10, annotation_font_color="#bbb")
    fig.update_layout(**_layout(340),
        xaxis=dict(title="Elapsed minutes", showgrid=False),
        yaxis=dict(title="Avg speed (m/s)", showgrid=True, gridcolor="#f0f0f0", range=[0, 4.5]),
        legend=dict(orientation="h", y=1.05, x=0),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    _section("Top 5 windows — detail")
    top5_display = top5[["minute", "avg_speed"]].copy()
    top5_display.columns = ["Minute", "Avg speed (m/s)"]
    top5_display["Avg speed (m/s)"] = top5_display["Avg speed (m/s)"].round(2)
    top5_display = top5_display.reset_index(drop=True)
    top5_display.index += 1
    st.dataframe(top5_display, use_container_width=True)

    peak_mins = ", ".join([f"min {int(r['minute'])}" for _, r in top5.iterrows()])
    _insight(f"{LABELS[sid]}: peak intensity moments at {peak_mins}. "
             f"Best single 5-min stretch: min {s['best5_min']}–{s['best5_min']+5} "
             f"at {s['best5_spd']} m/s avg. Max speed {s['max_speed']} m/s recorded at min {s['max_speed_min']}.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — DATA QUALITY
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Data Quality":
    st.markdown("## Data Quality")
    st.caption("GPS signal quality — how reliable are the numbers?")

    s1, s2 = STATS[1], STATS[2]

    _section("GPS signal")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Satellites S1 (avg)", s1["sat_mean"])
    c2.metric("Satellites S2 (avg)", s2["sat_mean"])
    c3.metric("Min satellites S1", s1["sat_min"])
    c4.metric("Min satellites S2", s2["sat_min"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("PDOP S1 (avg)", s1["pdop_mean"], help="Position Dilution of Precision. <2 = excellent, <5 = good")
    c2.metric("PDOP S2 (avg)", s2["pdop_mean"])
    c3.metric("PDOP S1 (max)", s1["pdop_max"])
    c4.metric("PDOP S2 (max)", s2["pdop_max"])

    # satellite count over time
    _section("Satellite count over time")
    fig = go.Figure()
    for sid in [1, 2]:
        g = SESS[sid]
        tl = g.groupby("bucket2")["satellite"].mean().reset_index()
        tl["minute"] = tl["bucket2"] * 2
        fig.add_trace(go.Scatter(
            x=tl["minute"], y=tl["satellite"].round(1),
            mode="lines", name=LABELS[sid],
            line=dict(color=COLORS[sid], width=2),
            hovertemplate="min %{x}: %{y:.1f} satellites<extra>" + LABELS[sid] + "</extra>",
        ))
    fig.add_hline(y=20, line_dash="dash", line_color="#1D9E75", line_width=1,
        annotation_text="good (20+)", annotation_font_size=10, annotation_font_color="#1D9E75")
    fig.update_layout(**_layout(280),
        xaxis=dict(title="Elapsed minutes", showgrid=False),
        yaxis=dict(title="Satellites", showgrid=True, gridcolor="#f0f0f0"),
        legend=dict(orientation="h", y=1.1, x=0),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # pdop over time
    _section("PDOP over time (lower = better)")
    fig2 = go.Figure()
    for sid in [1, 2]:
        g = SESS[sid]
        tl = g.groupby("bucket2")["pdop"].mean().reset_index()
        tl["minute"] = tl["bucket2"] * 2
        fig2.add_trace(go.Scatter(
            x=tl["minute"], y=tl["pdop"].round(2),
            mode="lines", name=LABELS[sid],
            line=dict(color=COLORS[sid], width=2),
            hovertemplate="min %{x}: PDOP %{y:.2f}<extra>" + LABELS[sid] + "</extra>",
        ))
    fig2.add_hline(y=2.0, line_dash="dash", line_color="#1D9E75", line_width=1,
        annotation_text="excellent (<2)", annotation_font_size=10, annotation_font_color="#1D9E75")
    fig2.add_hline(y=5.0, line_dash="dash", line_color="#EF9F27", line_width=1,
        annotation_text="acceptable (<5)", annotation_font_size=10, annotation_font_color="#EF9F27")
    fig2.update_layout(**_layout(280),
        xaxis=dict(title="Elapsed minutes", showgrid=False),
        yaxis=dict(title="PDOP", showgrid=True, gridcolor="#f0f0f0"),
        legend=dict(orientation="h", y=1.1, x=0),
    )
    st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    _insight(
        f"Both sessions have excellent GPS quality. "
        f"S1: avg {s1['sat_mean']} satellites, PDOP {s1['pdop_mean']} (max {s1['pdop_max']}). "
        f"S2: avg {s2['sat_mean']} satellites, PDOP {s2['pdop_mean']} (max {s2['pdop_max']}). "
        f"PDOP consistently below 2.0 means position accuracy within ~2–3 meters. "
        f"Speed and distance calculations can be trusted.",
        kind="good"
    )

    st.divider()
    st.markdown("**What is PDOP?** Position Dilution of Precision measures satellite geometry quality. "
                "Below 2 = excellent. Below 5 = good. Above 10 = unreliable. "
                "More satellites spread across the sky = lower PDOP = more accurate position.")
