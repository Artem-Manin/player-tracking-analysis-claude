import streamlit as st
import pandas as pd
import numpy as np
import folium
from pathlib import Path
import math, re, requests
from datetime import timezone, timedelta

# python -m streamlit run sprint_map.py

st.set_page_config(page_title="Sprint Map", layout="wide")

CEST = timezone(timedelta(hours=2))
def to_cest(ts): return ts.replace(tzinfo=timezone.utc).astimezone(CEST)
def fmt_time(ts): return to_cest(ts).strftime("%H:%M:%S")

# ── pitch constants ───────────────────────────────────────────────────────────
COL = {"forward": "#1D9E75", "back": "#E8724A", "lateral": "#aaaaaa"}

FAR_LEFT   = [48.255722, 16.360389]
FAR_RIGHT  = [48.255783, 16.361191]
NEAR_LEFT  = [48.255281, 16.360473]
NEAR_RIGHT = [48.255346, 16.361270]
CORNERS    = [FAR_LEFT, FAR_RIGHT, NEAR_RIGHT, NEAR_LEFT]
CENTER     = [sum(c[0] for c in CORNERS) / 4, sum(c[1] for c in CORNERS) / 4]

lat0 = sum(c[0] for c in CORNERS) / 4
lon0 = sum(c[1] for c in CORNERS) / 4
R    = 6371000
COS  = math.cos(math.radians(lat0))

def to_xy(pt):
    return ((pt[1]-lon0)*COS*(math.pi/180*R), (pt[0]-lat0)*(math.pi/180*R))

far_mid  = [(FAR_LEFT[0]+FAR_RIGHT[0])/2,  (FAR_LEFT[1]+FAR_RIGHT[1])/2]
near_mid = [(NEAR_LEFT[0]+NEAR_RIGHT[0])/2,(NEAR_LEFT[1]+NEAR_RIGHT[1])/2]
fm = np.array(to_xy(far_mid)); nm = np.array(to_xy(near_mid))
u  = (fm - nm) / np.linalg.norm(fm - nm)
v  = np.array([-u[1], u[0]])

def haversine(a, b):
    dlat = math.radians(b[0]-a[0]); dlon = math.radians(b[1]-a[1])
    h = math.sin(dlat/2)**2 + math.cos(math.radians(a[0]))*math.cos(math.radians(b[0]))*math.sin(dlon/2)**2
    return 2*R*math.asin(math.sqrt(h))

def sprint_count(spd):
    count, in_s = 0, False
    for v in (spd > 5.0).values:
        if v and not in_s: count += 1; in_s = True
        elif not v: in_s = False
    return count

def parse_time(val):
    s = str(val).strip()
    for fmt in ("%d.%m.%Y %H:%M:%S.%f", "%d.%m.%Y %H:%M:%S"):
        try: return pd.to_datetime(s, format=fmt)
        except: pass
    try: return pd.to_datetime(int(float(s)), unit="ms")
    except: return pd.NaT

def player_name_from(path: str, df: pd.DataFrame) -> str:
    if "playerName" in df.columns:
        vals = df["playerName"].dropna()
        if not vals.empty:
            name = str(vals.iloc[0]).strip()
            if name: return name
    m = re.match(r"^([a-zA-Z]+)_", Path(path).stem)
    if m: return m.group(1).capitalize()
    return Path(path).stem[:20]

def reverse_geocode(lat, lon):
    try:
        r = requests.get(
            "https://nominatim.openstreetmap.org/reverse",
            params={"lat": lat, "lon": lon, "format": "json", "accept-language": "en"},
            headers={"User-Agent": "coach-dashboard/1.0"}, timeout=4,
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

@st.cache_data
def discover_sessions(folder: str):
    sessions = []
    for csv_path in sorted(Path(folder).glob("*.csv")):
        try:
            df = pd.read_csv(csv_path, low_memory=False)
        except Exception:
            continue
        df["time"] = df["epoch_time"].apply(parse_time)
        player = player_name_from(str(csv_path), df)
        vienna = (df.latitude>48)&(df.latitude<49)&(df.longitude>16)&(df.longitude<17)
        df_v = df[vienna].dropna(subset=["time"])

        for sess_id in sorted(df["session"].dropna().unique()):
            g = df_v[df_v["session"] == sess_id]
            if len(g) < 30: continue
            hi = g[g["speed"] >= 1.5]
            if len(hi) == 0: continue
            g2 = g.loc[hi.index[0]:hi.index[-1]]
            dur = (g2.time.max() - g2.time.min()).total_seconds() / 60
            if dur < 10: continue
            date = to_cest(g2.time.min()).date()
            location = reverse_geocode(g2.latitude.mean(), g2.longitude.mean())
            if "steinb" not in location.lower():
                continue
            label = f"{date.strftime('%d.%m.%Y')}  ·  {player}  ·  {location}"
            sessions.append({
                "label":      label,
                "path":       str(csv_path),
                "session_id": int(sess_id),
                "player":     player,
                "date":       date,
                "location":   location,
            })

    sessions.sort(key=lambda x: (x["date"], x["player"], x["location"]))
    return sessions

@st.cache_data
def load_session(path: str, session_id: int):
    df = pd.read_csv(path, low_memory=False)
    df["time"] = df["epoch_time"].apply(parse_time)
    g = df[df["session"] == session_id].sort_values("time").reset_index(drop=True)
    gv = g[(g.latitude>48)&(g.latitude<49)&(g.longitude>16)&(g.longitude<17)].dropna(subset=["time"]).reset_index(drop=True)
    if len(gv) < 60:
        raise ValueError("Not enough valid GPS data")

    gv = gv.copy()
    gv["roll"] = gv["speed"].rolling(60, min_periods=10, center=True).mean()
    THRESH = 1.0
    try:
        si = next(i for i in range(len(gv)-60) if gv.roll.iloc[i]>THRESH and gv.roll.iloc[i:i+60].mean()>THRESH)
        ei = next(i for i in range(len(gv)-1, 60, -1) if gv.roll.iloc[i]>THRESH and gv.roll.iloc[i-60:i].mean()>THRESH)
    except StopIteration:
        si, ei = 0, len(gv)-1

    gross_start = gv.time.iloc[0]
    gross_end   = gv.time.iloc[-1]
    net = gv.iloc[si:ei+1].reset_index(drop=True).copy()
    net_start   = net.time.iloc[0]
    net_end     = net.time.iloc[-1]
    gross_min   = (gross_end - gross_start).total_seconds() / 60
    net_min     = (net_end - net_start).total_seconds() / 60
    trim_start  = (net_start - gross_start).total_seconds() / 60
    trim_end    = (gross_end - net_end).total_seconds() / 60

    spd  = net["speed"]
    dist = (spd * 0.5).sum() / 1000
    sc   = sprint_count(spd)

    net["x"]      = (net.longitude - lon0)*COS*(math.pi/180*R)
    net["y"]      = (net.latitude  - lat0)*(math.pi/180*R)
    net["along"]  = net["x"]*u[0] + net["y"]*u[1]
    net["across"] = net["x"]*v[0] + net["y"]*v[1]

    in_s = False; sprints_raw = []; si2 = None
    for i, row in net.iterrows():
        if row.speed > 5.0 and not in_s: in_s = True; si2 = i
        elif row.speed <= 5.0 and in_s:
            in_s = False
            seg = net.loc[si2:i-1]
            if len(seg) >= 2: sprints_raw.append(seg)

    sprint_records = []
    for s in sprints_raw:
        da = s["along"].iloc[-1] - s["along"].iloc[0]
        if abs(da) > 80: continue
        direction = "forward" if da > 2 else ("back" if da < -2 else "lateral")
        coords = [[float(r.latitude), float(r.longitude)] for _, r in s.iterrows()]
        path_dist = sum(haversine(coords[i], coords[i+1]) for i in range(len(coords)-1))
        sprint_records.append({
            "path":         coords,
            "spd":          round(float(s.speed.max()), 1),
            "path_dist":    round(path_dist, 1),
            "displacement": round(float(abs(da)), 1),
            "dir":          direction,
        })

    track = [[float(r.latitude), float(r.longitude)] for _, r in net.iloc[::4].iterrows()]

    return {
        "player":       player_name_from(path, df),
        "gross_start":  gross_start, "gross_end":  gross_end,  "gross_min":  round(gross_min, 1),
        "net_start":    net_start,   "net_end":    net_end,    "net_min":    round(net_min, 1),
        "trim_start":   round(trim_start, 1), "trim_end": round(trim_end, 1),
        "dist_km":      round(dist, 2),
        "avg_speed":    round(float(spd.mean()), 2),
        "median_speed": round(float(spd.median()), 2),
        "max_speed":    round(float(spd.max()), 2),
        "sprints_tot":  sc,
        "track":        track,
        "sprints":      sprint_records,
    }

# ══════════════════════════════════════════════════════════════════════════════
DATA_DIR = Path(__file__).parent

with st.spinner("Scanning sessions…"):
    sessions = discover_sessions(str(DATA_DIR))

if not sessions:
    st.error("No Steinbüchlweg sessions found.")
    st.stop()

with st.sidebar:
    st.markdown("### 📅 Session")
    labels       = [s["label"] for s in sessions]
    chosen_label = st.selectbox("Date · Player · Place", labels)
    chosen       = sessions[labels.index(chosen_label)]

    st.markdown("---")
    st.markdown("### 🎛 Layers")
    show_track   = st.toggle("GPS track",  value=True)
    show_pitch   = st.toggle("Pitch",      value=True)
    show_sprints = st.toggle("Sprints",    value=True)

    st.markdown("---")
    st.markdown("### 🔍 Sprint filter")
    show_fwd = st.checkbox("Forward",  value=True)
    show_bck = st.checkbox("Back",     value=True)
    show_lat = st.checkbox("Lateral",  value=True)
    dir_filter = {d for d, on in [("forward", show_fwd), ("back", show_bck), ("lateral", show_lat)] if on}

with st.spinner("Loading session…"):
    try:
        data = load_session(chosen["path"], chosen["session_id"])
    except Exception as e:
        st.error(f"Could not load session: {e}")
        st.stop()

sprints = data["sprints"]
fwd_n   = sum(1 for s in sprints if s["dir"] == "forward")
bck_n   = sum(1 for s in sprints if s["dir"] == "back")
lat_n   = sum(1 for s in sprints if s["dir"] == "lateral")
visible = [s for s in sprints if s["dir"] in dir_filter]

# ── header ────────────────────────────────────────────────────────────────────
st.markdown(f"## {data['player']} · Sprint Map · {chosen['date'].strftime('%d.%m.%Y')}")
st.caption(chosen["location"])

# ── session overview — one line ───────────────────────────────────────────────
st.markdown("#### Session overview")
st.markdown(f"""
<div style="display:flex;align-items:center;gap:24px;padding:10px 14px;
            background:#f9f9f7;border-radius:6px;border-left:4px solid #388adc;
            font-size:13px;flex-wrap:wrap">
  <span>⏱ <b>{fmt_time(data['gross_start'])}–{fmt_time(data['gross_end'])}</b> gross
        &nbsp;→&nbsp;
        <b>{fmt_time(data['net_start'])}–{fmt_time(data['net_end'])}</b> net
        &nbsp;(trimmed {data['trim_start']:.0f}+{data['trim_end']:.0f} min)</span>
  <span>📏 <b>{data['dist_km']} km</b></span>
  <span>⚡ avg <b>{data['avg_speed']} m/s</b>
        · med <b>{data['median_speed']} m/s</b>
        · max <b>{data['max_speed']} m/s</b></span>
  <span>🏃 <b>{len(sprints)}</b> sprints:
        <span style="color:#1D9E75">{fwd_n} fwd</span>
        · <span style="color:#E8724A">{bck_n} back</span>
        · <span style="color:#888">{lat_n} lat</span></span>
</div>""", unsafe_allow_html=True)

st.markdown("---")

# ── map ───────────────────────────────────────────────────────────────────────
m = folium.Map(location=CENTER, zoom_start=19, max_zoom=22, tiles=None)

folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="Esri World Imagery", name="Satellite",
    overlay=False, control=True, max_zoom=22, max_native_zoom=19,
).add_to(m)
folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}",
    attr="Esri Labels", name="Labels",
    overlay=True, control=True, opacity=0.5, max_zoom=22, max_native_zoom=19,
).add_to(m)

if show_pitch:
    folium.Polygon(
        locations=CORNERS + [CORNERS[0]],
        color="#e8c832", weight=2.5, fill=True,
        fill_color="#e8c832", fill_opacity=0.05, dash_array="6 4",
        tooltip="Pitch: 59.8 × 49.4 m",
    ).add_to(m)
    for pts, label in [([FAR_LEFT, FAR_RIGHT], "Far goal"), ([NEAR_LEFT, NEAR_RIGHT], "Near goal")]:
        folium.PolyLine(locations=pts, color="#e8c832", weight=5, opacity=0.95, tooltip=label).add_to(m)
        mid = [(pts[0][0]+pts[1][0])/2, (pts[0][1]+pts[1][1])/2]
        folium.Marker(
            location=mid,
            icon=folium.DivIcon(
                html=f'<div style="background:rgba(18,18,18,0.88);border:1px solid #555;border-radius:4px;padding:2px 7px;font-size:11px;font-weight:600;color:#e8c832;white-space:nowrap">{label}</div>',
                icon_size=(80,22), icon_anchor=(40,11),
            ),
        ).add_to(m)

if show_track:
    folium.PolyLine(
        data["track"], color="#388adc", weight=2, opacity=0.4,
        tooltip=f"{data['player']} track",
    ).add_to(m)

if show_sprints:
    for i, s in enumerate(visible):
        col  = COL[s["dir"]]
        path = s["path"]
        tooltip = f"S{i+1} · {s['dir']} · {s['path_dist']}m · {s['spd']} m/s"

        # glow
        folium.PolyLine(path, color=col, weight=10, opacity=0.15).add_to(m)
        # main line
        folium.PolyLine(path, color=col, weight=4, opacity=0.9, tooltip=tooltip).add_to(m)
        # start dot (filled)
        folium.CircleMarker(path[0], radius=7, color="#fff", fill=True,
                            fill_color=col, fill_opacity=1.0, weight=2,
                            tooltip=tooltip).add_to(m)
        # end dot (ring) — indicates direction
        folium.CircleMarker(path[-1], radius=5, color=col, fill=True,
                            fill_color="#fff", fill_opacity=1.0, weight=2.5,
                            tooltip=tooltip).add_to(m)

folium.LayerControl(collapsed=False).add_to(m)

# Render as raw HTML — avoids st_folium widget caching issues entirely
import streamlit.components.v1 as components
map_html = m.get_root().render()
components.html(map_html, height=640, scrolling=False)

# ── sprint detail table ───────────────────────────────────────────────────────
if sprints:
    st.markdown("---")
    st.markdown("#### Sprint detail")
    rows = []
    for i, s in enumerate(sprints):
        rows.append({
            "#":                i + 1,
            "Direction":        s["dir"],
            "Path dist (m)":    s["path_dist"],
            "Displacement (m)": s["displacement"],
            "Max speed (m/s)":  s["spd"],
            "km/h":             round(s["spd"] * 3.6, 1),
            "GPS points":       len(s["path"]),
        })
    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True, hide_index=True,
        column_config={
            "Direction":        st.column_config.TextColumn("Direction"),
            "Path dist (m)":    st.column_config.NumberColumn("Path dist (m)",    format="%.1f"),
            "Displacement (m)": st.column_config.NumberColumn("Displacement (m)", format="%.1f"),
            "Max speed (m/s)":  st.column_config.NumberColumn("Max speed (m/s)",  format="%.1f"),
            "km/h":             st.column_config.NumberColumn("km/h",             format="%.1f"),
        }
    )
