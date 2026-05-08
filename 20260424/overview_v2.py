import streamlit as st
import pandas as pd
from pathlib import Path
import requests
import re

# python -m streamlit run overview_v2.py

st.set_page_config(page_title="Session Overview", layout="wide")

DATA_FOLDER = Path(__file__).parent

def sprint_count(spd):
    count, in_s = 0, False
    for v in (spd > 5.0).values:
        if v and not in_s: count += 1; in_s = True
        elif not v: in_s = False
    return count

def parse_time(val):
    s = str(val).strip()
    for fmt in ("%d.%m.%Y %H:%M:%S.%f", "%d.%m.%Y %H:%M:%S"):
        try:
            return pd.to_datetime(s, format=fmt)
        except Exception:
            pass
    try:
        return pd.to_datetime(int(float(s)), unit="ms")
    except Exception:
        return pd.NaT

def reverse_geocode(lat, lon):
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

def player_name_from_file(path: Path, df: pd.DataFrame) -> str:
    if "playerName" in df.columns:
        vals = df["playerName"].dropna()
        if not vals.empty:
            name = str(vals.iloc[0]).strip()
            if name:
                return name
    m = re.match(r"^([a-zA-Z]+)_", path.stem)
    if m:
        return m.group(1).capitalize()
    return path.stem[:20]

@st.cache_data
def discover_players(folder: Path):
    seen: dict = {}
    for csv_path in sorted(folder.glob("*.csv")):
        try:
            df_head = pd.read_csv(csv_path, nrows=10)
        except Exception:
            continue
        name = player_name_from_file(csv_path, df_head)
        seen.setdefault(name, []).append(str(csv_path))
    return seen

@st.cache_data
def load_all_players(player_map_items: tuple) -> pd.DataFrame:
    all_rows = []
    for player_name, csv_paths in player_map_items:
        frames = []
        for p in csv_paths:
            try:
                frames.append(pd.read_csv(p))
            except Exception:
                continue
        if not frames:
            continue

        df = pd.concat(frames, ignore_index=True)

        display_name = player_name
        if "playerName" in df.columns:
            vals = df["playerName"].dropna()
            if not vals.empty:
                n = str(vals.iloc[0]).strip()
                if n:
                    display_name = n

        df = df[(df["latitude"] != 0) & (df["longitude"] != 0)].copy()
        if df.empty:
            continue

        df["time"] = df["epoch_time"].apply(parse_time)
        df = df.dropna(subset=["time"])

        for sess in sorted(df["session"].unique()):
            g = df[df["session"] == sess].sort_values("time").reset_index(drop=True)

            gross_start = g["time"].iloc[0]
            gross_end   = g["time"].iloc[-1]
            gross_dur   = (gross_end - gross_start).total_seconds() / 60

            high_speed = g[g["speed"] >= 1.5]
            if len(high_speed) > 0:
                g = g.loc[high_speed.index[0]:high_speed.index[-1]].reset_index(drop=True)

            g["elapsed_min"] = (g["time"] - g["time"].iloc[0]).dt.total_seconds() / 60
            spd = g["speed"]
            dur = g["elapsed_min"].max()
            if dur < 1:
                continue

            dist       = (spd * 0.5).sum() / 1000
            active_min = (spd > 0.5).sum() * 0.5 / 60
            sc         = sprint_count(spd)
            hi         = round(((spd >= 3.0).mean()) * 100, 1)

            first20 = g[g["elapsed_min"] <= 20]["speed"].mean()
            last20  = g[g["elapsed_min"] >= dur - 20]["speed"].mean()
            fade    = round((first20 - last20) / first20 * 100, 1) if first20 else 0.0

            location = reverse_geocode(g["latitude"].mean(), g["longitude"].mean())

            all_rows.append({
                "Player":        display_name,
                "Session":       int(sess),
                "Date":          g["time"].iloc[0].date(),
                "Location":      location,
                "Duration (min)": round(dur, 1),
                "Distance (km)": round(dist, 2),
                "Dist/min (m)":  round(dist * 1000 / dur, 1),
                "Active (min)":  round(active_min, 1),
                "Avg speed":     round(float(spd.mean()), 2),
                "Max speed":     round(float(spd.max()), 2),
                "Hi-intensity %": hi,
                "Sprints":       sc,
                "Sprints/min":   round(sc / dur, 3),
                "Fade index %":  fade,
            })

    return pd.DataFrame(all_rows)


# ── main ──────────────────────────────────────────────────────────────────────
with st.spinner("Scanning session files…"):
    player_map = discover_players(DATA_FOLDER)

if not player_map:
    st.error(f"No CSV files found in {DATA_FOLDER}")
    st.stop()

with st.spinner("Loading all players…"):
    table_df = load_all_players(tuple(player_map.items()))

if table_df.empty:
    st.warning("No valid sessions found.")
    st.stop()

st.markdown("## Session overview — all players")
st.caption(f"{table_df['Player'].nunique()} players · {len(table_df)} sessions · click any column header to sort")

st.dataframe(
    table_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Session":        st.column_config.NumberColumn("Session", format="%d"),
        "Date":           st.column_config.DateColumn("Date", format="DD.MM.YYYY"),
        "Duration (min)": st.column_config.NumberColumn("Duration (min)", format="%.1f"),
        "Distance (km)":  st.column_config.NumberColumn("Distance (km)", format="%.2f"),
        "Dist/min (m)":   st.column_config.NumberColumn("Dist/min (m)", format="%.1f"),
        "Active (min)":   st.column_config.NumberColumn("Active (min)", format="%.1f"),
        "Avg speed":      st.column_config.NumberColumn("Avg speed (m/s)", format="%.2f"),
        "Max speed":      st.column_config.NumberColumn("Max speed (m/s)", format="%.2f"),
        "Hi-intensity %": st.column_config.NumberColumn("Hi-intensity %", format="%.1f"),
        "Sprints":        st.column_config.NumberColumn("Sprints", format="%d"),
        "Sprints/min":    st.column_config.NumberColumn("Sprints/min", format="%.3f"),
        "Fade index %":   st.column_config.NumberColumn("Fade index %", format="%.1f"),
    },
)