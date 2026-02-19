"""
loader.py
---------
Generic loader for player tracking files (GPS+IMU or IMU-only).

Normalises column names, parses/reconstructs timestamps, converts raw
sensor integers to SI units, and returns a clean DataFrame plus a
metadata dict that tells downstream code what is and is not available.

IMU scaling assumptions (MPU-6000 family, typical for sport trackers):
  Accelerometer : ±4 g range  → scale = 8 192 LSB/g
  Gyroscope     : ±250 dps range → scale = 131 LSB/(°/s)
  Temperature   : raw / 256  → °C   (empirically validated against
                               Feb 2026 Vienna weather: –22 to –1 °C ✓)

Speed unit in file 1 is m/s (confirmed by Haversine cross-check).
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ACC_SCALE = 8_192.0          # LSB per g
GYRO_SCALE = 131.0           # LSB per °/s
TEMP_SCALE = 256.0           # LSB per °C

# Football speed zones (km/h) – UEFA / EPTS standard
SPEED_BINS_KMH = [0, 2, 7, 14, 19, 23, 999]
SPEED_LABELS = ["Standing", "Walking", "Jogging", "Running", "High Speed", "Sprinting"]

# Canonical column map: {normalised_name: [possible raw variants]}
COLUMN_ALIASES: dict[str, list[str]] = {
    "timestamp":  ["EpochTime", "epoch_time", "Timestamp", "timestamp"],
    "latitude":   ["Latitude", "latitude"],
    "longitude":  ["Longitude", "longitude"],
    "altitude":   ["Altitude", "altitude"],
    "satellite":  ["Satellite", "satellite"],
    "pdop":       ["PDOP", "pdop"],
    "speed_ms":   ["Speed", "speed"],
    "heading":    ["Heading", "heading"],
    "acc_x_raw":  ["AccX", "accX"],
    "acc_y_raw":  ["AccY", "accY"],
    "acc_z_raw":  ["AccZ", "accZ"],
    "rot_x_raw":  ["RotX", "rotX"],
    "rot_y_raw":  ["RotY", "rotY"],
    "rot_z_raw":  ["RotZ", "rotZ"],
    "pitch_deg":  ["Pitch", "pitch"],
    "roll_deg":   ["Roll", "roll"],
    "temp_raw":   ["Temp", "temp"],
    "session":    ["Session", "session"],
    "step":       ["Step", "step"],
    "crc":        ["CRC", "crc"],
    "count":      ["Count", "cnt"],
    "player_id":  ["playerId"],
    "device_id":  ["deviceId"],
    "field_id":   ["soccerFieldId"],
    "synced":     ["synced"],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename raw columns to canonical names (only renames what exists)."""
    rename_map: dict[str, str] = {}
    raw_cols = set(df.columns)
    for canonical, variants in COLUMN_ALIASES.items():
        for v in variants:
            if v in raw_cols:
                rename_map[v] = canonical
                break
    return df.rename(columns=rename_map)


def _parse_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """Try to parse the timestamp column from various formats."""
    if "timestamp" not in df.columns:
        return df

    sample = df["timestamp"].dropna().iloc[0] if not df["timestamp"].dropna().empty else None
    if sample is None:
        return df

    # String timestamps like '04.02.2026 17:31:31.200'
    if isinstance(sample, str):
        try:
            df["timestamp"] = pd.to_datetime(df["timestamp"], format="%d.%m.%Y %H:%M:%S.%f", utc=False)
            return df
        except Exception:
            pass
        try:
            df["timestamp"] = pd.to_datetime(df["timestamp"], infer_datetime_format=True)
            return df
        except Exception:
            pass

    # Unix epoch (ms or s)
    if pd.api.types.is_numeric_dtype(df["timestamp"]):
        ts = df["timestamp"]
        if ts.max() > 1e12:  # milliseconds
            df["timestamp"] = pd.to_datetime(ts, unit="ms", utc=True)
        else:
            df["timestamp"] = pd.to_datetime(ts, unit="s", utc=True)

    return df


def _reconstruct_timestamp(df: pd.DataFrame, start: str, freq_ms: int = 500) -> pd.DataFrame:
    """
    Build a synthetic timestamp column when none is available.
    Records are assumed consecutive with fixed interval.
    """
    start_dt = pd.Timestamp(start)  # naive → treated as local (CET)
    df["timestamp"] = [start_dt + pd.Timedelta(milliseconds=i * freq_ms) for i in range(len(df))]
    df["timestamp_reconstructed"] = True
    return df


def _convert_imu(df: pd.DataFrame) -> pd.DataFrame:
    """Convert raw integer IMU values to SI / readable units."""
    for axis in ["x", "y", "z"]:
        raw_col = f"acc_{axis}_raw"
        if raw_col in df.columns:
            df[f"acc_{axis}_g"] = df[raw_col] / ACC_SCALE          # g
            df[f"acc_{axis}_ms2"] = df[f"acc_{axis}_g"] * 9.80665  # m/s²

        raw_col = f"rot_{axis}_raw"
        if raw_col in df.columns:
            df[f"rot_{axis}_dps"] = df[raw_col] / GYRO_SCALE       # °/s

    if "temp_raw" in df.columns:
        df["temp_c"] = df["temp_raw"] / TEMP_SCALE

    return df


def _add_speed_zone(df: pd.DataFrame) -> pd.DataFrame:
    if "speed_ms" in df.columns and df["speed_ms"].notna().any():
        speed_kmh = df["speed_ms"] * 3.6
        df["speed_kmh"] = speed_kmh
        df["speed_zone"] = pd.cut(
            speed_kmh,
            bins=SPEED_BINS_KMH,
            labels=SPEED_LABELS,
            right=False,
        )
    return df


def _add_utc_cet(df: pd.DataFrame) -> pd.DataFrame:
    """Add UTC and CET/CEST columns from timestamp."""
    if "timestamp" not in df.columns:
        return df

    ts = df["timestamp"]

    # Make UTC-aware if naive (file 1 timestamps are local Vienna time = CET = UTC+1)
    if ts.dt.tz is None:
        ts_utc = ts.dt.tz_localize("Europe/Vienna").dt.tz_convert("UTC")
    else:
        ts_utc = ts.dt.tz_convert("UTC")

    df["ts_utc"] = ts_utc
    df["ts_cet"] = ts_utc.dt.tz_convert("Europe/Vienna")
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def load_file(
    path: str | Path,
    imu_only_start: Optional[str] = None,
    imu_only_freq_ms: int = 500,
) -> tuple[pd.DataFrame, dict]:
    """
    Load a player tracking CSV and return (df, meta).

    Parameters
    ----------
    path            : path to CSV
    imu_only_start  : if the file has no valid timestamp, supply the session
                      start as an ISO string or 'DD.MM.YYYY HH:MM' string.
                      Used to reconstruct synthetic timestamps.
    imu_only_freq_ms: interval between records when timestamps are absent (ms).

    Returns
    -------
    df   : cleaned DataFrame with normalised columns and SI-unit IMU columns
    meta : dict with keys:
             has_gps         bool
             has_timestamp   bool
             timestamp_reconstructed bool
             file_name       str
             n_rows          int
             duration_s      float | None
             session_start_utc str | None
             session_end_utc   str | None
             available_insights list[str]
             unavailable_insights list[str]
    """
    path = Path(path)
    df = pd.read_csv(path)
    df = _normalise_columns(df)

    # ── Timestamp ──────────────────────────────────────────────────────────
    ts_reconstructed = False
    has_timestamp = False

    if "timestamp" in df.columns:
        # Check if the column is all-zero / all-null (file 2 case)
        ts_col = df["timestamp"]
        if pd.api.types.is_numeric_dtype(ts_col) and (ts_col == 0).all():
            df.drop(columns=["timestamp"], inplace=True)
        else:
            df = _parse_timestamp(df)
            has_timestamp = True

    if "timestamp" not in df.columns:
        if imu_only_start:
            df = _reconstruct_timestamp(df, imu_only_start, imu_only_freq_ms)
            ts_reconstructed = True
            has_timestamp = True
        else:
            warnings.warn(f"{path.name}: no timestamp found and none supplied. Time-series features unavailable.")

    if has_timestamp:
        df = df.sort_values("timestamp").reset_index(drop=True)
        df = _add_utc_cet(df)

    # ── GPS ────────────────────────────────────────────────────────────────
    has_gps = (
        "latitude" in df.columns
        and "longitude" in df.columns
        and df["latitude"].notna().any()
        and (df["latitude"] != 0).any()
    )

    # ── IMU conversion ─────────────────────────────────────────────────────
    df = _convert_imu(df)

    # ── Speed zones ────────────────────────────────────────────────────────
    if has_gps:
        df = _add_speed_zone(df)

    # ── Duration ───────────────────────────────────────────────────────────
    duration_s = None
    session_start = session_end = None
    if has_timestamp and "ts_utc" in df.columns:
        t0 = df["ts_utc"].iloc[0]
        t1 = df["ts_utc"].iloc[-1]
        duration_s = (t1 - t0).total_seconds()
        session_start = str(t0)
        session_end = str(t1)

    # ── What can we compute? ───────────────────────────────────────────────
    all_insights = [
        "exploratory_analysis",
        "gps_speed_validation",
        "position_heatmap",
        "goalkeeper_clustering",
        "speed_distribution",
        "distance_total",
        "speed_trajectories",
        "outlier_detection",
        "imu_movement_detection",
        "asymmetry_analysis",
        "shot_pass_header_detection",
        "footedness",
        "fatigue_analysis",
    ]
    gps_insights = {
        "gps_speed_validation", "position_heatmap", "goalkeeper_clustering",
        "speed_distribution", "distance_total", "speed_trajectories", "fatigue_analysis",
    }
    ts_insights = {"fatigue_analysis", "imu_movement_detection", "shot_pass_header_detection"}

    available = []
    unavailable = []
    for ins in all_insights:
        needs_gps = ins in gps_insights
        needs_ts = ins in ts_insights
        ok = True
        if needs_gps and not has_gps:
            ok = False
        if needs_ts and not has_timestamp:
            ok = False
        (available if ok else unavailable).append(ins)

    meta = {
        "file_name": path.name,
        "n_rows": len(df),
        "has_gps": has_gps,
        "has_timestamp": has_timestamp,
        "timestamp_reconstructed": ts_reconstructed,
        "duration_s": duration_s,
        "session_start_utc": session_start,
        "session_end_utc": session_end,
        "available_insights": available,
        "unavailable_insights": unavailable,
    }

    return df, meta
