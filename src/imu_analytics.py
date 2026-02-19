"""
imu_analytics.py
----------------
All analyses that rely on IMU (accelerometer + gyroscope) data.

Unit assumptions (applied by loader.py before these functions run):
  acc_{x,y,z}_g    – acceleration in g
  acc_{x,y,z}_ms2  – acceleration in m/s²
  rot_{x,y,z}_dps  – angular velocity in °/s
  pitch_deg, roll_deg – device angles in degrees

Axis convention assumed (sensor worn on upper back / vest):
  X – lateral (left–right)
  Y – forward–backward (sagittal)
  Z – vertical

These are data-driven heuristics validated against the file structure.
All thresholds are documented and configurable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import find_peaks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _total_acc(df: pd.DataFrame) -> pd.Series:
    """Resultant acceleration magnitude (g)."""
    return np.sqrt(df["acc_x_g"] ** 2 + df["acc_y_g"] ** 2 + df["acc_z_g"] ** 2)


def _total_rot(df: pd.DataFrame) -> pd.Series:
    """Resultant angular velocity magnitude (°/s)."""
    return np.sqrt(df["rot_x_dps"] ** 2 + df["rot_y_dps"] ** 2 + df["rot_z_dps"] ** 2)


# ---------------------------------------------------------------------------
# Outlier detection
# ---------------------------------------------------------------------------
def detect_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flag outliers in acceleration and (if present) speed using IQR method.
    Adds boolean columns: 'outlier_acc', 'outlier_speed'.
    """
    df = df.copy()

    acc_mag = _total_acc(df)
    q1, q3 = acc_mag.quantile(0.25), acc_mag.quantile(0.75)
    iqr = q3 - q1
    df["acc_magnitude_g"] = acc_mag
    df["outlier_acc"] = (acc_mag < q1 - 3 * iqr) | (acc_mag > q3 + 3 * iqr)

    if "speed_ms" in df.columns and df["speed_ms"].notna().any():
        s = df["speed_ms"]
        q1s, q3s = s.quantile(0.25), s.quantile(0.75)
        iqrs = q3s - q1s
        df["outlier_speed"] = (s < q1s - 3 * iqrs) | (s > q3s + 3 * iqrs)

    return df


# ---------------------------------------------------------------------------
# Special movement detection
# ---------------------------------------------------------------------------
TWIST_ROT_THRESHOLD_DPS = 100      # min rotation magnitude for a twist
LEAN_ROLL_THRESHOLD_DEG = 15       # min |roll| for a lean
TURN_YAW_THRESHOLD_DPS = 80        # min rotZ magnitude for a turn
MIN_PEAK_DISTANCE_SAMPLES = 3      # min samples between events (~1.5 s at 2 Hz)


def detect_special_movements(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect twists, leans and turns using IMU data.

    Appends boolean columns:
      is_twist  – high resultant rotation
      is_lean   – large lateral roll angle
      is_turn   – large yaw angular velocity

    NOTE: These are conservative heuristics intended as starting points
    for manual validation against video.
    """
    df = df.copy()
    df["rot_magnitude_dps"] = _total_rot(df)

    # Twist: sudden high total rotation
    df["is_twist"] = df["rot_magnitude_dps"] > TWIST_ROT_THRESHOLD_DPS

    # Lean: significant roll angle
    if "roll_deg" in df.columns:
        df["is_lean"] = df["roll_deg"].abs() > LEAN_ROLL_THRESHOLD_DEG
    else:
        df["is_lean"] = False

    # Turn: strong yaw (Z-axis rotation for vertical-axis sensor)
    if "rot_z_dps" in df.columns:
        df["is_turn"] = df["rot_z_dps"].abs() > TURN_YAW_THRESHOLD_DPS
    else:
        df["is_turn"] = False

    return df


# ---------------------------------------------------------------------------
# Asymmetry analysis
# ---------------------------------------------------------------------------
def asymmetry_analysis(df: pd.DataFrame) -> dict:
    """
    Compute left–right asymmetry in lateral acceleration and rotation.

    Returns dict with:
      acc_x_mean_left  – mean lateral acc on left-dominant side (g)
      acc_x_mean_right – mean lateral acc on right-dominant side (g)
      asymmetry_index  – (|left| - |right|) / (|left| + |right|) × 100  (%)
      dominant_side    – 'left' | 'right' | 'symmetric'
    """
    if "acc_x_g" not in df.columns:
        return {}

    acc_x = df["acc_x_g"].dropna()
    rot_x = df["rot_x_dps"].dropna() if "rot_x_dps" in df.columns else pd.Series(dtype=float)

    left_acc = acc_x[acc_x < 0].abs().mean()   # negative X = left
    right_acc = acc_x[acc_x > 0].abs().mean()  # positive X = right

    if np.isnan(left_acc) or np.isnan(right_acc) or (left_acc + right_acc) == 0:
        asymmetry_idx = 0.0
    else:
        asymmetry_idx = (left_acc - right_acc) / (left_acc + right_acc) * 100

    dominant = (
        "left" if asymmetry_idx > 5
        else "right" if asymmetry_idx < -5
        else "symmetric"
    )

    result = {
        "acc_x_mean_left_g": round(float(left_acc), 4),
        "acc_x_mean_right_g": round(float(right_acc), 4),
        "asymmetry_index_pct": round(float(asymmetry_idx), 2),
        "dominant_side": dominant,
        "n_left_samples": int((acc_x < 0).sum()),
        "n_right_samples": int((acc_x > 0).sum()),
    }

    if len(rot_x) > 0:
        result["rot_x_mean_left_dps"] = round(float(rot_x[rot_x < 0].abs().mean()), 4)
        result["rot_x_mean_right_dps"] = round(float(rot_x[rot_x > 0].abs().mean()), 4)

    return result


# ---------------------------------------------------------------------------
# Shot / pass / header detection
# ---------------------------------------------------------------------------
# Thresholds derived from sports science literature on wearable IMU analysis.
# They are intentionally conservative to reduce false positives.
SHOT_ACC_THRESHOLD_G = 2.5       # large resultant acc spike
SHOT_ROT_THRESHOLD_DPS = 120     # accompanied by high rotation
HEADER_VERTICAL_ACC_G = 2.0      # large Z-axis impulse
HEADER_PITCH_CHANGE_DEG = 20     # accompanied by pitch change
PASS_ACC_THRESHOLD_G = 1.5       # moderate acc spike (lighter than shot)


def detect_action_events(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect probable shots, passes and headers from IMU peak analysis.

    Appends columns:
      acc_magnitude_g  (if not already present)
      event_shot       bool
      event_pass       bool
      event_header     bool

    IMPORTANT: these are signal-based heuristics. Manual video validation
    is strongly recommended before drawing conclusions.
    """
    df = df.copy()

    if "acc_magnitude_g" not in df.columns:
        df["acc_magnitude_g"] = _total_acc(df)
    if "rot_magnitude_dps" not in df.columns:
        df["rot_magnitude_dps"] = _total_rot(df)

    acc = df["acc_magnitude_g"].fillna(0).values
    rot = df["rot_magnitude_dps"].fillna(0).values

    # --- Shots: high acc + high rot ---
    shot_peaks, _ = find_peaks(acc, height=SHOT_ACC_THRESHOLD_G, distance=MIN_PEAK_DISTANCE_SAMPLES)
    shot_mask = np.zeros(len(df), dtype=bool)
    for idx in shot_peaks:
        if rot[idx] > SHOT_ROT_THRESHOLD_DPS:
            shot_mask[idx] = True
    df["event_shot"] = shot_mask

    # --- Headers: high Z-acc + pitch change ---
    if "acc_z_g" in df.columns and "pitch_deg" in df.columns:
        z_acc = df["acc_z_g"].fillna(0).values
        pitch = df["pitch_deg"].fillna(0).values
        pitch_change = np.abs(np.gradient(pitch))
        header_peaks, _ = find_peaks(
            np.abs(z_acc), height=HEADER_VERTICAL_ACC_G, distance=MIN_PEAK_DISTANCE_SAMPLES
        )
        header_mask = np.zeros(len(df), dtype=bool)
        for idx in header_peaks:
            if pitch_change[idx] > HEADER_PITCH_CHANGE_DEG:
                header_mask[idx] = True
        df["event_header"] = header_mask
    else:
        df["event_header"] = False

    # --- Passes: moderate acc spike, not already a shot ---
    pass_peaks, _ = find_peaks(acc, height=PASS_ACC_THRESHOLD_G, distance=MIN_PEAK_DISTANCE_SAMPLES)
    pass_mask = np.zeros(len(df), dtype=bool)
    for idx in pass_peaks:
        if not shot_mask[idx] and acc[idx] < SHOT_ACC_THRESHOLD_G:
            pass_mask[idx] = True
    df["event_pass"] = pass_mask

    return df


# ---------------------------------------------------------------------------
# Footedness / versatility
# ---------------------------------------------------------------------------
def estimate_footedness(df: pd.DataFrame) -> dict:
    """
    Heuristic foot dominance from lateral acceleration asymmetry during
    high-intensity movement events (shots + passes).

    Uses the sign of lateral acc (X-axis) at detected action events.
    Left foot kicks tend to generate negative X impulse for sensor on
    the back (and vice versa for right foot) — this is an assumption
    that should be validated with ground-truth labels.
    """
    if "event_shot" not in df.columns:
        df = detect_action_events(df)
    if "acc_x_g" not in df.columns:
        return {}

    events = df[df["event_shot"] | df["event_pass"]]
    if events.empty:
        return {"note": "No action events detected to estimate footedness."}

    left_kicks = (events["acc_x_g"] < 0).sum()
    right_kicks = (events["acc_x_g"] > 0).sum()
    total = left_kicks + right_kicks

    return {
        "estimated_left_foot_actions": int(left_kicks),
        "estimated_right_foot_actions": int(right_kicks),
        "left_pct": round(left_kicks / total * 100, 1) if total > 0 else None,
        "right_pct": round(right_kicks / total * 100, 1) if total > 0 else None,
        "likely_dominant_foot": (
            "left" if left_kicks > right_kicks else "right" if right_kicks > left_kicks else "ambidextrous"
        ),
        "note": "Heuristic estimate — validate against video.",
    }


# ---------------------------------------------------------------------------
# Fatigue analysis (requires timestamp)
# ---------------------------------------------------------------------------
def fatigue_analysis(df: pd.DataFrame, window_minutes: int = 10) -> pd.DataFrame:
    """
    Track drop in peak speed over rolling time windows to proxy fatigue.

    Returns a DataFrame with columns:
      window_start, window_end, peak_speed_ms, peak_speed_kmh, mean_speed_ms
    """
    if "ts_utc" not in df.columns or "speed_ms" not in df.columns:
        return pd.DataFrame()

    df = df.copy()
    df = df.set_index("ts_utc").sort_index()

    rule = f"{window_minutes}min"
    grouped = df["speed_ms"].resample(rule)

    rows = []
    for window_start, group in grouped:
        if group.empty:
            continue
        rows.append(
            {
                "window_start": window_start,
                "window_end": window_start + pd.Timedelta(minutes=window_minutes),
                "peak_speed_ms": group.max(),
                "peak_speed_kmh": round(group.max() * 3.6, 2),
                "mean_speed_ms": round(group.mean(), 4),
                "mean_speed_kmh": round(group.mean() * 3.6, 2),
                "n_samples": len(group),
            }
        )

    return pd.DataFrame(rows)
