"""
gk_analysis.py
--------------
Goalkeeper role detection via 3-minute window aggregation.

Two approaches:
  1. Rule-based  — explicit positional + speed thresholds
  2. KMeans (k=2) — unsupervised on position + mobility features

Field orientation assumed for File 1:
  East (high Longitude / high x_m) = OUR GOAL
  West (low Longitude  / low x_m)  = Opponent goal
  x_m = distance from western edge (opponent side)
  y_m = distance from southern edge

Grid layout (5 rows × 3 cols):
  Row 0 = Our Goal zone (Eastern strip)   ← bottom of vertical display
  Row 4 = Opponent zone (Western strip)   ← top of vertical display
  Col 0 = South sideline
  Col 1 = Centre
  Col 2 = North sideline
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from math import radians, cos
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WINDOW_SECONDS = 180          # 3-minute aggregation windows
N_GRID_COLS = 3               # across width  (S / Centre / N)
N_GRID_ROWS = 5               # along length  (our goal → opponent)

# Rule-based thresholds (tunable)
GK_X_FRACTION = 0.55          # our goal side = x_m > this fraction of field length
GK_Y_FRACTION_LOW = 0.20      # central band lower bound (fraction of width)
GK_Y_FRACTION_HIGH = 0.80     # central band upper bound
GK_SPEED_THRESHOLD_MS = 0.9   # mean speed (m/s) threshold — GK is slow


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------
def add_metric_coords(df: pd.DataFrame) -> tuple[pd.DataFrame, float, float]:
    """
    Add x_m (W→E, opponent→our goal) and y_m (S→N) columns in metres.
    Returns (df, x_max, y_max).
    """
    lat0 = df["latitude"].mean()
    lon_m_per_deg = 111_319 * cos(radians(lat0))
    lat_m_per_deg = 111_319

    lon_min = df["longitude"].min()
    lat_min = df["latitude"].min()

    df = df.copy()
    df["x_m"] = (df["longitude"] - lon_min) * lon_m_per_deg
    df["y_m"] = (df["latitude"] - lat_min) * lat_m_per_deg

    return df, float(df["x_m"].max()), float(df["y_m"].max())


def add_grid_cells(df: pd.DataFrame, x_max: float, y_max: float) -> pd.DataFrame:
    """
    Assign grid_row (0=our goal, N_GRID_ROWS-1=opponent) and
    grid_col (0=South, N_GRID_COLS-1=North) to every record.
    """
    df = df.copy()
    x_edges = np.linspace(0, x_max, N_GRID_ROWS + 1)
    y_edges = np.linspace(0, y_max, N_GRID_COLS + 1)

    # raw_row: 0=West(opponent), N_GRID_ROWS-1=East(our goal)
    raw_row = pd.cut(
        df["x_m"], bins=x_edges, labels=range(N_GRID_ROWS), include_lowest=True
    ).astype("Int64")

    # flip so row 0 = our goal (East) — bottom of vertical pitch display
    df["grid_row"] = (N_GRID_ROWS - 1) - raw_row
    df["grid_col"] = pd.cut(
        df["y_m"], bins=y_edges, labels=range(N_GRID_COLS), include_lowest=True
    ).astype("Int64")

    return df


# ---------------------------------------------------------------------------
# 3-minute window aggregation
# ---------------------------------------------------------------------------
def build_windows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate records into 3-minute windows.
    Returns one row per window with position, mobility, and grid features.
    """
    df = df.copy()

    if "ts_utc" in df.columns:
        col = df["ts_utc"]
        # Ensure datetime — handle epoch-ms integers or strings
        if not pd.api.types.is_datetime64_any_dtype(col):
            col = pd.to_datetime(col, utc=True, unit="ms", errors="coerce")
            if col.isna().all():
                col = pd.to_datetime(df["ts_utc"], utc=True, errors="coerce")
        df["ts_utc"] = col
        t0 = df["ts_utc"].min()
        elapsed = (df["ts_utc"] - t0).dt.total_seconds()
    else:
        # fallback: assume 0.5 s between rows
        elapsed = pd.Series(range(len(df)), dtype=float) * 0.5
    df["_window"] = (elapsed // WINDOW_SECONDS).astype(int)

    agg = df.groupby("_window").agg(
        mean_x=("x_m", "mean"),
        mean_y=("y_m", "mean"),
        std_x=("x_m", "std"),
        std_y=("y_m", "std"),
        mean_speed=("speed_ms", "mean"),
        max_speed=("speed_ms", "max"),
        pct_slow=("speed_ms", lambda s: (s < 1.5).mean()),   # % standing/walking
        n_records=("x_m", "count"),
    ).reset_index()
    agg.rename(columns={"_window": "window_idx"}, inplace=True)
    agg["window_start_s"] = agg["window_idx"] * WINDOW_SECONDS
    agg["std_x"] = agg["std_x"].fillna(0)
    agg["std_y"] = agg["std_y"].fillna(0)
    return agg


# ---------------------------------------------------------------------------
# Rule-based classification
# ---------------------------------------------------------------------------
def rule_based_gk(windows: pd.DataFrame, x_max: float, y_max: float) -> pd.DataFrame:
    """
    Label each window as GK (True) or Outfield (False) using explicit rules.

    GK criteria:
      - Mean position in our defensive half: x_m > GK_X_FRACTION × x_max
      - Mean position in central corridor: y_m between 20%–80% of width
      - Mean speed below threshold

    All three criteria must hold simultaneously.
    """
    windows = windows.copy()
    windows["rule_gk"] = (
        (windows["mean_x"] > GK_X_FRACTION * x_max)
        & (windows["mean_y"] > GK_Y_FRACTION_LOW * y_max)
        & (windows["mean_y"] < GK_Y_FRACTION_HIGH * y_max)
        & (windows["mean_speed"] < GK_SPEED_THRESHOLD_MS)
    )

    # Detailed breakdown for transparency
    windows["rule_in_our_half"] = windows["mean_x"] > GK_X_FRACTION * x_max
    windows["rule_central_y"]   = (
        (windows["mean_y"] > GK_Y_FRACTION_LOW * y_max)
        & (windows["mean_y"] < GK_Y_FRACTION_HIGH * y_max)
    )
    windows["rule_slow"]        = windows["mean_speed"] < GK_SPEED_THRESHOLD_MS
    return windows


# ---------------------------------------------------------------------------
# Unsupervised clustering
# ---------------------------------------------------------------------------
CLUSTER_FEATURES = ["mean_x", "mean_y", "std_x", "std_y", "mean_speed", "pct_slow"]


def cluster_gk(windows: pd.DataFrame, x_max: float) -> tuple[pd.DataFrame, dict]:
    """
    KMeans (k=2) on positional + mobility features.

    The GK cluster is identified as the one with:
      - Higher mean_x (closer to our goal, East side)
      - Lower mean_speed

    Returns (windows with 'cluster_label' and 'cluster_gk' columns, info dict).
    """
    windows = windows.copy()
    X = windows[CLUSTER_FEATURES].fillna(0).values

    # Need at least 2 windows to run k=2 clustering
    if len(windows) < 2:
        # Fall back: mark all windows by rule_gk if available, else all False
        windows["cluster_label"] = 0
        windows["cluster_gk"] = windows.get("rule_gk", pd.Series([False] * len(windows), index=windows.index))
        info = {
            "gk_cluster_id": None,
            "outfield_cluster_id": None,
            "features_used": CLUSTER_FEATURES,
            "cluster_stats": {},
            "fallback": True,
            "reason": f"Only {len(windows)} window(s) — need ≥ 2 for clustering.",
        }
        return windows, info

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_clusters = min(2, len(windows))
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    labels = km.fit_predict(X_scaled)
    windows["cluster_label"] = labels

    # Identify which cluster is GK
    cluster_stats = (
        windows.groupby("cluster_label")[["mean_x", "mean_speed", "pct_slow"]]
        .mean()
    )

    if n_clusters == 1:
        # Only one cluster — assign it as GK if it looks like GK, else outfield
        gk_cluster = 0
        outfield_cluster = None
    else:
        # GK cluster = higher x (our goal side) + lower speed
        x_norm = (cluster_stats["mean_x"] - cluster_stats["mean_x"].min()) / (
            cluster_stats["mean_x"].max() - cluster_stats["mean_x"].min() + 1e-9
        )
        s_norm = (cluster_stats["mean_speed"] - cluster_stats["mean_speed"].min()) / (
            cluster_stats["mean_speed"].max() - cluster_stats["mean_speed"].min() + 1e-9
        )
        gk_cluster = int((x_norm - s_norm).idxmax())
        outfield_cluster = 1 - gk_cluster

    windows["cluster_gk"] = windows["cluster_label"] == gk_cluster

    # Cluster summaries
    info = {
        "gk_cluster_id": gk_cluster,
        "outfield_cluster_id": outfield_cluster if n_clusters > 1 else None,
        "fallback": False,
        "features_used": CLUSTER_FEATURES,
        "cluster_stats": cluster_stats.to_dict(),
    }

    return windows, info


# ---------------------------------------------------------------------------
# Agreement analysis
# ---------------------------------------------------------------------------
def compare_methods(windows: pd.DataFrame) -> dict:
    """
    Compare rule-based vs clustering GK labels.
    Requires both 'rule_gk' and 'cluster_gk' columns.
    """
    n = len(windows)
    agree = (windows["rule_gk"] == windows["cluster_gk"]).sum()
    both_gk = (windows["rule_gk"] & windows["cluster_gk"]).sum()
    rule_only = (windows["rule_gk"] & ~windows["cluster_gk"]).sum()
    cluster_only = (~windows["rule_gk"] & windows["cluster_gk"]).sum()
    neither = (~windows["rule_gk"] & ~windows["cluster_gk"]).sum()

    return {
        "n_windows": n,
        "agreement_pct": round(agree / n * 100, 1),
        "both_gk": int(both_gk),
        "rule_only_gk": int(rule_only),
        "cluster_only_gk": int(cluster_only),
        "neither_gk": int(neither),
        "rule_total_gk": int(windows["rule_gk"].sum()),
        "cluster_total_gk": int(windows["cluster_gk"].sum()),
    }
