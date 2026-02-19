"""
gps_analytics.py
----------------
All analyses that require GPS latitude / longitude data.

Functions
---------
haversine_distance(df)         → df with 'dist_m' and 'speed_haversine_ms' cols
gps_speed_validation(df)       → metrics dict + annotated df
total_distance(df)             → float (metres)
speed_distribution(df)         → zone counts Series
goalkeeper_clustering(df, n)   → df with 'gk_cluster' column + cluster summary
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from math import radians, sin, cos, sqrt, atan2
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Haversine helper
# ---------------------------------------------------------------------------
def _hav(lat1, lon1, lat2, lon2) -> float:
    R = 6_371_000.0
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlam = radians(lon2 - lon1)
    a = sin(dphi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(dlam / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


def haversine_distance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 'dist_m' (step distance) and 'speed_haversine_ms' (recomputed speed)
    columns to the DataFrame.  Requires latitude, longitude, timestamp cols.
    """
    lats = df["latitude"].values
    lons = df["longitude"].values

    dist = np.zeros(len(df))
    for i in range(1, len(df)):
        dist[i] = _hav(lats[i - 1], lons[i - 1], lats[i], lons[i])
    df = df.copy()
    df["dist_m"] = dist

    if "ts_utc" in df.columns:
        dt = df["ts_utc"].diff().dt.total_seconds().values
        with np.errstate(divide="ignore", invalid="ignore"):
            speed_hav = np.where(dt > 0, dist / dt, np.nan)
        df["speed_haversine_ms"] = speed_hav

    return df


# ---------------------------------------------------------------------------
# GPS speed validation
# ---------------------------------------------------------------------------
def gps_speed_validation(df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """
    Compare device-reported speed with Haversine-recomputed speed.

    Returns
    -------
    metrics : dict with mae, rmse, correlation, n_flagged
    df_val  : df with error columns and 'flagged' bool column
    """
    df = haversine_distance(df)

    if "speed_haversine_ms" not in df.columns or "speed_ms" not in df.columns:
        return {}, df

    valid = df[["speed_ms", "speed_haversine_ms"]].dropna()
    error = valid["speed_ms"] - valid["speed_haversine_ms"]

    mae = error.abs().mean()
    rmse = float(np.sqrt((error ** 2).mean()))
    corr = valid["speed_ms"].corr(valid["speed_haversine_ms"])

    # Flag rows where absolute error exceeds 2× RMSE
    threshold = 2 * rmse
    df = df.copy()
    df["speed_error_ms"] = df["speed_ms"] - df["speed_haversine_ms"]
    df["speed_flagged"] = df["speed_error_ms"].abs() > threshold

    n_flagged = int(df["speed_flagged"].sum())

    metrics = {
        "mae_ms": round(mae, 4),
        "rmse_ms": round(rmse, 4),
        "correlation": round(corr, 4),
        "n_flagged": n_flagged,
        "flag_threshold_ms": round(threshold, 4),
        "n_valid_rows": len(valid),
    }

    return metrics, df


# ---------------------------------------------------------------------------
# Total distance
# ---------------------------------------------------------------------------
def total_distance(df: pd.DataFrame) -> float:
    """Return total distance covered in metres (requires dist_m column)."""
    if "dist_m" not in df.columns:
        df = haversine_distance(df)
    return float(df["dist_m"].sum())


# ---------------------------------------------------------------------------
# Speed distribution by zone
# ---------------------------------------------------------------------------
def speed_zone_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Return counts and % time in each speed zone."""
    if "speed_zone" not in df.columns:
        return pd.DataFrame()

    counts = df["speed_zone"].value_counts().rename("count")
    pct = (counts / counts.sum() * 100).round(1).rename("pct")
    result = pd.concat([counts, pct], axis=1).reset_index()
    result.columns = ["zone", "count", "pct"]

    # Preserve natural zone order
    from src.loader import SPEED_LABELS
    result["zone"] = pd.Categorical(result["zone"], categories=SPEED_LABELS, ordered=True)
    return result.sort_values("zone").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Goalkeeper clustering
# ---------------------------------------------------------------------------
def goalkeeper_clustering(df: pd.DataFrame, n_clusters: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Cluster player positions using KMeans on lat/lon.

    The goalkeeper cluster is heuristically identified as the cluster
    with the lowest mean speed and located closest to one of the
    two goal-line extremes (min or max latitude in the data).

    Returns
    -------
    df         : original df with 'position_cluster' int column added
    cluster_summary : DataFrame describing each cluster
    """
    pos = df[["latitude", "longitude"]].dropna().copy()

    # Convert to metres relative to centre for meaningful distances
    lat0 = pos["latitude"].mean()
    lon0 = pos["longitude"].mean()
    pos["x_m"] = (pos["longitude"] - lon0) * 111_319 * np.cos(np.radians(lat0))
    pos["y_m"] = (pos["latitude"] - lat0) * 111_319

    scaler = StandardScaler()
    X = scaler.fit_transform(pos[["x_m", "y_m"]])

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    pos["position_cluster"] = labels

    df = df.copy()
    df["position_cluster"] = np.nan
    df.loc[pos.index, "position_cluster"] = labels.astype(float)
    df["position_cluster"] = df["position_cluster"].astype("Int64")

    # Build summary
    summary_rows = []
    for c in range(n_clusters):
        mask = df["position_cluster"] == c
        sub = df[mask]
        row = {
            "cluster": c,
            "n_points": int(mask.sum()),
            "pct_time": round(mask.sum() / len(df) * 100, 1),
            "mean_lat": round(sub["latitude"].mean(), 6),
            "mean_lon": round(sub["longitude"].mean(), 6),
        }
        if "speed_ms" in sub.columns:
            row["mean_speed_ms"] = round(sub["speed_ms"].mean(), 3)
            row["mean_speed_kmh"] = round(sub["speed_ms"].mean() * 3.6, 2)
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)

    # Tag likely goalkeeper cluster: lowest speed + close to goal-line
    if "mean_speed_ms" in summary.columns:
        lat_extremes = [df["latitude"].min(), df["latitude"].max()]

        def dist_to_goalline(row):
            return min(abs(row["mean_lat"] - e) for e in lat_extremes)

        summary["dist_to_goalline_deg"] = summary.apply(dist_to_goalline, axis=1)
        # Score: lower speed + closer to goal line = more GK-like
        speed_score = (summary["mean_speed_ms"] - summary["mean_speed_ms"].min()) / (
            summary["mean_speed_ms"].max() - summary["mean_speed_ms"].min() + 1e-9
        )
        gline_score = (summary["dist_to_goalline_deg"] - summary["dist_to_goalline_deg"].min()) / (
            summary["dist_to_goalline_deg"].max() - summary["dist_to_goalline_deg"].min() + 1e-9
        )
        summary["gk_score"] = speed_score + gline_score  # lower = more GK-like
        summary["likely_gk"] = summary["gk_score"] == summary["gk_score"].min()

    return df, summary
