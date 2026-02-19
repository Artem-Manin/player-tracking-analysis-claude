# ⚽ Football Player Tracker

A Python analytics platform for football (soccer) player tracking data from GPS + IMU wearable sensor devices. Includes a Streamlit dashboard, reusable analytics modules, and a Jupyter notebook for exploratory analysis.

---

## 📁 Repository Structure

```
football-tracker/
├── data/                          # Source CSV files (not tracked if sensitive)
│   ├── new_player_data_2026_02_06_174048.csv     # GPS + IMU
│   └── player_activity_imu_2026_02_16.csv        # IMU-only
├── src/
│   ├── loader.py                  # Generic file loader & normaliser
│   ├── gps_analytics.py           # GPS-based analyses
│   ├── imu_analytics.py           # IMU-based analyses
│   └── plots.py                   # Plotly figure factories
├── notebooks/
│   └── exploratory_analysis.ipynb # Step-by-step EDA notebook
├── streamlit_app/
│   ├── Home.py                    # Main page (file upload + session summary)
│   └── pages/
│       ├── 1_Exploratory_Analysis.py
│       ├── 2_GPS_Speed_Validation.py
│       ├── 3_Position_Heatmap.py
│       ├── 4_Speed_Distribution.py
│       ├── 5_IMU_Movements.py
│       ├── 6_Action_Events.py
│       └── 7_Asymmetry_Fatigue.py
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run the Streamlit app

```bash
streamlit run streamlit_app/Home.py
```

Then open [http://localhost:8501](http://localhost:8501).

**Option A — Upload your own files** via the sidebar file uploader.  
**Option B — Click "Load bundled demo files"** to analyse the two included sessions.

---

## 📊 Insights & Pages

| Page | Insight | Requires GPS | Requires Timestamp |
|------|---------|:------------:|:------------------:|
| 🔍 Exploratory Analysis | Shape, columns, stats, nulls | — | — |
| 📡 GPS Speed Validation | Haversine vs device speed | ✓ | ✓ |
| 🗺️ Position Heatmap | Density map + GK clustering | ✓ | — |
| ⚡ Speed Distribution | UEFA/EPTS speed zones | ✓ | — |
| 🔄 Movement Detection | Twists, leans, turns | — | — |
| 🦵 Action Events | Shots, passes, headers, footedness | — | — |
| ⚖️ Asymmetry & Fatigue | L/R asymmetry + peak speed drop | — | fatigue only |

If a file lacks the required data, the page shows a clear explanation rather than an error.

---

## 🔧 Supported File Formats

The loader handles **any CSV** from this device family, with or without GPS, automatically:

- Normalises column names (case, underscores, `Count` vs `cnt`, etc.)
- Parses timestamps in `DD.MM.YYYY HH:MM:SS.mmm` format or Unix epoch
- **Reconstructs timestamps** for IMU-only files with no valid timestamp — supply a start time and 500 ms interval
- Converts raw IMU integers to SI units (see below)

---

## 📐 IMU Unit Conversion

| Raw column | Canonical column | Scale | Unit |
|-----------|-----------------|-------|------|
| `AccX/Y/Z` | `acc_x/y/z_g` | ÷ 8 192 | g (±4 g range) |
| `AccX/Y/Z` | `acc_x/y/z_ms2` | × 9.80665 | m/s² |
| `RotX/Y/Z` | `rot_x/y/z_dps` | ÷ 131 | °/s (±250 dps range) |
| `Temp` | `temp_c` | ÷ 256 | °C |
| `Pitch/Roll` | `pitch_deg/roll_deg` | as-is | ° |

These scales match a **16-bit MPU-6000 family sensor** at ±4 g / ±250 °/s ranges.  
Validated empirically: temperature reads −22 to −1 °C for a February 2026 Vienna session ✓.

---

## 📍 GPS Notes

- **File 1** (`new_player_data_2026_02_06_174048.csv`): full GPS + IMU, session 2026-02-04 17:31–18:55 CET.
- **File 2** (`player_activity_imu_2026_02_16.csv`): IMU-only; latitude/longitude/speed/epoch_time are all zero. Timestamps are reconstructed from 2026-02-16 19:30 CET at 500 ms intervals.

The GPS bounding box for File 1 is approximately **18 m × 35 m** — consistent with a goalkeeping or set-piece drill, not full-pitch tracking.

---

## ⚠️ Caveats & Honest Limitations

- **Action event detection** (shots, passes, headers) and **footedness** are conservative signal-based heuristics. They require video validation before being used as ground truth.
- **Goalkeeper clustering** uses a heuristic score (low speed + proximity to goal-line). With a bounding box of only 18 × 35 m, cluster spatial separation is small — treat as micro-zone analysis.
- **Asymmetry analysis** assumes X-axis = lateral direction with sensor worn on the upper back / vest. Verify sensor orientation before interpreting left/right results.
- **IMU scaling factors** are assumed from MPU-6000 family standard ranges. If the device uses different full-scale settings, recalibrate the constants in `src/loader.py`.

---

## 🧩 Extending the Platform

To add a new insight:
1. Add an analytics function to `src/gps_analytics.py` or `src/imu_analytics.py`
2. Add a figure factory to `src/plots.py`
3. Create a new page in `streamlit_app/pages/`
4. Update the `available_insights` logic in `src/loader.py` if the insight has data prerequisites

---

## 🐍 Python Version

Tested with Python 3.10+.
