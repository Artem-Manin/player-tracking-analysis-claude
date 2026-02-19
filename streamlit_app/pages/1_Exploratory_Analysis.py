"""Page 1 — Exploratory Analysis"""
import importlib.util
from pathlib import Path

import pandas as pd
import streamlit as st

# Load shared helpers by absolute path (CWD-independent)
_shared = importlib.util.module_from_spec(
    s := importlib.util.spec_from_file_location("shared", Path(__file__).resolve().parent / "shared.py")
)
s.loader.exec_module(_shared)
ROOT = _shared.ROOT
require_session, get_df, get_meta = _shared.require_session, _shared.get_df, _shared.get_meta
DARK_CSS = _shared.DARK_CSS

st.set_page_config(page_title="Exploratory Analysis", page_icon="🔍", layout="wide")
st.markdown(DARK_CSS, unsafe_allow_html=True)
st.markdown("# 🔍 Exploratory Analysis")
st.markdown("Basic profile of each loaded file: shape, columns, types, and sample data.")
st.divider()


def show_eda(df, meta, label):
    st.markdown(f"## {label} — `{meta.get('file_name', '')}`")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", f"{len(df):,}")
    col2.metric("Columns", f"{len(df.columns)}")
    col3.metric("Has GPS", "✓" if meta.get("has_gps") else "✗")
    col4.metric("Has Timestamp", "✓" if meta.get("has_timestamp") else "✗")

    st.markdown("### Columns & Data Types")
    dtype_df = pd.DataFrame({"Column": df.columns, "Type": [str(t) for t in df.dtypes]})
    st.dataframe(dtype_df, use_container_width=True, hide_index=True)

    st.markdown("### Sample Data (first 10 rows)")
    display_cols = [c for c in df.columns if not c.endswith("_raw")]
    st.dataframe(df[display_cols].head(10), use_container_width=True)

    st.markdown("### Descriptive Statistics")
    num_cols = [c for c in df.select_dtypes(include="number").columns if not c.endswith("_raw")]
    if num_cols:
        st.dataframe(df[num_cols].describe().T.round(4), use_container_width=True)

    st.markdown("### Missing Values")
    null_counts = df.isnull().sum()
    null_counts = null_counts[null_counts > 0]
    if null_counts.empty:
        st.success("No missing values found.")
    else:
        null_df = null_counts.reset_index()
        null_df.columns = ["Column", "Null Count"]
        null_df["% Missing"] = (null_df["Null Count"] / len(df) * 100).round(2)
        st.dataframe(null_df, use_container_width=True, hide_index=True)

    if meta.get("timestamp_reconstructed"):
        st.info(
            "⏱️ Timestamps in this file were **reconstructed** synthetically "
            f"(500 ms intervals starting {meta.get('session_start_utc', 'N/A')} UTC).",
            icon="ℹ️",
        )
    st.divider()


if "df1" in st.session_state:
    show_eda(get_df("df1"), get_meta("meta1"), "File 1")
if "df2" in st.session_state:
    show_eda(get_df("df2"), get_meta("meta2"), "File 2")
if "df1" not in st.session_state and "df2" not in st.session_state:
    require_session("df1")
