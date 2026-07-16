"""
Module: upload.py
Handles CSV upload, validation, target selection, and problem type detection.
"""

import streamlit as st
import pandas as pd
import numpy as np
from utils.detector import detect_problem_type


MAX_ROWS = 50_000
MAX_COLS = 100


def validate_dataframe(df: pd.DataFrame) -> list[str]:
    errors = []
    if df.shape[0] < 10:
        errors.append("Dataset must have at least 10 rows.")
    if df.shape[1] < 2:
        errors.append("Dataset must have at least 2 columns.")
    if df.shape[0] > MAX_ROWS:
        errors.append(f"Dataset exceeds {MAX_ROWS} rows. Trim it before uploading.")
    if df.shape[1] > MAX_COLS:
        errors.append(f"Dataset exceeds {MAX_COLS} columns.")
    duplicate_pct = df.duplicated().sum() / len(df) * 100
    if duplicate_pct > 80:
        errors.append(f"Over 80% duplicate rows ({duplicate_pct:.1f}%). Check your data.")
    constant_cols = [c for c in df.columns if df[c].nunique() <= 1]
    if len(constant_cols) == df.shape[1]:
        errors.append("All columns are constant. No useful data found.")
    return errors


def render_upload():
    st.title("📂 Step 1: Upload Dataset")
    st.markdown("Upload a structured CSV file. The system will validate it and detect the ML task automatically.")

    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded is None:
        st.info("👆 Upload a CSV to get started. Supported tasks: Classification, Regression, Clustering.")
        with st.expander("📋 Dataset Requirements"):
            st.markdown("""
- **Format:** CSV with headers in the first row
- **Size:** 10 – 50,000 rows, up to 100 columns
- **Types:** Numeric, categorical (string), boolean
- **Target column:** Required for supervised tasks (classification/regression)
- **No support for:** images, text blobs, timestamps as primary features
            """)
        return

    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        return

    errors = validate_dataframe(df)
    if errors:
        for e in errors:
            st.error(f"❌ {e}")
        return

    st.success(f"✅ Loaded {df.shape[0]} rows × {df.shape[1]} columns")

    st.subheader("Preview (first 5 rows)")
    st.dataframe(df.head(), use_container_width=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rows", df.shape[0])
    with col2:
        st.metric("Columns", df.shape[1])
    with col3:
        missing_pct = df.isnull().mean().mean() * 100
        st.metric("Missing Values", f"{missing_pct:.1f}%")

    # Target column selection
    st.subheader("🎯 Target Column Selection")
    st.markdown("Select the column you want to **predict**. Leave blank for clustering (unsupervised).")

    col_options = ["— None (Clustering) —"] + list(df.columns)
    target_sel = st.selectbox("Target Column", col_options)

    if target_sel == "— None (Clustering) —":
        target_col = None
        problem_type = "clustering"
        st.info("🔵 Clustering mode selected — no target column required.")
    else:
        target_col = target_sel
        problem_type = detect_problem_type(df, target_col)
        badge = {
            "classification": "🟢 Classification",
            "regression": "🟠 Regression",
        }.get(problem_type, "❓ Unknown")
        st.success(f"**Auto-detected task:** {badge}")
        with st.expander("Why this detection?"):
            col = df[target_col]
            nuniq = col.nunique()
            dtype = col.dtype
            st.write(f"- dtype: `{dtype}`")
            st.write(f"- unique values: `{nuniq}` out of `{len(col)}`")
            if problem_type == "classification":
                st.write("- Low unique count or non-numeric → Classification")
                st.write(f"- Classes: {sorted(col.dropna().unique().tolist()[:10])}")
            else:
                st.write("- Numeric with high cardinality → Regression")

    # Override option
    if target_col is not None:
        override = st.radio(
            "Override detected task?",
            [f"Use auto-detected: {problem_type.title()}", "Classification", "Regression"],
            horizontal=True,
        )
        if "Classification" in override and "auto" not in override:
            problem_type = "classification"
        elif "Regression" in override and "auto" not in override:
            problem_type = "regression"

    if st.button("✅ Confirm & Continue", type="primary"):
        st.session_state.df_raw = df
        st.session_state.target_col = target_col
        st.session_state.problem_type = problem_type
        feature_cols = [c for c in df.columns if c != target_col] if target_col else list(df.columns)
        st.session_state.feature_cols = feature_cols
        st.success("Dataset saved. Proceed to Data Analysis →")
        st.session_state.step = 1
        st.rerun()
