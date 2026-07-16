"""
Module: eda.py
Exploratory Data Analysis — distributions, correlations, missing value map.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def render_eda():
    st.title("🔍 Step 2: Exploratory Data Analysis")

    df = st.session_state.df_raw
    target = st.session_state.target_col
    problem_type = st.session_state.problem_type

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    # ── Overview ──────────────────────────────────────────────────────────────
    st.subheader("📋 Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Rows", df.shape[0])
    col2.metric("Total Columns", df.shape[1])
    col3.metric("Numeric Cols", len(numeric_cols))
    col4.metric("Categorical Cols", len(categorical_cols))

    with st.expander("📊 Full Descriptive Statistics"):
        st.dataframe(df.describe(include="all").T, use_container_width=True)

    # ── Missing values ─────────────────────────────────────────────────────────
    st.subheader("🕳️ Missing Values")
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if missing.empty:
        st.success("No missing values found.")
    else:
        missing_df = pd.DataFrame({
            "Column": missing.index,
            "Missing Count": missing.values,
            "Missing %": (missing.values / len(df) * 100).round(2),
        })
        st.dataframe(missing_df, use_container_width=True)
        fig = px.bar(
            missing_df, x="Column", y="Missing %",
            title="Missing Values by Column (%)",
            color="Missing %", color_continuous_scale="Reds",
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Distribution of numeric columns ───────────────────────────────────────
    st.subheader("📈 Numeric Column Distributions")
    if numeric_cols:
        show_cols = st.multiselect(
            "Select columns to plot", numeric_cols,
            default=numeric_cols[:min(4, len(numeric_cols))],
        )
        if show_cols:
            rows = (len(show_cols) + 1) // 2
            fig = make_subplots(rows=rows, cols=2, subplot_titles=show_cols)
            for i, col in enumerate(show_cols):
                r, c = divmod(i, 2)
                fig.add_trace(
                    go.Histogram(x=df[col].dropna(), name=col, showlegend=False),
                    row=r + 1, col=c + 1,
                )
            fig.update_layout(height=300 * rows, title_text="Histograms")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No numeric columns found.")

    # ── Categorical distributions ──────────────────────────────────────────────
    if categorical_cols:
        st.subheader("📊 Categorical Column Distributions")
        cat_sel = st.selectbox("Select categorical column", categorical_cols)
        vc = df[cat_sel].value_counts().reset_index()
        vc.columns = [cat_sel, "count"]
        if len(vc) <= 30:
            fig = px.bar(vc, x=cat_sel, y="count", title=f"Value Counts: {cat_sel}")
        else:
            fig = px.bar(vc.head(30), x=cat_sel, y="count", title=f"Top 30 Values: {cat_sel}")
            st.info("Showing top 30 categories only.")
        st.plotly_chart(fig, use_container_width=True)

    # ── Target analysis ────────────────────────────────────────────────────────
    if target:
        st.subheader(f"🎯 Target Column: `{target}`")
        if problem_type == "classification":
            vc = df[target].value_counts().reset_index()
            vc.columns = [target, "count"]
            fig = px.pie(vc, names=target, values="count", title="Class Distribution")
            st.plotly_chart(fig, use_container_width=True)
            # Class imbalance warning
            max_pct = vc["count"].max() / vc["count"].sum() * 100
            if max_pct > 80:
                st.warning(f"⚠️ Class imbalance detected: dominant class = {max_pct:.1f}%. Consider resampling.")
        else:
            fig = px.histogram(df, x=target, title=f"Distribution of {target}", nbins=40)
            st.plotly_chart(fig, use_container_width=True)

    # ── Correlation matrix ─────────────────────────────────────────────────────
    if len(numeric_cols) >= 2:
        st.subheader("🔗 Correlation Matrix")
        corr_cols = numeric_cols[:20]  # cap at 20 for readability
        corr = df[corr_cols].corr()
        fig = px.imshow(
            corr, text_auto=".2f", aspect="auto",
            color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
            title="Pearson Correlation (capped at 20 numeric cols)",
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

        # Highlight highly correlated pairs
        high_corr = []
        for i in range(len(corr.columns)):
            for j in range(i + 1, len(corr.columns)):
                v = abs(corr.iloc[i, j])
                if v > 0.85:
                    high_corr.append((corr.columns[i], corr.columns[j], round(v, 3)))
        if high_corr:
            st.warning(f"⚠️ {len(high_corr)} highly correlated pairs (|r| > 0.85) detected.")
            st.dataframe(
                pd.DataFrame(high_corr, columns=["Col A", "Col B", "|r|"]),
                use_container_width=True,
            )

    # ── Outlier summary ────────────────────────────────────────────────────────
    if numeric_cols:
        st.subheader("⚠️ Outlier Summary (IQR method)")
        outlier_rows = []
        for col in numeric_cols:
            s = df[col].dropna()
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            n_out = ((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum()
            if n_out > 0:
                outlier_rows.append({"Column": col, "Outliers": n_out, "% of rows": round(n_out / len(df) * 100, 2)})
        if outlier_rows:
            st.dataframe(pd.DataFrame(outlier_rows), use_container_width=True)
        else:
            st.success("No significant outliers detected.")
