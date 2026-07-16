"""
Module: preprocessing.py
Missing value handling, encoding, scaling, feature selection.
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, mutual_info_classif
import plotly.express as px


def _handle_missing(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    df = df.copy()
    strategy = config.get("missing_strategy", "median")
    for col in df.columns:
        if df[col].isnull().sum() == 0:
            continue
        if df[col].dtype in [np.float64, np.int64, float, int]:
            if strategy == "mean":
                df[col].fillna(df[col].mean(), inplace=True)
            elif strategy == "median":
                df[col].fillna(df[col].median(), inplace=True)
            elif strategy == "drop_rows":
                df.dropna(subset=[col], inplace=True)
            else:
                df[col].fillna(df[col].median(), inplace=True)
        else:
            if strategy == "drop_rows":
                df.dropna(subset=[col], inplace=True)
            else:
                mode_val = df[col].mode()
                df[col].fillna(mode_val[0] if len(mode_val) > 0 else "Unknown", inplace=True)
    return df


def _encode_categoricals(df: pd.DataFrame, target: str | None, config: dict) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    encoders = {}
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    enc_strategy = config.get("encoding", "label")

    for col in cat_cols:
        if col == target:
            if target and df[target].dtype == object:
                le = LabelEncoder()
                df[target] = le.fit_transform(df[target].astype(str))
                encoders[target] = le
            continue
        if enc_strategy == "onehot" and df[col].nunique() <= 15:
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
        else:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le

    return df, encoders


def _scale_features(X: pd.DataFrame, method: str) -> tuple[pd.DataFrame, object]:
    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    else:
        return X, None
    cols = X.columns.tolist()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=cols)
    return X_scaled, scaler


def _select_features(X: pd.DataFrame, y: pd.Series, problem_type: str, k: int) -> list[str]:
    k = min(k, X.shape[1])
    try:
        if problem_type == "classification":
            selector = SelectKBest(f_classif, k=k)
        else:
            selector = SelectKBest(f_regression, k=k)
        selector.fit(X.fillna(0), y)
        mask = selector.get_support()
        return X.columns[mask].tolist()
    except Exception:
        return X.columns.tolist()[:k]


def render_preprocessing():
    st.title("🧹 Step 3: Preprocessing")

    df = st.session_state.df_raw.copy()
    target = st.session_state.target_col
    problem_type = st.session_state.problem_type
    feature_cols = st.session_state.feature_cols

    missing_pct = df[feature_cols].isnull().mean().mean() * 100
    st.markdown(f"**Missing values in features:** `{missing_pct:.1f}%`")

    # ── Configuration ──────────────────────────────────────────────────────────
    st.subheader("⚙️ Configure Preprocessing")

    col1, col2, col3 = st.columns(3)
    with col1:
        missing_strategy = st.selectbox(
            "Missing Value Strategy",
            ["median", "mean", "drop_rows"],
            help="median/mean fills numerics; mode fills categoricals; drop_rows removes them.",
        )
    with col2:
        encoding = st.selectbox(
            "Categorical Encoding",
            ["label", "onehot"],
            help="Label: integer codes (fast). One-hot: binary columns (max 15 unique vals).",
        )
    with col3:
        scaling = st.selectbox(
            "Feature Scaling",
            ["standard", "minmax", "none"],
            help="StandardScaler (zero mean), MinMaxScaler (0–1), or none.",
        )

    # Feature selection (supervised only)
    do_feature_sel = False
    k_features = len(feature_cols)
    if problem_type in ("classification", "regression") and len(feature_cols) > 5:
        st.subheader("🔬 Feature Selection")
        do_feature_sel = st.checkbox(
            "Apply automatic feature selection (SelectKBest)", value=True
        )
        if do_feature_sel:
            k_features = st.slider(
                "Number of top features to keep",
                min_value=2,
                max_value=min(len(feature_cols), 30),
                value=min(len(feature_cols), 10),
            )

    # Remove ID-like columns
    st.subheader("🗑️ Drop Columns")
    id_candidates = [c for c in feature_cols if df[c].nunique() == len(df)]
    if id_candidates:
        st.warning(f"These columns have all unique values (likely IDs): `{id_candidates}`")
    drop_cols = st.multiselect(
        "Manually drop columns (optional)", feature_cols, default=id_candidates
    )

    if st.button("🚀 Run Preprocessing", type="primary"):
        with st.spinner("Preprocessing..."):
            config = {
                "missing_strategy": missing_strategy,
                "encoding": encoding,
                "scaling": scaling,
            }

            # Drop columns
            df_work = df.copy()
            if drop_cols:
                df_work = df_work.drop(columns=drop_cols)

            # Missing values
            df_work = _handle_missing(df_work, config)

            # Encode
            df_work, encoders = _encode_categoricals(df_work, target, config)
            st.session_state.label_encoders = encoders

            # Feature / target split
            if target:
                remaining_features = [c for c in df_work.columns if c != target and c not in drop_cols]
                X = df_work[remaining_features]
                y = df_work[target]

                # Feature selection
                if do_feature_sel and problem_type in ("classification", "regression"):
                    selected = _select_features(X, y, problem_type, k_features)
                    X = X[selected]
                    st.info(f"Selected {len(selected)} features: `{selected}`")

                    # Feature importance chart
                    if problem_type == "classification":
                        from sklearn.feature_selection import f_classif
                        scores, _ = f_classif(X.fillna(0), y)
                    else:
                        from sklearn.feature_selection import f_regression
                        scores, _ = f_regression(X.fillna(0), y)
                    fi_df = pd.DataFrame({"Feature": X.columns, "Score": scores}).sort_values("Score", ascending=False)
                    fig = px.bar(fi_df, x="Score", y="Feature", orientation="h", title="Feature Importance (F-score)")
                    st.plotly_chart(fig, use_container_width=True)

                # Scale
                if scaling != "none":
                    X, scaler = _scale_features(X, scaling)
                    st.session_state.scaler = scaler

                df_processed = pd.concat([X.reset_index(drop=True), y.reset_index(drop=True)], axis=1)
                st.session_state.feature_cols = X.columns.tolist()
            else:
                # Clustering — no target
                X = df_work[[c for c in df_work.columns if c not in drop_cols]]
                # Drop any remaining non-numeric
                X = X.select_dtypes(include=[np.number])
                if scaling != "none":
                    X, scaler = _scale_features(X, scaling)
                    st.session_state.scaler = scaler
                df_processed = X
                st.session_state.feature_cols = X.columns.tolist()

            st.session_state.df_processed = df_processed
            st.session_state.preprocessing_config = config

        st.success(f"✅ Preprocessing done. Shape: {df_processed.shape}")
        st.dataframe(df_processed.head(), use_container_width=True)

        # Before/after comparison
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Original shape", f"{df.shape[0]} × {df.shape[1]}")
        with col2:
            st.metric("Processed shape", f"{df_processed.shape[0]} × {df_processed.shape[1]}")
