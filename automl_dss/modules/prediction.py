"""
Module: prediction.py
Single-row manual input prediction + batch CSV prediction.
"""

import streamlit as st
import pandas as pd
import numpy as np
import io


def _make_input_widget(col_name: str, df: pd.DataFrame) -> float | str | None:
    sample = df[col_name].dropna()
    if sample.empty:
        return st.number_input(col_name, value=0.0)
    if pd.api.types.is_numeric_dtype(sample):
        return st.number_input(
            col_name,
            value=float(sample.median()),
            min_value=float(sample.min()),
            max_value=float(sample.max()),
        )
    else:
        unique_vals = sorted(sample.unique().tolist())
        if len(unique_vals) <= 20:
            return st.selectbox(col_name, unique_vals)
        return st.text_input(col_name, value=str(unique_vals[0]))


def _preprocess_input(input_dict: dict, feature_cols: list, encoders: dict, scaler) -> np.ndarray:
    row = pd.DataFrame([input_dict])
    for col, le in encoders.items():
        if col in row.columns:
            val = str(row[col].iloc[0])
            if val in le.classes_:
                row[col] = le.transform([val])[0]
            else:
                row[col] = 0
    row = row[feature_cols].fillna(0)
    if scaler is not None:
        row = pd.DataFrame(scaler.transform(row), columns=feature_cols)
    return row.values


def render_prediction():
    st.title("🎯 Step 6: Predictions")

    model = st.session_state.best_model
    best_name = st.session_state.best_model_name
    feature_cols = st.session_state.feature_cols
    encoders = st.session_state.label_encoders
    scaler = st.session_state.scaler
    problem_type = st.session_state.problem_type
    df_raw = st.session_state.df_raw
    target = st.session_state.target_col

    if model is None:
        st.warning("No model trained yet.")
        return

    if problem_type == "clustering":
        st.info("Clustering does not support new predictions in this version. See the Results tab for cluster assignments.")
        # Allow download of clustered data
        df_clustered = st.session_state.df_processed
        if "Cluster" in df_clustered.columns:
            csv = df_clustered.to_csv(index=False).encode()
            st.download_button("⬇️ Download Clustered Data", csv, "clustered_data.csv", "text/csv")
        return

    st.success(f"Using: **{best_name}**")

    tab1, tab2 = st.tabs(["🔢 Manual Input", "📁 Batch Prediction (CSV)"])

    with tab1:
        st.subheader("Enter values for each feature")
        input_dict = {}
        cols = st.columns(min(3, len(feature_cols)))
        for i, feat in enumerate(feature_cols):
            with cols[i % len(cols)]:
                if feat in df_raw.columns:
                    input_dict[feat] = _make_input_widget(feat, df_raw)
                else:
                    input_dict[feat] = st.number_input(feat, value=0.0)

        if st.button("🔮 Predict", type="primary"):
            try:
                X_input = _preprocess_input(input_dict, feature_cols, encoders, scaler)
                prediction = model.predict(X_input)[0]

                # Decode label if classification and encoder exists
                if problem_type == "classification" and target in encoders:
                    le = encoders[target]
                    try:
                        prediction = le.inverse_transform([int(prediction)])[0]
                    except Exception:
                        pass

                st.success(f"### Prediction: `{prediction}`")

                # Confidence for classification
                if problem_type == "classification" and hasattr(model, "predict_proba"):
                    proba = model.predict_proba(X_input)[0]
                    classes = model.classes_
                    # Decode class labels
                    if target in encoders:
                        try:
                            classes = encoders[target].inverse_transform(classes)
                        except Exception:
                            pass
                    conf_df = pd.DataFrame({
                        "Class": [str(c) for c in classes],
                        "Probability": proba,
                    }).sort_values("Probability", ascending=False)
                    import plotly.express as px
                    fig = px.bar(conf_df, x="Class", y="Probability",
                                 title="Prediction Confidence", color="Probability",
                                 color_continuous_scale="Blues")
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Prediction failed: {e}")

    with tab2:
        st.subheader("Upload a CSV with the same feature columns (no target)")
        batch_file = st.file_uploader("Upload batch CSV", type=["csv"], key="batch")

        if batch_file:
            try:
                df_batch = pd.read_csv(batch_file)
                st.write(f"Loaded {df_batch.shape[0]} rows")

                missing_feats = [f for f in feature_cols if f not in df_batch.columns]
                if missing_feats:
                    st.error(f"Missing columns: {missing_feats}")
                else:
                    df_input = df_batch[feature_cols].copy()
                    for col, le in encoders.items():
                        if col in df_input.columns and col != target:
                            df_input[col] = df_input[col].astype(str).apply(
                                lambda v: le.transform([v])[0] if v in le.classes_ else 0
                            )
                    df_input = df_input.fillna(0)
                    if scaler is not None:
                        df_input = pd.DataFrame(scaler.transform(df_input), columns=feature_cols)

                    preds = model.predict(df_input.values)

                    if problem_type == "classification" and target in encoders:
                        try:
                            preds = encoders[target].inverse_transform(preds.astype(int))
                        except Exception:
                            pass

                    df_batch["Prediction"] = preds

                    if problem_type == "classification" and hasattr(model, "predict_proba"):
                        proba = model.predict_proba(df_input.values)
                        df_batch["Confidence"] = proba.max(axis=1).round(4)

                    st.dataframe(df_batch.head(20), use_container_width=True)

                    csv_out = df_batch.to_csv(index=False).encode()
                    st.download_button("⬇️ Download Predictions", csv_out,
                                       "predictions.csv", "text/csv")
            except Exception as e:
                st.error(f"Batch prediction failed: {e}")
