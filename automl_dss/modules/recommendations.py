"""
Module: recommendations.py
Smart decision support — rule-based recommendations derived from model results.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px


def _generate_classification_recs(models: dict, best_name: str, df: pd.DataFrame, target: str) -> list[dict]:
    recs = []
    best = models[best_name]

    # Accuracy thresholds
    acc = best["test_accuracy"]
    if acc < 0.60:
        recs.append({"level": "🔴 Critical", "message": f"Model accuracy is low ({acc:.2%}). Dataset may lack discriminating features, or more data is needed.", "action": "Consider feature engineering, collecting more samples, or simplifying the problem."})
    elif acc < 0.75:
        recs.append({"level": "🟡 Warning", "message": f"Accuracy ({acc:.2%}) is moderate. There is room for improvement.", "action": "Try hyperparameter tuning, ensemble methods, or address class imbalance if present."})
    else:
        recs.append({"level": "🟢 Good", "message": f"Accuracy ({acc:.2%}) is acceptable for most use cases.", "action": "Validate on real-world unseen data before deployment."})

    # Class imbalance
    vc = df[target].value_counts(normalize=True)
    if vc.max() > 0.80:
        recs.append({"level": "🟡 Warning", "message": f"Class imbalance detected: dominant class = {vc.max():.1%}. Accuracy may be misleading.", "action": "Use F1-score / ROC-AUC as primary metric. Consider SMOTE or class weights."})

    # CV vs test gap
    cv_acc = best["cv_accuracy_mean"]
    gap = abs(cv_acc - acc)
    if gap > 0.10:
        recs.append({"level": "🟡 Warning", "message": f"Large gap between CV accuracy ({cv_acc:.2%}) and test accuracy ({acc:.2%}) — possible overfitting.", "action": "Reduce model complexity, add regularization, or increase training data."})

    return recs


def _generate_regression_recs(models: dict, best_name: str) -> list[dict]:
    recs = []
    best = models[best_name]
    r2 = best["test_r2"]
    rmse = best["rmse"]
    mae = best["mae"]

    if r2 < 0:
        recs.append({"level": "🔴 Critical", "message": f"R² is negative ({r2:.4f}) — model is worse than a mean baseline.", "action": "Check for data leakage, target encoding issues, or fundamental feature–target mismatch."})
    elif r2 < 0.50:
        recs.append({"level": "🔴 Critical", "message": f"R² = {r2:.4f} — model explains less than 50% of variance.", "action": "Engineer better features or verify the correct target column was selected."})
    elif r2 < 0.70:
        recs.append({"level": "🟡 Warning", "message": f"R² = {r2:.4f} — moderate fit. Useful for exploratory analysis but not precise prediction.", "action": "Investigate non-linear relationships, feature interactions, or outliers."})
    else:
        recs.append({"level": "🟢 Good", "message": f"R² = {r2:.4f} — strong predictive power.", "action": "Validate with fresh data. Monitor for distribution shift in production."})

    if rmse > mae * 2:
        recs.append({"level": "🟡 Warning", "message": "RMSE is significantly higher than MAE — large outlier errors present.", "action": "Investigate outlier rows and decide whether to cap/remove them."})

    return recs


def _generate_clustering_recs(models: dict, best_name: str) -> list[dict]:
    recs = []
    best = models[best_name]
    sil = best["silhouette"]
    k = best["k"]

    if sil < 0.25:
        recs.append({"level": "🔴 Critical", "message": f"Silhouette = {sil:.4f} — clusters overlap significantly.", "action": "Try different k values, different scaling, or consider if the data has real cluster structure."})
    elif sil < 0.50:
        recs.append({"level": "🟡 Warning", "message": f"Silhouette = {sil:.4f} — clusters are weakly separated.", "action": "Explore DBSCAN for arbitrary shapes, or apply dimensionality reduction before clustering."})
    else:
        recs.append({"level": "🟢 Good", "message": f"Silhouette = {sil:.4f} — clusters are reasonably well-separated.", "action": f"Proceed with k={k} clusters. Profile each cluster to understand segment characteristics."})

    return recs


def _general_data_recs(df: pd.DataFrame) -> list[dict]:
    recs = []
    missing_pct = df.isnull().mean().mean() * 100
    if missing_pct > 20:
        recs.append({"level": "🟡 Warning", "message": f"{missing_pct:.1f}% of values are missing — imputation may have introduced bias.", "action": "Collect cleaner data or use domain expertise to fill critical fields."})
    n_rows = df.shape[0]
    if n_rows < 200:
        recs.append({"level": "🟡 Warning", "message": f"Small dataset: {n_rows} rows. Model generalization may be poor.", "action": "Collect more data. Use cross-validation metrics (not just test accuracy) for evaluation."})
    return recs


def render_recommendations():
    st.title("💡 Step 7: Smart Recommendations")

    models = st.session_state.models
    best_name = st.session_state.best_model_name
    problem_type = st.session_state.problem_type
    df_raw = st.session_state.df_raw
    target = st.session_state.target_col

    if not models or not best_name:
        st.warning("No results available. Complete model training first.")
        return

    st.subheader(f"🏆 Best Model: `{best_name}`")

    # Generate recs
    recs = []
    recs += _general_data_recs(df_raw)
    if problem_type == "classification":
        recs += _generate_classification_recs(models, best_name, df_raw, target)
    elif problem_type == "regression":
        recs += _generate_regression_recs(models, best_name)
    else:
        recs += _generate_clustering_recs(models, best_name)

    # Display recs
    st.subheader("📋 Recommendations")
    for rec in recs:
        with st.expander(f"{rec['level']} — {rec['message'][:80]}..."):
            st.markdown(f"**Finding:** {rec['message']}")
            st.markdown(f"**Suggested Action:** {rec['action']}")

    # Decision summary
    st.subheader("📄 Decision Summary")
    st.markdown(f"""
| Item | Value |
|---|---|
| Problem Type | {problem_type.title()} |
| Best Model | {best_name} |
| Dataset Size | {df_raw.shape[0]} rows × {df_raw.shape[1]} cols |
| Features Used | {len(st.session_state.feature_cols)} |
| Recommendations | {len(recs)} |
    """)

    # Export report
    st.subheader("⬇️ Export")
    report_lines = [
        "AutoML DSS — Decision Report",
        "=" * 40,
        f"Problem Type: {problem_type.title()}",
        f"Best Model: {best_name}",
        f"Dataset: {df_raw.shape[0]} rows × {df_raw.shape[1]} cols",
        "",
        "Recommendations:",
    ]
    for r in recs:
        report_lines.append(f"\n{r['level']}")
        report_lines.append(f"  Finding: {r['message']}")
        report_lines.append(f"  Action : {r['action']}")

    report_text = "\n".join(report_lines)
    st.download_button(
        "⬇️ Download Report (.txt)", report_text.encode(),
        "automl_dss_report.txt", "text/plain"
    )
