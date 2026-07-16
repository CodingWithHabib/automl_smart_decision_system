"""
Module: results.py
Visualizes model results — confusion matrix, ROC, residuals, cluster plots.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.inspection import permutation_importance
from sklearn.decomposition import PCA


def _confusion_matrix_fig(y_test, y_pred, labels):
    cm = confusion_matrix(y_test, y_pred)
    fig = px.imshow(
        cm,
        labels={"x": "Predicted", "y": "Actual", "color": "Count"},
        x=[str(l) for l in labels],
        y=[str(l) for l in labels],
        text_auto=True,
        color_continuous_scale="Blues",
        title="Confusion Matrix",
    )
    return fig


def _roc_curve_fig(model, X_test, y_test):
    if not hasattr(model, "predict_proba"):
        return None
    try:
        y_scores = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_scores)
        roc_auc = auc(fpr, tpr)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"AUC = {roc_auc:.4f}"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash"), name="Random"))
        fig.update_layout(
            title="ROC Curve",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
        )
        return fig
    except Exception:
        return None


def _residual_plot(y_test, y_pred):
    residuals = np.array(y_test) - np.array(y_pred)
    fig = px.scatter(
        x=y_pred, y=residuals,
        labels={"x": "Predicted", "y": "Residuals"},
        title="Residual Plot",
    )
    fig.add_hline(y=0, line_dash="dash", line_color="red")
    return fig


def _feature_importance_fig(model, feature_cols):
    if hasattr(model, "feature_importances_"):
        fi = model.feature_importances_
        fi_df = pd.DataFrame({"Feature": feature_cols, "Importance": fi})
        fi_df = fi_df.sort_values("Importance", ascending=False).head(20)
        fig = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                     title="Feature Importances (top 20)", color="Importance",
                     color_continuous_scale="Viridis")
        return fig
    elif hasattr(model, "coef_"):
        coef = model.coef_.flatten()[:len(feature_cols)]
        fi_df = pd.DataFrame({"Feature": feature_cols[:len(coef)], "Coefficient": np.abs(coef)})
        fi_df = fi_df.sort_values("Coefficient", ascending=False).head(20)
        fig = px.bar(fi_df, x="Coefficient", y="Feature", orientation="h",
                     title="|Coefficients| (top 20)")
        return fig
    return None


def render_results():
    st.title("📊 Step 5: Results & Analysis")

    models = st.session_state.models
    best_name = st.session_state.best_model_name
    problem_type = st.session_state.problem_type
    feature_cols = st.session_state.feature_cols
    df = st.session_state.df_processed
    target = st.session_state.target_col

    if not models:
        st.warning("No trained models found. Go back and train models.")
        return

    st.success(f"🏆 Best Model: **{best_name}**")

    if problem_type == "classification":
        best = models[best_name]
        model = best["model"]
        y_pred = best["y_pred"]
        y_test = st.session_state.get("y_test", df[target].values)
        X_test = st.session_state.get("X_test")

        col1, col2, col3, col4, col5 = st.columns(5)

        col1.metric("Test Accuracy", f"{best['test_accuracy']:.4f}")
        col2.metric("Precision", f"{best['precision']:.4f}")
        col3.metric("Recall", f"{best['recall']:.4f}")
        col4.metric("F1 Score", f"{best['f1']:.4f}")
        col5.metric("CV Accuracy", f"{best['cv_accuracy_mean']:.4f}")

        classes = sorted(np.unique(y_test).tolist())

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(_confusion_matrix_fig(y_test, y_pred, classes), use_container_width=True)
            from sklearn.metrics import confusion_matrix

            cm = confusion_matrix(y_test, y_pred)
            TN, FP, FN, TP = cm.ravel()

            st.subheader("Confusion Matrix Breakdown")

            col1, col2, col3, col4 = st.columns(4)

            col1.metric("TP", TP)
            col2.metric("TN", TN)
            col3.metric("FP", FP)
            col4.metric("FN", FN)
        with col2:
            if len(classes) == 2 and X_test is not None:
                roc_fig = _roc_curve_fig(model, X_test, y_test)
                if roc_fig:
                    st.plotly_chart(roc_fig, use_container_width=True)
                else:
                    st.info("ROC curve unavailable for this model.")
            else:
                st.info("ROC curve shown only for binary classification.")

        # Feature importance
        fi_fig = _feature_importance_fig(model, feature_cols)
        if fi_fig:
            st.plotly_chart(fi_fig, use_container_width=True)

        # All models comparison
        st.subheader("All Models — Test Accuracy")
        acc_data = {k: v["test_accuracy"] for k, v in models.items()}
        fig = px.bar(x=list(acc_data.keys()), y=list(acc_data.values()),
                     labels={"x": "Model", "y": "Test Accuracy"},
                     color=list(acc_data.values()), color_continuous_scale="Teal")
        st.plotly_chart(fig, use_container_width=True)

    elif problem_type == "regression":
        best = models[best_name]
        model = best["model"]
        y_pred = best["y_pred"]
        y_test = best["y_test"]

        col1, col2, col3 = st.columns(3)
        col1.metric("Test R²", f"{best['test_r2']:.4f}")
        col2.metric("MAE", f"{best['mae']:.4f}")
        col3.metric("RMSE", f"{best['rmse']:.4f}")

        col1, col2 = st.columns(2)
        with col1:
            fig = px.scatter(x=y_test, y=y_pred,
                             labels={"x": "Actual", "y": "Predicted"},
                             title="Actual vs Predicted")
            fig.add_shape(type="line", x0=min(y_test), y0=min(y_test),
                          x1=max(y_test), y1=max(y_test),
                          line=dict(color="red", dash="dash"))
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.plotly_chart(_residual_plot(y_test, y_pred), use_container_width=True)

        fi_fig = _feature_importance_fig(model, feature_cols)
        if fi_fig:
            st.plotly_chart(fi_fig, use_container_width=True)

        # Model comparison
        st.subheader("All Models — R² Comparison")
        r2_data = {k: max(v["test_r2"], 0) for k, v in models.items()}
        fig = px.bar(x=list(r2_data.keys()), y=list(r2_data.values()),
                     labels={"x": "Model", "y": "R²"},
                     color=list(r2_data.values()), color_continuous_scale="Blues")
        st.plotly_chart(fig, use_container_width=True)

    else:
        # Clustering
        best = models[best_name]
        labels = best["labels"]
        X = df[feature_cols].values

        col1, col2 = st.columns(2)
        col1.metric("Optimal k", best["k"])
        col2.metric("Silhouette Score", f"{best['silhouette']:.4f}")

        # PCA 2D projection
        st.subheader("Cluster Visualization (PCA 2D)")
        pca = PCA(n_components=2, random_state=42)
        X_2d = pca.fit_transform(X)
        pca_df = pd.DataFrame(X_2d, columns=["PC1", "PC2"])
        pca_df["Cluster"] = labels.astype(str)
        fig = px.scatter(pca_df, x="PC1", y="PC2", color="Cluster",
                         title=f"Clusters — {best_name} (PCA projection)")
        st.plotly_chart(fig, use_container_width=True)

        # Cluster sizes
        sizes = pd.Series(labels).value_counts().sort_index()
        fig = px.bar(x=[f"Cluster {k}" for k in sizes.index], y=sizes.values,
                     title="Cluster Sizes", labels={"x": "Cluster", "y": "Count"})
        st.plotly_chart(fig, use_container_width=True)
