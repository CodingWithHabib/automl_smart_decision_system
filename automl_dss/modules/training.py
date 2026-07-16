"""
Module: training.py
Trains multiple models, cross-validates, compares, selects best.
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    silhouette_score
)
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
import time
import plotly.express as px
import plotly.graph_objects as go


# ── Model Catalogs ─────────────────────────────────────────────────────────────

CLASSIFICATION_MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC(probability=True, random_state=42),
    "Naive Bayes": GaussianNB(),
}

REGRESSION_MODELS = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(random_state=42),
    "Lasso Regression": Lasso(random_state=42),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "KNN": KNeighborsRegressor(n_neighbors=5),
    "SVR": SVR(),
}

CLUSTERING_MODELS = {
    "KMeans": None,        # special — needs k
    "Agglomerative": None, # special — needs k
}


def _train_classification(X_train, X_test, y_train, y_test, selected_models, cv_folds):
    results = {}
    for name in selected_models:
        model = CLASSIFICATION_MODELS[name]
        t0 = time.time()
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        elapsed = time.time() - t0

        n_classes = len(np.unique(y_train))
        roc = None
        if n_classes == 2 and hasattr(model, "predict_proba"):
            try:
                roc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
            except Exception:
                pass

        results[name] = {
            "model": model,
            "cv_accuracy_mean": cv_scores.mean(),
            "cv_accuracy_std": cv_scores.std(),
            "test_accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
             "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
            "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
            "roc_auc": roc,
            "train_time": elapsed,
            "y_pred": y_pred,
        }
    return results


def _train_regression(X_train, X_test, y_train, y_test, selected_models, cv_folds):
    results = {}
    for name in selected_models:
        model = REGRESSION_MODELS[name]
        t0 = time.time()
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="r2")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        elapsed = time.time() - t0

        results[name] = {
            "model": model,
            "cv_r2_mean": cv_scores.mean(),
            "cv_r2_std": cv_scores.std(),
            "test_r2": r2_score(y_test, y_pred),
            "mae": mean_absolute_error(y_test, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            "train_time": elapsed,
            "y_pred": y_pred,
            "y_test": y_test,
        }
    return results


def _train_clustering(X, k_range):
    results = {}
    inertias, silhouettes = [], []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        sil = silhouette_score(X, labels) if k > 1 else -1
        inertias.append(km.inertia_)
        silhouettes.append(sil)
        results[f"KMeans k={k}"] = {
            "model": km,
            "k": k,
            "inertia": km.inertia_,
            "silhouette": sil,
            "labels": labels,
        }

    # Elbow + silhouette charts
    elbow_df = pd.DataFrame({"k": list(k_range), "Inertia": inertias, "Silhouette": silhouettes})
    return results, elbow_df


def render_training():
    st.title("🏋️ Step 4: Model Training")

    df = st.session_state.df_processed
    target = st.session_state.target_col
    problem_type = st.session_state.problem_type
    feature_cols = st.session_state.feature_cols

    # ── Settings ───────────────────────────────────────────────────────────────
    st.subheader("⚙️ Training Configuration")
    col1, col2 = st.columns(2)

    if problem_type in ("classification", "regression"):
        with col1:
            test_size = st.slider("Test set size (%)", 10, 40, 20) / 100
        with col2:
            cv_folds = st.slider("Cross-validation folds", 3, 10, 5)

        catalog = CLASSIFICATION_MODELS if problem_type == "classification" else REGRESSION_MODELS
        st.subheader("🤖 Select Models to Train")
        selected_models = st.multiselect(
            "Models", list(catalog.keys()), default=list(catalog.keys())[:5]
        )
        if not selected_models:
            st.warning("Select at least one model.")
            return

        X = df[feature_cols].values
        y = df[target].values

        if st.button("🚀 Train All Models", type="primary"):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42,
                stratify=y if problem_type == "classification" else None,
            )
            progress = st.progress(0)
            status = st.empty()
            all_results = {}

            for i, name in enumerate(selected_models):
                status.text(f"Training {name}...")
                if problem_type == "classification":
                    res = _train_classification(X_train, X_test, y_train, y_test, [name], cv_folds)
                else:
                    res = _train_regression(X_train, X_test, y_train, y_test, [name], cv_folds)
                all_results.update(res)
                progress.progress((i + 1) / len(selected_models))

            status.empty()
            progress.empty()

            st.session_state.models = all_results
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test

            # Select best model
            if problem_type == "classification":
                best_name = max(all_results, key=lambda k: all_results[k]["cv_accuracy_mean"])
                metric_key = "cv_accuracy_mean"
            else:
                best_name = max(all_results, key=lambda k: all_results[k]["cv_r2_mean"])
                metric_key = "cv_r2_mean"

            st.session_state.best_model = all_results[best_name]["model"]
            st.session_state.best_model_name = best_name
            st.session_state.metrics = all_results

            st.success(f"✅ Training complete! Best model: **{best_name}** ({metric_key}: {all_results[best_name][metric_key]:.4f})")
            _show_comparison_table(all_results, problem_type)

    else:
        # Clustering
        st.subheader("🔵 Clustering Configuration")
        col1, col2 = st.columns(2)
        with col1:
            k_min = st.number_input("Min clusters (k)", 2, 10, 2)
        with col2:
            k_max = st.number_input("Max clusters (k)", 3, 15, 8)

        if k_max <= k_min:
            st.error("Max k must be greater than min k.")
            return

        X = df[feature_cols].values

        if st.button("🚀 Run Clustering", type="primary"):
            with st.spinner("Clustering in progress..."):
                results, elbow_df = _train_clustering(X, range(int(k_min), int(k_max) + 1))

            st.session_state.models = results

            # Charts
            col1, col2 = st.columns(2)
            with col1:
                fig = px.line(elbow_df, x="k", y="Inertia", markers=True, title="Elbow Method")
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig = px.line(elbow_df, x="k", y="Silhouette", markers=True, title="Silhouette Score")
                st.plotly_chart(fig, use_container_width=True)

            # Best k by silhouette
            best_k_row = elbow_df.loc[elbow_df["Silhouette"].idxmax()]
            best_k = int(best_k_row["k"])
            best_name = f"KMeans k={best_k}"
            st.session_state.best_model = results[best_name]["model"]
            st.session_state.best_model_name = best_name
            st.session_state.metrics = results
            # Store cluster labels in processed df
            st.session_state.df_processed["Cluster"] = results[best_name]["labels"]
            st.success(f"✅ Best clustering: **{best_name}** (Silhouette: {best_k_row['Silhouette']:.4f})")


def _show_comparison_table(results: dict, problem_type: str):
    st.subheader("📊 Model Comparison")
    rows = []
    for name, r in results.items():
        if problem_type == "classification":
          rows.append({
    "Model": name,
    "CV Accuracy": f"{r['cv_accuracy_mean']:.4f} ± {r['cv_accuracy_std']:.4f}",
    "Test Accuracy": f"{r['test_accuracy']:.4f}",
    "Precision": f"{r['precision']:.4f}",
    "Recall": f"{r['recall']:.4f}",
    "F1 Score": f"{r['f1']:.4f}",
    "ROC-AUC": f"{r['roc_auc']:.4f}" if r["roc_auc"] else "N/A",
    "Train Time (s)": f"{r['train_time']:.2f}",
})
        else:
            rows.append({
                "Model": name,
                "CV R²": f"{r['cv_r2_mean']:.4f} ± {r['cv_r2_std']:.4f}",
                "Test R²": f"{r['test_r2']:.4f}",
                "MAE": f"{r['mae']:.4f}",
                "RMSE": f"{r['rmse']:.4f}",
                "Train Time (s)": f"{r['train_time']:.2f}",
            })
    df_cmp = pd.DataFrame(rows)
    st.dataframe(df_cmp, use_container_width=True)

    # Bar chart
    if problem_type == "classification":
        vals = {k: v["cv_accuracy_mean"] for k, v in results.items()}
        metric_label = "CV Accuracy"
    else:
        vals = {k: max(v["cv_r2_mean"], 0) for k, v in results.items()}
        metric_label = "CV R²"
    fig = px.bar(
        x=list(vals.keys()), y=list(vals.values()),
        labels={"x": "Model", "y": metric_label},
        title=f"Model Comparison — {metric_label}",
        color=list(vals.values()), color_continuous_scale="Greens",
    )
    st.plotly_chart(fig, use_container_width=True)
