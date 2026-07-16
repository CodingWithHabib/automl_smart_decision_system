"""
tests/test_training.py
Smoke tests for model training functions.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pytest
from modules.training import _train_classification, _train_regression, _train_clustering
from sklearn.datasets import make_classification, make_regression, make_blobs


def test_classification_returns_expected_keys():
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    X_train, X_test = X[:80], X[80:]
    y_train, y_test = y[:80], y[80:]
    results = _train_classification(X_train, X_test, y_train, y_test,
                                    ["Logistic Regression"], cv_folds=3)
    r = results["Logistic Regression"]
    assert "test_accuracy" in r
    assert "f1" in r
    assert "cv_accuracy_mean" in r
    assert 0.0 <= r["test_accuracy"] <= 1.0


def test_regression_returns_expected_keys():
    X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)
    X_train, X_test = X[:80], X[80:]
    y_train, y_test = y[:80], y[80:]
    results = _train_regression(X_train, X_test, y_train, y_test,
                                ["Linear Regression"], cv_folds=3)
    r = results["Linear Regression"]
    assert "test_r2" in r
    assert "rmse" in r
    assert "mae" in r


def test_clustering_returns_silhouette():
    X, _ = make_blobs(n_samples=100, centers=3, random_state=42)
    results, elbow_df = _train_clustering(X, range(2, 6))
    assert "KMeans k=3" in results
    assert results["KMeans k=3"]["silhouette"] > 0
    assert "Silhouette" in elbow_df.columns


def test_multiple_classifiers_train():
    X, y = make_classification(n_samples=150, n_features=8, random_state=42)
    X_train, X_test = X[:120], X[120:]
    y_train, y_test = y[:120], y[120:]
    models_to_test = ["Decision Tree", "Random Forest", "Naive Bayes"]
    results = _train_classification(X_train, X_test, y_train, y_test,
                                    models_to_test, cv_folds=3)
    assert len(results) == 3
    for name in models_to_test:
        assert name in results
