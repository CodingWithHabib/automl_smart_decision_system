"""
tests/test_preprocessing.py
Tests for preprocessing helpers.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
import pytest

# Import internal helpers directly
from modules.preprocessing import _handle_missing, _encode_categoricals, _scale_features


def make_df():
    return pd.DataFrame({
        "age": [25, np.nan, 30, 35, np.nan],
        "city": ["Karachi", "Lahore", np.nan, "Lahore", "Karachi"],
        "salary": [50000, 60000, 55000, np.nan, 70000],
        "hired": [1, 0, 1, 1, 0],
    })


def test_median_fill_numerics():
    df = make_df()
    result = _handle_missing(df, {"missing_strategy": "median"})
    assert result["age"].isnull().sum() == 0
    assert result["salary"].isnull().sum() == 0


def test_mode_fill_categoricals():
    df = make_df()
    result = _handle_missing(df, {"missing_strategy": "median"})
    assert result["city"].isnull().sum() == 0


def test_mean_fill():
    df = make_df()
    result = _handle_missing(df, {"missing_strategy": "mean"})
    assert result["age"].isnull().sum() == 0


def test_label_encoding():
    df = pd.DataFrame({"city": ["A", "B", "A", "C"], "target": [0, 1, 0, 1]})
    encoded, encoders = _encode_categoricals(df, "target", {"encoding": "label"})
    assert encoded["city"].dtype in [np.int32, np.int64, int]
    assert "city" in encoders


def test_standard_scaling_zero_mean():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0], "b": [10.0, 20.0, 30.0, 40.0, 50.0]})
    scaled, scaler = _scale_features(df, "standard")
    assert abs(scaled["a"].mean()) < 1e-9
    assert abs(scaled["b"].mean()) < 1e-9


def test_minmax_scaling_range():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
    scaled, _ = _scale_features(df, "minmax")
    assert scaled["a"].min() >= 0.0
    assert scaled["a"].max() <= 1.0


def test_no_scaling_returns_same():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    result, scaler = _scale_features(df, "none")
    assert scaler is None
    pd.testing.assert_frame_equal(result, df)
