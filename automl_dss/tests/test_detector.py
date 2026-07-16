"""
tests/test_detector.py
Unit tests for problem type detection.
Run: pytest tests/
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
import pytest
from utils.detector import detect_problem_type


def test_string_target_is_classification():
    df = pd.DataFrame({"x": [1, 2, 3], "y": ["cat", "dog", "cat"]})
    assert detect_problem_type(df, "y") == "classification"


def test_binary_int_target_is_classification():
    df = pd.DataFrame({"x": range(100), "y": [0, 1] * 50})
    assert detect_problem_type(df, "y") == "classification"


def test_continuous_float_is_regression():
    df = pd.DataFrame({
        "x": range(200),
        "y": np.random.uniform(0, 1000, 200),
    })
    assert detect_problem_type(df, "y") == "regression"


def test_low_cardinality_int_is_classification():
    df = pd.DataFrame({"x": range(100), "y": np.random.choice([1, 2, 3, 4, 5], 100)})
    assert detect_problem_type(df, "y") == "classification"


def test_high_cardinality_int_is_regression():
    df = pd.DataFrame({"x": range(1000), "y": np.arange(1000) * 2 + np.random.randn(1000)})
    assert detect_problem_type(df, "y") == "regression"


def test_bool_target_is_classification():
    df = pd.DataFrame({"x": range(20), "y": [True, False] * 10})
    assert detect_problem_type(df, "y") == "classification"
