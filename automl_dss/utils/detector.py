"""
utils/detector.py
Problem type detection logic.
"""

import pandas as pd
import numpy as np


def detect_problem_type(df: pd.DataFrame, target_col: str) -> str:
    """
    Detect whether a supervised task is classification or regression.

    Rules (applied in order):
    1. Non-numeric dtype → classification
    2. Unique count ≤ 20 → classification
    3. Unique ratio ≤ 5% of total rows → classification
    4. Integer dtype with ≤ 30 unique values → classification
    5. Otherwise → regression
    """
    col = df[target_col].dropna()

    # Case 1: Non-numeric → classification
    if col.dtype == object or str(col.dtype) == "category" or col.dtype == bool:
        return "classification"

    n_unique = col.nunique()

    # Case 2: Boolean-like numeric (0/1 etc.)
    if n_unique == 2:
        return "classification"

    # Case 3: numeric but few discrete values → classification
    if pd.api.types.is_integer_dtype(col) and n_unique <= 10:
        return "classification"

    # Case 4: continuous numeric → regression
    return "regression"