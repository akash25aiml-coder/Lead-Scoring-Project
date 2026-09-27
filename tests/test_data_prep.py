"""
Tests for the leakage/cleaning logic — the part of this project most
student submissions never verify. Run: pytest tests/ from repo root.
"""
import sys
import os
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from data_prep import (
    load_and_clean, split_features, fit_rare_category_map,
    group_rare_categories, LEAKAGE_COLS, ZERO_VARIANCE_COLS,
)

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "Leads.csv")


@pytest.fixture(scope="module")
def cleaned_df():
    return load_and_clean(DATA_PATH)


def test_leakage_columns_are_dropped(cleaned_df):
    for col in LEAKAGE_COLS:
        assert col not in cleaned_df.columns, f"Leakage column '{col}' was not dropped"


def test_zero_variance_columns_are_dropped(cleaned_df):
    for col in ZERO_VARIANCE_COLS:
        assert col not in cleaned_df.columns, f"Zero-variance column '{col}' was not dropped"


def test_no_select_placeholder_remains(cleaned_df):
    for col in cleaned_df.select_dtypes(exclude="number").columns:
        assert "Select" not in cleaned_df[col].values, (
            f"'Select' placeholder leaked through in column '{col}' — "
            f"must be normalized to NaN before imputation"
        )


def test_target_column_present(cleaned_df):
    assert "Converted" in cleaned_df.columns
    assert set(cleaned_df["Converted"].unique()).issubset({0, 1})


def test_split_features_separates_target(cleaned_df):
    X, y, num_cols, cat_cols = split_features(cleaned_df)
    assert "Converted" not in X.columns
    assert len(y) == len(X)
    assert set(num_cols) | set(cat_cols) == set(X.columns)


def test_rare_category_grouping_is_train_only():
    """The core leakage-safety guarantee: rare-category thresholds must be
    computed from training data only, never from validation/test data."""
    train = pd.DataFrame({"City": ["A"] * 90 + ["B"] * 5 + ["C"] * 5})
    freq_map = fit_rare_category_map(train, cols=["City"], threshold=0.10)
    assert "B" in freq_map["City"]
    assert "C" in freq_map["City"]
    assert "A" not in freq_map["City"]

    # Apply the TRAIN-derived map to a differently-distributed test set —
    # test-set frequencies must NOT influence the grouping decision
    test = pd.DataFrame({"City": ["A"] * 2 + ["B"] * 2 + ["C"] * 2})
    grouped = group_rare_categories(test, freq_map, cols=["City"])
    assert (grouped["City"] == "Other").sum() == 4  # B's and C's collapsed
    assert (grouped["City"] == "A").sum() == 2       # A untouched
