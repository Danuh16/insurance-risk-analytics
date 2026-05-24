"""
Unit tests for data_loader.py
"""
import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data_loader import load_data, get_summary


@pytest.fixture
def sample_df():
    """Minimal mock dataframe mimicking the insurance dataset."""
    return pd.DataFrame({
        "PolicyID": [1, 2, 3],
        "TransactionMonth": ["2015-03-01 00:00:00", "2015-04-01 00:00:00", "2015-05-01 00:00:00"],
        "TotalPremium": ["21.9", "50.0", "0"],
        "TotalClaims": [".0", "25.0", "0"],
        "Province": ["Gauteng", "Western Cape", "Gauteng"],
        "Gender": ["Male", "Female", "Not specified"],
        "IsVATRegistered": ["True", "False", "True"],
    })


def test_derived_metrics(sample_df):
    """LossRatio and Margin should be created correctly."""
    df = sample_df.copy()
    df["TotalPremium"] = pd.to_numeric(df["TotalPremium"], errors="coerce")
    df["TotalClaims"] = pd.to_numeric(df["TotalClaims"], errors="coerce")
    df["LossRatio"] = df["TotalClaims"] / df["TotalPremium"].replace(0, np.nan)
    df["Margin"] = df["TotalPremium"] - df["TotalClaims"]

    assert "LossRatio" in df.columns
    assert "Margin" in df.columns
    assert df.loc[0, "Margin"] == pytest.approx(21.9, rel=1e-3)
    assert df.loc[1, "LossRatio"] == pytest.approx(0.5, rel=1e-3)
    # Zero premium => LossRatio is NaN
    assert np.isnan(df.loc[2, "LossRatio"])


def test_transaction_month_dtype(sample_df):
    """TransactionMonth should parse as datetime."""
    df = sample_df.copy()
    df["TransactionMonth"] = pd.to_datetime(df["TransactionMonth"], errors="coerce")
    assert pd.api.types.is_datetime64_any_dtype(df["TransactionMonth"])


def test_numeric_coercion(sample_df):
    """TotalPremium and TotalClaims should become float after coercion."""
    df = sample_df.copy()
    df["TotalPremium"] = pd.to_numeric(df["TotalPremium"], errors="coerce")
    df["TotalClaims"] = pd.to_numeric(df["TotalClaims"], errors="coerce")
    assert pd.api.types.is_float_dtype(df["TotalPremium"])


def test_summary_keys(sample_df):
    """get_summary should return expected keys (using a prepped df)."""
    df = sample_df.copy()
    df["TransactionMonth"] = pd.to_datetime(df["TransactionMonth"], errors="coerce")
    df["TotalPremium"] = pd.to_numeric(df["TotalPremium"], errors="coerce")
    df["TotalClaims"] = pd.to_numeric(df["TotalClaims"], errors="coerce")
    df["LossRatio"] = df["TotalClaims"] / df["TotalPremium"].replace(0, float("nan"))
    df["Margin"] = df["TotalPremium"] - df["TotalClaims"]

    summary = get_summary(df)
    assert "rows" in summary
    assert "columns" in summary
    assert "missing_pct" in summary
