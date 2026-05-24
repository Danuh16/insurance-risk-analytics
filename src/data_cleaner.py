"""
data_cleaner.py
---------------
DVC pipeline stage: raw → cleaned insurance dataset.

Cleaning steps applied:
1. Drop columns that are 100% missing (no information value).
2. Remove records with zero or negative TotalPremium (invalid policies).
3. Fill remaining categorical NaNs with 'Unknown'.
4. Cap extreme TotalClaims outliers at the 99.9th percentile.
5. Recalculate LossRatio and Margin on clean values.

Run directly:
    python src/data_cleaner.py

Or via DVC:
    dvc repro
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_loader import load_data

RAW_PATH = Path(__file__).resolve().parents[1] / "data" / "insurance_data.txt"
CLEAN_PATH = Path(__file__).resolve().parents[1] / "data" / "insurance_data_cleaned.txt"
METRICS_PATH = Path(__file__).resolve().parents[1] / "reports" / "cleaning_metrics.json"


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Apply all cleaning transformations.

    Returns
    -------
    cleaned_df : pd.DataFrame
    metrics    : dict  — statistics about what was changed
    """
    metrics = {"raw_rows": len(df), "raw_cols": len(df.columns)}

    # Step 1 — Drop 100%-missing columns
    fully_missing = [c for c in df.columns if df[c].isna().all()]
    df.drop(columns=fully_missing, inplace=True)
    metrics["dropped_empty_cols"] = len(fully_missing)
    metrics["dropped_col_names"] = fully_missing

    # Step 2 — Remove invalid (non-positive premium) rows
    before = len(df)
    df = df[df["TotalPremium"] > 0].copy()
    metrics["removed_zero_premium_rows"] = before - len(df)

    # Step 3 — Fill categorical NaNs
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    for col in cat_cols:
        null_count = df[col].isna().sum()
        if null_count > 0:
            df[col] = df[col].fillna("Unknown")
    metrics["categorical_cols_imputed"] = len(cat_cols)

    # Step 4 — Cap TotalClaims outliers at 99.9th percentile
    cap_value = df["TotalClaims"].quantile(0.999)
    n_capped = (df["TotalClaims"] > cap_value).sum()
    df["TotalClaims"] = df["TotalClaims"].clip(upper=cap_value)
    metrics["claims_cap_value"] = round(float(cap_value), 2)
    metrics["claims_rows_capped"] = int(n_capped)

    # Step 5 — Recompute derived metrics
    df["LossRatio"] = df["TotalClaims"] / df["TotalPremium"].replace(0, np.nan)
    df["Margin"] = df["TotalPremium"] - df["TotalClaims"]

    metrics["clean_rows"] = len(df)
    metrics["clean_cols"] = len(df.columns)
    metrics["overall_loss_ratio"] = round(
        float(df["TotalClaims"].sum() / df["TotalPremium"].sum()), 6
    )

    return df, metrics


def main():
    print("=" * 55)
    print("DVC STAGE: clean_data")
    print("=" * 55)

    print(f"  Reading: {RAW_PATH}")
    df = load_data(RAW_PATH)
    print(f"  Raw shape: {df.shape}")

    print("  Applying cleaning transformations...")
    df_clean, metrics = clean_data(df)

    print(f"  Clean shape: {df_clean.shape}")
    print(f"  Rows removed (zero premium): {metrics['removed_zero_premium_rows']:,}")
    print(f"  Cols dropped (100% missing): {metrics['dropped_empty_cols']}")
    print(f"  TotalClaims capped at: R{metrics['claims_cap_value']:,.2f}")
    print(f"  Clean Loss Ratio: {metrics['overall_loss_ratio']:.4f}")

    # Save cleaned data
    df_clean.to_csv(CLEAN_PATH, sep="|", index=False)
    print(f"  Saved: {CLEAN_PATH}")

    # Save metrics for DVC tracking
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"  Metrics: {METRICS_PATH}")
    print("  Done: clean_data stage complete")


if __name__ == "__main__":
    main()
