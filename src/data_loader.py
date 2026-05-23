"""
data_loader.py
--------------
Handles loading and initial type-casting of the ACIS insurance dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "insurance_data.txt"


def load_data(filepath: str | Path = DATA_PATH) -> pd.DataFrame:
    """
    Load the pipe-delimited insurance dataset and apply correct dtypes.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe with proper column types.
    """
    df = pd.read_csv(
        filepath,
        sep="|",
        low_memory=False,
        na_values=["", " ", "NA", "N/A", "nan", "NaN"],
    )

    # --- Date columns ---
    df["TransactionMonth"] = pd.to_datetime(df["TransactionMonth"], errors="coerce")

    # --- Numeric columns that may come in as strings ---
    numeric_cols = [
        "TotalPremium",
        "TotalClaims",
        "SumInsured",
        "CalculatedPremiumPerTerm",
        "CustomValueEstimate",
        "CapitalOutstanding",
        "kilowatts",
        "cubiccapacity",
        "Cylinders",
        "NumberOfDoors",
        "RegistrationYear",
        "NumberOfVehiclesInFleet",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- Boolean columns ---
    bool_cols = [
        "IsVATRegistered", "AlarmImmobiliser", "TrackingDevice",
        "NewVehicle", "WrittenOff", "Rebuilt", "Converted", "CrossBorder",
    ]
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].map({"True": True, "False": False, True: True, False: False})

    # --- Derived metrics ---
    df["LossRatio"] = df["TotalClaims"] / df["TotalPremium"].replace(0, np.nan)
    df["Margin"] = df["TotalPremium"] - df["TotalClaims"]

    return df


def get_summary(df: pd.DataFrame) -> dict:
    """Return basic dataset summary as a dict."""
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "date_range": (
            df["TransactionMonth"].min(),
            df["TransactionMonth"].max(),
        ),
        "missing_pct": (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
    }
