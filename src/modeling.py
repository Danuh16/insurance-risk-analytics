"""
modeling.py
-----------
ML pipeline for ACIS risk-based premium prediction.

Two sub-pipelines
-----------------
1. Claim Severity  — XGBoost / RF / Linear regression on log(TotalClaims)
                     (subset: policies where TotalClaims > 0)
2. Claim Frequency — XGBoost / RF / Logistic regression classifier for HasClaim
                     (all policies; negatives undersampled for training speed)

Combined premium framework
--------------------------
  Risk Premium = P(claim) × E[Severity | claim] × (1 + expense_loading + profit_margin)
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier, XGBRegressor

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Column lists
# ---------------------------------------------------------------------------

# Columns to drop before modeling (leaky, 100% missing, high-cardinality IDs, admin)
_DROP_COLS = [
    # 100% missing
    "AlarmImmobiliser", "TrackingDevice", "NewVehicle",
    # 78% missing
    "CustomValueEstimate",
    # Target-derived (leakage)
    "LossRatio", "Margin",
    # IDs / timestamps (handled via feature engineering)
    "UnderwrittenCoverID", "PolicyID",
    "TransactionMonth", "VehicleIntroDate",
    # High-cardinality vehicle descriptors
    "make", "Model", "mmcode",
    # Near-duplicate geographic fields
    "MainCrestaZone", "SubCrestaZone",
    # Low/zero signal
    "Country", "IsVATRegistered", "Citizenship",
    "Title", "Language", "Bank", "AccountType",
    "LegalType", "ItemType",
    # Near-fully-null binary flags
    "WrittenOff", "Rebuilt", "Converted", "CrossBorder",
    "NumberOfVehiclesInFleet",
    # Admin fields
    "ExcessSelected", "StatutoryClass", "StatutoryRiskType",
]

# Categorical columns to label-encode
_CAT_COLS = [
    "Province", "Gender", "VehicleType", "bodytype",
    "CoverType", "CoverCategory", "CoverGroup", "Section", "Product",
    "TermFrequency", "MaritalStatus",
]

# Numeric columns (impute with median if missing)
_NUM_COLS = [
    "SumInsured", "CalculatedPremiumPerTerm",
    "kilowatts", "cubiccapacity", "Cylinders", "NumberOfDoors",
    "RegistrationYear", "CapitalOutstanding", "PostalCode",
]


# ---------------------------------------------------------------------------
# Feature preparation
# ---------------------------------------------------------------------------

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer, impute, and encode features for model input.

    Steps
    -----
    1. Engineer derived features (vehicle_age, policy_month, log transforms,
       premium rate ratio) from the original DataFrame *before* dropping columns.
    2. Drop irrelevant / leaky columns and the model targets.
    3. Impute: median for numeric, 'Unknown' for categorical.
    4. Label-encode all categorical columns.
    5. Drop any residual non-numeric columns (e.g. raw date strings).

    Parameters
    ----------
    df : raw or lightly-cleaned DataFrame (output of load_data or clean_data)

    Returns
    -------
    pd.DataFrame  — fully numeric, model-ready feature matrix
    """
    out = df.copy()

    # --- 1. Feature engineering ---
    # Policy date features
    if "TransactionMonth" in out.columns:
        tx = pd.to_datetime(out["TransactionMonth"], errors="coerce")
        out["policy_year"]  = tx.dt.year.fillna(2014).astype(int)
        out["policy_month"] = tx.dt.month.fillna(6).astype(int)
    else:
        out["policy_year"]  = 2014
        out["policy_month"] = 6

    # Vehicle age (capped to sensible range)
    out["vehicle_age"] = (
        out["policy_year"] - out["RegistrationYear"].clip(1950, 2020)
    ).clip(0, 50)

    # Premium rate (premium per unit of sum insured)
    out["premium_rate"] = (
        out["CalculatedPremiumPerTerm"]
        / out["SumInsured"].replace(0, np.nan)
    ).fillna(0).clip(0, 2)

    # Log-transforms of right-skewed monetary columns
    for col in ["SumInsured", "CalculatedPremiumPerTerm", "CapitalOutstanding"]:
        if col in out.columns:
            out[f"log_{col}"] = np.log1p(out[col].clip(lower=0))

    # --- 2. Drop irrelevant / leaky / target columns ---
    targets  = ["TotalClaims", "TotalPremium", "HasClaim"]
    to_drop  = [c for c in _DROP_COLS + targets if c in out.columns]
    out.drop(columns=to_drop, inplace=True, errors="ignore")

    # --- 3. Impute ---
    # Categorical: fill NaN with "Unknown"
    for col in _CAT_COLS:
        if col in out.columns:
            out[col] = out[col].fillna("Unknown")

    # Numeric: fill NaN with column median
    for col in _NUM_COLS:
        if col in out.columns and out[col].isnull().any():
            out[col] = out[col].fillna(out[col].median())

    # Derived numeric: fill any NaN introduced by engineering
    for col in ["vehicle_age", "policy_year", "policy_month",
                "premium_rate", "log_SumInsured",
                "log_CalculatedPremiumPerTerm", "log_CapitalOutstanding"]:
        if col in out.columns and out[col].isnull().any():
            out[col] = out[col].fillna(out[col].median())

    # --- 4. Label-encode categoricals ---
    for col in _CAT_COLS:
        if col in out.columns:
            le = LabelEncoder()
            out[col] = le.fit_transform(out[col].astype(str))

    # --- 5. Drop residual non-numeric columns ---
    non_num = out.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_num:
        out.drop(columns=non_num, inplace=True, errors="ignore")

    return out


def get_feature_names(df: pd.DataFrame) -> list[str]:
    """Return the list of feature column names that prepare_features would produce."""
    return prepare_features(df.head(10)).columns.tolist()


# ---------------------------------------------------------------------------
# Dataset builders
# ---------------------------------------------------------------------------

def build_severity_dataset(
    df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Build train / test splits for Claim Severity prediction.

    Target : log1p(TotalClaims)  — back-transformed at evaluation time.
    Rows   : only policies where TotalClaims > 0.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    claim_df = df[df["TotalClaims"] > 0].copy()
    y = np.log1p(claim_df["TotalClaims"])
    X = prepare_features(claim_df)
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def build_frequency_dataset(
    df: pd.DataFrame,
    neg_to_pos_ratio: int = 50,
    test_size: float = 0.20,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Build train / test splits for Claim Frequency prediction (binary).

    Target          : HasClaim (1 if TotalClaims > 0 else 0)
    neg_to_pos_ratio: negatives kept per positive (undersampling for speed).
                      Set to None to use the full dataset.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    pos = df[df["TotalClaims"] > 0].copy()
    neg = df[df["TotalClaims"] == 0].copy()

    if neg_to_pos_ratio is not None:
        n_neg = min(len(pos) * neg_to_pos_ratio, len(neg))
        neg   = neg.sample(n=n_neg, random_state=random_state)

    work = pd.concat([pos, neg], ignore_index=True)
    work["HasClaim"] = (work["TotalClaims"] > 0).astype(int)

    y = work["HasClaim"].copy()
    X = prepare_features(work)

    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y,
    )


# ---------------------------------------------------------------------------
# Model trainers
# ---------------------------------------------------------------------------

def train_severity_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
) -> dict:
    """
    Train Linear Regression, Random Forest, and XGBoost regressors.

    All models predict log1p(TotalClaims).

    Returns
    -------
    dict  {name: fitted model}
    """
    # Linear Regression — wrapped in a scaling pipeline
    lr_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model",  LinearRegression()),
    ])
    lr_pipe.fit(X_train, y_train)

    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=3,
        random_state=random_state,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    xgb = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        verbosity=0,
        n_jobs=-1,
    )
    xgb.fit(X_train, y_train)

    return {
        "Linear Regression": lr_pipe,
        "Random Forest":     rf,
        "XGBoost":           xgb,
    }


def train_frequency_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
) -> dict:
    """
    Train Logistic Regression, Random Forest, and XGBoost classifiers.

    Class imbalance is addressed via class_weight='balanced' (LR / RF) and
    scale_pos_weight (XGBoost).

    Returns
    -------
    dict  {name: fitted model}
    """
    n_pos = int(y_train.sum())
    n_neg = int((y_train == 0).sum())
    scale_pw = n_neg / n_pos if n_pos > 0 else 1.0

    lr_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model",  LogisticRegression(
            class_weight="balanced",
            max_iter=500,
            random_state=random_state,
        )),
    ])
    lr_pipe.fit(X_train, y_train)

    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pw,
        eval_metric="logloss",
        random_state=random_state,
        verbosity=0,
        n_jobs=-1,
    )
    xgb.fit(X_train, y_train)

    return {
        "Logistic Regression": lr_pipe,
        "Random Forest":       rf,
        "XGBoost":             xgb,
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_regression(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    label: str = "",
) -> dict:
    """
    Evaluate a regression model.

    Predictions and y_test are assumed to be in log1p space; both are
    back-transformed with expm1 before computing ZAR-space metrics.

    Returns
    -------
    dict with keys: Model, RMSE (ZAR), R² (ZAR), RMSE (log), R² (log)
    """
    y_pred_log = model.predict(X_test)
    y_pred     = np.expm1(y_pred_log)
    y_true     = np.expm1(y_test)

    rmse     = np.sqrt(mean_squared_error(y_true, y_pred))
    r2       = r2_score(y_true, y_pred)
    rmse_log = np.sqrt(mean_squared_error(y_test, y_pred_log))
    r2_log   = r2_score(y_test, y_pred_log)

    return {
        "Model":      label,
        "RMSE (ZAR)": round(rmse,     2),
        "R² (ZAR)":   round(r2,       4),
        "RMSE (log)": round(rmse_log, 4),
        "R² (log)":   round(r2_log,   4),
    }


def evaluate_classification(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    label: str = "",
) -> dict:
    """
    Evaluate a binary classification model on claim frequency.

    Returns
    -------
    dict with keys: Model, Accuracy, Precision, Recall, F1, ROC-AUC
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return {
        "Model":     label,
        "Accuracy":  round(accuracy_score(y_test, y_pred),                4),
        "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "Recall":    round(recall_score(y_test, y_pred,    zero_division=0), 4),
        "F1":        round(f1_score(y_test, y_pred,        zero_division=0), 4),
        "ROC-AUC":   round(roc_auc_score(y_test, y_prob),  4),
    }


def build_comparison_table(results: list[dict]) -> pd.DataFrame:
    """Convert a list of evaluation dicts to a formatted DataFrame."""
    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# SHAP feature importance
# ---------------------------------------------------------------------------

def compute_shap_values(
    model,
    X: pd.DataFrame,
    model_type: str = "tree",
    n_background: int = 100,
) -> tuple:
    """
    Compute SHAP values for a fitted model.

    Parameters
    ----------
    model      : fitted estimator or sklearn Pipeline
    X          : feature DataFrame (subset recommended for speed)
    model_type : "tree" (RF / XGBoost) | "linear" (LR)
    n_background : rows to use as SHAP background sample (tree models)

    Returns
    -------
    (shap_values : np.ndarray, explainer)
    """
    import shap

    # Unwrap Pipeline to get the core estimator
    inner = model.named_steps["model"] if hasattr(model, "named_steps") else model

    if model_type == "linear":
        if hasattr(model, "named_steps"):
            X_t = model.named_steps["scaler"].transform(X)
        else:
            X_t = X.values
        explainer  = shap.LinearExplainer(inner, X_t)
        shap_vals  = explainer.shap_values(X_t)
    else:
        bg         = shap.sample(X, n_background, random_state=42)
        explainer  = shap.TreeExplainer(inner, data=bg)
        shap_out   = explainer(X)
        shap_vals  = shap_out.values
        # For binary classifiers, TreeExplainer may return (n, features, 2)
        if shap_vals.ndim == 3:
            shap_vals = shap_vals[:, :, 1]

    return shap_vals, explainer


# ---------------------------------------------------------------------------
# Risk-based premium framework
# ---------------------------------------------------------------------------

def compute_risk_premium(
    freq_model,
    sev_model,
    X: pd.DataFrame,
    expense_loading: float = 0.15,
    profit_margin: float = 0.05,
) -> pd.Series:
    """
    Compute an indication of the risk-based premium for each policy.

    Formula
    -------
    Pure Premium  = P(claim) × E[Severity | claim]
    Risk Premium  = Pure Premium × (1 + expense_loading + profit_margin)

    Parameters
    ----------
    freq_model      : fitted claim-frequency classifier
    sev_model       : fitted claim-severity regressor (log space)
    X               : feature matrix (same columns as training)
    expense_loading : e.g. 0.15 for 15% expense loading
    profit_margin   : e.g. 0.05 for 5% profit target

    Returns
    -------
    pd.Series  — computed risk premium per row
    """
    p_claim   = freq_model.predict_proba(X)[:, 1]
    severity  = np.expm1(sev_model.predict(X))
    pure_prem = p_claim * severity
    total     = pure_prem * (1 + expense_loading + profit_margin)
    return pd.Series(total, index=X.index, name="RiskPremium")

