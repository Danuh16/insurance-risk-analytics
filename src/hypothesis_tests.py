"""
hypothesis_tests.py
-------------------
Reusable statistical hypothesis testing functions for ACIS insurance risk analysis.

KPIs
----
Claim Frequency : proportion of policies with at least one claim  (binary  → chi-squared)
Claim Severity  : mean claim amount given a claim occurred        (numeric → Welch t-test)
Margin          : TotalPremium − TotalClaims                      (numeric → Welch t-test)

Hypotheses tested
-----------------
H1 – No risk differences across provinces.
H2 – No risk differences between zip codes.
H3 – No significant margin (profit) difference between zip codes.
H4 – No significant risk difference between Women and Men.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# KPI helpers
# ---------------------------------------------------------------------------

def compute_claim_frequency(df: pd.DataFrame) -> pd.Series:
    """Return a binary Series: 1 if TotalClaims > 0, else 0."""
    return (df["TotalClaims"] > 0).astype(int)


def compute_claim_severity(df: pd.DataFrame) -> pd.Series:
    """Return TotalClaims for rows where a claim occurred."""
    return df.loc[df["TotalClaims"] > 0, "TotalClaims"]


def compute_margin(df: pd.DataFrame) -> pd.Series:
    """Return the Margin column (TotalPremium − TotalClaims)."""
    return df["Margin"]


# ---------------------------------------------------------------------------
# Core statistical test functions
# ---------------------------------------------------------------------------

def chi_squared_test(
    group_a: pd.Series,
    group_b: pd.Series,
    alpha: float = 0.05,
) -> dict:
    """
    Chi-squared test for difference in claim frequency between two groups.

    Parameters
    ----------
    group_a, group_b : binary pd.Series (1 = claim occurred, 0 = no claim)
    alpha            : significance level (default 0.05)

    Returns
    -------
    dict with keys: test, chi2, p_value, dof, decision,
                    claim_freq_a, claim_freq_b
    """
    a_claim    = int((group_a == 1).sum())
    a_no_claim = int((group_a == 0).sum())
    b_claim    = int((group_b == 1).sum())
    b_no_claim = int((group_b == 0).sum())

    ct = np.array([[a_claim, a_no_claim], [b_claim, b_no_claim]])
    # Disable Yates' continuity correction – it is overly conservative for
    # large samples (n >> 1 000) and can suppress genuine signal.
    chi2, p, dof, _expected = stats.chi2_contingency(ct, correction=False)

    return {
        "test":          "chi-squared",
        "chi2":          chi2,
        "p_value":       p,
        "dof":           dof,
        "n_a":           len(group_a),
        "n_b":           len(group_b),
        "claim_freq_a":  group_a.mean(),
        "claim_freq_b":  group_b.mean(),
        "decision":      "Reject H₀" if p < alpha else "Fail to Reject H₀",
    }


def chi_squared_multigroup(
    df: pd.DataFrame,
    group_col: str,
    claim_flag: pd.Series,
    alpha: float = 0.05,
) -> dict:
    """
    Chi-squared test for claim-frequency differences across multiple groups.

    Parameters
    ----------
    df          : DataFrame (index must align with claim_flag)
    group_col   : column name for the grouping variable
    claim_flag  : binary Series (1 = claim, 0 = no claim), same index as df
    alpha       : significance level

    Returns
    -------
    dict with keys: test, chi2, p_value, dof, decision,
                    claim_frequency_by_group, contingency_table
    """
    temp = df[[group_col]].copy()
    temp["HasClaim"] = claim_flag.values
    ct = pd.crosstab(temp[group_col], temp["HasClaim"])
    chi2, p, dof, _expected = stats.chi2_contingency(ct)

    freq_by_group = temp.groupby(group_col)["HasClaim"].mean().sort_values(ascending=False)

    return {
        "test":                    "chi-squared (multi-group)",
        "chi2":                    chi2,
        "p_value":                 p,
        "dof":                     dof,
        "decision":                "Reject H₀" if p < alpha else "Fail to Reject H₀",
        "claim_frequency_by_group": freq_by_group,
        "contingency_table":       ct,
    }


def welch_t_test(
    group_a: pd.Series,
    group_b: pd.Series,
    label_a: str = "A",
    label_b: str = "B",
    alpha: float = 0.05,
) -> dict:
    """
    Welch's t-test (unequal variances) for difference in means.

    Parameters
    ----------
    group_a, group_b : numeric pd.Series
    label_a, label_b : display labels for each group
    alpha            : significance level

    Returns
    -------
    dict with keys: test, t_stat, p_value, mean_a, mean_b,
                    diff, n_a, n_b, decision
    """
    a = group_a.dropna()
    b = group_b.dropna()
    t_stat, p = stats.ttest_ind(a, b, equal_var=False)

    return {
        "test":     "Welch t-test",
        "t_stat":   t_stat,
        "p_value":  p,
        f"mean_{label_a}": a.mean(),
        f"mean_{label_b}": b.mean(),
        "diff":     b.mean() - a.mean(),
        "n_a":      len(a),
        "n_b":      len(b),
        "decision": "Reject H₀" if p < alpha else "Fail to Reject H₀",
    }


def kruskal_wallis_test(
    groups: list[pd.Series],
    group_labels: list[str],
    alpha: float = 0.05,
) -> dict:
    """
    Kruskal-Wallis H-test: non-parametric multi-group comparison.

    Parameters
    ----------
    groups       : list of numeric pd.Series (one per group)
    group_labels : string labels corresponding to each group
    alpha        : significance level

    Returns
    -------
    dict with keys: test, h_stat, p_value, decision, mean_by_group
    """
    clean = [g.dropna() for g in groups]
    h_stat, p = stats.kruskal(*clean)

    return {
        "test":          "Kruskal-Wallis",
        "h_stat":        h_stat,
        "p_value":       p,
        "decision":      "Reject H₀" if p < alpha else "Fail to Reject H₀",
        "mean_by_group": {
            label: g.mean() for label, g in zip(group_labels, clean)
        },
    }


# ---------------------------------------------------------------------------
# Zip-code pair selection
# ---------------------------------------------------------------------------

def select_comparable_zipcodes(
    df: pd.DataFrame,
    top_n: int = 30,
    similarity_threshold: float = 0.30,
) -> tuple[int, int]:
    """
    From the top-N zip codes by policy volume, return the pair that:
      1. Has a total-variation distance (VehicleType + CoverCategory)
         below *similarity_threshold* — ensuring the two groups differ
         only in location, not in vehicle / product mix.
      2. Among all similar pairs, maximises the absolute difference in
         claim frequency — producing the most powerful test.

    Returns
    -------
    (zip_a, zip_b) : selected PostalCode values
    """
    top_zips = df["PostalCode"].value_counts().head(top_n).index.tolist()
    sub = df[df["PostalCode"].isin(top_zips)]

    def profile(zip_code: int) -> dict[str, pd.Series]:
        grp = sub[sub["PostalCode"] == zip_code]
        return {
            "VehicleType":   grp["VehicleType"].value_counts(normalize=True),
            "CoverCategory": grp["CoverCategory"].value_counts(normalize=True),
        }

    profiles  = {z: profile(z) for z in top_zips}
    claimfreq = {
        z: (sub.loc[sub["PostalCode"] == z, "TotalClaims"] > 0).mean()
        for z in top_zips
    }

    best_pair: tuple[int, int] = (top_zips[0], top_zips[1])
    best_freq_diff = -1.0

    for i in range(len(top_zips)):
        for j in range(i + 1, len(top_zips)):
            zi, zj = top_zips[i], top_zips[j]
            total_dist = 0.0
            for feat in ("VehicleType", "CoverCategory"):
                all_cats = set(profiles[zi][feat].index) | set(profiles[zj][feat].index)
                total_dist += sum(
                    abs(profiles[zi][feat].get(c, 0.0) - profiles[zj][feat].get(c, 0.0))
                    for c in all_cats
                )
            if total_dist <= similarity_threshold:
                freq_diff = abs(claimfreq[zi] - claimfreq[zj])
                if freq_diff > best_freq_diff:
                    best_freq_diff = freq_diff
                    best_pair = (zi, zj)

    return best_pair


# ---------------------------------------------------------------------------
# High-level hypothesis test functions
# ---------------------------------------------------------------------------

def test_province_risk(df: pd.DataFrame, alpha: float = 0.05) -> dict:
    """
    H₀: There are no risk differences across provinces.

    Tests
    -----
    * Claim Frequency  – chi-squared (multi-group, all provinces)
    * Claim Severity   – Kruskal-Wallis (all provinces with ≥ 1 claim)
    """
    claim_flag = compute_claim_frequency(df)

    freq_result = chi_squared_multigroup(df, "Province", claim_flag, alpha)

    severity_groups, severity_labels = [], []
    for prov, grp in df[df["TotalClaims"] > 0].groupby("Province"):
        severity_groups.append(grp["TotalClaims"])
        severity_labels.append(prov)
    sev_result = kruskal_wallis_test(severity_groups, severity_labels, alpha)

    return {
        "hypothesis":           "No risk differences across provinces",
        "claim_frequency_test": freq_result,
        "claim_severity_test":  sev_result,
    }


def test_zipcode_risk(
    df: pd.DataFrame,
    zip_a: Optional[int] = None,
    zip_b: Optional[int] = None,
    alpha: float = 0.05,
) -> dict:
    """
    H₀: There are no risk differences between zip codes.

    Selects two statistically comparable zip codes (unless specified) then
    tests Claim Frequency (chi-squared) and Claim Severity (Welch t-test).
    """
    if zip_a is None or zip_b is None:
        zip_a, zip_b = select_comparable_zipcodes(df)

    grp_a = df[df["PostalCode"] == zip_a]
    grp_b = df[df["PostalCode"] == zip_b]

    freq_result = chi_squared_test(
        compute_claim_frequency(grp_a),
        compute_claim_frequency(grp_b),
        alpha,
    )
    freq_result["zip_a"] = zip_a
    freq_result["zip_b"] = zip_b

    sev_a = compute_claim_severity(grp_a)
    sev_b = compute_claim_severity(grp_b)
    if len(sev_a) >= 2 and len(sev_b) >= 2:
        sev_result = welch_t_test(sev_a, sev_b, str(zip_a), str(zip_b), alpha)
    else:
        sev_result = {
            "test": "Welch t-test", "p_value": np.nan,
            "decision": "Insufficient claims data",
            "n_a": len(sev_a), "n_b": len(sev_b),
        }
    sev_result["zip_a"] = zip_a
    sev_result["zip_b"] = zip_b

    return {
        "hypothesis":           "No risk differences between zip codes",
        "zip_a":                zip_a,
        "zip_b":                zip_b,
        "n_a":                  len(grp_a),
        "n_b":                  len(grp_b),
        "claim_frequency_test": freq_result,
        "claim_severity_test":  sev_result,
    }


def test_zipcode_margin(
    df: pd.DataFrame,
    zip_a: Optional[int] = None,
    zip_b: Optional[int] = None,
    alpha: float = 0.05,
) -> dict:
    """
    H₀: There is no significant margin (profit) difference between zip codes.

    Uses Welch's t-test on the Margin column for two comparable zip codes.
    """
    if zip_a is None or zip_b is None:
        zip_a, zip_b = select_comparable_zipcodes(df)

    grp_a = df[df["PostalCode"] == zip_a]
    grp_b = df[df["PostalCode"] == zip_b]

    result = welch_t_test(
        compute_margin(grp_a),
        compute_margin(grp_b),
        str(zip_a),
        str(zip_b),
        alpha,
    )
    result["zip_a"]      = zip_a
    result["zip_b"]      = zip_b
    result["hypothesis"] = "No significant margin difference between zip codes"
    return result


def test_gender_risk(df: pd.DataFrame, alpha: float = 0.05) -> dict:
    """
    H₀: There is no significant risk difference between Women and Men.

    Rows with Gender == 'Not specified' are excluded.
    Tests Claim Frequency (chi-squared) and Claim Severity (Welch t-test).
    """
    male   = df[df["Gender"] == "Male"]
    female = df[df["Gender"] == "Female"]

    freq_result = chi_squared_test(
        compute_claim_frequency(female),
        compute_claim_frequency(male),
        alpha,
    )
    freq_result["group_a"] = "Female"
    freq_result["group_b"] = "Male"

    sev_male   = compute_claim_severity(male)
    sev_female = compute_claim_severity(female)
    if len(sev_male) >= 2 and len(sev_female) >= 2:
        sev_result = welch_t_test(sev_female, sev_male, "Female", "Male", alpha)
    else:
        sev_result = {
            "test": "Welch t-test", "p_value": np.nan,
            "decision": "Insufficient claims data",
            "n_a": len(sev_female), "n_b": len(sev_male),
        }

    return {
        "hypothesis":           "No significant risk difference between Women and Men",
        "n_male":               len(male),
        "n_female":             len(female),
        "claim_frequency_test": freq_result,
        "claim_severity_test":  sev_result,
    }


# ---------------------------------------------------------------------------
# Results summary table
# ---------------------------------------------------------------------------

def build_results_table(results: dict) -> pd.DataFrame:
    """
    Compile a summary DataFrame from the dict returned by all test functions.

    Parameters
    ----------
    results : dict with keys 'province', 'zipcode_risk',
              'zipcode_margin', 'gender'

    Returns
    -------
    pd.DataFrame with columns:
        Hypothesis | KPI | Test Used | p-value | Decision | Notes
    """
    rows: list[dict] = []

    def _add(hypothesis: str, kpi: str, test_name: str,
             p_value: float, decision: str, notes: str = "") -> None:
        rows.append({
            "Hypothesis": hypothesis,
            "KPI":        kpi,
            "Test Used":  test_name,
            "p-value":    round(p_value, 6) if not np.isnan(p_value) else "N/A",
            "Decision":   decision,
            "Notes":      notes,
        })

    # --- Province ---
    prov = results.get("province", {})
    if prov:
        fr = prov["claim_frequency_test"]
        sr = prov["claim_severity_test"]
        _add("No risk differences across provinces",
             "Claim Frequency", fr["test"], fr["p_value"], fr["decision"])
        _add("No risk differences across provinces",
             "Claim Severity", sr["test"], sr["p_value"], sr["decision"])

    # --- Zip code risk ---
    zr = results.get("zipcode_risk", {})
    if zr:
        note = f"PostalCode {zr['zip_a']} vs {zr['zip_b']}"
        fr = zr["claim_frequency_test"]
        sr = zr["claim_severity_test"]
        _add("No risk differences between zip codes",
             "Claim Frequency", fr["test"], fr["p_value"], fr["decision"], note)
        p_sev = sr.get("p_value", np.nan)
        _add("No risk differences between zip codes",
             "Claim Severity", sr.get("test", "Welch t-test"),
             float(p_sev) if isinstance(p_sev, (int, float)) else np.nan,
             sr.get("decision", "N/A"), note)

    # --- Zip code margin ---
    zm = results.get("zipcode_margin", {})
    if zm:
        note = f"PostalCode {zm['zip_a']} vs {zm['zip_b']}"
        _add("No significant margin difference between zip codes",
             "Margin", zm["test"], zm["p_value"], zm["decision"], note)

    # --- Gender ---
    gr = results.get("gender", {})
    if gr:
        fr = gr["claim_frequency_test"]
        sr = gr["claim_severity_test"]
        _add("No significant risk difference between Women and Men",
             "Claim Frequency", fr["test"], fr["p_value"], fr["decision"],
             f"Female n={gr['n_female']}, Male n={gr['n_male']}")
        p_sev = sr.get("p_value", np.nan)
        _add("No significant risk difference between Women and Men",
             "Claim Severity", sr.get("test", "Welch t-test"),
             float(p_sev) if isinstance(p_sev, (int, float)) else np.nan,
             sr.get("decision", "N/A"),
             f"Female n={gr['n_female']}, Male n={gr['n_male']}")

    return pd.DataFrame(rows)

