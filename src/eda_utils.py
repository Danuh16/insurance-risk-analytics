"""
eda_utils.py
------------
Reusable EDA helper functions: plotting, aggregation, outlier detection.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path

FIGURES_DIR = Path(__file__).resolve().parents[1] / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Colour palette consistent with ACIS brand ──────────────────────────────
PALETTE = {
    "primary": "#1B4F72",
    "accent": "#E74C3C",
    "neutral": "#7F8C8D",
    "positive": "#27AE60",
    "highlight": "#F39C12",
}


def set_style():
    """Apply consistent matplotlib style."""
    plt.rcParams.update({
        "figure.dpi": 120,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.facecolor": "white",
    })


def plot_numerical_distributions(df: pd.DataFrame, cols: list, save: bool = True):
    """
    Histograms + KDE for a list of numerical columns.
    Shows the distribution shape and highlights skewness.
    """
    set_style()
    n = len(cols)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, col in zip(axes, cols):
        data = df[col].dropna()
        ax.hist(data, bins=60, color=PALETTE["primary"], alpha=0.75, edgecolor="white")
        ax.axvline(data.median(), color=PALETTE["accent"], linestyle="--", linewidth=1.5,
                   label=f"Median: {data.median():,.0f}")
        ax.axvline(data.mean(), color=PALETTE["highlight"], linestyle="--", linewidth=1.5,
                   label=f"Mean: {data.mean():,.0f}")
        ax.set_title(f"Distribution of {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
        ax.legend()
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    fig.tight_layout()
    if save:
        path = FIGURES_DIR / "numerical_distributions.png"
        fig.savefig(path, bbox_inches="tight")
    return fig


def plot_categorical_bars(df: pd.DataFrame, col: str, top_n: int = 15,
                          target: str = "TotalPremium", save: bool = True):
    """Bar chart: average TotalPremium and record count by category."""
    set_style()
    grp = (
        df.groupby(col)
        .agg(AvgPremium=(target, "mean"), Count=(target, "count"))
        .sort_values("AvgPremium", ascending=False)
        .head(top_n)
        .reset_index()
    )

    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax2 = ax1.twinx()

    bars = ax1.bar(grp[col], grp["AvgPremium"], color=PALETTE["primary"], alpha=0.8, label="Avg Premium")
    ax2.plot(grp[col], grp["Count"], color=PALETTE["accent"], marker="o", linewidth=2, label="Count")

    ax1.set_title(f"Average {target} by {col} (Top {top_n})")
    ax1.set_xlabel(col)
    ax1.set_ylabel(f"Average {target}", color=PALETTE["primary"])
    ax2.set_ylabel("Policy Count", color=PALETTE["accent"])
    ax1.tick_params(axis="x", rotation=45)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    fig.tight_layout()
    if save:
        path = FIGURES_DIR / f"bar_{col}.png"
        fig.savefig(path, bbox_inches="tight")
    return fig


def plot_loss_ratio_by_group(df: pd.DataFrame, group_col: str, save: bool = True):
    """
    Horizontal bar chart: Loss Ratio by a categorical grouping column.
    Red bars above 1.0 (unprofitable), green bars below.
    """
    set_style()
    grp = (
        df.groupby(group_col)
        .apply(lambda x: x["TotalClaims"].sum() / x["TotalPremium"].sum()
               if x["TotalPremium"].sum() > 0 else np.nan)
        .rename("LossRatio")
        .dropna()
        .sort_values(ascending=True)
    )

    colors = [PALETTE["accent"] if v >= 1 else PALETTE["positive"] for v in grp.values]

    fig, ax = plt.subplots(figsize=(10, max(4, len(grp) * 0.45)))
    ax.barh(grp.index, grp.values, color=colors, edgecolor="white")
    ax.axvline(1.0, color="black", linewidth=1.5, linestyle="--", label="Break-even (LR = 1.0)")
    ax.set_title(f"Loss Ratio by {group_col}")
    ax.set_xlabel("Loss Ratio (TotalClaims / TotalPremium)")
    ax.legend()

    for i, (val, label) in enumerate(zip(grp.values, grp.index)):
        ax.text(val + 0.01, i, f"{val:.3f}", va="center", fontsize=9)

    fig.tight_layout()
    if save:
        path = FIGURES_DIR / f"loss_ratio_{group_col}.png"
        fig.savefig(path, bbox_inches="tight")
    return fig


def plot_scatter_premium_vs_claims(df: pd.DataFrame, hue_col: str = "Province",
                                   sample_n: int = 5000, save: bool = True):
    """Scatter: TotalPremium vs TotalClaims coloured by a category."""
    set_style()
    sample = df[df["TotalPremium"] > 0].sample(min(sample_n, len(df)), random_state=42)

    fig, ax = plt.subplots(figsize=(10, 6))
    categories = sample[hue_col].dropna().unique()
    cmap = plt.cm.get_cmap("tab20", len(categories))

    for i, cat in enumerate(categories):
        sub = sample[sample[hue_col] == cat]
        ax.scatter(sub["TotalPremium"], sub["TotalClaims"], alpha=0.4, s=15,
                   color=cmap(i), label=str(cat))

    # Diagonal reference line (LR = 1)
    max_val = max(sample["TotalPremium"].max(), sample["TotalClaims"].max())
    ax.plot([0, max_val], [0, max_val], "k--", linewidth=1, label="LR = 1.0 (break-even)")

    ax.set_title(f"TotalPremium vs TotalClaims — coloured by {hue_col}")
    ax.set_xlabel("TotalPremium")
    ax.set_ylabel("TotalClaims")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    fig.tight_layout()
    if save:
        path = FIGURES_DIR / f"scatter_premium_claims_{hue_col}.png"
        fig.savefig(path, bbox_inches="tight")
    return fig


def plot_correlation_matrix(df: pd.DataFrame, cols: list, save: bool = True):
    """Heatmap of correlation matrix for selected numerical columns."""
    set_style()
    corr = df[cols].corr()

    fig, ax = plt.subplots(figsize=(len(cols) * 1.2, len(cols) * 1.0))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
        center=0, ax=ax, linewidths=0.5, cbar_kws={"shrink": 0.8}
    )
    ax.set_title("Correlation Matrix — Key Financial & Vehicle Features")
    fig.tight_layout()
    if save:
        path = FIGURES_DIR / "correlation_matrix.png"
        fig.savefig(path, bbox_inches="tight")
    return fig


def plot_boxplots(df: pd.DataFrame, cols: list, save: bool = True):
    """Box plots for outlier detection on numerical columns."""
    set_style()
    fig, axes = plt.subplots(1, len(cols), figsize=(6 * len(cols), 5))
    if len(cols) == 1:
        axes = [axes]

    for ax, col in zip(axes, cols):
        data = df[col].dropna()
        bp = ax.boxplot(data, patch_artist=True,
                        boxprops=dict(facecolor=PALETTE["primary"], alpha=0.6),
                        medianprops=dict(color=PALETTE["accent"], linewidth=2),
                        flierprops=dict(marker=".", color=PALETTE["neutral"], alpha=0.3))
        ax.set_title(f"Box Plot: {col}")
        ax.set_ylabel(col)
        q1, q3 = data.quantile(0.25), data.quantile(0.75)
        iqr = q3 - q1
        n_outliers = ((data < q1 - 1.5 * iqr) | (data > q3 + 1.5 * iqr)).sum()
        ax.set_xlabel(f"Outliers (IQR): {n_outliers:,} ({n_outliers/len(data)*100:.1f}%)")

    fig.tight_layout()
    if save:
        path = FIGURES_DIR / "boxplots_outliers.png"
        fig.savefig(path, bbox_inches="tight")
    return fig


def plot_temporal_trends(df: pd.DataFrame, save: bool = True):
    """
    Line chart: monthly TotalPremium, TotalClaims, and Loss Ratio over time.
    Key creative insight chart.
    """
    set_style()
    monthly = (
        df.groupby("TransactionMonth")
        .agg(TotalPremium=("TotalPremium", "sum"),
             TotalClaims=("TotalClaims", "sum"),
             PolicyCount=("PolicyID", "count"))
        .assign(LossRatio=lambda x: x["TotalClaims"] / x["TotalPremium"])
        .reset_index()
    )

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

    ax1.fill_between(monthly["TransactionMonth"], monthly["TotalPremium"],
                     alpha=0.3, color=PALETTE["primary"], label="Total Premium")
    ax1.plot(monthly["TransactionMonth"], monthly["TotalPremium"],
             color=PALETTE["primary"], linewidth=2)
    ax1.fill_between(monthly["TransactionMonth"], monthly["TotalClaims"],
                     alpha=0.3, color=PALETTE["accent"], label="Total Claims")
    ax1.plot(monthly["TransactionMonth"], monthly["TotalClaims"],
             color=PALETTE["accent"], linewidth=2)
    ax1.set_title("Monthly Premium vs Claims — Feb 2014 to Aug 2015")
    ax1.set_ylabel("Amount (ZAR)")
    ax1.legend()
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"R{x/1e6:.1f}M"))

    ax2.plot(monthly["TransactionMonth"], monthly["LossRatio"],
             color=PALETTE["highlight"], linewidth=2.5, marker="o", markersize=5)
    ax2.axhline(1.0, color="red", linestyle="--", linewidth=1.2, label="Break-even (LR=1.0)")
    ax2.fill_between(monthly["TransactionMonth"], monthly["LossRatio"], 1.0,
                     where=(monthly["LossRatio"] > 1),
                     alpha=0.25, color=PALETTE["accent"], label="Unprofitable zone")
    ax2.fill_between(monthly["TransactionMonth"], monthly["LossRatio"], 1.0,
                     where=(monthly["LossRatio"] <= 1),
                     alpha=0.15, color=PALETTE["positive"], label="Profitable zone")
    ax2.set_title("Monthly Loss Ratio")
    ax2.set_ylabel("Loss Ratio")
    ax2.set_xlabel("Month")
    ax2.legend()

    fig.tight_layout()
    if save:
        path = FIGURES_DIR / "temporal_trends.png"
        fig.savefig(path, bbox_inches="tight")
    return fig


def detect_outliers_iqr(df: pd.DataFrame, col: str) -> pd.Series:
    """Return boolean mask of IQR outliers for a column."""
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    return (df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)


def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """Return a sorted dataframe of missing value counts and percentages."""
    missing = df.isnull().sum()
    pct = (missing / len(df) * 100).round(2)
    report = pd.DataFrame({"Missing Count": missing, "Missing %": pct})
    return report[report["Missing Count"] > 0].sort_values("Missing %", ascending=False)
