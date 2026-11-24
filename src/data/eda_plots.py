"""
EDA plots for every column of H2.csv.

Run from the project root:
    python -m src.data.eda_plots

Saves the figures to outputs/figures/.
"""
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # works without a display; remove to show plots interactively
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.data.load_data import load_raw_data

sns.set_theme(style="whitegrid")

FIGURES_DIR = Path(__file__).resolve().parents[2] / "outputs" / "figures"


def is_discrete(s: pd.Series, max_unique: int = 15) -> bool:
    """Treat non-numeric or low-cardinality columns as categorical."""
    return (not pd.api.types.is_numeric_dtype(s)) or s.nunique(dropna=True) <= max_unique


def overview(s: pd.Series) -> None:
    print("=" * 60)
    print(f"FEATURE: {s.name}")
    print("=" * 60)
    print(f"dtype            : {s.dtype}")
    print(f"rows             : {len(s):,}")
    print(f"missing          : {s.isna().sum():,} ({s.isna().mean():.2%})")
    print(f"unique values    : {s.nunique(dropna=True):,}")
    print(f"duplicated values: {s.duplicated().sum():,}")


def analyze_categorical(s: pd.Series, out: str) -> None:
    s = s.astype(object).fillna("Missing")  # show missing values as their own bar
    counts = s.value_counts(dropna=False)
    pct = s.value_counts(normalize=True, dropna=False).mul(100).round(2)
    table = pd.DataFrame({"count": counts, "percent": pct})
    print("\nValue counts:")
    print(table.head(20).to_string())

    if len(counts) == 2:
        ratio = counts.max() / counts.min()
        print(f"\nBinary feature. Majority/minority ratio: {ratio:.2f}")
        if ratio > 3:
            print("  -> Noticeably imbalanced; consider this if it is a target variable.")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    top = counts.head(20)
    sns.barplot(x=top.index.astype(str), y=top.values, ax=axes[0], color="#4C72B0")
    axes[0].set_title(f"Counts: {s.name}")
    axes[0].set_xlabel(s.name)
    axes[0].set_ylabel("count")
    axes[0].tick_params(axis="x", rotation=45)
    for i, v in enumerate(top.values):
        axes[0].text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=8)

    axes[1].pie(top.values, labels=top.index.astype(str), autopct="%1.1f%%", startangle=90)
    axes[1].set_title(f"Proportions: {s.name}")
    plt.tight_layout()
    path = os.path.join(out, f"{s.name}_categorical.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\nSaved plot -> {path}")


def analyze_numeric(s: pd.Series, out: str) -> None:
    x = s.dropna()
    desc = x.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    print("\nDescriptive statistics:")
    print(desc.round(3).to_string())
    print(f"\nskewness : {x.skew():.3f}")
    print(f"kurtosis : {x.kurt():.3f}")
    print(f"zeros    : {(x == 0).sum():,} ({(x == 0).mean():.2%})")
    print(f"negatives: {(x < 0).sum():,}")

    # IQR outliers
    q1, q3 = x.quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = x[(x < lo) | (x > hi)]
    print(f"\nIQR outlier bounds: [{lo:.2f}, {hi:.2f}]")
    print(f"outliers: {len(outliers):,} ({len(outliers) / len(x):.2%})")

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    sns.histplot(x, kde=True, bins=40, ax=axes[0, 0], color="#4C72B0")
    axes[0, 0].axvline(x.mean(), color="red", ls="--", label=f"mean={x.mean():.1f}")
    axes[0, 0].axvline(x.median(), color="green", ls="--", label=f"median={x.median():.1f}")
    axes[0, 0].legend()
    axes[0, 0].set_title("Histogram + KDE")

    sns.boxplot(x=x, ax=axes[0, 1], color="#55A868")
    axes[0, 1].set_title("Boxplot")

    sns.violinplot(x=x, ax=axes[1, 0], color="#C44E52")
    axes[1, 0].set_title("Violin plot")

    from scipy import stats
    stats.probplot(x, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title("Q-Q plot (normality)")

    plt.tight_layout()
    path = os.path.join(out, f"{s.name}_numeric.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\nSaved plot -> {path}")


def relationship_to_other_features(df: pd.DataFrame, col: str, out: str, top_n: int = 10) -> None:
    """Quick look at how the feature relates to numeric columns."""
    num = df.select_dtypes(include=np.number)
    if col not in num.columns or num.shape[1] < 2:
        return
    corr = num.corr()[col].drop(col).sort_values(key=np.abs, ascending=False).head(top_n)
    print(f"\nTop {len(corr)} correlations with {col} (Pearson):")
    print(corr.round(3).to_string())

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=corr.values, y=corr.index, ax=ax, color="#4C72B0")
    ax.set_title(f"Correlation with {col}")
    plt.tight_layout()
    path = os.path.join(out, f"{col}_correlations.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved plot -> {path}")


def make_all_plots(df: pd.DataFrame, out: str = str(FIGURES_DIR)) -> None:
    """Run the EDA for every column and save the figures in `out`."""
    os.makedirs(out, exist_ok=True)
    print(f"Dataset shape: {df.shape[0]:,} rows x {df.shape[1]} columns")

    for col in df.columns:
        s = df[col]
        overview(s)

        if is_discrete(s):
            analyze_categorical(s, out)
        else:
            analyze_numeric(s, out)

        relationship_to_other_features(df, col, out)


def main() -> None:
    make_all_plots(load_raw_data())


if __name__ == "__main__":
    main()