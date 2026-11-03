"""Data quality checks for H2.csv (City Hotel).

Run: python -m src.data.quality
Saves six tables to outputs/tables/.
"""
from pathlib import Path

import pandas as pd

from src.data.load_data import load_raw_data

TABLES_DIR = Path("outputs/tables")


def _save(table: pd.DataFrame, name: str) -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLES_DIR / name, index=False)
    print(f"saved {TABLES_DIR / name}")


def column_types(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"column": df.columns,
                         "dtype": df.dtypes.astype(str).values})


def summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    stats = df.select_dtypes(include="number").describe().T
    stats.index.name = "column"
    return stats.reset_index()


def missing_values(df: pd.DataFrame) -> pd.DataFrame:
    count = df.isna().sum()
    table = pd.DataFrame({
        "column": df.columns,
        "missing_count": count.values,
        "missing_percent": (count / len(df) * 100).round(2).values,
    })
    return table.sort_values("missing_count", ascending=False)


def duplicates(df: pd.DataFrame) -> pd.DataFrame:
    n_dup = int(df.duplicated().sum())
    return pd.DataFrame({
        "total_rows": [len(df)],
        "duplicate_rows": [n_dup],
        "duplicate_percent": [round(n_dup / len(df) * 100, 2)],
    })


def unusual_values(df: pd.DataFrame) -> pd.DataFrame:
    guests = df["Adults"] + df["Children"].fillna(0) + df["Babies"]
    nights = df["StaysInWeekendNights"] + df["StaysInWeekNights"]
    checks = {
        "ADR == 0": df["ADR"] == 0,
        "ADR < 0": df["ADR"] < 0,
        "ADR > 1000": df["ADR"] > 1000,
        "Adults == 0": df["Adults"] == 0,
        "Total guests == 0": guests == 0,
        "Total nights == 0": nights == 0,
        "Total nights > 30": nights > 30,
        "LeadTime > 365": df["LeadTime"] > 365,
        "Babies > 0 and Adults == 0": (df["Babies"] > 0) & (df["Adults"] == 0),
        "Children missing": df["Children"].isna(),
        "DaysInWaitingList > 0": df["DaysInWaitingList"] > 0,
        "BookingChanges > 5": df["BookingChanges"] > 5,
        "RequiredCarParkingSpaces > 2": df["RequiredCarParkingSpaces"] > 2,
    }
    rows = [{"check": name, "rows": int(mask.sum()),
             "percent": round(mask.sum() / len(df) * 100, 2)}
            for name, mask in checks.items()]
    return pd.DataFrame(rows)


def cardinality(df: pd.DataFrame) -> pd.DataFrame:
    text_cols = df.select_dtypes(exclude="number").columns
    return pd.DataFrame({
        "column": text_cols,
        "unique_values": [df[c].nunique() for c in text_cols],
    }).sort_values("unique_values", ascending=False)


def quality_report(df: pd.DataFrame) -> None:
    _save(column_types(df), "column_types.csv")
    _save(summary_statistics(df), "summary_statistics.csv")
    _save(missing_values(df), "missing_values.csv")
    _save(duplicates(df), "duplicates.csv")
    _save(unusual_values(df), "unusual_values.csv")
    _save(cardinality(df), "cardinality.csv")


if __name__ == "__main__":
    quality_report(load_raw_data())