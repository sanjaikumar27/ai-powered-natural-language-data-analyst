"""Dataset summaries and evidence-backed insight generation."""
from __future__ import annotations
import pandas as pd
from .data_cleaning import quality_report


def descriptive_statistics(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df.select_dtypes(include="number")
    if numeric.empty:
        return pd.DataFrame(index=["count", "mean", "median", "mode", "std", "min", "max"])
    stats = numeric.describe().T
    stats["median"] = numeric.median()
    modes = numeric.mode(dropna=True)
    stats["mode"] = [modes[col].iloc[0] if col in modes and not modes[col].dropna().empty else None for col in numeric.columns]
    return stats[["count", "mean", "median", "mode", "std", "min", "max"]]


def correlation_pairs(df: pd.DataFrame, threshold: float = 0.7) -> pd.DataFrame:
    numeric = df.select_dtypes(include="number")
    if numeric.shape[1] < 2:
        return pd.DataFrame(columns=["column_1", "column_2", "correlation"])
    corr = numeric.corr(numeric_only=True)
    rows = []
    for i, a in enumerate(corr.columns):
        for b in corr.columns[i+1:]:
            value = corr.loc[a, b]
            if pd.notna(value) and abs(value) >= threshold:
                rows.append({"column_1": a, "column_2": b, "correlation": float(value)})
    return pd.DataFrame(rows).sort_values("correlation", key=lambda s: s.abs(), ascending=False) if rows else pd.DataFrame(columns=["column_1", "column_2", "correlation"])


def generate_insights(df: pd.DataFrame) -> list[str]:
    insights: list[str] = []
    numeric = df.select_dtypes(include="number")
    categorical = df.select_dtypes(exclude="number")
    if df.empty:
        return ["The dataset has no rows to analyze."]
    insights.append(f"The dataset contains {len(df):,} rows and {len(df.columns):,} columns.")
    for col in numeric.columns:
        series = numeric[col].dropna()
        if series.empty:
            continue
        if series.nunique() > 1:
            low, high = series.min(), series.max()
            insights.append(f"{col}: observed values range from {low:,.3g} to {high:,.3g}; mean {series.mean():,.3g}, median {series.median():,.3g} (computed from {len(series):,} non-missing rows).")
    for col in categorical.columns:
        counts = df[col].dropna().astype(str).value_counts()
        if len(counts):
            insights.append(f"{col}: most frequent value is {counts.index[0]!r} ({int(counts.iloc[0]):,} rows).")
    corr = correlation_pairs(df)
    if not corr.empty:
        row = corr.iloc[0]
        insights.append(f"{row['column_1']} and {row['column_2']} have a Pearson correlation of {row['correlation']:.2f}; this is association, not evidence of causation.")
    quality = quality_report(df)
    if quality["missing_cells"]:
        insights.append(f"{quality['missing_cells']:,} cells are missing across {len(quality['missing_by_column'])} columns.")
    if quality["duplicate_rows"]:
        insights.append(f"{quality['duplicate_rows']:,} duplicate rows were detected.")
    return insights[:15]
