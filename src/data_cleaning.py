"""Data-quality checks and opt-in cleaning transformations."""
from __future__ import annotations
import pandas as pd


def quality_report(df: pd.DataFrame) -> dict:
    missing = df.isna().sum()
    numeric = df.select_dtypes(include="number")
    outliers: dict[str, int] = {}
    for col in numeric.columns:
        series = numeric[col].dropna()
        if len(series) < 4:
            outliers[col] = 0
            continue
        q1, q3 = series.quantile([0.25, 0.75])
        iqr = q3 - q1
        outliers[col] = int(((series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)).sum())
    return {
        "rows": int(len(df)), "columns": int(len(df.columns)),
        "missing_by_column": {str(k): int(v) for k, v in missing.items() if v > 0},
        "missing_cells": int(missing.sum()), "duplicate_rows": int(df.duplicated().sum()),
        "outliers_by_column": outliers,
    }


def clean_dataset(df: pd.DataFrame, *, drop_duplicates: bool = False,
                  missing_strategy: str = "keep", drop_empty_columns: bool = False) -> pd.DataFrame:
    result = df.copy()
    if drop_empty_columns:
        result = result.dropna(axis=1, how="all")
    if drop_duplicates:
        result = result.drop_duplicates().reset_index(drop=True)
    if missing_strategy == "drop_rows":
        result = result.dropna().reset_index(drop=True)
    elif missing_strategy in {"fill_median_mode", "fill_zero"}:
        for col in result.columns:
            if not result[col].isna().any():
                continue
            if missing_strategy == "fill_zero" and pd.api.types.is_numeric_dtype(result[col]):
                result[col] = result[col].fillna(0)
            elif pd.api.types.is_numeric_dtype(result[col]):
                median = result[col].median()
                result[col] = result[col].fillna(median)
            else:
                mode = result[col].mode(dropna=True)
                result[col] = result[col].fillna(mode.iloc[0] if not mode.empty else "Unknown")
    elif missing_strategy != "keep":
        raise ValueError("Unknown missing-value strategy.")
    return result
