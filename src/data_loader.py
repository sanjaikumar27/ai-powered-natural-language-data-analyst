"""Validated readers for user-provided CSV and Excel files."""
from __future__ import annotations
from io import BytesIO
import re
import pandas as pd

MAX_ROWS = 1_000_000
MAX_COLUMNS = 250


def load_dataset(name: str, content: bytes) -> pd.DataFrame:
    if not content:
        raise ValueError("The uploaded file is empty.")
    suffix = name.lower().rsplit(".", 1)[-1] if "." in name else ""
    try:
        if suffix == "csv":
            df = pd.read_csv(BytesIO(content), low_memory=False)
        elif suffix in {"xlsx", "xlsm", "xls"}:
            df = pd.read_excel(BytesIO(content))
        else:
            raise ValueError("Unsupported file type. Please upload a CSV or Excel file.")
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Could not read this file: {exc}") from exc
    if df.empty and len(df.columns) == 0:
        raise ValueError("No tabular data was found in the uploaded file.")
    if len(df) > MAX_ROWS or len(df.columns) > MAX_COLUMNS:
        raise ValueError(f"Dataset is too large. Maximum supported size is {MAX_ROWS:,} rows and {MAX_COLUMNS} columns.")
    # Make headers unique and usable without changing their meaning.
    seen: dict[str, int] = {}
    headers = []
    for i, column in enumerate(df.columns):
        base = str(column).strip() or f"column_{i+1}"
        pandas_duplicate = re.match(r"^(.*)\.(\d+)$", base)
        if pandas_duplicate and pandas_duplicate.group(1) in seen:
            base = pandas_duplicate.group(1)
        seen[base] = seen.get(base, 0) + 1
        headers.append(base if seen[base] == 1 else f"{base}_{seen[base]}")
    df.columns = headers
    return df
