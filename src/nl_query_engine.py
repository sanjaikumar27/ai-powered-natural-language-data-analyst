"""Safe natural-language query planner. No eval, exec, or generated code is used."""
from __future__ import annotations
from dataclasses import dataclass
import re
import pandas as pd

@dataclass
class QueryResult:
    answer: str
    explanation: str
    data: pd.DataFrame | None = None
    chart_type: str | None = None
    columns_used: list[str] | None = None
    operation: str | None = None


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def _find_column(text: str, columns) -> str | None:
    ntext = f" {_norm(text)} "
    matches = []
    for col in columns:
        nc = _norm(col)
        variants = {nc}
        if nc.endswith("y"):
            variants.add(nc[:-1] + "ies")
        elif nc.endswith("s"):
            variants.add(nc[:-1])
        else:
            variants.add(nc + "s")
        if any(v and f" {v} " in ntext for v in variants):
            matches.append((len(nc), str(col)))
    return max(matches)[1] if matches else None


def _top_n(text: str) -> int | None:
    match = re.search(r"\btop\s+(\d+)\b", _norm(text))
    if match:
        return min(int(match.group(1)), 100)
    words = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
             "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
    match = re.search(r"\btop\s+(one|two|three|four|five|six|seven|eight|nine|ten)\b", _norm(text))
    return words[match.group(1)] if match else None


def answer_question(df: pd.DataFrame, question: str) -> QueryResult:
    q = question.strip()
    if not q:
        return QueryResult("Please enter a question.", "A question is required.")
    qn = _norm(q)
    cols = list(df.columns)
    measure = _find_column(q, df.select_dtypes(include="number").columns)
    group = _find_column(q, cols)
    if group == measure:
        group = None
    # Prefer the categorical column explicitly referenced alongside a measure.
    if measure:
        other = [c for c in cols if c != measure and _find_column(q, [c])]
        group = other[0] if other else None

    if re.search(r"\b(unusual|outlier|outliers|anomal|extreme values)\b", qn):
        values = []
        for col in df.select_dtypes(include="number").columns:
            s = df[col].dropna()
            if len(s) >= 4:
                q1, q3 = s.quantile([.25, .75]); iqr = q3 - q1
                count = int(((s < q1-1.5*iqr) | (s > q3+1.5*iqr)).sum())
                if count: values.append({"column": col, "outlier_count": count})
        out = pd.DataFrame(values)
        total = int(out["outlier_count"].sum()) if not out.empty else 0
        return QueryResult(f"Detected {total:,} IQR-based outlier value(s) across {len(out)} numeric column(s).", "Outliers are values outside 1.5×IQR fences; this is a screening heuristic, not a claim that a value is erroneous.", out, "bar" if not out.empty else None, list(out.columns), "IQR outlier scan")

    if re.search(r"\b(correlat|relationship between)\b", qn):
        num = df.select_dtypes(include="number")
        if num.shape[1] < 2:
            return QueryResult("At least two numeric columns are needed to compute correlations.", "No correlation matrix can be calculated for this dataset.")
        corr = num.corr(numeric_only=True)
        return QueryResult("Correlation matrix calculated from numeric columns. Correlation does not imply causation.", "The values are Pearson correlations using pairwise complete numeric observations.", corr.reset_index(names="column"), "heatmap", list(num.columns), "Pearson correlation matrix")

    date_col = _find_column(q, cols)
    if re.search(r"\b(month|monthly|trend|over time|by date|daily|yearly)\b", qn):
        if date_col is None or not ("date" in _norm(date_col) or "time" in _norm(date_col) or pd.api.types.is_datetime64_any_dtype(df[date_col])):
            date_candidates = [c for c in cols if "date" in _norm(c) or "time" in _norm(c) or pd.api.types.is_datetime64_any_dtype(df[c])]
            date_col = date_candidates[0] if date_candidates else None
    if re.search(r"\b(month|monthly|trend|over time|by date|daily|yearly)\b", qn) and date_col and ("date" in _norm(date_col) or pd.api.types.is_datetime64_any_dtype(df[date_col]) or "month" in qn):
        dates = pd.to_datetime(df[date_col], errors="coerce")
        if dates.notna().any():
            if measure is None:
                return QueryResult("Name a numeric measure to summarize over time, for example 'monthly sales trend'.", "The date column was identified, but no numeric measure was selected.")
            temp = pd.DataFrame({"period": dates.dt.to_period("M").astype(str), measure: pd.to_numeric(df[measure], errors="coerce")})
            grouped = temp.dropna().groupby("period", as_index=False)[measure].sum().sort_values("period")
            return QueryResult(f"{measure} totals across {len(grouped)} monthly period(s).", f"Converted {date_col} to calendar months and summed {measure} for each month.", grouped, "line", [date_col, measure], "monthly sum")

    agg = None
    if re.search(r"\b(average|avg|mean)\b", qn): agg = "mean"
    elif re.search(r"\b(total|sum|revenue|sales|profit)\b", qn): agg = "sum"
    elif re.search(r"\b(highest|largest|maximum|max|most)\b", qn): agg = "max"
    elif re.search(r"\b(lowest|smallest|minimum|min|least)\b", qn): agg = "min"
    elif re.search(r"\b(count|how many|number of)\b", qn): agg = "count"
    elif re.search(r"\b(median)\b", qn): agg = "median"

    if agg in {"sum", "mean", "median", "count"} and re.search(r"\b(by|per|each|for every|category|region|product)\b", qn):
        if group is None:
            categorical = [c for c in df.select_dtypes(include=["object", "category", "bool"]).columns if c != measure]
            if not categorical:
                categorical = [c for c in df.select_dtypes(exclude="number").columns if c != measure and "date" not in _norm(c) and "time" not in _norm(c)]
            group = categorical[0] if categorical else None
        if group is None:
            return QueryResult("I couldn't identify a grouping column. Mention a category, region, or other column name.", "Grouped aggregation needs a category column.")
        if agg != "count" and measure is None:
            return QueryResult("I couldn't identify a numeric measure. Include a column name such as sales or revenue.", "The requested aggregation needs a numeric column.")
        if agg == "count": result = df.groupby(group, dropna=False).size().rename("count").reset_index()
        else: result = df.groupby(group, dropna=False)[measure].agg(agg).reset_index()
        n = _top_n(q)
        if n: result = result.nlargest(n, "count" if agg == "count" else measure)
        else: result = result.sort_values("count" if agg == "count" else measure, ascending=False)
        label = {"sum":"sum", "mean":"average", "median":"median", "count":"count"}[agg]
        top = result.iloc[0] if not result.empty else None
        detail = f" Top group: {top[group]!r} ({top['count'] if agg == 'count' else top[measure]:,.3g})." if top is not None else ""
        measure_text = "rows" if agg == "count" else measure
        return QueryResult(f"Computed {label} {measure_text} for {len(result)} group(s).{detail}", f"Grouped by {group} and applied {label} to {measure_text}. Values are computed from the uploaded data.", result, "bar", [group] + ([] if agg == "count" else [measure]), f"groupby {label}")

    if agg in {"max", "min"} and measure is not None:
        n = _top_n(q)
        if group and group != measure:
            grouped = df.groupby(group, dropna=False)[measure].sum().reset_index(name=measure)
            result = grouped.nlargest(n if n else 1, measure) if agg == "max" else grouped.nsmallest(n if n else 1, measure)
            row = result.iloc[0]
            return QueryResult(f"{group} with {'highest' if agg == 'max' else 'lowest'} total {measure}: {row[group]} ({row[measure]:,.3g}).", f"Summed {measure} by {group} and selected the {'largest' if agg == 'max' else 'smallest'} result.", result, "bar", [group, measure], "grouped sum then rank")
        series = pd.to_numeric(df[measure], errors="coerce")
        idx = series.idxmax() if agg == "max" else series.idxmin()
        row = df.loc[idx]
        out = pd.DataFrame([row])
        return QueryResult(f"{measure} {'maximum' if agg == 'max' else 'minimum'} is {series.loc[idx]:,.3g}.", f"Selected the row with the {'largest' if agg == 'max' else 'smallest'} value in {measure}.", out, None, [measure], f"column {agg}")

    if agg == "count" or re.search(r"\b(how many|number of rows|row count|record count)\b", qn):
        return QueryResult(f"The dataset contains {len(df):,} rows.", "Counted the uploaded data rows.", None, None, [], "row count")
    if measure and re.search(r"\b(average|avg|mean|median|total|sum)\b", qn):
        series = pd.to_numeric(df[measure], errors="coerce").dropna()
        if series.empty: return QueryResult(f"No numeric values were found in {measure}.", "The selected column contains no usable numeric observations.")
        fn = "mean" if re.search(r"\b(average|avg|mean)\b", qn) else "median" if "median" in qn else "sum"
        value = getattr(series, fn)()
        return QueryResult(f"{fn.title()} {measure}: {value:,.4g}.", f"Applied {fn} to {len(series):,} non-missing values in {measure}.", None, None, [measure], fn)
    return QueryResult("I couldn't safely map that question to a supported analysis.", "Try a question such as 'What is the average revenue by region?', 'Top 5 products by sales', 'Which category has the highest profit?', 'Show monthly sales trend', or 'Are there unusual values?'.")


def execute_validated_plan(df: pd.DataFrame, plan: dict) -> QueryResult:
    """Execute a small allowlisted plan produced by an optional LLM; reject anything else."""
    allowed = {"group_aggregate", "column_stat", "row_count"}
    if not isinstance(plan, dict) or plan.get("operation") not in allowed:
        raise ValueError("Unsupported analysis plan.")
    op = plan["operation"]
    if op == "row_count":
        return QueryResult(f"The dataset contains {len(df):,} rows.", "Counted uploaded rows.", operation=op)
    if op == "column_stat":
        col, agg = plan.get("measure"), plan.get("aggregation")
        if col not in df.select_dtypes(include="number").columns or agg not in {"mean", "median", "sum", "min", "max"}:
            raise ValueError("Invalid numeric column or statistic in analysis plan.")
        value = getattr(df[col].dropna(), agg)()
        return QueryResult(f"{agg.title()} {col}: {value:,.4g}.", f"Applied {agg} to non-missing values in {col}.", columns_used=[col], operation=op)
    group, measure = plan.get("group_by"), plan.get("measure")
    agg = plan.get("aggregation")
    if group not in df.columns or measure not in df.select_dtypes(include="number").columns or agg not in {"sum", "mean", "median", "count"}:
        raise ValueError("Invalid group, numeric measure, or aggregation in analysis plan.")
    series = df.groupby(group, dropna=False)[measure]
    result = (df.groupby(group, dropna=False).size().rename("count").reset_index() if agg == "count" else series.agg(agg).reset_index())
    n = plan.get("top_n")
    if n is not None:
        if not isinstance(n, int) or n < 1 or n > 100: raise ValueError("top_n must be an integer from 1 to 100.")
        result = result.nlargest(n, "count" if agg == "count" else measure)
    return QueryResult(f"Computed {agg} {measure} by {group}.", f"Allowlisted group aggregation: {agg} of {measure}, grouped by {group}.", result, "bar", [group, measure], op)
