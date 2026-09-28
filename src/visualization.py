"""Plotly figure helpers for tabular EDA and query results."""
from __future__ import annotations
import pandas as pd
import plotly.express as px


def automatic_charts(df: pd.DataFrame) -> list[tuple[str, object]]:
    charts = []
    numeric = list(df.select_dtypes(include="number").columns)
    categorical = list(df.select_dtypes(exclude="number").columns)
    if numeric:
        charts.append((f"Distribution of {numeric[0]}", px.histogram(df, x=numeric[0], nbins=30, title=f"Distribution of {numeric[0]}", template="plotly_white")))
        charts.append((f"Outliers in {numeric[0]}", px.box(df, y=numeric[0], title=f"Spread of {numeric[0]}", template="plotly_white")))
    if categorical and numeric:
        grouped = df.groupby(categorical[0], dropna=False)[numeric[0]].sum().reset_index().sort_values(numeric[0], ascending=False).head(20)
        charts.append((f"{numeric[0]} by {categorical[0]}", px.bar(grouped, x=categorical[0], y=numeric[0], title=f"{numeric[0]} by {categorical[0]} (top 20)", template="plotly_white")))
    if len(numeric) >= 2:
        charts.append(("Numeric correlation heatmap", px.imshow(df[numeric].corr(numeric_only=True), text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title="Numeric correlation matrix", template="plotly_white")))
    if len(categorical) >= 1 and len(numeric) == 0:
        counts = df[categorical[0]].astype(str).value_counts().head(20).rename_axis(categorical[0]).reset_index(name="count")
        charts.append((f"Counts by {categorical[0]}", px.bar(counts, x=categorical[0], y="count", title="Top categories by row count", template="plotly_white")))
    return charts


def query_figure(result, x_col: str | None = None, y_col: str | None = None, chart_type: str | None = None):
    data = result.data
    if data is None or data.empty or len(data.columns) < 2:
        return None
    x = x_col if x_col in data.columns else data.columns[0]
    y = y_col if y_col in data.columns else next((c for c in data.columns if c != x and pd.api.types.is_numeric_dtype(data[c])), None)
    kind = chart_type or result.chart_type or "bar"
    if kind == "heatmap" and "column" in data.columns:
        matrix = data.set_index("column")
        return px.imshow(matrix, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title="Correlation heatmap", template="plotly_white")
    if not y: return None
    if kind == "line": return px.line(data, x=x, y=y, markers=True, template="plotly_white")
    if kind == "scatter": return px.scatter(data, x=x, y=y, template="plotly_white")
    if kind == "pie": return px.pie(data, names=x, values=y, template="plotly_white")
    if kind == "histogram": return px.histogram(data, x=y, template="plotly_white")
    if kind == "box": return px.box(data, x=x, y=y, template="plotly_white")
    return px.bar(data, x=x, y=y, template="plotly_white")
