"""Self-contained HTML analysis report export."""
from __future__ import annotations
import html
import pandas as pd
from .eda import descriptive_statistics, generate_insights
from .data_cleaning import quality_report


def build_html_report(df: pd.DataFrame, answers: list[dict] | None = None, charts: list[tuple[str, object]] | None = None) -> str:
    quality = quality_report(df)
    stats = descriptive_statistics(df)
    insights = generate_insights(df)
    esc = html.escape
    parts = ["<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width'><title>Data Analysis Report</title><style>body{font:16px/1.6 system-ui,sans-serif;max-width:1100px;margin:40px auto;padding:0 20px;color:#172033}h1,h2{color:#163a5f}table{border-collapse:collapse;width:100%;margin:16px 0;font-size:14px}th,td{border:1px solid #dbe3eb;padding:8px;text-align:left}th{background:#eff5fa}li{margin:6px 0}.meta{color:#526273}.card{background:#f5f8fb;padding:14px 18px;border-radius:10px}</style></head><body>"]
    parts.append("<h1>AI-Powered Natural Language Data Analyst</h1><p class='meta'>Generated locally from the uploaded dataset. No data has been sent to an external service by this report generator.</p>")
    parts.append(f"<div class='card'><b>Dataset dimensions:</b> {quality['rows']:,} rows × {quality['columns']:,} columns &nbsp; <b>Missing cells:</b> {quality['missing_cells']:,} &nbsp; <b>Duplicate rows:</b> {quality['duplicate_rows']:,}</div>")
    parts.append("<h2>Column types</h2><ul>" + "".join(f"<li>{esc(str(c))}: {esc(str(df[c].dtype))}</li>" for c in df.columns) + "</ul>")
    if not stats.empty: parts.append("<h2>Descriptive statistics</h2>" + stats.to_html(border=0, na_rep="—", float_format=lambda x: f"{x:,.4g}"))
    parts.append("<h2>Computed insights</h2><ul>" + "".join(f"<li>{esc(i)}</li>" for i in insights) + "</ul>")
    if answers:
        parts.append("<h2>Questions and answers</h2>")
        for item in answers:
            parts.append(f"<div class='card'><b>Q:</b> {esc(item.get('question',''))}<br><b>A:</b> {esc(item.get('answer',''))}<br><span class='meta'>{esc(item.get('explanation',''))}</span></div>")
    for title, fig in charts or []:
        parts.append(f"<h2>{esc(title)}</h2>")
        try: parts.append(fig.to_html(full_html=False, include_plotlyjs="cdn"))
        except Exception: pass
    parts.append(f"<footer class='meta'>Report generated from {len(df):,} uploaded rows.</footer></body></html>")
    return "\n".join(parts)
