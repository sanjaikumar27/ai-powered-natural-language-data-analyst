"""Optional OpenAI-compatible planner; sends no uploaded cell values."""
from __future__ import annotations
import json
import os
import urllib.request
from .nl_query_engine import execute_validated_plan


def plan_with_llm(df, question: str):
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not configured; local analysis remains available.")
    columns = [{"name": str(c), "dtype": str(df[c].dtype), "numeric": bool(__import__("pandas").api.types.is_numeric_dtype(df[c]))} for c in df.columns]
    payload = {
        "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": "Translate a question into one safe data analysis JSON plan. Never write code or SQL. Allowed schemas: {operation:'row_count'}; {operation:'column_stat',measure:<numeric column>,aggregation:'mean|median|sum|min|max'}; {operation:'group_aggregate',group_by:<column>,measure:<numeric column>,aggregation:'sum|mean|median|count',top_n:<optional integer>}. Use only exact provided column names. If unsupported, return {operation:'unsupported'}."},
            {"role": "user", "content": json.dumps({"columns": columns, "question": question}, ensure_ascii=False)}
        ]
    }
    base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    req = urllib.request.Request(base + "/chat/completions", data=json.dumps(payload).encode(), headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            result = json.loads(response.read().decode())
        raw = result["choices"][0]["message"]["content"]
        plan = json.loads(raw)
    except Exception as exc:
        raise RuntimeError(f"LLM request failed: {exc}") from exc
    return execute_validated_plan(df, plan)
