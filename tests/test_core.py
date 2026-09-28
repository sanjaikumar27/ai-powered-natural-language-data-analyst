from io import BytesIO
import pandas as pd
import pytest
from src.data_loader import load_dataset
from src.data_cleaning import clean_dataset, quality_report
from src.nl_query_engine import answer_question, execute_validated_plan
from src.visualization import automatic_charts
from src.report_generator import build_html_report
from src.ml_engine import train_predictor, cluster_records


def sample():
    return pd.DataFrame({"product":["A","B","A","C","B","C","A","B","C","A","B","C","A","B","C","A"], "region":["N","S"]*8, "sales":[10,20,15,30,25,35,12,22,18,32,27,40,14,24,19,38], "profit":[2,4,3,6,5,7,2,5,3,7,5,8,3,5,4,8], "when":pd.date_range("2025-01-01", periods=16, freq="D"), "missing":[None if i==0 else float(i) for i in range(16)]})


def test_csv_upload_and_headers():
    data = b"sales,sales,region\n1,2,N\n3,4,S\n"
    df = load_dataset("x.csv", data)
    assert list(df.columns) == ["sales", "sales_2", "region"]
    assert df.shape == (2,3)


def test_excel_upload():
    b = BytesIO(); sample().to_excel(b, index=False)
    df = load_dataset("sample.xlsx", b.getvalue())
    assert len(df) == 16 and "sales" in df.columns


def test_rejects_invalid_upload():
    with pytest.raises(ValueError): load_dataset("bad.json", b"{}")
    with pytest.raises(ValueError): load_dataset("x.csv", b"")


def test_quality_and_cleaning():
    df = pd.DataFrame({"x":[1,2,2,None], "cat":["a","a","a","b"], "empty":[None]*4})
    q = quality_report(df)
    assert q["duplicate_rows"] == 1 and q["missing_cells"] == 5
    cleaned = clean_dataset(df, drop_duplicates=True, missing_strategy="fill_median_mode", drop_empty_columns=True)
    assert len(cleaned) == 3 and "empty" not in cleaned.columns and cleaned.isna().sum().sum() == 0


def test_grouped_question_calculated_from_rows():
    r = answer_question(sample(), "What is the average sales by region?")
    assert "Computed average sales" in r.answer
    expected = sample().groupby("region").sales.mean().max()
    assert r.data["sales"].max() == pytest.approx(expected)
    assert r.columns_used == ["region", "sales"]


def test_top_category_question_and_unsupported_question():
    r = answer_question(sample(), "Which product has the highest sales?")
    assert r.data.iloc[0]["product"] == "C"
    top = answer_question(sample(), "Show the top five products by sales")
    assert len(top.data) == 3
    unclear = answer_question(sample(), "What is the average revenue for each region?")
    assert "numeric measure" in unclear.answer
    bad = answer_question(sample(), "Predict the next lottery number")
    assert "couldn't safely map" in bad.answer


def test_monthly_trend_infers_date_column():
    df = pd.DataFrame({"order_date": pd.to_datetime(["2025-01-02", "2025-01-18", "2025-02-03"]), "sales": [2, 3, 7]})
    result = answer_question(df, "Show monthly sales trend")
    assert result.data.to_dict("records") == [{"period": "2025-01", "sales": 5}, {"period": "2025-02", "sales": 7}]


def test_safe_plan_allowlist():
    r = execute_validated_plan(sample(), {"operation":"group_aggregate", "group_by":"region", "measure":"sales", "aggregation":"sum", "top_n":2})
    assert len(r.data) == 2
    with pytest.raises(ValueError): execute_validated_plan(sample(), {"operation":"anything", "code":"import os"})


def test_charts_and_report():
    charts = automatic_charts(sample())
    assert len(charts) >= 3
    report = build_html_report(sample(), [{"question":"test?", "answer":"yes"}], charts[:1])
    assert "Descriptive statistics" in report and "test?" in report and "plotly" in report.lower()


def test_regression_and_classification_evaluation():
    df = sample()
    reg = train_predictor(df, "sales", ["profit", "product", "region"], "regression")
    assert "mean_absolute_error" in reg["metrics"] and reg["test_rows"] > 0
    cls_df = df.copy(); cls_df["class"] = ["high" if x >= 25 else "low" for x in cls_df.sales]
    clf = train_predictor(cls_df, "class", ["profit", "product", "region"], "classification")
    assert "accuracy" in clf["metrics"] and "weighted_f1" in clf["metrics"]
    with pytest.raises(ValueError): train_predictor(df, "sales", ["sales"], "regression")


def test_clustering():
    out = cluster_records(sample(), ["sales", "profit"], 2)
    assert "cluster" in out and set(out["cluster"].unique()) <= {0,1}
