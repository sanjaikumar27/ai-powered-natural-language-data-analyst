from __future__ import annotations
from io import BytesIO
from pathlib import Path
import os
import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

from src.data_loader import load_dataset
from src.data_cleaning import clean_dataset, quality_report
from src.eda import descriptive_statistics, correlation_pairs, generate_insights
from src.nl_query_engine import answer_question
from src.llm_planner import plan_with_llm
from src.visualization import automatic_charts, query_figure
from src.ml_engine import train_predictor, cluster_records
from src.report_generator import build_html_report

ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env")
st.set_page_config(page_title="Data Analyst AI", page_icon="◈", layout="wide", initial_sidebar_state="expanded")
st.markdown("""<style>
:root{--ink:#14223a;--muted:#63748b;--blue:#2767d4;--line:#e4eaf2}
.block-container{padding-top:2rem;padding-bottom:3rem;max-width:1480px}
[data-testid="stSidebar"]{background:#f5f8fc;border-right:1px solid #e3eaf2}
.hero{padding:1.7rem 2rem;border-radius:18px;background:linear-gradient(118deg,#102a43 0%,#174d75 58%,#2767d4 100%);color:white;margin-bottom:1.2rem}
.hero h1{font-size:2.15rem;line-height:1.2;margin:0 0 .45rem;color:white}.hero p{color:#d9e9fb;margin:0;font-size:1rem}
.eyebrow{text-transform:uppercase;letter-spacing:.13em;font-size:.72rem;font-weight:700;color:#b9d9ff;margin-bottom:.5rem}
[data-testid="stMetric"]{background:white;border:1px solid var(--line);padding:1rem;border-radius:14px;box-shadow:0 4px 14px #183d6510}
[data-testid="stMetricLabel"]{color:var(--muted)}
section[data-testid="stTabs"] button{font-weight:650}
.small-note{color:#66788f;font-size:.88rem}
</style>""", unsafe_allow_html=True)

if "raw_df" not in st.session_state: st.session_state.raw_df = None
if "clean_df" not in st.session_state: st.session_state.clean_df = None
if "qa_history" not in st.session_state: st.session_state.qa_history = []
if "saved_charts" not in st.session_state: st.session_state.saved_charts = []

with st.sidebar:
    st.markdown("## ◈ Data Analyst AI")
    st.caption("Explore data. Ask better questions.")
    st.divider()
    uploaded = st.file_uploader("Upload a dataset", type=["csv", "xlsx", "xls", "xlsm"], help="Files are processed in this app session. Limits: 1,000,000 rows and 250 columns.")
    sample_button = st.button("Load sample sales dataset", use_container_width=True)
    if uploaded is not None:
        try:
            new_df = load_dataset(uploaded.name, uploaded.getvalue())
            if st.session_state.get("active_upload") != (uploaded.name, uploaded.size):
                st.session_state.raw_df = new_df
                st.session_state.clean_df = new_df.copy()
                st.session_state.qa_history = []
                st.session_state.saved_charts = []
                st.session_state.active_upload = (uploaded.name, uploaded.size)
                st.toast(f"Loaded {len(new_df):,} rows")
        except ValueError as exc:
            st.error(str(exc))
    if sample_button:
        st.session_state.raw_df = pd.read_csv(ROOT / "data" / "sample_sales.csv", parse_dates=["order_date"])
        st.session_state.clean_df = st.session_state.raw_df.copy()
        st.session_state.qa_history = []
        st.session_state.saved_charts = []
        st.session_state.active_upload = None
    st.divider()
    page = st.radio("Workspace", ["Overview", "Clean data", "Ask your data", "Visualizations", "Machine learning", "Report"], label_visibility="collapsed")
    st.divider()
    st.markdown("**Analysis mode**")
    use_llm = st.checkbox("Use optional AI question planner", value=False, help="If checked, the question and column names/types are sent to the configured OpenAI-compatible API. Uploaded rows and cell values are not sent.")
    if use_llm:
        st.warning("Only your question and schema are sent to the configured API—not the dataset rows. Verify your provider's data policy.")
    st.caption("Local analysis works without an API key.")


df = st.session_state.clean_df
st.markdown("<div class='hero'><div class='eyebrow'>Your private workspace · Local-first analysis</div><h1>AI-Powered Natural Language Data Analyst</h1><p>Explore files, ask questions in plain English, and turn real data into clear decisions.</p></div>", unsafe_allow_html=True)
if df is None:
    left, right = st.columns([1.2, .8])
    with left:
        st.subheader("From raw file to clear answers")
        st.write("Upload a CSV or Excel file—or load the included demo—to start exploring. Core analysis runs locally; the optional AI planner shares only the question and column schema after you explicitly enable it.")
        st.markdown("- **Inspect** missing values, duplicates, outliers, and distributions\n- **Ask** questions without writing Python or SQL\n- **Visualize** patterns with interactive charts\n- **Evaluate** optional prediction and clustering models\n- **Export** a shareable HTML report and cleaned CSV")
    with right:
        st.info("**Quick start**\n\n1. Choose **Load sample sales dataset** in the sidebar.\n2. Open **Ask your data**.\n3. Try: `What are the top 5 products by sales?`\n4. Visit **Report** to export the computed summary.")
    st.stop()

quality = quality_report(df)
if df.empty: st.warning("This dataset has no rows. Upload a dataset with records to continue.")

if page == "Overview":
    st.subheader("Dataset overview")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", f"{len(df.columns):,}")
    c3.metric("Missing cells", f"{quality['missing_cells']:,}")
    c4.metric("Duplicate rows", f"{quality['duplicate_rows']:,}")
    active_upload = st.session_state.get("active_upload")
    active_name = active_upload[0] if active_upload else "sample_sales.csv"
    st.caption(f"Currently analyzing **{active_name}** · {len(df):,} records")
    t1,t2,t3 = st.tabs(["Data preview", "Column profile", "Quality checks"])
    with t1:
        st.dataframe(df.head(25), use_container_width=True, hide_index=True)
        st.caption(f"Previewing {min(25,len(df))} of {len(df):,} rows.")
    with t2:
        profile = pd.DataFrame({"Column": df.columns, "Data type": [str(df[c].dtype) for c in df.columns], "Non-missing": [int(df[c].notna().sum()) for c in df.columns], "Missing": [int(df[c].isna().sum()) for c in df.columns], "Distinct": [int(df[c].nunique(dropna=True)) for c in df.columns]})
        st.dataframe(profile, use_container_width=True, hide_index=True)
        stats = descriptive_statistics(df)
        if not stats.empty: st.markdown("**Numeric descriptive statistics**"); st.dataframe(stats, use_container_width=True)
    with t3:
        a,b = st.columns(2)
        with a:
            st.markdown("**Missing values**")
            missing = pd.Series(quality["missing_by_column"], name="Missing cells")
            st.dataframe(missing if not missing.empty else pd.DataFrame({"Status":["No missing cells detected"]}), use_container_width=True)
        with b:
            st.markdown("**IQR outlier scan**")
            out = pd.Series(quality["outliers_by_column"], name="Potential outliers")
            st.dataframe(out if not out.empty else pd.DataFrame({"Status":["No numeric columns to scan"]}), use_container_width=True)
        st.markdown("**Highly correlated numeric columns (|r| ≥ 0.70)**")
        corr = correlation_pairs(df)
        st.dataframe(corr if not corr.empty else pd.DataFrame({"Status":["No pairs crossed the threshold (or fewer than two numeric columns)."]}), use_container_width=True, hide_index=not corr.empty)
    st.markdown("### Evidence-backed highlights")
    for insight in generate_insights(df)[:6]: st.markdown(f"- {insight}")

elif page == "Clean data":
    st.subheader("Clean data")
    st.write("Preview changes before downloading. Your original upload stays unchanged; cleaning is applied to this session's working copy.")
    with st.form("clean_form"):
        c1,c2 = st.columns(2)
        drop_dup = c1.checkbox("Remove duplicate rows", value=False)
        drop_empty = c1.checkbox("Remove entirely empty columns", value=True)
        strategy = c2.selectbox("Missing-value handling", ["keep", "drop_rows", "fill_median_mode", "fill_zero"], format_func=lambda x:{"keep":"Keep missing values", "drop_rows":"Drop rows containing missing values", "fill_median_mode":"Fill numeric median / categorical mode", "fill_zero":"Fill numeric missing values with zero; categorical with mode"}[x])
        submitted = st.form_submit_button("Preview and apply cleaning", type="primary")
    if submitted:
        try:
            cleaned = clean_dataset(st.session_state.raw_df if st.session_state.raw_df is not None else df, drop_duplicates=drop_dup, missing_strategy=strategy, drop_empty_columns=drop_empty)
            st.session_state.clean_df = cleaned
            st.success(f"Working copy updated: {len(cleaned):,} rows × {len(cleaned.columns)} columns.")
        except Exception as exc: st.error(str(exc))
    current = st.session_state.clean_df
    st.dataframe(current.head(30), use_container_width=True, hide_index=True)
    st.download_button("Download cleaned CSV", current.to_csv(index=False).encode("utf-8"), file_name="cleaned_dataset.csv", mime="text/csv", type="primary")

elif page == "Ask your data":
    st.subheader("Ask your data")
    st.caption("Answers and charts are calculated from the current dataset. No generated code is run.")
    with st.form("question_form", clear_on_submit=True):
        question = st.text_input("What would you like to know?", placeholder="e.g. What are the top 5 products by sales?")
        ask = st.form_submit_button("Analyze question", type="primary")
    if ask and question:
        try:
            with st.spinner("Computing from your data…"):
                if use_llm:
                    result = plan_with_llm(df, question)
                else:
                    result = answer_question(df, question)
            record = {"question": question, "answer": result.answer, "explanation": result.explanation, "columns_used": result.columns_used or [], "operation": result.operation or "local rule-based plan"}
            st.session_state.qa_history.insert(0, record)
            st.session_state.qa_history = st.session_state.qa_history[:20]
            if result.data is not None and not result.data.empty:
                fig = query_figure(result)
                if fig is not None:
                    fig.update_layout(title=question, margin=dict(t=60,b=30))
                    st.session_state.saved_charts.insert(0,(question,fig))
                    st.session_state.saved_charts = st.session_state.saved_charts[:10]
            st.session_state.latest_result = result
        except Exception as exc:
            st.error(f"Could not complete the analysis: {exc}")
    if "latest_result" in st.session_state:
        result = st.session_state.latest_result
        st.markdown("### Answer")
        st.success(result.answer)
        st.write(result.explanation)
        st.caption(f"Operation: {result.operation or 'local deterministic calculation'} · Columns: {', '.join(result.columns_used or []) or 'row count / dataset-wide'}")
        if result.data is not None:
            st.dataframe(result.data, use_container_width=True, hide_index=False)
            if result.chart_type == "heatmap" and "column" in result.data.columns:
                fig = px.imshow(result.data.set_index("column"), text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title="Correlation heatmap")
            else: fig = query_figure(result)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
                st.download_button("Download chart as interactive HTML", fig.to_html(include_plotlyjs="cdn", full_html=True).encode(), file_name="analysis_chart.html", mime="text/html")
    if st.session_state.qa_history:
        with st.expander("Recent questions", expanded=False):
            for item in st.session_state.qa_history: st.markdown(f"**{item['question']}**\n\n{item['answer']}\n\n*{item['explanation']}*\n")

elif page == "Visualizations":
    st.subheader("Automatic visualizations")
    charts = automatic_charts(df)
    if not charts: st.info("No suitable chart found. Add numeric and/or categorical columns.")
    for i,(title,fig) in enumerate(charts):
        with st.container(border=True):
            st.markdown(f"#### {title}")
            st.plotly_chart(fig, use_container_width=True, key=f"auto_{i}")
            st.download_button("Download interactive chart (HTML)", fig.to_html(include_plotlyjs="cdn", full_html=True).encode(), file_name=f"chart_{i+1}.html", mime="text/html", key=f"download_chart_{i}")
    st.divider()
    st.markdown("### Customize a chart")
    numeric_cols = list(df.select_dtypes(include="number").columns)
    if numeric_cols:
        categorical_cols = list(df.select_dtypes(include=["object", "category", "bool"]).columns)
        x_options = list(dict.fromkeys(categorical_cols + list(df.columns)))
        x_default = x_options.index(categorical_cols[0]) if categorical_cols else 0
        ctrl1,ctrl2,ctrl3 = st.columns(3)
        x_col = ctrl1.selectbox("X axis / category", x_options, index=x_default)
        y_col = ctrl2.selectbox("Numeric measure", numeric_cols)
        kind = ctrl3.selectbox("Chart type", ["Bar", "Line", "Scatter", "Histogram", "Box"])
        filtered = df.copy()
        filter_col = st.selectbox("Optional category filter", ["No filter"] + categorical_cols)
        if filter_col != "No filter":
            values = filtered[filter_col].dropna().astype(str).unique().tolist()
            selected = st.multiselect(f"Keep {filter_col}", values, default=values)
            filtered = filtered[filtered[filter_col].astype(str).isin(selected)]
        chart_kind = kind.lower()
        if chart_kind == "histogram":
            custom_fig = px.histogram(filtered, x=y_col, title=f"Distribution of {y_col}", template="plotly_white")
        elif chart_kind == "box":
            custom_fig = px.box(filtered, x=x_col if x_col in categorical_cols else None, y=y_col, title=f"{y_col} distribution", template="plotly_white")
        else:
            creator = {"bar": px.bar, "line": px.line, "scatter": px.scatter}[chart_kind]
            custom_fig = creator(filtered, x=x_col, y=y_col, title=f"{y_col} by {x_col}", template="plotly_white")
        custom_fig.update_layout(xaxis_title=x_col if chart_kind != "histogram" else y_col, yaxis_title="Count" if chart_kind == "histogram" else y_col)
        st.plotly_chart(custom_fig, use_container_width=True, key="custom_chart")
        st.download_button("Download customized chart (HTML)", custom_fig.to_html(include_plotlyjs="cdn", full_html=True).encode(), file_name="custom_chart.html", mime="text/html", key="download_custom_chart")
    else:
        st.info("Customized charts need at least one numeric column.")

elif page == "Machine learning":
    st.subheader("Predictive analytics")
    st.info("Models are optional. A held-out test split is used; imputation, scaling, and category encoding are fitted only on the training data. Do not interpret predictions as causal explanations.")
    task = st.radio("Analysis type", ["regression", "classification", "clustering"], horizontal=True)
    numeric = list(df.select_dtypes(include="number").columns)
    if task in {"regression","classification"}:
        if not df.columns.any(): st.warning("No columns available.")
        else:
            target = st.selectbox("Target column", list(df.columns), index=(0 if not numeric else list(df.columns).index(numeric[-1])))
            candidate_features = [c for c in df.columns if c != target]
            features = st.multiselect("Input features", candidate_features, default=candidate_features[:min(4,len(candidate_features))])
            test_ratio = st.slider("Test data share", .15, .4, .2, .05)
            if st.button("Train and evaluate", type="primary"):
                try:
                    outcome = train_predictor(df, target, features, task, test_ratio)
                    st.success(f"Trained on {outcome['train_rows']:,} rows; evaluated on {outcome['test_rows']:,} held-out rows.")
                    a,b = st.columns(2)
                    for i,(name,value) in enumerate(outcome["metrics"].items()): (a if i%2==0 else b).metric(name.replace("_"," ").title(), f"{value:.4f}" if pd.notna(value) else "Not defined")
                    st.caption("Accuracy is the proportion of correct classes. Weighted F1 combines precision and recall, weighted by class frequency. MAE is average absolute error in the target's units; R² compares error against a mean-only baseline and may be negative.")
                    st.markdown("**Example held-out predictions**")
                    st.dataframe(outcome["predictions"].head(25), use_container_width=True, hide_index=True)
                except Exception as exc: st.error(f"Model not trained: {exc}")
    else:
        feats = st.multiselect("Numeric columns", numeric, default=numeric[:min(3,len(numeric))])
        count = st.number_input("Number of clusters", min_value=2, max_value=max(2,min(10,len(df)-1)), value=min(3,max(2,len(df)-1)), step=1)
        if st.button("Cluster records", type="primary"):
            try:
                clustered = cluster_records(df, feats, int(count))
                st.dataframe(clustered.head(100), use_container_width=True, hide_index=True)
                st.plotly_chart(px.scatter(clustered, x=feats[0], y=feats[1] if len(feats)>1 else feats[0], color="cluster", title="K-means groups", template="plotly_white"), use_container_width=True)
                st.download_button("Download cluster labels", clustered.to_csv(index=False).encode(), file_name="clustered_dataset.csv", mime="text/csv")
            except Exception as exc: st.error(f"Clustering not run: {exc}")

elif page == "Report":
    st.subheader("Analysis report")
    st.write("Generate an HTML report from the current working dataset, including calculated data quality, statistics, highlights, your questions, and charts.")
    with st.expander("Report contents", expanded=True):
        st.markdown(f"- Dataset overview: **{len(df):,} rows**, **{len(df.columns)} columns**\n- Quality checks: **{quality['missing_cells']:,}** missing cells, **{quality['duplicate_rows']:,}** duplicates\n- **{len(st.session_state.qa_history)}** recent questions and computed answers\n- **{len(st.session_state.saved_charts)}** generated charts")
    report = build_html_report(df, st.session_state.qa_history, st.session_state.saved_charts[:4])
    st.download_button("Download HTML analysis report", report.encode("utf-8"), file_name="data_analysis_report.html", mime="text/html", type="primary")
    st.caption("The report is generated locally. Charts load Plotly from its public CDN when opened online.")

st.divider()
st.caption("Data Analyst AI · Numbers are computed from the active dataset. Correlation is not causation. Use model outputs as exploratory estimates, not guarantees.")
