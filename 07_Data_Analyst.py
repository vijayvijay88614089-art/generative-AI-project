from pathlib import Path
import time
import os
import tempfile
import streamlit as st
import pandas as pd
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "InsightForge_AI_mplconfig"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
import matplotlib.pyplot as plt
import seaborn as sns
from modules.data_loader import load_table, profile
from modules.analyst import analyze
from modules.database import save_query, history
from modules.chart_generator import build_chart
from modules.upload_manager import project_uploads
from modules.transformer_inference import DEFAULT_QWEN_MODEL, ollama_model_status
from modules.insights import generate_insights, build_html_report

st.title("AI Data Analyst")
st.write("Ask natural-language questions about a local dataset. The local planner handles common requests. For other questions, optionally use Qwen locally or Groq when configured; generated expressions are safety-checked and run in a time-limited worker.")
uploaded = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"], key="analyst_upload")
local_files = project_uploads("datasets", {".csv", ".xlsx", ".xls"})
local_choice = st.selectbox(
    "Or choose a dataset from uploads/datasets",
    [""] + [str(path) for path in local_files],
    format_func=lambda value: "Select a project file…" if not value else Path(value).name,
)
if uploaded or local_choice:
    try:
        frame = load_table(uploaded if uploaded else local_choice)
        if uploaded:
            active_key = f"upload:{getattr(uploaded, 'file_id', uploaded.name)}:{uploaded.size}"
        else:
            stat = Path(local_choice).stat()
            active_key = f"file:{Path(local_choice).resolve()}:{stat.st_size}:{stat.st_mtime_ns}"
        st.session_state["active_df"] = frame
        st.session_state["active_dataset_name"] = uploaded.name if uploaded else Path(local_choice).name
        st.session_state["active_dataset_key"] = active_key
    except Exception as exc: st.error(f"Could not load file: {exc}"); st.stop()
else:
    if "active_df" not in st.session_state:
        st.session_state["active_df"] = pd.read_csv(Path(__file__).resolve().parents[1] / "sample_data" / "sales_demo.csv", parse_dates=["date"])
    st.session_state.setdefault("active_dataset_key", "builtin:sales_demo")
    frame = st.session_state["active_df"]

info = profile(frame)
st.caption(f"Active dataset: {st.session_state.get('active_dataset_name', 'Built-in sales demo')}")
c1,c2,c3,c4 = st.columns(4)
c1.metric("Rows", f"{info['rows']:,}"); c2.metric("Columns", info["columns"]); c3.metric("Duplicate rows", info["duplicates"]); c4.metric("Missing cells", sum(info["missing"].values()))
with st.expander("Dataset understanding", expanded=True):
    st.write("**Dimensions:**", frame.shape, "· **Numeric columns:**", ", ".join(map(str, info["numeric"])) or "none", "· **Categorical columns:**", ", ".join(map(str, info["categorical"])) or "none")
    st.dataframe(pd.DataFrame({"Column": list(info["types"]), "Type": list(info["types"].values()), "Missing": list(info["missing"].values())}), hide_index=True, width="stretch")
    st.dataframe(frame.head(100), width="stretch")
    st.write("Statistical summary"); st.dataframe(frame.describe(include="all").transpose(), width="stretch")
    numeric_cols = frame.select_dtypes(include="number").columns.tolist()
    if numeric_cols:
        selected_numeric = st.selectbox("Distribution (Matplotlib + Seaborn)", numeric_cols)
        fig, ax = plt.subplots(figsize=(8, 3))
        sns.histplot(frame[selected_numeric].dropna(), kde=len(frame[selected_numeric].dropna()) > 1, ax=ax, color="#55d6be")
        ax.set_title(f"Distribution of {selected_numeric}")
        st.pyplot(fig)
        plt.close(fig)

with st.expander("Automated insights and report", expanded=False):
    st.write("Run deterministic checks for missing data, duplicates, numeric ranges, potential outliers, category concentration, and strong correlations. Results are computed locally; no LLM is required.")
    active_key = st.session_state.get("active_dataset_key", "builtin:sales_demo")
    if st.session_state.get("insights_dataset_key") != active_key:
        st.session_state.pop("dataset_insights", None)
        st.session_state.pop("insights_dataset_key", None)
    if st.button("Generate dataset insights", key="generate_dataset_insights"):
        try:
            st.session_state["dataset_insights"] = generate_insights(frame)
            st.session_state["insights_dataset_key"] = active_key
        except Exception as exc:
            st.error(f"Could not generate dataset insights: {type(exc).__name__}: {exc}")
    insights = st.session_state.get("dataset_insights")
    if insights and st.session_state.get("insights_dataset_key") == active_key:
        st.caption(f"Computed for {st.session_state.get('active_dataset_name', 'Built-in sales demo')}; values are descriptive signals to review, not causal claims.")
        st.dataframe(pd.DataFrame(insights["findings"]), hide_index=True, width="stretch")
        st.dataframe(pd.DataFrame(insights["quality"]), hide_index=True, width="stretch")
        dataset_name = st.session_state.get("active_dataset_name", "sales_demo.csv")
        report_name = f"{Path(dataset_name).stem or 'dataset'}_insights.html"
        st.download_button(
            "Download self-contained HTML report",
            data=build_html_report(insights, dataset_name).encode("utf-8"),
            file_name=report_name,
            mime="text/html",
            key="download_dataset_insights",
        )

qwen_ready, qwen_message = ollama_model_status(DEFAULT_QWEN_MODEL)
use_qwen = st.checkbox(
    "Use Qwen2.5-Coder for questions the local planner cannot answer",
    value=False,
    disabled=not qwen_ready,
)
if not qwen_ready:
    st.info(f"{qwen_message} Rule-based analysis still works without Qwen.")
if use_qwen:
    st.info(f"Qwen runs locally through Ollama using `{DEFAULT_QWEN_MODEL}`. The model must be pulled before analysis.")
question = st.text_input("Ask about this dataset", placeholder="Which region has the highest revenue?")
include_rag = st.checkbox("Include retrieved document context from RAG session", value=False, disabled=not bool(st.session_state.get("rag_chunks")))
if st.button("Analyze question", type="primary", disabled=not bool(question.strip())):
    with st.spinner("Identifying answer type, building the analysis plan, validating and calculating…"):
        rag_context = ""
        if include_rag and st.session_state.get("rag_chunks"):
            from modules.rag_pipeline import retrieve
            hits = retrieve(st.session_state["rag_chunks"], question, top_k=3)
            rag_context = "\n\n".join(f"[{h['chunk'].source}, page {h['chunk'].page or 'n/a'}] {h['chunk'].text}" for h in hits)
        result = analyze(
            question,
            frame,
            rag_context=rag_context,
            local_model_id=DEFAULT_QWEN_MODEL if use_qwen else None,
        )
    save_query(result, question)
    st.session_state["last_analysis"] = {"question": question, "result": result}
    if result["ok"]: st.success("Calculation completed and output type validated.")
    else: st.warning(f"Analysis not completed: {result['error']}")

last = st.session_state.get("last_analysis")
if last:
    result = last["result"]
    st.subheader("Answer")
    if result["ok"]:
        st.markdown(result["explanation"])
        a,b,c,d = st.columns(4)
        a.metric("Answer type", result["answer_type"]); b.metric("Time", f"{result['elapsed_seconds']:.3f}s"); c.metric("Correction attempts", result["attempts"]); d.metric("Plan source", result["source"])
        with st.expander("Calculation, few-shot examples, and evidence", expanded=True):
            st.code(result["expression"], language="python")
            st.write("**Computed result:**", result["value"])
            st.write("**Dataset evidence:**", f"{result['metadata']['rows']} rows; columns {', '.join(result['metadata']['column_names'])}; missing values {result['metadata']['missing']}")
            st.write("**Similar examples retrieved:**")
            st.json(result["examples"])
        fig = build_chart(frame, last["question"], result.get("value"))
        if fig is not None: st.plotly_chart(fig, width="stretch")
        st.caption(f"Explanation path: {result['explanation_source']}. Numeric value comes from the validated local calculation.")
    else:
        st.code(result["expression"], language="python")
        st.caption(f"Answer type: {result['answer_type']} · attempts: {result['attempts']} · elapsed: {result['elapsed_seconds']:.3f}s")

st.subheader("Recent local query history")
rows = history(15)
if rows: st.dataframe([{"Time":r[0],"Question":r[1],"Answer / error":r[2],"Success":bool(r[3]),"Seconds":r[4]} for r in rows], width="stretch", hide_index=True)
else: st.caption("Query history is empty. Dataset rows are not stored in history.")
