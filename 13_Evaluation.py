from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px
from modules.evaluation import run_evaluation
from modules.database import history, connect

st.title("Project Evaluation")
st.write("Run the bundled question set against the current sample dataset. Metrics below are computed from actual executions in this app session; they are not the published SemEval paper's scores.")
if st.button("Run demo evaluation", type="primary"):
    frame = pd.read_csv(Path(__file__).resolve().parents[1] / "sample_data" / "sales_demo.csv", parse_dates=["date"])
    with st.spinner("Executing six typed questions through the analyst pipeline…"):
        table, metrics = run_evaluation(frame)
    st.session_state["evaluation_table"] = table
    st.session_state["evaluation_metrics"] = metrics
table = st.session_state.get("evaluation_table")
metrics = st.session_state.get("evaluation_metrics")
if metrics:
    a,b,c,d = st.columns(4)
    a.metric("Exact match", f"{metrics['exact_match_accuracy']:.1%}")
    b.metric("Execution success", f"{metrics['execution_success_rate']:.1%}")
    c.metric("Answer type validation", f"{metrics['answer_type_validation_accuracy']:.1%}")
    d.metric("Mean response", f"{metrics['mean_response_seconds']:.3f}s")
    st.dataframe(table, hide_index=True, width="stretch")
    st.plotly_chart(px.bar(table.melt(id_vars="Question", value_vars=["Exact match","Execution success","Type valid"], var_name="Metric", value_name="Passed"), x="Metric", y="Passed", color="Metric", title="Measured result by question", barmode="group"), width="stretch")
    if metrics["error_correction_success_rate"] is None: st.caption(metrics["error_correction_note"])
    else: st.metric("Correction success rate", f"{metrics['error_correction_success_rate']:.1%}")
    st.caption("RAG relevance is not measured unless a benchmark corpus and relevance labels are supplied. Results are specific to this six-question synthetic demo set.")
else:
    st.info("No evaluation has been run yet. Click the button to calculate metrics from actual test questions.")
st.subheader("Evaluation history")
with connect() as db:
    runs = db.execute("SELECT created_at,metrics_json FROM evaluation_runs ORDER BY id DESC LIMIT 10").fetchall()
if runs: st.dataframe([{"Created":r[0],"Metrics":r[1]} for r in runs], hide_index=True, width="stretch")
