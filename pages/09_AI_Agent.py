from pathlib import Path
import streamlit as st
import pandas as pd
from modules.ai_agent import run, TOOLS

st.title("AI Agent")
st.write("This is a bounded, inspectable agent loop: classify the task, select from named read-only tools, run the constrained analyst, validate the result, and summarize the action log.")
if "active_df" not in st.session_state:
    st.session_state["active_df"] = pd.read_csv(Path(__file__).resolve().parents[1] / "sample_data" / "sales_demo.csv", parse_dates=["date"])
frame = st.session_state["active_df"]
st.caption(f"Current table: {len(frame)} rows × {len(frame.columns)} columns. Upload a file first on the AI Data Analyst page to reuse it.")
question = st.text_input("Agent task", "Compare revenue by region and show a chart")
if st.button("Run agent plan", type="primary"):
    with st.spinner("Running the selected read-only tools…"):
        outcome = run(question, frame, st.session_state.get("rag_chunks", []))
    st.session_state["agent_outcome"] = outcome
outcome = st.session_state.get("agent_outcome")
if outcome:
    st.subheader("Plan")
    st.dataframe(outcome["plan"], hide_index=True, width="stretch")
    st.subheader("Tool permissions")
    st.dataframe([{"Tool": k, "Permission": v} for k,v in TOOLS.items()], hide_index=True, width="stretch")
    st.subheader("Completed actions")
    st.dataframe(outcome["actions"], hide_index=True, width="stretch")
    result = outcome["result"]
    if result and result["ok"]:
        st.success(f"Verified result: {result['value']}")
        st.code(result["expression"], language="python")
    elif result:
        st.warning(result["error"])
    if outcome.get("chart") is not None:
        st.plotly_chart(outcome["chart"], width="stretch")
    if outcome.get("evidence"):
        st.subheader("Retrieved document evidence")
        for hit in outcome["evidence"]:
            chunk = hit["chunk"]
            with st.expander(f"{chunk.source} · page {chunk.page or 'n/a'} · similarity {hit['score']:.3f}"):
                st.write(chunk.text)
st.info("Agent safety boundary: no shell, arbitrary Python, or mutation tools are exposed. Document retrieval uses only files already indexed in the RAG page session.")
