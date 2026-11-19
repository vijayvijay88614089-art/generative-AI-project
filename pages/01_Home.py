from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px

st.title("🔎 InsightForge AI")
st.markdown("### Your evidence-first data analyst copilot")
st.write("Upload a table, ask a question in plain language, and inspect the calculation, evidence, and visualization behind the answer.")

col1, col2, col3 = st.columns(3)
col1.metric("Analysis", "CSV + Excel")
col2.metric("Grounding", "Dataset + RAG")
col3.metric("Safety", "AST allowlist + worker process")

st.subheader("Start with the built-in demo")
sample = pd.read_csv(Path(__file__).resolve().parents[1] / "sample_data" / "sales_demo.csv", parse_dates=["date"])
st.dataframe(sample.head(8), width="stretch")
chart = sample.groupby("region", as_index=False)["revenue"].sum().sort_values("revenue", ascending=False)
st.plotly_chart(px.bar(chart, x="region", y="revenue", color="region", title="Demo revenue by region"), width="stretch")
st.caption("Demo data is synthetic and included for demonstration; it is not a business benchmark.")

st.subheader("What you can explore")
left, right = st.columns(2)
left.markdown("**Build with**\n\n- Natural-language tabular analysis\n- Prompt techniques and injection defense\n- PDF/TXT retrieval assistant\n- Bounded tool-using agent")
right.markdown("**Learn about**\n\n- Generative model families\n- RNN, LSTM, GRU, attention and transformers\n- Multimodal systems\n- Evaluation, limitations, and responsible use")
