import streamlit as st

st.title("Industry Applications")
use_cases = {
    "Healthcare": "Summarize clinical literature or operational data with clinician review; protect health information and do not treat generated text as a diagnosis.",
    "Banking and finance": "Explain aggregate portfolio trends and support analyst workflows; test fairness, audit decisions, and keep regulated decisions under approved controls.",
    "Education": "Create practice explanations and adapt examples; preserve student privacy and check accuracy and accessibility.",
    "Retail and e-commerce": "Analyze product performance, catalog content, and customer feedback; audit for demographic stereotypes and deceptive personalization.",
    "Marketing": "Draft and compare campaign variants; verify factual claims, rights, disclosures, and audience appropriateness.",
    "Manufacturing": "Summarize sensor anomalies and maintenance notes; validate against physical inspection and safety procedures.",
    "Software development": "Draft code, explain errors, and summarize repositories; review security, license, and correctness before release.",
    "Customer support": "Retrieve policy passages and draft responses; cite sources, route uncertain cases, and avoid exposing customer records.",
}
domain = st.selectbox("Select an industry", list(use_cases))
st.subheader(domain)
st.write(use_cases[domain])
st.markdown("**InsightForge workflow:** upload an approved dataset → inspect data quality → ask a bounded analytical question → verify result and evidence → share with review context.")
st.caption("These are illustrative workflow examples, not validated deployments or performance claims.")
