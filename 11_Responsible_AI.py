import streamlit as st
import pandas as pd

st.title("Responsible AI")
st.write("Generative systems can make polished mistakes. Responsible use combines technical controls with clear accountability and human review.")
st.dataframe(pd.DataFrame([
    ["Bias", "Representation gaps or historical discrimination", "Check subgroup coverage and impacts; consult affected stakeholders"],
    ["Privacy", "Sensitive data exposure through logs or hosted APIs", "Minimize collection, avoid storing raw uploads, review where API content is sent"],
    ["Hallucinations", "Plausible but unsupported text or numbers", "Ground numeric outputs in calculations and show evidence"],
    ["Deepfakes", "Synthetic media used to impersonate or mislead", "Label generated media and verify provenance where possible"],
    ["Copyright", "Training or output may implicate protected works", "Review licenses and applicable policy before publication"],
    ["Misinformation", "False claims gain reach through generated content", "Cite sources, verify claims, and keep uncertainty visible"],
    ["Transparency", "Users cannot see how an answer was produced", "Show model/provider, expression, data evidence, limits, and review path"],
], columns=["Risk","How it appears","Mitigation"]), hide_index=True, width="stretch")
st.subheader("InsightForge safeguards")
st.markdown("- Uploaded datasets stay in app-session memory; query history stores questions, answers, and expressions but not the table itself.\n- Table code is expression-only, AST-validated, and evaluated in a short-lived worker with a five-second timeout.\n- Data uploads are capped at 100,000 rows and 100 columns.\n- Hosted Groq options send the selected prompt or media to Groq; uploaders are told before use.\n- Local answers show calculations and evidence. Correlation is never described as causation.")
st.warning("A timeout and row cap are not a full operating-system memory sandbox. Do not use this educational application for high-impact decisions without independent validation and security review.")
st.subheader("Human review checklist")
st.checkbox("I checked that this dataset is appropriate and permitted for this use.")
st.checkbox("I verified the answer against the source records and considered missingness and bias.")
st.checkbox("I will not use this result as the sole basis for a consequential decision.")
