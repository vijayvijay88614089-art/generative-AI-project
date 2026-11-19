import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

st.title("AI Fundamentals")
st.write("Artificial intelligence is the broad field of systems that perform tasks associated with perception, reasoning, learning, or action. Generative AI learns patterns in data and produces new content; predictive systems often choose labels or values from a known set.")
st.table(pd.DataFrame([
    ["Goal", "Estimate a label/value", "Sample from a learned data distribution"],
    ["Example", "Predict whether a transaction is fraud", "Draft an explanation or synthesize an image"],
    ["Typical output", "Class / score / forecast", "Text / image / audio / structured content"],
    ["Model example", "Logistic regression classifier", "VAE, GAN, autoregressive Transformer"],
], columns=["Dimension", "Discriminative / predictive", "Generative"]))
with st.expander("Probability refresher"):
    st.latex(r"P(x)=\sum_z P(x\mid z)P(z) \qquad P(z\mid x)=\frac{P(x\mid z)P(z)}{P(x)}")
    st.write("A generative model can describe how observations arise from latent variables. Sampling then uses the learned conditional distribution. Model confidence is not the same as truth.")
with st.expander("Neural network recap"):
    st.latex(r"h^{(l+1)}=\sigma(W^{(l)}h^{(l)}+b^{(l)})")
    st.write("Weights and biases are learned by minimizing a loss with gradient-based optimization. Hidden layers form representations; the loss and data define what the system learns.")
st.subheader("Tiny probability demonstration")
logits = st.slider("Logit for outcome A", -4.0, 4.0, 1.0, .1)
probs = np.exp([logits, 0]) / np.exp([logits, 0]).sum()
st.plotly_chart(px.bar(pd.DataFrame({"Outcome": ["A", "B"], "Probability": probs}), x="Outcome", y="Probability", range_y=[0,1]), width="stretch")
st.caption("Softmax demonstration only; these are illustrative logits, not model predictions.")
st.markdown("**Applications:** drafting, translation, code assistance, synthetic media, molecule design, data analysis, and multimodal interfaces. Each use needs task-specific evaluation and human oversight.")
