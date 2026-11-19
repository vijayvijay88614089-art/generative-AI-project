import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from modules.education import attention_matrix

st.title("Deep Learning Models")
st.write("Sequence models process ordered inputs such as words, events, sensor readings, or transactions.")
st.dataframe(pd.DataFrame([
    ["RNN", "One recurrent hidden state", "Short sequences; simple baselines", "Vanishing/exploding gradients"],
    ["LSTM", "Input, forget, output gates + cell state", "Longer dependencies", "More parameters and sequential computation"],
    ["BiLSTM", "LSTM in forward and reverse directions", "Full-sequence tagging", "Cannot use future context in streaming settings"],
    ["GRU", "Update and reset gates", "Compact sequence modeling", "Still sequential over time"],
    ["BiGRU", "Forward and reverse GRUs", "Offline sequence classification", "Future context unavailable in causal streaming"],
    ["Attention", "Weighted content lookup", "Direct context selection", "Quadratic attention cost in vanilla self-attention"],
], columns=["Model","Core idea","Strength","Trade-off"]), hide_index=True, width="stretch")
st.subheader("Interactive attention heatmap")
text = st.text_input("Enter a short sequence (space-separated)", "sales rose in east")
tokens = text.split()[:12]
if tokens:
    weights = attention_matrix(tokens)
    st.plotly_chart(px.imshow(weights, x=tokens, y=tokens, text_auto=".2f", aspect="auto", color_continuous_scale="Tealgrn", labels={"x":"Key token", "y":"Query token", "color":"Weight"}), width="stretch")
    st.caption("Illustrative scaled dot-product attention calculated from deterministic toy token vectors—not learned model attention.")
st.subheader("Working sequence model examples")
st.write("PyTorch implementations of RNN, LSTM, BiLSTM, GRU, and BiGRU classifiers are in `models/`. They define runnable forward passes; they are intentionally untrained and must not be interpreted as useful predictions.")
with st.expander("LSTM equations"):
    st.latex(r"f_t=\sigma(W_f[h_{t-1},x_t]+b_f),\quad i_t=\sigma(W_i[h_{t-1},x_t]+b_i)")
    st.latex(r"c_t=f_t\odot c_{t-1}+i_t\odot\tilde{c}_t,\quad h_t=o_t\odot\tanh(c_t)")
