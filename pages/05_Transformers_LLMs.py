import os
import importlib.util
import streamlit as st
import pandas as pd
from pathlib import Path
from modules.code_generator import groq_completion
from modules.transformer_inference import DEFAULT_QWEN_MODEL, is_ollama_model_id, model_capacity_status, ollama_model_status

st.title("Transformers and LLMs")
st.write("Transformers use attention to build contextual token representations. They can process tokens in parallel during training, while autoregressive text generation predicts one token at a time.")
st.code("Text → Tokenization → Embeddings + Positions → Self-Attention / MLP blocks → Output head → Next-token sampling", language="text")
st.latex(r"\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V")
st.dataframe(pd.DataFrame([
    ["Pretraining", "Learn broad representations from large corpora using self-supervised objectives", "A foundation model acquires general language patterns"],
    ["Fine-tuning", "Continue training on task/domain examples", "Adapt a base model to structured customer support"],
    ["Prompting", "Condition inference without changing model weights", "Set an analyst persona and required answer format"],
    ["RAG", "Retrieve relevant external evidence at inference time", "Ground a summary in an uploaded policy PDF"],
], columns=["Method","What changes","Example"]), width="stretch", hide_index=True)
st.subheader("Foundation and multimodal models")
st.write("Foundation models are broadly pretrained models that can be adapted to many downstream tasks. Multimodal systems connect two or more modalities, such as text with images, audio, or video. Pretraining data and capabilities vary by model; do not assume a text model can inspect media.")
st.subheader("Hosted transformer text-generation demo")
st.caption("Uses the configured Groq model. The entered prompt is sent to Groq when you click Generate; do not include private data unless you have authorization to share it.")
prompt = st.text_area("Prompt", "Explain why a sales analyst should verify a chart before sharing it.")
has_groq = bool(os.getenv("GROQ_API_KEY"))
if not has_groq: st.info("Hosted generation needs your Groq API key. Copy `.env.example` to `.env` beside `app.py`, set `GROQ_API_KEY=your_key`, then restart Streamlit. Keep the key private.")
if st.button("Generate with Groq", disabled=not has_groq):
    try:
        with st.spinner("Generating from the hosted model…"): output = groq_completion(prompt)
        st.markdown(output)
        st.caption("Actual API response. Model ID comes from GROQ_MODEL or defaults to openai/gpt-oss-20b.")
    except Exception as exc: st.warning(f"Generation unavailable: {exc}")
st.subheader("Optional local Hugging Face inference")
local_path = st.text_input("Model ID or local model directory", DEFAULT_QWEN_MODEL)
local_prompt = st.text_input("Local generation prompt", "A data analyst should verify a chart because")
has_transformers = importlib.util.find_spec("transformers") is not None
st.caption("Qwen2.5-Coder 1.5B runs locally through Ollama. Pull the model once with `ollama pull qwen2.5-coder:1.5b`. Larger Hugging Face models use Transformers and need much more RAM.")
if is_ollama_model_id(local_path):
    capacity_ok, capacity_message = ollama_model_status(local_path)
else:
    capacity_ok, capacity_message = model_capacity_status(local_path)
if not has_transformers and not is_ollama_model_id(local_path): st.info("Install the optional ML packages with `pip install -r requirements-ml.txt`.")
if not capacity_ok: st.info(capacity_message)
if st.button("Run local Transformer", disabled=not bool(local_path.strip()) or (not has_transformers and not is_ollama_model_id(local_path)) or not capacity_ok):
    try:
        from modules.transformer_inference import generate_local
        with st.spinner("Loading model; first use may download its weights…"): output = generate_local(local_path, local_prompt)
        st.write(output)
    except Exception as exc: st.warning(f"Local inference unavailable: {exc}")
st.caption("Use the default Ollama model tag or enter a Hugging Face model ID/local model folder.")
