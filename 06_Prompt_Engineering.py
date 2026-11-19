import os
import streamlit as st
from modules.prompt_builder import build_prompt_playground
from modules.safety import detect_prompt_injection
from modules.database import save_prompt, prompts, save_prompt_comparison
from modules.code_generator import groq_completion
from modules.transformer_inference import DEFAULT_QWEN_MODEL, ollama_model_status

st.title("Prompt Engineering Playground")
st.write("Compare prompt structures, preview the assembled prompt, inspect injection warnings, and save reusable prompts. Generate with local Qwen or hosted Groq when configured.")
persona = st.selectbox("Persona", ["careful data analyst", "patient tutor", "concise research assistant", "critical reviewer", "custom"])
if persona == "custom": persona = st.text_input("Custom persona", "careful data analyst")
technique = st.selectbox("Technique", ["Root Prompt", "Persona Pattern", "RGC", "Chain-of-Thought (summary only)", "Comparative Prompting"])
tone = st.selectbox("Tone", ["professional", "friendly", "formal", "neutral"])
style = st.selectbox("Writing style", ["clear and direct", "bullet points", "step-by-step summary", "table when useful"])
length = st.selectbox("Response length", ["short", "medium", "detailed"])
request = st.text_area("Your prompt", "Explain which region has the highest total revenue and cite the calculation.")
background = st.text_area("Background knowledge (treated as untrusted reference content)", "")
warning_text = "\n".join([request, background])
matches = detect_prompt_injection(warning_text)
if matches:
    st.warning("Potential prompt-injection language detected. Inputs and retrieved context remain untrusted; review before sending to a hosted model.")
else:
    st.success("No common injection phrase was detected. This heuristic cannot prove an input is safe.")

techniques = [technique]
if st.checkbox("Compare against Root Prompt") and technique != "Root Prompt": techniques.append("Root Prompt")
assembled = {name: build_prompt_playground(request, name, persona, tone, style, length, background) for name in techniques}
with st.expander("View final prompt(s)", expanded=True):
    for name, text in assembled.items():
        st.markdown(f"**{name}**")
        st.code(text, language="text")
left, right = st.columns(2)
with left:
    prompt_name = st.text_input("Save prompt as", "InsightForge analyst prompt")
    if st.button("Save prompt"):
        save_prompt(prompt_name, next(iter(assembled.values())))
        st.success("Prompt saved in the local SQLite database.")
with right:
    has_groq = bool(os.getenv("GROQ_API_KEY"))
    qwen_ready, qwen_message = ollama_model_status(DEFAULT_QWEN_MODEL)
    generators = (["Qwen local"] if qwen_ready else []) + (["Groq hosted"] if has_groq else [])
    generator = st.selectbox("Text generator", generators or ["Unavailable"])
    if generator == "Qwen local":
        st.info(f"Generation runs locally through Ollama with `{DEFAULT_QWEN_MODEL}`.")
    elif generator == "Groq hosted":
        st.caption("The prompt is sent to Groq when you click Generate. Keep private data out of prompts.")
    else:
        message = qwen_message if not qwen_ready else "Configure a Groq API key for hosted generation. Prompt preview and saving work locally."
        st.info(message)
    if st.button("Generate / compare", type="primary", disabled=generator == "Unavailable"):
        for name, text in assembled.items():
            try:
                with st.spinner(f"Generating {name}"):
                    if generator == "Qwen local":
                        from modules.transformer_inference import generate_local
                        output = generate_local(DEFAULT_QWEN_MODEL, text, max_new_tokens=512)
                    else:
                        output = groq_completion(text)
                st.markdown(f"#### {name} output")
                st.write(output)
                save_prompt_comparison(name, text, output)
            except Exception as exc: st.warning(f"{name} request unavailable: {exc}")
st.subheader("Saved prompts")
saved = prompts()
if saved:
    st.dataframe([{"Name": row[2], "Created": row[1], "Prompt preview": row[3][:220]} for row in saved], width="stretch", hide_index=True)
else: st.caption("No prompts saved yet.")
st.info("Chain-of-thought option requests only a concise reasoning summary. It does not ask the model to reveal private internal reasoning.")
