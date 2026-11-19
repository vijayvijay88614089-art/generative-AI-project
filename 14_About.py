import streamlit as st
from pathlib import Path

st.title("About InsightForge AI")
st.markdown("**Project title:** InsightForge AI: An Intelligent Data Analyst Copilot Using Generative AI and Advanced Prompt Engineering")
st.markdown("""
### Abstract
InsightForge AI is an educational application that turns tabular questions into bounded, inspectable calculations. It combines metadata-aware prompts, answer-type classification, few-shot examples, AST validation, isolated expression execution, result checks, retrieval-augmented document answers, and optional hosted language-model explanations. It also contains learning demonstrations for generative models, sequence architectures, prompt engineering, multimodal AI, and responsible deployment.

### Objectives
- Let a user upload and inspect CSV/Excel tables.
- Answer supported questions with computed outputs, relevant evidence, and visualizations.
- Demonstrate a research-inspired LLM-to-Pandas workflow while restricting executable code.
- Teach prompt design, RAG, agents, deep learning foundations, and responsible AI.
- Measure real local execution behavior without reporting invented results.

### Methodology
The tabular analysis pipeline identifies an expected output type, gathers dataset metadata, retrieves similar type-filtered examples, creates a Pandas expression, validates its syntax against an AST allowlist, executes it in a short-lived worker with a timeout, validates the result type, and optionally requests a concise explanation. Groq is optional; without a key, a deterministic local planner handles common operations and unsupported questions return a clear error. Uploaded documents are chunked and searched locally with TF-IDF; dense Sentence Transformers + FAISS retrieval is optional.

### Research inspiration
Gao, Chen, and Su's I2R-NLP SemEval-2025 system describes answer-type identification, metadata and few-shot prompting, Pandas-function generation, and multi-turn error correction. Its published findings belong to that paper's benchmark and setup; this application reports only metrics produced by its own evaluation page.

### Limitations
- The local question planner covers common aggregations and comparisons, not unrestricted analytical language.
- Hosted generation, image interpretation, and audio transcription require a Groq API key and transmit the selected input to the provider.
- The local RAG default uses sparse TF-IDF rather than a dense embedding model.
- The PyTorch generative demos train small models on bundled 8×8 handwritten digits; StyleGAN is described but not reproduced.
- CycleGAN is demonstrated as a cycle-consistency concept, not as a trained GAN.
- The worker process has a time limit and dataset cap but no hard OS-level memory isolation.

### Future scope
Add a broader labeled tabular benchmark, calibrated uncertainty, reproducible seed-locked comparisons of prompt strategies, persisted vector indexes, optional local multimodal models, accessibility refinements, and stronger OS/container sandboxing.
""")
st.markdown("### References")
st.markdown("- Gao, Y., Chen, B., & Su, J. (2025). [I2R-NLP at SemEval-2025 Task 8: Question Answering on Tabular Data](https://aclanthology.org/2025.semeval-1.14/).")
st.markdown("- Vaswani et al. (2017). [Attention Is All You Need](https://arxiv.org/abs/1706.03762).")
st.markdown("- Kingma & Welling (2014). [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114).")
st.markdown("- Goodfellow et al. (2014). [Generative Adversarial Nets](https://arxiv.org/abs/1406.2661).")
