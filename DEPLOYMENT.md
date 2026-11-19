# Deploy InsightForge AI

## Recommended host: Streamlit Community Cloud

The app is prepared for Community Cloud. The project was smoke-checked locally with Python 3.11 and Streamlit 1.66.1. The project folder must be the GitHub repository root so `app.py`, `requirements.txt`, `pages/`, `modules/`, `models/`, `assets/`, and `sample_data/` are all available together.

1. Create a GitHub repository and upload/push the **contents of this project folder**. Include the sample data and source files. Do not include `.venv/`, `.env`, `.streamlit/secrets.toml`, uploaded datasets/documents, or local SQLite files.
2. In Streamlit Community Cloud, create an app and select that repository, its branch, and `app.py` as the main file.
3. In the app's advanced settings, select Python 3.11, which is the version used for local checks.
4. If you want hosted text/audio/image generation, add secrets in the Cloud settings. The app reads these keys from Community Cloud secrets:

   ```toml
   GROQ_API_KEY = "your-private-key"
   GROQ_MODEL = "openai/gpt-oss-20b"
   GROQ_AUDIO_MODEL = "whisper-large-v3-turbo"
   GROQ_VISION_MODEL = "qwen/qwen3.8-27b"
   ```

   Add only the values you use. Never commit an API key to GitHub. Without `GROQ_API_KEY`, local CSV analysis, charts, lessons, basic TF-IDF retrieval, and generative-model demonstrations still work; hosted generation and audio transcription remain unavailable.
5. Deploy the app. Community Cloud will install packages from `requirements.txt` and provide a public `*.streamlit.app` URL.

## Hosting limitations

- Ollama is a local desktop service. On a hosted app, `127.0.0.1` refers to the app container, so the desktop's Qwen/Ollama service is not available there. Use a configured hosted model for remote generation.
- Uploaded files are processed by the hosted app while in use. Do not upload sensitive information unless you are authorized to process it on that host.
- The app's SQLite history and the host's local filesystem are not durable storage. Data may be lost when the hosted container restarts. The application does not persist uploaded datasets or RAG document text.
- Optional PyTorch, Transformers, Sentence Transformers/FAISS, and video packages are not in the base `requirements.txt`; keep them out of a small demo deployment unless you specifically need those features.

## Local deployment check

From this folder, run:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The `.streamlit/config.toml` applies the same theme locally and in the hosted app. The ignored `.streamlit/secrets.toml` file can be used for local Streamlit secrets, while `.env` continues to work for local development.
