import streamlit as st
import importlib.util
from modules.rag_pipeline import extract_chunks, retrieve, retrieve_dense, answer
from modules.database import connect
from modules.upload_manager import project_uploads

st.title("RAG Document Assistant")
st.write("Upload PDF or TXT files, split extracted text into overlapping chunks, retrieve relevant passages with a local TF-IDF vectorizer, and review citations/snippets before accepting an answer.")
uploads = st.file_uploader("Upload reference documents", type=["pdf", "txt"], accept_multiple_files=True)
local_docs = project_uploads("documents", {".pdf", ".txt"})
local_choices = st.multiselect(
    "Or choose project files from uploads/documents",
    local_docs,
    format_func=lambda path: path.name,
    key="rag_local_files",
)
if "rag_chunks" not in st.session_state: st.session_state["rag_chunks"] = []
if uploads or local_choices:
    chunks = []
    documents = [(item.name, item.getvalue()) for item in (uploads or [])]
    documents.extend((path.name, path.read_bytes()) for path in local_choices)
    signature = tuple(sorted((name, __import__("hashlib").sha256(payload).hexdigest()) for name, payload in documents))
    for name, payload in documents:
        try: chunks.extend(extract_chunks(name, payload))
        except Exception as exc: st.error(f"Could not read {name}: {exc}")
    if chunks:
        if st.session_state.get("rag_upload_signature") != signature:
            st.session_state["rag_chunks"] = chunks
            st.session_state["rag_upload_signature"] = signature
            with connect() as db:
                for name in sorted({c.source for c in chunks}):
                    db.execute("INSERT INTO documents(name,chunks) VALUES(?,?)", (name, sum(c.source == name for c in chunks)))
        st.success(f"Indexed {len(chunks)} chunks from {len({c.source for c in chunks})} document(s) in this session.")
if st.session_state["rag_chunks"]:
    dense_available = all(importlib.util.find_spec(package) is not None for package in ("sentence_transformers", "faiss"))
    backends = ["TF-IDF local (lightweight)"]
    if dense_available: backends.append("Sentence Transformers + FAISS")
    else: st.info("Dense retrieval is disabled until Sentence Transformers and FAISS are installed. TF-IDF retrieval is ready now.")
    backend = st.radio("Retrieval backend", backends)
    question = st.text_input("Ask about the uploaded references")
    if st.button("Retrieve and answer", type="primary", disabled=not bool(question.strip())):
        if backend.startswith("Sentence"):
            try:
                hits = retrieve_dense(st.session_state["rag_chunks"], question)
            except Exception as exc:
                st.warning(f"Dense retrieval is unavailable; using local TF-IDF instead. Details: {exc}")
                hits = retrieve(st.session_state["rag_chunks"], question)
        else:
            hits = retrieve(st.session_state["rag_chunks"], question)
        response, source = answer(question, hits)
        st.markdown(response)
        st.caption(source)
        st.subheader("Retrieved evidence")
        for hit in hits:
            chunk = hit["chunk"]
            match_type = hit.get("match_type", "Semantic similarity")
            with st.expander(f"{chunk.source} · page {chunk.page or 'n/a'} · {match_type}: {hit['score']:.3f}"):
                st.write(chunk.text)
else:
    st.info("Upload a reference document to begin. No source text is retained in SQLite; chunks live only in the active app session.")
st.markdown("**Retrieval backend:** TF-IDF vectors are used as the zero-setup local fallback. Optional Sentence Transformers + FAISS can be installed from `requirements-ml.txt` and configured in the pipeline for dense semantic retrieval.")
st.warning("Retrieved passages are untrusted input. A passage cannot override the system prompt. Check the source text and citations before relying on generated summaries.")
