import streamlit as st
import importlib.util
import os
from modules.multimodal import transcribe_audio, synthesize_speech, describe_image, summarize_video

st.title("Multimodal AI")
st.write("Multimodal systems connect language with other input/output types. Audio and image features below use configured hosted models. Calls transmit the selected media to Groq, so use only files you are authorized to share.")
has_groq = bool(os.getenv("GROQ_API_KEY"))
tab_text, tab_audio, tab_image, tab_video = st.tabs(["Text", "Audio", "Image", "Video"])
with tab_text:
    st.write("Text input is the primary interface for this project. The Prompt Engineering Playground and RAG assistant show two common patterns: structured instruction and external grounding.")
with tab_audio:
    audio = st.file_uploader("Audio file", type=["wav","mp3","m4a","ogg","flac","webm"], key="audio_upload")
    if not has_groq: st.info("Audio transcription needs your Groq API key. Copy `.env.example` to `.env` beside `app.py`, set `GROQ_API_KEY=your_key`, then restart Streamlit.")
    if st.button("Transcribe audio", disabled=audio is None or not has_groq):
        try:
            with st.spinner("Transcribing with Groq Whisper…"): text = transcribe_audio(audio.getvalue(), audio.name)
            st.text_area("Transcript", text, height=180)
        except Exception as exc: st.warning(f"Transcription unavailable: {exc}")
    speech_text = st.text_input("Text to speak (offline system voice)", "InsightForge reports the answer with evidence.")
    has_tts = importlib.util.find_spec("pyttsx3") is not None
    if not has_tts: st.info("Offline speech needs the optional `requirements-video.txt` packages.")
    if st.button("Synthesize speech", disabled=not has_tts):
        try:
            audio_bytes = synthesize_speech(speech_text)
            st.audio(audio_bytes, format="audio/wav")
        except Exception as exc: st.warning(f"Speech synthesis unavailable: {exc}")
with tab_image:
    image = st.file_uploader("Image", type=["png","jpg","jpeg","webp"], key="image_upload")
    ask = st.text_input("Image question", "Describe the visible chart and its labels.")
    if image: st.image(image, caption=image.name, width="stretch")
    if not has_groq: st.info("Image understanding needs your Groq API key. Copy `.env.example` to `.env` beside `app.py`, set `GROQ_API_KEY=your_key`, then restart Streamlit. The image stays local without the key.")
    if st.button("Ask vision model", disabled=image is None or not has_groq):
        try:
            with st.spinner("Analyzing the image with Groq vision…"): description = describe_image(image.getvalue(), image.type, ask)
            st.write(description)
        except Exception as exc: st.warning(f"Vision model unavailable: {exc}")
with tab_video:
    st.write("A working sparse-frame demo samples keyframes with OpenCV, captions each frame with the configured vision model, then summarizes timestamped captions. It does not transcribe video audio.")
    video = st.file_uploader("Video file", type=["mp4","mov","webm","avi"], key="video_upload")
    has_video = importlib.util.find_spec("cv2") is not None
    if not has_video: st.info("Video keyframe extraction needs the optional `requirements-video.txt` packages.")
    if not has_groq: st.info("Video captions need your Groq API key. Copy `.env.example` to `.env` beside `app.py`, set `GROQ_API_KEY=your_key`, then restart Streamlit. Sampled frames are sent to Groq only when you request a summary.")
    if st.button("Summarize sampled keyframes", disabled=video is None or not has_video or not has_groq):
        try:
            with st.spinner("Sampling and describing keyframes…"): summary = summarize_video(video.getvalue(), video.name)
            st.write(summary)
            st.caption("Summary is based on at most five sampled frames; speech/audio is not included.")
        except Exception as exc: st.warning(f"Video summary unavailable: {exc}")
    st.caption("Video support requires OpenCV from requirements-video.txt and a Groq key with a vision model configured. Uploaded video is processed temporarily and sampled frames are sent to Groq.")
