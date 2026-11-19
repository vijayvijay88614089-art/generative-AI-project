from __future__ import annotations
import os
from pathlib import Path


def transcribe_audio(payload: bytes, filename: str) -> str:
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("Set GROQ_API_KEY to use Groq Whisper transcription. No transcript is fabricated in local mode.")
    from groq import Groq
    import tempfile
    suffix = Path(filename).suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix) as temp:
        temp.write(payload); temp.flush()
        with open(temp.name, "rb") as audio:
            response = Groq(api_key=os.environ["GROQ_API_KEY"]).audio.transcriptions.create(
                file=(filename, audio.read()), model=os.getenv("GROQ_AUDIO_MODEL", "whisper-large-v3-turbo"), response_format="text")
    return str(response)


def synthesize_speech(text: str) -> bytes:
    """Offline text-to-speech through Windows system voices, when pyttsx3 is installed."""
    try:
        import pyttsx3
    except ImportError as exc:
        raise RuntimeError("Install optional speech support with `pip install pyttsx3`.") from exc
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp:
        path = temp.name
    try:
        engine = pyttsx3.init()
        engine.save_to_file(text[:2000], path)
        engine.runAndWait()
        return Path(path).read_bytes()
    finally:
        try: os.unlink(path)
        except OSError: pass


def describe_image(payload: bytes, mime_type: str, question: str = "Describe the image.") -> str:
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("Set GROQ_API_KEY and a vision-capable model to run image understanding.")
    import base64
    from groq import Groq
    data = base64.b64encode(payload).decode("ascii")
    model = os.getenv("GROQ_VISION_MODEL", "qwen/qwen3.8-27b")
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    response = client.chat.completions.create(model=model, messages=[{"role": "user", "content": [
        {"type": "text", "text": question}, {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{data}"}}]}], max_completion_tokens=500)
    return response.choices[0].message.content or ""


def summarize_video(payload: bytes, filename: str, max_frames: int = 5) -> str:
    """Extract sparse keyframes locally, caption each via configured vision API, summarize captions."""
    import tempfile
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Install optional video support with `pip install -r requirements-video.txt` (OpenCV).") from exc
    import base64
    import numpy as np
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("Set GROQ_API_KEY and a vision-capable model to summarize video.")
    suffix = Path(filename).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix) as temp:
        temp.write(payload); temp.flush()
        capture = cv2.VideoCapture(temp.name)
        if not capture.isOpened(): raise RuntimeError("Could not decode the uploaded video. Check the codec or format.")
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)); fps = float(capture.get(cv2.CAP_PROP_FPS) or 1)
        if total <= 0: capture.release(); raise RuntimeError("The video has no readable frames.")
        indices = np.linspace(0, total - 1, min(max_frames, total), dtype=int)
        captions = []
        for frame_no in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_no)); ok, frame = capture.read()
            if not ok: continue
            frame = cv2.resize(frame, (min(768, frame.shape[1]), min(768, frame.shape[0])))
            ok, encoded = cv2.imencode(".jpg", frame)
            if not ok: continue
            caption = describe_image(encoded.tobytes(), "image/jpeg", "Describe the visible scene and any readable text; do not infer events outside this frame.")
            captions.append({"time_seconds": round(float(frame_no / fps), 2), "caption": caption})
        capture.release()
    if not captions: raise RuntimeError("No keyframes could be captioned.")
    from .code_generator import groq_completion
    prompt = ("Summarize this video's sampled frame captions in order. State that these are sparse keyframes and "
              "do not invent audio or actions between frames. Preserve timestamps.\n" + str(captions))
    return groq_completion(prompt)
