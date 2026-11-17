"""Speech-to-text with faster-whisper.

Audio is decoded with ffmpeg straight into a numpy array instead of letting
faster-whisper open the file through PyAV. Newer PyAV releases (>=16) dropped
an argument faster-whisper still passes, which made every transcription fail.
"""
import asyncio
import os
import subprocess
import tempfile
import threading

import numpy as np

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "en") or None
SAMPLE_RATE = 16000

_model = None
_model_lock = threading.Lock()
# Whisper on CPU is heavy; run one transcription at a time.
_transcribe_lock = threading.Lock()


class TranscriptionError(Exception):
    pass


def _get_model():
    global _model
    with _model_lock:
        if _model is None:
            from faster_whisper import WhisperModel
            _model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
    return _model


def _decode(audio_bytes: bytes, suffix: str) -> np.ndarray:
    # A temp file (not a pipe) so formats with the index at the end, like
    # iPhone .m4a voice memos, still decode.
    with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
        tmp.write(audio_bytes)
        tmp.flush()
        try:
            proc = subprocess.run(
                ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
                 "-i", tmp.name, "-f", "f32le", "-ac", "1", "-ar", str(SAMPLE_RATE), "pipe:1"],
                capture_output=True, timeout=600,
            )
        except FileNotFoundError:
            raise TranscriptionError("ffmpeg is not installed on the server")
    if proc.returncode != 0:
        err = proc.stderr.decode(errors="ignore").strip()[-300:]
        raise TranscriptionError(f"Could not decode audio: {err or 'unknown format'}")
    return np.frombuffer(proc.stdout, dtype=np.float32)


def transcribe_sync(audio_bytes: bytes, filename: str = "audio.webm", prompt: str = None) -> str:
    if not audio_bytes:
        raise TranscriptionError("The recording was empty")
    suffix = os.path.splitext(filename or "")[1] or ".webm"
    audio = _decode(audio_bytes, suffix)
    if audio.size < SAMPLE_RATE // 2:
        raise TranscriptionError("The recording was too short")
    with _transcribe_lock:
        segments, _ = _get_model().transcribe(audio, language=WHISPER_LANGUAGE, vad_filter=True,
                                                initial_prompt=prompt)
        return " ".join(s.text.strip() for s in segments).strip()


async def transcribe(audio_bytes: bytes, filename: str = "audio.webm", prompt: str = None) -> str:
    """Transcribe without blocking the event loop."""
    return await asyncio.to_thread(transcribe_sync, audio_bytes, filename, prompt)
