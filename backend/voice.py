"""
Voice helpers: speech-to-text (Whisper) and text-to-speech (edge-tts).

- transcribe(): Whisper via transformers ASR pipeline.
- synthesize_speech(): edge-tts natural neural voice -> mp3 path.

Both are import-light at module load; heavy models are only created when first
used (and cached by the caller via @st.cache_resource where relevant).
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

import torch

from config import WHISPER_MODEL_ID, DEFAULT_STT_LANGUAGE, TTS_VOICE, AUDIO_DIR


# ---------------------------------------------------------------------------
# Speech-to-text
# ---------------------------------------------------------------------------
def load_whisper(model_id: str = WHISPER_MODEL_ID):
    """Build a Whisper ASR pipeline. Cache the return value in the UI layer."""
    from transformers import pipeline

    device = 0 if torch.cuda.is_available() else -1
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    return pipeline(
        task="automatic-speech-recognition",
        model=model_id,
        device=device,
        torch_dtype=dtype,
    )


def transcribe(asr_pipeline, audio_path: str, language: str = DEFAULT_STT_LANGUAGE) -> str:
    """Transcribe an audio file to text using a prepared Whisper pipeline."""
    result = asr_pipeline(
        str(audio_path),
        generate_kwargs={"task": "transcribe", "language": language},
    )
    return result["text"].strip()


# ---------------------------------------------------------------------------
# Text-to-speech
# ---------------------------------------------------------------------------
def _clean_for_speech(text: str) -> str:
    """Strip markdown so the spoken answer sounds natural."""
    text = re.sub(r"\|.*?\|", " ", text)          # drop markdown table rows
    text = re.sub(r"[#*`_>\-]+", " ", text)        # markdown symbols
    text = re.sub(r"\s+", " ", text).strip()
    return text


def synthesize_speech(text: str, out_name: str = "answer.mp3", voice: str = TTS_VOICE) -> str:
    """
    Generate natural speech with edge-tts and return the mp3 path.

    Falls back to pyttsx3 (offline) if edge-tts is unavailable.
    """
    spoken = _clean_for_speech(text) or "Sorry, there is nothing to read."
    out_path = str(Path(AUDIO_DIR) / out_name)

    try:
        import edge_tts

        async def _save():
            await edge_tts.Communicate(text=spoken, voice=voice).save(out_path)

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import nest_asyncio

                nest_asyncio.apply()
                loop.run_until_complete(_save())
            else:
                loop.run_until_complete(_save())
        except RuntimeError:
            asyncio.run(_save())
        return out_path
    except Exception:
        # Offline fallback
        import pyttsx3

        wav_path = str(Path(out_path).with_suffix(".wav"))
        engine = pyttsx3.init()
        engine.save_to_file(spoken, wav_path)
        engine.runAndWait()
        return wav_path
