"""
One-time model downloader for the Nursing AI Assistant.

Run this once after installing the Python requirements:

    python setup_models.py

It makes sure every model the app needs is present locally:
  1. Embedding model  (BAAI/bge-base-en-v1.5)   — from Hugging Face
  2. Speech-to-text   (openai/whisper-small)     — from Hugging Face
  3. Chat LLM         (qwen2.5)                  — pulled into Ollama
  4. Vision model     (qwen2.5vl:7b)             — pulled into Ollama

Steps 1-2 need internet (a few hundred MB to ~1 GB, cached afterwards).
Steps 3-4 need Ollama running (`ollama serve`) and download several GB.
Anything that fails is reported but does not stop the rest — you can re-run
this script any time; already-present models are skipped quickly.
"""

import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from config import EMBEDDING_MODEL, WHISPER_MODEL_ID, LLM_MODEL, VISION_MODEL


def hr(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def download_embeddings() -> bool:
    hr(f"1/4  Embedding model: {EMBEDDING_MODEL}")
    try:
        from langchain_huggingface import HuggingFaceEmbeddings

        emb = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            encode_kwargs={"normalize_embeddings": True},
        )
        emb.embed_query("warm-up")  # force the weights to load
        print("   OK — embedding model ready.")
        return True
    except Exception as exc:
        print(f"   FAILED: {exc}")
        return False


def download_whisper() -> bool:
    hr(f"2/4  Speech-to-text model: {WHISPER_MODEL_ID}")
    try:
        from transformers import pipeline

        pipeline(task="automatic-speech-recognition", model=WHISPER_MODEL_ID)
        print("   OK — Whisper model ready.")
        return True
    except Exception as exc:
        print(f"   FAILED: {exc}")
        print("   (Voice input will be unavailable until this succeeds.)")
        return False


def pull_ollama(model: str, index: str) -> bool:
    hr(f"{index}  Ollama model: {model}")
    try:
        import ollama

        existing = [m.get("model", m.get("name", "")) for m in ollama.list().get("models", [])]
        base = model.split(":")[0]
        if any(n == model or n.split(":")[0] == base for n in existing):
            print("   Already present — skipping.")
            return True

        print("   Pulling (this can take several minutes)…")
        last = ""
        for prog in ollama.pull(model, stream=True):
            status = prog.get("status", "")
            if status and status != last:
                print(f"     {status}")
                last = status
        print("   OK — model pulled.")
        return True
    except Exception as exc:
        print(f"   FAILED: {exc}")
        print("   Make sure Ollama is installed and running (`ollama serve`).")
        return False


def main():
    print("Setting up models for the Nursing AI Assistant…")
    results = {
        "embeddings": download_embeddings(),
        "whisper": download_whisper(),
        "qwen2.5 (chat)": pull_ollama(LLM_MODEL, "3/4"),
        "qwen2.5vl (vision)": pull_ollama(VISION_MODEL, "4/4"),
    }

    hr("Summary")
    for name, ok in results.items():
        print(f"   {'OK   ' if ok else 'FAILED'}  {name}")

    if all(results.values()):
        print("\nAll set! You can now launch the app with run.bat (or: streamlit run app.py)")
    else:
        print("\nSome items failed. The app still runs, but related features may be limited.")
        print("Re-run `python setup_models.py` after fixing the issue above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
