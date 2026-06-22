"""
Central configuration for the Nursing AI UI/UX PoC.

All paths are resolved relative to the project root so the app keeps working
even if the whole "AI evaluation project with nursing team" folder is moved.
"""

from pathlib import Path

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
# This file lives in:  <PROJECT_ROOT>/ui_ux PoC/config.py
POC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = POC_DIR.parent

# Local data folder (everything the app needs lives inside the PoC folder so
# the whole thing is self-contained and can be copied to a colleague as-is).
DATA_DIR = POC_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# Knowledge base vector store — bundled inside the PoC folder.
VECTOR_DB_PATH = str(DATA_DIR / "vector_db")
COLLECTION_NAME = "unified_ai_knowledge_base"

# Local SQLite store for the generated question bank (created on first run)
QUESTION_DB_PATH = str(DATA_DIR / "question_bank.db")

# Uploaded source documents land here, one sub-folder per document
# (holds the converted content.md + extracted image assets used by ingestion).
UPLOADS_DIR = DATA_DIR / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR = str(UPLOADS_DIR)

# Temp folder for recorded audio / generated speech
AUDIO_DIR = POC_DIR / "data" / "audio"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

# IMH brand logo
LOGO_PATH = str(POC_DIR / "backend" / "imh_logo_image.png")

# ----------------------------------------------------------------------------
# Models  (must match the embedding model used to build the vector DB)
# ----------------------------------------------------------------------------
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
LLM_MODEL = "qwen2.5"            # served locally by Ollama
VISION_MODEL = "qwen2.5vl:7b"    # used to describe images during ingestion
                                 # (llama3.2-vision needs Ollama >= 0.4 for the
                                 #  'mllama' architecture; qwen2.5vl works widely)
LLM_TEMPERATURE_ANSWER = 0.1
LLM_TEMPERATURE_GENERATE = 0.2

# Retrieval
RETRIEVER_TOP_K = 3
HYBRID_WEIGHTS = [0.5, 0.5]      # [BM25, vector]

# ----------------------------------------------------------------------------
# Ingestion  (must mirror information_vectorization_v4.ipynb so newly uploaded
# documents are chunked/embedded exactly like the original knowledge base)
# ----------------------------------------------------------------------------
INGEST_CHUNK_SIZE = 1400
INGEST_CHUNK_OVERLAP = 200
INGEST_MAX_IMAGE_RESOLUTION = (800, 800)
INGEST_MAX_IMAGE_TOKENS = 150
# Supported upload types (everything is converted to markdown + image assets)
SUPPORTED_UPLOAD_TYPES = ["pdf", "docx", "md", "txt"]

# Voice
WHISPER_MODEL_ID = "openai/whisper-small"
DEFAULT_STT_LANGUAGE = "en"
TTS_VOICE = "en-US-JennyNeural"

# ----------------------------------------------------------------------------
# UI theme  (ocean blue + white)
# ----------------------------------------------------------------------------
THEME = {
    "deep":     "#023E8A",   # deep ocean
    "primary":  "#0077B6",   # ocean blue
    "accent":   "#00B4D8",   # bright sea
    "light":    "#90E0EF",   # shallow water
    "mist":     "#CAF0F8",   # sea mist
    "white":    "#FFFFFF",
    "ink":      "#03263F",   # near-black navy text
}
