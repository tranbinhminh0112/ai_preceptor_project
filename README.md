# 🌊 Nursing AI Assistant

A Streamlit app over a nursing/psychiatry knowledge base. It answers clinical
questions (grounded, with sources), generates exam-ready questions, manages a
question bank, and lets you **upload your own documents** into the knowledge
base. Everything runs **locally** — the LLM and vision model are served by
[Ollama](https://ollama.com), embeddings run on your machine.

---

## Features

| Page | What it does |
|------|--------------|
| 💬 **Chat** | Ask clinical questions by text or voice. Answers come strictly from the knowledge base (no hallucination), with source references. Optional spoken reply. |
| 📝 **Question Generator** | Generate **QnA / MCQ / short-answer** questions. Pick **reference sources** to draw from, choose difficulty mix, then save the ones you like. |
| 📚 **Question Bank** | Browse / filter saved questions and **export everything to Excel**. |
| 📂 **Documents** | Upload **PDF / DOCX / Markdown / TXT** into the knowledge base. Images are described and tables summarised automatically, then everything is embedded and becomes searchable in Chat. Delete documents here too. |

---

## Quick start (3 steps)

### 1. Install the prerequisites
- **Python 3.11+** — <https://www.python.org/downloads/> (tick *“Add python.exe to PATH”*).
- **Ollama** — <https://ollama.com/download>. After installing, leave it running.

### 2. Run setup (one time)
Open this folder and **double-click `setup.bat`**. It will:
1. create a local virtual environment (`.venv`),
2. install all Python packages from `requirements.txt`,
3. download every model the app needs (see table below).

> ⚠️ Make sure **Ollama is running** before/while this runs — the chat & vision
> models are pulled through it. Total download is several GB and only happens once.

*(Prefer the command line? See [Manual setup](#manual-setup) below.)*

### 3. Launch
Double-click **`run.bat`**. A browser tab opens at <http://localhost:8502>.

---

## Models used

| Purpose | Model | Where it comes from | Approx size |
|---------|-------|--------------------|-------------|
| Embeddings (search) | `BAAI/bge-base-en-v1.5` | Hugging Face (auto) | ~440 MB |
| Speech-to-text | `openai/whisper-small` | Hugging Face (auto) | ~970 MB |
| Chat LLM | `qwen2.5` | Ollama | ~4.7 GB |
| Image descriptions | `qwen2.5vl:7b` | Ollama | ~6 GB |

`setup_models.py` downloads/pulls all of these. The embedding model also
auto-downloads on first app launch if it’s missing. You can re-run
`python setup_models.py` any time; already-present models are skipped.

---

## Manual setup

```powershell
# from inside the "ui_ux PoC" folder
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# make sure Ollama is running in another window:  ollama serve
.\.venv\Scripts\python.exe setup_models.py

# launch
.\.venv\Scripts\python.exe -m streamlit run app.py --server.port 8502
```

**GPU note:** `requirements.txt` installs the **CPU** build of PyTorch (works
everywhere, just slower). For an NVIDIA GPU, install the CUDA build instead:
```
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

---

## Project structure

```
ui_ux PoC/
├── app.py                 # Streamlit UI (Chat / Generator / Bank / Documents)
├── config.py              # all paths + model names (single source of truth)
├── backend/
│   ├── rag_engine.py      # bge embeddings + Chroma + BM25 hybrid + qwen2.5
│   ├── question_gen.py    # grounded question generation (JSON-constrained)
│   ├── ingest.py          # file → markdown → enrich → embed pipeline
│   ├── documents.py       # list / delete documents in the knowledge base
│   ├── voice.py           # Whisper (STT) + edge-tts/pyttsx3 (TTS)
│   └── database.py        # SQLite store + Excel export for the question bank
├── assets/styles.css      # theme
├── data/
│   ├── vector_db/         # the knowledge base (Chroma) — bundled
│   ├── uploads/           # documents you upload (created at runtime)
│   ├── audio/             # recorded / generated speech (created at runtime)
│   └── question_bank.db   # saved questions (created at runtime)
├── requirements.txt
├── setup.bat              # one-time: venv + deps + models
├── setup_models.py        # downloads/pulls all models
└── run.bat                # launches the app
```

The knowledge base lives in `data/vector_db/`, so the folder is **self-contained**.

---

## Troubleshooting

- **The window closes right after “Loading weights”.** Already handled by `run.bat`
  (it sets `KMP_DUPLICATE_LIB_OK=TRUE` to stop a PyTorch/scikit-learn OpenMP clash).
  If you launch manually, set that env var first.
- **“Ollama is not reachable” on the Documents page / chat errors.** Start Ollama
  (`ollama serve`) and make sure `qwen2.5` and `qwen2.5vl:7b` are pulled
  (`ollama list`). Image enrichment is skipped gracefully if the vision model is missing.
- **`unknown model architecture: 'mllama'`.** Your Ollama is older than 0.4. This app
  uses `qwen2.5vl:7b` for vision (not `llama3.2-vision`), so just update Ollama if you
  see this, or ignore it — ingestion still works without image descriptions.
- **First run is slow / downloads a lot.** That’s the models downloading once; later
  launches are fast.

---

## Notes for shipping to GitHub

- `.gitignore` already excludes the `.venv`, model caches, uploads, audio and the
  question-bank DB.
- The bundled `data/vector_db/` (~7 MB) **is** committed so the app works after
  cloning. If your repo is public or the documents are sensitive/copyrighted,
  uncomment `data/vector_db/` in `.gitignore` to keep it out — colleagues can then
  add documents themselves via the **Documents** page.
