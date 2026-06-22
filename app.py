"""
Nursing AI Assistant — UI/UX Proof of Concept
================================================

A clean, professional interface over the nursing knowledge base.

Pages
-----
1. Chat               — grounded medical Q&A (text or voice in, optional voice out).
2. Question Generator — create QnA / MCQ / short-answer items from the KB and save them.
3. Question Bank      — browse, filter and manage saved questions.

Run:  streamlit run "app.py"
"""

import hashlib
from pathlib import Path

import streamlit as st

from config import AUDIO_DIR, DEFAULT_STT_LANGUAGE, TTS_VOICE, LOGO_PATH
import backend.database as db


# ---------------------------------------------------------------------------
# Page + theme
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Nursing AI Assistant", page_icon=LOGO_PATH, layout="centered")

# IMH logo in the corner (top-left of the app + top of the sidebar)
if Path(LOGO_PATH).exists():
    st.logo(LOGO_PATH, size="large", link="https://www.imh.com.sg")


def inject_css():
    css_path = Path(__file__).parent / "assets" / "styles.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


inject_css()


def page_header(title: str, subtitle: str):
    st.markdown(
        f"<div class='page-head'><h1>{title}</h1><p>{subtitle}</p></div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Cached heavy resources
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading knowledge base…")
def get_rag_engine():
    from backend.rag_engine import RAGEngine
    return RAGEngine()


@st.cache_resource(show_spinner="Loading speech-to-text model…")
def get_whisper():
    from backend.voice import load_whisper
    return load_whisper()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def difficulty_dot(level: str) -> str:
    level = (level or "medium").lower()
    cls = {"easy": "dot-easy", "medium": "dot-medium", "hard": "dot-hard"}.get(level, "dot-medium")
    return f"<span class='dot {cls}'></span>"


def source_names(grouped) -> list:
    """Unique source document names, in order."""
    return [g["source"] for g in (grouped or [])]


def render_source_line(grouped):
    names = source_names(grouped)
    if not names:
        return
    st.markdown(
        f"<div class='src-line'><b>Sources:</b> {' · '.join(names)}</div>",
        unsafe_allow_html=True,
    )


def speak(text: str, key: str) -> str:
    from backend.voice import synthesize_speech
    return synthesize_speech(text, out_name=f"{key}.mp3", voice=st.session_state.get("tts_voice", TTS_VOICE))


def meta_row(item: dict) -> str:
    parts = [item.get("question_type", ""), item.get("difficulty_level", "")]
    srcs = item.get("knowledge_source", [])
    if srcs:
        # show just the document name, before the first " | "
        docs = sorted({str(s).split(" | ")[0] for s in srcs})
        parts.append(", ".join(docs))
    text = "  ·  ".join(p for p in parts if p)
    return f"<div class='q-meta'>{difficulty_dot(item.get('difficulty_level'))}{text}</div>"


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        "<div class='brand'>Nursing AI Assistant</div>"
        "<div class='brand-sub'>Institute of Mental Health</div>",
        unsafe_allow_html=True,
    )
    st.write("")
    page = st.radio(
        "Navigate",
        ["Chat", "Question Generator", "Question Bank", "Documents"],
        label_visibility="collapsed",
    )

    st.divider()
    with st.expander("Voice settings"):
        st.session_state.setdefault("tts_voice", TTS_VOICE)
        st.session_state["tts_voice"] = st.selectbox(
            "Reply voice",
            ["en-US-JennyNeural", "en-US-AriaNeural", "en-GB-SoniaNeural", "en-US-GuyNeural"],
            index=0,
        )
        st.session_state.setdefault("stt_lang", DEFAULT_STT_LANGUAGE)
        st.session_state["stt_lang"] = st.selectbox("Question language", ["en", "vi", "zh"], index=0)

    st.divider()
    st.caption(f"{db.count_questions()} questions saved")


# ===========================================================================
# PAGE 1 — CHAT
# ===========================================================================
def page_chat():
    try:
        engine = get_rag_engine()
    except Exception as exc:
        st.error("Knowledge base could not be loaded.")
        st.info(
            "This usually means the local embedding model is not cached yet, "
            "or the machine cannot reach Hugging Face to download it."
        )
        st.exception(exc)
        return
    page_header("Chat", "Ask clinical questions by text or voice. Answers come only from the knowledge base.")

    st.session_state.setdefault("messages", [])

    c1, c2 = st.columns([1, 1])
    voice_reply = c1.toggle("Read answers aloud", value=False)
    if c2.button("Clear conversation", type="secondary"):
        st.session_state["messages"] = []
        st.rerun()

    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                st.markdown(
                    f"<div class='src-line'><b>Sources:</b> {' · '.join(msg['sources'])}</div>",
                    unsafe_allow_html=True,
                )
            if msg.get("audio") and Path(msg["audio"]).exists():
                st.audio(msg["audio"])

    # voice input
    with st.expander("Ask with your voice"):
        audio = st.audio_input("Record a question", key="voice_q", label_visibility="collapsed")
        voice_query = None
        if audio is not None:
            digest = hashlib.md5(audio.getvalue()).hexdigest()
            if st.session_state.get("last_voice_digest") != digest:
                st.session_state["last_voice_digest"] = digest
                wav_path = Path(AUDIO_DIR) / "question.wav"
                wav_path.write_bytes(audio.getvalue())
                with st.spinner("Transcribing…"):
                    from backend.voice import transcribe
                    voice_query = transcribe(get_whisper(), str(wav_path), language=st.session_state["stt_lang"])
                st.caption(f"Heard: “{voice_query}”")

    query = st.chat_input("Type a clinical question…") or voice_query

    if query:
        st.session_state["messages"].append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                from backend.rag_engine import group_sources
                answer, docs = engine.answer(query)
                grouped = group_sources(docs)
            st.markdown(answer)
            render_source_line(grouped)

            audio_path = None
            if voice_reply:
                with st.spinner("Generating audio…"):
                    audio_path = speak(answer, f"reply_{len(st.session_state['messages'])}")
                if audio_path and Path(audio_path).exists():
                    st.audio(audio_path, autoplay=True)

        st.session_state["messages"].append({
            "role": "assistant",
            "content": answer,
            "sources": source_names(grouped),
            "audio": audio_path,
        })


# ===========================================================================
# PAGE 2 — QUESTION GENERATOR
# ===========================================================================
def page_generator():
    try:
        engine = get_rag_engine()
    except Exception as exc:
        st.error("Knowledge base could not be loaded.")
        st.info(
            "Question generation depends on the same vector database and embedding model "
            "used by the chat page."
        )
        st.exception(exc)
        return
    page_header("Question Generator", "Generate exam-ready questions from the knowledge base, then save the ones you want.")

    import backend.documents as docs
    available_sources = [r["source"] for r in docs.list_documents()]

    with st.form("gen_form"):
        topic = st.text_input("Topic", value="Mental health nursing")
        c1, c2, c3 = st.columns(3)
        qna_count = c1.number_input("QnA", 0, 10, 2)
        mcq_count = c2.number_input("MCQ", 0, 10, 2)
        short_count = c3.number_input("Short answer", 0, 10, 2)
        difficulties = st.multiselect(
            "Difficulty levels", ["easy", "medium", "hard"], default=["easy", "medium", "hard"]
        )
        selected_sources = st.multiselect(
            "Reference sources",
            available_sources,
            default=[],
            help="Pick one or more documents to draw questions from. Leave empty to use the whole knowledge base.",
        )
        submitted = st.form_submit_button("Generate")

    if submitted:
        from backend.question_gen import generate_question_batch
        try:
            with st.spinner("Generating from the knowledge base…"):
                batch = generate_question_batch(
                    engine.retriever,
                    topic=topic,
                    qna_count=int(qna_count),
                    mcq_count=int(mcq_count),
                    short_answer_count=int(short_count),
                    difficulty_levels=tuple(difficulties) or ("easy", "medium", "hard"),
                    sources=selected_sources,
                    vector_db=engine.vector_db,
                )
            st.session_state["generated_batch"] = batch
        except Exception as exc:
            st.error(f"Generation failed: {exc}")

    batch = st.session_state.get("generated_batch")
    if not batch:
        return

    st.divider()
    head_l, head_r = st.columns([3, 1])
    head_l.markdown(f"**{len(batch['all_questions'])} questions** · {batch['topic']}")
    if head_r.button("Save all"):
        n = db.save_many(batch["all_questions"], topic=batch["topic"])
        st.success(f"Saved {n} questions.")

    for idx, item in enumerate(batch["all_questions"]):
        if item["question_type"] == "mcq" and item.get("options"):
            opts = "".join(f"<li>{o}</li>" for o in item["options"])
            answer_html = f"<ul>{opts}</ul><div class='q-answer'>Correct: <b>{item.get('correct_answer','')}</b></div>"
        else:
            answer_html = f"<div class='q-answer'>{item.get('answer','')}</div>"

        st.markdown(
            f"<div class='q-card'><div class='q-text'>{idx+1}. {item['question']}</div>"
            f"{answer_html}{meta_row(item)}</div>",
            unsafe_allow_html=True,
        )
        if st.button("Save", key=f"save_{idx}", type="secondary"):
            db.save_question(item, topic=batch["topic"])
            st.toast("Saved")


# ===========================================================================
# PAGE 3 — QUESTION BANK
# ===========================================================================
def page_bank():
    page_header("Question Bank", "Saved questions with difficulty and source. Filter and manage them here.")

    f1, f2, f3 = st.columns([1, 1, 2])
    diff = f1.selectbox("Difficulty", ["All", "easy", "medium", "hard"])
    qtype = f2.selectbox("Type", ["All", "qna", "mcq", "short_answer"])
    search = f3.text_input("Search")

    rows = db.fetch_questions(difficulty=diff, question_type=qtype, search=search or None)
    total = db.count_questions()

    h1, h2, h3 = st.columns([3, 1, 1])
    h1.caption(f"Showing {len(rows)} of {total} question(s)")
    if total:
        h2.download_button(
            "Export Excel",
            data=db.export_xlsx(),
            file_name="question_bank.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            help="Download all saved questions as an Excel file.",
        )
    if rows and h3.button("Clear all", type="secondary"):
        db.clear_all()
        st.rerun()

    if not rows:
        st.info("No saved questions yet. Generate some in the Question Generator.")
        return

    for row in rows:
        if row["question_type"] == "mcq" and row.get("options"):
            opts = "".join(f"<li>{o}</li>" for o in row["options"])
            answer_html = f"<ul>{opts}</ul><div class='q-answer'>Correct: <b>{row.get('correct_answer','')}</b></div>"
        else:
            answer_html = f"<div class='q-answer'>{row.get('answer','')}</div>"

        c_card, c_del = st.columns([12, 1])
        with c_card:
            st.markdown(
                f"<div class='q-card'><div class='q-text'>{row['question']}</div>"
                f"{answer_html}{meta_row(row)}</div>",
                unsafe_allow_html=True,
            )
        if c_del.button("✕", key=f"del_{row['id']}", type="secondary"):
            db.delete_question(row["id"])
            st.rerun()


# ===========================================================================
# PAGE 4 — DOCUMENTS (upload + manage the knowledge base)
# ===========================================================================
def page_documents():
    import backend.documents as docs
    import backend.ingest as ingest
    from config import SUPPORTED_UPLOAD_TYPES, LLM_MODEL, VISION_MODEL

    page_header(
        "Documents",
        "Upload new source documents into the knowledge base, or remove existing ones. "
        "Everything you add here becomes searchable on the Chat page.",
    )

    # --- enrichment status ---
    status = ingest.ollama_status()
    if status["up"] and status["has_vision"] and status["has_llm"]:
        st.success("Ollama is running — full ingestion (image descriptions + table summaries) is available.")
    elif status["up"]:
        missing = [m for m, ok in [(LLM_MODEL, status["has_llm"]), (VISION_MODEL, status["has_vision"])] if not ok]
        st.warning(
            f"Ollama is running but these models are not pulled: **{', '.join(missing)}**. "
            "Documents will still be added, but without AI image/table enrichment. "
            "Run `ollama pull " + " && ollama pull ".join(missing) + "` to enable it."
        )
    else:
        st.warning(
            "Ollama is not reachable. Documents will still be added (text + tables as raw markdown), "
            "but without AI image descriptions or table summaries. Start it with `ollama serve`."
        )

    # --- upload + ingest ---
    with st.form("upload_form"):
        files = st.file_uploader(
            "Choose document(s)",
            type=SUPPORTED_UPLOAD_TYPES,
            accept_multiple_files=True,
            help="PDF, DOCX, Markdown or TXT. Each file becomes one source document.",
        )
        c1, c2, c3 = st.columns(3)
        use_vision = c1.toggle("Describe images", value=True, help="Use the vision model on figures/diagrams.")
        use_tables = c2.toggle("Summarise tables", value=True, help="Convert tables into searchable prose.")
        use_keywords = c3.toggle("Extract keywords", value=True)
        submitted = st.form_submit_button("Ingest into knowledge base", type="primary")

    if submitted:
        if not files:
            st.error("Please choose at least one file first.")
        else:
            engine_ok = True
            try:
                embeddings = get_rag_engine().embeddings
            except Exception as exc:
                engine_ok = False
                st.error("Could not load the embedding model needed for ingestion.")
                st.exception(exc)

            if engine_ok:
                results = []
                for f in files:
                    with st.status(f"Ingesting **{f.name}**…", expanded=True) as box:
                        try:
                            stats = ingest.ingest_file(
                                f.name,
                                f.getvalue(),
                                embeddings,
                                use_vision=use_vision,
                                use_table_summary=use_tables,
                                use_keywords=use_keywords,
                                log=lambda m: box.write(m),
                            )
                            results.append(stats)
                            box.update(
                                label=f"✓ {stats['source']} — {stats['chunks']} chunk(s)",
                                state="complete",
                                expanded=False,
                            )
                        except Exception as exc:
                            box.update(label=f"✕ {f.name} failed", state="error")
                            st.exception(exc)

                if results:
                    # New chunks are in the DB — rebuild the cached RAG engine so the
                    # Chat / Generator pages pick them up immediately.
                    get_rag_engine.clear()
                    total = sum(r["chunks"] for r in results)
                    st.success(f"Done. Added {total} chunk(s) across {len(results)} document(s).")

    # --- manage existing documents ---
    st.divider()
    rows = docs.list_documents()
    st.markdown(f"**Knowledge base** · {len(rows)} document(s) · {docs.total_chunks()} chunk(s)")

    if not rows:
        st.info("No documents yet. Upload one above to get started.")
        return

    for row in rows:
        c_info, c_del = st.columns([6, 1])
        with c_info:
            bits = [f"**{row['chunks']}** chunks"]
            if row["images"]:
                bits.append(f"{row['images']} image(s)")
            if row["tables"]:
                bits.append(f"{row['tables']} table(s)")
            st.markdown(
                f"<div class='q-card'><div class='q-text'>{row['source']}</div>"
                f"<div class='q-meta'>{'  ·  '.join(bits)}</div></div>",
                unsafe_allow_html=True,
            )
        if c_del.button("Delete", key=f"deldoc_{row['source']}", type="secondary"):
            n = docs.delete_document(row["source"])
            get_rag_engine.clear()
            st.toast(f"Removed '{row['source']}' ({n} chunks)")
            st.rerun()


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
db.init_db()
if page == "Chat":
    page_chat()
elif page == "Question Generator":
    page_generator()
elif page == "Documents":
    page_documents()
else:
    page_bank()
