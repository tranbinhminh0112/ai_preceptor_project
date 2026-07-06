from __future__ import annotations

import hashlib
from pathlib import Path

import streamlit as st

import backend.database as db
import backend.documents as docs
import backend.ingest as ingest
from config import AUDIO_DIR, DEFAULT_STT_LANGUAGE, LLM_MODEL, SUPPORTED_UPLOAD_TYPES, TTS_VOICE, VISION_MODEL

from .common import get_user_profile, go_to, logo_tag, render_portal_nav


@st.cache_resource(show_spinner="Loading knowledge base...")
def get_rag_engine():
    from backend.rag_engine import RAGEngine

    return RAGEngine()


@st.cache_resource(show_spinner="Loading speech-to-text model...")
def get_whisper():
    from backend.voice import load_whisper

    return load_whisper()


def source_names(grouped) -> list[str]:
    return [g["source"] for g in (grouped or [])]


def render_source_line(grouped):
    names = source_names(grouped)
    if not names:
        return
    st.markdown(
        f"<div class='src-line'><b>Sources:</b> {' · '.join(names)}</div>",
        unsafe_allow_html=True,
    )


def difficulty_dot(level: str) -> str:
    level = (level or "medium").lower()
    cls = {"easy": "dot-easy", "medium": "dot-medium", "hard": "dot-hard"}.get(level, "dot-medium")
    return f"<span class='dot {cls}'></span>"


def meta_row(item: dict) -> str:
    parts = [item.get("question_type", ""), item.get("difficulty_level", "")]
    srcs = item.get("knowledge_source", [])
    if srcs:
        docs_only = sorted({str(src).split(" | ")[0] for src in srcs})
        parts.append(", ".join(docs_only))
    return f"<div class='q-meta'>{difficulty_dot(item.get('difficulty_level'))}{' · '.join(p for p in parts if p)}</div>"


def speak(text: str, key: str) -> str:
    from backend.voice import synthesize_speech

    return synthesize_speech(text, out_name=f"{key}.mp3", voice=st.session_state.get("tts_voice", TTS_VOICE))


def _audio_extension(audio_file) -> str:
    mime = (getattr(audio_file, "type", "") or "").lower()
    mapping = {
        "audio/wav": ".wav",
        "audio/x-wav": ".wav",
        "audio/mpeg": ".mp3",
        "audio/mp3": ".mp3",
        "audio/mp4": ".m4a",
        "audio/webm": ".webm",
        "audio/ogg": ".ogg",
    }
    fallback = Path(getattr(audio_file, "name", "recording.wav")).suffix or ".wav"
    return mapping.get(mime, fallback)


def _init_state():
    defaults = {
        "assistant_messages": [],
        "generated_batch": None,
        "tts_voice": TTS_VOICE,
        "stt_lang": DEFAULT_STT_LANGUAGE,
        "assistant_last_voice_digest": "",
        "assistant_last_voice_transcript": "",
        "assistant_pending_voice_query": "",
        "assistant_voice_error": "",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def _render_header(profile: dict):
    st.markdown(
        f"""
        <div class="page-header-card">
            <div class="header-main">
                <div class="header-brand">{logo_tag(44)}<span>AI Learning Assistant</span></div>
                <div class="header-sub">Grounded chat, question authoring, question bank management, and document ingestion</div>
            </div>
            <div class="header-user">{profile['display_name']}<br><span>{profile['role']} · {profile['staff_id']}</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_sidebar():
    with st.sidebar:
        if st.button("Dashboard", use_container_width=True):
            go_to("menu")
        st.markdown("---")
        st.markdown("**Quick References**")
        for idx, prompt in enumerate(
            [
                "What are common symptoms of anxiety in psychiatric patients?",
                "Summarise the Mental State Examination domains.",
                "Generate revision questions about de-escalation techniques.",
                "Show me documents related to safety protocols.",
            ]
        ):
            if st.button(prompt, key=f"quick_ref_{idx}", use_container_width=True):
                st.session_state["assistant_pending_voice_query"] = prompt
        st.markdown("---")
        st.session_state["tts_voice"] = st.selectbox(
            "Reply voice",
            ["en-US-JennyNeural", "en-US-AriaNeural", "en-GB-SoniaNeural", "en-US-GuyNeural"],
            index=["en-US-JennyNeural", "en-US-AriaNeural", "en-GB-SoniaNeural", "en-US-GuyNeural"].index(st.session_state["tts_voice"]) if st.session_state["tts_voice"] in ["en-US-JennyNeural", "en-US-AriaNeural", "en-GB-SoniaNeural", "en-US-GuyNeural"] else 0,
        )
        st.session_state["stt_lang"] = st.selectbox(
            "Question language",
            ["en", "vi", "zh"],
            index=["en", "vi", "zh"].index(st.session_state["stt_lang"]) if st.session_state["stt_lang"] in ["en", "vi", "zh"] else 0,
        )
        st.markdown("---")
        st.caption(f"{db.count_questions()} questions saved")
        st.caption(f"{docs.count_documents()} source documents")


def _render_chat_tab():
    try:
        engine = get_rag_engine()
    except Exception as exc:
        st.error("Knowledge base could not be loaded.")
        st.info("This usually means the local embedding model is not cached yet or the machine cannot reach Hugging Face on first setup.")
        st.exception(exc)
        return

    st.markdown("<div class='section-label'>Chat</div>", unsafe_allow_html=True)
    st.caption("Ask clinical questions by text or voice. Answers come only from the knowledge base.")

    top_left, top_right = st.columns([1, 1])
    voice_reply = top_left.toggle("Read answers aloud", value=False)
    if top_right.button("Clear conversation", type="secondary"):
        st.session_state["assistant_messages"] = []
        st.session_state["assistant_pending_voice_query"] = ""
        st.rerun()

    for message in st.session_state["assistant_messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                st.markdown(
                    f"<div class='src-line'><b>Sources:</b> {' · '.join(message['sources'])}</div>",
                    unsafe_allow_html=True,
                )
            if message.get("audio") and Path(message["audio"]).exists():
                st.audio(message["audio"])

    with st.expander("Ask with your voice"):
        audio = st.audio_input("Record a question", key="assistant_voice_input", label_visibility="collapsed")
        if audio is not None:
            payload = audio.getvalue()
            digest = hashlib.md5(payload).hexdigest()
            if st.session_state["assistant_last_voice_digest"] != digest:
                st.session_state["assistant_last_voice_digest"] = digest
                st.session_state["assistant_voice_error"] = ""
                st.session_state["assistant_pending_voice_query"] = ""
                audio_path = Path(AUDIO_DIR) / f"question_{digest}{_audio_extension(audio)}"
                audio_path.write_bytes(payload)
                try:
                    from backend.voice import transcribe

                    with st.spinner("Transcribing..."):
                        transcript = transcribe(get_whisper(), str(audio_path), language=st.session_state["stt_lang"]).strip()
                    if transcript:
                        st.session_state["assistant_last_voice_transcript"] = transcript
                        st.session_state["assistant_pending_voice_query"] = transcript
                    else:
                        st.session_state["assistant_last_voice_transcript"] = ""
                        st.session_state["assistant_voice_error"] = "No speech was detected. Please try recording again."
                except Exception as exc:
                    st.session_state["assistant_last_voice_transcript"] = ""
                    st.session_state["assistant_voice_error"] = f"Voice transcription failed: {exc}"

        if st.session_state["assistant_last_voice_transcript"]:
            st.caption(f"Heard: “{st.session_state['assistant_last_voice_transcript']}”")
        if st.session_state["assistant_voice_error"]:
            st.warning(st.session_state["assistant_voice_error"])

    typed_query = st.chat_input("Type a clinical question...")
    pending_query = st.session_state.get("assistant_pending_voice_query", "")
    query = typed_query or pending_query
    if query and pending_query and query == pending_query:
        st.session_state["assistant_pending_voice_query"] = ""

    if not query:
        return

    st.session_state["assistant_messages"].append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    grouped = []
    answer = ""
    audio_path = None
    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                from backend.rag_engine import group_sources

                answer, retrieved_docs = engine.answer(query)
                grouped = group_sources(retrieved_docs)
            st.markdown(answer)
            render_source_line(grouped)
        except Exception as exc:
            answer = f"I ran into an error while generating the answer: {exc}"
            st.error(answer)

        if voice_reply and answer:
            try:
                with st.spinner("Generating audio..."):
                    audio_path = speak(answer, f"reply_{len(st.session_state['assistant_messages'])}")
                if audio_path and Path(audio_path).exists():
                    st.audio(audio_path, autoplay=True)
            except Exception as exc:
                st.warning(f"Audio playback could not be generated: {exc}")

    st.session_state["assistant_messages"].append(
        {
            "role": "assistant",
            "content": answer,
            "sources": source_names(grouped),
            "audio": audio_path,
        }
    )


def _render_generator_tab():
    try:
        engine = get_rag_engine()
    except Exception as exc:
        st.error("Knowledge base could not be loaded.")
        st.info("Question generation depends on the same vector database and embedding model used by the chat page.")
        st.exception(exc)
        return

    available_sources = [row["source"] for row in docs.list_documents()]
    st.markdown("<div class='section-label'>Question Generator</div>", unsafe_allow_html=True)
    st.caption("Generate exam-ready questions from the knowledge base, then save the ones you want.")

    with st.form("assistant_gen_form"):
        topic = st.text_input("Topic", value="Mental health nursing")
        c1, c2, c3 = st.columns(3)
        qna_count = c1.number_input("QnA", 0, 10, 2)
        mcq_count = c2.number_input("MCQ", 0, 10, 2)
        short_count = c3.number_input("Short answer", 0, 10, 2)
        difficulties = st.multiselect("Difficulty levels", ["easy", "medium", "hard"], default=["easy", "medium", "hard"])
        selected_sources = st.multiselect(
            "Reference sources",
            available_sources,
            default=[],
            help="Pick one or more documents to draw questions from. Leave empty to use the whole knowledge base.",
        )
        submitted = st.form_submit_button("Generate", type="primary")

    if submitted:
        from backend.question_gen import generate_question_batch

        try:
            with st.spinner("Generating from the knowledge base..."):
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

    requested = batch.get("requested_counts", {})
    generated = batch.get("generated_counts", {})
    note = batch.get("generation_note")

    st.divider()
    head_left, head_right = st.columns([3, 1])
    head_left.markdown(
        f"**{len(batch['all_questions'])} questions** · {batch['topic']} · QnA {generated.get('qna', 0)}/{requested.get('qna', 0)} · MCQ {generated.get('mcq', 0)}/{requested.get('mcq', 0)} · Short {generated.get('short_answer', 0)}/{requested.get('short_answer', 0)}"
    )
    if note:
        st.info(note)
    if head_right.button("Save all"):
        saved = db.save_many(batch["all_questions"], topic=batch["topic"])
        st.success(f"Saved {saved} questions.")

    for idx, item in enumerate(batch["all_questions"]):
        if item["question_type"] == "mcq" and item.get("options"):
            options_html = "".join(f"<li>{option}</li>" for option in item["options"])
            answer_html = f"<ul>{options_html}</ul><div class='q-answer'>Correct: <b>{item.get('correct_answer', '')}</b></div>"
        else:
            answer_html = f"<div class='q-answer'>{item.get('answer', '')}</div>"

        st.markdown(
            f"<div class='q-card'><div class='q-text'>{idx + 1}. {item['question']}</div>{answer_html}{meta_row(item)}</div>",
            unsafe_allow_html=True,
        )
        if st.button("Save", key=f"generated_save_{idx}", type="secondary"):
            db.save_question(item, topic=batch["topic"])
            st.toast("Saved")


def _render_bank_tab():
    st.markdown("<div class='section-label'>Question Bank</div>", unsafe_allow_html=True)
    st.caption("Saved questions with difficulty and source. Filter and manage them here.")

    f1, f2, f3 = st.columns([1, 1, 2])
    diff = f1.selectbox("Difficulty", ["All", "easy", "medium", "hard"], key="bank_diff")
    qtype = f2.selectbox("Type", ["All", "qna", "mcq", "short_answer"], key="bank_type")
    search = f3.text_input("Search", key="bank_search")

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
        )
    if rows and h3.button("Clear all", type="secondary"):
        db.clear_all()
        st.rerun()

    if not rows:
        st.info("No saved questions yet. Generate some in the Question Generator tab.")
        return

    for row in rows:
        if row["question_type"] == "mcq" and row.get("options"):
            options_html = "".join(f"<li>{option}</li>" for option in row["options"])
            answer_html = f"<ul>{options_html}</ul><div class='q-answer'>Correct: <b>{row.get('correct_answer', '')}</b></div>"
        else:
            answer_html = f"<div class='q-answer'>{row.get('answer', '')}</div>"

        card_col, del_col = st.columns([12, 1])
        with card_col:
            st.markdown(
                f"<div class='q-card'><div class='q-text'>{row['question']}</div>{answer_html}{meta_row(row)}</div>",
                unsafe_allow_html=True,
            )
        with del_col:
            if st.button("✕", key=f"bank_del_{row['id']}", type="secondary"):
                db.delete_question(row["id"])
                st.rerun()


def _render_documents_tab():
    st.markdown("<div class='section-label'>Documents</div>", unsafe_allow_html=True)
    st.caption("Upload new source documents into the knowledge base, or remove existing ones. Everything you add here becomes searchable in Chat.")

    status = ingest.ollama_status()
    if status["up"] and status["has_vision"] and status["has_llm"]:
        st.success("Ollama is running and full ingestion is available.")
    elif status["up"]:
        missing = [model for model, ok in [(LLM_MODEL, status["has_llm"]), (VISION_MODEL, status["has_vision"])] if not ok]
        st.warning(
            f"Ollama is running but these models are not pulled yet: {', '.join(missing)}. Documents will still be added without full AI enrichment."
        )
    else:
        st.warning("Ollama is not reachable. Documents can still be added, but without image or table enrichment.")

    with st.form("assistant_upload_form"):
        files = st.file_uploader(
            "Choose document(s)",
            type=SUPPORTED_UPLOAD_TYPES,
            accept_multiple_files=True,
            help="PDF, DOCX, Markdown or TXT. Each file becomes one source document.",
        )
        c1, c2, c3 = st.columns(3)
        use_vision = c1.toggle("Describe images", value=True)
        use_tables = c2.toggle("Summarise tables", value=True)
        use_keywords = c3.toggle("Extract keywords", value=True)
        submitted = st.form_submit_button("Ingest into knowledge base", type="primary")

    if submitted:
        if not files:
            st.error("Please choose at least one file first.")
        else:
            try:
                embeddings = get_rag_engine().embeddings
            except Exception as exc:
                st.error("Could not load the embedding model needed for ingestion.")
                st.exception(exc)
                embeddings = None

            if embeddings is not None:
                results = []
                for file in files:
                    with st.status(f"Ingesting **{file.name}**...", expanded=True) as box:
                        try:
                            stats = ingest.ingest_file(
                                file.name,
                                file.getvalue(),
                                embeddings,
                                use_vision=use_vision,
                                use_table_summary=use_tables,
                                use_keywords=use_keywords,
                                log=lambda message: box.write(message),
                            )
                            results.append(stats)
                            box.update(label=f"✓ {stats['source']} — {stats['chunks']} chunk(s)", state="complete", expanded=False)
                        except Exception as exc:
                            box.update(label=f"✕ {file.name} failed", state="error")
                            st.exception(exc)

                if results:
                    get_rag_engine.clear()
                    total_chunks = sum(item["chunks"] for item in results)
                    st.success(f"Done. Added {total_chunks} chunk(s) across {len(results)} document(s).")

    st.divider()
    rows = docs.list_documents()
    st.markdown(f"**Knowledge base** · {len(rows)} document(s) · {docs.total_chunks()} chunk(s)")
    if not rows:
        st.info("No documents yet. Upload one above to get started.")
        return

    for row in rows:
        info_col, del_col = st.columns([6, 1])
        with info_col:
            parts = [f"**{row['chunks']}** chunks"]
            if row["images"]:
                parts.append(f"{row['images']} image(s)")
            if row["tables"]:
                parts.append(f"{row['tables']} table(s)")
            st.markdown(
                f"<div class='q-card'><div class='q-text'>{row['source']}</div><div class='q-meta'>{' · '.join(parts)}</div></div>",
                unsafe_allow_html=True,
            )
        with del_col:
            if st.button("Delete", key=f"doc_del_{row['source']}", type="secondary"):
                deleted = docs.delete_document(row["source"])
                get_rag_engine.clear()
                st.toast(f"Removed '{row['source']}' ({deleted} chunks)")
                st.rerun()


def render():
    _init_state()
    profile = get_user_profile()
    render_portal_nav("assistant")
    _render_header(profile)
    _render_sidebar()

    st.markdown(
        """
        <div class="panel-card">
            <div class="panel-kicker">Unified Assistant</div>
            <div class="panel-title">Chat, question generation, question bank, and source documents are grouped together here.</div>
            <div class="panel-copy">This keeps the learning workflow in one place instead of splitting it across separate navigation items.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    chat_tab, generator_tab, bank_tab, documents_tab = st.tabs(["Chat", "Question Generator", "Question Bank", "Documents"])
    with chat_tab:
        _render_chat_tab()
    with generator_tab:
        _render_generator_tab()
    with bank_tab:
        _render_bank_tab()
    with documents_tab:
        _render_documents_tab()
