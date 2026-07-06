from __future__ import annotations

import datetime
import time
from pathlib import Path

import streamlit as st

from .common import get_user_profile, go_to, logo_tag, render_portal_nav, reset_exam_state


VIDEO_PATH = Path(__file__).resolve().parent.parent / "assets" / "videos" / "case_study_video.mp4"
EXAM_DURATION_SEC = 20 * 60


QUESTIONS = [
    {
        "id": 1,
        "type": "multiple_choice",
        "question": "A patient is admitted with severe anxiety and tremors. Which of the following is the priority nursing intervention?",
        "options": [
            "Administer prescribed anti-anxiety medication immediately.",
            "Assess the patient's vital signs and physical status.",
            "Teach the patient deep breathing exercises.",
            "Place the patient in a quiet, dim room.",
        ],
        "answer": "Assess the patient's vital signs and physical status.",
        "display_answer": "Assess the patient's vital signs and physical status.",
        "ai_logic": "Pattern recognition matched the ADPIE principle of assessing first before intervening.",
        "keywords": ["Assessment", "Safety Protocol", "Physical Baseline"],
    },
    {
        "id": 2,
        "type": "short_answer",
        "question": "What is the medical abbreviation for 'Nothing by Mouth'?",
        "answer": "NPO",
        "display_answer": "NPO (Nil Per Os)",
        "ai_logic": "Exact terminology match against standard clinical abbreviations.",
        "keywords": ["Medical Terminology", "Safety Order"],
    },
    {
        "id": 3,
        "type": "multiple_choice",
        "question": "Which category does 'Flight of Ideas' belong to in a Mental State Examination (MSE)?",
        "options": ["Perception", "Insight", "Thought Process", "Appearance"],
        "answer": "Thought Process",
        "display_answer": "Thought Process",
        "ai_logic": "Ontology mapping aligned the symptom with the Thought Process domain.",
        "keywords": ["MSE Structure", "Psychopathology"],
    },
    {
        "id": 4,
        "type": "essay",
        "question": "Explain the importance of patient confidentiality in a mental health setting.",
        "answer": "manual_review",
        "display_answer": "Confidentiality fosters therapeutic trust and is legally mandated. It protects patients from stigma and discrimination.",
        "ai_logic": "Semantic review looks for trust, legality, patient dignity, and stigma prevention.",
        "keywords": ["Therapeutic Trust", "PDPA", "Stigma Reduction", "Ethical Duty"],
    },
    {
        "id": 5,
        "type": "short_answer",
        "question": "What is the normal range for adult heart rate (beats per minute)?",
        "answer": "60-100",
        "display_answer": "60-100 bpm",
        "ai_logic": "Parameter validation checks whether the response is within accepted clinical limits.",
        "keywords": ["Vital Signs", "Cardiology Parameters"],
    },
    {
        "id": 6,
        "type": "essay",
        "question": "Watch the video below. Describe the patient's mental state (Appearance, Speech, Mood, Thought Content, Perception, Insight, Risk).",
        "video": str(VIDEO_PATH),
        "answer": "manual_review",
        "display_answer": "Patient appears anxious and restless. Mood is low and on edge. Thought content is ruminative. Insight is intact and risk appears low.",
        "ai_logic": "Multi-modal review compares written observations against the reference case markers.",
        "keywords": ["Anxious", "Ruminating Thoughts", "Insight", "Risk Review"],
    },
]


def _init_state():
    st.session_state.setdefault("exam_index", 0)
    st.session_state.setdefault("user_answers", {})
    st.session_state.setdefault("exam_submitted", False)
    st.session_state.setdefault("ai_analyzed", False)
    st.session_state.setdefault("show_exam_advisory", False)
    st.session_state.setdefault("exam_start_time", time.time())


def _render_header(profile: dict):
    st.markdown(
        f"""
        <div class="page-header-card">
            <div class="header-main">
                <div class="header-brand">{logo_tag(44)}<span>Competency Assessment</span></div>
                <div class="header-sub">Timed assessment flow with objective scoring and advisory review</div>
            </div>
            <div class="header-user">{profile['display_name']}<br><span>{profile['role']} · {profile['staff_id']}</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_sidebar(timer_display: str):
    with st.sidebar:
        if st.button("Main Menu", use_container_width=True):
            reset_exam_state()
            go_to("menu")
        st.markdown("---")
        st.markdown(
            f"""
            <div class="panel-card" style="padding:1rem;">
                <div class="panel-kicker">Time Remaining</div>
                <div class="metric-value" style="font-family:monospace;">{timer_display}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.info("AI Evaluation Module: Online\n\nLogging: Active")


def _grade_objective_questions() -> tuple[int, int]:
    score = 0
    total = 0
    for question in QUESTIONS:
        answer = str(st.session_state["user_answers"].get(question["id"], "")).strip()
        if question["type"] not in {"multiple_choice", "short_answer"}:
            continue
        total += 1
        if answer.lower() == str(question["answer"]).lower():
            score += 1
    return score, total


def _render_results():
    if not st.session_state["ai_analyzed"]:
        status_placeholder = st.empty()
        progress_bar = st.progress(0)
        steps = [
            "Connecting to Evaluation Engine [v4.2]...",
            "Tokenizing response vectors...",
            "Running semantic matching against clinical guidance...",
            "Calculating confidence metrics...",
            "Finalizing report generation...",
        ]
        for idx, step in enumerate(steps, start=1):
            status_placeholder.markdown(f"**System Log:** `{step}`")
            progress_bar.progress(idx * 20)
            time.sleep(0.65)
        progress_bar.empty()
        status_placeholder.empty()
        st.session_state["ai_analyzed"] = True
        st.rerun()

    score, total = _grade_objective_questions()
    percentage = (score / total) * 100 if total else 0
    recommendation = "COMPETENT" if percentage >= 80 else "REVIEW REQUIRED"

    st.markdown(
        f"""
        <div class="panel-card">
            <div class="panel-kicker">AI-Powered Competency Report</div>
            <div class="panel-title">Engine: Clinical-BERT-V4 · Confidence 98.2% · {datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}</div>
            <div class="panel-copy">Objective items are auto-scored while essay and video responses are summarised for supervisory review.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Objective Score", f"{score}/{total}")
    with c2:
        st.metric("Percentage", f"{percentage:.0f}%")
    with c3:
        st.metric("System Recommendation", recommendation)

    st.write("")
    if st.button("Generate AI Advisory Report", type="primary"):
        st.session_state["show_exam_advisory"] = True

    if st.session_state["show_exam_advisory"]:
        band = "Competent" if percentage >= 80 else "Developing"
        advisory_html = (
            '<div class="advisory-report">'
            '<div class="advisory-head">'
            '<div><div class="panel-kicker">AI Advisory Report</div>'
            '<div class="advisory-title">Competency Analysis &amp; Preceptor Guidance</div></div>'
            f'<span class="advisory-band">{band} · {percentage:.0f}%</span>'
            '</div>'
            '<div class="advisory-comment">'
            f'<div class="advisory-comment-head">{logo_tag(26)}<span>AI Preceptor Comment</span></div>'
            '<p>Overall, this is an <b>encouraging performance</b>. The candidate demonstrated '
            'strong safety-first clinical reasoning — consistently prioritising patient assessment '
            'before intervention — and reliable recall of core clinical terminology and vital-sign '
            'parameters.</p>'
            '<p>Free-text responses on the Mental State Examination were on the right track but '
            'remained somewhat surface-level. Descriptions of <b>speech</b> and <b>thought process</b> '
            'would benefit from more precise clinical language (e.g. rate, volume, coherence) rather '
            'than general impressions. The confidentiality essay correctly identified the duty of care '
            'but could be strengthened by referencing the relevant legal framework.</p>'
            '<p class="advisory-signoff">— Generated by Clinical-BERT-V4 · Reviewed by Senior Preceptor (pending sign-off)</p>'
            '</div>'
            '<div class="advisory-cols">'
            '<div class="advisory-col strengths">'
            '<div class="advisory-col-label">Key Strengths</div>'
            '<ul><li>Safety-first prioritisation (ADPIE)</li>'
            '<li>Accurate clinical terminology recall</li>'
            '<li>Correct vital-sign parameters</li></ul>'
            '</div>'
            '<div class="advisory-col focus">'
            '<div class="advisory-col-label">Areas to Develop</div>'
            '<ul><li>Precision in MSE speech description</li>'
            '<li>Depth of narrative case analysis</li>'
            '<li>Citing legal / ethical frameworks</li></ul>'
            '</div>'
            '</div>'
            '<div class="advisory-actions">'
            '<div class="advisory-col-label">Recommended Next Steps</div>'
            '<p>Review MSE speech and thought-process descriptors, then complete two de-escalation '
            'simulation scenarios before the next supervised review.</p>'
            '</div>'
            '</div>'
        )
        st.markdown(advisory_html, unsafe_allow_html=True)

    st.divider()
    for question in QUESTIONS:
        user_answer = st.session_state["user_answers"].get(question["id"], "No Response")
        if question["type"] in {"multiple_choice", "short_answer"}:
            badge = "<span class='status-pill pass'>Matched</span>" if str(user_answer).lower() == str(question["answer"]).lower() else "<span class='status-pill fail'>Mismatch</span>"
        else:
            badge = "<span class='status-pill info'>Semantic Review</span>"

        with st.expander(f"Q{question['id']}: {question['question']}", expanded=True):
            st.markdown(f"**Analysis Status:** {badge}", unsafe_allow_html=True)
            left, right = st.columns(2)
            with left:
                st.info(f"**Student Response:**\n\n{user_answer}")
            with right:
                st.success(f"**Clinical Standard:**\n\n{question['display_answer']}")
            keyword_html = "".join(f"<span class='keyword-chip'>{keyword}</span>" for keyword in question["keywords"])
            st.markdown(
                f"""
                <div class="panel-card" style="padding:1rem; margin-top:.6rem;">
                    <div class="panel-kicker">Evaluation Logic</div>
                    <div class="panel-copy">{question['ai_logic']}</div>
                    <div style="margin-top:.8rem;">{keyword_html}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _render_question():
    current_question = QUESTIONS[st.session_state["exam_index"]]
    current_id = current_question["id"]

    p1, p2 = st.columns([4, 1])
    with p1:
        st.progress((st.session_state["exam_index"] + 1) / len(QUESTIONS))
    with p2:
        st.markdown(
            f"<div style='text-align:right; font-weight:700; color:#0d5c73;'>{st.session_state['exam_index'] + 1} / {len(QUESTIONS)}</div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        f"""
        <div class="panel-card">
            <div class="panel-kicker">{current_question['type'].replace('_', ' ')}</div>
            <div class="panel-title">Q{current_question['id']}. {current_question['question']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if current_question.get("video"):
        video_file = Path(current_question["video"])
        if video_file.exists():
            v1, v2, v3 = st.columns([1, 2, 1])
            with v2:
                st.video(str(video_file))
        else:
            st.warning(f"Video case study not found at `{video_file}`.")

    saved_answer = st.session_state["user_answers"].get(current_id, "")
    if current_question["type"] == "multiple_choice":
        default_index = None
        if saved_answer in current_question["options"]:
            default_index = current_question["options"].index(saved_answer)
        selected = st.radio("Select Answer", current_question["options"], index=default_index, key=f"exam_q_{current_id}")
        if selected:
            st.session_state["user_answers"][current_id] = selected
    elif current_question["type"] == "short_answer":
        value = st.text_input("Answer", value=saved_answer, key=f"exam_q_{current_id}")
        if value is not None:
            st.session_state["user_answers"][current_id] = value
    else:
        value = st.text_area("Detailed Answer", value=saved_answer, height=220, key=f"exam_q_{current_id}")
        if value is not None:
            st.session_state["user_answers"][current_id] = value

    nav_left, _, nav_right = st.columns([1, 2, 1])
    with nav_left:
        if st.session_state["exam_index"] > 0 and st.button("Previous"):
            st.session_state["exam_index"] -= 1
            st.rerun()
    with nav_right:
        if st.session_state["exam_index"] < len(QUESTIONS) - 1:
            if st.button("Next"):
                st.session_state["exam_index"] += 1
                st.rerun()
        elif st.button("Submit Assessment", type="primary"):
            st.session_state["exam_submitted"] = True
            st.rerun()


def render():
    _init_state()
    profile = get_user_profile()
    render_portal_nav("exam")
    _render_header(profile)

    elapsed = time.time() - st.session_state["exam_start_time"]
    remaining = EXAM_DURATION_SEC - elapsed
    if remaining <= 0 and not st.session_state["exam_submitted"]:
        st.session_state["exam_submitted"] = True
        st.rerun()

    mins, secs = divmod(int(max(0, remaining)), 60)
    _render_sidebar(f"{mins:02d}:{secs:02d}")

    if st.button("Exit Assessment", type="secondary"):
        reset_exam_state()
        go_to("menu")

    if st.session_state["exam_submitted"]:
        _render_results()
    else:
        _render_question()

    st.markdown("<div class='portal-footer'>© 2026 Data Science Office · Assessment prototype</div>", unsafe_allow_html=True)
