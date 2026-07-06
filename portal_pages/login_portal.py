from __future__ import annotations

import streamlit as st

import backend.database as db
import backend.documents as docs

from .common import get_user_profile, go_to, login_user, logo_tag, render_portal_nav


def render_login():
    st.markdown(
        """
        <div class="portal-shell">
            <div class="login-hero">
                <div class="eyebrow">Institute of Mental Health</div>
                <h1>Clinical Learning Portal</h1>
                <p>Unified workspace for guided learning, grounded Q&amp;A, question generation, and competency review.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _, col_mid, _ = st.columns([1.12, 1.05, 1.12])
    with col_mid:
        with st.container(border=True):
            st.markdown(f"<div style='text-align:center'>{logo_tag(66)}</div>", unsafe_allow_html=True)
            st.markdown("<div class='login-title'>IMH Learning Portal</div>", unsafe_allow_html=True)
            st.markdown("<div class='login-subtitle'>Secure staff access for nursing education workflows</div>", unsafe_allow_html=True)

            staff_id = st.text_input("Staff ID", placeholder="Ex: RN-8821", key="login_staff_id")
            st.text_input("Password", type="password", placeholder="Enter password", key="login_password")
            st.caption("Prototype login for internal demo routing.")

            if st.button("Authenticate", type="primary", use_container_width=True):
                if (staff_id or "").strip():
                    login_user(staff_id)
                    st.rerun()
                else:
                    st.error("Staff ID is required.")

    st.markdown(
        "<div class='portal-footer'>© 2026 Data Science Office · Nursing AI Assistant prototype</div>",
        unsafe_allow_html=True,
    )


def _menu_card(section: str, title: str, desc: str, button_label: str, target: str):
    with st.container(border=True):
        st.markdown(
            f"""
            <div class="panel-kicker">{section}</div>
            <div class="menu-card-title">{title}</div>
            <div class="menu-card-desc">{desc}</div>
            """,
            unsafe_allow_html=True,
        )
        if st.button(button_label, key=f"menu_{target}", use_container_width=True):
            go_to(target)


def render_menu():
    profile = get_user_profile()
    render_portal_nav("menu")

    top_left, top_right = st.columns([5, 1.2])
    with top_left:
        st.markdown(
            f"""
            <div class="dashboard-hero">
                <div class="eyebrow">Welcome back</div>
                <h1>{profile['display_name']}</h1>
                <p>{profile['role']} · {profile['unit']} · ID: {profile['staff_id']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with top_right:
        st.markdown(
            """
            <div class="panel-card" style="padding:1rem; margin-top:.15rem;">
                <div class="panel-kicker">Environment</div>
                <div class="panel-copy">Local demo workspace is ready for navigation, review, and assessment.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(
            f"<div class='metric-card'><div class='metric-label'>Saved Questions</div><div class='metric-value'>{db.count_questions()}</div></div>",
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f"<div class='metric-card'><div class='metric-label'>Knowledge Docs</div><div class='metric-value'>{docs.count_documents()}</div></div>",
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            f"<div class='metric-card'><div class='metric-label'>Indexed Chunks</div><div class='metric-value'>{docs.total_chunks()}</div></div>",
            unsafe_allow_html=True,
        )
    with m4:
        st.markdown(
            "<div class='metric-card'><div class='metric-label'>Exam Modules</div><div class='metric-value'>1 Live</div></div>",
            unsafe_allow_html=True,
        )

    st.write("")
    c1, c2, c3 = st.columns(3)
    with c1:
        _menu_card(
            "Learning Review",
            "Learning History",
            "Review progress snapshots, study activity, advisory notes, and recent learning artefacts.",
            "View History",
            "history",
        )
    with c2:
        _menu_card(
            "Knowledge Tools",
            "AI Learning Assistant",
            "Use grounded chat, generate questions, manage the question bank, and curate source documents in one place.",
            "Open Assistant",
            "assistant",
        )
    with c3:
        _menu_card(
            "Assessment",
            "Competency Exam",
            "Run the assessment flow with timer, case study review, scoring summary, and advisory report.",
            "Start Assessment",
            "exam",
        )

    st.markdown(
        """
        <div class="panel-card" style="margin-top:1.2rem;">
            <div class="panel-kicker">System Notice</div>
            <div class="panel-title">This portal is using the local knowledge base currently bundled in this project.</div>
            <div class="panel-copy">Uploaded documents become searchable inside the assistant workspace immediately after ingestion.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
