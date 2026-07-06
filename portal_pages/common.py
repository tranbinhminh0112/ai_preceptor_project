from __future__ import annotations

import base64
from functools import lru_cache
from pathlib import Path

import streamlit as st

from config import LOGO_PATH


DEFAULT_USER_PROFILE = {
    "display_name": "Tran Binh Minh",
    "role": "Junior Staff Nurse",
    "unit": "Ward 5",
    "staff_id": "RN-8821",
}


def init_portal_state():
    defaults = {
        "logged_in": False,
        "current_page": "login",
        "user_profile": dict(DEFAULT_USER_PROFILE),
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def go_to(page_name: str):
    st.session_state["current_page"] = page_name
    st.rerun()


def reset_exam_state():
    for key in [
        "exam_index",
        "user_answers",
        "exam_submitted",
        "exam_start_time",
        "ai_analyzed",
        "show_exam_advisory",
    ]:
        st.session_state.pop(key, None)


def logout():
    reset_exam_state()
    st.session_state["logged_in"] = False
    st.session_state["current_page"] = "login"
    for key in [
        "assistant_messages",
        "generated_batch",
        "assistant_last_voice_digest",
        "assistant_last_voice_transcript",
        "assistant_pending_voice_query",
        "assistant_voice_error",
    ]:
        st.session_state.pop(key, None)
    st.rerun()


def login_user(staff_id: str):
    clean_id = (staff_id or "").strip() or DEFAULT_USER_PROFILE["staff_id"]
    profile = dict(DEFAULT_USER_PROFILE)
    profile["staff_id"] = clean_id
    if clean_id.upper() != DEFAULT_USER_PROFILE["staff_id"]:
        profile["display_name"] = "IMH Staff"
    st.session_state["user_profile"] = profile
    st.session_state["logged_in"] = True
    st.session_state["current_page"] = "menu"


def get_user_profile() -> dict:
    return dict(st.session_state.get("user_profile", DEFAULT_USER_PROFILE))


@lru_cache(maxsize=1)
def get_logo_base64() -> str:
    logo_path = Path(LOGO_PATH)
    if not logo_path.exists():
        return ""
    return base64.b64encode(logo_path.read_bytes()).decode("utf-8")


def logo_tag(height: int = 48) -> str:
    encoded = get_logo_base64()
    if not encoded:
        return ""
    return f'<img src="data:image/png;base64,{encoded}" style="height:{height}px; object-fit:contain;">'


def render_portal_nav(active_page: str):
    labels = [
        ("menu", "Dashboard"),
        ("assistant", "AI Learning Assistant"),
        ("history", "History"),
        ("exam", "Assessment"),
    ]

    st.markdown("<div class='top-nav-shell'>", unsafe_allow_html=True)
    cols = st.columns([1, 1, 1, 1, 0.9])
    for col, (page_name, label) in zip(cols[:4], labels):
        with col:
            button_type = "primary" if active_page == page_name else "secondary"
            if st.button(label, key=f"topnav_{page_name}", type=button_type, use_container_width=True):
                if page_name != active_page:
                    if active_page == "exam":
                        reset_exam_state()
                    go_to(page_name)
    with cols[4]:
        if st.button("Log Out", key="topnav_logout", type="secondary", use_container_width=True):
            logout()
    st.markdown("</div>", unsafe_allow_html=True)
