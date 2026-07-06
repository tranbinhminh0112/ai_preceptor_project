from pathlib import Path

import streamlit as st

import backend.database as db
from config import LOGO_PATH
from portal_pages import assistant_workspace, examination_page, history_page, login_portal
from portal_pages.common import get_logo_base64, init_portal_state


st.set_page_config(
    page_title="Nursing AI Assistant",
    page_icon=LOGO_PATH,
    layout="wide",
    initial_sidebar_state="collapsed",
)


def inject_css():
    css_path = Path(__file__).parent / "assets" / "styles.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def render_fixed_brand():
    encoded = get_logo_base64()
    if not encoded:
        return
    st.markdown(
        f"""
        <div class="fixed-brand">
            <img src="data:image/png;base64,{encoded}" alt="IMH logo" />
            <span>IMH</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


inject_css()
render_fixed_brand()
db.init_db()
init_portal_state()


if not st.session_state.get("logged_in", False):
    login_portal.render_login()
else:
    current_page = st.session_state.get("current_page", "menu")
    if current_page == "history":
        history_page.render()
    elif current_page == "assistant":
        assistant_workspace.render()
    elif current_page == "exam":
        examination_page.render()
    else:
        login_portal.render_menu()
