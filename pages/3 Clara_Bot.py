# pages/3_Clara_Bot.py
import streamlit as st
from ai_model.ui_components import load_css, render_clara, render_user
from ai_model.conversation import ClaraConversation
# from ai_model.xai import last_reasoning, render_reasoning_panel

# ------------------------------
# Page setup
# ------------------------------
st.set_page_config(page_title="Clara — Ethical Support Chat", page_icon="🦋", layout="centered")
load_css()

st.title("🦋 Clara — Ethical Support Chat")
st.markdown("""
**Clara** (Care, Legitimacy, and Accountability in Responsible AI) is an ethical and caring AI assistant powered by the lightweight **Phi-3 Mini** model.
She explores how responsible AI can assist citizens in contexts of integrity, public service, and digital ethics.
""")
st.caption("Developed by Fernando Forattini, 2025")

# ------------------------------
# Initialize conversation
# ------------------------------
if "clara" not in st.session_state:
    st.session_state.clara = ClaraConversation()
    st.session_state.clara.set_mode("support")

# ------------------------------
# Display chat history (skip re-rendered streamed messages)
# ------------------------------
for msg in st.session_state.clara.history:
    if msg.role == "assistant":
        render_clara(msg.content)
    else:
        render_user(msg.content)

# ------------------------------
# Chat input
# ------------------------------
user_input = st.chat_input(
    "Ask Clara for service updates, report a concern, or explore your rights and support options."
)

if user_input:
    # Render user bubble
    render_user(user_input)
    # Process Clara’s response (includes live streaming inside the bubble)
    st.session_state.clara.process_user_input(user_input)
