import base64
from pathlib import Path
import streamlit as st
import time

# ------------------------------
# CSS
# ------------------------------
def load_css():
    """Load Clara’s CSS styling."""
    st.markdown("""
    <style>
      [data-testid="stChatMessage"] { margin-bottom: 1.1rem !important; }

      .clara-container {
          display: flex;
          align-items: flex-start;
          gap: 1rem;
          margin: 1rem 0;
      }

      .clara-container img {
          width: 100px; height: 100px;
          border-radius: 50%;
          object-fit: cover;
          border: 2px solid #bfa15a;
          flex-shrink: 0;
      }

      .clara-text {
          background-color: #f7f9fc;
          padding: 1rem;
          border-radius: 1rem;
          box-shadow: 0 1px 4px rgba(0,0,0,0.08);
          font-size: 1.05rem; line-height: 1.5; color: #333;
      }

      .clara-name {
          font-weight: 700;
          margin-bottom: 0.3rem;
          color: #0b3d91;
      }

      .user-container {
          display: flex;
          flex-direction: row-reverse;
          align-items: flex-start;
          gap: 1rem;
          margin: 1rem 0;
      }

      .user-container img {
          width: 100px; height: 100px;
          border-radius: 50%;
          object-fit: cover;
          border: 2px solid #4f6da3;
          flex-shrink: 0;
      }

      .user-text {
          background-color: #eef4ff;
          padding: 1rem;
          border-radius: 1rem;
          box-shadow: 0 1px 4px rgba(0,0,0,0.05);
          font-size: 1.05rem; line-height: 1.5; color: #1f2a44;
          text-align: right;
      }

      .user-name {
          font-weight: 600;
          margin-bottom: 0.3rem;
          color: #274472;
      }
    </style>
    """, unsafe_allow_html=True)

# ------------------------------
# Avatar helpers
# ------------------------------
def load_avatar(filename: str) -> str:
    base_dir = Path(__file__).resolve()
    search_paths = [
        base_dir.parent / "images" / filename,
        base_dir.parents[1] / "images" / filename,
    ]
    for avatar_path in search_paths:
        if avatar_path.exists():
            with avatar_path.open("rb") as img:
                return base64.b64encode(img.read()).decode("utf-8")
    return ""

CLARA_AVATAR_B64 = load_avatar("clara.png")
USER_AVATAR_B64 = load_avatar("reply.png")

# ------------------------------
# Rendering helpers
# ------------------------------
def render_clara(text: str):
    avatar = f'<img src="data:image/png;base64,{CLARA_AVATAR_B64}" alt="Clara">'
    st.markdown(f"""
    <div class="clara-container">
        {avatar}
        <div class="clara-text">
            <div class="clara-name">Clara</div>
            {text}
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_user(text: str):
    avatar = f'<img src="data:image/png;base64,{USER_AVATAR_B64}" alt="User">'
    st.markdown(f"""
    <div class="user-container">
        {avatar}
        <div class="user-text">
            <div class="user-name">You</div>
            {text}
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_clara_stream(text_generator):
    """
    Display Clara's streaming text inside her chat bubble.
    The text_generator yields chunks from stream_response().
    """
    from .ui_components import CLARA_AVATAR_B64  # re-use existing avatar
    avatar_tag = f'<img src="data:image/png;base64,{CLARA_AVATAR_B64}" alt="Clara avatar" />'

    container = st.empty()
    partial_text = ""

    for chunk in text_generator:
        partial_text += chunk
        html = f"""
        <div class="clara-container">
            {avatar_tag}
            <div class="clara-text">
                <div class="clara-name">Clara</div>
                {partial_text}
            </div>
        </div>
        """
        container.markdown(html, unsafe_allow_html=True)
        time.sleep(0.025)  # Adjust typing speed (lower = faster)

    return partial_text

def render_reflection(text: str, container=None):
    """Render Clara’s ethical reflection before her main answer."""
    markdown = container.markdown if container is not None else st.markdown
    markdown(f"""
    <div class="clara-container">
        <div class="clara-text" style="
            background-color: #fffbe6;
            border-left: 4px solid #bfa15a;
            font-style: italic;
            color: #444;
        ">
            🪞 <b>Clara’s Reflection:</b> {text}
        </div>
    </div>
    """, unsafe_allow_html=True)
