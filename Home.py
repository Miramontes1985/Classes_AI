# home.py
import streamlit as st
from datetime import datetime

# ------------------------------
# Page configuration
# ------------------------------
st.set_page_config(
    page_title="AI for Business & Ethics — DCU Teaching App",
    page_icon="💼",
    layout="wide"
)

# ------------------------------
# Header
# ------------------------------
st.title("💼 AI for Business and Ethics — Teaching Platform")
st.markdown("#### Dublin City University · Lero — The Science Foundation Ireland Research Centre for Software")
st.divider()

# ------------------------------
# Overview
# ------------------------------
st.write(
    """
    Welcome to this interactive platform designed to **demonstrate and discuss Artificial Intelligence in business and public-sector contexts**.  
    Developed by Fernando Forattini for teaching and outreach at **Dublin City University (DCU)**, the app illustrates how AI systems are built, trained, and governed — 
    and how issues such as **bias, fairness, and accountability** arise in real organisational environments.
    """
)

st.markdown("### 🧩 Modules Included")
st.write(
    """
    So far, the app includes 4 interactive modules:
    1. **Before Training:** Observe how a generic AI model behaves out of the box.  
    2. **Fine-Tuning in Action:** Watch a model being trained on inclusive, domain-specific data.  
    3. **After Training:** See how behaviour changes after alignment.  
    4. **Identity Profiling (Vision Bias):** Examine bias in AI image classification systems.  
    """
)

# ------------------------------
# Author Section
# ------------------------------
st.markdown("---")
st.subheader("👨‍🏫 Author & Context")
st.markdown(
    f"""
    **Developed by [Fernando Miramontes Forattini](mailto:fernandomforattini@gmail.com)**  
    Post-Doctoral Research Fellow, Marie Skłodowska-Curie Actions  
    [Dublin City University](https://www.dcu.ie/) · [Lero, the SFI Research Centre for Software](https://lero.ie/)  

    **Purpose:** Created for educational use in courses and workshops on *AI, Ethics, and Digital Governance*.  
    This application is open for demonstration, research, and classroom engagement under a non-commercial licence.  
    """
)

# ------------------------------
# Footer
# ------------------------------
st.divider()
st.caption(
    f"© {datetime.now().year} — Dublin City University · Fernando Miramontes Forattini · "
    "For teaching and research purposes only."
)
