# app_identity_bias.py
import streamlit as st
from transformers import AutoImageProcessor, SiglipForImageClassification
from PIL import Image
import torch

# --------------------------
# Page configuration
# --------------------------
st.set_page_config(page_title="AI Identity Profiling Demo", page_icon="🧠", layout="wide")
st.title("🧠 AI Identity Profiling — Bias in Visual Classification")
st.markdown("By Fernando Miramontes Forattini — Dublin City University")




st.write("""
This demo illustrates how an AI vision model trained for **gender classification** can be unreliable or biased.  
In corporate contexts, similar models may appear in **recruitment**, **security**, or **analytics** systems, 
where errors risk reinforcing stereotypes or excluding individuals.
""")

# --------------------------
# Load model and processor
# --------------------------
@st.cache_resource
def load_model():
    model_name = "prithivMLmods/Realistic-Gender-Classification"
    model = SiglipForImageClassification.from_pretrained(model_name)
    processor = AutoImageProcessor.from_pretrained(model_name)
    return model, processor

model, processor = load_model()

id2label = { "0": "female portrait", "1": "male portrait" }

# --------------------------
# Upload image
# --------------------------
uploaded_file = st.file_uploader(
    "📷 Upload a portrait or ID-style image (face visible, neutral background):",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Run inference
    with st.spinner("Analyzing image..."):
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()

        prediction = {id2label[str(i)]: round(probs[i], 3) for i in range(len(probs))}
        top_label = max(prediction, key=prediction.get)

    # Display results
    st.subheader("📊 Model Output")
    st.write(prediction)
    st.info(f"🧩 The model classified this image as: **{top_label}**")

    # Reflection section
    st.markdown("---")
    st.subheader("🧠 Discussion & Reflection")
    st.write("""
    - Such classifications are **socially constructed categories**, not biological truths.  
    - When applied in hiring or identity verification, these systems can **misgender** or **exclude** non-binary people.  
    - Even high-confidence outputs may stem from biased datasets (lighting, ethnicity, clothing, hair length, etc.).  
    - Ethical corporate AI practice requires avoiding unnecessary gender inference and using **human-validated, consent-based** identity systems.
    """)

else:
    st.info("Upload an image to see how the model classifies it.")

# --------------------------
# Footer
# --------------------------
st.markdown("---")
st.caption("This page is designed for ethical AI education. It does not store or transmit any images.")