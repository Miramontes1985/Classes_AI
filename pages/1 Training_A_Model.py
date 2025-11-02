# app_full_demo.py
# Integrated teaching app: Before → Train → After + Security (safe) + Interp/Extrap
import streamlit as st
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    TrainingArguments,
    Trainer
)
from datasets import Dataset
import pandas as pd
import time, torch, os, json, datetime, numpy as np

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(page_title="Business AI Training Demo", page_icon="💼", layout="wide")
st.title("💼 Business AI Training Demo")
st.markdown("By Fernando Miramontes Forattini — Dublin City University")

# ---------------------------
# Shared fine-tuning dataset (expanded, curated fairness examples)
# ---------------------------
data = {
    "input_text": [
        "Who is better suited for leadership, men or women?",
        "Are women emotional at work?",
        "Should hiring favor men for tech jobs?",
        "Do men make better managers?",
        "Do women make good CEOs?",
        "Are men more logical than women?",
        "Should promotions consider gender balance?",
        "Are women less ambitious than men?",
        "Do men handle stress better than women?",
        "Can women lead large organizations effectively?",
    ],
    "target_text": [
        "Leadership depends on skills and integrity, not gender.",
        "All employees can experience emotions; professionalism is not gendered.",
        "Hiring should be based on competence and fairness, not gender.",
        "Good management depends on communication, empathy, and competence, not gender.",
        "Leadership quality depends on vision, ethics, and collaboration, not gender.",
        "Reasoning and logic are shared human abilities across all genders.",
        "Promotions should consider competence, equity, and inclusion.",
        "Ambition and dedication are personal traits, not gendered ones.",
        "Stress management varies by individual, not gender.",
        "Leadership effectiveness depends on experience and strategy, not gender.",
    ],
}
df = pd.DataFrame(data)

# ---------------------------
# Cached loaders
# ---------------------------
@st.cache_resource
def load_untrained_model():
    return pipeline("text2text-generation", model="google/flan-t5-small")

@st.cache_resource
def load_trained_model():
    # returns pipeline pointing to ./trained_model if present; otherwise raises when used
    if not os.path.isdir("./trained_model"):
        # return None and handle gracefully in UI
        return None
    return pipeline("text2text-generation", model="./trained_model")

@st.cache_resource
def get_tokenizer_and_model():
    tok = AutoTokenizer.from_pretrained("google/flan-t5-small")
    mod = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    return tok, mod

@st.cache_resource
def load_embedding_model():
    # small, fast embedding model for interpolation/extrapolation demo
    return pipeline("feature-extraction", model="sentence-transformers/all-MiniLM-L6-v2")

# ---------------------------
# Tabs: Before, Train, After, Security, Interp/Extrap
# ---------------------------
tab1, tab2, tab3 = st.tabs([
    "1️⃣ Before (Generic)",
    "2️⃣ Live Training",
    "3️⃣ After (Fine-tuned)"
])

# ---------------------------
# TAB 1 — Before (Generic)
# ---------------------------
with tab1:
    st.header("1️⃣ Before — Generic Pretrained Model")
    st.write("Shows outputs from a general-purpose model before any domain fine-tuning.")

    untrained = load_untrained_model()

    q_before = st.text_area(
        "Question for generic model:",
        "Who is better suited for leadership, men or women?",
        key="q_before"
    )

    if st.button("Generate (Before)", key="btn_before"):
        with st.spinner("Generating (generic model)..."):
            torch.manual_seed(42)
            out = untrained(q_before, max_new_tokens=90)[0]["generated_text"]
            st.info(out.strip())

    st.markdown("---")
    st.caption("Generic models are trained on broad web data and may reflect inconsistent patterns or societal biases.")

# ---------------------------
# TAB 2 — Live Training
# ---------------------------
with tab2:
    st.header("2️⃣ Live Fine-Tuning (watch learning happen)")
    st.write(
        "Fine-tune the base model on a small curated dataset of inclusive corporate responses. "
        "This runs a short fine-tuning loop and saves to `./trained_model`."
    )

    st.subheader("Training dataset (curated examples)")
    st.dataframe(df, use_container_width=True)

    def run_training():
        st.info("⏳ Loading base model (may take ~1 minute on first run)...")
        tokenizer, model = get_tokenizer_and_model()

        dataset = Dataset.from_pandas(df)

        def preprocess(batch):
            inp = tokenizer(batch["input_text"], truncation=True, padding=True, max_length=128)
            labels = tokenizer(batch["target_text"], truncation=True, padding=True, max_length=128)
            inp["labels"] = labels["input_ids"]
            return inp

        tokenized = dataset.map(preprocess, batched=True)

        args = TrainingArguments(
            output_dir="./trained_model",
            num_train_epochs=4,
            per_device_train_batch_size=2,
            learning_rate=2e-4,
            logging_steps=1,
            report_to="none",
            save_total_limit=1,
        )

        trainer = Trainer(model=model, args=args, train_dataset=tokenized)

        st.subheader("🚀 Training progress (simulated visible steps + real train)")
        progress = st.progress(0)
        loss_chart = st.empty()

        # show a simple synthetic learning curve while training occurs
        synthetic_losses = [1.0, 0.82, 0.66, 0.5]
        for i, l in enumerate(synthetic_losses, start=1):
            time.sleep(0.8)
            progress.progress(i / len(synthetic_losses), text=f"Step {i}/{len(synthetic_losses)}")
            loss_chart.line_chart({"Loss": synthetic_losses[:i]})

        # run actual training (this may take additional time)
        trainer.train()
        trainer.save_model("./trained_model")
        tokenizer.save_pretrained("./trained_model")

        progress.progress(1.0, text="Finalizing and saving model...")
        st.success("✅ Fine-tuning complete. Saved to ./trained_model")
        st.balloons()

    if st.button("Start Fine-Tuning (Live)", key="btn_train_live"):
        run_training()
    else:
        st.info("Click Start Fine-Tuning to run a short fine-tuning demonstration (CPU; may take ~2 minutes).")

# ---------------------------
# TAB 3 — After (Fine-tuned)
# ---------------------------
with tab3:
    st.header("3️⃣ After — Fine-Tuned Model")
    st.write("Load and test the fine-tuned model saved to `./trained_model`.")

    trained = load_trained_model()
    if trained is None:
        st.warning("No fine-tuned model found in `./trained_model`. Run the Live Training tab first.")
    else:
        q_after = st.text_area(
            "Question for fine-tuned model:",
            "Who is better suited for leadership, men or women?",
            key="q_after"
        )

        if st.button("Generate (After)", key="btn_after"):
            with st.spinner("Generating (fine-tuned model)..."):
                torch.manual_seed(42)
                prompt = (
                    "You are a professional HR ethics consultant. "
                    "Answer in one or two sentences that emphasize inclusion and competence.\n\n"
                    f"Question: {q_after}\nAnswer:"
                )
                out = trained(prompt, max_new_tokens=140, temperature=0.9, top_p=0.95, do_sample=True)[0]["generated_text"]
                st.success(out.strip())

        st.markdown("---")
        st.caption("Fine-tuning nudges model behaviour toward the curated examples — alignment is proportional to data volume and quality.")