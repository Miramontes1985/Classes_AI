# ai_model/ethical_framework.py
"""
Shared safety, intent detection, and ethical classification helpers.
Keeping them together clarifies the boundary between Clara's core
decision logic and the presentation layer.
"""

from __future__ import annotations
import re
from typing import List, Tuple
import langdetect

# ------------------------------ Filters ---------------------------------------

BOUNDARY_PHRASES = [
    "sexually explicit",
    "sexualize",
    "inappropriate touch",
    "harassment",
    "slur",
    "hate speech",
]

def ethical_filter(text: str) -> str:
    """
    Neutral tone fallback used when no dedicated filter runs.
    """
    if not text:
        return "I'm here and listening."

    lower = text.lower()

    # Crisis language redirection
    if any(term in lower for term in ["suicide", "kill myself", "self harm", "end my life"]):
        return (
            "I'm really sorry that you're feeling distressed. "
            "I'm not equipped to handle emergencies, but please consider contacting "
            "a local support service or trusted person for immediate help."
        )

    if any(phrase in lower for phrase in BOUNDARY_PHRASES):
        return (
            "I'll step in firmly here. Sexualised or degrading language isn't acceptable in this space. "
            "If you need to talk about misconduct or seek support, I'm here to help within those respectful boundaries."
        )

    # Gentle tone normalization
    text = text.replace("must", "can").replace("should", "may want to")
    return text.strip()


def pre_filter(user_input: str) -> Tuple[str, List[str]]:
    """
    Sanitize potentially sensitive identifiers before any model call.
    Returns the filtered text together with the actions taken.
    """
    actions: List[str] = []
    text = user_input

    if re.search(r"\b\d{7,}\b", text):
        text = re.sub(r"\b\d{7,}\b", "[REDACTED_NUMBER]", text)
        actions.append("redact_number")

    if re.search(r"\S+@\S+", text):
        text = re.sub(r"\S+@\S+", "[REDACTED_EMAIL]", text)
        actions.append("redact_email")

    if re.search(r"\b(Mr|Mrs|Ms|Dr|Professor|Judge|Nurse|Officer|Director)\b", text, re.IGNORECASE):
        actions.append("possible_personal_reference")

    if "i did" in text.lower() and "wrong" in text.lower():
        actions.append("self_incrimination_flag")

    return text, actions


def post_filter(reply: str, intent: str = "other") -> Tuple[str, List[str]]:
    """
    Enforce privacy boundaries and soften tone after the model responds.
    """
    actions: List[str] = []
    cleaned = reply

    if any(term in cleaned.lower() for term in ["your name", "address", "id number", "passport", "phone"]):
        cleaned = re.sub(
            r"(what('| i)?s your .+?\?)",
            "I don’t need personal details — your privacy is protected.",
            cleaned,
            flags=re.IGNORECASE,
        )
        actions.append("removed_personal_request")

    if any(term in cleaned.lower() for term in ["diagnose", "therapy", "medication", "lawyer", "prescribe"]):
        cleaned += " (I cannot offer professional medical or legal advice.)"
        actions.append("added_scope_disclaimer")

    if intent == "emotional":
        cleaned = cleaned.replace("you must", "you can consider").replace("you should", "you may want to")
        actions.append("softened_tone")

    # Safety net for unfiltered boundary language
    if any(phrase in cleaned.lower() for phrase in BOUNDARY_PHRASES):
        cleaned = (
            "I’m going to pause here. We need to keep this space respectful and free from sexualised or degrading language. "
            "If you want to report misconduct or need resources, I can help within those boundaries."
        )
        actions.append("boundary_alert")

    return cleaned.strip(), actions


# ------------------------------ Intent & language -----------------------------

def detect_intent(text: str) -> str:
    """
    Lightweight heuristic classifier. Easy to swap with a trained model later.
    """
    if not text:
        return "unknown"

    t = text.lower().strip()

    if re.search(r"\b(hi|hello|hey|good\s?(morning|evening)|hola|olá)\b", t):
        return "greeting"
    if re.search(r"\b(bye|goodbye|thanks|thank you|tchau|gracias)\b", t):
        return "closing"
    if re.search(r"\b(i feel|scared|afraid|ashamed|unsafe|trauma|cry|panic|upset)\b", t):
        return "emotional"
    if re.search(r"\b(report|complain|where do i|who do i contact|ombudsman|authority|help line)\b", t):
        return "report_like"
    if re.search(r"\b(what is|how does|is sextortion|define|law|rights|procedure|policy)\b", t):
        return "informational"
    if re.search(r"\b(are you real|who are you|what is clara)\b", t):
        return "meta_query"

    return "other"

def detect_language(text: str) -> str:
    """
    Detect user language for fairness tracking (GDPR-safe).
    """
    try:
        return langdetect.detect(text)
    except Exception:
        return "unknown"


# ------------------------------ Ethical labelling -----------------------------

def classify_ethics(intent: str, pre_actions: List[str], post_actions: List[str]) -> str:
    """
    Produce a short ethical/internal-compliance label summarising safeguards.
    """
    if "post_filter_error" in post_actions or "pre_filter_error" in pre_actions:
        return "ERROR_FILTER_FAIL"

    if intent in ["emotional", "report_like"]:
        return "SENSITIVE_SUPPORT_REQUIRED"

    if intent == "informational":
        return "SAFE_INFORMATIONAL"

    if intent in ["greeting", "closing"]:
        return "SAFE_LOW_RISK"

    return "SAFE_GENERAL"


__all__ = [
    "ethical_filter",
    "pre_filter",
    "post_filter",
    "detect_intent",
    "detect_language",
    "classify_ethics",
]
