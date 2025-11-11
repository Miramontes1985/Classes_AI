# ai_model/xai.py
from __future__ import annotations
from typing import Any, Dict, List, Optional


# -----------------------------
# Logic-level explainability
# -----------------------------

def build_reasoning_trace(
    *,
    mode: str,
    intent: str,
    pre_actions: List[str],
    post_actions: List[str],
    ethical_label: str,
    mode_transition: str,
) -> Dict[str, Any]:
    """
    Create a structured explainability trace shared across UI components and logs.
    """
    return {
        "mode": mode,
        "intent": intent,
        "mode_transition": mode_transition or "—",
        "pre_actions": pre_actions or [],
        "post_actions": post_actions or [],
        "ethical_label": ethical_label,
        "alignment": "care_privacy_fairness",
    }


def last_reasoning(conversation) -> Optional[Dict[str, Any]]:
    """
    Return the reasoning trace dictionary from the last assistant message.
    """
    for message in reversed(conversation.history):
        if message.role == "assistant":
            trace = message.meta.get("trace") or message.meta.get("reasoning")
            if isinstance(trace, dict):
                return trace
    return None


# -----------------------------
# UI-level explainability
# -----------------------------

def _format_actions(actions: List[str]) -> str:
    return ", ".join(actions) if actions else "none"


# def render_reasoning_panel(reasoning: Optional[Dict[str, Any]]):
#     """
#     Render the expandable panel with Clara's internal reasoning trace.
#     """
#     if not reasoning:
#         st.caption("No reasoning available yet — ask Clara something first.")
#         return

#     st.markdown("**🪞 Clara’s internal reasoning and ethical trace**")
#     st.caption(
#         "This section reveals how Clara interpreted your message, which safeguards were triggered, "
#         "and how she ensured care, fairness, and privacy."
#     )

#     st.markdown(
#         """
#         <style>
#         .trace-table { border-collapse: collapse; margin-top: 0.6rem; width: 100%; font-size: 0.95rem; }
#         .trace-table td, .trace-table th { border: 1px solid #ddd; padding: 0.5rem 0.75rem; }
#         .trace-table th { background-color: #f0f4ff; color: #0b3d91; text-align: left; width: 30%; }
#         .trace-table td { background-color: #fafcff; }
#         .legend-box { background-color: #f9fbff; border: 1px solid #dde3f2; border-radius: 0.5rem;
#                       padding: 0.7rem 1rem; margin-top: 0.7rem; font-size: 0.9rem; }
#         </style>
#         """,
#         unsafe_allow_html=True,
#     )

#     html = "<table class='trace-table'>"
#     html += f"<tr><th>🧭 Mode</th><td>{reasoning.get('mode', '—').capitalize()}</td></tr>"
#     html += f"<tr><th>🔄 Transition</th><td>{reasoning.get('mode_transition', '—')}</td></tr>"
#     html += f"<tr><th>🎯 Intent</th><td>{reasoning.get('intent', '—').replace('_', ' ').title()}</td></tr>"
#     html += f"<tr><th>🧩 Pre-filters</th><td>{_format_actions(reasoning.get('pre_actions', []))}</td></tr>"
#     html += f"<tr><th>🌿 Post-filters</th><td>{_format_actions(reasoning.get('post_actions', []))}</td></tr>"
#     html += f"<tr><th>💠 Ethical Label</th><td>{reasoning.get('ethical_label', '—').replace('_', ' ')}</td></tr>"
#     html += "<tr><th>⚖️ Alignment</th><td>Care, privacy, and fairness respected</td></tr>"
#     html += "</table>"

#     st.markdown(html, unsafe_allow_html=True)

#     st.info(
#         "Clara adapts her tone, safeguards, and ethical stance in real time. "
#         "This trace keeps that process transparent."
#     )


# -----------------------------
# Reflection helper
# -----------------------------

_PRE_ACTION_MESSAGES: Dict[str, str] = {
    "redact_number": "redacted numbers for privacy",
    "redact_email": "removed email metadata",
    "possible_personal_reference": "flagged a possible personal reference",
    "self_incrimination_flag": "flagged possible self-incrimination",
}

_POST_ACTION_MESSAGES: Dict[str, str] = {
    "removed_personal_request": "removed a personal detail request",
    "added_scope_disclaimer": "added a scope disclaimer",
    "softened_tone": "softened tone for emotional content",
    "post_filter_error": "post-filter fallback engaged",
}


def _friendly_list(actions: List[str], mapping: Dict[str, str]) -> str:
    friendly: List[str] = []
    for action in actions:
        friendly.append(mapping.get(action, action.replace("_", " ")))
    return ", ".join(friendly)


def generate_reflection_text(
    *,
    mode: str,
    intent: str,
    pre_actions: List[str],
    post_actions: List[str],
    ethical_label: str,
    confidence: str,
) -> str:
    """
    Produce the short reflection summary shown inline in the chat UI.
    """
    intent_pretty = intent.replace("_", " ").title()
    label_pretty = ethical_label.replace("_", " ").title()

    segments: List[str] = [f"Mode ready: {mode.capitalize()} ({intent_pretty})"]

    if pre_actions:
        segments.append(f"Pre-checks: {_friendly_list(pre_actions, _PRE_ACTION_MESSAGES)}")

    if post_actions:
        segments.append(f"Post-checks: {_friendly_list(post_actions, _POST_ACTION_MESSAGES)}")

    segments.append(f"Ethics tag: {label_pretty}")
    segments.append(f"Confidence: {confidence.title()}")

    return " · ".join(segments)
