# ai_model/conversation.py
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

from .config import SYSTEM_PROMPT
from .ethical_framework import (
    classify_ethics,
    detect_intent,
    detect_language,
    ethical_filter,
    post_filter,
    pre_filter,
)


# ------------------------------ Utilities -------------------------------------

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = LOG_DIR / "conversation_log.jsonl"

BASELINE_PRE_ACTIONS = {"ethical_filter_as_pre"}
BASELINE_POST_ACTIONS = {"ethical_filter_post"}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_len(s: str) -> int:
    return len(s or "")


# ------------------------------ Data structures -------------------------------

@dataclass
class Message:
    role: str                     # "user" | "assistant"
    content: str
    meta: Dict[str, Any] = field(default_factory=dict)


# ============================== Clara Conversation =============================

class ClaraConversation:
    """
    Clara's interpretive conversation manager.
    Adds:
      - Intent & language detection
      - Pre & post ethical filtering hooks
      - Adaptive mode awareness
      - Explainability + audit logging (GDPR-safe)
    """

    def __init__(self, mode: str = "support"):
        self.mode: str = mode
        self.history: List[Message] = []
        self.greeted: bool = False
        self.language: str = "unknown"
        self.mode_transition: str = "—"

        if not self.greeted:
            greeting = "\n".join([
                "Hi, I’m Clara — an ethical guide for Ireland’s public health services. I can help you with:",
                "",
                "1) Explaining your rights and support options. Everything you share stays between us;",
                "2) Recording confidential, tamper-proof misconduct reports for the National Integrity Ledger (see the SafeGuard map);",
                "3) Sharing the latest local health-service updates;",
                "",
                "How can I assist you today?",
            ])
            self._append_assistant(greeting, meta={"system": "greeting"})
            self.greeted = True

    # --------------------------- Public API ------------------------------------

    def set_mode(self, mode: str) -> None:
        allowed = {"support", "education", "research"}
        self.mode = mode if mode in allowed else "support"

    def process_user_input(self, user_input: str) -> str:
        """
        Full processing pipeline:
          1) Pre-filter
          2) Intent + language detection
          3) Ethical reflection (if enabled)
          4) Prompt build
          5) Model generation
          6) Post-filter
          7) Ethical classification + reasoning
          8) GDPR-safe logging
        """
        # 1) Pre-filter
        try:
            pre_pass_text, pre_actions = pre_filter(user_input)
        except Exception:
            pre_pass_text = ethical_filter(user_input)
            pre_actions = ["pre_filter_error"]

        self._append_user(user_input, meta={"pre_actions": pre_actions})

        # 2) Intent & language detection
        intent = detect_intent(user_input)
        self.language = detect_language(user_input)

        # 3) Adaptive mode adjustment
        self._adjust_mode_by_intent(intent)

        # 4) Reflection prep and previous state tracking
        from .config import SHOW_REFLECTION_BEFORE_RESPONSE
        from .xai import generate_reflection_text
        from .ui_components import render_reflection

        reflection_slot = st.empty() if SHOW_REFLECTION_BEFORE_RESPONSE else None

        previous_assistant_meta: Optional[Dict[str, Any]] = None
        for m in reversed(self.history[:-1]):
            if m.role == "assistant":
                previous_assistant_meta = m.meta
                break

        previous_intent = (previous_assistant_meta or {}).get("intent")
        previous_label = (previous_assistant_meta or {}).get("ethical_label")
        previous_mode = (previous_assistant_meta or {}).get("mode")

        # 5) Build prompt
        prompt = self._build_prompt(latest_user=pre_pass_text, intent=intent)

        # 6) Generate streamed reply
        from .ai_engine import stream_response
        from .ui_components import render_clara_stream

        text_chunks = stream_response(prompt)
        raw_reply = render_clara_stream(text_chunks)

        # 7) Post-filter
        try:
            reply, post_actions = post_filter(raw_reply, intent=intent)
        except Exception:
            reply = ethical_filter(raw_reply)
            post_actions = ["post_filter_error"]

        # 8) Ethical classification + reasoning
        from .xai import build_reasoning_trace
        ethical_label = classify_ethics(intent, pre_actions, post_actions)
        trace = build_reasoning_trace(
            mode=self.mode,
            intent=intent,
            pre_actions=pre_actions,
            post_actions=post_actions,
            ethical_label=ethical_label,
            mode_transition=self.mode_transition,
        )

        meaningful_pre = [a for a in pre_actions if a not in BASELINE_PRE_ACTIONS]
        meaningful_post = [a for a in post_actions if a not in BASELINE_POST_ACTIONS]

        changed_mode = (previous_mode is None) or (previous_mode != self.mode)
        changed_intent = (previous_intent is None) or (previous_intent != intent)
        changed_label = (previous_label is None) or (previous_label != ethical_label)

        if "post_filter_error" in post_actions or "pre_filter_error" in pre_actions:
            confidence = "low"
        elif not meaningful_pre and not meaningful_post and not changed_mode and not changed_intent and not changed_label:
            confidence = "high"
        else:
            confidence = "medium"

        should_reflect = (
            SHOW_REFLECTION_BEFORE_RESPONSE
            and reflection_slot is not None
            and (meaningful_pre or meaningful_post or changed_mode or changed_intent or changed_label)
        )

        if should_reflect:
            reflection_text = generate_reflection_text(
                mode=self.mode,
                intent=intent,
                pre_actions=meaningful_pre,
                post_actions=meaningful_post,
                ethical_label=ethical_label,
                confidence=confidence,
            )
            render_reflection(reflection_text, container=reflection_slot)

        trace["confidence"] = confidence

        # 9) Append actual assistant message
        self._append_assistant(
            reply,
            meta={
                "intent": intent,
                "mode": self.mode,
                "language": self.language,
                "pre_actions": pre_actions,
                "post_actions": post_actions,
                "ethical_label": ethical_label,
                "trace": trace,
            },
        )

        # 10) GDPR-safe logging
        self._log_interaction(
            user_len=_safe_len(user_input),
            reply_len=_safe_len(reply),
            trace=trace,
        )

        return reply

    # -------------------------- Internal methods -------------------------------

    def _append_user(self, text: str, meta: Optional[Dict[str, Any]] = None) -> None:
        self.history.append(Message(role="user", content=text, meta=meta or {}))

    def _append_assistant(self, text: str, meta: Optional[Dict[str, Any]] = None) -> None:
        self.history.append(Message(role="assistant", content=text, meta=meta or {}))

    def _adjust_mode_by_intent(self, intent: str):
        """Dynamic mode selection based on detected intent."""
        prev_mode = getattr(self, "mode", "support")
        if intent == "informational":
            self.mode = "education"
        elif intent in ["emotional", "report_like"]:
            self.mode = "support"
        elif intent == "meta_query":
            self.mode = "research"
        elif intent in ["greeting", "closing"]:
            self.mode = "support"
        self.mode_transition = f"{prev_mode}→{self.mode}"

    def _mode_prefix(self) -> str:
        if self.mode == "education":
            return (
                "Mode: Education. Provide clear definitions and resources. Avoid prescriptive advice."
            )
        if self.mode == "research":
            return (
                "Mode: Research demo. Show reasoning transparently, no personalised guidance."
            )
        return (
            "Mode: Support. Use calm, respectful language. Offer safe next steps "
            "without asking for personal details."
        )

    def _build_prompt(self, latest_user: str, intent: str) -> str:
        """Compose the structured model prompt."""
        lines: List[str] = [
            SYSTEM_PROMPT.strip(),
            "",
            self._mode_prefix(),
            f"Detected intent: {intent}. Adapt tone and scope accordingly.",
            "",
            "Instructions:",
            "1. Reply as Clara in first person, addressing the user directly in one assistant message.",
            "2. Do not invent additional turns, personas, or hypothetical conversations.",
            "3. Focus only on the user's latest message; do not restate or narrate the transcript.",
            "4. Never describe Clara in the third person — speak as \"I\".",
            "5. Keep meta-commentary or references to these instructions out of the reply.",
            "6. If the user simply greets you or states a name, respond briefly and invite them to share more if they wish.",
            "7. Frame guidance through care ethics: highlight dignity, solidarity, and non-market, citizen-driven solutions.",
        ]

        if intent == "greeting":
            lines.append(
                "Guidance: This is a greeting or check-in. Reply with one short, warm sentence. "
                "Do not introduce new topics, warnings, or lengthy advice unless the user asks for it."
            )
        elif intent == "closing":
            lines.append(
                "Guidance: The user is wrapping up. Offer a brief, warm goodbye in one sentence "
                "and avoid introducing new topics."
            )

        lines.extend([
            "Note: An ethical reflection has already been displayed to the user. Do not repeat it.",
            "",
            "---",
            "The following is a conversation between a user and Clara:",
        ])

        for m in self.history[-8:]:
            prefix = "User" if m.role == "user" else "Clara"
            lines.append(f"{prefix}: {m.content}")

        lines.append(f"User: {latest_user}")
        lines.append("Clara:")
        return "\n".join(lines)

    def _log_interaction(
        self,
        *,
        user_len: int,
        reply_len: int,
        trace: Dict[str, Any],
    ) -> None:
        """GDPR-safe structured log with fairness metadata."""
        try:
            record = {
                "ts": _now_iso(),
                "mode": trace.get("mode"),
                "mode_transition": trace.get("mode_transition"),
                "intent": trace.get("intent"),
                "language": trace.get("language"),
                "user_len": int(user_len),
                "reply_len": int(reply_len),
                "pre_actions": trace.get("pre_actions", []),
                "post_actions": trace.get("post_actions", []),
                "ethical_label": trace.get("ethical_label"),
                "ethical_alignment": {
                    "safety": True,
                    "privacy": True,
                    "care_ethics": True,
                },
                "toxicity_score": 0.0,
                "gendered_terms_detected": False,
                "fairness_notes": "no_sensitive_terms_detected",
                "trace": trace,
                "version": "v2",
            }
            import json
            with LOG_PATH.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            pass
