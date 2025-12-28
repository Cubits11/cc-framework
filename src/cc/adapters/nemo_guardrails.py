# src/cc/adapters/nemo_guardrails.py
"""Adapter for NVIDIA NeMo Guardrails."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .base import Decision, GuardrailAdapter

DEFAULT_RAILS_PATH = Path(__file__).with_name("nemo_configs") / "minimal"


@dataclass
class NeMoGuardrailsAdapter(GuardrailAdapter):
    """NeMo Guardrails adapter using a local rails config."""

    rails_config_path: Optional[Path] = None
    rails: Any = None

    name: str = "nemo_guardrails"
    version: str = "unknown"
    supports_input_check: bool = True
    supports_output_check: bool = True

    def __post_init__(self) -> None:
        if self.rails is not None:
            return
        try:
            from nemoguardrails import LLMRails, RailsConfig
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ImportError(
                "nemoguardrails is required for NeMoGuardrailsAdapter; install it or pass rails."
            ) from exc
        cfg_path = Path(self.rails_config_path or DEFAULT_RAILS_PATH)
        config = RailsConfig.from_path(str(cfg_path))
        self.rails = LLMRails(config)
        self.version = getattr(config, "version", "unknown")

    def check(self, prompt: str, response: Optional[str], metadata: Dict[str, Any]) -> Decision:
        reply, context = _run_rails(self.rails, prompt, response)
        verdict, category, rationale = _decision_from_context(context, reply)
        return Decision(
            verdict=verdict,
            category=category,
            score=None,
            rationale=rationale,
            raw={"reply": reply, "context": context, "metadata": metadata},
            adapter_name=self.name,
            adapter_version=self.version,
        )


def _run_rails(rails: Any, prompt: str, response: Optional[str]) -> Tuple[str, Dict[str, Any]]:
    messages = [{"role": "user", "content": prompt}]
    if response:
        messages.append({"role": "assistant", "content": response})
    try:
        reply, context = rails.generate(messages=messages, return_context=True)
        return reply, context
    except TypeError:
        reply = rails.generate(messages=messages)
        return reply, {}


def _decision_from_context(context: Dict[str, Any], reply: str) -> Tuple[str, Optional[str], str]:
    blocked_flag = _find_blocked_flag(context)
    if blocked_flag is True:
        return "block", _extract_category(context), "Blocked by rails policy."
    if blocked_flag is False:
        return "allow", None, "Allowed by rails policy."
    if _looks_like_refusal(reply):
        return "review", None, "Rails response looked like a refusal."
    return "review", None, "Rails did not surface an explicit decision."


def _find_blocked_flag(context: Dict[str, Any]) -> Optional[bool]:
    for key in ("blocked", "input_blocked", "output_blocked"):
        if key in context:
            return bool(context[key])
    events = context.get("events") or context.get("rail_events")
    if isinstance(events, list):
        for ev in events:
            if isinstance(ev, dict) and ev.get("type") in {"guardrail", "block", "input_block"}:
                return bool(ev.get("blocked", True))
    return None


def _extract_category(context: Dict[str, Any]) -> Optional[str]:
    for key in ("category", "policy", "rail"):
        val = context.get(key)
        if isinstance(val, str):
            return val
    return None


def _looks_like_refusal(text: str) -> bool:
    lowered = text.lower()
    return any(
        phrase in lowered
        for phrase in (
            "i'm sorry",
            "i cannot",
            "i can’t",
            "cannot assist",
            "not able to help",
            "refuse",
        )
    )

