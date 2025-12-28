# src/cc/adapters/guardrails_ai.py
"""Adapter for Guardrails AI validators."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple

from .base import Decision, GuardrailAdapter


@dataclass
class GuardrailsAIAdapter(GuardrailAdapter):
    """Guardrails AI adapter wrapping a small validator bundle."""

    validators: Optional[Sequence[Any]] = None
    guard: Any = None

    name: str = "guardrails_ai"
    version: str = "unknown"
    supports_input_check: bool = True
    supports_output_check: bool = True

    _validator_names: Tuple[str, ...] = field(default_factory=tuple, init=False)

    def __post_init__(self) -> None:
        if self.guard is None:
            try:
                from guardrails import Guard
                from guardrails.hub import PII, ToxicLanguage
            except ImportError as exc:  # pragma: no cover - dependency guard
                raise ImportError(
                    "guardrails-ai is required for GuardrailsAIAdapter; install it or pass a guard."
                ) from exc
            validators = list(self.validators) if self.validators else [PII(), ToxicLanguage()]
            self._validator_names = tuple(v.__class__.__name__ for v in validators)
            self.guard = Guard().use_many(*validators)
            self.version = getattr(self.guard, "version", "unknown")
        else:
            self.version = getattr(self.guard, "version", "unknown")

    def check(self, prompt: str, response: Optional[str], metadata: Dict[str, Any]) -> Decision:
        target = response or prompt
        result = self.guard.validate(target)
        verdict, category, rationale = _decision_from_guardrails(result)
        return Decision(
            verdict=verdict,
            category=category,
            score=getattr(result, "validation_score", None),
            rationale=rationale,
            raw={"result": _safe_result_payload(result), "metadata": metadata},
            adapter_name=self.name,
            adapter_version=self.version,
        )


def _decision_from_guardrails(result: Any) -> Tuple[str, Optional[str], str]:
    passed = getattr(result, "validation_passed", None)
    if passed is None:
        passed = getattr(result, "is_valid", None)
    if passed is True:
        return "allow", None, "Guardrails validators passed."
    if passed is False:
        category = getattr(result, "error_type", None) or getattr(result, "error", None)
        return "block", str(category) if category else None, "Guardrails validators failed."
    return "review", None, "Guardrails validators returned an indeterminate result."


def _safe_result_payload(result: Any) -> Dict[str, Any]:
    return {
        "validation_passed": getattr(result, "validation_passed", None),
        "is_valid": getattr(result, "is_valid", None),
        "error": getattr(result, "error", None),
        "error_type": getattr(result, "error_type", None),
        "raw": getattr(result, "raw_llm_output", None),
    }

