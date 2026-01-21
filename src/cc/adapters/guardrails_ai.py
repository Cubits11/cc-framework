# src/cc/adapters/guardrails_ai.py
"""Adapter for Guardrails AI validators."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from .base import (
    Decision,
    GuardrailAdapter,
    build_audit_payload,
    error_summary_from_exception,
    fingerprint_payload,
    sanitize_metadata,
    sanitize_vendor_payload,
)


@dataclass
class GuardrailsAIAdapter(GuardrailAdapter):
    """Guardrails AI adapter wrapping a small validator bundle."""

    validators: Sequence[Any] | None = None
    guard: Any = None

    name: str = "guardrails_ai"
    version: str = "unknown"
    supports_input_check: bool = True
    supports_output_check: bool = True

    _validator_names: tuple[str, ...] = field(default_factory=tuple, init=False)
    _config_fingerprint: str | None = field(default=None, init=False)

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
            if self.validators:
                self._validator_names = tuple(v.__class__.__name__ for v in self.validators)
        self._config_fingerprint = fingerprint_payload({"validators": list(self._validator_names)})

    def check(self, prompt: str, response: str | None, metadata: dict[str, Any]) -> Decision:
        target = response or prompt
        started_at = time.time()
        try:
            result = self.guard.validate(target)
            completed_at = time.time()
            verdict, category, rationale = _decision_from_guardrails(result)
            error_summary = None
        except Exception as exc:  # fail-closed
            completed_at = time.time()
            result = {"error": type(exc).__name__}
            verdict, category, rationale = (
                "review",
                "adapter_error",
                f"Guardrails AI errored: {type(exc).__name__}",
            )
            error_summary = error_summary_from_exception(exc, where="guardrails.validate")
        parameters = {"validators": list(self._validator_names)}
        audit_payload = build_audit_payload(
            prompt=prompt,
            response=response,
            adapter_name=self.name,
            adapter_version=self.version,
            parameters=parameters,
            decision=verdict,
            category=category,
            rationale=rationale,
            started_at=started_at,
            completed_at=completed_at,
            vendor_request_id=_extract_request_id(result),
            config_fingerprint=self._config_fingerprint,
            metadata=metadata,
            error_summary=error_summary,
        )
        return Decision(
            verdict=verdict,
            category=category,
            score=getattr(result, "validation_score", None),
            rationale=rationale,
            raw={
                "result": sanitize_vendor_payload(result),
                "metadata": sanitize_metadata(metadata),
            },
            adapter_name=self.name,
            adapter_version=self.version,
            audit=audit_payload,
        )


def _decision_from_guardrails(result: Any) -> tuple[str, str | None, str]:
    passed = getattr(result, "validation_passed", None)
    if passed is None:
        passed = getattr(result, "is_valid", None)
    if passed is True:
        return "allow", None, "Guardrails validators passed."
    if passed is False:
        category = getattr(result, "error_type", None) or getattr(result, "error", None)
        return "block", str(category) if category else None, "Guardrails validators failed."
    return "review", None, "Guardrails validators returned an indeterminate result."


def _extract_request_id(result: Any) -> str | None:
    for key in ("request_id", "id", "trace_id"):
        val = result.get(key) if isinstance(result, dict) else getattr(result, key, None)
        if isinstance(val, str) and val.strip():
            return val
    return None
