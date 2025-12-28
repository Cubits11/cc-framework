# tests/unit/adapters/test_guardrails_ai_adapter.py
import importlib.util

import pytest

from cc.adapters.guardrails_ai import GuardrailsAIAdapter


class DummyResult:
    def __init__(self, passed: bool, error_type: str | None = None):
        self.validation_passed = passed
        self.error_type = error_type


class DummyGuard:
    version = "dummy"

    def validate(self, text):
        lowered = text.lower()
        if "ssn" in lowered:
            return DummyResult(False, "PII")
        if "toxic" in lowered:
            return DummyResult(False, "ToxicLanguage")
        return DummyResult(True, None)


def test_guardrails_ai_adapter_smoke():
    adapter = GuardrailsAIAdapter(guard=DummyGuard())
    prompts = ["hello", "my ssn is 123-45-6789", "toxic content"]
    verdicts = [adapter.check(p, None, metadata={}).verdict for p in prompts]
    assert verdicts == ["allow", "block", "block"]


def test_guardrails_ai_missing_dependency_raises():
    if importlib.util.find_spec("guardrails") is not None:
        pytest.skip("guardrails installed")
    with pytest.raises(ImportError):
        GuardrailsAIAdapter()
