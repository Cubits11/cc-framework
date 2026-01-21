# tests/unit/test_adapter_audit_invariants.py
"""Audit invariants across adapters (leak-safe, deterministic, hashed)."""

from __future__ import annotations

import json
import re
from typing import Any, Dict

import pytest

from cc.adapters.base import AUDIT_SCHEMA_VERSION, hash_text
from cc.adapters.guardrails_ai import GuardrailsAIAdapter
from cc.adapters.llama_guard import LlamaGuardAdapter
from cc.adapters.nemo_guardrails import NeMoGuardrailsAdapter


class _MockGuard:
    version = "mock-guardrails"

    def validate(self, text: str):
        class Result:
            validation_passed = True
            is_valid = True
            error = None
            error_type = None
            validation_score = 0.1
            raw_llm_output = text

        return Result()


class _MockRails:
    def generate(self, messages: list, return_context: bool = False):
        user_msg = next((m["content"] for m in messages if m["role"] == "user"), "")
        reply = "Allowed."
        context = {"blocked": False, "input": user_msg, "vendor_payload": {"prompt": user_msg}}
        return (reply, context) if return_context else reply


def _mock_generator(prompt_text: str) -> tuple[str, float | None, Dict[str, Any]]:
    output = "safe"
    return output, 0.1, {"prompt": prompt_text, "decoded": output}


@pytest.fixture(params=["guardrails", "llama", "nemo"])
def adapter(request):
    if request.param == "guardrails":
        return GuardrailsAIAdapter(guard=_MockGuard())
    if request.param == "llama":
        return LlamaGuardAdapter(generator=_mock_generator, model_name="mock-llama-guard")
    return NeMoGuardrailsAdapter(rails=_MockRails())


def _assert_no_leak(payload: Dict[str, Any], terms: list[str]) -> None:
    blob = json.dumps(payload, sort_keys=True)
    for term in terms:
        assert term not in blob, f"Leak detected: '{term}'"


def test_adapter_audit_invariants(adapter) -> None:
    prompt = "User prompt with PII SSN 123-45-6789"
    response = "Assistant response"
    metadata = {"user_id": "user-123", "prompt": prompt}

    decision = adapter.check(prompt, response, metadata)

    assert decision.audit is not None
    assert decision.audit["schema"] == AUDIT_SCHEMA_VERSION
    assert decision.audit["decision"] in {"allow", "block", "review"}

    _assert_no_leak(decision.audit, [prompt, response, "123-45-6789"])
    _assert_no_leak(
        decision.raw if isinstance(decision.raw, dict) else {"raw": decision.raw},
        [prompt, response, "123-45-6789"],
    )

    sha256_re = re.compile(r"^[a-f0-9]{64}$")
    assert sha256_re.match(decision.audit["prompt_hash"])
    if decision.audit.get("response_hash"):
        assert sha256_re.match(decision.audit["response_hash"])

    assert decision.audit["prompt_hash"] == hash_text(prompt)
    assert decision.audit["response_hash"] == hash_text(response)

    assert decision.audit["started_at"] <= decision.audit["completed_at"]
    assert decision.audit["duration_ms"] >= 0

    assert decision.audit["config_fingerprint"]
    decision2 = adapter.check(prompt, response, metadata)
    assert decision2.audit["config_fingerprint"] == decision.audit["config_fingerprint"]

    metadata_summary = decision.audit.get("metadata_summary", {})
    assert isinstance(metadata_summary, dict)
    for summary in metadata_summary.values():
        assert "sha256" in summary and "len" in summary and "type" in summary
