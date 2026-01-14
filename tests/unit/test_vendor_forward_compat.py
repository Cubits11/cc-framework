# tests/unit/test_vendor_forward_compat.py
"""Forward-compatibility: new vendor fields must not leak prompt data."""
from __future__ import annotations

import json

import pytest

from cc.adapters.guardrails_ai import GuardrailsAIAdapter
from cc.adapters.llama_guard import LlamaGuardAdapter
from cc.adapters.nemo_guardrails import NeMoGuardrailsAdapter


class _GuardWithNewFields:
    version = "mock-guardrails"

    def validate(self, text: str):
        class Result:
            validation_passed = True
            is_valid = True
            error = None
            error_type = None
            new_prompt_field = text
            nested = {"prompt": text, "content": text}

        return Result()


class _RailsWithNewFields:
    def generate(self, messages: list, return_context: bool = False):
        user_msg = next((m["content"] for m in messages if m["role"] == "user"), "")
        reply = "Allowed."
        context = {"new_prompt_field": user_msg, "nested": {"content": user_msg}}
        return (reply, context) if return_context else reply


def _generator_with_new_fields(prompt_text: str):
    return "safe", 0.1, {"new_prompt_field": prompt_text, "nested": {"prompt": prompt_text}}


@pytest.mark.parametrize("adapter", [
    GuardrailsAIAdapter(guard=_GuardWithNewFields()),
    LlamaGuardAdapter(generator=_generator_with_new_fields, model_name="mock-llama-guard"),
    NeMoGuardrailsAdapter(rails=_RailsWithNewFields()),
])
def test_vendor_forward_compat_no_leak(adapter) -> None:
    prompt = "sensitive prompt"
    decision = adapter.check(prompt, None, {})
    audit_blob = json.dumps(decision.audit, sort_keys=True)
    raw_blob = json.dumps(decision.raw, sort_keys=True)
    assert prompt not in audit_blob
    assert prompt not in raw_blob
