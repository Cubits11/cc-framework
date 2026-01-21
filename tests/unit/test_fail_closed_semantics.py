# tests/unit/test_fail_closed_semantics.py
"""Fail-closed behavior under exceptions and concurrent load."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

from cc.adapters.guardrails_ai import GuardrailsAIAdapter
from cc.adapters.llama_guard import LlamaGuardAdapter
from cc.adapters.nemo_guardrails import NeMoGuardrailsAdapter


class _BrokenGuard:
    version = "broken"

    def validate(self, text: str):
        raise RuntimeError(f"guard fail: {text}")


class _BrokenRails:
    def generate(self, messages: list, return_context: bool = False):
        raise TimeoutError("rails timeout")


def _broken_generator(prompt_text: str):
    raise ValueError("generation failed")


def test_fail_closed_guardrails_ai() -> None:
    adapter = GuardrailsAIAdapter(guard=_BrokenGuard())
    decision = adapter.check("prompt", None, {})
    assert decision.verdict == "review"
    assert decision.category == "adapter_error"
    assert decision.audit["decision"] == "review"
    assert decision.audit.get("error_summary", {}).get("type") == "RuntimeError"
    assert "guard fail" not in json.dumps(decision.audit)


def test_fail_closed_llama_guard() -> None:
    adapter = LlamaGuardAdapter(generator=_broken_generator, model_name="mock-llama-guard")
    decision = adapter.check("prompt", None, {})
    assert decision.verdict == "review"
    assert decision.category == "adapter_error"
    assert decision.audit.get("error_summary", {}).get("type") == "ValueError"


def test_fail_closed_nemo_guardrails() -> None:
    adapter = NeMoGuardrailsAdapter(rails=_BrokenRails())
    decision = adapter.check("prompt", None, {})
    assert decision.verdict == "review"
    assert decision.category == "adapter_error"
    assert decision.audit.get("error_summary", {}).get("type") == "TimeoutError"


def test_fail_closed_concurrency_stress() -> None:
    class FlakyGuard:
        version = "flaky"

        def validate(self, text: str):
            if "fail" in text:
                raise RuntimeError("boom")

            class Result:
                validation_passed = True
                is_valid = True

            return Result()

    adapter = GuardrailsAIAdapter(guard=FlakyGuard())
    prompts = [f"item-{i}-{'fail' if i % 3 == 0 else 'ok'}" for i in range(30)]

    def _run(prompt: str):
        return adapter.check(prompt, None, {})

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(_run, prompts))

    assert len(results) == len(prompts)
    for prompt, decision in zip(prompts, results):
        if "fail" in prompt:
            assert decision.verdict == "review"
            assert decision.audit.get("error_summary", {}).get("type") == "RuntimeError"
        else:
            assert decision.verdict in {"allow", "block", "review"}
