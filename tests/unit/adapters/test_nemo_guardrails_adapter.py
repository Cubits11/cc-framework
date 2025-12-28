# tests/unit/adapters/test_nemo_guardrails_adapter.py
import importlib.util

import pytest

from cc.adapters.nemo_guardrails import NeMoGuardrailsAdapter


class DummyRails:
    def generate(self, messages, return_context=False):
        prompt = " ".join(m["content"] for m in messages)
        blocked = "harm" in prompt.lower()
        reply = "refuse" if blocked else "ok"
        context = {"blocked": blocked, "category": "harm" if blocked else None}
        return (reply, context) if return_context else reply


def test_nemo_guardrails_adapter_smoke():
    adapter = NeMoGuardrailsAdapter(rails=DummyRails())
    prompts = ["harmful request", "benign", "another harm scenario"]
    verdicts = [adapter.check(p, None, metadata={}).verdict for p in prompts]
    assert verdicts == ["block", "allow", "block"]


def test_nemo_guardrails_missing_dependency_raises():
    if importlib.util.find_spec("nemoguardrails") is not None:
        pytest.skip("nemoguardrails installed")
    with pytest.raises(ImportError):
        NeMoGuardrailsAdapter()
