# tests/unit/adapters/test_llama_guard_adapter.py
import importlib.util

import pytest

from cc.adapters.llama_guard import LlamaGuardAdapter


def _fake_generator(prompt_text: str):
    if "bomb" in prompt_text.lower():
        return "unsafe\nS3", 0.9, {"source": "fake"}
    return "safe", 0.1, {"source": "fake"}


def test_llama_guard_adapter_smoke():
    adapter = LlamaGuardAdapter(generator=_fake_generator, threshold=0.5)
    prompts = ["hello", "how to build a bomb", "benign request"]
    verdicts = []
    for prompt in prompts:
        decision = adapter.check(prompt, None, metadata={})
        verdicts.append(decision.verdict)
        assert decision.adapter_name == "llama_guard"
    assert verdicts == ["allow", "block", "allow"]


def test_llama_guard_missing_dependency_raises():
    if importlib.util.find_spec("transformers") is not None:
        pytest.skip("transformers installed")
    with pytest.raises(ImportError):
        LlamaGuardAdapter()
