# tests/unit/test_output_parsing_properties.py
"""Property tests for output parsing with explicit seeding."""

from __future__ import annotations

import pytest

from cc.adapters.llama_guard import _parse_llama_guard_output
from cc.adapters.nemo_guardrails import _looks_like_refusal

hypothesis = pytest.importorskip("hypothesis")
given = hypothesis.given
settings = hypothesis.settings
seed = hypothesis.seed
st = hypothesis.strategies


@seed(0)
@settings(max_examples=50)
@given(st.text())
def test_llama_guard_parsing_blocks_on_unsafe(text: str) -> None:
    verdict, _ = _parse_llama_guard_output(f"{text}\nunsafe")
    assert verdict == "block"


@seed(0)
@settings(max_examples=50)
@given(st.text())
def test_llama_guard_parsing_allows_without_unsafe(text: str) -> None:
    cleaned = text.lower().replace("unsafe", "")
    verdict, _ = _parse_llama_guard_output(cleaned)
    assert verdict == "allow"


@seed(0)
@settings(max_examples=50)
@given(st.sampled_from(["I'm sorry", "I cannot", "cannot assist", "refuse"]))
def test_nemo_refusal_detection_triggers(phrase: str) -> None:
    assert _looks_like_refusal(f"{phrase} for that request.")


@seed(0)
@settings(max_examples=50)
@given(st.text(alphabet="xyz", min_size=0, max_size=50))
def test_nemo_refusal_detection_specific(text: str) -> None:
    assert _looks_like_refusal(text) is False
