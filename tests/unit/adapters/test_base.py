# tests/unit/adapters/test_base.py
import pytest

from cc.adapters.base import Decision


def test_decision_invariants_accept_valid():
    decision = Decision(
        verdict="allow",
        category=None,
        score=0.2,
        rationale="ok",
        raw={"payload": True},
        adapter_name="dummy",
        adapter_version="0.0",
    )
    assert decision.verdict == "allow"


def test_decision_invalid_verdict_raises():
    with pytest.raises(ValueError):
        Decision(
            verdict="deny",  # type: ignore[arg-type]
            category=None,
            score=None,
            rationale=None,
            raw={"payload": True},
            adapter_name="dummy",
            adapter_version="0.0",
        )


def test_decision_raw_type_enforced():
    with pytest.raises(TypeError):
        Decision(
            verdict="allow",
            category=None,
            score=None,
            rationale=None,
            raw=123,  # type: ignore[arg-type]
            adapter_name="dummy",
            adapter_version="0.0",
        )
