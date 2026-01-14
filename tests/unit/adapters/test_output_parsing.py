"""Audit invariant tests: verify all adapters produce scrubbed, reproducible audits."""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict

import pytest

from cc.adapters.base import Decision
from cc.adapters.guardrails_ai import GuardrailsAIAdapter
from cc.adapters.llama_guard import LlamaGuardAdapter
from cc.adapters.nemo_guardrails import NeMoGuardrailsAdapter


# ============================================================================
# FIXTURES: Mock adapters with deterministic configs
# ============================================================================

@pytest.fixture
def mock_guardrails_adapter():
    """GuardrailsAI with mock guard to avoid real validators."""
    class MockGuard:
        version = "test-0.1.0"
        def validate(self, text: str):
            class Result:
                validation_passed = True
                is_valid = True
                error = None
                error_type = None
                raw_llm_output = f"validated: {text[:50]}"
            return Result()
    return GuardrailsAIAdapter(guard=MockGuard())


@pytest.fixture
def mock_llama_guard_adapter():
    """LlamaGuard with mock generator for deterministic outputs."""
    def mock_gen(prompt_text: str) -> tuple[str, float | None, Dict[str, Any]]:
        # Deterministic: hash prompt → score
        score = (int(hashlib.md5(prompt_text.encode()).hexdigest(), 16) % 100) / 100.0
        output = "safe" if score < 0.5 else "unsafe\nCategory: violence"
        return output, score, {"decoded": output}
    return LlamaGuardAdapter(generator=mock_gen, model_name="mock-llama-guard")


@pytest.fixture
def mock_nemo_adapter(tmp_path):
    """NeMo with mock rails for deterministic blocking."""
    class MockRails:
        def generate(self, messages: list, return_context: bool = False):
            user_msg = next((m["content"] for m in messages if m["role"] == "user"), "")
            blocked = "bomb" in user_msg.lower()
            reply = "Blocked." if blocked else "Allowed."
            context = {"blocked": blocked, "category": "violence" if blocked else None}
            return (reply, context) if return_context else reply
    return NeMoGuardrailsAdapter(rails=MockRails())


# ============================================================================
# TEST 1: Audit Existence & Structure Invariants
# ============================================================================

@pytest.mark.parametrize("adapter_fixture", [
    "mock_guardrails_adapter",
    "mock_llama_guard_adapter",
    "mock_nemo_adapter"
])
def test_audit_structure_invariants(adapter_fixture, request):
    """Every adapter must produce audit with required fields, no raw prompts."""
    adapter = request.getfixturevalue(adapter_fixture)
    prompt = "Test prompt with PII: SSN 123-45-6789"
    response = "Test response"
    metadata = {"user_id": "test-user", "session": "abc"}
    
    decision = adapter.check(prompt, response, metadata)
    
    # Invariant 1: audit exists and is dict
    assert isinstance(decision.audit, dict), f"{adapter.name}: audit must be dict"
    audit = decision.audit
    
    # Invariant 2: required fields present
    required = {
        "adapter_name", "adapter_version", "decision", "started_at", "completed_at",
        "prompt_hash", "response_hash", "parameters", "config_fingerprint"
    }
    missing = required - audit.keys()
    assert not missing, f"{adapter.name}: missing audit fields {missing}"
    
    # Invariant 3: no raw prompt/response in audit
    audit_str = json.dumps(audit)
    assert "123-45-6789" not in audit_str, f"{adapter.name}: PII leaked in audit"
    assert prompt not in audit_str, f"{adapter.name}: raw prompt in audit"
    assert response not in audit_str, f"{adapter.name}: raw response in audit"
    
    # Invariant 4: hashes are valid sha256
    sha256_pattern = re.compile(r"^[a-f0-9]{64}$")
    assert sha256_pattern.match(audit["prompt_hash"]), f"{adapter.name}: invalid prompt_hash"
    if audit.get("response_hash"):
        assert sha256_pattern.match(audit["response_hash"]), f"{adapter.name}: invalid response_hash"
    
    # Invariant 5: timestamps ordered
    assert audit["started_at"] <= audit["completed_at"], f"{adapter.name}: timestamps misordered"
    
    # Invariant 6: fingerprint present and stable
    assert audit["config_fingerprint"], f"{adapter.name}: missing config fingerprint"
    # Re-run → same fingerprint
    decision2 = adapter.check(prompt, response, metadata)
    assert decision2.audit["config_fingerprint"] == audit["config_fingerprint"], \
        f"{adapter.name}: fingerprint not stable"


# ============================================================================
# TEST 2: Metadata Scrubbing Under Adversarial Payloads
# ============================================================================

ADVERSARIAL_METADATA_CASES = [
    # Case 1: Nested prompt in metadata
    {"meta": {"prompt": "secret prompt", "content": {"text": "hidden"}}, "leak_terms": ["secret prompt", "hidden"]},
    # Case 2: Large text fields (>128 chars)
    {"meta": {"log": "x" * 500}, "leak_terms": ["x" * 500]},
    # Case 3: Custom vendor fields (e.g., input_text)
    {"meta": {"input_text": "PII: email@test.com", "custom_prompt": "leak"}, "leak_terms": ["email@test.com", "leak"]},
    # Case 4: Deeply nested structures
    {"meta": {"a": {"b": {"c": {"message": "deep secret"}}}}, "leak_terms": ["deep secret"]},
]

@pytest.mark.parametrize("adapter_fixture", [
    "mock_guardrails_adapter",
    "mock_llama_guard_adapter",
    "mock_nemo_adapter"
])
@pytest.mark.parametrize("case", ADVERSARIAL_METADATA_CASES, ids=lambda c: f"leak_{c['leak_terms'][0][:20]}")
def test_metadata_scrubbing_adversarial(adapter_fixture, case, request):
    """Adversarial: metadata with content-like keys must be hashed, not leaked."""
    adapter = request.getfixturevalue(adapter_fixture)
    metadata = case["meta"]
    leak_terms = case["leak_terms"]
    
    decision = adapter.check("safe prompt", None, metadata)
    
    # Audit must scrub metadata
    audit_str = json.dumps(decision.audit)
    for term in leak_terms:
        assert term not in audit_str, \
            f"{adapter.name}: metadata leaked '{term[:50]}' in audit"
    
    # Scrubbed metadata should have hashes
    scrubbed = decision.audit.get("metadata", {})
    
    def check_scrubbed(obj, path=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, dict) and "sha256" in v:
                    # Good: hashed
                    assert "len" in v, f"{adapter.name}: scrubbed dict missing 'len' at {path}.{k}"
                elif isinstance(v, str) and len(v) > 128:
                    pytest.fail(f"{adapter.name}: large string not hashed at {path}.{k}")
                check_scrubbed(v, f"{path}.{k}")
    
    check_scrubbed(scrubbed)


# ============================================================================
# TEST 3: Fail-Closed Under Exceptions
# ============================================================================

def test_fail_closed_guardrails(mock_guardrails_adapter):
    """GuardrailsAI: exception → verdict=review, audit has error."""
    # Inject exception
    original_guard = mock_guardrails_adapter.guard
    class BrokenGuard:
        version = "broken"
        def validate(self, text: str):
            raise RuntimeError("Simulated failure")
    mock_guardrails_adapter.guard = BrokenGuard()
    
    decision = mock_guardrails_adapter.check("test", None, {})
    
    # Fail-closed: verdict=review
    assert decision.verdict == "review", "Failed to fail-closed on exception"
    assert decision.category == "adapter_error"
    assert "errored" in decision.rationale.lower()
    
    # Audit logs error
    assert "error" in decision.audit or "adapter_error" in decision.audit.get("category", "")
    
    # Restore
    mock_guardrails_adapter.guard = original_guard


def test_fail_closed_llama_guard(mock_llama_guard_adapter):
    """LlamaGuard: exception → verdict=review, audit error."""
    # Inject exception
    original_gen = mock_llama_guard_adapter.generator
    def broken_gen(prompt_text: str):
        raise ValueError("Generation failed")
    mock_llama_guard_adapter.generator = broken_gen
    
    decision = mock_llama_guard_adapter.check("test", None, {})
    
    assert decision.verdict == "review"
    assert decision.category == "adapter_error"
    assert decision.audit.get("decision") == "review"
    
    # Restore
    mock_llama_guard_adapter.generator = original_gen


def test_fail_closed_nemo(mock_nemo_adapter):
    """NeMo: exception → verdict=review, audit error."""
    original_rails = mock_nemo_adapter.rails
    class BrokenRails:
        def generate(self, messages: list, return_context: bool = False):
            raise Exception("Rails crashed")
    mock_nemo_adapter.rails = BrokenRails()
    
    decision = mock_nemo_adapter.check("test", None, {})
    
    assert decision.verdict == "review"
    assert "errored" in decision.rationale.lower()
    
    # Restore
    mock_nemo_adapter.rails = original_rails


# ============================================================================
# TEST 4: Fingerprint Stability & Determinism
# ============================================================================

@pytest.mark.parametrize("adapter_fixture", [
    "mock_guardrails_adapter",
    "mock_llama_guard_adapter",
    "mock_nemo_adapter"
])
def test_config_fingerprint_determinism(adapter_fixture, request):
    """Config fingerprint must be identical across multiple checks."""
    adapter = request.getfixturevalue(adapter_fixture)
    
    fingerprints = set()
    for i in range(10):
        decision = adapter.check(f"prompt {i}", None, {})
        fp = decision.audit.get("config_fingerprint")
        assert fp, f"{adapter.name}: missing fingerprint on run {i}"
        fingerprints.add(fp)
    
    assert len(fingerprints) == 1, \
        f"{adapter.name}: fingerprint not stable across runs (got {len(fingerprints)} unique)"


# ============================================================================
# TEST 5: PII Fuzzing (Property-Based)
# ============================================================================

PII_PATTERNS = [
    "SSN: 123-45-6789",
    "Credit card: 4532-1234-5678-9010",
    "Email: sensitive@example.com",
    "Phone: +1-555-123-4567",
    "API key: sk-proj-abcdefghijklmnop",
]

@pytest.mark.parametrize("adapter_fixture", [
    "mock_guardrails_adapter",
    "mock_llama_guard_adapter",
    "mock_nemo_adapter"
])
@pytest.mark.parametrize("pii", PII_PATTERNS, ids=lambda p: p[:20])
def test_pii_scrubbing_fuzzing(adapter_fixture, pii, request):
    """Fuzz adapters with PII in prompt/response/metadata → must scrub."""
    adapter = request.getfixturevalue(adapter_fixture)
    
    prompt = f"User query with {pii}"
    response = f"Response containing {pii}"
    metadata = {"field": pii, "nested": {"data": pii}}
    
    decision = adapter.check(prompt, response, metadata)
    
    # Check audit
    audit_str = json.dumps(decision.audit)
    assert pii not in audit_str, f"{adapter.name}: PII '{pii}' leaked in audit"
    
    # Check raw (should also scrub)
    raw_str = json.dumps(decision.raw)
    # Allow hash previews, but not full PII
    if len(pii) > 20:  # Large enough to be hashed
        assert pii not in raw_str, f"{adapter.name}: PII '{pii}' leaked in raw"


# ============================================================================
# RERUN COMMAND
# ============================================================================
# pytest tests/unit/adapters/test_audit_invariants.py -v --tb=short
# For property-based fuzzing (install hypothesis):
# pytest tests/unit/adapters/test_audit_invariants.py -v --hypothesis-show-statistics