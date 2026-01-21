"""Adversarial tests: leak detection, vendor updates, score stability."""

from __future__ import annotations

import json

import pytest

from cc.adapters.guardrails_ai import GuardrailsAIAdapter

# ============================================================================
# TEST 1: Adversarial Leak Detection (Injection Attacks)
# ============================================================================

ADVERSARIAL_PROMPTS = [
    # SQL-injection-style
    "User query: '; DROP TABLE users; --",
    # PII extraction attempt
    "Extract all PII from: SSN 123-45-6789, email: test@example.com",
    # Nested JSON
    '{"prompt": "hidden injection", "nested": {"data": "secret"}}',
    # Unicode/special chars
    "Prompt with unicode: \u202e\u202d\u200b PII: credit card 4532123456789010",
    # Large payload
    "A" * 10000 + " PII: phone +1-555-123-4567",
]


@pytest.mark.parametrize("adversarial_prompt", ADVERSARIAL_PROMPTS, ids=lambda p: p[:30])
def test_adversarial_prompt_scrubbing(adversarial_prompt):
    """Adversarial: crafted prompts → must be hashed in audit, not leaked."""

    class MockGuard:
        version = "adversarial-test"

        def validate(self, text: str):
            class Result:
                validation_passed = True
                is_valid = True
                error = None
                error_type = None
                raw_llm_output = f"validated: {text[:50]}"

            return Result()

    adapter = GuardrailsAIAdapter(guard=MockGuard())
    metadata = {"user_input": adversarial_prompt, "nested": {"field": adversarial_prompt}}

    decision = adapter.check(adversarial_prompt, None, metadata)

    # Audit must not contain raw prompt
    audit_str = json.dumps(decision.audit)

    # Extract unique substrings to check
    unique_parts = [
        adversarial_prompt[:50],
        adversarial_prompt[-50:] if len(adversarial_prompt) > 50 else adversarial_prompt,
    ]

    for part in unique_parts:
        if len(part) > 10:  # Ignore very short strings
            assert part not in audit_str, f"Adversarial prompt leaked in audit: {part[:100]}"

    # Should have prompt_hash
    assert decision.audit["prompt_hash"], "Missing prompt_hash"
    assert len(decision.audit["prompt_hash"]) == 64, "Invalid hash length"
