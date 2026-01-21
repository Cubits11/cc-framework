# tests/unit/core/models/test_models_attack_result.py

"""
AttackResult and related time/hash semantics.

Scope:
- Construction + iso_time behaviour
- Validation (transcript_hash length, utility_score constraints)
- from_transcript semantics (salt, world_bit, session_id)
- Hashing + immutability
- Semantic hashing (exclude metadata fields)
- Thread-safety and basic parallel uniqueness
"""

import concurrent.futures
import math
import time
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from cc.core.models import (
    MAX_REASONABLE_UNIX_TIMESTAMP,
    AttackResult,
    WorldBit,
    _hash_text,
    _iso_from_unix,
)
from tests._factories import mk_attack_result

# ---------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------


def assert_immutable(instance: Any, field: str, new_value: Any) -> None:
    """
    Ensure models are effectively immutable from the outside.

    We treat any attempt to reassign a field as an error, regardless of
    whether it manifests as AttributeError, ValidationError, or TypeError.
    """
    with pytest.raises((AttributeError, ValidationError, TypeError)):
        setattr(instance, field, new_value)


# ---------------------------------------------------------------------
# Basic construction & iso_time behaviour
# ---------------------------------------------------------------------


@given(
    success=st.booleans(),
    attack_id=st.text(min_size=1, max_size=32),
    guardrails_applied=st.text(min_size=1, max_size=64),
    rng_seed=st.integers(),
    timestamp=st.floats(
        allow_nan=False,
        allow_infinity=False,
        max_value=MAX_REASONABLE_UNIX_TIMESTAMP,
    ),
)
def test_attack_result_valid_and_iso_time_shape(
    success: bool,
    attack_id: str,
    guardrails_applied: str,
    rng_seed: int,
    timestamp: float,
):
    """
    AttackResult construction + iso_time formatting.

    WHAT:
        Build AttackResult instances across a range of parameters and check:
        - world_bit stored correctly
        - iso_time matches _iso_from_unix(timestamp)
        - iso_time has sensible ISO-8601 UTC structure
    WHY:
        iso_time is logged and used in audit trails.
    THREAT:
        If iso_time diverged from timestamp or changed format silently,
        downstream analysis would break.
    """
    if timestamp <= 0:
        # Normalize negative/zero timestamps to "now" – mirrors model logic.
        timestamp = time.time()

    ar = mk_attack_result(
        world_bit=WorldBit.BASELINE,
        success=success,
        attack_id=attack_id,
        guardrails_applied=guardrails_applied,
        rng_seed=rng_seed,
        timestamp=timestamp,
    )
    assert ar.world_bit is WorldBit.BASELINE

    iso = ar.iso_time

    # Must exactly match helper
    assert iso == _iso_from_unix(ar.timestamp)

    # General ISO-UTC shape: YYYY-MM-DDTHH:MM:SS(.ms)Z
    assert iso.endswith("Z")
    body = iso[:-1]
    assert "T" in body
    date_part, time_part = body.split("T")
    # Date must have 3 components
    assert len(date_part.split("-")) == 3
    # Time has at least HH:MM:SS
    assert ":" in time_part


def test_attack_result_transcript_hash_length_enforced():
    """
    transcript_hash must be a full hex digest, not an arbitrary short string.
    """
    with pytest.raises(ValidationError):
        AttackResult(
            world_bit=WorldBit.BASELINE,
            success=True,
            attack_id="id",
            transcript_hash="short",
            guardrails_applied="ga",
            rng_seed=42,
        )


def test_attack_result_transcript_hash_hex_enforced():
    """
    transcript_hash must be hex-encoded, not arbitrary characters.
    """
    with pytest.raises(ValidationError):
        AttackResult(
            world_bit=WorldBit.BASELINE,
            success=True,
            attack_id="id",
            transcript_hash="g" * 64,
            guardrails_applied="ga",
            rng_seed=42,
        )


def test_attack_result_invalid_utility_nan_and_inf():
    """
    utility_score must be finite; NaN and ±inf are rejected.

    WHY:
        Many stats routines assume finite utilities; allowing NaN/inf would
        create silent propagation bugs.
    """
    base = mk_attack_result(
        world_bit=WorldBit.BASELINE,
        success=True,
        attack_id="id",
        guardrails_applied="ga",
        rng_seed=42,
    )
    common_kwargs = dict(
        world_bit=WorldBit.BASELINE,
        success=True,
        attack_id="id",
        transcript_hash=base.transcript_hash,
        guardrails_applied="ga",
        rng_seed=42,
        timestamp=base.timestamp,
    )
    with pytest.raises(ValidationError):
        AttackResult(**common_kwargs, utility_score=math.nan)
    with pytest.raises(ValidationError):
        AttackResult(**common_kwargs, utility_score=float("inf"))
    with pytest.raises(ValidationError):
        AttackResult(**common_kwargs, utility_score=float("-inf"))


# ---------------------------------------------------------------------
# from_transcript semantics
# ---------------------------------------------------------------------


def test_attack_result_from_transcript_and_salt_variants():
    """
    from_transcript should honour salt and world_bit semantics.

    WHAT:
        Compare two AttackResult instances differing only in salt.
    WHY:
        Ensures that the transcript_hash really is a salted hash and that
        world_bit is carried through correctly.
    """
    ar1 = AttackResult.from_transcript(
        world_bit=WorldBit.PROTECTED,
        success=False,
        attack_id="id",
        transcript=b"binary",
        guardrails_applied="ga",
        rng_seed=42,
    )
    ar2 = AttackResult.from_transcript(
        world_bit=WorldBit.PROTECTED,
        success=False,
        attack_id="id",
        transcript=b"binary",
        guardrails_applied="ga",
        rng_seed=42,
        salt=b"x",
    )
    assert ar1.world_bit is WorldBit.PROTECTED
    assert ar1.transcript_hash == _hash_text(b"binary")
    assert ar2.transcript_hash == _hash_text(b"binary", salt=b"x")
    assert ar1.transcript_hash != ar2.transcript_hash

    # Baseline default for sessions/request IDs
    assert isinstance(ar1.session_id, str)
    assert getattr(ar1, "request_id", None) is None or isinstance(ar1.request_id, str)


def test_attack_result_model_hash_alias_and_immutability():
    """
    model_hash() is an alias for blake3_hash(), and instances are immutable.
    """
    ar = AttackResult.from_transcript(
        world_bit=WorldBit.BASELINE,
        success=True,
        attack_id="id",
        transcript="hello",
        guardrails_applied="none",
        rng_seed=0,
    )
    h = ar.model_hash()
    assert h == ar.blake3_hash()
    assert_immutable(ar, "success", False)


# ---------------------------------------------------------------------
# Semantic hashing (exclude metadata fields)
# ---------------------------------------------------------------------


def test_attack_result_semantic_hash_excludes_metadata_fields():
    """
    Two AttackResults differing only in metadata should share a semantic hash.

    WHAT:
        Create two logically identical results but with different metadata
        (timestamp/updated_at/etc.), then:
          - default blake3_hash: different
          - blake3_hash(exclude=metadata): identical
    WHY:
        Deduplication and semantic caching should depend on the "core"
        experimental content, not when/where the result was created.
    """
    base_kwargs = dict(
        world_bit=WorldBit.BASELINE,
        success=True,
        attack_id="attack-1",
        guardrails_applied="ga",
        rng_seed=123,
    )

    ar1 = mk_attack_result(**base_kwargs, timestamp=0.0)
    ar2 = mk_attack_result(**base_kwargs, timestamp=1.0)

    # Default hashes should usually differ because of metadata.
    h1 = ar1.blake3_hash(use_cache=False)
    h2 = ar2.blake3_hash(use_cache=False)
    assert h1 != h2

    # Define which fields are considered "metadata" for semantic hashing.
    metadata_fields = {"timestamp", "updated_at", "creator_id", "request_id"}

    sem1 = ar1.blake3_hash(exclude=metadata_fields, use_cache=False)
    sem2 = ar2.blake3_hash(exclude=metadata_fields, use_cache=False)
    assert sem1 == sem2


# ---------------------------------------------------------------------
# Thread-safety & parallel creation
# ---------------------------------------------------------------------


def test_attack_result_from_transcript_thread_safety():
    """
    from_transcript should be safe under concurrent use.

    WHAT:
        Call from_transcript in parallel and:
          - no crashes
          - each result has a valid 64-char hex transcript_hash
    WHY:
        Experiments may create AttackResults from multiple threads.
    THREAT:
        Shared mutable state inside from_transcript would cause flakiness.
    """

    def worker(i: int) -> AttackResult:
        return AttackResult.from_transcript(
            world_bit=WorldBit.BASELINE if i % 2 == 0 else WorldBit.PROTECTED,
            success=bool(i % 2),
            attack_id=f"id-{i}",
            transcript=f"payload-{i}",
            guardrails_applied="none",
            rng_seed=i,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as ex:
        results = list(ex.map(worker, range(64)))

    assert len(results) == 64
    for i, ar in enumerate(results):
        assert isinstance(ar.transcript_hash, str)
        assert len(ar.transcript_hash) == 64
        assert ar.attack_id == f"id-{i}"


def test_attack_result_request_id_uniqueness_under_parallel_creation():
    """
    request_id should be unique across many parallel AttackResult creations.

    WHAT:
        Create many results in parallel and check that request_id collisions
        do not occur (if request_id exists).
    WHY:
        request_id is intended to be a unique handle for an execution.
    """

    def worker(i: int) -> AttackResult:
        return AttackResult.from_transcript(
            world_bit=WorldBit.BASELINE,
            success=bool(i % 2),
            attack_id=f"req-{i}",
            transcript=f"payload-{i}",
            guardrails_applied="none",
            rng_seed=i,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as ex:
        results = list(ex.map(worker, range(200)))

    # If the model doesn't have a request_id field yet, this test becomes a no-op.
    if getattr(results[0], "request_id", None) is None:
        pytest.skip("AttackResult.request_id not implemented; skipping uniqueness test")

    ids = [r.request_id for r in results]
    assert len(ids) == len(set(ids)), "Collision detected in request_id generation"
