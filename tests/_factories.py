"""Shared factories for test data construction."""

from __future__ import annotations

from cc.core.models import AttackResult, WorldBit


def _normalize_world_bit(world_bit: int | WorldBit) -> WorldBit:
    if isinstance(world_bit, WorldBit):
        return world_bit
    return WorldBit(int(world_bit))


def _deterministic_transcript(
    *,
    world_bit: int | WorldBit,
    success: bool,
    attack_id: str,
    guardrails_applied: str,
    rng_seed: int,
) -> str:
    wb = _normalize_world_bit(world_bit)
    return (
        f"attack_id={attack_id}|world_bit={int(wb)}|success={success}|"
        f"guardrails_applied={guardrails_applied}|rng_seed={rng_seed}"
    )


def mk_attack_result(
    world_bit: int | WorldBit,
    success: bool,
    attack_id: str,
    guardrails_applied: str,
    rng_seed: int,
    timestamp: float = 0.0,
) -> AttackResult:
    """
    Canonical AttackResult factory for tests.

    Uses AttackResult.from_transcript with a deterministic transcript string,
    ensuring transcript_hash satisfies invariants and remains stable.
    """
    transcript = _deterministic_transcript(
        world_bit=world_bit,
        success=success,
        attack_id=attack_id,
        guardrails_applied=guardrails_applied,
        rng_seed=rng_seed,
    )
    return AttackResult.from_transcript(
        world_bit=_normalize_world_bit(world_bit),
        success=success,
        attack_id=attack_id,
        transcript=transcript,
        guardrails_applied=guardrails_applied,
        rng_seed=rng_seed,
        timestamp=timestamp,
    )
