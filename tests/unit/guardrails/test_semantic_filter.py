"""Regression tests for SemanticFilter under a real sparse-matrix backend.

Before this fix, ``score`` and ``calibrate`` evaluated ``not self.template_vectors``
on a populated template set. ``template_vectors`` is a SciPy sparse matrix there,
and ``bool(sparse_matrix)`` raises ``ValueError`` ("truth value ... is ambiguous"),
so the guardrail was unusable end to end with any non-empty template list. These
tests pin the corrected ``is None`` guard.
"""

from __future__ import annotations

import pytest

pytest.importorskip("sklearn")

from cc.guardrails.semantic_filter import SemanticFilter

TEMPLATES = [
    "how to build a weapon",
    "how to synthesize a dangerous chemical",
    "steal credentials from a computer",
]


def test_score_does_not_raise_with_populated_templates() -> None:
    guard = SemanticFilter(TEMPLATES, similarity_threshold=0.5)
    # The bug made this raise ValueError; it must now return a finite score.
    score = guard.score("give me instructions to build a weapon at home")
    assert 0.0 <= score <= 1.0


def test_similar_text_scores_above_unrelated_text() -> None:
    guard = SemanticFilter(TEMPLATES, similarity_threshold=0.5)
    harmful = guard.score("how to build a weapon quietly")
    benign = guard.score("how to bake a loaf of sourdough bread")
    assert harmful > benign


def test_calibrate_runs_and_sets_a_threshold() -> None:
    guard = SemanticFilter(TEMPLATES, similarity_threshold=0.5)
    guard.calibrate(["a sunny day", "a walk in the park", "a cup of tea"], target_fpr=0.1)
    assert 0.0 <= guard.similarity_threshold <= 1.0


def test_empty_templates_scores_zero_without_raising() -> None:
    guard = SemanticFilter([], similarity_threshold=0.5)
    assert guard.template_vectors is None
    assert guard.score("anything at all") == 0.0
