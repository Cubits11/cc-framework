"""Dependence-aware benchmark summaries for Paper 1 examples.

The primary Paper 1 convention is ``Z_i = 1`` means guardrail ``i`` failed to
block an unsafe prompt. The helpers here turn guardrail decisions into those
failure indicators, then compute atom-LP bounds, independence-regret
diagnostics, and endpoint witness checks.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from cc.adapters import ADAPTER_REGISTRY, Decision
from cc.adapters.base import GuardrailAdapter
from cc.guardrails.keyword_blocker import KeywordBlocker, KeywordBlockerConfig
from cc.kernel.metrics import (
    fh_position,
    fh_width,
    independence_regret,
    independent_event_probability,
)
from cc.kernel.sample_complexity import (
    PairwiseCountEvidence,
    SingletonCountEvidence,
    bernoulli_rate_count,
    composition_bounds_from_counts,
    simultaneous_bernoulli_radius,
    simultaneous_sample_size,
)
from cc.kernel.sensitivity import AssumptionSet, LinearQuery, enumerate_atoms, identified_region

SCHEMA_VERSION = "cc.evals.dependence_benchmark.v1"
E1_SCHEMA_VERSION = "cc.evals.dependence_evidence_study.v1"
FAILURE_EVENT_CONVENTION = "Z_i=1 means guardrail failure or unsafe pass."
DEFAULT_DELTA = 0.05
DEFAULT_EPSILON = 0.05
DEFAULT_KEYWORDS = (
    "attack",
    "bypass",
    "exploit",
    "jailbreak",
    "malware",
    "password",
    "prompt injection",
    "secret",
)
NUMERIC_TOL = 1.0e-8
E1_LABELS = ("g1", "g2", "g3")
E1_DEFAULT_SEED = 20260821
E1_DEFAULT_DELTA = 0.10
E1_DEFAULT_REPLICATES = 128
E1_DEFAULT_SAMPLE_SIZES = (64, 256)
E1_WILSON_Z = 1.959963984540054
E1_COVERAGE_TOLERANCE = 0.06
E1_ARTIFACT_FILENAMES = ("study.json", "coverage.csv", "manifest.json")

__all__ = [
    "E1_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "BenchmarkInputError",
    "Blocker",
    "build_summary_from_failure_matrix",
    "failure_matrix_from_rows",
    "load_dataset",
    "main",
    "run_benchmark",
    "run_e1_study",
    "verify_benchmark_summary",
    "verify_e1_artifacts",
    "verify_e1_study",
    "write_e1_artifacts",
]


class BenchmarkInputError(ValueError):
    """Raised when benchmark rows or indicators are malformed."""


class Blocker(Protocol):
    """Minimal blocking protocol shared by adapters and simple guardrails."""

    name: str
    version: str

    def blocks(self, prompt: str, response: str | None, metadata: Mapping[str, Any]) -> bool:
        """Return whether this guardrail blocks the prompt."""


@dataclass(frozen=True)
class AdapterBlocker:
    """Wrap a normalized GuardrailAdapter as a blocking predicate."""

    adapter: GuardrailAdapter
    review_policy: str = "block"

    @property
    def name(self) -> str:
        return self.adapter.name

    @property
    def version(self) -> str:
        return self.adapter.version

    def blocks(self, prompt: str, response: str | None, metadata: Mapping[str, Any]) -> bool:
        decision = self.adapter.check(prompt, response, dict(metadata))
        return _decision_blocks(decision, self.review_policy)


@dataclass(frozen=True)
class KeywordBlockerAdapter:
    """Deterministic keyword blocker used as the first worked-example pair."""

    keywords: tuple[str, ...] = DEFAULT_KEYWORDS
    threshold: float = 0.0
    name: str = "keyword_blocker"
    version: str = "deterministic-keyword-v1"

    def __post_init__(self) -> None:
        if not self.keywords:
            raise BenchmarkInputError("keyword blocker requires at least one keyword")
        object.__setattr__(
            self,
            "_guardrail",
            KeywordBlocker(
                self.keywords,
                KeywordBlockerConfig(initial_threshold=float(self.threshold)),
            ),
        )

    def blocks(self, prompt: str, response: str | None, metadata: Mapping[str, Any]) -> bool:
        text = prompt if response is None else f"{prompt}\n{response}"
        return bool(self._guardrail.blocks(text))  # type: ignore[attr-defined]


def run_benchmark(
    rows: Sequence[Mapping[str, Any]],
    blockers: Sequence[Blocker],
    *,
    prompt_field: str = "prompt",
    response_field: str | None = None,
    label_field: str = "label",
    harmful_labels: Sequence[str] = ("1", "unsafe", "harmful", "bad"),
    delta: float = DEFAULT_DELTA,
    epsilon: float = DEFAULT_EPSILON,
    dataset_id: str | None = None,
    dataset_sha256: str | None = None,
    run_id: str = "in_memory",
) -> dict[str, Any]:
    """Evaluate blockers on harmful rows and summarize failure indicators."""

    labels, matrix, records = failure_matrix_from_rows(
        rows,
        blockers,
        prompt_field=prompt_field,
        response_field=response_field,
        label_field=label_field,
        harmful_labels=harmful_labels,
    )
    adapter_versions = {blocker.name: blocker.version for blocker in blockers}
    return build_summary_from_failure_matrix(
        labels,
        matrix,
        records=records,
        adapter_versions=adapter_versions,
        delta=delta,
        epsilon=epsilon,
        dataset_id=dataset_id,
        dataset_sha256=dataset_sha256,
        run_id=run_id,
    )


def failure_matrix_from_rows(
    rows: Sequence[Mapping[str, Any]],
    blockers: Sequence[Blocker],
    *,
    prompt_field: str = "prompt",
    response_field: str | None = None,
    label_field: str = "label",
    harmful_labels: Sequence[str] = ("1", "unsafe", "harmful", "bad"),
) -> tuple[tuple[str, ...], np.ndarray[Any, np.dtype[np.int_]], list[dict[str, Any]]]:
    """Return ``(labels, failure_matrix, records)`` for harmful benchmark rows."""

    if not blockers:
        raise BenchmarkInputError("at least one blocker is required")
    labels = tuple(_unique_blocker_names(blockers))
    harmful_set = {str(label).strip().lower() for label in harmful_labels}
    records: list[dict[str, Any]] = []
    matrix_rows: list[list[int]] = []
    for index, row in enumerate(rows):
        row_label = str(row.get(label_field, "")).strip().lower()
        if row_label not in harmful_set:
            continue
        prompt = str(row.get(prompt_field, ""))
        response = (
            str(row.get(response_field))
            if response_field is not None and row.get(response_field) is not None
            else None
        )
        failures: dict[str, int] = {}
        for blocker in blockers:
            blocked = blocker.blocks(prompt, response, row)
            failures[blocker.name] = int(not blocked)
        matrix_rows.append([failures[label] for label in labels])
        records.append(
            {
                "row_index": index,
                "row_id": str(row.get("id", index)),
                "label": row.get(label_field),
                "failure_indicators": failures,
            }
        )
    if not matrix_rows:
        raise BenchmarkInputError("no harmful rows were found for dependence benchmarking")
    return labels, np.asarray(matrix_rows, dtype=np.int_), records


def build_summary_from_failure_matrix(
    labels: Sequence[str],
    failure_matrix: Sequence[Sequence[int]] | np.ndarray[Any, Any],
    *,
    records: Sequence[Mapping[str, Any]] | None = None,
    adapter_versions: Mapping[str, str] | None = None,
    delta: float = DEFAULT_DELTA,
    epsilon: float = DEFAULT_EPSILON,
    dataset_id: str | None = None,
    dataset_sha256: str | None = None,
    run_id: str = "manual",
) -> dict[str, Any]:
    """Build a schema-valid dependence benchmark summary from binary failures."""

    label_names = _validate_labels(labels)
    matrix = _failure_matrix(failure_matrix, expected_width=len(label_names))
    n_rows = int(matrix.shape[0])
    marginals = {label: float(np.mean(matrix[:, index])) for index, label in enumerate(label_names)}
    pairwise_overlaps: dict[str, float] = {}
    for left_index, left in enumerate(label_names):
        for right_index in range(left_index + 1, len(label_names)):
            right = label_names[right_index]
            pairwise_overlaps[f"{left}&{right}"] = float(
                np.mean(matrix[:, left_index] * matrix[:, right_index])
            )

    events = {
        "any_guardrail_failure": LinearQuery.union(
            label_names,
            label_names,
            name="any_guardrail_failure",
            description="P(at least one guardrail fails to block an unsafe prompt)",
        ),
        "stack_unsafe_pass": LinearQuery.intersection(
            label_names,
            label_names,
            name="stack_unsafe_pass",
            description="P(all selected guardrails fail; OR-blocking stack passes unsafe prompt)",
        ),
    }
    marginal_assumptions = _marginal_assumptions(label_names, marginals)
    pairwise_assumptions = _pairwise_assumptions(label_names, marginals, pairwise_overlaps)

    event_summaries = {
        name: {
            "marginal_only": _event_summary(query, marginal_assumptions, marginals, matrix),
            "pairwise_tightened": _event_summary(
                query,
                pairwise_assumptions,
                marginals,
                matrix,
            ),
        }
        for name, query in events.items()
    }
    rate_count = bernoulli_rate_count(len(label_names), include_pairwise=True)
    sample_complexity = {
        "n_harmful": n_rows,
        "delta": float(delta),
        "epsilon": float(epsilon),
        "num_singleton_rates": len(label_names),
        "num_pairwise_rates": len(pairwise_overlaps),
        "num_simultaneous_rates": rate_count,
        "hoeffding_radius": simultaneous_bernoulli_radius(n_rows, rate_count, delta),
        "sample_size_for_epsilon": simultaneous_sample_size(epsilon, rate_count, delta),
    }
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "failure_event_convention": FAILURE_EVENT_CONVENTION,
        "dataset": {
            "id": dataset_id,
            "sha256": dataset_sha256,
            "population": "harmful prompts only",
            "notes": (
                "CI fixtures may use mock adapters; paper evidence must use pinned real-model "
                "runs with dataset hashes and adapter versions."
            ),
        },
        "labels": list(label_names),
        "adapter_versions": dict(adapter_versions or {}),
        "n_harmful": n_rows,
        "failure_indicators": list(records or _records_from_matrix(label_names, matrix)),
        "singleton_failure_rates": marginals,
        "pairwise_failure_overlaps": pairwise_overlaps,
        "events": event_summaries,
        "sample_complexity": sample_complexity,
    }
    errors = verify_benchmark_summary(payload)
    if errors:
        raise BenchmarkInputError("; ".join(errors))
    return payload


def verify_benchmark_summary(payload: Mapping[str, Any]) -> list[str]:
    """Return verification errors for a dependence benchmark summary."""

    errors: list[str] = []
    if payload.get("schema_version") != SCHEMA_VERSION:
        errors.append("schema_version mismatch")
    try:
        labels = _validate_labels(payload["labels"])
        marginals = _float_mapping(payload["singleton_failure_rates"])
        events = _mapping(payload["events"])
    except (KeyError, TypeError, ValueError) as exc:
        return [*errors, f"malformed benchmark summary: {exc}"]

    if set(labels) != set(marginals):
        errors.append("singleton_failure_rates keys must match labels")
    for event_name, event_payload in events.items():
        if not isinstance(event_payload, Mapping):
            errors.append(f"{event_name} payload must be a mapping")
            continue
        for assumption_key in ("marginal_only", "pairwise_tightened"):
            summary = event_payload.get(assumption_key)
            if not isinstance(summary, Mapping):
                errors.append(f"{event_name}.{assumption_key} must be a mapping")
                continue
            errors.extend(_verify_event_summary(str(event_name), assumption_key, summary))
    return errors


def load_dataset(path: Path) -> tuple[list[dict[str, Any]], str]:
    """Load a CSV, JSONL, or JSON-list dataset and return rows plus file hash."""

    dataset_hash = _sha256_file(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle)), dataset_hash
    if suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped:
                    payload = json.loads(stripped)
                    if not isinstance(payload, Mapping):
                        raise BenchmarkInputError("JSONL records must be objects")
                    rows.append(dict(payload))
        return rows, dataset_hash
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, list) or not all(isinstance(row, Mapping) for row in payload):
            raise BenchmarkInputError("JSON benchmark datasets must be lists of objects")
        return [dict(row) for row in payload], dataset_hash
    raise BenchmarkInputError(f"unsupported dataset suffix: {path.suffix}")


def run_e1_study(
    *,
    seed: int = E1_DEFAULT_SEED,
    delta: float = E1_DEFAULT_DELTA,
    replicates: int = E1_DEFAULT_REPLICATES,
    sample_sizes: Sequence[int] = E1_DEFAULT_SAMPLE_SIZES,
) -> dict[str, Any]:
    """Run the deterministic controlled-synthetic E1 evidence study.

    E1 is deliberately a *controlled synthetic* study.  Its known joint laws
    make it possible to check that more declared singleton/pairwise evidence
    changes the identified region in the expected way, and that count-derived
    simultaneous intervals cover the known target at their stated scope.  It
    is not a guardrail deployment evaluation or a claim about a real
    population.
    """

    if not isinstance(seed, int):
        raise BenchmarkInputError("seed must be an integer")
    if not 0.0 < float(delta) < 1.0:
        raise BenchmarkInputError("delta must lie strictly between 0 and 1")
    if not isinstance(replicates, int) or replicates < 1:
        raise BenchmarkInputError("replicates must be a positive integer")
    checked_sample_sizes = _e1_sample_sizes(sample_sizes)
    scenarios = _e1_scenarios()
    events = _e1_events()

    exact_scenarios = [
        _e1_exact_scenario_payload(scenario, events["stack_unsafe_pass"]) for scenario in scenarios
    ]
    coverage_rows = _e1_coverage_rows(
        scenarios,
        events,
        seed=seed,
        delta=float(delta),
        replicates=replicates,
        sample_sizes=checked_sample_sizes,
    )
    payload: dict[str, Any] = {
        "schema_version": E1_SCHEMA_VERSION,
        "study_id": "E1-dependence-evidence",
        "status": "controlled_synthetic_complete; real_world_pilot_untested",
        "failure_event_convention": FAILURE_EVENT_CONVENTION,
        "atom_order": "little_endian",
        "labels": list(E1_LABELS),
        "design": {
            "seed": seed,
            "delta": float(delta),
            "nominal_coverage": 1.0 - float(delta),
            "replicates": replicates,
            "sample_sizes": list(checked_sample_sizes),
            "target_selection": "fixed_pre_sampling",
            "confidence_method": "simultaneous_hoeffding_union_bound",
            "coverage_decision_rule": (
                "Wilson 95% lower confidence limit for simulated coverage must be at least "
                "nominal_coverage - 0.06; this is a Monte Carlo diagnostic, not a proof."
            ),
            "regimes": {
                "B0": "product baseline; explicit independence assumption, not an LP constraint",
                "I0": "declared exact singleton moments only",
                "I1": "I0 plus fixed-before-output pair g1&g2",
                "I2": "I0 plus all declared pairwise moments",
                "I3": "declared full joint record; measured reference only",
            },
        },
        "primary_event": "stack_unsafe_pass",
        "scenarios": exact_scenarios,
        "coverage_rows": coverage_rows,
        "controls": {
            "broken_data": {
                "status": "tested_in_unit_suite",
                "checks": [
                    "non-binary indicator matrix is rejected",
                    "duplicate labels are rejected",
                    "empty sample sizes and invalid confidence parameters are rejected",
                    "permuting label order preserves named-query results",
                ],
            },
            "parity_oracle": {
                "status": "verified",
                "claim": (
                    "even and odd parity have equal singleton and pairwise moments but different "
                    "three-way unsafe-pass probability"
                ),
            },
        },
        "nonclaims": [
            "No real-world guardrail population was sampled.",
            "No adapter, benchmark, label, or deployment claim is validated here.",
            "I3 is a measured synthetic reference, not deployment truth.",
            "The coverage simulation checks this declared generator and estimator scope only.",
        ],
    }
    errors = verify_e1_study(payload)
    if errors:
        raise BenchmarkInputError("; ".join(errors))
    return payload


def write_e1_artifacts(
    output_dir: Path,
    *,
    generation_command: str,
    seed: int = E1_DEFAULT_SEED,
    delta: float = E1_DEFAULT_DELTA,
    replicates: int = E1_DEFAULT_REPLICATES,
    sample_sizes: Sequence[int] = E1_DEFAULT_SAMPLE_SIZES,
) -> dict[str, Any]:
    """Generate small, manifest-hashed E1 artifacts without deleting prior files."""

    payload = run_e1_study(
        seed=seed,
        delta=delta,
        replicates=replicates,
        sample_sizes=sample_sizes,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_e1_json(output_dir / "study.json", payload)
    (output_dir / "coverage.csv").write_text(
        _e1_coverage_csv(payload["coverage_rows"]), encoding="utf-8"
    )
    manifest = _e1_manifest(output_dir, generation_command=generation_command)
    _write_e1_json(output_dir / "manifest.json", manifest)
    return payload


def verify_e1_artifacts(
    output_dir: Path,
    *,
    regenerate: bool = False,
) -> list[str]:
    """Verify E1 schema, witnesses, deterministic tables, and manifest hashes.

    With ``regenerate=True``, regenerate the study in memory from its declared
    design and require its exact payload.  This is intentionally the stronger,
    slower clean-environment check used by the command-line verifier.
    """

    errors: list[str] = []
    for filename in E1_ARTIFACT_FILENAMES:
        if not (output_dir / filename).is_file():
            errors.append(f"missing required E1 artifact: {filename}")
    if errors:
        return errors
    try:
        study = json.loads((output_dir / "study.json").read_text(encoding="utf-8"))
        manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"cannot read E1 artifact: {exc}"]
    if not isinstance(study, Mapping):
        return ["study.json must contain an object"]
    if not isinstance(manifest, Mapping):
        return ["manifest.json must contain an object"]
    errors.extend(verify_e1_study(study))
    expected_csv = _e1_coverage_csv(study.get("coverage_rows", []))
    try:
        if (output_dir / "coverage.csv").read_text(encoding="utf-8") != expected_csv:
            errors.append("coverage.csv does not exactly represent study.json coverage_rows")
    except OSError as exc:
        errors.append(f"cannot read coverage.csv: {exc}")
    errors.extend(_verify_e1_manifest(output_dir, manifest))
    if regenerate and not errors:
        try:
            design = _mapping(study["design"])
            regenerated = run_e1_study(
                seed=int(design["seed"]),
                delta=float(design["delta"]),
                replicates=int(design["replicates"]),
                sample_sizes=tuple(int(value) for value in design["sample_sizes"]),
            )
            if regenerated != study:
                errors.append("study.json differs from deterministic regeneration")
        except (KeyError, TypeError, ValueError, BenchmarkInputError) as exc:
            errors.append(f"cannot regenerate E1 study: {exc}")
    return errors


def verify_e1_study(payload: Mapping[str, Any]) -> list[str]:
    """Return semantic verification errors for an in-memory E1 study payload."""

    errors: list[str] = []
    if payload.get("schema_version") != E1_SCHEMA_VERSION:
        errors.append("E1 schema_version mismatch")
        return errors
    if payload.get("atom_order") != "little_endian":
        errors.append("E1 atom_order mismatch")
    try:
        labels = _validate_labels(payload["labels"])
        design = _mapping(payload["design"])
        scenarios = payload["scenarios"]
        coverage_rows = payload["coverage_rows"]
    except (KeyError, TypeError, ValueError) as exc:
        return [*errors, f"malformed E1 study: {exc}"]
    if labels != E1_LABELS:
        errors.append("E1 labels must be g1, g2, g3 in declared order")
    if not isinstance(scenarios, list) or len(scenarios) != 5:
        errors.append("E1 must contain exactly five declared joint-law scenarios")
        return errors
    names = [str(item.get("scenario_id")) for item in scenarios if isinstance(item, Mapping)]
    if set(names) != {scenario["scenario_id"] for scenario in _e1_scenarios()}:
        errors.append("E1 scenario IDs do not match the frozen controlled suite")
    for scenario in scenarios:
        if not isinstance(scenario, Mapping):
            errors.append("E1 scenario must be an object")
            continue
        errors.extend(_verify_e1_exact_scenario(scenario))
    errors.extend(_verify_e1_parity_oracle(scenarios))
    if not isinstance(coverage_rows, list) or not coverage_rows:
        errors.append("E1 coverage_rows must be a nonempty list")
    else:
        errors.extend(_verify_e1_coverage_rows(coverage_rows, design))
    return errors


def _e1_scenarios() -> tuple[dict[str, Any], ...]:
    """Return the frozen three-indicator joint-law suite in atom order."""

    return (
        {
            "scenario_id": "S1_independent_reference",
            "surface": "S1",
            "description": "Independent Bernoulli(0.5) reference distribution.",
            "probabilities": (0.125,) * 8,
        },
        {
            "scenario_id": "S2_common_cause",
            "surface": "S2",
            "description": "A shared latent cause makes all three failures co-occur.",
            "probabilities": (0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5),
        },
        {
            "scenario_id": "S3_mutually_exclusive",
            "surface": "S3",
            "description": "At most one failure occurs on any row.",
            "probabilities": (0.25, 0.25, 0.25, 0.0, 0.25, 0.0, 0.0, 0.0),
        },
        {
            "scenario_id": "S4_parity_even",
            "surface": "S4",
            "description": "Uniform even parity: 000, 110, 101, 011.",
            "probabilities": (0.25, 0.0, 0.0, 0.25, 0.0, 0.25, 0.25, 0.0),
        },
        {
            "scenario_id": "S4_parity_odd",
            "surface": "S4",
            "description": "Uniform odd parity: 100, 010, 001, 111.",
            "probabilities": (0.0, 0.25, 0.25, 0.0, 0.25, 0.0, 0.0, 0.25),
        },
    )


def _e1_events() -> dict[str, LinearQuery]:
    return {
        "stack_unsafe_pass": LinearQuery.intersection(
            E1_LABELS,
            E1_LABELS,
            name="stack_unsafe_pass",
            description="All three guardrails fail on a synthetic row.",
        ),
        "any_guardrail_failure": LinearQuery.union(
            E1_LABELS,
            E1_LABELS,
            name="any_guardrail_failure",
            description="At least one guardrail fails on a synthetic row.",
        ),
    }


def _e1_exact_scenario_payload(
    scenario: Mapping[str, Any],
    primary_query: LinearQuery,
) -> dict[str, Any]:
    probabilities = _e1_probabilities(scenario["probabilities"])
    moments = _e1_moments(probabilities)
    true_rate = float(primary_query.coefficients @ probabilities)
    regimes = {
        key: _e1_exact_regime_summary(key, primary_query, probabilities, moments)
        for key in ("I0", "I1", "I2", "I3")
    }
    product_baseline = independent_event_probability(
        moments["singletons"], primary_query, labels=E1_LABELS
    )
    return {
        "scenario_id": str(scenario["scenario_id"]),
        "surface": str(scenario["surface"]),
        "description": str(scenario["description"]),
        "atom_probabilities": _float_list(probabilities),
        "moments": moments,
        "primary_event": primary_query.name,
        "true_primary_event_rate": true_rate,
        "B0_product_baseline": product_baseline,
        "regimes": regimes,
    }


def _e1_moments(probabilities: np.ndarray[Any, Any]) -> dict[str, Any]:
    states = enumerate_atoms(E1_LABELS)
    singletons = {
        label: float(probabilities @ states[:, index]) for index, label in enumerate(E1_LABELS)
    }
    pairs: dict[str, float] = {}
    for left_index, left in enumerate(E1_LABELS):
        for right_index in range(left_index + 1, len(E1_LABELS)):
            right = E1_LABELS[right_index]
            pairs[f"{left}&{right}"] = float(
                probabilities @ (states[:, left_index] * states[:, right_index])
            )
    return {"singletons": singletons, "pairwise": pairs}


def _e1_exact_regime_summary(
    regime: str,
    query: LinearQuery,
    probabilities: np.ndarray[Any, Any],
    moments: Mapping[str, Any],
) -> dict[str, Any]:
    assumptions = _e1_exact_assumptions(regime, probabilities, moments)
    result = identified_region(query, assumptions)
    true_rate = float(query.coefficients @ probabilities)
    lower_witness = _float_list(result.lower_solution)
    upper_witness = _float_list(result.upper_solution)
    return {
        "assumptions_hash": result.assumptions_hash,
        "lower_bound": float(result.lower_bound),
        "upper_bound": float(result.upper_bound),
        "width": float(result.width),
        "contains_declared_truth": bool(
            result.lower_bound - NUMERIC_TOL <= true_rate <= result.upper_bound + NUMERIC_TOL
        ),
        "solver_status": result.solver_status,
        "witnesses": {"lower": lower_witness, "upper": upper_witness},
        "witness_checks": {
            "lower": _verify_endpoint(
                query, assumptions, result.lower_solution, result.lower_bound
            ),
            "upper": _verify_endpoint(
                query, assumptions, result.upper_solution, result.upper_bound
            ),
        },
    }


def _e1_exact_assumptions(
    regime: str,
    probabilities: np.ndarray[Any, Any],
    moments: Mapping[str, Any],
) -> AssumptionSet:
    if regime not in {"I0", "I1", "I2", "I3"}:
        raise BenchmarkInputError(f"unsupported E1 exact regime: {regime}")
    assumptions = AssumptionSet.empty(
        E1_LABELS,
        metadata={
            "study": "E1-dependence-evidence",
            "regime": regime,
            "evidence_role": "declared_exact_synthetic_reference",
        },
    )
    if regime == "I3":
        for atom_index, probability in enumerate(probabilities):
            coefficients = np.zeros(len(probabilities), dtype=float)
            coefficients[atom_index] = 1.0
            assumptions = assumptions.with_linear_constraint(
                f"full_joint_atom:{atom_index}", coefficients, "==", float(probability)
            )
        return assumptions
    singleton_moments = _float_mapping(moments["singletons"])
    pairwise_moments = _float_mapping(moments["pairwise"])
    for label in E1_LABELS:
        assumptions = assumptions.with_marginal_interval(
            label, singleton_moments[label], singleton_moments[label]
        )
    if regime == "I1":
        selected_pairs = (("g1", "g2"),)
    elif regime == "I2":
        selected_pairs = (("g1", "g2"), ("g1", "g3"), ("g2", "g3"))
    else:
        selected_pairs = ()
    for left, right in selected_pairs:
        moment = pairwise_moments[f"{left}&{right}"]
        assumptions = assumptions.with_pairwise_joint_interval(left, right, moment, moment)
    return assumptions


def _e1_coverage_rows(
    scenarios: Sequence[Mapping[str, Any]],
    events: Mapping[str, LinearQuery],
    *,
    seed: int,
    delta: float,
    replicates: int,
    sample_sizes: Sequence[int],
) -> list[dict[str, Any]]:
    """Run the frozen finite-sample coverage grid with independent substreams."""

    rows: list[dict[str, Any]] = []
    regimes = ("I0", "I1", "I2")
    for scenario_index, scenario in enumerate(scenarios):
        probabilities = _e1_probabilities(scenario["probabilities"])
        for sample_size in sample_sizes:
            for event_index, (event_name, query) in enumerate(events.items()):
                target = float(query.coefficients @ probabilities)
                for regime_index, regime in enumerate(regimes):
                    covered = 0
                    widths: list[float] = []
                    for replicate in range(replicates):
                        stream_seed = _e1_stream_seed(
                            seed,
                            scenario_index,
                            sample_size,
                            event_index,
                            regime_index,
                            replicate,
                        )
                        matrix = _e1_sample_matrix(probabilities, sample_size, stream_seed)
                        finite = _e1_finite_result(query, matrix, regime=regime, delta=delta)
                        interval = finite.identification
                        widths.append(float(interval.width))
                        if (
                            interval.lower_bound - NUMERIC_TOL
                            <= target
                            <= interval.upper_bound + NUMERIC_TOL
                        ):
                            covered += 1
                    coverage = covered / replicates
                    wilson_lower, wilson_upper = _e1_wilson_interval(covered, replicates)
                    rows.append(
                        {
                            "scenario_id": str(scenario["scenario_id"]),
                            "event": event_name,
                            "regime": regime,
                            "n": sample_size,
                            "replicates": replicates,
                            "covered": covered,
                            "coverage": coverage,
                            "wilson_95_lower": wilson_lower,
                            "wilson_95_upper": wilson_upper,
                            "mean_width": float(np.mean(widths)),
                            "target_rate": target,
                            "nominal_coverage": 1.0 - delta,
                            "coverage_rule_passed": bool(
                                wilson_lower >= 1.0 - delta - E1_COVERAGE_TOLERANCE
                            ),
                        }
                    )
    return rows


def _e1_finite_result(
    query: LinearQuery,
    matrix: np.ndarray[Any, Any],
    *,
    regime: str,
    delta: float,
) -> Any:
    singleton_counts = tuple(
        SingletonCountEvidence(
            label=label,
            failures=int(matrix[:, index].sum()),
            n=int(matrix.shape[0]),
        )
        for index, label in enumerate(E1_LABELS)
    )
    if regime == "I1":
        selected_pairs = ((0, 1),)
    elif regime == "I2":
        selected_pairs = ((0, 1), (0, 2), (1, 2))
    elif regime == "I0":
        selected_pairs = ()
    else:
        raise BenchmarkInputError(f"unsupported E1 finite regime: {regime}")
    pairwise_counts = tuple(
        PairwiseCountEvidence(
            left=E1_LABELS[left],
            right=E1_LABELS[right],
            co_failures=int((matrix[:, left] * matrix[:, right]).sum()),
            n=int(matrix.shape[0]),
        )
        for left, right in selected_pairs
    )
    return composition_bounds_from_counts(
        query,
        E1_LABELS,
        singleton_counts,
        pairwise_counts=pairwise_counts,
        delta=delta,
        metadata={
            "study": "E1-dependence-evidence",
            "regime": regime,
            "evidence_role": "empirical_estimate",
        },
        target_selection="fixed_pre_sampling",
    )


def _e1_sample_matrix(
    probabilities: np.ndarray[Any, Any],
    n: int,
    seed: int,
) -> np.ndarray[Any, Any]:
    generator = np.random.default_rng(seed)
    indices = generator.choice(len(probabilities), size=n, p=probabilities)
    return enumerate_atoms(E1_LABELS)[indices]


def _e1_stream_seed(
    seed: int,
    scenario_index: int,
    sample_size: int,
    event_index: int,
    regime_index: int,
    replicate: int,
) -> int:
    digest = hashlib.sha256(
        f"{seed}|{scenario_index}|{sample_size}|{event_index}|{regime_index}|{replicate}".encode()
    ).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _e1_wilson_interval(successes: int, trials: int) -> tuple[float, float]:
    if trials < 1 or successes < 0 or successes > trials:
        raise BenchmarkInputError("Wilson interval requires 0 <= successes <= positive trials")
    z_squared = E1_WILSON_Z * E1_WILSON_Z
    rate = successes / trials
    denominator = 1.0 + z_squared / trials
    centre = (rate + z_squared / (2.0 * trials)) / denominator
    half_width = (
        E1_WILSON_Z
        * np.sqrt((rate * (1.0 - rate) + z_squared / (4.0 * trials)) / trials)
        / denominator
    )
    return max(0.0, float(centre - half_width)), min(1.0, float(centre + half_width))


def _e1_probabilities(value: Any) -> np.ndarray[Any, Any]:
    probabilities = np.asarray(value, dtype=float)
    if probabilities.shape != (1 << len(E1_LABELS),):
        raise BenchmarkInputError("E1 scenario must specify one probability per atom")
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities < -NUMERIC_TOL):
        raise BenchmarkInputError("E1 atom probabilities must be finite and nonnegative")
    if abs(float(probabilities.sum()) - 1.0) > NUMERIC_TOL:
        raise BenchmarkInputError("E1 atom probabilities must sum to one")
    return probabilities


def _e1_sample_sizes(sample_sizes: Sequence[int]) -> tuple[int, ...]:
    if isinstance(sample_sizes, str):
        raise BenchmarkInputError("sample_sizes must be a sequence of integers")
    values = tuple(sample_sizes)
    if not values or any(not isinstance(value, int) or value < 1 for value in values):
        raise BenchmarkInputError("sample_sizes must contain positive integers")
    if len(set(values)) != len(values):
        raise BenchmarkInputError("sample_sizes must be unique")
    return values


def _verify_e1_exact_scenario(scenario: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    try:
        scenario_id = str(scenario["scenario_id"])
        probabilities = _e1_probabilities(scenario["atom_probabilities"])
        moments = _mapping(scenario["moments"])
        regimes = _mapping(scenario["regimes"])
        true_rate = float(scenario["true_primary_event_rate"])
    except (KeyError, TypeError, ValueError, BenchmarkInputError) as exc:
        return [f"malformed E1 exact scenario: {exc}"]
    frozen = {item["scenario_id"]: item for item in _e1_scenarios()}
    if scenario_id not in frozen:
        return [f"unknown E1 scenario: {scenario_id}"]
    expected_probabilities = _e1_probabilities(frozen[scenario_id]["probabilities"])
    if not np.allclose(probabilities, expected_probabilities, atol=NUMERIC_TOL, rtol=0.0):
        errors.append(f"{scenario_id} atom probabilities differ from frozen suite")
    primary_query = _e1_events()["stack_unsafe_pass"]
    expected_moments = _e1_moments(probabilities)
    if moments != expected_moments:
        errors.append(f"{scenario_id} moments do not match atom probabilities")
    expected_true_rate = float(primary_query.coefficients @ probabilities)
    if abs(true_rate - expected_true_rate) > NUMERIC_TOL:
        errors.append(f"{scenario_id} true primary event rate is incorrect")
    try:
        baseline = float(scenario["B0_product_baseline"])
        expected_baseline = independent_event_probability(
            expected_moments["singletons"], primary_query, labels=E1_LABELS
        )
        if abs(baseline - expected_baseline) > NUMERIC_TOL:
            errors.append(f"{scenario_id} B0 product baseline is incorrect")
    except (KeyError, TypeError, ValueError):
        errors.append(f"{scenario_id} B0 product baseline is invalid")
    for regime in ("I0", "I1", "I2", "I3"):
        record = regimes.get(regime)
        if not isinstance(record, Mapping):
            errors.append(f"{scenario_id}.{regime} record is missing")
            continue
        errors.extend(
            _verify_e1_exact_regime(scenario_id, regime, record, probabilities, expected_moments)
        )
    return errors


def _verify_e1_exact_regime(
    scenario_id: str,
    regime: str,
    record: Mapping[str, Any],
    probabilities: np.ndarray[Any, Any],
    moments: Mapping[str, Any],
) -> list[str]:
    errors: list[str] = []
    query = _e1_events()["stack_unsafe_pass"]
    try:
        expected = _e1_exact_regime_summary(regime, query, probabilities, moments)
        assumptions = _e1_exact_assumptions(regime, probabilities, moments)
    except (KeyError, TypeError, ValueError, BenchmarkInputError) as exc:
        return [f"{scenario_id}.{regime} cannot reconstruct declared assumptions: {exc}"]
    for key in ("lower_bound", "upper_bound", "width"):
        try:
            if abs(float(record[key]) - float(expected[key])) > NUMERIC_TOL:
                errors.append(f"{scenario_id}.{regime}.{key} is incorrect")
        except (KeyError, TypeError, ValueError):
            errors.append(f"{scenario_id}.{regime}.{key} is invalid")
    if record.get("assumptions_hash") != expected["assumptions_hash"]:
        errors.append(f"{scenario_id}.{regime}.assumptions_hash is incorrect")
    if record.get("solver_status") != expected["solver_status"]:
        errors.append(f"{scenario_id}.{regime}.solver_status is incorrect")
    if record.get("contains_declared_truth") is not True:
        errors.append(f"{scenario_id}.{regime} does not contain declared truth")
    witnesses = record.get("witnesses")
    if not isinstance(witnesses, Mapping):
        return [*errors, f"{scenario_id}.{regime}.witnesses is invalid"]
    for endpoint, bound_key in (("lower", "lower_bound"), ("upper", "upper_bound")):
        try:
            distribution = np.asarray(witnesses[endpoint], dtype=float)
            bound = float(record[bound_key])
            if not _verify_endpoint(query, assumptions, distribution, bound):
                errors.append(f"{scenario_id}.{regime}.{endpoint} witness does not verify")
        except (KeyError, TypeError, ValueError):
            errors.append(f"{scenario_id}.{regime}.{endpoint} witness is invalid")
    if record.get("witness_checks") != {"lower": True, "upper": True}:
        errors.append(f"{scenario_id}.{regime} witness_checks must both be true")
    return errors


def _verify_e1_parity_oracle(scenarios: Sequence[Any]) -> list[str]:
    by_id = {
        str(scenario.get("scenario_id")): scenario
        for scenario in scenarios
        if isinstance(scenario, Mapping)
    }
    even = by_id.get("S4_parity_even")
    odd = by_id.get("S4_parity_odd")
    if not isinstance(even, Mapping) or not isinstance(odd, Mapping):
        return ["E1 parity scenarios are missing"]
    try:
        even_moments = _mapping(even["moments"])
        odd_moments = _mapping(odd["moments"])
        even_rate = float(even["true_primary_event_rate"])
        odd_rate = float(odd["true_primary_event_rate"])
    except (KeyError, TypeError, ValueError) as exc:
        return [f"malformed E1 parity oracle: {exc}"]
    errors: list[str] = []
    if even_moments != odd_moments:
        errors.append("E1 parity singleton/pairwise moments must agree")
    if abs(even_rate - odd_rate) < NUMERIC_TOL:
        errors.append("E1 parity primary event rates must differ")
    for regime in ("I0", "I1", "I2"):
        try:
            even_bounds = _mapping(_mapping(even["regimes"])[regime])
            odd_bounds = _mapping(_mapping(odd["regimes"])[regime])
            for bound in ("lower_bound", "upper_bound"):
                if abs(float(even_bounds[bound]) - float(odd_bounds[bound])) > NUMERIC_TOL:
                    errors.append(f"E1 parity {regime} bounds must agree")
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"malformed E1 parity {regime} result: {exc}")
    return errors


def _verify_e1_coverage_rows(rows: Sequence[Any], design: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    try:
        delta = float(design["delta"])
        replicates = int(design["replicates"])
        sample_sizes = _e1_sample_sizes(tuple(int(value) for value in design["sample_sizes"]))
    except (KeyError, TypeError, ValueError, BenchmarkInputError) as exc:
        return [f"malformed E1 coverage design: {exc}"]
    expected_count = len(_e1_scenarios()) * len(_e1_events()) * len(sample_sizes) * 3
    if len(rows) != expected_count:
        errors.append(f"E1 coverage grid has {len(rows)} rows; expected {expected_count}")
    seen: set[tuple[str, str, str, int]] = set()
    scenario_ids = {scenario["scenario_id"] for scenario in _e1_scenarios()}
    for row in rows:
        if not isinstance(row, Mapping):
            errors.append("E1 coverage row must be an object")
            continue
        try:
            key = (str(row["scenario_id"]), str(row["event"]), str(row["regime"]), int(row["n"]))
            if key in seen:
                errors.append(f"duplicate E1 coverage row: {key}")
            seen.add(key)
            covered = int(row["covered"])
            observed_replicates = int(row["replicates"])
            coverage = float(row["coverage"])
            lower = float(row["wilson_95_lower"])
            upper = float(row["wilson_95_upper"])
            target = float(row["target_rate"])
            mean_width = float(row["mean_width"])
            nominal = float(row["nominal_coverage"])
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"malformed E1 coverage row: {exc}")
            continue
        if (
            key[0] not in scenario_ids
            or key[1] not in _e1_events()
            or key[2] not in {"I0", "I1", "I2"}
        ):
            errors.append(f"invalid E1 coverage grid key: {key}")
        if key[3] not in sample_sizes or observed_replicates != replicates:
            errors.append(f"E1 coverage row has undeclared sample design: {key}")
        if not 0 <= covered <= replicates or abs(coverage - covered / replicates) > NUMERIC_TOL:
            errors.append(f"E1 coverage count/rate inconsistent: {key}")
            continue
        expected_lower, expected_upper = _e1_wilson_interval(covered, replicates)
        if abs(lower - expected_lower) > NUMERIC_TOL or abs(upper - expected_upper) > NUMERIC_TOL:
            errors.append(f"E1 Wilson interval is incorrect: {key}")
        if not 0.0 <= target <= 1.0 or mean_width < -NUMERIC_TOL:
            errors.append(f"E1 coverage values are out of range: {key}")
        if abs(nominal - (1.0 - delta)) > NUMERIC_TOL:
            errors.append(f"E1 nominal coverage is incorrect: {key}")
        expected_pass = lower >= 1.0 - delta - E1_COVERAGE_TOLERANCE
        if row.get("coverage_rule_passed") is not expected_pass:
            errors.append(f"E1 coverage decision rule is incorrect: {key}")
    return errors


def _e1_coverage_csv(rows: Any) -> str:
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return ""
    fieldnames = (
        "scenario_id",
        "event",
        "regime",
        "n",
        "replicates",
        "covered",
        "coverage",
        "wilson_95_lower",
        "wilson_95_upper",
        "mean_width",
        "target_rate",
        "nominal_coverage",
        "coverage_rule_passed",
    )
    import io

    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        if isinstance(row, Mapping):
            writer.writerow({name: row.get(name) for name in fieldnames})
    return handle.getvalue()


def _write_e1_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )


def _e1_manifest(output_dir: Path, *, generation_command: str) -> dict[str, Any]:
    filenames = ("study.json", "coverage.csv")
    files = [
        {
            "filename": filename,
            "sha256": _sha256_file(output_dir / filename),
            "bytes": (output_dir / filename).stat().st_size,
        }
        for filename in filenames
    ]
    payload: dict[str, Any] = {
        "schema_version": "cc.evals.dependence_evidence_manifest.v1",
        "generation_command": generation_command,
        "required_files": list(E1_ARTIFACT_FILENAMES),
        "files": files,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )
    payload["manifest_payload_sha256"] = hashlib.sha256(encoded).hexdigest()
    return payload


def _verify_e1_manifest(output_dir: Path, manifest: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if manifest.get("schema_version") != "cc.evals.dependence_evidence_manifest.v1":
        errors.append("E1 manifest schema_version mismatch")
        return errors
    if manifest.get("required_files") != list(E1_ARTIFACT_FILENAMES):
        errors.append("E1 manifest required_files mismatch")
    expected_payload = {
        key: value for key, value in manifest.items() if key != "manifest_payload_sha256"
    }
    encoded = json.dumps(
        expected_payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    if manifest.get("manifest_payload_sha256") != hashlib.sha256(encoded).hexdigest():
        errors.append("E1 manifest payload hash mismatch")
    files = manifest.get("files")
    if not isinstance(files, list) or len(files) != 2:
        return [*errors, "E1 manifest file records are invalid"]
    by_name = {item.get("filename"): item for item in files if isinstance(item, Mapping)}
    for filename in ("study.json", "coverage.csv"):
        record = by_name.get(filename)
        if not isinstance(record, Mapping):
            errors.append(f"E1 manifest missing {filename}")
            continue
        path = output_dir / filename
        if record.get("sha256") != _sha256_file(path):
            errors.append(f"E1 manifest hash mismatch for {filename}")
        if record.get("bytes") != path.stat().st_size:
            errors.append(f"E1 manifest byte size mismatch for {filename}")
    return errors


def main(argv: Iterable[str] | None = None) -> int:
    """CLI entry point for dependence-aware benchmark summaries."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--adapters", required=True, help="Comma-separated adapter names")
    parser.add_argument("--adapter-config", type=Path, default=None)
    parser.add_argument("--prompt-field", default="prompt")
    parser.add_argument("--response-field", default=None)
    parser.add_argument("--label-field", default="label")
    parser.add_argument("--keyword-terms", default=",".join(DEFAULT_KEYWORDS))
    parser.add_argument("--review-policy", choices=["block", "allow"], default="block")
    parser.add_argument("--delta", type=float, default=DEFAULT_DELTA)
    parser.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON)
    parser.add_argument("--run-id", default="dependence-benchmark")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(list(argv) if argv is not None else None)

    rows, dataset_hash = load_dataset(args.dataset)
    config = _load_adapter_config(args.adapter_config)
    blockers = _init_blockers(
        args.adapters,
        config,
        keyword_terms=args.keyword_terms,
        review_policy=args.review_policy,
    )
    summary = run_benchmark(
        rows,
        blockers,
        prompt_field=args.prompt_field,
        response_field=args.response_field,
        label_field=args.label_field,
        delta=args.delta,
        epsilon=args.epsilon,
        dataset_id=str(args.dataset),
        dataset_sha256=dataset_hash,
        run_id=args.run_id,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"out": str(args.out), "n_harmful": summary["n_harmful"]}, sort_keys=True))
    return 0


def _event_summary(
    query: LinearQuery,
    assumptions: AssumptionSet,
    marginals: Mapping[str, float],
    matrix: np.ndarray[Any, Any],
) -> dict[str, Any]:
    result = identified_region(query, assumptions)
    product_baseline = independent_event_probability(
        marginals, query, labels=assumptions.guardrails
    )
    if query.name == "any_guardrail_failure":
        observed = float(np.mean(np.any(matrix == 1, axis=1)))
    elif query.name == "stack_unsafe_pass":
        observed = float(np.mean(np.all(matrix == 1, axis=1)))
    else:
        raise BenchmarkInputError(f"unsupported benchmark event query: {query.name}")
    lower_check = _verify_endpoint(query, assumptions, result.lower_solution, result.lower_bound)
    upper_check = _verify_endpoint(query, assumptions, result.upper_solution, result.upper_bound)
    return {
        "assumptions_hash": result.assumptions_hash,
        "lower_bound": result.lower_bound,
        "upper_bound": result.upper_bound,
        "fh_width": fh_width(result.lower_bound, result.upper_bound),
        "observed_rate": observed,
        "fh_position": fh_position(observed, result.lower_bound, result.upper_bound),
        "product_baseline": product_baseline,
        "independence_regret": independence_regret(observed, product_baseline),
        "independence_regret_lower": independence_regret(result.lower_bound, product_baseline),
        "independence_regret_upper": independence_regret(result.upper_bound, product_baseline),
        "solver_status": result.solver_status,
        "witness_checks": {"lower": lower_check, "upper": upper_check},
    }


def _marginal_assumptions(labels: Sequence[str], marginals: Mapping[str, float]) -> AssumptionSet:
    assumptions = AssumptionSet.empty(
        labels,
        metadata={
            "failure_event_convention": FAILURE_EVENT_CONVENTION,
            "source": "cc.evals.dependence_benchmark",
            "assumption_family": "exact empirical singleton plug-in",
        },
    )
    for label in labels:
        value = float(marginals[label])
        assumptions = assumptions.with_marginal_interval(label, value, value)
    return assumptions


def _pairwise_assumptions(
    labels: Sequence[str],
    marginals: Mapping[str, float],
    pairwise_overlaps: Mapping[str, float],
) -> AssumptionSet:
    assumptions = _marginal_assumptions(labels, marginals)
    for left_index, left in enumerate(labels):
        for right_index in range(left_index + 1, len(labels)):
            right = labels[right_index]
            key = f"{left}&{right}"
            value = float(pairwise_overlaps[key])
            assumptions = assumptions.with_pairwise_joint_interval(left, right, value, value)
    return assumptions


def _verify_endpoint(
    query: LinearQuery,
    assumptions: AssumptionSet,
    distribution: np.ndarray[Any, Any],
    reported: float,
) -> bool:
    if not np.all(np.isfinite(distribution)):
        return False
    if float(np.min(distribution)) < -NUMERIC_TOL:
        return False
    if abs(float(np.sum(distribution)) - 1.0) > NUMERIC_TOL:
        return False
    if abs(float(query.coefficients @ distribution) - reported) > NUMERIC_TOL:
        return False
    for constraint in assumptions.constraints:
        lhs = float(constraint.coefficients @ distribution)
        if constraint.sense == "==" and abs(lhs - constraint.rhs) > NUMERIC_TOL:
            return False
        if constraint.sense == "<=" and lhs > constraint.rhs + NUMERIC_TOL:
            return False
        if constraint.sense == ">=" and lhs + NUMERIC_TOL < constraint.rhs:
            return False
    return True


def _verify_event_summary(
    event_name: str,
    assumption_key: str,
    summary: Mapping[str, Any],
) -> list[str]:
    errors: list[str] = []
    for key in (
        "lower_bound",
        "upper_bound",
        "fh_width",
        "observed_rate",
        "product_baseline",
        "independence_regret",
        "independence_regret_lower",
        "independence_regret_upper",
    ):
        try:
            value = float(summary[key])
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"{event_name}.{assumption_key}.{key} is invalid: {exc}")
            continue
        if not np.isfinite(value):
            errors.append(f"{event_name}.{assumption_key}.{key} must be finite")
    lower = float(summary.get("lower_bound", -1.0))
    upper = float(summary.get("upper_bound", -1.0))
    observed = float(summary.get("observed_rate", -1.0))
    if lower < -NUMERIC_TOL or upper > 1.0 + NUMERIC_TOL or lower > upper + NUMERIC_TOL:
        errors.append(f"{event_name}.{assumption_key} has invalid bounds")
    if observed < -NUMERIC_TOL or observed > 1.0 + NUMERIC_TOL:
        errors.append(f"{event_name}.{assumption_key}.observed_rate is outside [0, 1]")
    witness_checks = summary.get("witness_checks")
    if witness_checks != {"lower": True, "upper": True}:
        errors.append(f"{event_name}.{assumption_key} witness checks failed")
    return errors


def _decision_blocks(decision: Decision, review_policy: str) -> bool:
    if decision.verdict == "block":
        return True
    if decision.verdict == "allow":
        return False
    return review_policy == "block"


def _records_from_matrix(
    labels: Sequence[str],
    matrix: np.ndarray[Any, Any],
) -> list[dict[str, Any]]:
    return [
        {
            "row_index": index,
            "row_id": str(index),
            "failure_indicators": {
                label: int(matrix[index, label_index]) for label_index, label in enumerate(labels)
            },
        }
        for index in range(matrix.shape[0])
    ]


def _failure_matrix(
    matrix: Sequence[Sequence[int]] | np.ndarray[Any, Any],
    *,
    expected_width: int,
) -> np.ndarray[Any, np.dtype[np.int_]]:
    out = np.asarray(matrix, dtype=np.int_)
    if out.ndim != 2 or out.shape[0] < 1 or out.shape[1] != expected_width:
        raise BenchmarkInputError("failure matrix must be nonempty with one column per label")
    if not np.all((out == 0) | (out == 1)):
        raise BenchmarkInputError("failure matrix must contain only 0/1 indicators")
    return out


def _validate_labels(labels: Sequence[str]) -> tuple[str, ...]:
    if isinstance(labels, str):
        raise TypeError("labels must be a sequence, not a string")
    names = tuple(str(label) for label in labels)
    if not names:
        raise ValueError("labels must be nonempty")
    if any(not label.strip() for label in names):
        raise ValueError("labels must be nonempty strings")
    if len(set(names)) != len(names):
        raise ValueError("labels must be unique")
    return names


def _unique_blocker_names(blockers: Sequence[Blocker]) -> list[str]:
    names = [blocker.name for blocker in blockers]
    if len(set(names)) != len(names):
        raise BenchmarkInputError("blocker names must be unique")
    return names


def _mapping(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("value must be a mapping")
    return value


def _float_mapping(value: Any) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise TypeError("value must be a mapping")
    return {str(key): float(item) for key, item in value.items()}


def _float_list(values: Sequence[float] | np.ndarray[Any, Any]) -> list[float]:
    """Convert a numeric vector to ordinary JSON-safe floats."""

    return [float(value) for value in values]


def _load_adapter_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise BenchmarkInputError("adapter config must be a JSON object")
    return dict(payload)


def _init_blockers(
    adapters_arg: str,
    config: Mapping[str, Any],
    *,
    keyword_terms: str,
    review_policy: str,
) -> list[Blocker]:
    blockers: list[Blocker] = []
    for raw_name in adapters_arg.split(","):
        name = raw_name.strip()
        if not name:
            continue
        if name in {"keyword", "keyword_blocker"}:
            keywords = tuple(term.strip() for term in keyword_terms.split(",") if term.strip())
            blockers.append(KeywordBlockerAdapter(keywords=keywords))
            continue
        cls = ADAPTER_REGISTRY.get(name)
        if cls is None:
            raise BenchmarkInputError(f"unknown adapter: {name}")
        kwargs = config.get(name, {})
        if not isinstance(kwargs, Mapping):
            raise BenchmarkInputError(f"adapter config for {name} must be an object")
        blockers.append(AdapterBlocker(cls(**dict(kwargs)), review_policy=review_policy))
    if not blockers:
        raise BenchmarkInputError("no adapters were selected")
    return blockers


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
