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
    bernoulli_rate_count,
    simultaneous_bernoulli_radius,
    simultaneous_sample_size,
)
from cc.kernel.sensitivity import AssumptionSet, LinearQuery, identified_region

SCHEMA_VERSION = "cc.evals.dependence_benchmark.v1"
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

__all__ = [
    "SCHEMA_VERSION",
    "BenchmarkInputError",
    "Blocker",
    "build_summary_from_failure_matrix",
    "failure_matrix_from_rows",
    "load_dataset",
    "main",
    "run_benchmark",
    "verify_benchmark_summary",
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
