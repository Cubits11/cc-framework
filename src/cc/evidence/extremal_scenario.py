"""Serializable extremal scenarios from Frechet and stress kernels."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from cc.kernel.frechet_classes import (
    FrechetBoundResult,
    atom_matrix,
    distribution_moments,
    event_probability,
)
from cc.kernel.stress import StressTestResult

EXTREMAL_SCENARIO_SCHEMA_VERSION = "cc.extremal_scenario.v1"
_DISTRIBUTION_TOL = 1.0e-8
_NEGATIVE_PROBABILITY_TOL = 1.0e-10
_DEFAULT_NON_CLAIMS = (
    "This scenario is an extremal or fitted evidence artifact, not a deployment approval.",
    "This scenario does not certify production safety or legal compliance.",
    "This scenario does not generalize beyond its stated marginals, constraints, and sample scope.",
    "An extremal_scenario artifact does not prove the endpoint scenario is likely; it proves or "
    "records a feasible endpoint/fitted scenario under the stated assumptions.",
)


class ExtremalModel(BaseModel):
    """Strict base model for extremal scenario artifacts."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        populate_by_name=True,
        serialize_by_alias=True,
        strict=True,
    )


class ScenarioKind(str, Enum):
    """Kind of endpoint or fitted scenario represented by the evidence object."""

    FRECHET_ENDPOINT = "frechet_endpoint"
    STRESS_ENDPOINT = "stress_endpoint"
    CONFIRMATORY_FAILURE_MATRIX = "confirmatory_failure_matrix"


class GuardrailOutcome(ExtremalModel):
    """One atom in the finite binary guardrail outcome table."""

    atom_index: int = Field(ge=0)
    failures: tuple[int, ...] = Field(min_length=1)
    probability: float = Field(ge=0.0, le=1.0)
    event_occurs: bool

    @field_validator("failures")
    @classmethod
    def _binary_failures(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if any(item not in (0, 1) for item in value):
            raise ValueError("failures must be binary indicators")
        return value


class ScenarioFeasibility(ExtremalModel):
    """Residual diagnostics against the scenario's defining constraints."""

    probability_sum: float
    min_probability: float
    max_probability: float
    negative_probability_count: int = Field(ge=0)
    total_probability_residual: float
    marginal_residuals: tuple[float, ...]
    marginal_residual_linf: float
    pairwise_residuals: dict[str, float] = Field(default_factory=dict)
    event_probability_residual: float | None = None
    bound_residual: float | None = None
    objective_residual: float | None = None
    max_abs_residual: float
    is_feasible: bool = True

    @model_validator(mode="after")
    def _require_finite_residuals(self) -> ScenarioFeasibility:
        values = [
            self.probability_sum,
            self.min_probability,
            self.max_probability,
            self.total_probability_residual,
            *self.marginal_residuals,
            self.marginal_residual_linf,
            *self.pairwise_residuals.values(),
        ]
        if self.event_probability_residual is not None:
            values.append(self.event_probability_residual)
        if self.bound_residual is not None:
            values.append(self.bound_residual)
        if self.objective_residual is not None:
            values.append(self.objective_residual)
        values.append(self.max_abs_residual)
        if any(not math.isfinite(float(value)) for value in values):
            raise ValueError("feasibility residuals must be finite")
        return self


class ExcludedEvidenceField(ExtremalModel):
    """Field intentionally excluded from scenario evidence."""

    field_name: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    status: Literal["excluded"] = "excluded"
    path: str = Field(min_length=1)
    replacement: str | None = None


class ExtremalScenario(ExtremalModel):
    """Full finite-atom extremal scenario with narrative and non-claims."""

    schema_: Literal["cc.extremal_scenario.v1"] = Field(
        default=EXTREMAL_SCENARIO_SCHEMA_VERSION,
        alias="schema",
    )
    schema_version: Literal["cc.extremal_scenario.v1"] = EXTREMAL_SCENARIO_SCHEMA_VERSION
    scenario_id: str = Field(min_length=1)
    kind: ScenarioKind
    source: str = Field(min_length=1)
    source_kernel: str = Field(min_length=1)
    source_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    guardrail_ids: tuple[str, ...] = Field(min_length=1)
    endpoint: str = Field(min_length=1)
    event: Literal["and", "or"]
    event_definition: str | None = None
    objective: str | None = None
    bound_value: float | None = None
    n_guardrails: int = Field(ge=1)
    event_probability: float = Field(ge=0.0, le=1.0)
    atom_table: tuple[GuardrailOutcome, ...]
    top_outcomes: tuple[GuardrailOutcome, ...]
    feasibility: ScenarioFeasibility
    narrative: str = Field(min_length=1)
    non_claims: tuple[str, ...] = Field(default_factory=lambda: _DEFAULT_NON_CLAIMS)
    excluded_evidence_fields: tuple[ExcludedEvidenceField, ...] = Field(default_factory=tuple)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_atom_table(self) -> ExtremalScenario:
        if len(self.guardrail_ids) != self.n_guardrails:
            raise ValueError("guardrail_ids length must match n_guardrails")
        expected = 1 << self.n_guardrails
        if len(self.atom_table) != expected:
            raise ValueError(
                f"atom_table length must be 2**n_guardrails ({expected}), got {len(self.atom_table)}"
            )
        if any(len(outcome.failures) != self.n_guardrails for outcome in self.atom_table):
            raise ValueError("all atom outcomes must match n_guardrails")
        if [outcome.atom_index for outcome in self.atom_table] != list(range(expected)):
            raise ValueError("atom_table must be ordered by atom_index")
        total = sum(outcome.probability for outcome in self.atom_table)
        if not math.isclose(total, 1.0, abs_tol=_DISTRIBUTION_TOL):
            raise ValueError(f"atom_table probabilities must sum to 1, got {total}")
        if not self.top_outcomes:
            raise ValueError("top_outcomes must contain at least one outcome")
        if any(outcome not in self.atom_table for outcome in self.top_outcomes):
            raise ValueError("top_outcomes must be drawn from atom_table")
        expected_top = _top_outcomes(self.atom_table, top_k=len(self.top_outcomes))
        if self.top_outcomes != expected_top:
            raise ValueError("top_outcomes must be sorted deterministically from atom_table")
        if self.event_definition is None and self.objective is None:
            raise ValueError("scenario must include event_definition or objective")
        return self

    @classmethod
    def from_frechet_result(
        cls,
        result: FrechetBoundResult,
        *,
        endpoint: Literal["lower", "upper"] = "upper",
        scenario_id: str | None = None,
        source_hash: str | None = None,
        narrative: str | None = None,
        top_k: int = 5,
    ) -> ExtremalScenario:
        """Build a scenario from a Frechet endpoint distribution."""

        distribution = (
            result.upper_distribution if endpoint == "upper" else result.lower_distribution
        )
        if distribution is None:
            raise ValueError("Frechet result does not include endpoint distributions")
        target = result.upper if endpoint == "upper" else result.lower
        n_guardrails = int(result.marginals.size)
        feasibility = _feasibility(
            distribution,
            event=result.event,
            expected_marginals=tuple(float(item) for item in result.marginals),
            expected_event_probability=float(target),
            pairwise_targets={
                f"{constraint.i},{constraint.j}": float(constraint.joint_probability)
                for constraint in result.pairwise
            },
        )
        event_prob = event_probability(distribution, event=result.event, n_events=n_guardrails)
        atom_table = _atom_table(distribution, event=result.event, n_events=n_guardrails)
        return cls(
            scenario_id=scenario_id
            or _scenario_id(
                "frechet",
                endpoint,
                {
                    "marginals": [float(item) for item in result.marginals],
                    "event": result.event,
                    "endpoint": endpoint,
                    "distribution": [float(item) for item in distribution],
                },
            ),
            kind=ScenarioKind.FRECHET_ENDPOINT,
            source="frechet",
            source_kernel="frechet",
            source_hash=source_hash,
            guardrail_ids=_default_guardrail_ids(n_guardrails),
            endpoint=endpoint,
            event=result.event,
            event_definition=(
                f"{result.event.upper()} composed binary guardrail-failure event "
                "under supplied marginals and side constraints."
            ),
            bound_value=float(target),
            n_guardrails=n_guardrails,
            event_probability=float(event_prob),
            atom_table=atom_table,
            top_outcomes=_top_outcomes(atom_table, top_k=top_k),
            feasibility=feasibility,
            narrative=narrative
            or (
                f"Frechet {endpoint} endpoint for the {result.event.upper()} composed "
                "failure event under the supplied marginals and side constraints."
            ),
            metadata={
                "frechet_lower": float(result.lower),
                "frechet_upper": float(result.upper),
                "frechet_width": float(result.width),
                "atom_order": "little_endian",
            },
        )

    @classmethod
    def from_stress_result(
        cls,
        result: StressTestResult,
        *,
        endpoint: Literal["stressed", "frechet_limit"] = "stressed",
        scenario_id: str | None = None,
        source_hash: str | None = None,
        narrative: str | None = None,
        top_k: int = 5,
    ) -> ExtremalScenario:
        """Build a scenario from a stress-test distribution."""

        if endpoint == "stressed":
            distribution = result.stressed_distribution
            target = result.stressed_risk
        else:
            distribution = result.frechet_limit_distribution
            target = result.frechet_upper
        n_guardrails = int(result.marginals.size)
        feasibility = _feasibility(
            distribution,
            event=result.event,
            expected_marginals=tuple(float(item) for item in result.marginals),
            expected_event_probability=float(target),
        )
        atom_table = _atom_table(distribution, event=result.event, n_events=n_guardrails)
        return cls(
            scenario_id=scenario_id
            or _scenario_id(
                "stress",
                endpoint,
                {
                    "event": result.event,
                    "endpoint": endpoint,
                    "distribution": [float(item) for item in distribution],
                    "budget": _jsonable(result.stress_budget),
                },
            ),
            kind=ScenarioKind.STRESS_ENDPOINT,
            source="stress",
            source_kernel="stress",
            source_hash=source_hash,
            guardrail_ids=_default_guardrail_ids(n_guardrails),
            endpoint=endpoint,
            event=result.event,
            event_definition=(
                f"{result.event.upper()} composed binary guardrail-failure event under stress."
            ),
            objective="Maximize endpoint event probability under the fixed-marginal stress budget.",
            bound_value=float(target),
            n_guardrails=n_guardrails,
            event_probability=float(target),
            atom_table=atom_table,
            top_outcomes=_top_outcomes(atom_table, top_k=top_k),
            feasibility=feasibility,
            narrative=narrative
            or (
                f"Stress {endpoint} scenario for the {result.event.upper()} composed "
                "failure event under the fixed-marginal stress budget."
            ),
            metadata={
                "baseline_risk": float(result.baseline_risk),
                "stressed_risk": float(result.stressed_risk),
                "risk_increase": float(result.risk_increase),
                "realized_distance": float(result.realized_distance),
                "frechet_lower": float(result.frechet_lower),
                "frechet_upper": float(result.frechet_upper),
                "gap_to_frechet_limit": float(result.gap_to_frechet_limit),
                "stress_budget": _jsonable(result.stress_budget),
                "method": result.method,
                "atom_order": "little_endian",
            },
        )

    @classmethod
    def from_confirmatory_failure_matrix(
        cls,
        failures: Sequence[Sequence[int | bool]],
        *,
        confirmatory_ci: tuple[float, float],
        event: Literal["and", "or"] = "and",
        scenario_id: str | None = None,
        source_hash: str | None = None,
        source_payload: Mapping[str, Any] | None = None,
        narrative: str | None = None,
        top_k: int = 5,
    ) -> ExtremalScenario:
        """Build empirical scenario evidence from a non-adaptive failure matrix."""

        distribution, n_guardrails = _distribution_from_failure_matrix(failures)
        event_prob = event_probability(distribution, event=event, n_events=n_guardrails)
        atom_table = _atom_table(distribution, event=event, n_events=n_guardrails)
        exclusions = excluded_evidence_fields_from_payload(source_payload or {})
        return cls(
            scenario_id=scenario_id
            or _scenario_id(
                "empirical_confirmatory",
                "failure_matrix",
                {
                    "event": event,
                    "distribution": [float(item) for item in distribution],
                    "confirmatory_ci": list(confirmatory_ci),
                },
            ),
            kind=ScenarioKind.CONFIRMATORY_FAILURE_MATRIX,
            source="empirical_confirmatory",
            source_kernel="empirical_confirmatory",
            source_hash=source_hash,
            guardrail_ids=_default_guardrail_ids(n_guardrails),
            endpoint="confirmatory_failure_matrix",
            event=event,
            event_definition=(
                f"{event.upper()} composed binary guardrail-failure event from a non-adaptive "
                "confirmatory failure matrix."
            ),
            bound_value=float(event_prob),
            n_guardrails=n_guardrails,
            event_probability=float(event_prob),
            atom_table=atom_table,
            top_outcomes=_top_outcomes(atom_table, top_k=top_k),
            feasibility=_feasibility(distribution, event=event),
            narrative=narrative
            or "Confirmatory empirical failure-matrix scenario from a non-adaptive sample.",
            excluded_evidence_fields=exclusions,
            metadata={
                "confirmatory_ci": [float(confirmatory_ci[0]), float(confirmatory_ci[1])],
                "n_inputs": len(failures),
                "atom_order": "little_endian",
            },
        )


def excluded_evidence_fields_from_payload(
    payload: Mapping[str, Any],
) -> tuple[ExcludedEvidenceField, ...]:
    """Identify adaptive fields that must not feed fitted empirical scenarios."""

    excluded: list[ExcludedEvidenceField] = []
    for path, _value in _walk(payload):
        leaf = path.rsplit(".", 1)[-1]
        if leaf in {
            "certificate_ci",
            "exploratory_ci",
            "adaptive_search_ci",
            "non_confirmatory_ci",
            "exploratory_certificate_ci",
        }:
            excluded.append(
                ExcludedEvidenceField(
                    field_name=leaf,
                    path=path,
                    reason=(
                        "Adaptive/post-selection interval is exploratory and is not surfaced "
                        "as confirmatory evidence."
                    ),
                    replacement="confirmatory_ci",
                )
            )
    return tuple(excluded)


def _atom_table(
    distribution: ArrayLike,
    *,
    event: Literal["and", "or"],
    n_events: int,
) -> tuple[GuardrailOutcome, ...]:
    pmf = _as_distribution(distribution, n_events)
    states = atom_matrix(n_events)
    rows: list[GuardrailOutcome] = []
    for idx, (state, probability) in enumerate(zip(states, pmf, strict=True)):
        failures = tuple(int(item) for item in state.tolist())
        event_occurs = all(failures) if event == "and" else any(failures)
        rows.append(
            GuardrailOutcome(
                atom_index=idx,
                failures=failures,
                probability=float(probability),
                event_occurs=bool(event_occurs),
            )
        )
    return tuple(rows)


def _top_outcomes(
    atom_table: Sequence[GuardrailOutcome],
    *,
    top_k: int,
) -> tuple[GuardrailOutcome, ...]:
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    rows = sorted(atom_table, key=lambda row: (-row.probability, row.atom_index))
    return tuple(rows[: min(top_k, len(rows))])


def _feasibility(
    distribution: ArrayLike,
    *,
    event: Literal["and", "or"],
    expected_marginals: Sequence[float] | None = None,
    expected_event_probability: float | None = None,
    pairwise_targets: Mapping[str, float] | None = None,
) -> ScenarioFeasibility:
    raw_pmf = _raw_distribution(distribution, None)
    pmf = _as_distribution(distribution, None)
    n_events = int(math.log2(pmf.size))
    probability_sum = float(np.sum(raw_pmf))
    total_residual = float(probability_sum - 1.0)
    min_probability = float(np.min(raw_pmf))
    max_probability = float(np.max(raw_pmf))
    negative_probability_count = int(np.sum(raw_pmf < 0.0))
    marginals, pairwise = distribution_moments(pmf, n_events)
    marginal_residuals: tuple[float, ...]
    if expected_marginals is None:
        marginal_residuals = tuple(0.0 for _ in range(n_events))
    else:
        expected = np.asarray(expected_marginals, dtype=float)
        if expected.shape != marginals.shape:
            raise ValueError("expected_marginals length must match distribution dimension")
        marginal_residuals = tuple(float(item) for item in (marginals - expected))
    marginal_residual_linf = (
        max(abs(float(item)) for item in marginal_residuals) if marginal_residuals else 0.0
    )

    pairwise_residuals: dict[str, float] = {}
    for key, expected_value in (pairwise_targets or {}).items():
        i_raw, j_raw = key.split(",", 1)
        i = int(i_raw)
        j = int(j_raw)
        pairwise_residuals[key] = float(pairwise[i, j] - float(expected_value))

    event_residual = None
    if expected_event_probability is not None:
        event_residual = float(
            event_probability(pmf, event=event, n_events=n_events) - expected_event_probability
        )
    residuals = [total_residual, *marginal_residuals, *pairwise_residuals.values()]
    if event_residual is not None:
        residuals.append(event_residual)
    max_abs_residual = max(abs(float(item)) for item in residuals) if residuals else 0.0
    return ScenarioFeasibility(
        probability_sum=probability_sum,
        min_probability=min_probability,
        max_probability=max_probability,
        negative_probability_count=negative_probability_count,
        total_probability_residual=total_residual,
        marginal_residuals=marginal_residuals,
        marginal_residual_linf=marginal_residual_linf,
        pairwise_residuals=pairwise_residuals,
        event_probability_residual=event_residual,
        bound_residual=event_residual,
        objective_residual=event_residual,
        max_abs_residual=max_abs_residual,
        is_feasible=negative_probability_count == 0 and max_abs_residual <= _DISTRIBUTION_TOL,
    )


def _distribution_from_failure_matrix(
    failures: Sequence[Sequence[int | bool]],
) -> tuple[NDArray[np.float64], int]:
    arr = np.asarray(failures, dtype=int)
    if arr.ndim != 2 or arr.shape[0] < 1 or arr.shape[1] < 2:
        raise ValueError("failures must be a non-empty 2D matrix with at least two guardrails")
    if not np.all((arr == 0) | (arr == 1)):
        raise ValueError("failures must be binary")
    n_events = int(arr.shape[1])
    weights = 1 << np.arange(n_events, dtype=int)
    atom_ids = arr @ weights
    counts = np.bincount(atom_ids, minlength=1 << n_events).astype(float)
    return counts / float(np.sum(counts)), n_events


def _as_distribution(
    distribution: ArrayLike,
    n_events: int | None,
) -> NDArray[np.float64]:
    pmf = _raw_distribution(distribution, n_events)
    total = float(np.sum(pmf))
    return cast(NDArray[np.float64], np.clip(pmf, 0.0, 1.0) / total)


def _raw_distribution(
    distribution: ArrayLike,
    n_events: int | None,
) -> NDArray[np.float64]:
    pmf = np.asarray(distribution, dtype=float)
    if pmf.ndim != 1:
        raise ValueError("distribution must be one-dimensional")
    if not np.all(np.isfinite(pmf)):
        raise ValueError("distribution must be finite")
    if np.any(pmf < -_NEGATIVE_PROBABILITY_TOL):
        raise ValueError("distribution entries must be nonnegative")
    if n_events is None:
        if pmf.size == 0 or pmf.size & (pmf.size - 1):
            raise ValueError("distribution length must be a positive power of two")
        n_events = int(math.log2(pmf.size))
    if pmf.size != 1 << int(n_events):
        raise ValueError("distribution length must equal 2**n_events")
    total = float(np.sum(pmf))
    if not math.isclose(total, 1.0, abs_tol=_DISTRIBUTION_TOL):
        raise ValueError(f"distribution must sum to 1, got {total}")
    return cast(NDArray[np.float64], pmf)


def _default_guardrail_ids(n_guardrails: int) -> tuple[str, ...]:
    return tuple(f"guardrail_{idx}" for idx in range(n_guardrails))


def _scenario_id(source_kernel: str, endpoint: str, payload: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(
        json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return f"scenario-{source_kernel}-{endpoint}-{digest[:12]}"


def _walk(value: Any, path: str = "$") -> list[tuple[str, Any]]:
    items: list[tuple[str, Any]] = [(path, value)]
    if isinstance(value, Mapping):
        for key, child in value.items():
            items.extend(_walk(child, f"{path}.{key}"))
    elif isinstance(value, list):
        for idx, child in enumerate(value):
            items.extend(_walk(child, f"{path}[{idx}]"))
    return items


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


__all__ = [
    "EXTREMAL_SCENARIO_SCHEMA_VERSION",
    "ExcludedEvidenceField",
    "ExtremalScenario",
    "GuardrailOutcome",
    "ScenarioFeasibility",
    "ScenarioKind",
    "excluded_evidence_fields_from_payload",
]
