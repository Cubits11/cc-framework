from __future__ import annotations

import json

import numpy as np
import pytest
from pydantic import ValidationError

from cc.evidence.extremal_scenario import (
    CONFIRMATORY_MATRIX_NOT_DEPLOYMENT_NON_CLAIM,
    FEASIBILITY_NOT_LIKELIHOOD_NON_CLAIM,
    FITTED_NOT_MODEL_TRUTH_NON_CLAIM,
    PROHIBITED_ADAPTIVE_FIELDS,
    SCENARIO_NOT_DEPLOYMENT_NON_CLAIM,
    SCENARIO_NOT_EXTERNAL_VALIDITY_NON_CLAIM,
    SCENARIO_NOT_LIFECYCLE_STATE_NON_CLAIM,
    STRESS_NOT_FORECAST_NON_CLAIM,
    ExtremalScenario,
    ScenarioKind,
    excluded_evidence_fields_from_payload,
)
from cc.kernel.cliff import cliff_certificate
from cc.kernel.frechet_classes import (
    FrechetBoundResult,
    PairwiseDependence,
    atom_matrix,
    frechet_bounds,
)
from cc.kernel.stress import BaselineDependence, StressBudget, stress_test
from cc.redteam.dependence_search import DiscoveredCliffReport, compute_dependence_metrics


def _simple_confirmatory_scenario() -> ExtremalScenario:
    return ExtremalScenario.from_confirmatory_failure_matrix(
        failures=[(0, 0), (1, 0), (0, 1), (1, 1)],
        confirmatory_ci=(0.25, 0.25),
    )


def _simple_frechet_scenario() -> ExtremalScenario:
    return ExtremalScenario.from_frechet_result(
        frechet_bounds([0.4, 0.6], event="and", return_distributions=True),
        endpoint="upper",
    )


def _simple_stress_scenario() -> ExtremalScenario:
    return ExtremalScenario.from_stress_result(
        stress_test(
            BaselineDependence(
                marginals=[0.2, 0.2],
                pairwise=[PairwiseDependence(0, 1, "joint_probability", 0.08)],
            ),
            StressBudget(0.02, metric="wasserstein"),
        )
    )


def test_frechet_builder_serializes_full_atom_table_and_feasibility() -> None:
    result = frechet_bounds([0.4, 0.6], event="and", return_distributions=True)

    scenario = ExtremalScenario.from_frechet_result(result, endpoint="upper", top_k=2)

    assert scenario.model_dump(mode="json")["schema"] == "cc.extremal_scenario.v1"
    assert scenario.kind is ScenarioKind.FRECHET_ENDPOINT
    assert scenario.epistemic_status == "extremal_feasibility_witness"
    assert "feasibility_not_likelihood" in scenario.boundary_tags
    assert "empirical_not_deployment_safety" in scenario.boundary_tags
    assert scenario.source == "frechet"
    assert scenario.source_kernel == "frechet"
    assert scenario.endpoint == "upper"
    assert scenario.guardrail_ids == ("guardrail_0", "guardrail_1")
    assert scenario.bound_value == pytest.approx(result.upper)
    assert scenario.event_probability == pytest.approx(result.upper)
    assert len(scenario.atom_table) == 4
    assert [row.failures for row in scenario.atom_table] == [
        tuple(int(item) for item in row) for row in atom_matrix(2).tolist()
    ]
    assert sum(row.probability for row in scenario.atom_table) == pytest.approx(1.0)
    assert scenario.feasibility.probability_sum == pytest.approx(1.0)
    assert scenario.feasibility.min_probability >= 0.0
    assert scenario.feasibility.max_probability <= 1.0
    assert scenario.feasibility.negative_probability_count == 0
    assert scenario.feasibility.marginal_residual_linf <= 1.0e-8
    assert scenario.feasibility.bound_residual == pytest.approx(0.0)
    assert scenario.feasibility.max_abs_residual <= 1.0e-8
    assert scenario.top_outcomes[0].probability >= scenario.top_outcomes[1].probability
    assert FEASIBILITY_NOT_LIKELIHOOD_NON_CLAIM in scenario.non_claims
    assert SCENARIO_NOT_DEPLOYMENT_NON_CLAIM in scenario.non_claims
    assert SCENARIO_NOT_EXTERNAL_VALIDITY_NON_CLAIM in scenario.non_claims
    assert SCENARIO_NOT_LIFECYCLE_STATE_NON_CLAIM in scenario.non_claims
    assert ExtremalScenario.model_validate_json(scenario.model_dump_json()) == scenario


def test_frechet_builder_records_pairwise_feasibility_residuals() -> None:
    result = frechet_bounds(
        [0.4, 0.6],
        pairwise=[PairwiseDependence(0, 1, "joint_probability", 0.25)],
        event="and",
        return_distributions=True,
    )

    scenario = ExtremalScenario.from_frechet_result(result, endpoint="lower")

    assert scenario.event_probability == pytest.approx(0.25)
    assert scenario.feasibility.pairwise_residuals["0,1"] == pytest.approx(0.0)
    assert scenario.feasibility.event_probability_residual == pytest.approx(0.0)
    assert "likely" in FEASIBILITY_NOT_LIKELIHOOD_NON_CLAIM
    assert FEASIBILITY_NOT_LIKELIHOOD_NON_CLAIM in scenario.non_claims


def test_stress_builder_uses_stressed_and_frechet_limit_distributions() -> None:
    result = stress_test(
        BaselineDependence(
            marginals=[0.2, 0.2],
            pairwise=[PairwiseDependence(0, 1, "joint_probability", 0.08)],
        ),
        StressBudget(0.02, metric="wasserstein"),
    )

    stressed = ExtremalScenario.from_stress_result(result, endpoint="stressed")
    limit = ExtremalScenario.from_stress_result(result, endpoint="frechet_limit")

    assert stressed.kind is ScenarioKind.STRESS_ENDPOINT
    assert stressed.epistemic_status == "stress_feasibility_witness"
    assert "stress_not_forecast" in stressed.boundary_tags
    assert stressed.source == "stress"
    assert stressed.source_kernel == "stress"
    assert stressed.event_probability == pytest.approx(result.stressed_risk)
    assert limit.event_probability == pytest.approx(result.frechet_upper)
    assert stressed.metadata["gap_to_frechet_limit"] == pytest.approx(result.gap_to_frechet_limit)
    assert stressed.feasibility.max_abs_residual <= 1.0e-8
    assert limit.feasibility.max_abs_residual <= 1.0e-8
    assert STRESS_NOT_FORECAST_NON_CLAIM in stressed.non_claims
    assert STRESS_NOT_FORECAST_NON_CLAIM in limit.non_claims


def test_full_atom_table_reconstructs_distribution() -> None:
    distribution = np.asarray([0.5, 0.2, 0.2, 0.1], dtype=float)
    scenario = ExtremalScenario.from_confirmatory_failure_matrix(
        failures=[
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
            (1, 0),
            (1, 0),
            (0, 1),
            (0, 1),
            (1, 1),
        ],
        confirmatory_ci=(0.1, 0.1),
    )
    reconstructed = np.asarray([row.probability for row in scenario.atom_table], dtype=float)

    assert scenario.kind is ScenarioKind.CONFIRMATORY_FAILURE_MATRIX
    assert scenario.epistemic_status == "confirmatory_empirical_summary"
    assert "confirmatory_not_external_validity" in scenario.boundary_tags
    assert "empirical_not_deployment_safety" in scenario.boundary_tags
    assert reconstructed == pytest.approx(distribution)
    assert scenario.event_probability == pytest.approx(0.1)
    assert CONFIRMATORY_MATRIX_NOT_DEPLOYMENT_NON_CLAIM in scenario.non_claims
    assert FITTED_NOT_MODEL_TRUTH_NON_CLAIM in scenario.non_claims


def test_atom_probabilities_must_sum_to_one_within_tolerance() -> None:
    result = FrechetBoundResult(
        lower=0.0,
        upper=0.0,
        event="and",
        marginals=np.asarray([0.5, 0.5], dtype=float),
        pairwise=(),
        lower_distribution=np.asarray([0.2, 0.2, 0.2, 0.2], dtype=float),
        upper_distribution=np.asarray([0.2, 0.2, 0.2, 0.2], dtype=float),
    )

    with pytest.raises(ValueError, match="sum to 1"):
        ExtremalScenario.from_frechet_result(result, endpoint="upper")


def test_negative_atom_probabilities_are_rejected() -> None:
    result = FrechetBoundResult(
        lower=0.0,
        upper=0.0,
        event="and",
        marginals=np.asarray([0.5, 0.5], dtype=float),
        pairwise=(),
        lower_distribution=np.asarray([1.1, -0.1, 0.0, 0.0], dtype=float),
        upper_distribution=np.asarray([1.1, -0.1, 0.0, 0.0], dtype=float),
    )

    with pytest.raises(ValueError, match="nonnegative"):
        ExtremalScenario.from_frechet_result(result, endpoint="upper")


def test_guardrail_ids_must_match_atom_dimensionality() -> None:
    scenario = _simple_confirmatory_scenario()
    payload = scenario.model_dump(mode="json")
    payload["guardrail_ids"] = ["only-one"]

    with pytest.raises(ValidationError, match="guardrail_ids"):
        ExtremalScenario.model_validate(payload)


def test_top_outcomes_are_sorted_deterministically() -> None:
    scenario = ExtremalScenario.from_confirmatory_failure_matrix(
        failures=[(0, 0), (1, 0), (0, 1), (1, 1)],
        confirmatory_ci=(0.25, 0.25),
        top_k=3,
    )

    assert [row.atom_index for row in scenario.top_outcomes] == [0, 1, 2]

    payload = scenario.model_dump(mode="json")
    payload["top_outcomes"] = list(reversed(payload["top_outcomes"]))

    with pytest.raises(ValidationError, match="top_outcomes"):
        ExtremalScenario.model_validate(payload)


def test_top_k_does_not_drop_full_atom_table_needed_for_audit() -> None:
    scenario = ExtremalScenario.from_confirmatory_failure_matrix(
        failures=[(0, 0), (1, 0), (0, 1), (1, 1)],
        confirmatory_ci=(0.25, 0.25),
        top_k=1,
    )

    assert len(scenario.top_outcomes) == 1
    assert len(scenario.atom_table) == 4


def test_frechet_endpoints_preserve_kernel_returned_distributions() -> None:
    result = frechet_bounds(
        [0.4, 0.6],
        pairwise=[PairwiseDependence(0, 1, "joint_probability", 0.25)],
        event="and",
        return_distributions=True,
    )

    lower = ExtremalScenario.from_frechet_result(result, endpoint="lower")
    upper = ExtremalScenario.from_frechet_result(result, endpoint="upper")

    assert [row.probability for row in lower.atom_table] == pytest.approx(result.lower_distribution)
    assert [row.probability for row in upper.atom_table] == pytest.approx(result.upper_distribution)


def test_adaptive_ci_fields_are_excluded_from_empirical_scenario_source_payload() -> None:
    source_payload = {
        "adaptive_report": {
            "certificate_ci": [0.8, 1.0],
            "exploratory_ci": [0.8, 1.0],
            "confirmatory_certificate_ci": [0.2, 0.4],
        }
    }

    scenario = ExtremalScenario.from_confirmatory_failure_matrix(
        failures=[(1, 1), (1, 0), (0, 1), (0, 0)],
        confirmatory_ci=(0.2, 0.4),
        source_payload=source_payload,
    )

    exclusions = excluded_evidence_fields_from_payload(source_payload)
    dumped = scenario.model_dump(mode="json")

    assert exclusions
    assert {item.field_name for item in scenario.excluded_evidence_fields} >= {
        "certificate_ci",
        "exploratory_ci",
    }
    assert scenario.excluded_evidence_fields[0].status == "excluded"
    assert scenario.excluded_evidence_fields[0].reason == (
        "Adaptive/post-selection interval is exploratory and is not surfaced as confirmatory evidence."
    )
    assert "certificate_ci" not in json.dumps(scenario.metadata, sort_keys=True)
    assert scenario.metadata["confirmatory_ci"] == [0.2, 0.4]
    assert {
        "field_name": "certificate_ci",
        "reason": (
            "Adaptive/post-selection interval is exploratory and is not surfaced as "
            "confirmatory evidence."
        ),
        "status": "excluded",
        "path": "$.adaptive_report.certificate_ci",
        "replacement": "confirmatory_ci",
    } in dumped["excluded_evidence_fields"]


def test_prohibited_adaptive_fields_constant_covers_known_exploratory_intervals() -> None:
    assert {
        "certificate_ci",
        "exploratory_ci",
        "adaptive_search_ci",
        "non_confirmatory_ci",
        "exploratory_certificate_ci",
    } <= PROHIBITED_ADAPTIVE_FIELDS


def test_discovered_cliff_report_interval_is_only_carried_as_exclusion() -> None:
    metrics = compute_dependence_metrics([(1, 1), (1, 0), (0, 1), (0, 0)])
    report = DiscoveredCliffReport(
        run_id="adaptive-report",
        created_at="2026-01-01T00:00:00+00:00",
        seed=7,
        baseline_metrics=metrics,
        discovered_metrics=metrics,
        tau_shift=0.0,
        tail_dependence_shift=0.0,
        objective_value=0.0,
        certificate=cliff_certificate(
            {"lambda_any": 0.5},
            {"lambda_any": (0.4, 0.6), "confidence_level": 0.95},
            critical_value=0.2,
        ),
        exploratory_ci=(0.4, 0.6),
    )
    source_payload = report.to_dict()
    source_payload["certificate_ci"] = [0.4, 0.6]

    scenario = ExtremalScenario.from_confirmatory_failure_matrix(
        failures=[(1, 1), (1, 0), (0, 1), (0, 0)],
        confirmatory_ci=(0.5, 0.5),
        source_payload=source_payload,
    )

    serialized = scenario.model_dump(mode="json")

    assert "certificate_ci" not in json.dumps(serialized["metadata"], sort_keys=True)
    assert any(
        item["field_name"] == "certificate_ci" and item["status"] == "excluded"
        for item in serialized["excluded_evidence_fields"]
    )
    assert any(
        item["field_name"] == "exploratory_ci" and item["status"] == "excluded"
        for item in serialized["excluded_evidence_fields"]
    )


def test_scenario_rejects_lifecycle_and_deployment_status_fields() -> None:
    scenario = _simple_confirmatory_scenario()

    payload = scenario.model_dump(mode="json")
    payload["claim_state"] = "supported"

    with pytest.raises(ValidationError, match="live claim/lifecycle/deployment fields"):
        ExtremalScenario.model_validate(payload)

    payload = scenario.model_dump(mode="json")
    payload["deployment_safe"] = True

    with pytest.raises(ValidationError, match="live claim/lifecycle/deployment fields"):
        ExtremalScenario.model_validate(payload)

    payload = scenario.model_dump(mode="json")
    payload["status"] = "pass"

    with pytest.raises(ValidationError, match="live claim/lifecycle/deployment fields"):
        ExtremalScenario.model_validate(payload)


def test_scenario_metadata_cannot_smuggle_overclaim_fields() -> None:
    scenario = _simple_confirmatory_scenario()

    payload = scenario.model_dump(mode="json")
    payload["metadata"]["deployment_safety_support"] = True

    with pytest.raises(ValidationError, match="forbidden overclaim field"):
        ExtremalScenario.model_validate(payload)

    payload = scenario.model_dump(mode="json")
    payload["metadata"]["model_truth_claim"] = "the fitted model is true"

    with pytest.raises(ValidationError, match="forbidden overclaim field"):
        ExtremalScenario.model_validate(payload)

    payload = scenario.model_dump(mode="json")
    payload["metadata"]["likelihood_claim"] = "this endpoint is likely"

    with pytest.raises(ValidationError, match="forbidden overclaim field"):
        ExtremalScenario.model_validate(payload)


def test_scenario_requires_mandatory_non_claims() -> None:
    scenario = _simple_frechet_scenario()
    payload = scenario.model_dump(mode="json")
    payload["non_claims"] = ["too weak"]

    with pytest.raises(ValidationError, match="mandatory non-claims"):
        ExtremalScenario.model_validate(payload)


def test_scenario_kind_epistemic_status_and_boundary_tags_must_match() -> None:
    scenario = _simple_stress_scenario()

    payload = scenario.model_dump(mode="json")
    payload["epistemic_status"] = "extremal_feasibility_witness"

    with pytest.raises(ValidationError, match="stress_endpoint must use"):
        ExtremalScenario.model_validate(payload)

    payload = scenario.model_dump(mode="json")
    payload["boundary_tags"] = ["empirical_not_deployment_safety"]

    with pytest.raises(ValidationError, match="stress_not_forecast"):
        ExtremalScenario.model_validate(payload)


def test_frechet_scenario_requires_feasibility_not_likelihood_boundary_tag() -> None:
    scenario = _simple_frechet_scenario()
    payload = scenario.model_dump(mode="json")
    payload["boundary_tags"] = ["empirical_not_deployment_safety"]

    with pytest.raises(ValidationError, match="feasibility_not_likelihood"):
        ExtremalScenario.model_validate(payload)


def test_confirmatory_matrix_requires_confirmatory_boundary_tag() -> None:
    scenario = _simple_confirmatory_scenario()
    payload = scenario.model_dump(mode="json")
    payload["boundary_tags"] = ["empirical_not_deployment_safety", "fitted_not_model_truth"]

    with pytest.raises(ValidationError, match="confirmatory_not_external_validity"):
        ExtremalScenario.model_validate(payload)


def test_confirmatory_ci_must_be_ordered_and_bounded() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        ExtremalScenario.from_confirmatory_failure_matrix(
            failures=[(0, 0), (1, 0), (0, 1), (1, 1)],
            confirmatory_ci=(-0.1, 0.5),
        )

    with pytest.raises(ValueError, match="lower cannot exceed upper"):
        ExtremalScenario.from_confirmatory_failure_matrix(
            failures=[(0, 0), (1, 0), (0, 1), (1, 1)],
            confirmatory_ci=(0.7, 0.2),
        )


def test_canonical_hash_is_stable_and_sensitive_to_boundary_content() -> None:
    scenario = _simple_frechet_scenario()

    same = ExtremalScenario.model_validate_json(scenario.model_dump_json())

    assert same.canonical_hash() == scenario.canonical_hash()

    payload = scenario.model_dump(mode="json")
    payload["narrative"] = payload["narrative"] + " changed"
    changed = ExtremalScenario.model_validate(payload)

    assert changed.canonical_hash() != scenario.canonical_hash()


def test_canonical_hash_is_sensitive_to_non_claim_boundary_content() -> None:
    scenario = _simple_frechet_scenario()
    payload = scenario.model_dump(mode="json")
    payload["non_claims"] = list(payload["non_claims"]) + [
        "Additional scoped caveat for a downstream verifier."
    ]

    changed = ExtremalScenario.model_validate(payload)

    assert changed.canonical_hash() != scenario.canonical_hash()


def test_feasible_endpoint_narrative_carries_not_likelihood_boundary() -> None:
    scenario = ExtremalScenario.from_frechet_result(
        frechet_bounds([0.3, 0.7], event="and", return_distributions=True),
        endpoint="upper",
    )

    serialized = json.dumps(scenario.model_dump(mode="json"), sort_keys=True).lower()

    assert "not a likelihood claim" in serialized
    assert "does not prove the endpoint world is likely" in serialized


def test_stress_scenario_narrative_carries_not_forecast_boundary() -> None:
    scenario = _simple_stress_scenario()
    serialized = json.dumps(scenario.model_dump(mode="json"), sort_keys=True).lower()

    assert "not a forecast" in serialized
    assert "stress scenario is a counterfactual" in serialized


def test_confirmatory_matrix_narrative_carries_not_deployment_boundary() -> None:
    scenario = _simple_confirmatory_scenario()
    serialized = json.dumps(scenario.model_dump(mode="json"), sort_keys=True).lower()

    assert "does not certify deployment safety" in serialized


def test_event_probability_must_match_atom_table_event_mass() -> None:
    scenario = _simple_confirmatory_scenario()
    payload = scenario.model_dump(mode="json")
    payload["event_probability"] = 0.99

    with pytest.raises(ValidationError, match="event_probability must match"):
        ExtremalScenario.model_validate(payload)


def test_scenario_identifiers_cannot_reuse_lifecycle_state_names() -> None:
    scenario = _simple_frechet_scenario()
    payload = scenario.model_dump(mode="json")
    payload["scenario_id"] = "supported"

    with pytest.raises(ValidationError, match="lifecycle state names"):
        ExtremalScenario.model_validate(payload)

    payload = scenario.model_dump(mode="json")
    payload["endpoint"] = "revoked"

    with pytest.raises(ValidationError, match="lifecycle state names"):
        ExtremalScenario.model_validate(payload)


def test_metadata_is_sorted_and_jsonable_after_validation() -> None:
    scenario = _simple_frechet_scenario()
    payload = scenario.model_dump(mode="json")
    payload["metadata"] = {"z": 1, "a": {"nested": np.float64(1.25)}}

    validated = ExtremalScenario.model_validate(payload)

    assert list(validated.metadata) == ["a", "z"]
    assert validated.metadata["a"] == {"nested": 1.25}
