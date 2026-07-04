from __future__ import annotations

import json

import numpy as np
import pytest

from cc.evidence.extremal_scenario import (
    ExtremalScenario,
    excluded_evidence_fields_from_payload,
)
from cc.kernel.frechet_classes import PairwiseDependence, atom_matrix, frechet_bounds
from cc.kernel.stress import BaselineDependence, StressBudget, stress_test


def test_frechet_builder_serializes_full_atom_table_and_feasibility() -> None:
    result = frechet_bounds([0.4, 0.6], event="and", return_distributions=True)

    scenario = ExtremalScenario.from_frechet_result(result, endpoint="upper", top_k=2)

    assert scenario.source_kernel == "frechet"
    assert scenario.endpoint == "upper"
    assert scenario.event_probability == pytest.approx(result.upper)
    assert len(scenario.atom_table) == 4
    assert [row.failures for row in scenario.atom_table] == [
        tuple(int(item) for item in row) for row in atom_matrix(2).tolist()
    ]
    assert sum(row.probability for row in scenario.atom_table) == pytest.approx(1.0)
    assert scenario.feasibility.max_abs_residual <= 1.0e-8
    assert scenario.top_outcomes[0].probability >= scenario.top_outcomes[1].probability
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

    assert stressed.source_kernel == "stress"
    assert stressed.event_probability == pytest.approx(result.stressed_risk)
    assert limit.event_probability == pytest.approx(result.frechet_upper)
    assert stressed.metadata["gap_to_frechet_limit"] == pytest.approx(result.gap_to_frechet_limit)
    assert stressed.feasibility.max_abs_residual <= 1.0e-8
    assert limit.feasibility.max_abs_residual <= 1.0e-8


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

    assert reconstructed == pytest.approx(distribution)
    assert scenario.event_probability == pytest.approx(0.1)


def test_certificate_ci_is_excluded_from_empirical_scenario_source_payload() -> None:
    source_payload = {
        "adaptive_report": {
            "certificate_ci": [0.8, 1.0],
            "confirmatory_certificate_ci": [0.2, 0.4],
        }
    }

    scenario = ExtremalScenario.from_confirmatory_failure_matrix(
        failures=[(1, 1), (1, 0), (0, 1), (0, 0)],
        confirmatory_ci=(0.2, 0.4),
        source_payload=source_payload,
    )

    assert excluded_evidence_fields_from_payload(source_payload)
    assert scenario.excluded_evidence_fields[0].path == "$.adaptive_report.certificate_ci"
    assert "certificate_ci" not in json.dumps(scenario.metadata, sort_keys=True)
    assert scenario.metadata["confirmatory_ci"] == [0.2, 0.4]
