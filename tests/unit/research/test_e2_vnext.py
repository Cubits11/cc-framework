from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.e2_vnext.power_design import (
    bias_corrected_delta,
    delta_variance,
    feasible_delta_bounds,
    raw_delta_from_counts,
    required_complete_pairs,
)
from experiments.e2_vnext.preflight import ACTIVATION_STOP, evaluate, load_json


ROOT = Path(__file__).resolve().parents[3]
E2_VNEXT = ROOT / "experiments" / "e2_vnext"


def test_paired_delta_variance_matches_central_margin_derivation() -> None:
    assert delta_variance(0.5, 0.5, 0.02) == pytest.approx(0.0621)
    assert feasible_delta_bounds(0.5, 0.5) == pytest.approx((-0.25, 0.25))


def test_delta_resolution_drives_2025_complete_pair_requirement() -> None:
    point = required_complete_pairs(0.5, 0.5, 0.02, power=0.95)
    assert point.required_complete_pairs == 2025
    assert point.variance_null == pytest.approx(0.0625)
    assert point.variance_alternative == pytest.approx(0.0621)


def test_raw_delta_and_superpopulation_bias_correction_are_distinct() -> None:
    raw = raw_delta_from_counts(n_11=6, n_10=2, n_01=2, n_00=10)
    assert raw == pytest.approx(0.14)
    assert bias_corrected_delta(raw, 20) == pytest.approx(0.14 * 20 / 19)
    with pytest.raises(ValueError):
        bias_corrected_delta(0.0, 1)


def test_frozen_preflight_stops_before_loader_on_two_recorded_blockers(tmp_path: Path) -> None:
    design = load_json(E2_VNEXT / "FROZEN_DESIGN.json")
    screen = load_json(E2_VNEXT / "CANDIDATE_SCREEN.json")
    result = evaluate(design, screen, tmp_path / "decision_attestation.json")
    assert result["activation_status"] == ACTIVATION_STOP
    assert "E2V-G0-ATTESTATION-MISSING" in result["reason_codes"]
    assert "E2V-TRAINING_OVERLAP-FAIL" in result["reason_codes"]
    assert "E2V-FRAME-INSUFFICIENT" not in result["reason_codes"]
    assert result["allowed_next_stage"] == "none"
    assert all("loader" not in str(gate["detail"]).lower() for gate in result["gates"])


def test_candidate_screen_has_capacity_but_never_converts_unknown_overlap_to_pass() -> None:
    design = json.loads((E2_VNEXT / "FROZEN_DESIGN.json").read_text(encoding="utf-8"))
    screen = json.loads((E2_VNEXT / "CANDIDATE_SCREEN.json").read_text(encoding="utf-8"))
    assert screen["sampling_frame"]["eligible_attack_items"] >= design["delta_power"]["minimum_complete_pairs"]
    assert screen["selected_candidate"]["ontology"] == "PASS"
    assert screen["selected_candidate"]["training_overlap"] == "FAIL"
