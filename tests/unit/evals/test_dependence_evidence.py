from __future__ import annotations

import json

import pytest

from cc.evals.dependence_benchmark import (
    BenchmarkInputError,
    run_e1_study,
    verify_e1_artifacts,
    verify_e1_study,
    write_e1_artifacts,
)


def test_e1_parity_proves_pairwise_evidence_does_not_identify_three_way_event() -> None:
    payload = run_e1_study(replicates=2, sample_sizes=(8,))

    assert verify_e1_study(payload) == []
    scenarios = {scenario["scenario_id"]: scenario for scenario in payload["scenarios"]}
    even = scenarios["S4_parity_even"]
    odd = scenarios["S4_parity_odd"]
    assert even["moments"] == odd["moments"]
    assert even["true_primary_event_rate"] == pytest.approx(0.0)
    assert odd["true_primary_event_rate"] == pytest.approx(0.25)

    for regime in ("I0", "I1", "I2"):
        assert even["regimes"][regime]["lower_bound"] == pytest.approx(
            odd["regimes"][regime]["lower_bound"]
        )
        assert even["regimes"][regime]["upper_bound"] == pytest.approx(
            odd["regimes"][regime]["upper_bound"]
        )
    assert even["regimes"]["I2"]["lower_bound"] == pytest.approx(0.0)
    assert even["regimes"]["I2"]["upper_bound"] == pytest.approx(0.25)
    assert even["regimes"]["I3"]["lower_bound"] == pytest.approx(0.0)
    assert odd["regimes"]["I3"]["lower_bound"] == pytest.approx(0.25)


def test_e1_rejects_malformed_designs() -> None:
    with pytest.raises(BenchmarkInputError, match="replicates"):
        run_e1_study(replicates=0)
    with pytest.raises(BenchmarkInputError, match="sample_sizes"):
        run_e1_study(sample_sizes=())
    with pytest.raises(BenchmarkInputError, match="delta"):
        run_e1_study(delta=1.0)


def test_e1_artifacts_detect_tampering(tmp_path) -> None:  # type: ignore[no-untyped-def]
    write_e1_artifacts(
        tmp_path,
        generation_command="unit-test",
        replicates=2,
        sample_sizes=(8,),
    )
    assert verify_e1_artifacts(tmp_path, regenerate=True) == []

    study = json.loads((tmp_path / "study.json").read_text(encoding="utf-8"))
    study["scenarios"][0]["regimes"]["I0"]["lower_bound"] = 0.123
    (tmp_path / "study.json").write_text(
        json.dumps(study, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    errors = verify_e1_artifacts(tmp_path)
    assert any("lower_bound is incorrect" in error for error in errors)
    assert any("manifest hash mismatch" in error for error in errors)
