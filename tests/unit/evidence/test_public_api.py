from __future__ import annotations

import cc.evidence as evidence


def test_new_evidence_public_imports_are_intentional() -> None:
    required = {
        "ClaimDecayPolicy",
        "ClaimDecayRecord",
        "DecayStatus",
        "DecayState",
        "evaluate_claim_decay",
        "ExtremalScenario",
        "ScenarioKind",
        "GuardrailOutcome",
        "CLAIM_DECAY_SCHEMA",
        "EXTREMAL_SCENARIO_SCHEMA",
    }

    assert required <= set(evidence.__all__)
    for name in required:
        assert hasattr(evidence, name)

    assert "excluded_evidence_fields_from_payload" not in evidence.__all__
    assert "ScenarioFeasibility" not in evidence.__all__
    assert "ExcludedEvidenceField" not in evidence.__all__
    assert "ConfiguredHazardPolicy" not in evidence.__all__
