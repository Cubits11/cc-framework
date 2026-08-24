#!/usr/bin/env python3
"""Evaluate E2-VNEXT's static gates without downloading or loading a model.

The program intentionally has no network, model-runtime, dataset-parser, or
prompt argument.  A NOT_ACTIVATED record is therefore evidence that the study
was stopped *before* a candidate could see an item.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ACTIVATION_PASS = "ACTIVATION_READY"
ACTIVATION_STOP = "NOT_ACTIVATED"


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON object at {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _gate(name: str, passed: bool, code: str, detail: str) -> dict[str, object]:
    return {"code": code, "detail": detail, "name": name, "passed": passed}


def _decision_gate(design: dict[str, Any], attestation_path: Path) -> dict[str, object]:
    if not attestation_path.exists():
        return _gate(
            "decision_owner_and_joint_detector_miss_cap",
            False,
            "E2V-G0-ATTESTATION-MISSING",
            "No signed pre-data attestation names an owner, an action, a joint detector-miss cap, and a benign false-block cap.",
        )
    try:
        attestation = load_json(attestation_path)
    except ValueError as exc:
        return _gate("decision_owner_and_joint_detector_miss_cap", False, "E2V-G0-ATTESTATION-INVALID", str(exc))
    required_text = ("decision_owner", "action_if_pass", "action_if_fail", "action_if_hold", "rationale", "attested_at_utc")
    missing_text = [name for name in required_text if not isinstance(attestation.get(name), str) or not attestation[name].strip()]
    caps = ("joint_detector_miss_cap", "serial_false_block_cap")
    invalid_caps = [
        name
        for name in caps
        if not isinstance(attestation.get(name), (int, float)) or not 0.0 < float(attestation[name]) < 1.0
    ]
    if missing_text or invalid_caps:
        return _gate(
            "decision_owner_and_joint_detector_miss_cap",
            False,
            "E2V-G0-ATTESTATION-INCOMPLETE",
            f"missing_text={missing_text}; invalid_caps={invalid_caps}",
        )
    if attestation.get("study_id") != design.get("study_id"):
        return _gate(
            "decision_owner_and_joint_detector_miss_cap",
            False,
            "E2V-G0-ATTESTATION-STUDY-MISMATCH",
            "The attestation does not name this study.",
        )
    return _gate(
        "decision_owner_and_joint_detector_miss_cap",
        True,
        "E2V-G0-PASS",
        "A nonempty owner attestation fixes the action and both caps before any score.",
    )


def _screen_gate(screen: dict[str, Any], field: str, name: str, pass_value: str = "PASS") -> dict[str, object]:
    status = screen.get("selected_candidate", {}).get(field)
    passed = status == pass_value
    return _gate(
        name,
        passed,
        f"E2V-{field.upper()}-{'PASS' if passed else 'FAIL'}",
        str(screen.get("selected_candidate", {}).get(f"{field}_rationale", "No rationale recorded.")),
    )


def evaluate(design: dict[str, Any], screen: dict[str, Any], attestation_path: Path) -> dict[str, object]:
    """Return every gate result; do not short-circuit hidden failures."""

    if design.get("study_id") != "E2-VNEXT":
        raise ValueError("unexpected study_id")
    gates: list[dict[str, object]] = [_decision_gate(design, attestation_path)]
    gates.append(_screen_gate(screen, "ontology", "ontology_compatibility_before_loader"))
    gates.append(_screen_gate(screen, "training_overlap", "training_overlap_eligibility"))
    gates.append(_screen_gate(screen, "maintenance", "maintenance_and_rights_eligibility"))
    capacity = screen.get("sampling_frame", {})
    required = design.get("delta_power", {}).get("minimum_complete_pairs")
    available = capacity.get("eligible_attack_items")
    capacity_pass = isinstance(required, int) and isinstance(available, int) and available >= required
    gates.append(
        _gate(
            "frame_capacity_for_delta_resolution",
            capacity_pass,
            "E2V-FRAME-PASS" if capacity_pass else "E2V-FRAME-INSUFFICIENT",
            f"eligible_attack_items={available}; required_complete_pairs={required}",
        )
    )
    selected_id = screen.get("selected_candidate", {}).get("id")
    selected = isinstance(selected_id, str) and bool(selected_id)
    gates.append(
        _gate(
            "one_outcome_blind_candidate_pair_selected",
            selected,
            "E2V-CANDIDATE-SELECTED" if selected else "E2V-CANDIDATE-NONE",
            "No substitution is permitted after this screen.",
        )
    )
    failures = [gate for gate in gates if not gate["passed"]]
    return {
        "activation_status": ACTIVATION_PASS if not failures else ACTIVATION_STOP,
        "allowed_next_stage": "artifact_retrieval_and_no_input_loader" if not failures else "none",
        "gate_order": [gate["name"] for gate in gates],
        "gates": gates,
        "reason_codes": [gate["code"] for gate in failures],
        "study_id": design["study_id"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--candidate-screen", type=Path, required=True)
    parser.add_argument("--decision-attestation", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--observed-at-utc", required=True)
    parser.add_argument("--require-pass", action="store_true")
    args = parser.parse_args()
    try:
        design = load_json(args.design)
        screen = load_json(args.candidate_screen)
        result = evaluate(design, screen, args.decision_attestation)
        result["candidate_screen_sha256"] = sha256_file(args.candidate_screen)
        result["design_sha256"] = sha256_file(args.design)
        result["observed_at_utc"] = args.observed_at_utc
        result["preflight_script_sha256"] = sha256_file(Path(__file__))
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(canonical_json(result) + "\n")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(canonical_json(result))
    return 0 if result["activation_status"] == ACTIVATION_PASS or not args.require_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
