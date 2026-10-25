"""E1: contract rows retain the frozen evaluation stratum through analysis.

The four evaluation strata are frozen in SITE's
experiments/e2/freeze/PREREG_SELECTION.md section 1. Calibration is a separate
population and must never become an evaluation observation stratum.
Set E2_SITE_ROOT to exercise the actual SITE analyzer alongside this contract.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))

from validate_e2_observations import (  # noqa: E402
    design_violations,
    load_rows,
    schema_violations,
)

SCHEMA = json.loads((ROOT / "schemas/cc.e2_observation_row.v1.json").read_text())
REFERENCE = ROOT / "examples/e2/observations.example.jsonl"
STRATA = {"harmful", "benign_eval", "harmful_jbb", "benign_jbb"}


def test_stratum_is_required_and_has_the_frozen_string_enum() -> None:
    assert "stratum" in SCHEMA["required"]
    definition = SCHEMA["properties"]["stratum"]
    assert definition["type"] == "string"
    assert set(definition["enum"]) == STRATA


@pytest.mark.parametrize("stratum", sorted(STRATA))
def test_frozen_evaluation_stratum_is_schema_valid(stratum: str) -> None:
    row = load_rows(REFERENCE)[0]
    row["stratum"] = stratum
    assert schema_violations([row], SCHEMA) == []


@pytest.mark.parametrize("stratum", [None, 1, True, [], {}, "unknown", "benign_calibration"])
def test_other_or_malformed_strata_are_violations(stratum: object) -> None:
    row = load_rows(REFERENCE)[0]
    row["stratum"] = stratum
    assert schema_violations([row], SCHEMA)


def test_missing_stratum_is_a_schema_violation() -> None:
    row = load_rows(REFERENCE)[0]
    row.pop("stratum", None)
    assert any("stratum" in message for message in schema_violations([row], SCHEMA))


def test_three_reference_rows_round_trip_into_actual_site_analyzer(tmp_path: Path) -> None:
    site_root = os.environ.get("E2_SITE_ROOT")
    if not site_root:
        pytest.skip("set E2_SITE_ROOT for the cross-repository analyzer check")
    run_dir = Path(site_root) / "experiments/e2/run"
    sys.path.insert(0, str(run_dir))
    spec = importlib.util.spec_from_file_location("e2_stratum_analyzer", run_dir / "analyze.py")
    assert spec is not None and spec.loader is not None
    analyzer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analyzer)

    rows = load_rows(REFERENCE)[:3]
    for row, guard in zip(rows, ("lg4", "lg3", "sg2b"), strict=True):
        row["guardrail_id"] = guard
    assert schema_violations(rows, SCHEMA) == []
    assert design_violations(rows) == []
    fixture = tmp_path / "reference-three.jsonl"
    fixture.write_text("".join(json.dumps(row) + "\n" for row in rows))

    events, missing, synthetic = analyzer.load(fixture)
    item_id = rows[0]["item_id"]
    assert events == {
        "harmful": {item_id: {row["guardrail_id"]: row["normalized_outcome"] for row in rows}}
    }
    assert analyzer.complete_cases(events["harmful"]) == events["harmful"]
    assert missing == {}
    assert synthetic is False
