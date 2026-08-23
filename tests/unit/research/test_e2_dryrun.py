"""The E2 dry run must stay an honest, conforming instrument rehearsal.

These tests hold the dry run to environment-robust invariants rather than
byte-equality of a regenerated semantic column (the TF-IDF mechanism depends on
the installed backend). They check that:

  * the committed reference observation set still passes the real E2 validator;
  * a fresh run emits a conforming, shared-item-complete observation set;
  * the pre-registered negative controls behave (the common-cause control hits
    its closed form; the independent recompute matches);
  * nothing here silently drifts into looking like a positive real-guardrail
    result — the report is flagged is_e2 == False and carries its non-claims.

What these tests do NOT establish: that the toy Delta means anything about real
guardrails. It does not, by construction (see docs/research/E2_DRYRUN_FINDING.md).
"""

from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = ROOT / "experiments" / "e2_dryrun"
REFERENCE = EXPERIMENT / "results" / "observations.jsonl"

sys.path.insert(0, str(ROOT / "scripts"))
validate_e2 = pytest.importorskip("validate_e2_observations")
pytest.importorskip("sklearn")


def _load_schema() -> dict:
    return json.loads(
        (ROOT / "schemas" / "cc.e2_observation_row.v1.json").read_text(encoding="utf-8")
    )


def _run_harness(tmp_path: Path) -> dict:
    import cc.evals.dependence_benchmark  # noqa: F401  (ensures package import path)

    module_globals = runpy.run_path(
        str(EXPERIMENT / "run_dryrun.py"), run_name="_dryrun_under_test"
    )
    rows, labels, matrix, _item_ids = module_globals["emit_rows"](
        module_globals["load_corpus"](),
        module_globals["build_guardrails"](
            [r["text"] for r in module_globals["load_corpus"]() if int(r["reference_label"]) == 0]
        ),
    )
    controls = module_globals["negative_controls"](matrix, labels)
    deltas = module_globals["delta_matrix"](matrix, labels)
    return {
        "rows": rows,
        "labels": labels,
        "matrix": matrix,
        "controls": controls,
        "deltas": deltas,
    }


def test_reference_snapshot_conforms() -> None:
    rows = validate_e2.load_rows(REFERENCE)
    schema = _load_schema()
    assert validate_e2.schema_violations(rows, schema) == []
    assert validate_e2.design_violations(rows) == []
    assert len(rows) == 66


def test_fresh_run_is_shared_item_complete(tmp_path: Path) -> None:
    result = _run_harness(tmp_path)
    rows = result["rows"]
    assert len(rows) == 66
    schema = _load_schema()
    assert validate_e2.schema_violations(rows, schema) == []
    assert validate_e2.design_violations(rows) == []
    # every guardrail saw every harmful item (Layer B)
    items = {r["item_id"] for r in rows}
    for guardrail in result["labels"]:
        covered = {r["item_id"] for r in rows if r["guardrail_id"] == guardrail}
        assert covered == items


def test_marginals_and_deltas_in_range() -> None:
    result = _run_harness(Path("/tmp"))
    for rate in result["deltas"]["singleton_failure_rates"].values():
        assert 0.0 <= rate <= 1.0
    for pair in result["deltas"]["pairs"].values():
        assert -1.0 <= pair["delta"] <= 1.0
        assert pair["fh_lower"] <= pair["p11_measured"] + 1e-9
        assert pair["p11_measured"] <= pair["fh_upper"] + 1e-9


def test_negative_controls_behave() -> None:
    result = _run_harness(Path("/tmp"))
    controls = result["controls"]
    # common-cause control must hit the closed form pA*(1-pA) exactly
    assert controls["duplicate_common_cause"]["matches"] is True
    observed = controls["duplicate_common_cause"]["observed_delta"]
    expected = controls["duplicate_common_cause"]["expected_delta"]
    assert abs(observed - expected) < 1e-6
    # independent recompute must match one of the pair deltas
    recomputed = controls["independent_recompute"]["value"]
    first_pair = next(iter(result["deltas"]["pairs"].values()))
    assert abs(recomputed - first_pair["delta"]) < 1e-6


def test_report_refuses_to_look_like_e2() -> None:
    report = json.loads((EXPERIMENT / "results" / "report.json").read_text(encoding="utf-8"))
    assert report["is_e2"] is False
    assert report["n_harmful"] == 22
    joined = " ".join(report["non_claims"]).lower()
    assert "not e2" in joined
    assert "not deployed guardrails" in joined or "toy" in joined
