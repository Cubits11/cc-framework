"""V1/V2: malformed observations yield violations, and timestamps are RFC 3339."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
VALIDATOR = ROOT / "scripts/validate_e2_observations.py"
SCHEMA = ROOT / "schemas/cc.e2_observation_row.v1.json"
REFERENCE = ROOT / "examples/e2/observations.example.jsonl"


def _rows() -> list[dict]:
    return [json.loads(line) for line in REFERENCE.read_text().splitlines()[:3]]


def _run(path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-B", str(VALIDATOR), str(path), "--schema", str(SCHEMA), "--json"],
        text=True,
        capture_output=True,
        check=False,
    )


def _payload(result: subprocess.CompletedProcess[str]) -> dict:
    assert "Traceback" not in result.stdout + result.stderr, result.stderr
    # Accept the existing success-summary suffix; the violations themselves
    # must be a structured JSON document on stdout, never exception prose.
    try:
        payload, _ = json.JSONDecoder().raw_decode(result.stdout.lstrip())
    except json.JSONDecodeError:
        pytest.fail(f"no structured validator output: stdout={result.stdout!r}, stderr={result.stderr!r}")
    assert isinstance(payload, dict)
    assert isinstance(payload.get("violations"), list)
    return payload


def _reject(result: subprocess.CompletedProcess[str], field: str | None = None) -> None:
    payload = _payload(result)
    assert result.returncode == 1, (result.returncode, payload, result.stderr)
    assert payload["violations"], payload
    assert all(isinstance(message, str) for message in payload["violations"])
    if field is not None:
        assert any(field in message for message in payload["violations"]), payload


def _write_rows(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "observations.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


@pytest.mark.parametrize("replicate", [[], {}, "not-an-int", None, 0.5, True])
def test_malformed_replicate_is_structured_violation(tmp_path: Path, replicate: object) -> None:
    rows = _rows()
    rows[0]["replicate_id"] = replicate
    _reject(_run(_write_rows(tmp_path, rows)), "replicate_id")


def test_mixed_replicate_types_do_not_crash_sorting(tmp_path: Path) -> None:
    rows = _rows()
    rows.append(dict(rows[0], replicate_id="1"))
    _reject(_run(_write_rows(tmp_path, rows)), "replicate_id")


@pytest.mark.parametrize("outcome", ["junk", [], {}, True, 1.5, None])
def test_malformed_outcome_is_structured_violation(tmp_path: Path, outcome: object) -> None:
    rows = _rows()
    rows[0]["normalized_outcome"] = outcome
    _reject(_run(_write_rows(tmp_path, rows)), "normalized_outcome")


@pytest.mark.parametrize(
    "field,value",
    [
        ("item_id", []),
        ("guardrail_id", {}),
        ("configuration_hash", []),
        ("missingness_code", []),
        ("shared_dependency_group", {}),
        ("guardrail_version", []),
        ("raw_outcome_detail", "score"),
        ("execution_timestamp", 1),
    ],
)
def test_other_malformed_fields_are_structured_violations(
    tmp_path: Path, field: str, value: object
) -> None:
    rows = _rows()
    rows[0][field] = value
    _reject(_run(_write_rows(tmp_path, rows)), field)


@pytest.mark.parametrize("row", [[], None, "row", 1, True])
def test_non_object_jsonl_rows_are_structured_violations(tmp_path: Path, row: object) -> None:
    path = tmp_path / "not-an-object.jsonl"
    path.write_text(json.dumps(row) + "\n")
    _reject(_run(path))


@pytest.mark.parametrize("content", [b'{"item_id":', b'\xff\xfe\n', b'', b'  \n\t\n'])
def test_unreadable_or_empty_observations_are_structured_violations(
    tmp_path: Path, content: bytes
) -> None:
    path = tmp_path / "malformed.jsonl"
    path.write_bytes(content)
    _reject(_run(path))


@pytest.mark.parametrize("kind", ["missing", "directory"])
def test_input_io_errors_are_structured_violations(tmp_path: Path, kind: str) -> None:
    path = tmp_path / "input.jsonl"
    if kind == "directory":
        path.mkdir()
    _reject(_run(path))


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_nonstandard_json_number_is_a_structured_violation(tmp_path: Path, constant: str) -> None:
    rows = _rows()
    rows[0]["raw_outcome_detail"] = {"score": float(constant)}
    _reject(_run(_write_rows(tmp_path, rows)))


@pytest.mark.parametrize(
    "timestamp",
    [
        "not-a-date",
        "2026-10-24",
        "2026-10-24T20:00:00",
        "2026-13-24T20:00:00Z",
        "2026-10-24T25:00:00Z",
        "2026-10-24T20:00:00+24:00",
        "2026-10-24T20:00:00+04",
    ],
)
def test_invalid_rfc3339_timestamp_is_a_violation(tmp_path: Path, timestamp: str) -> None:
    rows = _rows()
    rows[0]["execution_timestamp"] = timestamp
    _reject(_run(_write_rows(tmp_path, rows)), "execution_timestamp")


@pytest.mark.parametrize(
    "timestamp",
    [
        "2026-10-24T20:00:00Z",
        "2026-10-24T20:00:00.123456Z",
        "2026-10-24T20:00:00-04:00",
        "2026-10-24T20:00:00+05:30",
    ],
)
def test_valid_rfc3339_timestamp_is_accepted(tmp_path: Path, timestamp: str) -> None:
    rows = _rows()
    rows[0]["execution_timestamp"] = timestamp
    result = _run(_write_rows(tmp_path, rows))
    payload = _payload(result)
    assert result.returncode == 0, (payload, result.stderr)
    assert payload["violations"] == []
