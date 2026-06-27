from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def test_verify_paper_artifacts_accepts_fresh_reproduction(tmp_path: Path) -> None:
    out_dir = _reproduce(tmp_path)

    result = _verify(out_dir)

    assert result.returncode == 0, result.stderr


def test_verify_paper_artifacts_catches_corrupted_witness_even_with_updated_hash(
    tmp_path: Path,
) -> None:
    out_dir = _reproduce(tmp_path)
    witness_path = out_dir / "minimal_witnesses.json"
    witnesses = json.loads(witness_path.read_text(encoding="utf-8"))
    witnesses["cases"][0]["witnesses"]["lower"]["distribution"][0] += 0.125
    _write_json(witness_path, witnesses)
    _refresh_manifest_entry(out_dir, "minimal_witnesses.json")

    result = _verify(out_dir)

    assert result.returncode != 0
    assert "distribution sums" in result.stderr or "marginal" in result.stderr


def test_verify_paper_artifacts_catches_manifest_hash_mismatch(tmp_path: Path) -> None:
    out_dir = _reproduce(tmp_path)
    manifest_path = out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"][0]["sha256"] = "0" * 64
    manifest["manifest_payload_sha256"] = _manifest_payload_hash(manifest)
    _write_json(manifest_path, manifest)

    result = _verify(out_dir)

    assert result.returncode != 0
    assert "Hash mismatch" in result.stderr


def _reproduce(tmp_path: Path) -> Path:
    out_dir = tmp_path / "paper"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/reproduce_paper.py",
            "--out",
            str(out_dir),
        ],
        cwd=ROOT,
        env=_env(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return out_dir


def _verify(out_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "scripts/verify_paper_artifacts.py",
            "--dir",
            str(out_dir),
        ],
        cwd=ROOT,
        env=_env(),
        capture_output=True,
        text=True,
        check=False,
    )


def _env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    return env


def _refresh_manifest_entry(out_dir: Path, filename: str) -> None:
    manifest_path = out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    path = out_dir / filename
    for entry in manifest["files"]:
        if entry["filename"] == filename:
            entry["sha256"] = _sha256(path)
            entry["bytes"] = path.stat().st_size
            break
    else:
        raise AssertionError(f"Missing manifest entry for {filename}")
    manifest["manifest_payload_sha256"] = _manifest_payload_hash(manifest)
    _write_json(manifest_path, manifest)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_payload_hash(manifest: dict[str, Any]) -> str:
    payload = dict(manifest)
    payload.pop("manifest_payload_sha256", None)
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
