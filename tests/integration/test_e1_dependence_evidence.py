from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_e1_scripts_reproduce_and_verify_a_small_clean_run(tmp_path: Path) -> None:
    artifacts = tmp_path / "e1"
    environment = {**os.environ, "PYTHONPATH": "src"}
    reproduce = subprocess.run(
        [
            sys.executable,
            "scripts/reproduce_e1_dependence_evidence.py",
            "--output-dir",
            str(artifacts),
            "--replicates",
            "2",
            "--sample-sizes",
            "8",
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert reproduce.returncode == 0, reproduce.stderr
    verify = subprocess.run(
        [
            sys.executable,
            "scripts/verify_e1_dependence_evidence.py",
            "--artifact-dir",
            str(artifacts),
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert verify.returncode == 0, verify.stderr
