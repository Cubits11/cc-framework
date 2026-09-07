"""The published reproduction must execute its corrections, not just define them."""

import subprocess
import sys
from pathlib import Path


def test_full_reproduction_executes_revision_two_calculations() -> None:
    root = Path(__file__).resolve().parents[3]
    result = subprocess.run(
        [
            sys.executable,
            str(root / "docs/research/epistemic-program/identical-shadows/identical_shadows.py"),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "strict counterexamples=3" in result.stdout
    assert "EXTERNAL CASES" in result.stdout
    assert "DISCLOSURE LADDER" in result.stdout
    assert "not information fractions" in result.stdout
