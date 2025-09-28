import csv
import subprocess
from pathlib import Path

import pytest

REQUIRED_COLUMNS = [
    "tpr_a",
    "tpr_b",
    "fpr_a",
    "fpr_b",
    "I1_lo",
    "I1_hi",
    "I0_lo",
    "I0_hi",
    "vbar1",
    "vbar0",
    "cc_hat",
    "ci_lo",
    "ci_hi",
    "ci_width",
    "D",
    "D_lamp",
    "bonferroni_call",
    "bhy_call",
]

SCAN_PATH = Path("results/week5_scan/scan.csv")


@pytest.fixture(scope="session")
def ensure_week5_scan() -> Path:
    subprocess.run(["make", "week5-pilot"], check=True)
    if not SCAN_PATH.exists():
        raise RuntimeError("Week5 scan did not produce scan.csv")
    return SCAN_PATH


def test_scan_schema_and_rows(ensure_week5_scan: Path) -> None:
    with ensure_week5_scan.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        for col in REQUIRED_COLUMNS:
            assert col in fieldnames, f"Missing column {col}"
        rows = list(reader)
    assert len(rows) >= 100, "Week5 pilot should log at least 100 rows"

    last = None
    for row in reversed(rows):
        if row.get("fpr_a"):
            last = row
            break
    if last is None:
        pytest.fail("No row contained fpr_a values")

    fpr = float(last["fpr_a"])
    assert 0.04 <= fpr <= 0.06, f"Calibrated FPR {fpr:.3f} out of bounds"
