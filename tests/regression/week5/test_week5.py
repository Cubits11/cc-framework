import csv
import os
import shutil
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

SCAN_PATH = Path("tests/fixtures/week5_scan/scan.csv")


@pytest.fixture(scope="session")
def ensure_week5_scan(tmp_path_factory: pytest.TempPathFactory) -> Path:
    if os.environ.get("CC_REFRESH_GOLDEN_ARTIFACTS") == "1":
        subprocess.run(["make", "week5-pilot"], check=True)
        generated_scan = Path("results/week5_scan/scan.csv")
        if not generated_scan.exists():
            raise RuntimeError("Week5 scan did not produce scan.csv")
        shutil.copyfile(generated_scan, SCAN_PATH)
        return SCAN_PATH

    if not SCAN_PATH.exists():
        pytest.skip(
            "Committed Week5 scan artifact is absent; set "
            "CC_REFRESH_GOLDEN_ARTIFACTS=1 to regenerate it."
        )

    tmp_scan = tmp_path_factory.mktemp("week5_scan") / "scan.csv"
    shutil.copyfile(SCAN_PATH, tmp_scan)
    return tmp_scan


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
        if row.get("fpr_b"):
            last = row
            break
    if last is None:
        pytest.fail("No row contained fpr_b values")

    fpr = float(last["fpr_b"])
    assert 0.04 <= fpr <= 0.06, f"Calibrated FPR {fpr:.3f} out of bounds"
