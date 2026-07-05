from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

FORBIDDEN_PARTS = {
    ".next",
    ".pytest_cache",
    ".venv",
    "__pycache__",
    "cdk.out",
    "checkpoints",
    "dist",
    "node_modules",
    "notebooks",
    "results",
    "runs",
    "site",
}

FORBIDDEN_PREFIXES = (
    "apps/",
    "artifacts/",
    "build/",
    "checkpoints/",
    "dist/",
    "experiments/",
    "infra/",
    "notebooks/",
    "paper/",
    "results/",
    "runs/",
)

FORBIDDEN_SUFFIXES = (
    ".ipynb",
    ".parquet",
    ".pkl",
    ".sqlite",
)


def test_built_wheel_excludes_non_package_artifacts(tmp_path: Path) -> None:
    out_dir = tmp_path / "dist"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--no-isolation",
            "--outdir",
            str(out_dir),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    wheels = sorted(out_dir.glob("*.whl"))
    assert len(wheels) == 1

    with zipfile.ZipFile(wheels[0]) as wheel:
        names = wheel.namelist()

    assert names
    for name in names:
        normalized = name.replace("\\", "/")
        parts = set(normalized.split("/"))
        assert not parts & FORBIDDEN_PARTS, normalized
        assert not normalized.startswith(FORBIDDEN_PREFIXES), normalized
        assert not normalized.endswith(FORBIDDEN_SUFFIXES), normalized


def test_importable_wheel_surfaces_are_classified_in_api_doc(tmp_path: Path) -> None:
    out_dir = tmp_path / "dist"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--no-isolation",
            "--outdir",
            str(out_dir),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    wheel_path = next(out_dir.glob("*.whl"))
    with zipfile.ZipFile(wheel_path) as wheel:
        names = wheel.namelist()

    packaged_surfaces = {
        f"cc.{parts[1]}"
        for name in names
        if name.startswith("cc/") and name.endswith(".py")
        for parts in [name.split("/")]
        if len(parts) > 2
    }

    api_doc = (ROOT / "docs" / "api.md").read_text(encoding="utf-8")
    missing = sorted(surface for surface in packaged_surfaces if f"`{surface}`" not in api_doc)

    assert "`cc.kernel.strict`" in api_doc
    assert missing == []
