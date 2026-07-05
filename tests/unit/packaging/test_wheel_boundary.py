from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]

FORBIDDEN_PARTS = {
    ".DS_Store",
    ".next",
    ".pytest_cache",
    ".venv",
    ".env",
    ".mypy_cache",
    ".ruff_cache",
    "__pycache__",
    "cdk.out",
    "checkpoints",
    "dist",
    "node_modules",
    "notebooks",
    "pip-wheel-metadata",
    "results",
    "runs",
    "site",
    "summaries",
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
    "raw/",
    "results/",
    "runs/",
    "summaries/",
    "tmp/",
)

FORBIDDEN_SUFFIXES = (
    ".csv.gz",
    ".dylib",
    ".ipynb",
    ".npy",
    ".npz",
    ".parquet",
    ".pkl",
    ".pyc",
    ".so",
    ".sqlite",
    ".tar",
    ".tgz",
    ".zip",
)


@pytest.fixture(scope="module")
def built_wheel(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out_dir = tmp_path_factory.mktemp("dist")
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
    return wheels[0]


def wheel_names(wheel_path: Path) -> list[str]:
    with zipfile.ZipFile(wheel_path) as wheel:
        return wheel.namelist()


def test_built_wheel_excludes_non_package_artifacts(built_wheel: Path) -> None:
    names = wheel_names(built_wheel)

    assert names
    for name in names:
        normalized = name.replace("\\", "/")
        parts = set(normalized.split("/"))
        assert not parts & FORBIDDEN_PARTS, normalized
        assert not normalized.startswith(FORBIDDEN_PREFIXES), normalized
        assert not normalized.endswith(FORBIDDEN_SUFFIXES), normalized
        assert "/.env" not in normalized
        assert "/summaries/" not in normalized


def test_built_wheel_does_not_contain_large_generated_payloads(built_wheel: Path) -> None:
    with zipfile.ZipFile(built_wheel) as wheel:
        for info in wheel.infolist():
            normalized = info.filename.replace("\\", "/")
            assert info.file_size < 2_000_000, normalized
            assert not normalized.endswith((".pdf", ".png", ".jpg", ".jpeg", ".gif", ".mp4"))


def test_importable_wheel_surfaces_are_classified_in_api_doc(built_wheel: Path) -> None:
    names = wheel_names(built_wheel)

    packaged_surfaces = {
        f"cc.{parts[1]}"
        for name in names
        if name.startswith("cc/") and name.endswith(".py")
        for parts in [name.split("/")]
        if len(parts) > 2
    }
    packaged_surfaces.add("cc.kernel.strict")

    api_doc = (ROOT / "docs" / "api.md").read_text(encoding="utf-8")
    missing = sorted(surface for surface in packaged_surfaces if f"`{surface}`" not in api_doc)

    assert "`cc.kernel.strict`" in api_doc
    assert missing == []


def test_artificial_wheel_leak_names_are_rejected_by_boundary_policy(tmp_path: Path) -> None:
    leaked = tmp_path / "leaked.whl"
    with zipfile.ZipFile(leaked, "w") as wheel:
        for name in [
            "cc/kernel/strict.py",
            "apps/dashboard/.next/server.js",
            "infra/cdk.out/tree.json",
            "runs/smoke/output.json",
            "summaries/claim-summary.json",
            ".venv/lib/python/site.py",
        ]:
            wheel.writestr(name, "x")

    names = wheel_names(leaked)
    violations = []
    for name in names:
        normalized = name.replace("\\", "/")
        parts = set(normalized.split("/"))
        if (
            parts & FORBIDDEN_PARTS
            or normalized.startswith(FORBIDDEN_PREFIXES)
            or normalized.endswith(FORBIDDEN_SUFFIXES)
        ):
            violations.append(normalized)

    assert violations == [
        "apps/dashboard/.next/server.js",
        "infra/cdk.out/tree.json",
        "runs/smoke/output.json",
        "summaries/claim-summary.json",
        ".venv/lib/python/site.py",
    ]
