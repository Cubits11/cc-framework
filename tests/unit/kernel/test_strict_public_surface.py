from __future__ import annotations

import ast
from pathlib import Path

import cc.kernel as broad_kernel
import cc.kernel.strict as strict_kernel

ROOT = Path(__file__).resolve().parents[3]
PAPER_FACING_FILES = (
    ROOT / "scripts" / "reproduce_paper.py",
    ROOT / "scripts" / "verify_paper_artifacts.py",
    ROOT / "examples" / "minimal" / "run_bounds.py",
)
DISALLOWED_PAPER_IMPORTS = {
    "cc.kernel.causal",
    "cc.kernel.ccf_models",
    "cc.kernel.cliff",
    "cc.kernel.frechet_classes",
    "cc.kernel.metrics",
    "cc.kernel.sample_complexity",
    "cc.kernel.sensitivity",
    "cc.kernel.sequential",
    "cc.kernel.stress",
}


def test_strict_surface_excludes_experimental_kernel_symbols() -> None:
    exported = set(strict_kernel.__all__)

    assert "strict" not in exported
    assert "causal" not in exported
    assert "ccf_models" not in exported
    assert "cliff" not in exported
    assert "sequential" not in exported
    assert "stress" not in exported
    assert "AssumptionSet" in exported
    assert "LinearQuery" in exported
    assert "identified_region" in exported
    assert "composition_bounds_from_counts" in exported


def test_broad_kernel_remains_backward_compatible() -> None:
    assert broad_kernel.AssumptionSet is strict_kernel.AssumptionSet
    assert hasattr(broad_kernel, "AnytimeBernoulliTester")
    assert "AnytimeBernoulliTester" in broad_kernel.__all__


def test_paper_facing_code_imports_kernel_through_strict_surface() -> None:
    for path in PAPER_FACING_FILES:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imported_modules = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        }
        assert "cc.kernel.strict" in imported_modules
        assert imported_modules.isdisjoint(DISALLOWED_PAPER_IMPORTS)


def test_paper_core_docs_name_strict_kernel_surface() -> None:
    paper_core = (ROOT / "docs" / "research" / "PAPER_CORE.md").read_text(encoding="utf-8")

    assert "cc.kernel.strict" in paper_core
