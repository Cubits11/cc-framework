from __future__ import annotations

import importlib
from pathlib import Path

import cc.kernel.strict as strict_kernel

ROOT = Path(__file__).resolve().parents[3]


def test_strict_kernel_imports_and_exports_stable_symbols() -> None:
    required = {
        "AssumptionSet",
        "IdentificationInfeasibleError",
        "LinearConstraint",
        "LinearQuery",
        "frechet_bounds",
        "fh_width",
        "identified_region",
        "independent_event_probability",
    }

    exported = set(strict_kernel.__all__)

    assert required <= exported
    for name in required:
        assert hasattr(strict_kernel, name)


def test_api_contract_document_exists_and_names_boundaries() -> None:
    api_doc = ROOT / "docs" / "api.md"
    text = api_doc.read_text(encoding="utf-8")

    assert "# Public API Contract" in text
    assert "## Stable API" in text
    assert "`cc.kernel.strict`" in text
    assert "## Importable But Experimental Or Backcompat" in text
    assert "`cc.kernel`" in text
    assert "`cc.core`" in text
    assert "`cc.evidence`" in text
    assert "## Reference Or Demo Only" in text
    assert "`cc.enterprise`" in text
    assert "`cc.adapters`" in text


def test_documented_experimental_modules_import_without_stability_claim() -> None:
    modules = [
        "cc.kernel",
        "cc.core",
        "cc.evidence",
        "cc.evals",
        "cc.cartographer",
    ]
    api_doc = (ROOT / "docs" / "api.md").read_text(encoding="utf-8")

    for module_name in modules:
        assert f"`{module_name}`" in api_doc
        assert importlib.import_module(module_name) is not None
