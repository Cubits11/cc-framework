from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path

import cc.kernel.strict as strict_kernel

ROOT = Path(__file__).resolve().parents[3]


def test_strict_kernel_imports_and_exports_stable_symbols() -> None:
    required = {
        "AssumptionSet",
        "BernoulliRateInterval",
        "FiniteSampleIdentificationResult",
        "FrechetBoundResult",
        "FrechetClassInfeasibleError",
        "IdentificationResult",
        "IdentificationInfeasibleError",
        "LinearConstraint",
        "LinearQuery",
        "assumption_set_from_counts",
        "cc_gain",
        "cc_shift",
        "classical_frechet_bounds",
        "enumerate_atoms",
        "frechet_bounds",
        "fh_position",
        "fh_width",
        "identified_region",
        "independence_regret",
        "independent_event_probability",
        "sample_size_for_radius",
    }

    exported = set(strict_kernel.__all__)

    assert required <= exported
    for name in required:
        assert hasattr(strict_kernel, name)


def test_api_contract_document_exists_and_names_boundaries() -> None:
    api_doc = ROOT / "docs" / "api.md"
    text = api_doc.read_text(encoding="utf-8")
    normalized = " ".join(text.lower().split())

    assert "# Public API Contract" in text
    assert "## Stable API" in text
    assert "`cc.kernel.strict`" in text
    assert "stable paper-facing api" in normalized
    assert "## Importable But Experimental Or Backcompat" in text
    assert "`cc.kernel`" in text
    assert "broad compatibility aggregate" in normalized
    assert "do not carry the stable compatibility guarantees of `cc.kernel.strict`" in normalized
    assert "`cc.core`" in text
    assert "`cc.evidence`" in text
    assert "## Reference Or Demo Only" in text
    assert "`cc.enterprise`" in text
    assert "`cc.adapters`" in text
    assert "`cc.redteam`" in text
    assert "dashboard helpers" in normalized
    assert "generated `runs`, `results`, `checkpoints`" in text
    assert "not a live aws or production-readiness claim" in normalized


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


def test_strict_kernel_import_does_not_eagerly_load_broad_experimental_kernel_modules() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json, sys; "
                "import cc.kernel.strict; "
                "print(json.dumps(sorted(name for name in sys.modules "
                "if name.startswith('cc.kernel.'))))"
            ),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    imported = set(json.loads(result.stdout))
    assert {
        "cc.kernel.strict",
        "cc.kernel.frechet_classes",
        "cc.kernel.metrics",
        "cc.kernel.sample_complexity",
        "cc.kernel.sensitivity",
    } <= imported
    assert imported.isdisjoint(
        {
            "cc.kernel.causal",
            "cc.kernel.ccf_models",
            "cc.kernel.cliff",
            "cc.kernel.sequential",
            "cc.kernel.stress",
        }
    )


def test_cartographer_cli_import_does_not_require_plotting_stack() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json, sys; "
                "import cc.cartographer.cli; "
                "print(json.dumps(sorted(name for name in sys.modules "
                "if name == 'matplotlib' or name.startswith('matplotlib.') "
                "or name == 'cc.cartographer.atlas')))"
            ),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == []


def test_optional_plotting_modules_import_without_matplotlib_installed() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import builtins; "
                "real_import = builtins.__import__\n"
                "def blocked_import(name, *args, **kwargs):\n"
                "    if name == 'matplotlib' or name.startswith('matplotlib.'):\n"
                "        raise ModuleNotFoundError(\"No module named 'matplotlib'\", "
                "name='matplotlib')\n"
                "    return real_import(name, *args, **kwargs)\n"
                "builtins.__import__ = blocked_import\n"
                "import cc.analysis.generate_figures\n"
                "import cc.cartographer.atlas\n"
                "import cc.cartographer.cli\n"
            ),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
