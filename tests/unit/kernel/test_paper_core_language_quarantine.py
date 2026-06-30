from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PAPER_CORE_TEXTS = (
    ROOT / "docs" / "research" / "PAPER_CORE.md",
    ROOT / "docs" / "theory" / "finite_sample_identification.md",
    ROOT / "docs" / "theory" / "theorem_ledger.md",
    *sorted((ROOT / "paper" / "sections").glob("*.tex")),
)
FORBIDDEN_TERMS = (
    "CC_max",
    "cc_max",
    "Delta_add",
    "delta_add",
    "Youden",
    "J-statistic",
    "J statistic",
    "proof-carrying",
    "Proof-carrying",
    "scalarization impossibility",
    "Scalarization impossibility",
    "dependence cliff",
    "dependence cliffs",
)


def test_paper_core_texts_do_not_use_legacy_metric_or_overclaim_terms() -> None:
    for path in PAPER_CORE_TEXTS:
        text = path.read_text(encoding="utf-8")
        for term in FORBIDDEN_TERMS:
            assert term not in text, f"{path} contains forbidden paper-core term {term!r}"


def test_product_coupling_is_labeled_as_baseline_in_paper_core_texts() -> None:
    for path in PAPER_CORE_TEXTS:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            normalized = line.lower()
            if "product coupling" not in normalized and "product-coupling" not in normalized:
                continue
            assert (
                "baseline" in normalized
                or "not truth" in normalized
                or "not the default model" in normalized
            ), f"{path}:{line_number} mentions product coupling without baseline labeling"
