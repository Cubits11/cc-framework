from pathlib import Path

import pytest

pytest.importorskip("cryptography")
pytest.importorskip("blake3")

from cc.core.evidence_bundle import EvidenceBundleConfig, run_evidence_bundle


def test_evidence_bundle_emits_leak_safe_artifacts(tmp_path: Path) -> None:
    prompt_source = tmp_path / "prompts.txt"
    prompt_source.write_text("user email: test@example.com\n", encoding="utf-8")

    config = EvidenceBundleConfig(
        prompt_source=prompt_source,
        guardrails=[{"name": "keyword_blocker", "params": {"keywords": ["secret"]}}],
        output_dir=tmp_path,
        run_id="bundle_test",
        seed=123,
        enable_plots=False,
        env_gates={"allow_real": False},
    )

    result = run_evidence_bundle(config)
    output_dir = Path(result["output_dir"])

    results_text = (output_dir / "results.jsonl").read_text(encoding="utf-8")
    metrics_text = (output_dir / "metrics.json").read_text(encoding="utf-8")
    manifest_text = (output_dir / "manifest.json").read_text(encoding="utf-8")

    assert "test@example.com" not in results_text
    assert "test@example.com" not in metrics_text
    assert "test@example.com" not in manifest_text

    assert (output_dir / "ledger.jsonl").exists()
    assert (output_dir / "attestation.json").exists()
    assert (output_dir / "bundle_hashes.json").exists()
