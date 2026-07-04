from pathlib import Path

import pytest

from cc.core.audit_runner import AuditRunConfig, run_audit


def test_audit_runner_rejects_run_id_path_traversal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    prompt_source = tmp_path / "prompts.txt"
    prompt_source.write_text("ordinary prompt\n", encoding="utf-8")

    config = AuditRunConfig(
        prompt_source=prompt_source,
        guardrails=[{"name": "keyword_blocker", "params": {"keywords": ["secret"]}}],
        output_dir=tmp_path / "runs",
        run_id="../escaped",
    )

    with pytest.raises(ValueError, match="run_id"):
        run_audit(config)

    assert not (tmp_path / "escaped").exists()
