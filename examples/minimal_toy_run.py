"""Minimal end-to-end audit demo.

Usage:
  python examples/minimal_toy_run.py

Outputs:
  output/<run_id>/
    execution_manifest.json
    results.jsonl
    attestation.json
"""

from pathlib import Path

from cc.core.audit_runner import AuditRunConfig, run_audit, verify_attestation


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    prompt_source = root / "datasets" / "attack_prompts" / "basic.txt"
    benign_source = root / "datasets" / "benign" / "safe_prompts.txt"
    output_dir = root / "output"

    cfg = AuditRunConfig(
        prompt_source=prompt_source,
        benign_calibration_source=benign_source,
        output_dir=output_dir,
        composition="any_block",
        guardrails=[
            {
                "name": "keyword_blocker",
                "params": {"keywords": ["password", "api_key", "secret", "hack", "exploit"]},
            }
        ],
    )

    result = run_audit(cfg)
    print("Audit run complete:")
    for key, value in result.items():
        print(f"  {key}: {value}")

    verified, reason = verify_attestation(
        Path(result["attestation_path"]),
        Path(result["manifest_path"]),
        Path(result["results_path"]),
    )
    print(f"Attestation check: {verified} ({reason})")


if __name__ == "__main__":
    main()
