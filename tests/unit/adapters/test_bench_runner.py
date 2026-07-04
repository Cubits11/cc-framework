# tests/unit/adapters/test_bench_runner.py
import json
import random
from pathlib import Path

from cc.adapters.base import Decision, GuardrailAdapter
from cc.cartographer.audit import verify_chain
from cc.evals.run_bench import run_benchmark
from cc.io.seeds import set_seed


class RandomAdapter(GuardrailAdapter):
    name = "random"
    version = "test"
    supports_input_check = True
    supports_output_check = False

    def check(self, prompt, response, metadata):
        verdict = "block" if random.random() > 0.5 else "allow"
        return Decision(
            verdict=verdict,
            category=None,
            score=None,
            rationale=None,
            raw={"prompt": prompt},
            adapter_name=self.name,
            adapter_version=self.version,
        )


class AuditedAdapter(GuardrailAdapter):
    name = "audited"
    version = "test"
    supports_input_check = True
    supports_output_check = False

    def check(self, prompt, response, metadata):
        blocked = bool(metadata.get("block", False))
        return Decision(
            verdict="block" if blocked else "allow",
            category=None,
            score=1.0 if blocked else 0.0,
            rationale=None,
            raw={"prompt": prompt},
            audit={"prompt": prompt, "blocked": blocked},
            adapter_name=self.name,
            adapter_version=self.version,
        )


def test_benchmark_determinism(tmp_path: Path):
    dataset = [
        {"prompt": "a", "label": 0},
        {"prompt": "b", "label": 1},
        {"prompt": "c", "label": 0},
    ]
    out_path = tmp_path / "bench.jsonl"
    set_seed(123)
    result1 = run_benchmark(
        dataset=dataset,
        adapters=[RandomAdapter()],
        composition="OR",
        prompt_field="prompt",
        response_field=None,
        label_field="label",
        review_policy="block",
        run_meta={"run_id": "r1", "config": {}},
        out_path=out_path,
    )
    out_path.unlink()
    set_seed(123)
    result2 = run_benchmark(
        dataset=dataset,
        adapters=[RandomAdapter()],
        composition="OR",
        prompt_field="prompt",
        response_field=None,
        label_field="label",
        review_policy="block",
        run_meta={"run_id": "r2", "config": {}},
        out_path=out_path,
    )
    assert result1["summary"]["composed"]["j"] == result2["summary"]["composed"]["j"]


def test_benchmark_writes_batched_hash_chains(tmp_path: Path):
    dataset = [
        {"prompt": "safe", "label": 0, "block": False},
        {"prompt": "unsafe", "label": 1, "block": True},
    ]
    out_path = tmp_path / "bench.jsonl"
    audit_path = tmp_path / "audit.jsonl"

    result = run_benchmark(
        dataset=dataset,
        adapters=[AuditedAdapter()],
        composition="OR",
        prompt_field="prompt",
        response_field=None,
        label_field="label",
        review_policy="block",
        run_meta={"run_id": "batched", "config": {}},
        out_path=out_path,
        audit_out_path=audit_path,
    )

    verify_chain(str(out_path))
    verify_chain(str(audit_path))

    output_records = [
        json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines()
    ]
    audit_records = [
        json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [record["record_type"] for record in output_records] == ["item", "item", "summary"]
    assert all(record["record_type"] == "guardrail_adapter_audit" for record in audit_records)
    assert len(audit_records) == len(dataset)
    assert output_records[-1]["summary"] == result["summary"]
