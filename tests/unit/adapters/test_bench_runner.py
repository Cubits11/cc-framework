# tests/unit/adapters/test_bench_runner.py
import random
from pathlib import Path

from cc.adapters.base import Decision, GuardrailAdapter
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
