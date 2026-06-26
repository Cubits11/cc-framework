import json
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from cc.core.attackers import AttackStrategy
from cc.core.guardrail_api import GuardrailAdapter
from cc.core.logging import ChainedJSONLLogger
from cc.core.models import AttackResult, WorldConfig
from cc.core.protocol import CausalInferenceEngine, TwoWorldProtocol
from cc.guardrails.base import Guardrail


class CountingGuardrail(Guardrail):
    """Guardrail that counts how often ``score`` is invoked."""

    def __init__(self) -> None:
        self.score_calls = 0

    def score(self, text: str) -> float:  # pragma: no cover - simple counter
        self.score_calls += 1
        return 0.9

    def blocks(self, text: str) -> bool:  # pragma: no cover - uses score
        return self.score(text) > 0.5

    def calibrate(self, benign_texts, target_fpr: float = 0.05) -> None:
        return None


class FailingGuardrail(Guardrail):
    """Guardrail that forces the protocol error logging path."""

    def score(self, text: str) -> float:
        raise RuntimeError("forced guardrail failure")

    def blocks(self, text: str) -> bool:
        return False

    def calibrate(self, benign_texts, target_fpr: float = 0.05) -> None:
        return None


class TinyAttacker(AttackStrategy):
    def generate_attack(self, history: list[dict]) -> dict:
        return {"prompt": "synthetic attack"}

    def update_strategy(self, attack: dict, result: dict) -> None:
        return None

    def reset(self, *, seed: int | None = None) -> None:
        return None


def test_apply_guardrail_stack_scores_once(tmp_path: Path) -> None:
    """Protocol should invoke ``score`` at most once per guardrail."""

    cg = CountingGuardrail()
    adapter = GuardrailAdapter(cg)
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    proto = TwoWorldProtocol(logger=ChainedJSONLLogger(str(log_dir / "audit.jsonl")))

    blocked, score, triggered = proto.apply_guardrail_stack([adapter], "attack")
    assert blocked is True
    assert score == 0.9
    assert triggered == ["CountingGuardrail"]
    assert cg.score_calls == 1


def test_guardrail_error_log_omits_raw_prompt(tmp_path: Path) -> None:
    secret_prompt = "secret_prompt_should_not_appear"
    log_path = tmp_path / "audit.jsonl"
    proto = TwoWorldProtocol(logger=ChainedJSONLLogger(str(log_path)))

    blocked, score, triggered = proto.apply_guardrail_stack(
        [GuardrailAdapter(FailingGuardrail())], secret_prompt
    )

    assert blocked is False
    assert score == 0.0
    assert triggered == []
    log_text = log_path.read_text(encoding="utf-8")
    assert secret_prompt not in log_text
    assert "text_preview" not in log_text
    assert "prompt_hash" in log_text


def test_legacy_bayesian_heuristic_requires_explicit_warning(tmp_path: Path) -> None:
    log_path = tmp_path / "audit.jsonl"

    with pytest.warns(RuntimeWarning, match="deprecated, unvalidated legacy heuristic"):
        proto = TwoWorldProtocol(
            logger=ChainedJSONLLogger(str(log_path)),
            enable_bayesian_stopping=True,
            legacy_bayesian_heuristic=True,
        )

    assert proto.bayesian_tester is not None
    assert proto.enable_bayesian_stopping is True


def test_default_protocol_uses_anytime_tester_without_legacy(tmp_path: Path) -> None:
    proto = TwoWorldProtocol(logger=ChainedJSONLLogger(str(tmp_path / "audit.jsonl")))

    assert proto.bayesian_tester is None
    assert proto.sequential_tester.result().e_value == 1.0


def test_run_writes_preregistration_style_analysis_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    proto = TwoWorldProtocol(
        logger=ChainedJSONLLogger(str(tmp_path / "audit.jsonl")),
        episode_length=1,
        random_seed=9,
        checkpoint_every=0,
    )
    worlds = {
        0: WorldConfig(world_id=0, baseline_success_rate=0.4, description="A-only"),
        1: WorldConfig(world_id=1, baseline_success_rate=0.2, description="A+B"),
    }

    with pytest.warns(RuntimeWarning, match="Cluster-bootstrap causal estimator failed"):
        proto.run_experiment(
            attacker=TinyAttacker(),
            world_configs=worlds,
            n_sessions=6,
            experiment_id="analysis_plan_test",
            checkpoint_every=0,
        )

    plan_json = tmp_path / "checkpoints" / "analysis_plan_test" / "analysis_plan.json"
    plan_md = tmp_path / "checkpoints" / "analysis_plan_test" / "analysis_plan.md"
    assert plan_json.exists()
    assert plan_md.exists()

    payload = json.loads(plan_json.read_text(encoding="utf-8"))
    assert payload["estimand"]["notation"] == "tau = E_P[Y_i(1) - Y_i(0)]"
    assert payload["pre_specified_analysis"]["significance_threshold_alpha"] == 0.05
    assert payload["pre_specified_analysis"]["causal_estimator"] == "cluster_bootstrap_ate"
    assert payload["actual_run"]["actual_sessions"] == 6
    assert "Identifying Assumptions" in plan_md.read_text(encoding="utf-8")


def test_causal_effect_cluster_robust_imbalanced_clusters() -> None:
    rng = np.random.default_rng(4)
    cluster_sizes = [12, 8, 6, 4]
    cluster_effects = [0.4, -0.4, 0.4, -0.2]
    base = 0.4
    treat = 0.05

    results = []
    idx = 0
    for cluster_id, (size, effect) in enumerate(zip(cluster_sizes, cluster_effects, strict=False)):
        for i in range(size):
            world_bit = i % 2
            p = base + effect + treat * world_bit
            p = min(max(p, 0.05), 0.95)
            success = bool(rng.random() < p)
            results.append(
                AttackResult(
                    world_bit=world_bit,
                    success=success,
                    attack_id=f"attack-{idx}",
                    transcript_hash=f"hash-{idx}",
                    guardrails_applied="none",
                    rng_seed=idx,
                    attack_strategy=f"cluster-{cluster_id}",
                )
            )
            idx += 1

    engine = CausalInferenceEngine()
    effect = engine.estimate_ate(results)

    w0 = [r.success for r in results if r.world_bit == 0]
    w1 = [r.success for r in results if r.world_bit == 1]
    p0 = float(np.mean(w0))
    p1 = float(np.mean(w1))
    n0 = len(w0)
    n1 = len(w1)
    var0 = max(p0 * (1.0 - p0), 1e-12)
    var1 = max(p1 * (1.0 - p1), 1e-12)
    naive_se = float(np.sqrt(var0 / n0 + var1 / n1))
    naive_df = max(min(n0, n1) - 1, 1)
    naive_ci_half = float(stats.t.ppf(0.975, naive_df)) * naive_se  # type: ignore[arg-type]

    assert (effect.ci_upper - effect.ate) > naive_ci_half

    if naive_se > 0.0:
        naive_t = effect.ate / naive_se
        crit = float(stats.t.ppf(0.975, naive_df))  # type: ignore[arg-type]
        naive_power = float(1.0 - stats.nct.cdf(crit, df=naive_df, nc=abs(naive_t)))  # type: ignore[arg-type]
        assert effect.power < naive_power
