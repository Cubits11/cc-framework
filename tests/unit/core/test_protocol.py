from pathlib import Path

import numpy as np
from scipy import stats

from cc.core.guardrail_api import GuardrailAdapter
from cc.core.logging import ChainedJSONLLogger
from cc.core.models import AttackResult
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


def test_apply_guardrail_stack_scores_once(tmp_path: Path) -> None:
    """Protocol should invoke ``score`` at most once per guardrail."""

    cg = CountingGuardrail()
    adapter = GuardrailAdapter(cg)
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    proto = TwoWorldProtocol(
        logger=ChainedJSONLLogger(str(log_dir / "audit.jsonl"))
    )

    blocked, score, triggered = proto.apply_guardrail_stack([adapter], "attack")
    assert blocked is True
    assert score == 0.9
    assert triggered == ["CountingGuardrail"]
    assert cg.score_calls == 1


def test_causal_effect_cluster_robust_imbalanced_clusters() -> None:
    rng = np.random.default_rng(4)
    cluster_sizes = [12, 8, 6, 4]
    cluster_effects = [0.4, -0.4, 0.4, -0.2]
    base = 0.4
    treat = 0.05

    results = []
    idx = 0
    for cluster_id, (size, effect) in enumerate(zip(cluster_sizes, cluster_effects)):
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
