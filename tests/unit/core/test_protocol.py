from pathlib import Path

from cc.core.guardrail_api import GuardrailAdapter
from cc.core.logging import ChainedJSONLLogger
from cc.core.protocol import TwoWorldProtocol
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
