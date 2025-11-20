from cc.guardrails.base import Guardrail
from cc.guardrails.composites import ANDGuardrail, ORGuardrail


class AlwaysBlock(Guardrail):
    def blocks(self, text: str) -> bool:
        return True

    def score(self, text: str) -> float:
        return 1.0

    def calibrate(self, benign_texts, target_fpr: float = 0.05) -> None:
        self.calibrated = True


class NeverBlock(Guardrail):
    def blocks(self, text: str) -> bool:
        return False

    def score(self, text: str) -> float:
        return 0.0

    def calibrate(self, benign_texts, target_fpr: float = 0.05) -> None:
        self.calibrated = True


def test_empty_stack():
    for cls in (ANDGuardrail, ORGuardrail):
        g = cls([])
        assert g.blocks("anything") is False
        assert g.score("anything") == 0.0


def test_conflicting_results_or_and():
    blocker = AlwaysBlock()
    passer = NeverBlock()

    or_guard = ORGuardrail([blocker, passer])
    assert or_guard.blocks("text") is True
    assert or_guard.score("text") == 1.0

    and_guard = ANDGuardrail([blocker, passer])
    assert and_guard.blocks("text") is False
    assert and_guard.score("text") == 0.0


def test_calibrate_delegation():
    blocker = AlwaysBlock()
    passer = NeverBlock()
    g = ORGuardrail([blocker, passer])
    g.calibrate(["benign"], 0.1)
    assert getattr(blocker, "calibrated", False) is True
    assert getattr(passer, "calibrated", False) is True