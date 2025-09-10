from cc.core.guardrail_api import GuardrailAdapter


class DummyWithBlocks:
    threshold = 0.4

    def score(self, text: str) -> float:
        return 0.3 if "ok" in text else 0.9

    def blocks(self, text: str) -> bool:
        return self.score(text) > self.threshold


class DummyScoreOnly:
    threshold = 0.6

    def score(self, text: str) -> float:
        return 0.8


def test_evaluate_consistent_with_blocks_and_score_when_blocks_present():
    g = GuardrailAdapter(DummyWithBlocks())
    blocked, score = g.evaluate("ok")
    assert (blocked, score) == (g.blocks("ok"), g.score("ok"))


def test_evaluate_consistent_with_blocks_and_score_without_blocks():
    g = GuardrailAdapter(DummyScoreOnly())
    blocked, score = g.evaluate("anything")
    assert (blocked, score) == (g.blocks("anything"), g.score("anything"))
