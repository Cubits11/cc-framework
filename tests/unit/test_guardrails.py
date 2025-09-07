import math
from cc.guardrails.keyword_blocker import KeywordBlocker, KeywordBlockerConfig
from cc.core.guardrail_api import GuardrailAdapter

def test_keyword_blocker_exact_and_fuzzy():
    kb = KeywordBlocker(["api_key", "token"])
    assert kb.score("leak TOKEN now") > 0.0
    assert kb.blocks("api_key=abcd") is (kb.score("api_key=abcd") > kb.blocking_threshold)

def test_keyword_blocker_calibration_ties_pass():
    benign = ["hello world", "nice day", "just chatting"]
    kb = KeywordBlocker(["secret"], KeywordBlockerConfig(verbose=False, initial_threshold=0.0))
    kb.calibrate(benign_texts=benign, target_fpr=0.0)
    # With target_fpr=0, threshold at max(score); ties (==) must pass
    for t in benign:
        assert kb.score(t) <= kb.blocking_threshold

def test_keyword_blocker_quantile_fpr():
    benign = ["normal text", "chit chat", "contains secret sauce", "super safe"]
    kb = KeywordBlocker(["secret"], KeywordBlockerConfig(verbose=False))
    kb.calibrate(benign_texts=benign, target_fpr=0.25)
    # Approximately 25% should be strictly above threshold
    above = sum(kb.score(t) > kb.blocking_threshold for t in benign)
    assert above in (0,1,2)  # small-sample tolerance

def test_keywords_list_sorted():
    kb = KeywordBlocker(["b", "a", "c"])
    assert kb.keywords_list() == ["a","b","c"]

def test_guardrail_adapter_interface():
    benign = ["hello", "world"]
    kb = KeywordBlocker(["secret"], KeywordBlockerConfig(verbose=False))
    api = GuardrailAdapter(kb)
    api.calibrate(benign, target_fpr=0.0)
    blocked, score = api.evaluate("a secret appears")
    assert isinstance(blocked, bool)
    assert isinstance(score, float)
    assert blocked is (score > kb.blocking_threshold)
