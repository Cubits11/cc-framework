import pytest

from experiments.correlation_cliff.simulate.config import ConfigError, SimConfig
from experiments.correlation_cliff.simulate import utils as U


def _marginals():
    return U.TwoWorldMarginals(
        w0=U.WorldMarginals(pA=0.2, pB=0.3),
        w1=U.WorldMarginals(pA=0.25, pB=0.35),
    )


def _base_kwargs():
    return dict(
        marginals=_marginals(),
        rule="OR",
        lambdas=[0.0, 0.5, 1.0],
        n=10,
        n_reps=3,
        seed=123,
        path="fh_linear",
    )


def test_simconfig_normalizes_rule_and_path_and_seed_policy():
    cfg = SimConfig(**{**_base_kwargs(), "rule": "or", "path": "FH_LINEAR", "seed_policy": "SEQUENTIAL"})
    assert cfg.rule == "OR"
    assert cfg.path == "fh_linear"
    assert cfg.seed_policy == "sequential"


def test_simconfig_rejects_duplicate_lambdas_after_rounding():
    with pytest.raises(ConfigError, match="duplicates"):
        SimConfig(**{**_base_kwargs(), "lambdas": [0.0, 0.0, 1.0]})


def test_lambda_index_for_seed_canonical_for_stable_per_cell_unsorted_allowed():
    # stable_per_cell is default, so unsorted lambdas are allowed; mapping is canonical sorted-by-value.
    cfg = SimConfig(**{**_base_kwargs(), "lambdas": [1.0, 0.5, 0.0]})
    assert cfg.lambda_index_for_seed(0.0) == 0
    assert cfg.lambda_index_for_seed(0.5) == 1
    assert cfg.lambda_index_for_seed(1.0) == 2


def test_sequential_policy_requires_nondecreasing_lambdas():
    with pytest.raises(ConfigError, match="non-decreasing"):
        SimConfig(**{**_base_kwargs(), "seed_policy": "sequential", "lambdas": [1.0, 0.5, 0.0]})


def test_rejects_bad_prob_tol():
    with pytest.raises(ConfigError, match="prob_tol"):
        SimConfig(**{**_base_kwargs(), "prob_tol": 1e-2})


@pytest.mark.parametrize("bad_n", [True, 0, -1])
def test_rejects_bad_n(bad_n):
    with pytest.raises(ConfigError):
        SimConfig(**{**_base_kwargs(), "n": bad_n})


@pytest.mark.parametrize("bad_seed", [True, -1])
def test_rejects_bad_seed(bad_seed):
    with pytest.raises(ConfigError):
        SimConfig(**{**_base_kwargs(), "seed": bad_seed})


def test_accepts_marginals_mapping_input():
    m = {"w0": {"pA": 0.2, "pB": 0.3}, "w1": {"pA": 0.25, "pB": 0.35}}
    cfg = SimConfig(**{**_base_kwargs(), "marginals": m})
    assert cfg.marginals.w0.pA == 0.2
    assert cfg.marginals.w1.pB == 0.35
