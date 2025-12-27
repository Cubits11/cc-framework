import pytest

from experiments.correlation_cliff.simulate.config import SimConfig, validate_cfg
from experiments.correlation_cliff.simulate import utils as U


def _marginals():
    return U.TwoWorldMarginals(
        w0=U.WorldMarginals(pA=0.2, pB=0.3),
        w1=U.WorldMarginals(pA=0.25, pB=0.35),
    )


def test_simconfig_normalizes_rule():
    cfg = SimConfig(marginals=_marginals(), rule="or", lambdas=[0.0, 1.0], n=10)
    assert cfg.rule == "OR"


def test_simconfig_rejects_duplicate_lambdas():
    with pytest.raises(ValueError):
        SimConfig(marginals=_marginals(), rule="OR", lambdas=[0.0, 0.0, 1.0], n=10)


def test_lambda_index_for_seed_canonical():
    cfg = SimConfig(marginals=_marginals(), rule="OR", lambdas=[1.0, 0.5, 0.0], n=10)
    # canonical order is [0.0, 0.5, 1.0]
    assert cfg.lambda_index_for_seed(0.0) == 0
    assert cfg.lambda_index_for_seed(0.5) == 1
    assert cfg.lambda_index_for_seed(1.0) == 2


def test_validate_cfg_rejects_bad_prob_tol():
    cfg = SimConfig(marginals=_marginals(), rule="OR", lambdas=[0.0, 1.0], n=10, prob_tol=1e-2)
    with pytest.raises(ValueError):
        validate_cfg(cfg)
