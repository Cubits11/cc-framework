import pytest

from experiments.correlation_cliff.simulate.config import ConfigError, SimConfig, _strict_int


def test_strict_int_rejects_float_coercion():
    with pytest.raises(ConfigError, match="no silent coercion"):
        _strict_int(100.0, "n")

    with pytest.raises(ConfigError, match="bool"):
        _strict_int(True, "n")

    assert _strict_int(100, "n") == 100


def test_sim_config_n_rejects_float():
    with pytest.raises(ConfigError, match="no silent coercion"):
        SimConfig(
            marginals=dict(w0=dict(pA=0.5, pB=0.5), w1=dict(pA=0.6, pB=0.6)),
            rule="OR",
            lambdas=[0.5],
            n=100.0,
            n_reps=10,
            seed=0,
            path="fh_linear",
        )


def test_batch_sampling_flag_validation():
    cfg = SimConfig(
        marginals=dict(w0=dict(pA=0.5, pB=0.5), w1=dict(pA=0.6, pB=0.6)),
        rule="OR",
        lambdas=[0.5],
        n=100,
        n_reps=10,
        seed=0,
        path="fh_linear",
        batch_sampling=True,
    )
    assert cfg.batch_sampling is True

    with pytest.raises(ConfigError, match="batch_sampling must be bool"):
        SimConfig(
            marginals=dict(w0=dict(pA=0.5, pB=0.5), w1=dict(pA=0.6, pB=0.6)),
            rule="OR",
            lambdas=[0.5],
            n=100,
            n_reps=10,
            seed=0,
            path="fh_linear",
            batch_sampling="yes",  # type: ignore[arg-type]
        )
