from experiments.correlation_cliff.simulate.cli import _cfg_from_dict


def test_cfg_from_dict_legacy_schema():
    d = {
        "marginals": {"w0": {"pA": 0.2, "pB": 0.3}, "w1": {"pA": 0.25, "pB": 0.35}},
        "rule": "OR",
        "n": 100,
        "n_reps": 3,
        "seed": 7,
        "lambdas": [0.0, 0.5, 1.0],
        "path": "fh_linear",
        "seed_policy": "stable_per_cell",
    }
    cfg = _cfg_from_dict(d)
    assert cfg.n == 100
    assert cfg.n_reps == 3
    assert cfg.seed == 7
    assert list(cfg.lambdas) == [0.0, 0.5, 1.0]
    assert cfg.path == "fh_linear"


def test_cfg_from_dict_pipeline_schema():
    d = {
        "marginals": {"w0": {"pA": 0.2, "pB": 0.3}, "w1": {"pA": 0.25, "pB": 0.35}},
        "composition": {"primary_rule": "AND"},
        "dependence_paths": {
            "primary": {"type": "fh_power", "gamma": 2.0, "lambda_grid_coarse": {"num": 5}}
        },
        "sampling": {"n_per_world": 50, "n_reps": 2, "seed": 123, "seed_policy": "stable_per_cell"},
        "simulate": {"prob_tol": 1e-12, "allow_tiny_negative": True, "tiny_negative_eps": 1e-15},
    }
    cfg = _cfg_from_dict(d)
    assert cfg.rule == "AND"
    assert cfg.path == "fh_power"
    assert cfg.path_params.get("gamma") == 2.0
    assert cfg.n == 50
    assert cfg.n_reps == 2
    assert len(cfg.lambdas) == 5
