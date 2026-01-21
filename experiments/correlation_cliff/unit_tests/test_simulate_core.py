import numpy as np
import pandas as pd
import pytest

from experiments.correlation_cliff.simulate import core
from experiments.correlation_cliff.simulate import utils as U
from experiments.correlation_cliff.simulate.config import SimConfig


def _marginals():
    return U.TwoWorldMarginals(
        w0=U.WorldMarginals(pA=0.2, pB=0.3),
        w1=U.WorldMarginals(pA=0.25, pB=0.35),
    )


def _base_cfg(**overrides):
    cfg = SimConfig(
        marginals=_marginals(),
        rule="OR",
        lambdas=[0.0, 0.5, 1.0],
        n=50,
        n_reps=3,
        seed=123,
        path="fh_linear",
        include_theory_reference=False,  # tests should not depend on optional overlays
        **overrides,
    )
    return cfg


def test_simulate_replicate_at_lambda_basic_schema_and_counts_sum():
    cfg = _base_cfg(seed_policy="stable_per_cell")
    row = core.simulate_replicate_at_lambda(
        cfg,
        lam=0.5,
        lam_index=cfg.lambda_index_for_seed(0.5),
        rep=0,
        rng=np.random.default_rng(999),  # ignored for stable_per_cell
    )

    assert row["lambda"] == 0.5
    assert row["rep"] == 0
    assert row["rule"] == "OR"
    assert row["path"] == "fh_linear"

    assert "world_valid_w0" in row and "world_valid_w1" in row
    assert row["world_valid_w0"] is True
    assert row["world_valid_w1"] is True
    assert row["worlds_valid"] is True
    assert row["row_ok"] is True

    # counts exist and sum to n for each world
    for w in (0, 1):
        n00 = row[f"n00_w{w}"]
        n01 = row[f"n01_w{w}"]
        n10 = row[f"n10_w{w}"]
        n11 = row[f"n11_w{w}"]
        assert all(isinstance(x, int) for x in (n00, n01, n10, n11))
        assert n00 + n01 + n10 + n11 == cfg.n


def test_simulate_grid_stable_per_cell_order_invariant():
    cfg_a = _base_cfg(seed_policy="stable_per_cell", lambdas=[0.0, 0.5, 1.0])
    cfg_b = _base_cfg(seed_policy="stable_per_cell", lambdas=[1.0, 0.5, 0.0])  # reordered

    df_a = (
        core.simulate_grid(cfg_a)
        .sort_values(["lambda", "rep"], kind="mergesort")
        .reset_index(drop=True)
    )
    df_b = (
        core.simulate_grid(cfg_b)
        .sort_values(["lambda", "rep"], kind="mergesort")
        .reset_index(drop=True)
    )

    # Select a stable subset of columns that must match exactly.
    cols = [
        "lambda",
        "rep",
        "n00_w0",
        "n01_w0",
        "n10_w0",
        "n11_w0",
        "n00_w1",
        "n01_w1",
        "n10_w1",
        "n11_w1",
        "JC_hat",
        "CC_hat",
        "worlds_valid",
        "row_ok",
    ]
    for c in cols:
        assert c in df_a.columns and c in df_b.columns

    # Expect exact equality for these deterministic outputs.
    pd.testing.assert_frame_equal(df_a[cols], df_b[cols], check_dtype=False)


def test_simulate_grid_sequential_is_order_dependent():
    cfg_a = _base_cfg(seed_policy="sequential", lambdas=[0.0, 0.5, 1.0])
    cfg_b = _base_cfg(seed_policy="sequential", lambdas=[1.0, 0.5, 0.0])

    df_a = (
        core.simulate_grid(cfg_a)
        .sort_values(["lambda", "rep"], kind="mergesort")
        .reset_index(drop=True)
    )
    df_b = (
        core.simulate_grid(cfg_b)
        .sort_values(["lambda", "rep"], kind="mergesort")
        .reset_index(drop=True)
    )

    # Very high probability these differ; we check a concrete column.
    assert not np.array_equal(df_a["n11_w0"].to_numpy(), df_b["n11_w0"].to_numpy())


def test_hard_fail_false_marks_world_invalid_instead_of_raising(monkeypatch):
    cfg = _base_cfg(hard_fail_on_invalid=False, seed_policy="stable_per_cell", lambdas=[0.5])

    def _bad_p11_from_path(pA, pB, lam, *, path, path_params):
        # Make world 1 fail by keying off its pA value (0.25 in _marginals()).
        if abs(float(pA) - 0.25) < 1e-12:
            raise RuntimeError("boom")
        return float(pA) * float(pB), {"ok": 1.0}

    monkeypatch.setattr(core, "p11_from_path", _bad_p11_from_path)

    df = core.simulate_grid(cfg)
    assert len(df) == cfg.n_reps * len(cfg.lambdas)

    row0 = df.iloc[0].to_dict()
    assert row0["world_valid_w0"] is True
    assert row0["world_valid_w1"] is False
    assert row0["worlds_valid"] is False
    assert row0["row_ok"] is False
    assert row0["world_error_stage_w1"] == "p11_from_path"
    assert "boom" in str(row0["world_error_msg_w1"])


def test_simulate_grid_hard_fail_false_catches_row_exception(monkeypatch):
    cfg = _base_cfg(hard_fail_on_invalid=False, seed_policy="stable_per_cell", lambdas=[0.0, 0.5])

    def _explode(*args, **kwargs):
        raise RuntimeError("replicate explode")

    monkeypatch.setattr(core, "simulate_replicate_at_lambda", _explode)

    df = core.simulate_grid(cfg)
    assert len(df) == cfg.n_reps * len(cfg.lambdas)
    assert df["row_ok"].fillna(False).astype(bool).sum() == 0
    assert (df["row_error_stage"] == "simulate_replicate_at_lambda").all()


def test_summarize_simulation_outputs_expected_columns():
    cfg = _base_cfg(seed_policy="stable_per_cell", lambdas=[0.0, 0.5], n_reps=5)
    df = core.simulate_grid(cfg)
    df_sum = core.summarize_simulation(df)

    assert len(df_sum) == 2
    assert "lambda" in df_sum.columns
    assert "CC_hat_mean" in df_sum.columns
    assert "CC_hat_q0025" in df_sum.columns
    assert "CC_hat_q0500" in df_sum.columns
    assert "CC_hat_q0975" in df_sum.columns
    assert "row_ok_rate" in df_sum.columns


def test_summarize_simulation_rejects_duplicate_quantiles_after_rounding():
    df = pd.DataFrame({"lambda": [0.0, 0.0], "rep": [0, 1], "CC_hat": [1.0, 2.0]})
    with pytest.raises(ValueError, match="duplicates"):
        core.summarize_simulation(df, quantiles=(0.5, 0.5000000000000001))
