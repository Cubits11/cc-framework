import pandas as pd

from experiments.correlation_cliff.simulate.config import SimConfig
from experiments.correlation_cliff.simulate.core import simulate_grid


def test_simulate_replicate_metadata_present():
    cfg = SimConfig(
        marginals={"w0": {"pA": 0.5, "pB": 0.5}, "w1": {"pA": 0.6, "pB": 0.6}},
        rule="OR",
        lambdas=[0.5],
        n=100,
        n_reps=3,
        seed=42,
        path="fh_linear",
    )
    df = simulate_grid(cfg)

    for w in (0, 1):
        assert f"sample_meta_sum_error_w{w}" in df.columns
        assert f"sample_meta_clipped_any_w{w}" in df.columns

    valid = df[df["world_valid_w0"] & df["world_valid_w1"]]
    assert valid["sample_meta_sum_error_w0"].notna().mean() > 0.95


def test_envelope_violation_boundary():
    cfg = SimConfig(
        marginals={"w0": {"pA": 0.5, "pB": 0.5}, "w1": {"pA": 0.5, "pB": 0.5}},
        rule="OR",
        lambdas=[0.0, 0.5, 1.0],
        n=500,
        n_reps=3,
        seed=0,
        path="fh_linear",
        envelope_tol=5e-3,
    )
    df = simulate_grid(cfg)
    valid = df[df["world_valid_w0"] & df["world_valid_w1"]]
    for _, row in valid.iterrows():
        jc_hat = row["JC_hat"]
        env_min = row["JC_env_min"]
        env_max = row["JC_env_max"]
        tol = cfg.envelope_tol
        expected_violation = (jc_hat < env_min - tol) or (jc_hat > env_max + tol)
        assert bool(row["JC_env_violation"]) == expected_violation


def test_batch_sampling_mode_toggles():
    cfg_batch = SimConfig(
        marginals={"w0": {"pA": 0.5, "pB": 0.5}, "w1": {"pA": 0.6, "pB": 0.6}},
        rule="OR",
        lambdas=[0.5],
        n=100,
        n_reps=5,
        seed=42,
        path="fh_linear",
        seed_policy="sequential",
        batch_sampling=True,
    )
    cfg_std = SimConfig(
        marginals={"w0": {"pA": 0.5, "pB": 0.5}, "w1": {"pA": 0.6, "pB": 0.6}},
        rule="OR",
        lambdas=[0.5],
        n=100,
        n_reps=5,
        seed=42,
        path="fh_linear",
        seed_policy="sequential",
        batch_sampling=False,
    )

    df_batch = simulate_grid(cfg_batch)
    df_std = simulate_grid(cfg_std)

    assert df_batch["batch_sampling_used"].iloc[0] == 1.0
    assert df_std["batch_sampling_used"].iloc[0] == 0.0
    assert df_batch["seed_policy_applied"].iloc[0] == "sequential_batch"
    assert df_std["seed_policy_applied"].iloc[0] == "sequential_standard"

    differs = (df_batch["n00_w0"].to_numpy() != df_std["n00_w0"].to_numpy()).any()
    assert differs

    for df, label in [(df_batch, "batch"), (df_std, "std")]:
        valid = df[df["world_valid_w0"] & df["world_valid_w1"]]
        nan_rate = valid["CC_hat"].isna().mean()
        assert nan_rate < 0.05, f"{label}: too many NaNs in CC_hat ({nan_rate:.1%})"


def test_determinism_stable_per_cell():
    cfg = SimConfig(
        marginals={"w0": {"pA": 0.5, "pB": 0.5}, "w1": {"pA": 0.6, "pB": 0.6}},
        rule="OR",
        lambdas=[0.5],
        n=100,
        n_reps=3,
        seed=42,
        path="fh_linear",
        seed_policy="stable_per_cell",
    )

    df1 = simulate_grid(cfg)
    df2 = simulate_grid(cfg)

    pd.testing.assert_frame_equal(
        df1[["CC_hat", "JC_hat", "phi_hat_w0"]],
        df2[["CC_hat", "JC_hat", "phi_hat_w0"]],
        check_dtype=False,
    )


def test_lambda_ordering_stable_per_cell():
    lambdas1 = [0.0, 0.25, 0.5, 0.75, 1.0]
    lambdas2 = [1.0, 0.5, 0.0, 0.75, 0.25]

    cfg_base = {
        "marginals": {"w0": {"pA": 0.5, "pB": 0.5}, "w1": {"pA": 0.6, "pB": 0.6}},
        "rule": "OR",
        "n": 100,
        "n_reps": 3,
        "seed": 42,
        "path": "fh_linear",
        "seed_policy": "stable_per_cell",
    }

    df1 = simulate_grid(SimConfig(**cfg_base, lambdas=lambdas1))
    df2 = simulate_grid(SimConfig(**cfg_base, lambdas=lambdas2))

    df1_sorted = df1.sort_values(["lambda", "rep"]).reset_index(drop=True)
    df2_sorted = df2.sort_values(["lambda", "rep"]).reset_index(drop=True)

    pd.testing.assert_frame_equal(
        df1_sorted[["lambda", "rep", "CC_hat", "JC_hat", "phi_hat_w0"]],
        df2_sorted[["lambda", "rep", "CC_hat", "JC_hat", "phi_hat_w0"]],
        check_dtype=False,
    )
