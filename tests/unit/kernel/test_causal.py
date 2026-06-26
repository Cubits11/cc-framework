from __future__ import annotations

from cc.kernel.causal import (
    compare_naive_vs_clustered_uncertainty,
    estimate_clustered_ate,
    generate_synthetic_clustered_two_world,
    run_coverage_simulation,
)


def test_cluster_bootstrap_estimator_is_seeded_and_reproducible() -> None:
    data = generate_synthetic_clustered_two_world(
        n_clusters=24,
        cluster_size=6,
        icc=0.2,
        treatment_effect=0.15,
        seed=100,
    )

    first = estimate_clustered_ate(
        data.outcome,
        data.world,
        data.cluster,
        bootstrap_reps=200,
        seed=123,
    )
    second = estimate_clustered_ate(
        data.outcome,
        data.world,
        data.cluster,
        bootstrap_reps=200,
        seed=123,
    )

    assert first.ate == second.ate
    assert first.se == second.se
    assert first.ci_lower == second.ci_lower
    assert first.ci_upper == second.ci_upper
    assert first.method == "cluster_bootstrap_ate"


def test_naive_variance_understates_clustered_uncertainty_on_known_icc_data() -> None:
    comparison = compare_naive_vs_clustered_uncertainty(
        n_clusters=32,
        cluster_size=8,
        icc=0.35,
        treatment_effect=0.2,
        bootstrap_reps=240,
        seed=7,
    )

    assert comparison["true_icc"] == 0.35
    assert comparison["cluster_se"] > comparison["naive_se"]
    assert comparison["se_understatement_ratio"] > 1.25


def test_coverage_simulation_grid_hits_nominal_within_monte_carlo_tolerance() -> None:
    rows = run_coverage_simulation(
        icc_values=[0.0, 0.25],
        cluster_sizes=[4, 10],
        n_clusters=36,
        true_effect=0.2,
        monte_carlo_reps=40,
        bootstrap_reps=120,
        seed=11,
    )

    assert len(rows) == 4
    for row in rows:
        assert row["within_tolerance"] is True
        assert (
            abs(row["cluster_bootstrap_coverage"] - row["nominal_coverage"])
            <= row["coverage_tolerance"]
        )
