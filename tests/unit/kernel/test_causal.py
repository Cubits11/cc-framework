from __future__ import annotations

import numpy as np

from cc.kernel import causal
from cc.kernel.causal import (
    compare_naive_vs_clustered_uncertainty,
    difference_in_means,
    estimate_clustered_ate,
    generate_synthetic_clustered_two_world,
    run_coverage_simulation,
)


def _materialized_bootstrap_effects(
    outcome: np.ndarray,
    world: np.ndarray,
    cluster: np.ndarray,
    *,
    bootstrap_reps: int,
    seed: int,
) -> np.ndarray:
    unique_clusters = np.unique(cluster)
    n_clusters = int(unique_clusters.size)
    rng = np.random.default_rng(seed)
    cluster_indices = [np.flatnonzero(cluster == label) for label in unique_clusters]
    effects: list[float] = []
    for _ in range(int(bootstrap_reps)):
        sampled = rng.integers(0, n_clusters, size=n_clusters)
        idx = np.concatenate([cluster_indices[int(j)] for j in sampled])
        wb = world[idx]
        if not np.any(wb == 0) or not np.any(wb == 1):
            continue
        effects.append(difference_in_means(outcome[idx], wb))
    return np.asarray(effects, dtype=float)


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


def test_cluster_bootstrap_matches_materialized_reference_with_variable_clusters() -> None:
    outcome = np.asarray(
        [
            1.0,
            1.2,
            1.4,
            0.2,
            0.4,
            0.8,
            0.9,
            1.1,
            -0.2,
            -0.1,
            0.0,
            0.3,
            0.5,
            0.7,
        ],
        dtype=float,
    )
    world = np.asarray([1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0], dtype=int)
    cluster = np.asarray([0, 0, 0, 1, 1, 2, 2, 2, 3, 4, 4, 4, 5, 5], dtype=int)
    bootstrap_reps = 400
    seed = 202

    reference = _materialized_bootstrap_effects(
        outcome,
        world,
        cluster,
        bootstrap_reps=bootstrap_reps,
        seed=seed,
    )
    estimate = estimate_clustered_ate(
        outcome,
        world,
        cluster,
        bootstrap_reps=bootstrap_reps,
        seed=seed,
    )

    assert estimate.bootstrap_valid_reps == reference.size
    assert np.isclose(estimate.ate, difference_in_means(outcome, world))
    assert np.isclose(estimate.se, np.std(reference, ddof=1))
    assert np.isclose(estimate.ci_lower, np.quantile(reference, 0.025))
    assert np.isclose(estimate.ci_upper, np.quantile(reference, 0.975))


def test_cluster_bootstrap_matches_materialized_reference_with_single_world_clusters() -> None:
    rng = np.random.default_rng(77)
    cluster_sizes = np.asarray([3, 5, 2, 6, 4, 7, 3, 5], dtype=int)
    cluster = np.repeat(np.arange(cluster_sizes.size), cluster_sizes)
    world_by_cluster = np.asarray([0, 1, 0, 1, 0, 1, 0, 1], dtype=int)
    world = np.repeat(world_by_cluster, cluster_sizes)
    outcome = rng.normal(0.0, 0.5, size=world.size) + 0.4 * world
    bootstrap_reps = 500
    seed = 8

    reference = _materialized_bootstrap_effects(
        outcome,
        world,
        cluster,
        bootstrap_reps=bootstrap_reps,
        seed=seed,
    )
    estimate = estimate_clustered_ate(
        outcome,
        world,
        cluster,
        bootstrap_reps=bootstrap_reps,
        seed=seed,
    )

    assert reference.size == estimate.bootstrap_valid_reps
    assert np.isfinite([estimate.se, estimate.ci_lower, estimate.ci_upper]).all()
    assert np.isclose(estimate.se, np.std(reference, ddof=1))
    assert np.isclose(estimate.ci_lower, np.quantile(reference, 0.025))
    assert np.isclose(estimate.ci_upper, np.quantile(reference, 0.975))


def test_cluster_bootstrap_does_not_materialize_observation_resamples(monkeypatch) -> None:
    data = generate_synthetic_clustered_two_world(
        n_clusters=12,
        cluster_size=4,
        icc=0.1,
        treatment_effect=0.2,
        seed=13,
    )

    calls = 0
    original_difference_in_means = causal.difference_in_means

    def counted_difference_in_means(outcome, world):  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        if calls > 1:
            raise AssertionError("bootstrap replicates should use cluster aggregates")
        return original_difference_in_means(outcome, world)

    monkeypatch.setattr(causal, "difference_in_means", counted_difference_in_means)

    estimate = estimate_clustered_ate(
        data.outcome,
        data.world,
        data.cluster,
        bootstrap_reps=120,
        seed=14,
    )
    assert estimate.bootstrap_valid_reps > 0
    assert calls == 1


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
