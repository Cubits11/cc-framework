from __future__ import annotations

import numpy as np

from cc.kernel.sequential import (
    AnytimeBernoulliTester,
    calibrate_false_stop_rate,
    independent_joint_miss_baseline,
    simulate_power_curve,
)


def test_update_many_matches_incremental_updates() -> None:
    outcomes = np.array([0, 0, 1, 0, 1, 1, 1, 0, 1], dtype=int)

    incremental = AnytimeBernoulliTester(null_rate=0.1, alpha=0.05)
    for outcome in outcomes:
        incremental_result = incremental.update(int(outcome))

    batch = AnytimeBernoulliTester(null_rate=0.1, alpha=0.05)
    batch_result = batch.update_many(outcomes)

    assert batch_result.n == incremental_result.n
    assert batch_result.successes == incremental_result.successes
    assert batch_result.stop_time == incremental_result.stop_time
    assert batch_result.e_value == incremental_result.e_value


def test_null_false_stop_rate_is_controlled_for_fixed_seed() -> None:
    result = calibrate_false_stop_rate(
        rng=np.random.default_rng(123),
        null_rate=0.1,
        alpha=0.05,
        n_trials=1_000,
        n_max=1_000,
    )

    assert result.false_stop_rate <= result.alpha + 2.0 * result.monte_carlo_se
    assert result.within_two_se is True


def test_power_curve_improves_with_larger_effects() -> None:
    rows = simulate_power_curve(
        rng=np.random.default_rng(456),
        null_rate=0.1,
        true_rates=np.array([0.15, 0.25]),
        alpha=0.05,
        n_trials=200,
        n_max=1_000,
    )

    assert rows[1].stop_probability > rows[0].stop_probability
    assert rows[1].expected_stop_time < rows[0].expected_stop_time


def test_independent_joint_miss_baseline() -> None:
    assert independent_joint_miss_baseline(0.2, 0.3) == 0.06
