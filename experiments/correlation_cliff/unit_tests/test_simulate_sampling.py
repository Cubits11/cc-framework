import numpy as np
import pytest

from experiments.correlation_cliff.simulate import sampling


def test_rng_for_cell_stable_and_distinct():
    a = sampling.rng_for_cell(seed=123, rep=0, lam_index=5, world=1).integers(0, 2**32, size=10)
    b = sampling.rng_for_cell(seed=123, rep=0, lam_index=5, world=1).integers(0, 2**32, size=10)
    assert np.array_equal(a, b)

    c = sampling.rng_for_cell(seed=123, rep=0, lam_index=5, world=2).integers(0, 2**32, size=10)
    assert not np.array_equal(a, c)


def test_validate_cell_probs_accepts_and_clips_tiny_negative():
    p = np.array([-5e-8, 0.5, 0.25, 0.25 + 5e-8], dtype=np.float64)
    out = sampling.validate_cell_probs(
        p,
        prob_tol=1e-6,
        allow_tiny_negative=True,
        tiny_negative_eps=1e-6,
    )
    assert out.shape == (4,)
    assert np.all(out >= 0.0)
    assert np.isclose(out.sum(), 1.0, atol=1e-6, rtol=0.0)


def test_validate_cell_probs_with_meta_tracks_clipping():
    p = np.array([-1e-8, 0.5, 0.25, 0.25000001], dtype=np.float64)
    out, meta = sampling.validate_cell_probs_with_meta(
        p,
        prob_tol=1e-6,
        allow_tiny_negative=True,
        tiny_negative_eps=1e-6,
    )
    assert out.shape == (4,)
    assert meta["clipped_any"] == 1.0
    assert meta["clipped_low"] == 1.0
    assert meta["clipped_high"] in (0.0, 1.0)


def test_validate_cell_probs_sum_tolerance_boundary():
    p_ok = np.array([0.25, 0.25, 0.25, 0.2500009], dtype=np.float64)
    out = sampling.validate_cell_probs(
        p_ok,
        prob_tol=1e-6,
        allow_tiny_negative=False,
        tiny_negative_eps=1e-6,
    )
    assert np.isclose(out.sum(), 1.0, atol=1e-6, rtol=0.0)

    p_bad = np.array([0.25, 0.25, 0.25, 0.250002], dtype=np.float64)
    with pytest.raises(ValueError, match="do not sum to 1"):
        sampling.validate_cell_probs(
            p_bad,
            prob_tol=1e-6,
            allow_tiny_negative=False,
            tiny_negative_eps=1e-6,
        )


def test_validate_cell_probs_clips_tiny_over_one():
    p = np.array([0.1, 0.2, 0.3, 0.4000000005], dtype=np.float64)
    out = sampling.validate_cell_probs(
        p,
        prob_tol=1e-6,
        allow_tiny_negative=True,
        tiny_negative_eps=1e-6,
    )
    assert np.all(out <= 1.0)
    assert np.isclose(out.sum(), 1.0, atol=1e-6, rtol=0.0)


def test_validate_cell_probs_rejects_bad_sum():
    p = np.array([0.25, 0.25, 0.25, 0.30], dtype=np.float64)
    with pytest.raises(ValueError, match="do not sum to 1"):
        sampling.validate_cell_probs(
            p,
            prob_tol=1e-6,
            allow_tiny_negative=False,
            tiny_negative_eps=1e-6,
        )


def test_validate_cell_probs_rejects_large_negative_even_if_allow_tiny():
    p = np.array([-1e-3, 0.5, 0.25, 0.251], dtype=np.float64)
    with pytest.raises(ValueError, match="Negative cell probability"):
        sampling.validate_cell_probs(
            p,
            prob_tol=1e-6,
            allow_tiny_negative=True,
            tiny_negative_eps=1e-6,
        )


def test_validate_cell_probs_rejects_nonfinite():
    p = np.array([0.25, np.nan, 0.25, 0.5], dtype=np.float64)
    with pytest.raises(ValueError, match="Non-finite"):
        sampling.validate_cell_probs(
            p,
            prob_tol=1e-6,
            allow_tiny_negative=False,
            tiny_negative_eps=1e-6,
        )


def test_draw_joint_counts_deterministic_single_cell():
    rng = np.random.default_rng(0)
    (n00, n01, n10, n11), meta = sampling.draw_joint_counts(
        rng,
        n=5,
        p00=0.0,
        p01=0.0,
        p10=0.0,
        p11=1.0,
        prob_tol=1e-12,
        allow_tiny_negative=False,
        tiny_negative_eps=1e-6,
        return_meta=True,
    )
    assert (n00, n01, n10, n11) == (0, 0, 0, 5)
    assert np.isclose(meta["p11_used"], 1.0)
    assert meta["p11_mismatch"] <= meta["p11_mismatch_tol"]


def test_draw_joint_counts_counts_sum_to_n():
    rng = np.random.default_rng(123)
    n = 100
    (n00, n01, n10, n11), meta = sampling.draw_joint_counts(
        rng,
        n=n,
        p00=0.1,
        p01=0.2,
        p10=0.3,
        p11=0.4,
        prob_tol=1e-12,
        allow_tiny_negative=False,
        tiny_negative_eps=1e-6,
        return_meta=True,
    )
    assert all(isinstance(x, int) for x in (n00, n01, n10, n11))
    assert n00 + n01 + n10 + n11 == n
    assert all(x >= 0 for x in (n00, n01, n10, n11))
    assert meta["sum_error"] <= 1e-12


def test_draw_joint_counts_rejects_non_generator_rng():
    with pytest.raises(TypeError, match="rng must be a numpy.random.Generator"):
        sampling.draw_joint_counts(
            rng="not a generator",  # type: ignore[arg-type]
            n=10,
            p00=0.25,
            p01=0.25,
            p10=0.25,
            p11=0.25,
            prob_tol=1e-12,
            allow_tiny_negative=False,
            tiny_negative_eps=1e-6,
        )


@pytest.mark.parametrize("bad_n", [0, -1, 5.0, True])
def test_draw_joint_counts_rejects_bad_n(bad_n):
    rng = np.random.default_rng(0)
    with pytest.raises((TypeError, ValueError)):
        sampling.draw_joint_counts(
            rng,
            n=bad_n,  # type: ignore[arg-type]
            p00=0.25,
            p01=0.25,
            p10=0.25,
            p11=0.25,
            prob_tol=1e-12,
            allow_tiny_negative=False,
            tiny_negative_eps=1e-6,
        )


def test_draw_joint_counts_batch_shape_and_sum():
    rng = np.random.default_rng(321)
    counts, meta = sampling.draw_joint_counts_batch(
        rng,
        n=50,
        p00=0.1,
        p01=0.2,
        p10=0.3,
        p11=0.4,
        size=5,
        prob_tol=1e-12,
        allow_tiny_negative=False,
        tiny_negative_eps=1e-6,
        return_meta=True,
    )
    assert counts.shape == (5, 4)
    assert np.all(counts.sum(axis=1) == 50)
    assert meta["p11_mismatch"] <= meta["p11_mismatch_tol"]


@pytest.mark.parametrize("bad_size", [0, -1, 3.2, True])
def test_draw_joint_counts_batch_rejects_bad_size(bad_size):
    rng = np.random.default_rng(0)
    with pytest.raises((TypeError, ValueError)):
        sampling.draw_joint_counts_batch(
            rng,
            n=10,
            p00=0.25,
            p01=0.25,
            p10=0.25,
            p11=0.25,
            size=bad_size,  # type: ignore[arg-type]
            prob_tol=1e-12,
            allow_tiny_negative=False,
            tiny_negative_eps=1e-6,
        )


def test_empirical_from_counts_or_basic_invariants():
    out = sampling.empirical_from_counts(
        n=10,
        n00=1,
        n01=2,
        n10=3,
        n11=4,
        rule="OR",
    )
    # core probabilities
    assert np.isclose(out["p00_hat"], 0.1)
    assert np.isclose(out["p11_hat"], 0.4)
    assert np.isclose(out["pA_hat"], (3 + 4) / 10.0)
    assert np.isclose(out["pB_hat"], (2 + 4) / 10.0)

    # OR should be pA + pB - p11
    expected_or = out["pA_hat"] + out["pB_hat"] - out["p11_hat"]
    assert np.isclose(out["pC_hat"], expected_or)

    # flags present
    assert out["phi_finite"] in (0.0, 1.0)
    assert out["tau_finite"] in (0.0, 1.0)


def test_empirical_from_counts_and_rule_is_p11():
    out = sampling.empirical_from_counts(
        n=10,
        n00=1,
        n01=2,
        n10=3,
        n11=4,
        rule="AND",
    )
    assert np.isclose(out["pC_hat"], out["p11_hat"])


def test_empirical_from_counts_rejects_bad_rule():
    with pytest.raises(ValueError, match="Invalid rule"):
        sampling.empirical_from_counts(
            n=10,
            n00=1,
            n01=2,
            n10=3,
            n11=4,
            rule="XOR",  # type: ignore[arg-type]
        )


def test_empirical_from_counts_rejects_sum_mismatch():
    with pytest.raises(ValueError, match="Counts do not sum to n"):
        sampling.empirical_from_counts(
            n=10,
            n00=1,
            n01=2,
            n10=3,
            n11=5,
            rule="OR",
        )


def test_empirical_from_counts_degeneracy_flags_exact():
    # A always 0 (n10+n11 == 0)
    out = sampling.empirical_from_counts(
        n=10,
        n00=7,
        n01=3,
        n10=0,
        n11=0,
        rule="OR",
    )
    assert out["degenerate_A"] == 1.0
    # B not degenerate (n01+n11 == 3)
    assert out["degenerate_B"] == 0.0
