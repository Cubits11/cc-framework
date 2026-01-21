"""
Enterprise-grade unit tests for experiments/correlation_cliff/simulate.py

Goals
-----
1) Enforce the "no silent lies" contract:
   - no silent renormalization
   - strict type handling where promised
   - explicit failure semantics

2) Enforce reproducibility contract:
   - stable_per_cell should be order-invariant (lambda order does not change results)
   - sequential is allowed to be order-dependent

3) Enforce schema stability:
   - simulate_replicate_at_lambda always emits stable identifiers + flags
   - simulate_grid returns deterministic ordering and expected row counts

Notes
-----
- These tests avoid SciPy dependence by default.
- gaussian_tau tests are conditional: they skip if SciPy is not installed.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest


# -----------------------------------------------------------------------------
# Import simulate module robustly (package mode preferred; script fallback allowed)
# -----------------------------------------------------------------------------
def _import_sim():
    try:
        from experiments.correlation_cliff import simulate_legacy as sim  # type: ignore

        return sim
    except Exception:
        import simulate_legacy as sim  # type: ignore

        return sim


sim = _import_sim()


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _base_marginals():
    """
    Choose marginals that:
    - are comfortably inside (0,1)
    - give Jbest > 0 (so CC is defined)
    """
    return sim.TwoWorldMarginals(
        w0=sim.WorldMarginals(pA=0.20, pB=0.35),
        w1=sim.WorldMarginals(pA=0.45, pB=0.35),
    )


def _make_cfg(
    *,
    lambdas=(0.0, 0.5, 1.0),
    n=50,
    n_reps=2,
    seed=123,
    rule="OR",
    path="fh_linear",
    path_params=None,
    seed_policy="stable_per_cell",
    hard_fail_on_invalid=True,
    include_theory_reference=False,
) -> "sim.SimConfig":
    if path_params is None:
        path_params = {}
    return sim.SimConfig(
        marginals=_base_marginals(),
        rule=rule,
        lambdas=list(lambdas),
        n=n,
        n_reps=n_reps,
        seed=seed,
        path=path,
        path_params=dict(path_params),
        seed_policy=seed_policy,
        hard_fail_on_invalid=hard_fail_on_invalid,
        include_theory_reference=include_theory_reference,
    )


def _have_scipy() -> bool:
    try:
        import scipy  # noqa: F401

        return True
    except Exception:
        return False


# -----------------------------------------------------------------------------
# Lambda grid builder tests
# -----------------------------------------------------------------------------
def test_build_linear_lambda_grid_basic_closed_both():
    g = sim.build_linear_lambda_grid(num=5, start=0.0, stop=1.0, closed="both")
    assert isinstance(g, np.ndarray)
    assert g.shape == (5,)
    assert float(g[0]) == 0.0
    assert float(g[-1]) == 1.0
    assert np.all(np.diff(g) > 0)


@pytest.mark.parametrize(
    "closed,expected_first,expected_last",
    [
        ("neither", 0.25, 0.75),
        ("left", 0.0, 0.75),
        ("right", 0.25, 1.0),
    ],
)
def test_build_linear_lambda_grid_closed_variants(closed, expected_first, expected_last):
    g = sim.build_linear_lambda_grid(num=4, start=0.0, stop=1.0, closed=closed)
    assert g.shape == (4,)
    assert np.all(np.diff(g) > 0)
    assert pytest.approx(float(g[0]), abs=1e-12) == expected_first
    assert pytest.approx(float(g[-1]), abs=1e-12) == expected_last


def test_build_linear_lambda_grid_rejects_bad_args():
    with pytest.raises(ValueError):
        sim.build_linear_lambda_grid(num=0)
    with pytest.raises(ValueError):
        sim.build_linear_lambda_grid(num=2, start=1.0, stop=0.0)
    with pytest.raises(ValueError):
        sim.build_linear_lambda_grid(num=2, start=-0.1, stop=1.0)
    with pytest.raises(ValueError):
        sim.build_linear_lambda_grid(num=1, closed="both")  # requires >=2


# -----------------------------------------------------------------------------
# Probability validation & sampling tests
# -----------------------------------------------------------------------------
def test_validate_cell_probs_happy_path_and_tiny_clipping():
    # tiny negative jitter should be clipped when allow_tiny_negative=True
    p = np.array([0.1, 0.2, 0.3, 0.4], dtype=float)
    p[0] = -1e-16
    p[3] = 1.0 - (p[0] + p[1] + p[2])  # keep sum ~1 but with tiny negative in p0
    out = sim._validate_cell_probs(
        p,
        prob_tol=1e-12,
        allow_tiny_negative=True,
        tiny_negative_eps=1e-15,
        context="unit-test",
    )
    assert out.shape == (4,)
    assert np.all(np.isfinite(out))
    assert float(out.min()) >= 0.0
    assert float(out.max()) <= 1.0
    assert abs(float(out.sum()) - 1.0) <= 1e-12


def test_validate_cell_probs_rejects_real_negative():
    p = np.array([-1e-6, 0.2, 0.3, 0.500001], dtype=float)  # sums ~1 but negative is real
    with pytest.raises(ValueError, match="Negative cell probability"):
        sim._validate_cell_probs(
            p,
            prob_tol=1e-9,
            allow_tiny_negative=True,
            tiny_negative_eps=1e-15,
        )


def test_validate_cell_probs_rejects_sum_mismatch_no_renormalization():
    p = np.array([0.1, 0.2, 0.3, 0.39], dtype=float)  # sum=0.99
    with pytest.raises(ValueError, match="do not sum to 1"):
        sim._validate_cell_probs(
            p,
            prob_tol=1e-12,
            allow_tiny_negative=True,
            tiny_negative_eps=1e-15,
        )


def test_draw_joint_counts_contract_and_sum():
    rng = np.random.default_rng(123)
    n00, n01, n10, n11 = sim._draw_joint_counts(
        rng,
        n=100,
        p00=0.1,
        p01=0.2,
        p10=0.3,
        p11=0.4,
        prob_tol=1e-12,
        allow_tiny_negative=True,
        tiny_negative_eps=1e-15,
        context="unit-test",
    )
    assert all(isinstance(x, int) for x in (n00, n01, n10, n11))
    assert n00 >= 0 and n01 >= 0 and n10 >= 0 and n11 >= 0
    assert (n00 + n01 + n10 + n11) == 100


def test_draw_joint_counts_rejects_float_n_even_if_integerish():
    rng = np.random.default_rng(0)
    with pytest.raises(TypeError, match="no silent coercion"):
        sim._draw_joint_counts(
            rng,
            n=100.0,
            p00=0.25,
            p01=0.25,
            p10=0.25,
            p11=0.25,
            prob_tol=1e-12,
            allow_tiny_negative=True,
            tiny_negative_eps=1e-15,
        )


# -----------------------------------------------------------------------------
# Empirical metric computation tests
# -----------------------------------------------------------------------------
def test_empirical_from_counts_or_formula_matches():
    # Construct a simple contingency table
    n = 100
    n00, n01, n10, n11 = 40, 20, 10, 30  # sums to 100
    out = sim._empirical_from_counts(n=n, n00=n00, n01=n01, n10=n10, n11=n11, rule="OR")

    pA = (n10 + n11) / n
    pB = (n01 + n11) / n
    p11 = n11 / n
    expected_pC = pA + pB - p11

    assert pytest.approx(out["pA_hat"], abs=1e-12) == pA
    assert pytest.approx(out["pB_hat"], abs=1e-12) == pB
    assert pytest.approx(out["p11_hat"], abs=1e-12) == p11
    assert pytest.approx(out["pC_hat"], abs=1e-12) == expected_pC


def test_empirical_from_counts_and_formula_matches():
    n = 100
    n00, n01, n10, n11 = 40, 20, 10, 30
    out = sim._empirical_from_counts(n=n, n00=n00, n01=n01, n10=n10, n11=n11, rule="AND")
    expected_pC = n11 / n
    assert pytest.approx(out["pC_hat"], abs=1e-12) == expected_pC


def test_empirical_from_counts_strict_type_rejections():
    with pytest.raises(TypeError):
        sim._empirical_from_counts(n=100, n00=1.0, n01=0, n10=0, n11=99, rule="OR")  # float count
    with pytest.raises(TypeError):
        sim._empirical_from_counts(n=True, n00=0, n01=0, n10=0, n11=0, rule="OR")  # bool n


# -----------------------------------------------------------------------------
# p11 path tests
# -----------------------------------------------------------------------------
def test_p11_from_path_fh_linear_in_bounds_and_meta_schema():
    pA, pB, lam = 0.2, 0.35, 0.6
    p11, meta = sim.p11_from_path(pA, pB, lam, path="fh_linear", path_params={})

    b = sim.fh_bounds(pA, pB)
    assert float(b.L) <= p11 <= float(b.U)

    # Required meta keys (numeric-only)
    for k in ("L", "U", "FH_width", "lam", "lam_eff", "raw_p11", "clip_amt", "clipped"):
        assert k in meta
        assert isinstance(meta[k], float)
        assert math.isfinite(meta[k]) or k in ("clip_amt", "clipped")  # these can be 0.0 safely


def test_p11_from_path_rejects_bad_clip_tol_policy():
    with pytest.raises(ValueError, match="clip_tol"):
        sim.p11_from_path(0.2, 0.3, 0.5, path="fh_linear", path_params={"clip_tol": 1e-3})


@pytest.mark.skipif(not _have_scipy(), reason="SciPy not installed; gaussian_tau path unavailable")
def test_p11_from_path_gaussian_tau_runs_and_meta_contains_tau_rho():
    p11, meta = sim.p11_from_path(
        0.2,
        0.35,
        0.75,
        path="gaussian_tau",
        path_params={"ppf_clip_eps": 1e-10},
    )
    b = sim.fh_bounds(0.2, 0.35)
    assert float(b.L) <= p11 <= float(b.U)
    assert "tau" in meta and "rho" in meta


@pytest.mark.skipif(
    _have_scipy(), reason="SciPy installed; gaussian_tau ImportError expectation not applicable"
)
def test_p11_from_path_gaussian_tau_raises_if_no_scipy():
    with pytest.raises(ImportError):
        sim.p11_from_path(
            0.2,
            0.35,
            0.75,
            path="gaussian_tau",
            path_params={"ppf_clip_eps": 1e-10},
        )


# -----------------------------------------------------------------------------
# SimConfig + cfg parsing validation tests
# -----------------------------------------------------------------------------
def test_simconfig_rejects_duplicate_lambdas():
    with pytest.raises(ValueError, match="duplicates"):
        _make_cfg(lambdas=(0.0, 0.5, 0.5, 1.0))


def test_validate_cfg_accepts_good_cfg():
    cfg = _make_cfg()
    sim._validate_cfg(cfg)  # should not raise


def test_cfg_from_dict_legacy_schema_smoke():
    d: Dict[str, Any] = {
        "marginals": {"w0": {"pA": 0.2, "pB": 0.35}, "w1": {"pA": 0.45, "pB": 0.35}},
        "rule": "OR",
        "n": 50,
        "n_reps": 2,
        "seed": 7,
        "lambdas": [0.0, 0.5, 1.0],
        "path": "fh_linear",
        "seed_policy": "stable_per_cell",
        "prob_tol": 1e-12,
        "allow_tiny_negative": True,
        "tiny_negative_eps": 1e-15,
        "include_theory_reference": False,
    }
    cfg = sim._cfg_from_dict(d)
    assert cfg.rule == "OR"
    assert cfg.path == "fh_linear"
    assert cfg.n == 50
    assert cfg.n_reps == 2
    assert list(cfg.lambdas) == [0.0, 0.5, 1.0]


def test_cfg_from_dict_pipeline_schema_smoke():
    d: Dict[str, Any] = {
        "marginals": {"w0": {"pA": 0.2, "pB": 0.35}, "w1": {"pA": 0.45, "pB": 0.35}},
        "composition": {"primary_rule": "OR"},
        "dependence_paths": {
            "primary": {
                "type": "fh_linear",
                "lambda_grid_coarse": {"start": 0.0, "stop": 1.0, "num": 3},
            }
        },
        "sampling": {"n_per_world": 50, "n_reps": 2, "seed": 7, "seed_policy": "stable_per_cell"},
        "simulate": {"include_theory_reference": False},
    }
    cfg = sim._cfg_from_dict(d)
    assert cfg.rule == "OR"
    assert cfg.path == "fh_linear"
    assert cfg.n == 50
    assert cfg.n_reps == 2
    assert list(cfg.lambdas) == [0.0, 0.5, 1.0]


def test_load_yaml_roundtrip_if_pyyaml_available(tmp_path: Path):
    pytest.importorskip("yaml")
    yml = tmp_path / "cfg.yaml"
    yml.write_text(
        "marginals:\n"
        "  w0: {pA: 0.2, pB: 0.35}\n"
        "  w1: {pA: 0.45, pB: 0.35}\n"
        "rule: OR\n"
        "n: 50\n"
        "n_reps: 2\n"
        "seed: 7\n"
        "lambdas: [0.0, 0.5, 1.0]\n"
        "path: fh_linear\n"
        "seed_policy: stable_per_cell\n"
        "include_theory_reference: false\n"
    )
    d = sim._load_yaml(str(yml))
    assert isinstance(d, dict)
    cfg = sim._cfg_from_dict(d)
    assert cfg.n == 50
    assert cfg.rule == "OR"


# -----------------------------------------------------------------------------
# Simulation harness tests (schema + determinism)
# -----------------------------------------------------------------------------
def test_simulate_replicate_at_lambda_schema_and_counts_sum():
    cfg = _make_cfg(n=40, n_reps=1, lambdas=(0.25,), seed_policy="stable_per_cell")
    # base_rng is ignored for stable_per_cell world RNG selection, but required by signature
    base_rng = np.random.default_rng(0)

    row = sim.simulate_replicate_at_lambda(cfg, lam=0.25, lam_index=0, rep=0, rng=base_rng)

    assert row["lambda"] == pytest.approx(0.25)
    assert row["rep"] == 0
    assert row["rule"] == "OR"
    assert "world_valid_w0" in row and "world_valid_w1" in row
    assert isinstance(row["world_valid_w0"], bool)
    assert isinstance(row["world_valid_w1"], bool)

    # If worlds are valid, counts should sum to n per world
    if row["world_valid_w0"]:
        s0 = int(row["n00_w0"]) + int(row["n01_w0"]) + int(row["n10_w0"]) + int(row["n11_w0"])
        assert s0 == cfg.n
    if row["world_valid_w1"]:
        s1 = int(row["n00_w1"]) + int(row["n01_w1"]) + int(row["n10_w1"]) + int(row["n11_w1"])
        assert s1 == cfg.n

    # JC envelope should exist
    assert "JC_env_min" in row and "JC_env_max" in row


def test_simulate_grid_row_count_and_sort_order():
    cfg = _make_cfg(n=30, n_reps=2, lambdas=(0.0, 0.5, 1.0), seed_policy="stable_per_cell")
    df = sim.simulate_grid(cfg)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2 * 3  # n_reps * n_lambdas
    assert list(df.columns)  # non-empty

    # Sorted by lambda then rep (mergesort stable)
    assert df["lambda"].is_monotonic_increasing
    # Within each lambda, rep should be non-decreasing
    for lam, g in df.groupby("lambda", sort=True):
        reps = list(pd.to_numeric(g["rep"], errors="coerce"))
        assert reps == sorted(reps)


def test_summarize_simulation_outputs_expected_quantile_columns():
    cfg = _make_cfg(n=30, n_reps=5, lambdas=(0.0, 0.5, 1.0), seed_policy="stable_per_cell")
    df = sim.simulate_grid(cfg)
    df_sum = sim.summarize_simulation(df)

    assert isinstance(df_sum, pd.DataFrame)
    assert len(df_sum) == 3  # one per lambda
    # Default quantiles produce q0025 q0500 q0975
    assert "CC_hat_q0025" in df_sum.columns
    assert "CC_hat_q0500" in df_sum.columns
    assert "CC_hat_q0975" in df_sum.columns
    assert "JC_hat_q0025" in df_sum.columns
    assert "row_ok_rate" in df_sum.columns
    assert "JC_env_violation_rate" in df_sum.columns


def test_seed_policy_stable_per_cell_order_invariant_results():
    """
    This enforces the contract:
      stable_per_cell => results do not change if lambdas are reordered.

    If this fails, it means your seeding is still keyed to loop index
    rather than a canonical lambda index derived from the lambda value.
    """
    lambdas_a = (0.0, 0.25, 0.5, 0.75, 1.0)
    lambdas_b = tuple(reversed(lambdas_a))

    cfg_a = _make_cfg(n=40, n_reps=3, lambdas=lambdas_a, seed=999, seed_policy="stable_per_cell")
    cfg_b = _make_cfg(n=40, n_reps=3, lambdas=lambdas_b, seed=999, seed_policy="stable_per_cell")

    df_a = sim.simulate_grid(cfg_a).sort_values(["lambda", "rep"]).reset_index(drop=True)
    df_b = sim.simulate_grid(cfg_b).sort_values(["lambda", "rep"]).reset_index(drop=True)

    # Compare a robust subset of deterministic columns
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
    ]
    # Some columns may be absent if a row failed; enforce row_ok before comparing
    if "row_ok" in df_a.columns and "row_ok" in df_b.columns:
        df_a = df_a[df_a["row_ok"].astype(bool)].reset_index(drop=True)
        df_b = df_b[df_b["row_ok"].astype(bool)].reset_index(drop=True)

    pd.testing.assert_frame_equal(df_a[cols], df_b[cols], check_dtype=False)


def test_seed_policy_sequential_is_allowed_to_be_order_dependent():
    """
    sequential => consumes RNG in loop order; reordering lambdas can change results.
    This test does NOT require inequality (could coincidentally match),
    but it ensures the code runs under sequential policy.
    """
    cfg = _make_cfg(n=20, n_reps=2, lambdas=(0.0, 0.5, 1.0), seed=42, seed_policy="sequential")
    df = sim.simulate_grid(cfg)
    assert len(df) == 6
    assert "CC_hat" in df.columns
