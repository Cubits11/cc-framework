from __future__ import annotations

import json
import math
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))  # so `import scripts` works
sys.path.insert(0, str(REPO_ROOT / "src"))  # so `import cc...` works even without PYTHONPATH

import numpy as np
import pytest

from cc.analysis.week7_utils import (
    bca_bootstrap,
    fh_bounds_intersection,
    fh_bounds_union,
    fh_envelope,
    independence_and,
    independence_or,
    stable_prod_one_minus,
    wilson_interval,
)
from scripts import make_week7_runs as runs_module


def _random_probabilities(rng: np.random.Generator, k: int) -> list[float]:
    return rng.uniform(0.05, 0.95, size=k).tolist()


def _percentile_interval(samples: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    return float(np.quantile(samples, alpha / 2.0)), float(np.quantile(samples, 1.0 - alpha / 2.0))


def _make_toy_config(path: Path) -> Path:
    cfg = {
        "fpr_window": [0.04, 0.06],
        "seeds": [2027, 2028],
        "episodes_per_config": 160,
        "threshold_grid": {"keyword": [0.6, 0.7], "regex": [0.5], "semantic": [0.7]},
        "or_compositions": [["keyword", "regex"]],
        "and_compositions": [["keyword", "regex", "semantic"]],
    }
    path.write_text(json.dumps(cfg))
    return path


@pytest.fixture(scope="module")
def toy_pipeline(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("week7")
    cfg_path = _make_toy_config(tmp / "grid.json")
    outdir = tmp / "summaries"
    figs = tmp / "figs"
    audit = tmp / "audit.jsonl"
    summary = tmp / "week7_summary.json"
    memo = tmp / "memo.md"

    def run(cmd: Sequence[str]) -> None:
        subprocess.run(cmd, check=True)

    run(
        [
            sys.executable,
            "scripts/make_week7_runs.py",
            "--config",
            str(cfg_path),
            "--outdir",
            str(outdir),
            "--audit",
            str(audit),
        ]
    )
    run(
        [
            sys.executable,
            "scripts/compute_independence.py",
            "--in",
            str(outdir),
            "--out",
            str(outdir),
        ]
    )
    run(
        [
            sys.executable,
            "scripts/compute_fh_envelope.py",
            "--in",
            str(outdir),
            "--out",
            str(outdir),
        ]
    )
    run([sys.executable, "scripts/make_week7_figs.py", "--in", str(outdir), "--out", str(figs)])
    run(
        [
            sys.executable,
            "scripts/write_week7_memo.py",
            "--summary",
            str(summary),
            "--out",
            str(memo),
            "--points",
            str(outdir),
        ]
    )

    points = []
    for path in sorted(outdir.glob("point_*.json")):
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        points.append(data)
    with summary.open("r", encoding="utf-8") as fh:
        summary_payload = json.load(fh)
    return {
        "tmp": tmp,
        "summaries": outdir,
        "figs": figs,
        "audit": audit,
        "points": points,
        "summary": summary_payload,
        "memo": memo,
    }


# --------------------------
# Category A: FH & envelopes
# --------------------------


def test_fh_single_rail_collapse():
    env = fh_envelope("serial_or", [0.8], [0.05])
    assert env.tpr_lower == pytest.approx(env.tpr_upper)
    assert env.fpr_lower == pytest.approx(env.fpr_upper)
    assert env.j_lower == pytest.approx(env.j_upper)


def test_or_independence_within_envelope():
    rng = np.random.default_rng(1)
    tprs = _random_probabilities(rng, 3)
    fprs = _random_probabilities(rng, 3)
    env = fh_envelope("serial_or", tprs, fprs)
    baseline = independence_or(tprs, fprs)
    assert env.tpr_lower - 1e-9 <= baseline["tpr"] <= env.tpr_upper + 1e-9
    assert env.fpr_lower - 1e-9 <= baseline["fpr"] <= env.fpr_upper + 1e-9
    assert env.j_lower - 1e-9 <= baseline["j"] <= env.j_upper + 1e-9


def test_and_independence_within_envelope():
    rng = np.random.default_rng(2)
    tprs = _random_probabilities(rng, 3)
    fprs = _random_probabilities(rng, 3)
    env = fh_envelope("parallel_and", tprs, fprs)
    baseline = independence_and(tprs, fprs)
    assert env.tpr_lower - 1e-9 <= baseline["tpr"] <= env.tpr_upper + 1e-9
    assert env.fpr_lower - 1e-9 <= baseline["fpr"] <= env.fpr_upper + 1e-9
    assert env.j_lower - 1e-9 <= baseline["j"] <= env.j_upper + 1e-9


def test_fh_intersection_formulae_match():
    marginals = [0.3, 0.4, 0.5]
    lower, upper = fh_bounds_intersection(marginals)
    assert lower == pytest.approx(max(0.0, sum(marginals) - (len(marginals) - 1)))
    assert upper == pytest.approx(min(marginals))


def test_fh_union_formulae_match():
    marginals = [0.2, 0.25, 0.15]
    lower, upper = fh_bounds_union(marginals)
    assert lower == pytest.approx(max(marginals))
    assert upper == pytest.approx(min(1.0, sum(marginals)))


def test_nested_sets_reach_extremes():
    marginals = [0.2, 0.4, 0.6]
    _lower, upper = fh_bounds_intersection(marginals)
    assert upper == pytest.approx(min(marginals))
    lower_u, _upper_u = fh_bounds_union(marginals)
    assert lower_u == pytest.approx(max(marginals))


def test_monotonicity_additional_rail():
    rng = np.random.default_rng(42)
    for _ in range(50):
        base_tprs = _random_probabilities(rng, 2)
        base_fprs = _random_probabilities(rng, 2)
        extra_tpr = rng.uniform(0.4, 0.9)
        extra_fpr = rng.uniform(0.01, 0.1)
        env_two_and = fh_envelope("parallel_and", base_tprs, base_fprs)
        env_three_and = fh_envelope(
            "parallel_and", [*base_tprs, extra_tpr], [*base_fprs, extra_fpr]
        )
        assert env_three_and.tpr_upper <= env_two_and.tpr_upper + 1e-9
        assert env_three_and.tpr_lower <= env_two_and.tpr_lower + 1e-9
        env_two_or = fh_envelope("serial_or", base_tprs, base_fprs)
        env_three_or = fh_envelope("serial_or", [*base_tprs, extra_tpr], [*base_fprs, extra_fpr])
        assert env_three_or.fpr_upper >= env_two_or.fpr_upper - 1e-9
        assert env_three_or.tpr_lower >= env_two_or.tpr_lower - 1e-9


def test_envelope_bounds_constrained():
    rng = np.random.default_rng(3)
    for _ in range(20):
        tprs = _random_probabilities(rng, 4)
        fprs = _random_probabilities(rng, 4)
        env = fh_envelope("serial_or", tprs, fprs)
        assert 0 <= env.tpr_lower <= env.tpr_upper <= 1
        assert 0 <= env.fpr_lower <= env.fpr_upper <= 1
        assert -1 <= env.j_lower <= env.j_upper <= 1


# --------------------------
# Category B: independence
# --------------------------


def test_stable_product_matches_direct():
    rng = np.random.default_rng(5)
    vals = rng.uniform(0.0, 0.5, size=20)
    prod_direct = float(np.prod(1.0 - vals))
    prod_stable = stable_prod_one_minus(vals)
    assert prod_direct == pytest.approx(prod_stable, abs=1e-12)

    near_one = np.full(200, 0.99)
    assert math.isfinite(stable_prod_one_minus(near_one))


def test_independence_matches_monte_carlo():
    rng = np.random.default_rng(6)
    tprs = [0.7, 0.8]
    fprs = [0.05, 0.04]
    baseline_or = independence_or(tprs, fprs)
    baseline_and = independence_and(tprs, fprs)

    trials = 40_000
    hits_or = np.zeros(trials, dtype=bool)
    false_or = np.zeros(trials, dtype=bool)
    for tpr, fpr in zip(tprs, fprs, strict=False):
        hits_or |= rng.random(trials) < tpr
        false_or |= rng.random(trials) < fpr
    assert hits_or.mean() == pytest.approx(baseline_or["tpr"], rel=0.02)
    assert false_or.mean() == pytest.approx(baseline_or["fpr"], rel=0.05)

    hits_and = np.ones(trials, dtype=bool)
    false_and = np.ones(trials, dtype=bool)
    for tpr, fpr in zip(tprs, fprs, strict=False):
        hits_and &= rng.random(trials) < tpr
        false_and &= rng.random(trials) < fpr
    assert hits_and.mean() == pytest.approx(baseline_and["tpr"], rel=0.05)
    assert false_and.mean() == pytest.approx(baseline_and["fpr"], rel=0.05)


# --------------------------
# Category C: uncertainty
# --------------------------


def test_wilson_interval_properties():
    ci_small = wilson_interval(5, 20)
    ci_large = wilson_interval(50, 200)
    assert 0.0 <= ci_small.lower <= ci_small.upper <= 1.0
    assert ci_large.width < ci_small.width


def test_bca_beats_percentile_coverage():
    rng = np.random.default_rng(7)
    true_p = 0.85
    cover_bca = []
    cover_perc = []
    for _ in range(80):
        data = rng.binomial(1, true_p, size=90)
        bca = bca_bootstrap(
            data, lambda xs: float(np.mean(xs[0])), rng=rng.integers(0, 10_000), n_bootstrap=500
        )
        boots = np.array([np.mean(data[rng.integers(0, data.size, data.size)]) for _ in range(500)])
        lo, hi = _percentile_interval(boots)
        cover_bca.append(bca.lower <= true_p <= bca.upper)
        cover_perc.append(lo <= true_p <= hi)
    bca_rate = np.mean(cover_bca)
    perc_rate = np.mean(cover_perc)
    assert abs(bca_rate - 0.95) <= 0.1
    assert bca_rate >= perc_rate - 0.05


def test_bca_width_shrinks_with_sample_size():
    rng = np.random.default_rng(8)
    small = rng.binomial(1, 0.7, size=40)
    large = rng.binomial(1, 0.7, size=400)
    width_small = bca_bootstrap(
        small, lambda xs: float(np.mean(xs[0])), rng=0, n_bootstrap=400
    ).width
    width_large = bca_bootstrap(
        large, lambda xs: float(np.mean(xs[0])), rng=1, n_bootstrap=400
    ).width
    assert width_large < width_small


def test_d_lamp_flags_low_denominator():
    classification, cc_l, d_lamp = runs_module.classify_regime(0.05, [0.08, 0.09], 0.04)
    assert d_lamp is True
    assert cc_l is None
    assert classification == "destructive"


# --------------------------
# Category D: regimes and figures
# --------------------------


def test_classification_independent_case():
    classification, cc_l, d_lamp = runs_module.classify_regime(0.12, [0.12, 0.11], 0.0)
    assert classification == "independent"
    assert pytest.approx(cc_l, rel=0.1) == 1.0
    assert d_lamp is False


def test_classification_constructive_case():
    classification, cc_l, _ = runs_module.classify_regime(0.2, [0.11, 0.12], 0.05)
    assert classification == "constructive"
    assert cc_l > 0


def test_classification_destructive_case():
    classification, cc_l, _ = runs_module.classify_regime(0.05, [0.15, 0.16], 0.1)
    assert classification == "destructive"
    assert cc_l < 0


def test_regime_labels_follow_delta_sign(toy_pipeline):
    for raw in toy_pipeline["points"]:
        delta_j = raw["empirical"]["j"] - max(m["j"] for m in raw["per_rail"].values())
        label = raw["classification"]["label"]
        if delta_j > 1e-4:
            assert label == "constructive"
        elif delta_j < -1e-4:
            assert label == "destructive"
        else:
            assert label == "independent"


# --------------------------
# Category E: robustness & edge cases
# --------------------------


def test_independence_requires_non_empty():
    with pytest.raises(ValueError):
        independence_or([], [])


def test_fh_bounds_reject_nan():
    with pytest.raises(ValueError):
        fh_bounds_intersection([0.2, math.nan])
    with pytest.raises(ValueError):
        fh_bounds_union([0.1, float("inf")])


def test_zero_fp_runs_recorded_in_summary(toy_pipeline):
    summary = toy_pipeline["summary"]
    assert "zero_fp_runs" in summary
    assert summary["zero_fp_runs"] >= 0


def test_audit_lines_include_hash(toy_pipeline):
    audit_path = toy_pipeline["audit"]
    lines = audit_path.read_text().strip().splitlines()
    assert lines
    payload = json.loads(lines[0])
    assert payload["config_hash"]


def test_figures_are_loadable(toy_pipeline):
    figs = toy_pipeline["figs"]
    pngs = list(figs.glob("*.png"))
    assert pngs, "expected figures to be generated"
    import matplotlib.image as mpimg

    for img in pngs:
        arr = mpimg.imread(img)
        assert arr.size > 0


def test_write_week7_memo_outputs_files(toy_pipeline):
    assert toy_pipeline["memo"].is_file()
    assert toy_pipeline["summary"]


def test_week7_summary_contains_regime_counts(toy_pipeline):
    summary = toy_pipeline["summary"]
    counts = summary["regime_counts"]
    assert set(counts.keys()) == {"constructive", "independent", "destructive"}


def test_pipeline_runtime_results_exist(toy_pipeline):
    outdir = toy_pipeline["summaries"]
    assert list(outdir.glob("point_*.json"))


def test_week7_pipeline_completes_quickly(tmp_path):
    cfg_path = _make_toy_config(tmp_path / "grid.json")
    outdir = tmp_path / "summaries"
    audit = tmp_path / "audit.jsonl"
    subprocess.run(
        [
            sys.executable,
            "scripts/make_week7_runs.py",
            "--config",
            str(cfg_path),
            "--outdir",
            str(outdir),
            "--audit",
            str(audit),
            "--limit",
            "1",
        ],
        check=True,
    )
    assert audit.exists()
    first_point = next(outdir.glob("point_*.json"))
    with first_point.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    assert data["episodes"] == 160


def test_summary_memo_generation(tmp_path):
    cfg_path = _make_toy_config(tmp_path / "grid.json")
    outdir = tmp_path / "summaries"
    audit = tmp_path / "audit.jsonl"
    summary = tmp_path / "summary.json"
    memo = tmp_path / "memo.md"
    subprocess.run(
        [
            sys.executable,
            "scripts/make_week7_runs.py",
            "--config",
            str(cfg_path),
            "--outdir",
            str(outdir),
            "--audit",
            str(audit),
            "--limit",
            "1",
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/compute_independence.py",
            "--in",
            str(outdir),
            "--out",
            str(outdir),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/compute_fh_envelope.py",
            "--in",
            str(outdir),
            "--out",
            str(outdir),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/write_week7_memo.py",
            "--summary",
            str(summary),
            "--out",
            str(memo),
            "--points",
            str(outdir),
        ],
        check=True,
    )
    assert summary.exists()
    assert memo.exists()
