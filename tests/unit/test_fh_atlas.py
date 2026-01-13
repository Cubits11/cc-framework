import numpy as np

from experiments.fh_atlas.manifest import hash_file
from experiments.fh_atlas.simulate_dependence import simulate_dependence
from theory.fh_bounds import (
    independence_serial_or_j,
    serial_or_composition_bounds,
    union_bounds,
)


def test_independence_within_fh_bounds_serial_or():
    rng = np.random.default_rng(123)
    miss_rates = rng.uniform(0.01, 0.2, size=3).tolist()
    fpr_rates = rng.uniform(0.0, 0.1, size=3).tolist()
    bounds = serial_or_composition_bounds(miss_rates, fpr_rates)
    tprs = [1.0 - m for m in miss_rates]
    j_indep = independence_serial_or_j(tprs, fpr_rates)
    assert bounds.j_lower - 1e-8 <= j_indep <= bounds.j_upper + 1e-8


def test_simulated_j_within_bounds():
    miss_rates = [0.05, 0.08]
    fpr_rates = [0.01, 0.02]
    bounds = serial_or_composition_bounds(miss_rates, fpr_rates)
    rng = np.random.default_rng(7)
    sim = simulate_dependence(
        tprs=[1.0 - m for m in miss_rates],
        fprs=fpr_rates,
        composition_type="serial_or",
        family="clayton",
        theta=1.0,
        sample_size=400,
        rng=rng,
    )
    assert bounds.j_lower - 0.1 <= sim.observed_j <= bounds.j_upper + 0.1


def test_fh_width_monotonic_union():
    p = 0.1
    widths = []
    for k in range(1, 5):
        bounds = union_bounds([p] * k)
        widths.append(bounds.width)
    assert widths == sorted(widths)


def test_manifest_hash_stable(tmp_path):
    file_path = tmp_path / "artifact.txt"
    file_path.write_text("test", encoding="utf-8")
    first = hash_file(file_path)
    second = hash_file(file_path)
    assert first == second
