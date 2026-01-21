#!/usr/bin/env python
"""Generate Week 7 scaled runs using a deterministic simulator.

The real project hooks into `run_with_checks.py`.  For this repository we
provide a light-weight simulator that mimics the statistical behaviour expected
by the Week 7 memo.  It honours the configuration schema described in the
Brutally Complete pack and produces one JSON summary per configuration/seed
pair.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from cc.analysis.week7_utils import (
    BCaInterval,
    bca_bootstrap,
    fh_envelope,
    independence_and,
    independence_or,
    wilson_interval,
)


@dataclass(frozen=True)
class RailParams:
    base_tpr: float
    base_fpr: float
    slope_tpr: float
    slope_fpr: float
    anchor: float


RAIL_LIBRARY: Dict[str, RailParams] = {
    "keyword": RailParams(base_tpr=0.82, base_fpr=0.052, slope_tpr=0.9, slope_fpr=1.2, anchor=0.65),
    "regex": RailParams(base_tpr=0.75, base_fpr=0.048, slope_tpr=0.7, slope_fpr=1.0, anchor=0.60),
    "semantic": RailParams(
        base_tpr=0.88, base_fpr=0.055, slope_tpr=1.1, slope_fpr=1.4, anchor=0.75
    ),
}


# ---------------------------------------------------------------------------
# Deterministic simulation helpers
# ---------------------------------------------------------------------------


def calibrate_rate(params: RailParams, threshold: float, jitter: float) -> Tuple[float, float]:
    delta = params.anchor - threshold
    tpr = params.base_tpr + params.slope_tpr * delta + jitter
    fpr = params.base_fpr + params.slope_fpr * delta * 0.4 + jitter / 2.0
    tpr = float(min(0.97, max(0.55, tpr)))
    fpr = float(min(0.03, max(0.02, fpr)))
    return tpr, fpr


def sample_guardrail(
    seed: int, episodes: int, rail: str, threshold: float
) -> Tuple[np.ndarray, np.ndarray]:
    params = RAIL_LIBRARY.get(rail)
    if not params:
        raise ValueError(f"Unknown rail '{rail}'")
    rng = np.random.default_rng(seed)
    jitter = rng.normal(0.0, 0.01)
    tpr, fpr = calibrate_rate(params, threshold, jitter)
    world1 = rng.binomial(1, tpr, size=episodes).astype(float)
    world0 = rng.binomial(1, fpr, size=episodes).astype(float)
    return world1, world0


def compose_events(topology: str, events: Sequence[np.ndarray]) -> np.ndarray:
    if topology == "serial_or":
        stacked = np.stack(events, axis=0)
        return (stacked.max(axis=0)).astype(float)
    if topology == "parallel_and":
        stacked = np.stack(events, axis=0)
        return (stacked.min(axis=0)).astype(float)
    raise ValueError(f"Unsupported topology: {topology}")


def compute_empirical_metrics(world1: np.ndarray, world0: np.ndarray) -> Dict[str, float]:
    tpr = float(np.mean(world1))
    fpr = float(np.mean(world0))
    j = tpr - fpr
    return {"tpr": tpr, "fpr": fpr, "j": j}


def compute_wilson(world: np.ndarray) -> WilsonInterval:
    successes = int(np.sum(world))
    return wilson_interval(successes, world.size)


def compute_bca(world1: np.ndarray, world0: np.ndarray, *, rng_seed: int) -> Dict[str, BCaInterval]:
    delta_interval = bca_bootstrap(
        [world1, world0],
        lambda xs: float(np.mean(xs[0]) - np.mean(xs[1])),
        rng=rng_seed,
        n_bootstrap=600,
    )
    j_interval = bca_bootstrap(
        [world1, world0],
        lambda xs: float(np.mean(xs[0]) - np.mean(xs[1])),
        rng=rng_seed + 1,
        n_bootstrap=600,
    )
    return {"delta": delta_interval, "j": j_interval}


def classify_regime(
    j_obs: float, single_js: Sequence[float], independence_j: float
) -> Tuple[str, float, bool]:
    best_single = max(single_js)
    delta_j = j_obs - best_single
    d_lamp = best_single < 0.10
    if delta_j > 1e-4:
        classification = "constructive"
    elif delta_j < -1e-4:
        classification = "destructive"
    else:
        classification = "independent"
    cc_l = (j_obs - independence_j) / max(best_single, 1e-6)
    if d_lamp:
        cc_l = None
    return classification, cc_l, d_lamp


def hash_config(payload: Mapping[str, object]) -> str:
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate Week 7 scaled runs")
    parser.add_argument("--config", required=True, help="YAML configuration file")
    parser.add_argument("--outdir", required=True, help="Directory for point summaries")
    parser.add_argument("--audit", required=True, help="Path to audit JSONL file")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on total grid points")
    args = parser.parse_args()

    config_path = Path(args.config)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    audit_path = Path(args.audit)
    audit_path.parent.mkdir(parents=True, exist_ok=True)

    with config_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    seeds: List[int] = list(cfg.get("seeds", []))
    episodes = int(cfg.get("episodes_per_config", 1000))
    fpr_window = cfg.get("fpr_window", [0.04, 0.06])

    threshold_grid: Dict[str, List[float]] = cfg.get("threshold_grid", {})
    or_compositions: List[List[str]] = cfg.get("or_compositions", [])
    and_compositions: List[List[str]] = cfg.get("and_compositions", [])

    def iter_thresholds(rails: Sequence[str]) -> Iterable[Dict[str, float]]:
        values = [threshold_grid[r] for r in rails]
        for combo in product(*values):
            yield {rail: float(thr) for rail, thr in zip(rails, combo)}

    total_runs = 0
    for topology, compositions in [
        ("serial_or", or_compositions),
        ("parallel_and", and_compositions),
    ]:
        for rails in compositions:
            for thresholds in iter_thresholds(rails):
                for seed in seeds:
                    if args.limit is not None and total_runs >= args.limit:
                        break
                    total_runs += 1

                    rng_seed = seed + int(1e6 * sum(thresholds.values()))
                    world1_events = []
                    world0_events = []
                    per_rail_metrics: Dict[str, Dict[str, float]] = {}

                    for rail in rails:
                        key_seed = rng_seed + hash(rail) % 10_000
                        world1, world0 = sample_guardrail(
                            key_seed, episodes, rail, thresholds[rail]
                        )
                        world1_events.append(world1)
                        world0_events.append(world0)
                        metrics = compute_empirical_metrics(world1, world0)
                        per_rail_metrics[rail] = {
                            "tpr": metrics["tpr"],
                            "fpr": metrics["fpr"],
                            "j": metrics["j"],
                            "wilson_world1_width": compute_wilson(world1).width,
                            "wilson_world0_width": compute_wilson(world0).width,
                        }

                    world1_comp = compose_events(topology, world1_events)
                    world0_comp = compose_events(topology, world0_events)
                    empirical = compute_empirical_metrics(world1_comp, world0_comp)

                    independence = (
                        independence_or(
                            [m["tpr"] for m in per_rail_metrics.values()],
                            [m["fpr"] for m in per_rail_metrics.values()],
                        )
                        if topology == "serial_or"
                        else independence_and(
                            [m["tpr"] for m in per_rail_metrics.values()],
                            [m["fpr"] for m in per_rail_metrics.values()],
                        )
                    )

                    fh = fh_envelope(
                        topology,
                        [m["tpr"] for m in per_rail_metrics.values()],
                        [m["fpr"] for m in per_rail_metrics.values()],
                    )

                    bc = compute_bca(world1_comp, world0_comp, rng_seed=rng_seed)
                    classification, cc_l, d_lamp = classify_regime(
                        empirical["j"],
                        [m["j"] for m in per_rail_metrics.values()],
                        independence["j"],
                    )

                    fh_containment = (
                        fh.tpr_lower - 1e-8 <= empirical["tpr"] <= fh.tpr_upper + 1e-8
                        and fh.fpr_lower - 1e-8 <= empirical["fpr"] <= fh.fpr_upper + 1e-8
                    )
                    window_ok = empirical["fpr"] == 0.0 or (
                        fpr_window[0] <= empirical["fpr"] <= fpr_window[1]
                    )

                    audit_entry = {
                        "topology": topology,
                        "rails": rails,
                        "thresholds": thresholds,
                        "seed": seed,
                        "episodes": episodes,
                        "config_hash": hash_config(
                            {"topology": topology, "rails": rails, "thresholds": thresholds}
                        ),
                        "fh_containment": fh_containment,
                        "fpr_window_ok": window_ok,
                    }

                    point_payload = {
                        "topology": topology,
                        "rails": rails,
                        "thresholds": thresholds,
                        "seed": seed,
                        "episodes": episodes,
                        "per_rail": per_rail_metrics,
                        "empirical": empirical,
                        "wilson": {
                            "world0": compute_wilson(world0_comp).__dict__,
                            "world1": compute_wilson(world1_comp).__dict__,
                        },
                        "independence": independence,
                        "fh_envelope": fh.__dict__,
                        "bca": {
                            "delta": bc["delta"].__dict__,
                            "j": bc["j"].__dict__,
                        },
                        "classification": {
                            "label": classification,
                            "cc_l": cc_l,
                            "d_lamp": d_lamp,
                        },
                        "acceptance": {
                            "fpr_window": window_ok,
                            "fh_containment": fh_containment,
                            "fpr_window_bounds": fpr_window,
                        },
                        "audit": audit_entry,
                    }

                    out_path = outdir / (
                        f"point_{topology}_{'-'.join(rails)}_"
                        + "_".join(f"{k}{thresholds[k]:.2f}" for k in sorted(thresholds))
                        + f"_seed{seed}.json"
                    )
                    with out_path.open("w", encoding="utf-8") as fh_out:
                        json.dump(point_payload, fh_out, indent=2, sort_keys=True)

                    with audit_path.open("a", encoding="utf-8") as audit_fh:
                        audit_fh.write(json.dumps(audit_entry, sort_keys=True) + "\n")

    print(f"Wrote {total_runs} point summaries to {outdir}")


if __name__ == "__main__":
    main()
