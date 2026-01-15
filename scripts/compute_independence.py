#!/usr/bin/env python
"""Compute independence baselines for Week 7 point summaries."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from cc.analysis.week7_utils import independence_and, independence_or


def load_point(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def write_point(path: Path, payload: Dict) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)


def compute_for_point(data: Dict) -> Dict:
    per_rail = data.get("per_rail", {})
    rails = data.get("rails", [])
    tprs: List[float] = []
    fprs: List[float] = []
    for rail in rails:
        metrics = per_rail.get(rail)
        if not metrics:
            raise ValueError(f"Missing per-rail metrics for {rail}")
        tprs.append(float(metrics["tpr"]))
        fprs.append(float(metrics["fpr"]))

    topology = data.get("topology")
    if topology == "serial_or":
        baseline = independence_or(tprs, fprs)
    elif topology == "parallel_and":
        baseline = independence_and(tprs, fprs)
    else:
        raise ValueError(f"Unsupported topology: {topology}")

    data.setdefault("independence", {}).update(baseline)
    data.setdefault("audit", {}).setdefault("independence_contained", None)
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Annotate point summaries with independence baselines")
    parser.add_argument("--in", dest="input_dir", required=True, help="Directory of point_*.json files")
    parser.add_argument("--out", dest="output_dir", required=True, help="Directory to write updated JSON")
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.glob("point_*.json"))
    if not files:
        raise SystemExit("No point_*.json files found; run make_week7_runs.py first")

    for path in files:
        data = load_point(path)
        updated = compute_for_point(data)
        write_point(out_dir / path.name, updated)


if __name__ == "__main__":
    main()
