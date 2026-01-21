#!/usr/bin/env python
"""Compute Fréchet–Hoeffding envelopes for Week 7 point summaries."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from cc.analysis.week7_utils import FHEnvelope, fh_envelope


def load_point(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def write_point(path: Path, payload: Dict) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)


def compute_for_point(data: Dict) -> Dict:
    topology = data.get("topology")
    rails = data.get("rails", [])
    per_rail = data.get("per_rail", {})

    tprs: List[float] = []
    fprs: List[float] = []
    for rail in rails:
        metrics = per_rail.get(rail)
        if not metrics:
            raise ValueError(f"Missing per-rail metrics for {rail}")
        tprs.append(float(metrics["tpr"]))
        fprs.append(float(metrics["fpr"]))

    envelope: FHEnvelope = fh_envelope(topology, tprs, fprs)
    fh_payload = {
        "tpr_lower": envelope.tpr_lower,
        "tpr_upper": envelope.tpr_upper,
        "fpr_lower": envelope.fpr_lower,
        "fpr_upper": envelope.fpr_upper,
        "j_lower": envelope.j_lower,
        "j_upper": envelope.j_upper,
    }
    data["fh_envelope"] = fh_payload

    independence = data.get("independence")
    if independence:
        contained = (
            fh_payload["tpr_lower"] - 1e-12
            <= independence["tpr"]
            <= fh_payload["tpr_upper"] + 1e-12
            and fh_payload["fpr_lower"] - 1e-12
            <= independence["fpr"]
            <= fh_payload["fpr_upper"] + 1e-12
            and fh_payload["j_lower"] - 1e-12 <= independence["j"] <= fh_payload["j_upper"] + 1e-12
        )
        data.setdefault("audit", {})["fh_contains_independence"] = bool(contained)
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Annotate point summaries with FH envelopes")
    parser.add_argument(
        "--in", dest="input_dir", required=True, help="Directory containing point_*.json"
    )
    parser.add_argument(
        "--out", dest="output_dir", required=True, help="Directory to write updated JSON"
    )
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.glob("point_*.json"))
    if not files:
        raise SystemExit("No point_*.json files to process")

    for path in files:
        updated = compute_for_point(load_point(path))
        write_point(out_dir / path.name, updated)


if __name__ == "__main__":
    main()
