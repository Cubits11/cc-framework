# src/cc/analysis/comparison_runner.py
"""
cc.analysis.comparison_runner
=============================

Run alternative destructive-interference metrics on a set of composability results
(Week 6-7) and summarize agreement/disagreement statistics.

Usage:
    python -m cc.analysis.comparison_runner \
        --results results/demo/raw.jsonl \
        --out-dir results/comparisons \
        [--threshold-j 0.05 --threshold-euclid 0.02 ...]
"""

import argparse
import json
from collections.abc import Iterable
from pathlib import Path

from cc.analysis.alternative_metrics import analyze_disagreements, compare_all_metrics, kappa_matrix


def load_results(path: Path) -> dict[str, dict]:
    """Load JSONL results and map to the required metric dictionary."""
    configs: dict[str, dict[str, float]] = {}
    with path.open() as f:
        for line in f:
            rec = json.loads(line)
            cfg = rec.get("config_name") or rec.get("cfg_name") or f"cfg_{len(configs)}"
            # Map raw metrics to keys expected by compare_all_metrics()
            configs[cfg] = {
                "tpr": rec["tpr_empirical"],
                "fpr": rec["fpr_empirical"],
                "j_obs": rec["j_empirical"],
                "j_indep": rec["j_independent_baseline"],
                "tpr_indep": rec["tpr_independent_baseline"],
                "fpr_indep": rec["fpr_independent_baseline"],
                "fh_lower": rec["fh_envelope_lower"],
                "fh_upper": rec["fh_envelope_upper"],
            }
    return configs


def main(argv: Iterable[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Compare interference metrics across configurations.")
    ap.add_argument("--results", required=True)
    ap.add_argument("--out-dir", required=True)
    # optional threshold arguments...
    args = ap.parse_args(argv)
    configs = load_results(Path(args.results))
    df = compare_all_metrics(configs)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "comparisons.csv", index=False)
    kappas = kappa_matrix(df, ["youden_class", "euclidean_class", "cost_class", "fh_class"])
    kappas.to_csv(out_dir / "kappa_matrix.csv")
    diss = analyze_disagreements(df)
    diss.to_csv(out_dir / "disagreements.csv", index=False)
    print(f"Processed {len(df)} configs, wrote results to {out_dir}")
