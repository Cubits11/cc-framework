# examples/adapter_disagreement_summary.py
"""Summarize adapter disagreement rates from benchmark JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def load_disagreements(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("record_type") != "summary":
                continue
            summary = rec.get("summary", {})
            disagreement = summary.get("disagreement_rates", {})
            run_id = rec.get("run_meta", {}).get("run_id")
            for pair, rate in disagreement.items():
                rows.append({"run_id": run_id, "pair": pair, "rate": rate})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute disagreement summary table.")
    ap.add_argument("--bench", required=True, help="JSONL path from run_bench.")
    ap.add_argument("--out", default="results/adapter_disagreements.csv")
    args = ap.parse_args()

    rows = load_disagreements(Path(args.bench))
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No disagreement data found in summary records.")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path.resolve()}")


if __name__ == "__main__":
    main()
