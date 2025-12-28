# examples/adapter_cliff_plot.py
"""Generate a cliff plot from benchmark JSONL summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import pandas as pd


def load_summaries(path: Path) -> List[Dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("record_type") == "summary":
                summary = rec.get("summary", {})
                meta = rec.get("run_meta", {})
                records.append(
                    {
                        "composition_parameter": meta.get("config", {}).get(
                            "composition_parameter"
                        ),
                        "cc_max": summary.get("cc_max"),
                    }
                )
    return records


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot CC cliff curve over parameter.")
    ap.add_argument("--bench", required=True, help="JSONL path from run_bench.")
    ap.add_argument("--out", default="figs/adapter_cliff_plot.png")
    args = ap.parse_args()

    data = load_summaries(Path(args.bench))
    df = pd.DataFrame(data).dropna()
    if df.empty:
        raise ValueError("No summary records with composition_parameter found.")
    df = df.sort_values("composition_parameter")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df["composition_parameter"], df["cc_max"], marker="o")
    ax.axhline(1.0, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Composition parameter")
    ax.set_ylabel("CC_max")
    ax.set_title("Cliff plot over composition parameter")
    ax.grid(True, alpha=0.3)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Wrote {out_path.resolve()}")


if __name__ == "__main__":
    main()
