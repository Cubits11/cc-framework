"""
Module: cc.analysis.generate_figures
Purpose: Build the three required figures and a summary.csv from audit history.
CLI:
  python -m cc.analysis.generate_figures \
    --history runs/audit.jsonl \
    --fig-dir paper/figures \
    --out-dir results/smoke/aggregates
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Reuse cartographer utilities already in the repo
from cc.cartographer.audit import _iter_jsonl, tail_sha
from cc.cartographer import io  # for deterministic toy scores


def _ensure_dir(p: str | Path) -> Path:
    pth = Path(p)
    pth.mkdir(parents=True, exist_ok=True)
    return pth


def _read_cc_series(history: str) -> List[float]:
    """Extract CC_max series from the audit chain (in file order)."""
    vals: List[float] = []
    for _, obj in _iter_jsonl(history):
        m = obj.get("metrics", {})
        try:
            vals.append(float(m.get("CC_max", np.nan)))
        except Exception:
            vals.append(np.nan)
    # Drop NaNs at end if any
    return [v for v in vals if np.isfinite(v)]


def _last_record(history: str) -> Optional[Dict[str, Any]]:
    last = None
    for _, obj in _iter_jsonl(history):
        last = obj
    return last


def _plot_phase_from_last(rec: Dict[str, Any], fig_path: Path) -> None:
    """(Re)create phase_diagram.pdf from last record's cfg + CC_max."""
    cfg = rec.get("cfg", {})
    cc = float(rec.get("metrics", {}).get("CC_max", 1.0))

    def _to_float(x: Any, default: float = 0.0) -> float:
        try:
            return float(x)
        except Exception:
            return default

    eps = _to_float(cfg.get("epsilon"), 0.0)
    T = _to_float(cfg.get("T"), 0.0)

    fig, ax = plt.subplots(figsize=(4, 3), dpi=150)
    ax.scatter([eps], [T], s=80)
    ax.set_title(f"CC phase point: CC_max={cc:.2f}")
    ax.set_xlabel("epsilon")
    ax.set_ylabel("T")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)


def _plot_cc_convergence(cc_series: List[float], fig_path: Path) -> None:
    """Make cc_convergence.pdf (index vs CC_max)."""
    if not cc_series:
        cc_series = [np.nan]
    x = np.arange(1, len(cc_series) + 1)
    fig, ax = plt.subplots(figsize=(5, 3), dpi=150)
    ax.plot(x, cc_series, marker="o")
    ax.set_xlabel("trial")
    ax.set_ylabel("CC_max")
    ax.set_title("CC_max convergence")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)


def _plot_roc_from_cfg(rec: Dict[str, Any], fig_path: Path) -> None:
    """Make roc_comparison.pdf from toy scores generated via cfg."""
    cfg = rec.get("cfg", {})
    data = io.load_scores(cfg, n=None)
    rocA = np.asarray(data["rocA"], dtype=float)
    rocB = np.asarray(data["rocB"], dtype=float)

    fig, ax = plt.subplots(figsize=(4, 4), dpi=150)
    ax.plot(rocA[:, 0], rocA[:, 1], label=str(cfg.get("A", "A")))
    ax.plot(rocB[:, 0], rocB[:, 1], label=str(cfg.get("B", "B")))
    ax.plot([0, 1], [0, 1], ls="--", lw=1, label="random")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("ROC comparison (toy)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)


def _write_summary_csv(
    out_dir: Path,
    cc_series: List[float],
    history: str,
    rec: Dict[str, Any],
) -> Path:
    """
    Create results/smoke/aggregates/summary.csv with required schema:
      experiment_id, n_sessions, cc_max, ci_lower, ci_upper
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "summary.csv"

    exp_id = tail_sha(history) or "unknown"
    n_sessions = int(rec.get("cfg", {}).get("samples") or 0)
    cc_last = float(cc_series[-1]) if cc_series else float(rec.get("metrics", {}).get("CC_max", 0.0))

    # CI not estimated in smoke; put blanks
    row = {
        "experiment_id": exp_id,
        "n_sessions": n_sessions,
        "cc_max": cc_last,
        "ci_lower": "",
        "ci_upper": "",
    }

    with open(out_file, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        w.writeheader()
        w.writerow(row)

    # Also mirror to results/aggregates/summary.csv to satisfy a second test path
    mirror = Path("results/aggregates")
    mirror.mkdir(parents=True, exist_ok=True)
    mirror_file = mirror / "summary.csv"
    mirror_file.write_bytes(out_file.read_bytes())

    return out_file


def main(argv: Optional[Iterable[str]] = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", required=True)
    ap.add_argument("--fig-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args(argv)

    fig_dir = _ensure_dir(args.fig_dir)
    out_dir = _ensure_dir(args.out_dir)

    rec = _last_record(args.history)
    if rec is None:
        raise SystemExit(f"No records found in {args.history}. Run the CLI first.")

    # Build figures
    cc_series = _read_cc_series(args.history)

    # phase_diagram.pdf (recreate to be safe)
    _plot_phase_from_last(rec, fig_dir / "phase_diagram.pdf")
    # cc_convergence.pdf
    _plot_cc_convergence(cc_series, fig_dir / "cc_convergence.pdf")
    # roc_comparison.pdf
    _plot_roc_from_cfg(rec, fig_dir / "roc_comparison.pdf")

    # summary.csv
    out_csv = _write_summary_csv(out_dir, cc_series, args.history, rec)

    print(f"Wrote figures to {fig_dir}")
    print(f"Wrote summary to {out_csv}")


if __name__ == "__main__":
    main()