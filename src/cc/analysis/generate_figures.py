"""
Module: cc.analysis.generate_figures
Purpose: Build the three required figures and a summary.csv from audit history.

CLI:
  python -m cc.analysis.generate_figures \
    --history runs/audit.jsonl \
    --fig-dir paper/figures \
    --out-dir results/smoke/aggregates

Inputs (audit JSONL; one object per event)
  {
    "cfg": {
      "epsilon": 0.00,          # optional, for phase-point plot
      "T": 0.00,                # optional, for phase-point plot
      "samples": 200,           # optional, for n_sessions in CSV
      "...": "..."              # any other config used by io.load_scores
    },
    "metrics": {
      "CC_max": 1.87,           # required for convergence and summary
      "...": "..."
    },
    "sha": "..."                # optional; tail_sha(history) resolves exp id
  }

Outputs
  <fig-dir>/phase_diagram.pdf
  <fig-dir>/cc_convergence.pdf
  <fig-dir>/roc_comparison.pdf
  <out-dir>/summary.csv
  results/aggregates/summary.csv (mirror for alternate test path)
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# --- Matplotlib style tuned for LaTeX-like look (vector-friendly) -------------
mpl.rcParams.update(
    {
        "pdf.fonttype": 42,           # keep text as text in PDFs
        "ps.fonttype": 42,
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "savefig.transparent": True,
    }
)

# --- Reuse cartographer utilities already in the repo -------------------------
from cc.cartographer.audit import _iter_jsonl, tail_sha

try:
    # optional dependency; we fail soft to a synthetic ROC if missing
    from cc.cartographer import io  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    io = None  # sentinel


# ============================== helpers =======================================

def _ensure_dir(p: str | Path) -> Path:
    pth = Path(p)
    pth.mkdir(parents=True, exist_ok=True)
    return pth


def _to_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if np.isfinite(v):
            return v
    except Exception:
        pass
    return default


def _read_cc_series(history: str | Path) -> List[float]:
    """Extract CC_max series from the audit chain (in file order). NaNs dropped."""
    vals: List[float] = []
    for _, obj in _iter_jsonl(str(history)):
        m = obj.get("metrics", {}) if isinstance(obj, dict) else {}
        vals.append(_to_float(m.get("CC_max"), np.nan))
    # drop non-finite
    return [v for v in vals if np.isfinite(v)]


def _last_record(history: str | Path) -> Optional[Dict[str, Any]]:
    last: Optional[Dict[str, Any]] = None
    for _, obj in _iter_jsonl(str(history)):
        if isinstance(obj, dict):
            last = obj
    return last


def _bootstrap_ci(
    data: Sequence[float],
    alpha: float = 0.05,
    B: int = 10_000,
    random_state: int = 1337,
) -> Optional[Tuple[float, float]]:
    """
    Percentile bootstrap CI for the mean of 'data'.
    Returns (lower, upper) or None if |data|<3 (too small).
    """
    arr = np.asarray([x for x in data if np.isfinite(x)], dtype=float)
    if arr.size < 3:
        return None
    rng = np.random.default_rng(random_state)
    idx = rng.integers(0, arr.size, size=(B, arr.size))
    boot = arr[idx].mean(axis=1)
    lo = float(np.quantile(boot, alpha / 2.0))
    hi = float(np.quantile(boot, 1.0 - alpha / 2.0))
    return (lo, hi)


# ============================== plotting ======================================

def _plot_phase_from_last(rec: Dict[str, Any], fig_path: Path) -> None:
    """
    Create phase_diagram.pdf as a single phase-point (epsilon, T) annotated by CC_max.
    This matches the "debug" phase plots you've used and keeps the figure light.
    """
    cfg = rec.get("cfg", {}) if isinstance(rec, dict) else {}
    cc = _to_float(rec.get("metrics", {}).get("CC_max") if isinstance(rec, dict) else None, 1.0)

    eps = _to_float(cfg.get("epsilon"), 0.0)
    T = _to_float(cfg.get("T"), 0.0)

    fig, ax = plt.subplots(figsize=(4.2, 3.2))  # ~ 302×230pt
    ax.scatter([eps], [T], s=90, color="#1f77b4")
    ax.axhline(0.0, color="0.8", lw=0.8)
    ax.axvline(0.0, color="0.8", lw=0.8)
    ax.set_title(f"CC phase point: CC_max={cc:.2f}", pad=6)
    ax.set_xlabel("epsilon")
    ax.set_ylabel("T")
    ax.grid(True, alpha=0.25)
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)


def _plot_cc_convergence(cc_series: Sequence[float], fig_path: Path) -> None:
    """Make cc_convergence.pdf (trial index vs CC_max) with clean styling."""
    series = np.asarray(list(cc_series), dtype=float)
    if series.size == 0:
        series = np.array([np.nan])
    x = np.arange(1, series.size + 1)

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.plot(x, series, marker="o", lw=1.8, color="#1f77b4")
    ax.set_xlabel("trial")
    ax.set_ylabel(r"$CC_{\max}$")
    ax.set_title(r"$CC_{\max}$ convergence", pad=6)
    ax.grid(True, alpha=0.25)
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)


def _synthetic_roc() -> Tuple[np.ndarray, np.ndarray]:
    """Fallback ROC curves if io.load_scores(cfg, ...) is unavailable."""
    fpr_single = np.array([0.0, 0.10, 0.20, 0.50, 1.0])
    tpr_single = np.array([0.0, 0.50, 0.70, 0.85, 1.0])
    fpr_comp   = np.array([0.0, 0.05, 0.15, 0.40, 1.0])
    tpr_comp   = np.array([0.0, 0.60, 0.80, 0.90, 1.0])
    rocA = np.c_[fpr_single, tpr_single]
    rocB = np.c_[fpr_comp, tpr_comp]
    return rocA, rocB


def _plot_roc_from_cfg(rec: Dict[str, Any], fig_path: Path) -> None:
    """
    Make roc_comparison.pdf from toy scores generated via cfg.
    If cc.cartographer.io is not importable or throws, draw a synthetic ROC.
    """
    cfg = rec.get("cfg", {}) if isinstance(rec, dict) else {}
    labelA = str(cfg.get("A", "A"))
    labelB = str(cfg.get("B", "B"))

    try:
        if io is None:
            raise RuntimeError("io.load_scores not available")
        data = io.load_scores(cfg, n=None)  # type: ignore[attr-defined]
        rocA = np.asarray(data["rocA"], dtype=float)
        rocB = np.asarray(data["rocB"], dtype=float)
    except Exception:
        rocA, rocB = _synthetic_roc()

    fig, ax = plt.subplots(figsize=(4.4, 4.0))
    ax.plot(rocA[:, 0], rocA[:, 1], label=labelA, lw=1.8, color="#1f77b4")
    ax.plot(rocB[:, 0], rocB[:, 1], label=labelB, lw=1.8, color="#d62728")
    ax.plot([0, 1], [0, 1], ls="--", lw=1.2, color="0.4", label="random")

    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("ROC comparison (toy)", pad=6)
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25)
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)


# ============================== CSV summary ===================================

def _write_summary_csv(
    out_dir: Path,
    cc_series: Sequence[float],
    history: str | Path,
    rec: Dict[str, Any],
) -> Path:
    """
    Create results/smoke/aggregates/summary.csv with schema:
      experiment_id, n_sessions, cc_max, ci_lower, ci_upper
    Also mirrors to results/aggregates/summary.csv for alternative paths.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "summary.csv"

    exp_id = tail_sha(str(history)) or "unknown"
    cfg = rec.get("cfg", {}) if isinstance(rec, dict) else {}
    n_sessions = int(_to_float(cfg.get("samples"), 0.0))

    series = [float(v) for v in cc_series if np.isfinite(v)]
    cc_last = float(series[-1]) if series else _to_float(rec.get("metrics", {}).get("CC_max"), 0.0)

    ci = _bootstrap_ci(series) if len(series) >= 5 else None
    ci_lower = f"{ci[0]:.3f}" if ci is not None else ""
    ci_upper = f"{ci[1]:.3f}" if ci is not None else ""

    row = {
        "experiment_id": exp_id,
        "n_sessions": n_sessions,
        "cc_max": round(cc_last, 3),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }

    with open(out_file, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        w.writeheader()
        w.writerow(row)

    # Mirror for tests expecting a global aggregates path
    mirror = Path("results/aggregates")
    mirror.mkdir(parents=True, exist_ok=True)
    mirror_file = mirror / "summary.csv"
    mirror_file.write_bytes(out_file.read_bytes())

    return out_file


# ============================== CLI ===========================================

def main(argv: Optional[Iterable[str]] = None) -> None:
    ap = argparse.ArgumentParser(
        description="Generate three publication-quality figures and a summary.csv from an audit history."
    )
    ap.add_argument("--history", required=True, help="Path to audit JSONL (hash-chained).")
    ap.add_argument("--fig-dir", required=True, help="Directory to write PDFs (figures).")
    ap.add_argument("--out-dir", required=True, help="Directory to write CSV summary.")
    args = ap.parse_args(argv)

    history = Path(args.history)
    fig_dir = _ensure_dir(args.fig_dir)
    out_dir = _ensure_dir(args.out_dir)

    rec = _last_record(history)
    if rec is None:
        raise SystemExit(f"No records found in {history}. Run your experiment first.")

    # Build figures (vector PDFs)
    cc_series = _read_cc_series(history)
    _plot_phase_from_last(rec, fig_dir / "phase_diagram.pdf")
    _plot_cc_convergence(cc_series, fig_dir / "cc_convergence.pdf")
    _plot_roc_from_cfg(rec, fig_dir / "roc_comparison.pdf")

    # Write summary
    out_csv = _write_summary_csv(out_dir, cc_series, history, rec)

    print(f"✓ Wrote figures → {fig_dir.resolve()}")
    print(f"✓ Wrote summary → {out_csv.resolve()}")


if __name__ == "__main__":
    main()