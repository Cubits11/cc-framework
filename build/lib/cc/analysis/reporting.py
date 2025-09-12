# src/cc/analysis/reporting.py
"""Aggregate CC/CCC evaluation artifacts into human-readable reports.

Lightweight helpers for transforming metric dictionaries produced by
:mod:`cc.analysis.cc_estimation` (CC) and CCC addenda CSVs into CSV/Markdown
summaries under `evaluation/`. Avoids heavy deps so it runs in CI.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, Optional, cast
import csv, hashlib, json

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

EVAL_DIR = Path("evaluation")
CCC_DIR = EVAL_DIR / "ccc" / "addenda"
REPORTS_DIR = EVAL_DIR / "reports"
RESULTS = Path("results")
RUNS = Path("runs")
AUDIT = RUNS / "audit.jsonl"

NEUTRAL_LO, NEUTRAL_HI = 0.95, 1.05  # 5% neutrality band

# ---------------------------------------------------------------------------
# Tiny utilities
# ---------------------------------------------------------------------------

def _read_json(path: Path) -> Dict[str, Any]:
    with path.open() as f:
        return cast(Dict[str, Any], json.load(f))

def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()[:12]

def _discover_cc_runs() -> List[Path]:
    return [p for p in RESULTS.glob("**/metrics_fixed.json")]

def _ci_width(lo: Optional[float], hi: Optional[float]) -> Optional[float]:
    if lo is None or hi is None:
        return None
    try:
        return float(hi) - float(lo)
    except Exception:
        return None

def _cc_decision_from_ci(lo: Optional[float], hi: Optional[float]) -> Optional[str]:
    if lo is None or hi is None:
        return None
    if lo > NEUTRAL_HI:
        return "constructive"
    if hi < NEUTRAL_LO:
        return "destructive"
    return "neutral"

# ---------------------------------------------------------------------------
# Public helpers used by notebooks/tests (kept stable)
# ---------------------------------------------------------------------------

def summarize_metrics(metrics: Dict[str, float], precision: int = 4) -> List[Tuple[str, float]]:
    """Stable, rounded list of (metric, value)."""
    out: List[Tuple[str, float]] = []
    for k, v in sorted(metrics.items()):
        try:
            out.append((k, round(float(v), precision)))
        except Exception:
            # non-floats: skip from numeric summary; keep CSV layer for raw
            continue
    return out

def metrics_to_markdown(summary: Iterable[Tuple[str, float]]) -> str:
    lines = ["| Metric | Value |", "| --- | --- |"]
    for metric, value in summary:
        lines.append(f"| {metric} | {value} |")
    return "\n".join(lines) + "\n"

def metrics_to_csv(summary: Iterable[Tuple[str, float]]) -> str:
    lines = ["metric,value"]
    for metric, value in summary:
        lines.append(f"{metric},{value}")
    return "\n".join(lines) + "\n"

# ---------------------------------------------------------------------------
# CC aggregation
# ---------------------------------------------------------------------------

def _collect_cc_rows() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for mfile in _discover_cc_runs():
        try:
            metrics = _read_json(mfile)
        except Exception:
            continue
        run_dir = mfile.parent
        meta_path = run_dir / "meta.json"
        meta = _read_json(meta_path) if meta_path.exists() else {}
        ci = metrics.get("CC_CI95") or [None, None]
        lo, hi = (ci[0], ci[1]) if isinstance(ci, (list, tuple)) and len(ci) == 2 else (None, None)
        # prefer stored decision; else compute from CI
        decision = metrics.get("decision") or _cc_decision_from_ci(lo, hi)
        # stable run_id heuristic
        run_id = run_dir.name
        if (run_dir.parent / "raw.jsonl").exists() and run_dir.name == "figs":
            run_id = run_dir.parent.name

        row = {
            "flavor": "CC",
            "run_id": run_id,
            "path": str(run_dir),
            "J_single_best": metrics.get("J_single_best"),
            "J_comp": metrics.get("J_comp"),
            "delta_J": metrics.get("delta_J"),
            "CC_max": metrics.get("CC_max"),
            "CC_CI95_lo": lo,
            "CC_CI95_hi": hi,
            "CC_CI95_width": _ci_width(lo, hi),
            "decision": decision,
            "policy_fpr_cap": (metrics.get("policy") or {}).get("fpr_cap"),
            "git_sha": meta.get("git_sha"),
            "raw_hash": _hash_file(run_dir.parent / "raw.jsonl")
                        if (run_dir.parent / "raw.jsonl").exists() else None,
        }
        rows.append(row)
    return rows

def _write_rows_csv(rows: List[Dict[str, Any]], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with out.open("w", newline="") as f:
            csv.writer(f).writerow(["flavor","run_id","path","J_single_best","J_comp","delta_J",
                                    "CC_max","CC_CI95_lo","CC_CI95_hi","CC_CI95_width",
                                    "decision","policy_fpr_cap","git_sha","raw_hash"])
        return
    header = list(rows[0].keys())
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader(); w.writerows(rows)

def _write_md(rows: List[Dict[str, Any]], out: Path, title: str) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", ""]
    if not rows:
        lines.append("_No runs discovered._")
    else:
        lines += ["| flavor | run_id | CC_max | 95% CI | width | decision | ΔJ | FPR cap |",
                  "|---|---|---:|---|---:|---|---:|---:|"]
        for r in rows:
            ci = f"[{r.get('CC_CI95_lo')}, {r.get('CC_CI95_hi')}]"
            lines.append(
              f"| {r.get('flavor')} | {r.get('run_id')} | {r.get('CC_max')} | "
              f"{ci} | {r.get('CC_CI95_width')} | {r.get('decision')} | "
              f"{r.get('delta_J')} | {r.get('policy_fpr_cap')} |"
            )
    out.write_text("\n".join(lines) + "\n")

# ---------------------------------------------------------------------------
# CCC aggregation (reads addenda CSVs if present)
# ---------------------------------------------------------------------------

def _read_ccc_addenda_csvs() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not CCC_DIR.exists():
        return rows
    for csv_path in CCC_DIR.glob("*.csv"):
        try:
            with csv_path.open() as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # tag with file and flavor
                    rec = {"flavor": "CCC", "source_file": str(csv_path)}
                    rec.update(row)
                    rows.append(rec)
        except Exception:
            continue
    return rows

# Normalize CCC columns a bit for merged views
def _normalize_ccc_rows(raw: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in raw:
        out.append({
            "flavor": "CCC",
            "run_id": r.get("run_id") or r.get("id") or r.get("pair") or "unknown",
            "path": r.get("path") or r.get("artifact_dir") or "",
            "CCC_score": r.get("CCC") or r.get("score") or r.get("ccc"),
            "CCC_CI95_lo": r.get("CI_lo") or r.get("ccc_lo"),
            "CCC_CI95_hi": r.get("CI_hi") or r.get("ccc_hi"),
            "decision": r.get("decision") or r.get("label"),
            "notes": r.get("notes") or "",
            "source_file": r.get("source_file", ""),
        })
    return out

# ---------------------------------------------------------------------------
# Build API
# ---------------------------------------------------------------------------

def build_cc() -> List[Dict[str, Any]]:
    rows = _collect_cc_rows()
    _write_rows_csv(rows, REPORTS_DIR / "summary_cc.csv")
    _write_md(rows, REPORTS_DIR / "readiness_cc.md", "CC Readiness Report")
    return rows

def build_ccc() -> List[Dict[str, Any]]:
    raw = _read_ccc_addenda_csvs()
    rows = _normalize_ccc_rows(raw)
    # write CCC-only summaries
    out_csv = REPORTS_DIR / "summary_ccc.csv"
    out_md  = REPORTS_DIR / "readiness_ccc.md"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        header = list(rows[0].keys())
        with out_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=header)
            w.writeheader(); w.writerows(rows)
    else:
        with out_csv.open("w", newline="") as f:
            csv.writer(f).writerow(["flavor","run_id","path","CCC_score","CCC_CI95_lo","CCC_CI95_hi","decision","notes","source_file"])
    _write_md(rows, out_md, "CCC Readiness Report")
    return rows

def build_all(mode: str = "all") -> None:
    """Build reports. mode ∈ {'cc','ccc','all'}."""
    mode = (mode or "all").lower()
    cc_rows: List[Dict[str, Any]] = []
    ccc_rows: List[Dict[str, Any]] = []
    if mode in ("cc", "all"):
        cc_rows = build_cc()
    if mode in ("ccc", "all"):
        ccc_rows = build_ccc()
    # merged pane
    merged: List[Dict[str, Any]] = []
    merged.extend(cc_rows)
    merged.extend(ccc_rows)
    _write_rows_csv(merged, REPORTS_DIR / "summary_all.csv")
    _write_md(merged, REPORTS_DIR / "readiness_all.md", "CC/CCC Readiness Report")
    print(f"[reporting] Built mode={mode}. See {REPORTS_DIR}/summary_* and readiness_*.md")
