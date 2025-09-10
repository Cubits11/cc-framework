# src/cc/cartographer/cli.py
"""
Cartographer CLI

Subcommands:
  - run             Execute a single run (scores → J's → CC metrics → figure → audit append)
  - verify-audit    Verify the tamper-evident JSONL audit chain
  - verify-stats    Lightweight bootstrap sanity check on score plumbing
  - build-reports   Aggregate CC/CCC results into CSV/Markdown under evaluation/reports

Examples:
  python -m cc.cartographer.cli run \
      --config experiments/configs/smoke.yaml \
      --samples 200 \
      --fig paper/figures/phase_diagram.pdf \
      --audit runs/audit.jsonl

  python -m cc.cartographer.cli verify-audit --audit runs/audit.jsonl
  python -m cc.cartographer.cli verify-stats --config experiments/configs/smoke.yaml --bootstrap 5000

  python -m cc.cartographer.cli build-reports --mode all
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, Mapping, Optional, Tuple, Literal, List

# Core cartographer surfaces
from cc.cartographer import atlas, audit, bounds, io, stats
# Reporting aggregator (CC/CCC)
from cc.analysis import reporting


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _norm_comp(x: Any) -> Literal["AND", "OR"]:
    """
    Normalize composition label to {'AND','OR'}.
    Defaults to 'AND' if unspecified/unrecognized.
    """
    s = str(x).strip().upper() if x is not None else "AND"
    return "AND" if s == "AND" else "OR"


def _maybe_post_github_comment(context_json: str, body: str) -> None:
    """
    Optional GitHub comment posting stub.

    Historical code attempted network side-effects; to keep this CLI
    tool hermetic and CI-friendly, we *emit* a structured line that a
    workflow step can consume to post a comment via `gh api` or an
    Action. If `context_json` is absent/invalid, this is a no-op.

    Expected `context_json` (example):
        {
          "repo": "owner/name",
          "issue_number": 123,
          "sha": "abcdef...",
          "actor": "octocat"
        }
    """
    if not context_json:
        return
    try:
        ctx = json.loads(context_json)
        payload = {
            "repo": ctx.get("repo"),
            "issue_number": ctx.get("issue_number"),
            "sha": ctx.get("sha"),
            "actor": ctx.get("actor"),
            "body": body,
        }
        # Single-line JSON for easy parsing in CI
        print(f"::github-comment::{json.dumps(payload, separators=(',',':'))}")
    except Exception:
        # Silently ignore malformed context to avoid breaking runs.
        return


# -----------------------------------------------------------------------------
# Subcommands
# -----------------------------------------------------------------------------

def _cmd_run(argv: List[str]) -> None:
    p = argparse.ArgumentParser(
        prog="cc.cartographer.cli run",
        description="Execute a single run: load scores, compute J/CI and CC, draw a figure, and append to audit."
    )
    p.add_argument("--config", required=True, help="Path to YAML config (experiments/configs/*.yaml)")
    p.add_argument("--samples", type=int, default=200, help="Override sample count for synthetic loaders")
    p.add_argument("--fig", required=True, help="Path to write the phase-point figure (PNG/PDF)")
    p.add_argument("--audit", required=True, help="Path to JSONL audit log (append-only)")
    p.add_argument("--post-comment", type=str, default="false", help="If 'true', emit GH comment payload line")
    p.add_argument("--github-context", type=str, default="", help="JSON string with repo/issue/sha/actor")
    args = p.parse_args(argv)

    # Load config and generate/ingest scores
    cfg: Mapping[str, Any] = dict(io.load_config(args.config))
    data: Mapping[str, Any] = io.load_scores(cfg, n=args.samples)

    # Per-component J and CI
    JA, JA_ci = stats.compute_j_ci(data["A0"], data["A1"])  # (J_A, (lo,hi))
    JB, JB_ci = stats.compute_j_ci(data["B0"], data["B1"])

    # Composite J: empirical if Comp* provided, else FH ceiling on ROC
    comp: Literal["AND", "OR"] = _norm_comp(cfg.get("comp", "AND"))
    if data.get("Comp0") is not None and data.get("Comp1") is not None:
        Jc, Jc_ci = stats.compute_j_ci(data["Comp0"], data["Comp1"])
        comp_label = "empirical"
    else:
        # frechet_upper expects ROC-like sequences or arrays
        Jc = float(bounds.frechet_upper(data["rocA"], data["rocB"], comp=comp))
        Jc_ci = (float("nan"), float("nan"))  # ceiling has no sampling CI
        comp_label = "UPPER BOUND"

    # Compose CC (and optionally Δ_add or others as your stats module defines)
    CC, Dadd = stats.compose_cc(JA, JB, Jc)

    # Figure and human-friendly entry/decision
    fig_path = atlas.plot_phase_point(cfg, CC, args.fig)
    entry, decision = atlas.compose_entry(
        cfg, JA, JA_ci, JB, JB_ci, Jc, Jc_ci, CC, Dadd, comp_label, fig_path
    )

    # Tamper-evident audit append
    rec = audit.make_record(cfg, JA, JA_ci, JB, JB_ci, Jc, Jc_ci, CC, Dadd, decision, [fig_path])
    sha = audit.append_jsonl(args.audit, rec)

    # Console outputs (human + machine)
    print(entry)
    print(decision)
    print(f"audit_sha: {sha}")

    # Optional GitHub comment emission
    if args.post_comment.lower() == "true" and args.github_context:
        _maybe_post_github_comment(args.github_context, entry + "\n\n" + decision + f"\n\n**audit_sha:** `{sha}`")


def _cmd_verify_audit(argv: List[str]) -> None:
    p = argparse.ArgumentParser(
        prog="cc.cartographer.cli verify-audit",
        description="Verify integrity of the append-only JSONL audit chain."
    )
    p.add_argument("--audit", required=True, help="Path to JSONL audit log")
    args = p.parse_args(argv)
    audit.verify_chain(args.audit)
    print("audit chain OK")


def _cmd_verify_stats(argv: List[str]) -> None:
    p = argparse.ArgumentParser(
        prog="cc.cartographer.cli verify-stats",
        description="Run bootstrap diagnostics to sanity-check score plumbing."
    )
    p.add_argument("--config", required=True, help="Path to YAML config")
    p.add_argument("--bootstrap", type=int, default=10_000, help="Bootstrap resamples (module may cap internally)")
    args = p.parse_args(argv)

    cfg = io.load_config(args.config)
    data = io.load_scores(cfg, n=None)
    stats.bootstrap_diagnostics(data, B=args.bootstrap)
    print("bootstrap diagnostics OK")


def _cmd_build_reports(argv: List[str]) -> None:
    p = argparse.ArgumentParser(
        prog="cc.cartographer.cli build-reports",
        description="Aggregate reports from existing artifacts under results/ and evaluation/ccc/addenda."
    )
    p.add_argument(
        "--mode",
        default="all",
        choices=["cc", "ccc", "all"],
        help="Which reports to build (default: all)"
    )
    args = p.parse_args(argv)
    reporting.build_all(mode=args.mode)


# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------

def main() -> None:
    if len(sys.argv) < 2:
        print(
            "Usage: python -m cc.cartographer.cli {run|verify-audit|verify-stats|build-reports} ...",
            file=sys.stderr,
        )
        sys.exit(2)

    cmd, argv = sys.argv[1], sys.argv[2:]
    if cmd == "run":
        _cmd_run(argv)
    elif cmd == "verify-audit":
        _cmd_verify_audit(argv)
    elif cmd == "verify-stats":
        _cmd_verify_stats(argv)
    elif cmd == "build-reports":
        _cmd_build_reports(argv)
    else:
        print(f"unknown subcommand: {cmd}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
