# src/cc/cartographer/cli.py
"""
Cartographer CLI

Subcommands:
  - run:           Execute a single run (scores→J’s→CC metrics→figure→audit append)
  - verify-audit:  Verify the tamper-evident JSONL audit chain
  - verify-stats:  Lightweight bootstrap sanity check on score plumbing

Examples:
  python -m cc.cartographer.cli run \
      --config src/cc/exp/configs/smoke.yaml \
      --samples 200 \
      --fig paper/figures/phase_diagram.pdf \
      --audit runs/audit.jsonl

  python -m cc.cartographer.cli verify-audit --audit runs/audit.jsonl
  python -m cc.cartographer.cli verify-stats --config src/cc/exp/configs/smoke.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, Mapping, Optional, Tuple

from cc.cartographer import atlas, audit, bounds, io, stats


# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------

def _norm_comp(x: Any) -> str:
    """Normalize composition label to {'AND','OR'} with a default."""
    s = str(x).strip().upper() if x is not None else "AND"
    return "AND" if s == "AND" else "OR"


def _maybe_post_github_comment(context_json: str, body: str) -> None:
    """
    Optional GitHub comment posting.

    The old code called `io.post_github_comment`, which isn't part of the io module.
    To keep the CLI self-contained and avoid network deps, we accept a JSON `context`
    string and simply print a structured line that a CI step can consume to post
    the comment (e.g., via `gh api` or a workflow action).

    Expected `context_json` (example):
        {
          "repo": "owner/name",
          "issue_number": 123,
          "sha": "abcdef...",
          "actor": "octocat"
        }

    If invalid or empty, this is a no-op.
    """
    if not context_json:
        return
    try:
        ctx = json.loads(context_json)
        if not isinstance(ctx, dict):
            return
        print("::group::GITHUB_COMMENT_CONTEXT")
        print(json.dumps({"context": ctx, "body": body}, indent=2))
        print("::endgroup::")
    except Exception:
        # Swallow; posting is best-effort and should not break runs.
        pass


# -----------------------------------------------------------------------------
# Subcommands
# -----------------------------------------------------------------------------

def _cmd_run(argv: list[str]) -> None:
    p = argparse.ArgumentParser(prog="cc.cartographer.cli run")
    p.add_argument("--config", required=True, help="Path to YAML config")
    p.add_argument("--samples", type=int, default=200, help="Override sample count")
    p.add_argument("--fig", required=True, help="Path to write the phase-point figure (PDF)")
    p.add_argument("--audit", required=True, help="Path to JSONL audit log")
    p.add_argument("--post-comment", type=str, default="false", help="Emit structured GH comment payload")
    p.add_argument("--github-context", type=str, default="", help="JSON string for GitHub context")
    args = p.parse_args(argv)

    # Load config + synthetic scores for smoke/e2e
    cfg = dict(io.load_config(args.config))
    data: Mapping[str, Any] = io.load_scores(cfg, n=args.samples)

    # Component J’s + CI via deterministic bootstrap
    JA, JA_ci = stats.compute_j_ci(data["A0"], data["A1"])
    JB, JB_ci = stats.compute_j_ci(data["B0"], data["B1"])

    # Compose: empirical if provided, else FH upper bound on ROC curves
    comp = _norm_comp(cfg.get("comp", "AND"))
    if data.get("Comp0") is not None and data.get("Comp1") is not None:
        Jc, Jc_ci = stats.compute_j_ci(data["Comp0"], data["Comp1"])
        comp_label = "empirical"
    else:
        # bounds.frechet_upper expects ROC-like sequences [(FPR,TPR), ...] or (K,2) arrays
        Jc = float(bounds.frechet_upper(data["rocA"], data["rocB"], comp=comp))
        Jc_ci = (None, None)  # ceiling has no sampling CI here
        comp_label = "UPPER BOUND"

    # Dashboard helpers (note: cc_max is a reporting normalization; not a theorem)
    CC, Dadd = stats.compose_cc(JA, JB, Jc)

    # Figure + journal entry
    fig_path = atlas.plot_phase_point(cfg, CC, args.fig)
    entry, decision = atlas.compose_entry(
        cfg, JA, JA_ci, JB, JB_ci, Jc, Jc_ci, CC, Dadd, comp_label, fig_path
    )

    # Tamper-evident audit append
    rec = audit.make_record(cfg, JA, JA_ci, JB, JB_ci, Jc, Jc_ci, CC, Dadd, decision, [fig_path])
    sha = audit.append_jsonl(args.audit, rec)

    # Console output (human + machine-readable)
    print(entry)
    print(decision)
    print(f"audit_sha: {sha}")

    # Optional GitHub comment payload emission
    if args.post_comment.lower() == "true" and args.github_context:
        _maybe_post_github_comment(
            args.github_context, entry + "\n\n" + decision + f"\n\n**audit_sha:** `{sha}`"
        )


def _cmd_verify_audit(argv: list[str]) -> None:
    p = argparse.ArgumentParser(prog="cc.cartographer.cli verify-audit")
    p.add_argument("--audit", required=True, help="Path to JSONL audit log")
    args = p.parse_args(argv)
    audit.verify_chain(args.audit)
    print("audit chain OK")


def _cmd_verify_stats(argv: list[str]) -> None:
    p = argparse.ArgumentParser(prog="cc.cartographer.cli verify-stats")
    p.add_argument("--config", required=True, help="Path to YAML config")
    p.add_argument("--bootstrap", type=int, default=10_000, help="Bootstrap resamples (capped internally)")
    args = p.parse_args(argv)

    cfg = io.load_config(args.config)
    data = io.load_scores(cfg, n=None)
    stats.bootstrap_diagnostics(data, B=args.bootstrap)
    print("bootstrap diagnostics OK")


# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------

def main() -> None:
    if len(sys.argv) < 2:
        print(
            "Usage: python -m cc.cartographer.cli {run|verify-audit|verify-stats} ...",
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
    else:
        print(f"unknown subcommand: {cmd}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()