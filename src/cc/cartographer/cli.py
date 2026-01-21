# src/cc/cartographer/cli.py
"""
Cartographer CLI

Subcommands:
  - run             Execute a single run (scores → J's → CC metrics → figure → audit append)
  - verify-audit    Verify the tamper-evident JSONL audit chain
  - verify-stats    Lightweight bootstrap sanity check on score plumbing
  - build-reports   Aggregate CC/CCC results into CSV/Markdown under evaluation/reports
  - methods         FH–Bernstein + Wilson + (optional) Bootstrap CC CI at a fixed θ

Examples:
  python -m cc.cartographer.cli run \
      --config experiments/configs/smoke.yaml \
      --samples 200 \
      --fig paper/figures/phase_diagram.pdf \
      --audit runs/audit.jsonl

  python -m cc.cartographer.cli verify-audit --audit runs/audit.jsonl
  python -m cc.cartographer.cli verify-stats --config experiments/configs/smoke.yaml --bootstrap 5000
  python -m cc.cartographer.cli build-reports --mode all

  # Week-3 methods (counts path)
  python -m cc.cartographer.cli methods \
      --D 0.55 \
      --tpr-a 0.72 --tpr-b 0.65 --fpr-a 0.035 --fpr-b 0.050 \
      --n1 1200 --n0 1200 \
      --k1 744 --k0 138 \
      --alpha-cap 0.04 --delta 0.05 \
      --figure-out figs/fig_week3_roc_fh.png \
      --json-out runs/week3_methods.json

  # Week-3 methods (bootstrap path)
  python -m cc.cartographer.cli methods \
      --D 0.55 \
      --tpr-a 0.72 --tpr-b 0.65 --fpr-a 0.035 --fpr-b 0.050 \
      --n1 1200 --n0 1200 \
      --y1-samples runs/y1_ab_and.bin \
      --y0-samples runs/y0_ab_or.bin \
      --alpha-cap 0.04 --delta 0.05 --bootstrap-B 2000 --seed 7 \
      --figure-out figs/fig_week3_roc_fh.png \
      --json-out runs/week3_methods.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, List, Literal, Mapping, Optional

import numpy as np

# Reporting aggregator (CC/CCC)
from cc.analysis import reporting

# Week-3 methods plumbing
from cc.analysis.cc_estimation import estimate_cc_methods_from_rates

# Core cartographer surfaces
from cc.cartographer import atlas, audit, bounds, io, stats
from cc.cartographer.intervals import cc_ci_bootstrap, cc_ci_wilson

# Optional figure helper (present if you added plot_roc_fh_slice earlier)
try:
    from cc.analysis.generate_figures import plot_roc_fh_slice
except Exception:
    plot_roc_fh_slice = None  # soft dependency


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
    Optional GitHub comment emission: we *emit* a structured line that CI can pick up.
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
        print(f"::github-comment::{json.dumps(payload, separators=(',', ':'))}")
    except Exception:
        return


def _read_binary_series(path: Path) -> np.ndarray:
    """
    Read 0/1 values from a text/csv file (newline or comma-separated).
    Raises if any value is not {0,1}.
    """
    txt = Path(path).read_text().strip().replace(",", " ")
    vals = [float(v) for v in txt.split() if v]
    arr = np.array(vals, dtype=float).ravel()
    if not np.isin(arr, [0.0, 1.0]).all():
        uniq = np.unique(arr)
        raise ValueError(f"{path} contains non-binary values: {uniq}")
    return arr


def _maybe_counts_to_phat(k: Optional[int], n: Optional[int]) -> Optional[float]:
    """
    Convert counts to proportion if both provided; otherwise return None.
    """
    if k is None and n is None:
        return None
    if k is None or n is None:
        raise ValueError("Provide both k and n for counts, or neither.")
    if n <= 0 or k < 0 or k > n:
        raise ValueError("Counts must satisfy 0 <= k <= n and n > 0.")
    return k / n


# -----------------------------------------------------------------------------
# Subcommands
# -----------------------------------------------------------------------------


def _cmd_run(argv: List[str]) -> None:
    p = argparse.ArgumentParser(
        prog="cc.cartographer.cli run",
        description="Execute a single run: load scores, compute J/CI and CC, draw a figure, and append to audit.",
    )
    p.add_argument(
        "--config", required=True, help="Path to YAML config (experiments/configs/*.yaml)"
    )
    p.add_argument(
        "--samples", type=int, default=200, help="Override sample count for synthetic loaders"
    )
    p.add_argument("--fig", required=True, help="Path to write the phase-point figure (PNG/PDF)")
    p.add_argument("--audit", required=True, help="Path to JSONL audit log (append-only)")
    p.add_argument(
        "--post-comment", type=str, default="false", help="If 'true', emit GH comment payload line"
    )
    p.add_argument(
        "--github-context", type=str, default="", help="JSON string with repo/issue/sha/actor"
    )
    args = p.parse_args(argv)

    # Load config and generate/ingest scores
    cfg: Mapping[str, Any] = dict(io.load_config(args.config))
    data: Mapping[str, Any] = io.load_scores(cfg, n=args.samples)

    # Per-component J and CI
    JA, JA_ci = stats.compute_j_ci(data["A0"], data["A1"])
    JB, JB_ci = stats.compute_j_ci(data["B0"], data["B1"])

    # Composite J: empirical if Comp* provided, else FH ceiling on ROC
    comp: Literal["AND", "OR"] = _norm_comp(cfg.get("comp", "AND"))
    if data.get("Comp0") is not None and data.get("Comp1") is not None:
        Jc, Jc_ci = stats.compute_j_ci(data["Comp0"], data["Comp1"])
        comp_label = "empirical"
    else:
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
    rec = audit.make_record(
        cfg, None, JA, JA_ci, JB, JB_ci, Jc, Jc_ci, CC, Dadd, decision, [fig_path]
    )
    sha = audit.append_jsonl(args.audit, rec)

    # Console outputs (human + machine)
    print(entry)
    print(decision)
    print(f"audit_sha: {sha}")

    # Optional GitHub comment emission
    if args.post_comment.lower() == "true" and args.github_context:
        _maybe_post_github_comment(
            args.github_context, entry + "\n\n" + decision + f"\n\n**audit_sha:** `{sha}`"
        )


def _cmd_verify_audit(argv: List[str]) -> None:
    p = argparse.ArgumentParser(
        prog="cc.cartographer.cli verify-audit",
        description="Verify integrity of the append-only JSONL audit chain.",
    )
    p.add_argument("--audit", required=True, help="Path to JSONL audit log")
    args = p.parse_args(argv)
    audit.verify_chain(args.audit)
    print("audit chain OK")


def _cmd_verify_stats(argv: List[str]) -> None:
    p = argparse.ArgumentParser(
        prog="cc.cartographer.cli verify-stats",
        description="Run bootstrap diagnostics to sanity-check score plumbing.",
    )
    p.add_argument("--config", required=True, help="Path to YAML config")
    p.add_argument(
        "--bootstrap",
        type=int,
        default=10_000,
        help="Bootstrap resamples (module may cap internally)",
    )
    args = p.parse_args(argv)

    cfg = io.load_config(args.config)
    data = io.load_scores(cfg, n=None)
    stats.bootstrap_diagnostics(data, B=args.bootstrap)
    print("bootstrap diagnostics OK")


def _cmd_build_reports(argv: List[str]) -> None:
    p = argparse.ArgumentParser(
        prog="cc.cartographer.cli build-reports",
        description="Aggregate reports from existing artifacts under results/ and evaluation/ccc/addenda.",
    )
    p.add_argument(
        "--mode",
        default="all",
        choices=["cc", "ccc", "all"],
        help="Which reports to build (default: all)",
    )
    args = p.parse_args(argv)
    reporting.build_all(mode=args.mode)


def _cmd_methods(argv: List[str]) -> None:
    """
    Week-3: FH–Bernstein + Wilson + (optional) Bootstrap CC CI at a fixed θ, with planner and figure.
    """
    p = argparse.ArgumentParser(
        prog="cc.cartographer.cli methods",
        description="Compute FH–Bernstein + Wilson + Bootstrap CC CIs at a fixed operating point θ.",
    )
    # Required operating context
    p.add_argument(
        "--D", type=float, required=True, help="Denominator (>0): D = min_r (1 - J_r(θ_r*))."
    )
    p.add_argument("--tpr-a", type=float, required=True)
    p.add_argument("--tpr-b", type=float, required=True)
    p.add_argument("--fpr-a", type=float, required=True)
    p.add_argument("--fpr-b", type=float, required=True)
    p.add_argument("--n1", type=int, required=True, help="Y=1 sample size.")
    p.add_argument("--n0", type=int, required=True, help="Y=0 sample size.")

    # Either counts or sample files
    p.add_argument("--k1", type=int, help="Successes for p1_hat (A∧B | Y=1).")
    p.add_argument("--k0", type=int, help="Successes for p0_hat (A∨B | Y=0).")
    p.add_argument("--y1-samples", type=str, help="Path to 0/1 series for Y=1 composite (A∧B).")
    p.add_argument("--y0-samples", type=str, help="Path to 0/1 series for Y=0 composite (A∨B).")

    # Policy + risk + planner
    p.add_argument("--alpha-cap", type=float, default=None, help="Policy cap α: binds I0 upper.")
    p.add_argument(
        "--delta", type=float, default=0.05, help="Two-sided risk for CIs (default 0.05)."
    )
    p.add_argument(
        "--target-t", type=float, default=None, help="Optional target half-width t for CC planner."
    )

    # Bootstrap controls
    p.add_argument(
        "--bootstrap-B", type=int, default=2000, help="Bootstrap replicates if samples provided."
    )
    p.add_argument("--seed", type=int, default=7, help="RNG seed for bootstrap.")

    # Outputs
    p.add_argument("--figure-out", type=str, default=None, help="Save ROC+FH figure (PNG).")
    p.add_argument("--json-out", type=str, default=None, help="Dump JSON with all numbers.")
    args = p.parse_args(argv)

    # Resolve p-hats from counts or sample files
    p1_hat = _maybe_counts_to_phat(args.k1, args.n1)
    p0_hat = _maybe_counts_to_phat(args.k0, args.n0)

    y1 = y0 = None
    if args.y1_samples or args.y0_samples:
        if not (args.y1_samples and args.y0_samples):
            raise SystemExit("Provide both --y1-samples and --y0-samples when bootstrapping.")
        y1 = _read_binary_series(Path(args.y1_samples))
        y0 = _read_binary_series(Path(args.y0_samples))
        if len(y1) != args.n1 or len(y0) != args.n0:
            print("Warning: n1/n0 differ from file lengths; using lengths from files.")
            args.n1 = len(y1)
            args.n0 = len(y0)
        p1_hat = float(np.mean(y1))
        p0_hat = float(np.mean(y0))

    if p1_hat is None or p0_hat is None:
        raise SystemExit("Provide either counts (k1,k0) or sample files (y1-samples,y0-samples).")

    # FH–Bernstein core (computes FH intervals, variance envelopes, and planner)
    report = estimate_cc_methods_from_rates(
        p1_hat=p1_hat,
        p0_hat=p0_hat,
        D=args.D,
        tpr_a=args.tpr_a,
        tpr_b=args.tpr_b,
        fpr_a=args.fpr_a,
        fpr_b=args.fpr_b,
        n1=args.n1,
        n0=args.n0,
        alpha_cap=args.alpha_cap,
        delta=args.delta,
        target_t=args.target_t,
    )

    # Wilson CC CI
    wil_lo, wil_hi = cc_ci_wilson(p1_hat, args.n1, p0_hat, args.n0, args.D, args.delta)

    # Bootstrap CC CI (if samples provided)
    boo_lo = boo_hi = None
    if y1 is not None and y0 is not None:
        boo_lo, boo_hi = cc_ci_bootstrap(
            y1, y0, args.D, args.delta, B=args.bootstrap_B, seed=args.seed
        )

    # Pretty print
    point = report["point"]
    bounds = report["bounds"]
    ci_b = report["ci"]

    def fmt_iv(iv: tuple[float, float]) -> str:
        return f"[{iv[0]:.4f}, {iv[1]:.4f}]"

    print("\n=== CC Methods @ θ ===")
    print(f"  p1_hat (A∧B|Y=1): {p1_hat:.4f} (n1={args.n1})")
    print(f"  p0_hat (A∨B|Y=0): {p0_hat:.4f} (n0={args.n0})")
    print(f"  D: {args.D:.6f}   CC_hat: {point['cc_hat']:.4f}")
    print(f"  I1 (FH AND, Y=1): {fmt_iv(tuple(bounds['I1']))}   v̄1: {bounds['vbar1']:.4f}")
    print(
        f"  I0 (FH OR , Y=0): {fmt_iv(tuple(bounds['I0']))}   v̄0: {bounds['vbar0']:.4f}   α-cap: {args.alpha_cap}"
    )

    print(f"\n  CIs (two-sided, δ = {args.delta:.3f}):")
    print(
        f"    FH–Bernstein: [{ci_b['lo']:.4f}, {ci_b['hi']:.4f}]  (planner t={ci_b.get('target_t')})"
    )
    print(f"    Wilson      : [{wil_lo:.4f}, {wil_hi:.4f}]")
    if boo_lo is not None:
        print(
            f"    Bootstrap   : [{boo_lo:.4f}, {boo_hi:.4f}]   (B={args.bootstrap_B}, seed={args.seed})"
        )
    else:
        print("    Bootstrap   : (skipped — provide --y1-samples/--y0-samples)")

    if ci_b.get("n1_star") is not None:
        print("\n  Planner (per-class, each term ≤ δ/2):")
        print(f"    n1* ≈ {ci_b['n1_star']:.1f}   n0* ≈ {ci_b['n0_star']:.1f}")

    # Optional figure
    if args.figure_out:
        if plot_roc_fh_slice is None:
            print(
                "Figure helper not available; add plot_roc_fh_slice() to cc.analysis.generate_figures."
            )
        else:
            out = Path(args.figure_out)
            out.parent.mkdir(parents=True, exist_ok=True)
            plot_roc_fh_slice(
                tpr_a=args.tpr_a,
                fpr_a=args.fpr_a,
                tpr_b=args.tpr_b,
                fpr_b=args.fpr_b,
                alpha_cap=args.alpha_cap,
                I1=tuple(bounds["I1"]),
                I0=tuple(bounds["I0"]),
                D=args.D,
                n1=args.n1,
                n0=args.n0,
                delta=args.delta,
                outpath=str(out),
            )
            print(f"\nFigure saved → {out}")

    # Optional JSON dump
    if args.json_out:
        payload = {
            "point": point,
            "bounds": bounds,
            "ci": ci_b,
            "wilson": {"lo": wil_lo, "hi": wil_hi},
            "bootstrap": (
                {"lo": boo_lo, "hi": boo_hi, "B": args.bootstrap_B, "seed": args.seed}
                if boo_lo is not None
                else None
            ),
        }
        Path(args.json_out).write_text(json.dumps(payload, indent=2))
        print(f"JSON saved → {args.json_out}")


# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------


def main() -> None:
    if len(sys.argv) < 2:
        print(
            "Usage: python -m cc.cartographer.cli {run|verify-audit|verify-stats|build-reports|methods} ...",
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
        _cmd_build_reports(argv)  # (fixed typo)
    elif cmd == "methods":
        _cmd_methods(argv)
    else:
        print(f"unknown subcommand: {cmd}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
