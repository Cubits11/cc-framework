# deployment/cli/cc/methods.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple

import click
import numpy as np

from cc.analysis.cc_estimation import estimate_cc_methods_from_rates
from cc.cartographer.intervals import (
    cc_ci_wilson,
    cc_ci_bootstrap,
)
from cc.analysis.generate_figures import plot_roc_fh_slice

def _read_binary_series(path: Path) -> np.ndarray:
    """Read 0/1 values from a text or csv file (one per line or comma-separated)."""
    txt = Path(path).read_text().strip().replace(",", " ")
    vals = [float(v) for v in txt.split() if v != ""]
    arr = np.array(vals, dtype=float).ravel()
    if not np.isin(arr, [0.0, 1.0]).all():
        uniq = np.unique(arr)
        raise ValueError(f"{path} contains non-binary values: {uniq}")
    return arr

def _maybe_counts_to_phat(k: Optional[int], n: Optional[int]) -> Optional[float]:
    if k is None and n is None:
        return None
    if k is None or n is None:
        raise click.BadParameter("Provide both k and n for counts, or neither.")
    if n <= 0 or k < 0 or k > n:
        raise click.BadParameter("Counts must satisfy 0 <= k <= n and n > 0.")
    return k / n

@click.command(context_settings={"show_default": True})
@click.option("--D", "D", type=float, required=True, help="Denominator (>0), D = min_r (1 - J_r(θ_r*)).")
@click.option("--tpr-a", type=float, required=True)
@click.option("--tpr-b", type=float, required=True)
@click.option("--fpr-a", type=float, required=True)
@click.option("--fpr-b", type=float, required=True)
@click.option("--n1", type=int, required=True, help="Number of Y=1 samples.")
@click.option("--n0", type=int, required=True, help="Number of Y=0 samples.")
@click.option("--k1", type=int, help="Successes for p1_hat (A∧B on Y=1).")
@click.option("--k0", type=int, help="Successes for p0_hat (A∨B on Y=0).")
@click.option("--y1-samples", type=click.Path(exists=True, dir_okay=False), help="Path to 0/1 series for Y=1 composite (A∧B).")
@click.option("--y0-samples", type=click.Path(exists=True, dir_okay=False), help="Path to 0/1 series for Y=0 composite (A∨B).")
@click.option("--alpha-cap", type=float, default=None, help="Policy cap α for FPR; binds I0 upper.")
@click.option("--delta", type=float, default=0.05, help="Two-sided risk for CIs.")
@click.option("--target-t", type=float, default=None, help="Optional target half-width t for CC (planner).")
@click.option("--bootstrap-B", type=int, default=2000, help="Bootstrap replicates (if samples provided).")
@click.option("--seed", type=int, default=7, help="Random seed for bootstrap.")
@click.option("--figure-out", type=click.Path(dir_okay=False), default=None, help="If provided, save ROC+FH figure here (PNG).")
@click.option("--json-out", type=click.Path(dir_okay=False), default=None, help="If provided, dump JSON with all numbers.")
def methods(
    D: float,
    tpr_a: float,
    tpr_b: float,
    fpr_a: float,
    fpr_b: float,
    n1: int,
    n0: int,
    k1: Optional[int],
    k0: Optional[int],
    y1_samples: Optional[str],
    y0_samples: Optional[str],
    alpha_cap: Optional[float],
    delta: float,
    target_t: Optional[float],
    bootstrap_B: int,
    seed: int,
    figure_out: Optional[str],
    json_out: Optional[str],
):
    """
    Print FH intervals, variance envelopes, CC_hat, and three CIs (Bernstein/ Wilson/ Bootstrap),
    plus per-class n* from the Bernstein planner. Optionally saves a ROC+FH figure and a JSON dump.
    """
    # Resolve p-hats from counts or sample files (one method must be chosen)
    p1_hat = _maybe_counts_to_phat(k1, n1)
    p0_hat = _maybe_counts_to_phat(k0, n0)

    y1 = y0 = None
    if y1_samples or y0_samples:
        if not (y1_samples and y0_samples):
            raise click.BadParameter("Provide both --y1-samples and --y0-samples when bootstrapping.")
        y1 = _read_binary_series(Path(y1_samples))
        y0 = _read_binary_series(Path(y0_samples))
        if len(y1) != n1 or len(y0) != n0:
            click.echo("Warning: provided n1/n0 do not match sample lengths; using lengths from files.")
            n1 = len(y1)
            n0 = len(y0)
        p1_hat = float(np.mean(y1))
        p0_hat = float(np.mean(y0))

    if p1_hat is None or p0_hat is None:
        raise click.BadParameter("Provide either counts (k1,k0) or sample files (y1-samples,y0-samples).")

    # Core FH–Bernstein workflow
    report = estimate_cc_methods_from_rates(
        p1_hat=p1_hat, p0_hat=p0_hat,
        D=D,
        tpr_a=tpr_a, tpr_b=tpr_b, fpr_a=fpr_a, fpr_b=fpr_b,
        n1=n1, n0=n0,
        alpha_cap=alpha_cap,
        delta=delta,
        target_t=target_t,
    )

    # Wilson CC CI
    wil_lo, wil_hi = cc_ci_wilson(p1_hat, n1, p0_hat, n0, D, delta)

    # Bootstrap CC CI (only if samples provided)
    boo_lo = boo_hi = None
    if y1 is not None and y0 is not None:
        boo_lo, boo_hi = cc_ci_bootstrap(y1, y0, D, delta, B=bootstrap_B, seed=seed)

    # Pretty print
    point = report["point"]
    bounds = report["bounds"]
    ci_b = report["ci"]

    def fmt_iv(iv: Tuple[float,float]) -> str:
        return f"[{iv[0]:.4f}, {iv[1]:.4f}]"

    click.echo("\n=== CC Methods @ θ ===")
    click.echo(f"  p1_hat (A∧B|Y=1): {p1_hat:.4f}  (n1={n1})")
    click.echo(f"  p0_hat (A∨B|Y=0): {p0_hat:.4f}  (n0={n0})")
    click.echo(f"  D: {D:.6f}   CC_hat: {point['cc_hat']:.4f}")
    click.echo(f"  I1 (FH AND, Y=1): {fmt_iv(tuple(bounds['I1']))}   v̄1: {bounds['vbar1']:.4f}")
    click.echo(f"  I0 (FH OR , Y=0): {fmt_iv(tuple(bounds['I0']))}   v̄0: {bounds['vbar0']:.4f}   α-cap: {alpha_cap}")

    click.echo("\n  CIs (two-sided, δ = {:.3f}):".format(delta))
    click.echo(f"    FH–Bernstein: [{ci_b['lo']:.4f}, {ci_b['hi']:.4f}]  (planner target t={ci_b.get('target_t')})")
    click.echo(f"    Wilson      : [{wil_lo:.4f}, {wil_hi:.4f}]")
    if boo_lo is not None:
        click.echo(f"    Bootstrap   : [{boo_lo:.4f}, {boo_hi:.4f}]   (B={bootstrap_B}, seed={seed})")
    else:
        click.echo(f"    Bootstrap   : (skipped — provide --y1-samples/--y0-samples)")

    if ci_b.get("n1_star") is not None:
        click.echo("\n  Planner (per-class, each term ≤ δ/2):")
        click.echo(f"    n1* ≈ {ci_b['n1_star']:.1f}   n0* ≈ {ci_b['n0_star']:.1f}")

    # Optional figure
    if figure_out:
        out = Path(figure_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        plot_roc_fh_slice(
            tpr_a=tpr_a, fpr_a=fpr_a,
            tpr_b=tpr_b, fpr_b=fpr_b,
            alpha_cap=alpha_cap,
            I1=tuple(bounds["I1"]),
            I0=tuple(bounds["I0"]),
            D=D, n1=n1, n0=n0, delta=delta,
            outpath=str(out),
        )
        click.echo(f"\nFigure saved → {out}")

    # Optional JSON
    if json_out:
        payload = {
            "point": point,
            "bounds": bounds,
            "ci": ci_b,
            "wilson": {"lo": wil_lo, "hi": wil_hi},
            "bootstrap": {"lo": boo_lo, "hi": boo_hi, "B": bootstrap_B, "seed": seed} if boo_lo is not None else None,
        }
        Path(json_out).write_text(json.dumps(payload, indent=2))
        click.echo(f"JSON saved → {json_out}")

if __name__ == "__main__":
    methods()
