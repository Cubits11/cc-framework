# src/cc/analysis/generate_figures.py
"""
Module: cc.analysis.generate_figures
Purpose: Build figures and summary.csv from audit history with Week-3 extensions.

CLI:
  python -m cc.analysis.generate_figures \
    --history runs/audit.jsonl \
    --fig-dir paper/figures \
    --out-dir results/smoke/aggregates
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

# -------------------- Matplotlib (paper-safe, colorblind-friendly) ------------
mpl.rcParams.update(
    {
        "pdf.fonttype": 42,
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
        # clean white backgrounds (fixes “dark checkerboard”)
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.edgecolor": "white",
        "savefig.transparent": False,
    }
)

# Accessible palette
CBLUE = "#1f77b4"
CRED  = "#d62728"
CGREY = "0.35"

# ----------------------- repo utilities (already present) ---------------------
from cc.cartographer.audit import _iter_jsonl, tail_sha

try:
    from cc.cartographer import io  # optional; for ROC toy loader
except Exception:
    io = None  # sentinel

# Bounds + CIs
from cc.cartographer.bounds import fh_intervals, cc_confint, envelope_over_rocs
from cc.analysis.cc_estimation import cc_confint_newcombe

# =============================================================================
# Helpers
# =============================================================================

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
    vals: List[float] = []
    for _, obj in _iter_jsonl(str(history)):
        m = obj.get("metrics", {}) if isinstance(obj, dict) else {}
        vals.append(_to_float(m.get("CC_max"), np.nan))
    return [v for v in vals if np.isfinite(v)]


def _last_record(history: str | Path) -> Optional[Dict[str, Any]]:
    last: Optional[Dict[str, Any]] = None
    for _, obj in _iter_jsonl(str(history)):
        if isinstance(obj, dict):
            last = obj
    return last


def _extract_phase_points(history: str | Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    eps, Ts, ccs = [], [], []
    for _, obj in _iter_jsonl(str(history)):
        if not isinstance(obj, dict):
            continue
        cfg = obj.get("cfg", {}) or {}
        met = obj.get("metrics", {}) or {}
        e = _to_float(cfg.get("epsilon"), np.nan)
        t = _to_float(cfg.get("T"), np.nan)
        c = _to_float(met.get("CC_max"), np.nan)
        if np.isfinite(e) and np.isfinite(t) and np.isfinite(c):
            eps.append(e); Ts.append(t); ccs.append(c)
    return np.array(eps), np.array(Ts), np.array(ccs)


def _auto_lims(x: float, y: float) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    span = max(0.05, 1.2 * max(abs(x), abs(y), 0.02))
    return (-span, span), (-span, span)

# =============================================================================
# Phase / convergence / ROC plots
# =============================================================================

def _plot_phase_surface(E: np.ndarray, T: np.ndarray, C: np.ndarray,
                        last_eps: float, last_T: float, fig_path: Path) -> None:
    from matplotlib.colors import Normalize
    if E.size < 3:
        _plot_phase_point(last_eps, last_T, float(C[-1]) if C.size else 1.0, fig_path, warned=False)
        return

    # best-so-far (global argmax across history)
    arg = int(np.nanargmax(C))
    e_best, t_best, c_best = float(E[arg]), float(T[arg]), float(C[arg])

    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    tri = mtri.Triangulation(E, T)
    norm = Normalize(vmin=max(0.0, float(np.nanmin(C))), vmax=float(np.nanmax(C)))
    cs = ax.tricontourf(tri, C, levels=21, cmap="cividis", norm=norm)
    ax.tricontour(tri, C, levels=[0.95, 1.00, 1.05], colors="k", linewidths=1.0)

    # last run (blue) + best overall (gold star)
    ax.scatter([last_eps], [last_T], s=80, color=CBLUE, edgecolor="white", lw=0.8, zorder=3, label="last run")
    ax.scatter([e_best], [t_best], s=180, marker="*", color="#DAA520", edgecolor="black", lw=0.6, zorder=4,
               label=f"best (CC={c_best:.2f})")

    pad_x = (E.max() - E.min()) * 0.08
    pad_y = (T.max() - T.min()) * 0.08
    ax.set_xlim(E.min() - pad_x, E.max() + pad_x)
    ax.set_ylim(T.min() - pad_y, T.max() + pad_y)

    ax.set_xlabel("epsilon"); ax.set_ylabel("T")
    ax.set_title(r"$CC_{\max}$ phase diagram", pad=6)
    cbar = fig.colorbar(cs, ax=ax); cbar.set_label(r"$CC_{\max}$")
    ax.legend(frameon=False, loc="best")
    _save_opaque(fig, fig_path); plt.close(fig)



def _plot_phase_point(eps: float, T: float, cc: float, fig_path: Path, warned: bool) -> None:
    fig, ax = plt.subplots(figsize=(4.6, 3.6), facecolor="white")
    ax.scatter([eps], [T], s=90, color=CBLUE)
    (x0, x1), (y0, y1) = _auto_lims(eps, T)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    subtitle = "" if not warned else " (cfg missing; defaults used)"
    ax.set_title(f"CC phase point: CC_max={cc:.2f}{subtitle}", pad=6)
    ax.set_xlabel("epsilon"); ax.set_ylabel("T")
    ax.grid(True, alpha=0.25)
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)


def _plot_cc_convergence(cc_series: Sequence[float], fig_path: Path) -> None:
    series = np.asarray(list(cc_series), dtype=float)
    if series.size == 0:
        series = np.array([np.nan])
    x = np.arange(1, series.size + 1)
    fig, ax = plt.subplots(figsize=(5.2, 3.6), facecolor="white")
    ax.plot(x, series, marker="o", lw=1.8, color=CBLUE)
    ax.set_xlabel("trial"); ax.set_ylabel(r"$CC_{\max}$")
    ax.set_title(r"$CC_{\max}$ convergence", pad=6)
    ax.grid(True, alpha=0.25)
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)


def _synthetic_roc() -> Tuple[np.ndarray, np.ndarray]:
    fpr_single = np.array([0.0, 0.10, 0.20, 0.50, 1.0])
    tpr_single = np.array([0.0, 0.50, 0.70, 0.85, 1.0])
    fpr_comp   = np.array([0.0, 0.05, 0.15, 0.40, 1.0])
    tpr_comp   = np.array([0.0, 0.60, 0.80, 0.90, 1.0])
    return np.c_[fpr_single, tpr_single], np.c_[fpr_comp, tpr_comp]


def _plot_roc_from_cfg(rec: Dict[str, Any], fig_path: Path) -> None:
    cfg = rec.get("cfg", {}) if isinstance(rec, dict) else {}
    labelA = str(cfg.get("A", "A"))
    labelB = str(cfg.get("B", "B"))
    try:
        if io is None:
            raise RuntimeError("io.load_scores not available")
        data = io.load_scores(cfg, n=None)  # type: ignore
        rocA = np.asarray(data["rocA"], dtype=float)
        rocB = np.asarray(data["rocB"], dtype=float)
        if rocA.ndim != 2 or rocA.shape[1] != 2 or rocB.ndim != 2 or rocB.shape[1] != 2:
            raise ValueError("Bad ROC shapes from loader")
    except Exception:
        rocA, rocB = _synthetic_roc()

    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.plot(rocA[:, 0], rocA[:, 1], label=labelA, lw=1.8, color=CBLUE)
    ax.plot(rocB[:, 0], rocB[:, 1], label=labelB, lw=1.8, color=CRED)
    ax.plot([0, 1], [0, 1], ls="--", lw=1.2, color="0.6", label="random")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title("ROC comparison (toy)", pad=6)
    ax.legend(frameon=False); ax.grid(True, alpha=0.25)
    _save_opaque(fig, fig_path); plt.close(fig)

# =============================================================================
# Week-3 figures
# =============================================================================

def plot_roc_fh_slice(
    *,
    tpr_a: float, fpr_a: float,
    tpr_b: float, fpr_b: float,
    alpha_cap: Optional[float],
    I1: Tuple[float, float],
    I0: Tuple[float, float],
    D: float,
    n1: int, n0: int,
    delta: float,
    outpath: str,
) -> None:
    """ROC slice with FH context + Bernstein vs Hoeffding tail inset."""
    import math
    fig = plt.figure(figsize=(7.0, 5.2), facecolor="white")
    ax = fig.add_axes([0.10, 0.12, 0.62, 0.80], facecolor="white")

    ax.plot([0, 1], [0, 1], ls="--", lw=1.0, color="0.6")
    ax.scatter([fpr_a, fpr_b], [tpr_a, tpr_b], s=60, marker="o", color=CBLUE)
    ax.text(fpr_a, tpr_a, "  A", va="center", ha="left")
    ax.text(fpr_b, tpr_b, "  B", va="center", ha="left")

    if alpha_cap is not None:
        ax.axvline(alpha_cap, ls=":", lw=1.2, color=CGREY)
        ax.text(alpha_cap, 0.02, " α-cap", rotation=90, va="bottom", ha="right")

    tpr_and = min(tpr_a, tpr_b)
    fpr_and = max(0.0, fpr_a + fpr_b - 1.0)
    tpr_or  = min(1.0, tpr_a + tpr_b)
    fpr_or  = max(fpr_a, fpr_b)
    ax.text(0.02, 0.95, f"AND@θ: TPR≤{tpr_and:.3f}, FPR≥{fpr_and:.3f}", transform=ax.transAxes)
    ax.text(0.02, 0.90, f"OR @θ: TPR≤{tpr_or:.3f},  FPR≥{fpr_or:.3f}",  transform=ax.transAxes)

    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.set_title("ROC slice with FH context at θ")

    ax2 = fig.add_axes([0.75, 0.58, 0.22, 0.32], facecolor="white")
    if I0[0] <= 0.5 <= I0[1]:
        vbar0 = 0.25
    else:
        vbar0 = max(I0[0]*(1-I0[0]), I0[1]*(1-I0[1]))
    eps = [i/1000.0 for i in range(1, 251)]
    bern = [2.0*math.exp(- n0 * (e*e) / (2.0*vbar0 + (2.0/3.0)*e)) for e in eps]
    hoe  = [2.0*math.exp(- 2.0 * n0 * (e*e)) for e in eps]
    ax2.plot(eps, bern, lw=1.2, label="Bernstein (Y=0)", color=CBLUE)
    ax2.plot(eps, hoe,  lw=1.2, label="Hoeffding", color=CRED, alpha=0.9)
    ax2.axhline(delta/2.0, ls=":", lw=1.0, color=CGREY)
    ax2.set_yscale("log"); ax2.set_xlabel("ε"); ax2.set_ylabel("tail prob")
    ax2.set_title("Tail bound (Y=0)"); ax2.legend(fontsize=8, loc="upper right")

    ax.text(0.02, -0.10, f"I1={I1}, I0={I0}, CC=(1-(p1-p0))/D, D={D:.3f}", transform=ax.transAxes)
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_fh_heatmap(
    roc_a: np.ndarray,
    roc_b: np.ndarray,
    *,
    comp: str = "AND",
    add_anchors: bool = True,
    outpath: str,
) -> Dict[str, Any]:
    """FH J-envelope heatmap over two ROC arrays. Returns analysis dict."""
    _, J = envelope_over_rocs(roc_a, roc_b, comp=comp, add_anchors=add_anchors)
    J_max = float(np.max(J)) if J.size else 0.0
    ia, ib = np.unravel_index(int(np.argmax(J)), J.shape)

    fig, ax = plt.subplots(figsize=(7.2, 4.8), facecolor="white")
    im = ax.imshow(J, origin="lower", cmap="cividis", interpolation="nearest", aspect="auto")
    ax.scatter([ib], [ia], marker="x", s=120, c="black", lw=2.0, zorder=3)
    ax.text(ib + 1, ia + 1, f"argmax\nJ={J_max:.3f}", va="center", color="black")
    ax.set_xlabel("Index in ROC_B"); ax.set_ylabel("Index in ROC_A")
    ax.set_title(f"FH Envelope on Composed J [{comp}]")
    cbar = plt.colorbar(im, ax=ax); cbar.set_label("FH ceiling on composed J")
    fig.savefig(outpath, dpi=300); plt.close(fig)

    return {"J_max": J_max, "argmax": (int(ia), int(ib)), "shape": J.shape, "outpath": outpath}


def _plot_cc_ci_figure(
    *, cc_hat: float, bern_lo: float, bern_hi: float, newc_lo: float, newc_hi: float,
    outpath: str, title: str = "CC CIs (FH–Bernstein vs. Newcombe)"
) -> None:
    """Internal: draw the CI comparison figure."""
    fig, ax = plt.subplots(figsize=(10.0, 5.0), facecolor="white")
    ax.set_xlim(0.0, 1.0); ax.set_xlabel("CC"); ax.set_title(title)

    ax.errorbar(x=cc_hat, y=0, xerr=[[cc_hat - bern_lo], [bern_hi - cc_hat]],
                fmt="o", color=CBLUE, capsize=4, lw=2.0, label="Bernstein (FH)")
    ax.errorbar(x=cc_hat, y=1, xerr=[[cc_hat - newc_lo], [newc_hi - cc_hat]],
                fmt="o", color=CRED, capsize=4, lw=2.0, label="Newcombe (Wilson)")
    ax.axvline(cc_hat, ls="--", color=CGREY, lw=1.2)

    txt = (
        f"CĈ={cc_hat:.3f}\n"
        f"Bernstein [{bern_lo:.3f}, {bern_hi:.3f}] (±{(bern_hi-bern_lo)/2:.3f})\n"
        f"Newcombe  [{newc_lo:.3f}, {newc_hi:.3f}] (±{(newc_hi-newc_lo)/2:.3f})"
    )
    ax.text(0.02, 0.05, txt, transform=ax.transAxes)
    ax.set_yticks([0, 1], ["Bernstein (FH)", "Newcombe (Wilson)"])
    ax.grid(True, axis="x", alpha=0.25); ax.legend(frameon=False, loc="lower right")
    fig.savefig(outpath, dpi=300); plt.close(fig)


def plot_cc_ci_comparison(
    *,
    # raw inputs (as used by scripts/make_week3_figs.py)
    p1_hat: float, n1: int, p0_hat: float, n0: int, D: float,
    tpr_a: float, tpr_b: float, fpr_a: float, fpr_b: float,
    alpha_cap: Optional[float] = None,
    delta: float = 0.05,
    outpath: str,
    title: str = "CC CIs (FH–Bernstein vs. Newcombe)",
) -> Dict[str, Any]:
    """
    Compute FH–Bernstein and Newcombe(Wilson) CIs for CC from raw inputs,
    plot the comparison, and return a small summary dict.
    """
    # FH intervals at θ (policy bound applied to I0 via alpha_cap)
    I1, I0 = fh_intervals(tpr_a, tpr_b, fpr_a, fpr_b, alpha_cap=alpha_cap)

    # FH–Bernstein CC CI
    bern_lo, bern_hi = cc_confint(
        n1=n1, n0=n0, p1_hat=p1_hat, p0_hat=p0_hat, D=D, I1=I1, I0=I0, delta=delta
    )

    # Newcombe CC CI (difference of proportions → CC transform)
    x1 = int(round(p1_hat * n1))
    x0 = int(round(p0_hat * n0))
    try:
        newc_lo, newc_hi = cc_confint_newcombe(x1, n1, x0, n0, D, delta=delta)
    except TypeError:
        # for older signature without delta
        newc_lo, newc_hi = cc_confint_newcombe(x1, n1, x0, n0, D)

    cc_hat = (1.0 - (p1_hat - p0_hat)) / float(D)

    _plot_cc_ci_figure(
        cc_hat=cc_hat, bern_lo=float(bern_lo), bern_hi=float(bern_hi),
        newc_lo=float(newc_lo), newc_hi=float(newc_hi),
        outpath=outpath, title=title,
    )

    return {
        "cc_hat": float(cc_hat),
        "bernstein": (float(bern_lo), float(bern_hi)),
        "newcombe": (float(newc_lo), float(newc_hi)),
        "I1": I1, "I0": I0,
        "outpath": outpath,
    }

# =============================================================================
# CSV summary + CLI
# =============================================================================

def _bootstrap_ci(
    data: Sequence[float],
    alpha: float = 0.05,
    B: int = 10_000,
    random_state: int = 1337,
) -> Optional[Tuple[float, float]]:
    arr = np.asarray([x for x in data if np.isfinite(x)], dtype=float)
    if arr.size < 3:
        return None
    rng = np.random.default_rng(random_state)
    idx = rng.integers(0, arr.size, size=(B, arr.size))
    boot = arr[idx].mean(axis=1)
    lo = float(np.quantile(boot, alpha / 2.0))
    hi = float(np.quantile(boot, 1.0 - alpha / 2.0))
    return (lo, hi)


def _write_summary_csv(
    out_dir: Path,
    cc_series: Sequence[float],
    history: str | Path,
    rec: Dict[str, Any],
) -> Path:
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

    mirror = Path("results/aggregates")
    mirror.mkdir(parents=True, exist_ok=True)
    mirror_file = mirror / "summary.csv"
    mirror_file.write_bytes(out_file.read_bytes())

    return out_file


def _add_week3_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--week3-figure", action="store_true",
                    help="Also produce Week-3 ROC slice figure with FH context.")
    ap.add_argument("--week3-theta", type=str, default=None,
                    help="θ as 'tpr_a,fpr_a,tpr_b,fpr_b'")
    ap.add_argument("--week3-I1", type=str, default=None,
                    help="I1 interval as 'lo,hi'")
    ap.add_argument("--week3-I0", type=str, default=None,
                    help="I0 interval as 'lo,hi'")
    ap.add_argument("--week3-D", type=float, default=None)
    ap.add_argument("--week3-n1", type=int, default=None)
    ap.add_argument("--week3-n0", type=int, default=None)
    ap.add_argument("--week3-delta", type=float, default=0.05)
    ap.add_argument("--week3-alpha-cap", type=float, default=None)
    ap.add_argument("--fh-heatmap", action="store_true")
    ap.add_argument("--ci-comparison", action="store_true")


def _parse_pair(txt: Optional[str]) -> Optional[Tuple[float, float]]:
    if not txt:
        return None
    parts = [p.strip() for p in str(txt).split(",")]
    if len(parts) != 2:
        raise SystemExit(f"Bad pair (expected 'lo,hi'): {txt}")
    return float(parts[0]), float(parts[1])


def _parse_quad(txt: Optional[str]) -> Optional[Tuple[float, float, float, float]]:
    if not txt:
        return None
    parts = [p.strip() for p in str(txt).split(",")]
    if len(parts) != 4:
        raise SystemExit(f"Bad quad (expected 'tpr_a,fpr_a,tpr_b,fpr_b'): {txt}")
    return float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])


def main(argv: Optional[Iterable[str]] = None) -> None:
    ap = argparse.ArgumentParser(
        description="Generate figures and summary.csv from audit history with Week-3 extensions."
    )
    ap.add_argument("--history", required=True, help="Path to audit JSONL (hash-chained).")
    ap.add_argument("--fig-dir", required=True, help="Directory to write PDFs (figures).")
    ap.add_argument("--out-dir", required=True, help="Directory to write CSV summary.")
    _add_week3_args(ap)
    args = ap.parse_args(argv)

    history = Path(args.history)
    fig_dir = _ensure_dir(args.fig_dir)
    out_dir = _ensure_dir(args.out_dir)

    rec = _last_record(history)
    if rec is None:
        raise SystemExit(f"No records found in {history}. Run your experiment first.")

    cc_series = _read_cc_series(history)
    E, T, C = _extract_phase_points(history)
    last_cfg = rec.get("cfg", {}) if isinstance(rec, dict) else {}
    last_eps = _to_float(last_cfg.get("epsilon"), 0.0)
    last_T   = _to_float(last_cfg.get("T"), 0.0)
    last_cc  = _to_float(rec.get("metrics", {}).get("CC_max"), 1.0)

    if E.size >= 10:
        _plot_phase_surface(E, T, C, last_eps, last_T, fig_dir / "phase_diagram.pdf")
    else:
        warned = (last_eps == 0.0 and last_T == 0.0)
        _plot_phase_point(last_eps, last_T, last_cc, fig_dir / "phase_diagram.pdf", warned=warned)

    _plot_cc_convergence(cc_series, fig_dir / "cc_convergence.pdf")
    _plot_roc_from_cfg(rec, fig_dir / "roc_comparison.pdf")

    out_csv = _write_summary_csv(out_dir, cc_series, history, rec)
    print(f"✓ Wrote figures → {fig_dir.resolve()}")
    print(f"✓ Wrote summary → {out_csv.resolve()}")

    if args.week3_figure:
        theta = _parse_quad(args.week3_theta)
        I1 = _parse_pair(args.week3_I1)
        I0 = _parse_pair(args.week3_I0)
        D  = args.week3_D
        n1 = args.week3_n1
        n0 = args.week3_n0
        delta = float(args.week3_delta) if args.week3_delta is not None else 0.05
        a_cap = args.week3_alpha_cap
        if not (theta and I1 and I0 and D and n1 and n0):
            raise SystemExit("Provide --week3-theta --week3-I1 --week3-I0 --week3-D --week3-n1 --week3-n0")
        tpr_a, fpr_a, tpr_b, fpr_b = theta
        out_png = fig_dir / "fig_week3_roc_fh.png"
        plot_roc_fh_slice(
            tpr_a=tpr_a, fpr_a=fpr_a, tpr_b=tpr_b, fpr_b=fpr_b,
            alpha_cap=a_cap, I1=I1, I0=I0, D=float(D), n1=int(n1), n0=int(n0),
            delta=float(delta), outpath=str(out_png),
        )
        print(f"✓ Wrote Week-3 figure → {out_png.resolve()}")

    if args.fh_heatmap:
        try:
            cfg2 = rec.get("cfg", {}) if isinstance(rec, dict) else {}
            if io is None:
                rocA, rocB = _synthetic_roc()
            else:
                data = io.load_scores(cfg2, n=None)  # type: ignore
                rocA = np.asarray(data["rocA"], dtype=float)
                rocB = np.asarray(data["rocB"], dtype=float)
            heatmap_path = fig_dir / "fh_heatmap.png"
            result = plot_fh_heatmap(rocA, rocB, comp="AND", add_anchors=True, outpath=str(heatmap_path))
            print(f"✓ Wrote FH heatmap → {heatmap_path.resolve()}  (J_max={result['J_max']:.3f} at {result['argmax']})")
        except Exception as e:
            print(f"⚠ Could not generate FH heatmap: {e}")

    if args.ci_comparison:
        try:
            met = rec.get("metrics", {}) if isinstance(rec, dict) else {}
            cfg2 = rec.get("cfg", {}) if isinstance(rec, dict) else {}
            p1_hat = _to_float(met.get("p1_hat"), 0.62)
            p0_hat = _to_float(met.get("p0_hat"), 0.09)
            n1 = int(_to_float(cfg2.get("n1"), 200))
            n0 = int(_to_float(cfg2.get("n0"), 200))
            D  = _to_float(cfg2.get("D"), 0.55)
            tpr_a = _to_float(cfg2.get("tpr_a"), 0.72)
            tpr_b = _to_float(cfg2.get("tpr_b"), 0.65)
            fpr_a = _to_float(cfg2.get("fpr_a"), 0.035)
            fpr_b = _to_float(cfg2.get("fpr_b"), 0.05)

            ci_path = fig_dir / "cc_ci_comparison.png"
            out = plot_cc_ci_comparison(
                p1_hat=p1_hat, n1=n1, p0_hat=p0_hat, n0=n0, D=D,
                tpr_a=tpr_a, tpr_b=tpr_b, fpr_a=fpr_a, fpr_b=fpr_b,
                alpha_cap=None, delta=0.05, outpath=str(ci_path)
            )
            print(f"✓ Wrote CC CI comparison → {ci_path.resolve()}  (CĈ={out['cc_hat']:.3f})")
        except Exception as e:
            print(f"⚠ Could not generate CI comparison: {e}")


if __name__ == "__main__":
    main()
