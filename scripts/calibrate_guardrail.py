#!/usr/bin/env python
"""
Calibration helper to tune toy guardrail thresholds to FPR ≈ 0.05.

Week-6 upgrades:
- --write-inplace and --write-config-out (mutually exclusive) to persist calibrated threshold
- --window-lo/--window-hi to enforce α-window (default [0.04, 0.06]); non-zero exit if violated
- Always write a flat calibration_summary.json with:
  {name, threshold, fpr, n_texts, alpha_cap, target_window, stack_fpr, timestamp}
- Print exactly one concise line with calibration outcome
- Only modify guardrails[0].params.threshold (others preserved)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

from cc.core.guardrail_api import GuardrailAdapter
from cc.core.logging import ChainedJSONLLogger
from cc.core.registry import build_guardrails
from cc.guardrails.toy_threshold import ToyThresholdGuardrail

# --------------------------
# YAML IO helpers
# --------------------------


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_config(cfg: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def update_threshold_in_config(path_in: Path, path_out: Path, thr: float) -> None:
    """
    Safely load YAML, update guardrails[0].params.threshold, and write to path_out.
    All other fields are preserved verbatim.
    """
    cfg = load_config(path_in)
    guardrails = cfg.get("guardrails")
    if not guardrails or not isinstance(guardrails, list):
        raise ValueError("Config must contain a non-empty 'guardrails' list.")
    g0 = guardrails[0] or {}
    params = dict(g0.get("params") or {})
    params["threshold"] = float(thr)
    g0["params"] = params
    guardrails[0] = g0
    cfg["guardrails"] = guardrails
    save_config(cfg, path_out)


# --------------------------
# Data + calibration helpers
# --------------------------


def load_benign_texts(
    dataset: Path, synthetic_vocab: List[str], harmful_vocab: List[str], n_synthetic: int = 200
) -> List[str]:
    texts: List[str] = []
    if dataset.is_file():
        texts.extend([line.strip() for line in dataset.read_text().splitlines() if line.strip()])
    elif dataset.is_dir():
        for file in sorted(dataset.rglob("*.txt")):
            texts.extend([line.strip() for line in file.read_text().splitlines() if line.strip()])
    rng = np.random.default_rng(13)
    vocab = [tok for tok in synthetic_vocab if tok]
    if not vocab:
        vocab = ["hello", "thanks", "please", "info"]
    harmful = [tok for tok in harmful_vocab if tok]
    for _ in range(max(0, n_synthetic - len(texts))):
        k = int(rng.integers(5, 9))
        tokens = rng.choice(vocab, size=k, replace=True)
        if harmful and rng.random() < 0.05:
            insert_at = int(rng.integers(0, k))
            tokens[insert_at] = rng.choice(harmful)
        texts.append(" ".join(tokens))
    return texts


def calibrate_guardrail_entry(
    entry: Dict[str, Any],
    cfg: Dict[str, Any],
    benign_texts: List[str],
    window_lo: float = 0.04,
    window_hi: float = 0.06,
) -> Dict[str, Any]:
    """
    Calibrate a ToyThresholdGuardrail to target FPR within [window_lo, window_hi] and
    <= alpha_cap if present.

    Returns dict with keys:
      name, threshold, fpr, n_texts, target, range_lo, range_hi
    """
    params = entry.setdefault("params", {})
    keywords = params.get("keywords", [])
    threshold = float(params.get("threshold", 0.2))
    alpha_cap = float(params.get("alpha_cap", cfg.get("alpha_cap", 0.05)))
    target = float(entry.get("fpr_target", cfg.get("alpha_cap", 0.05)))
    target = min(max(target, 0.0), alpha_cap)

    guardrail = ToyThresholdGuardrail(keywords=keywords, threshold=threshold, alpha_cap=alpha_cap)
    scores = [guardrail.score(text) for text in benign_texts]
    n = len(scores)
    if n == 0:
        return {
            "name": entry.get("name"),
            "threshold": guardrail.get_threshold(),
            "fpr": 0.0,
            "n_texts": 0,
            "target": target,
            "range_lo": float(window_lo),
            "range_hi": float(window_hi),
        }

    scores_sorted = sorted(scores, reverse=True)
    desired_lo = max(0.0, min(alpha_cap, float(window_lo)))
    desired_hi = max(desired_lo, min(alpha_cap, float(window_hi)))

    min_k = int(np.ceil(desired_lo * n))
    max_k = int(np.floor(desired_hi * n))
    if max_k < min_k:
        max_k = min_k
    candidate_k = list(range(min_k, max_k + 1))
    if not candidate_k:
        candidate_k = [int(round(target * n))]
    candidate_k = [int(np.clip(k, 0, n)) for k in candidate_k]

    best = None
    for k in candidate_k:
        if k <= 0:
            thr = float(scores_sorted[0]) + 1e-6
        elif k >= n:
            thr = float(scores_sorted[-1]) - 1e-6
        else:
            high = float(scores_sorted[k - 1])
            low = float(scores_sorted[k])
            thr = float(np.clip((high + low) / 2.0, 0.0, 1.0))
        guardrail.set_threshold(thr)
        fpr = guardrail.false_positive_rate(benign_texts)
        diff = abs(fpr - target)
        in_window = desired_lo <= fpr <= desired_hi
        if best is None or (in_window and not best[0]) or (in_window == best[0] and diff < best[2]):
            best = (in_window, thr, diff, fpr)

    if best is None:
        chosen_threshold = guardrail.get_threshold()
        final_fpr = guardrail.false_positive_rate(benign_texts)
    else:
        guardrail.set_threshold(best[1])
        chosen_threshold = guardrail.get_threshold()
        final_fpr = best[3]

    # Safety: cap at alpha_cap if somehow exceeded
    if final_fpr > alpha_cap:
        guardrail.set_threshold(min(1.0, chosen_threshold + 1e-3))
        chosen_threshold = guardrail.get_threshold()
        final_fpr = guardrail.false_positive_rate(benign_texts)

    params["threshold"] = chosen_threshold  # reflect back into entry
    return {
        "name": entry.get("name"),
        "threshold": chosen_threshold,
        "fpr": final_fpr,
        "n_texts": n,
        "target": target,
        "range_lo": desired_lo,
        "range_hi": desired_hi,
    }


def estimate_stack_fpr(cfg: Dict[str, Any], benign_texts: List[str]) -> Optional[float]:
    guardrail_cfgs = cfg.get("guardrails") or []
    if not guardrail_cfgs or not benign_texts:
        return None
    try:
        stack = build_guardrails(guardrail_cfgs)
    except Exception:
        return None
    adapters = [GuardrailAdapter(guardrail) for guardrail in stack]
    blocked = 0
    for text in benign_texts:
        try:
            for adapter in adapters:
                is_blocked, _score = adapter.evaluate(text)
                if is_blocked:
                    blocked += 1
                    break
        except Exception:
            return None
    return blocked / float(len(benign_texts))


# --------------------------
# CLI
# --------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate toy guardrail thresholds (Week-6).")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    parser.add_argument(
        "--dataset",
        default="datasets/benign",
        help="Directory/file with benign texts used for FPR calibration.",
    )
    parser.add_argument(
        "--summary", required=True, help="Path to write calibration_summary.json (flat object)."
    )
    parser.add_argument(
        "--audit",
        default="runs/audit_week6.jsonl",
        help="Append calibration event to this JSONL chain.",
    )
    # Week-6 write-back options (mutually exclusive)
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--write-inplace",
        action="store_true",
        help="Update the provided config file in-place with calibrated threshold.",
    )
    group.add_argument(
        "--write-config-out",
        default=None,
        help="Write a derived YAML config with the calibrated threshold.",
    )
    # α-window
    parser.add_argument(
        "--window-lo", type=float, default=0.04, help="Lower bound for α-window (inclusive)."
    )
    parser.add_argument(
        "--window-hi", type=float, default=0.06, help="Upper bound for α-window (inclusive)."
    )
    return parser.parse_args()


def _pick_entry_for_calibration(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Choose which guardrail entry to calibrate.
    Preference: name == 'toy_threshold' else guardrails[0].
    """
    guardrails = cfg.get("guardrails") or []
    if not guardrails:
        raise ValueError("Config missing 'guardrails'.")
    # Prefer explicit toy_threshold
    for e in guardrails:
        if (e or {}).get("name") == "toy_threshold":
            return e
    # Fallback to first entry
    return guardrails[0]


def main() -> None:
    args = parse_args()

    cfg_path = Path(args.config)
    dataset_path = Path(args.dataset)
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = load_config(cfg_path)

    # Prepare benign corpus
    attacker = cfg.get("attacker", {}) or {}
    params = attacker.get("params", {})
    benign_vocab = params.get("vocab_benign", [])
    harmful_vocab = params.get("vocab_harmful", [])
    benign_texts = load_benign_texts(dataset_path, benign_vocab, harmful_vocab)

    # Choose the entry to calibrate
    entry = _pick_entry_for_calibration(cfg)
    name = entry.get("name", "toy_threshold")

    # Calibrate
    result = calibrate_guardrail_entry(
        entry, cfg, benign_texts, window_lo=float(args.window_lo), window_hi=float(args.window_hi)
    )
    alpha_cap = float(cfg.get("alpha_cap", 0.05))
    lo = float(args.window_lo)
    hi = float(args.window_hi)

    stack_fpr = estimate_stack_fpr(cfg, benign_texts)

    # Persist flat calibration summary
    flat_summary = {
        "name": name,
        "threshold": float(result["threshold"]),
        "fpr": float(result["fpr"]),
        "n_texts": int(result["n_texts"]),
        "alpha_cap": alpha_cap,
        "target_window": [lo, hi],
        "stack_fpr": None if stack_fpr is None else float(stack_fpr),
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(flat_summary, f, indent=2)

    # Write-back logic
    derived_out: Optional[Path] = None
    if args.write_inplace:
        tmp_out = cfg_path.with_suffix(cfg_path.suffix + ".tmp")
        update_threshold_in_config(cfg_path, tmp_out, flat_summary["threshold"])
        # Atomic-ish replace
        tmp_out.replace(cfg_path)
        derived_out = cfg_path
    elif args.write_config_out:
        derived_out = Path(args.write_config_out)
        derived_out.parent.mkdir(parents=True, exist_ok=True)
        update_threshold_in_config(cfg_path, derived_out, flat_summary["threshold"])

    # Print single concise line
    print(
        f"Calibrated {flat_summary['name']} \u2192 threshold={flat_summary['threshold']:.6f}, "
        f"FPR={flat_summary['fpr']:.4f} on {flat_summary['n_texts']} benign "
        f"(α∈[{lo:.2f},{hi:.2f}])."
    )

    # α-window enforcement
    in_window = lo <= flat_summary["fpr"] <= hi
    exit_code = 0 if in_window else 1
    if not in_window:
        print(
            f"ERROR: Measured FPR={flat_summary['fpr']:.4f} outside target window [{lo:.2f},{hi:.2f}].",
            file=sys.stderr,
        )

    # Audit (best-effort, never overrides exit code)
    try:
        logger = ChainedJSONLLogger(str(args.audit))
        logger.log(
            {
                "event": "guardrail_calibration",
                "name": flat_summary["name"],
                "threshold": flat_summary["threshold"],
                "fpr": flat_summary["fpr"],
                "n_texts": flat_summary["n_texts"],
                "alpha_cap": flat_summary["alpha_cap"],
                "window": [lo, hi],
                "config_in": str(cfg_path),
                "config_out": str(derived_out) if derived_out else None,
                "dataset": str(dataset_path),
                "ts": int(time.time()),
            }
        )
    except Exception:
        pass

    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
