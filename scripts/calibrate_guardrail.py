"""Calibration helper to tune toy guardrail thresholds to FPR ≈ 0.05."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import yaml

from cc.core.logging import ChainedJSONLLogger
from cc.guardrails.toy_threshold import ToyThresholdGuardrail


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_config(cfg: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def load_benign_texts(dataset: Path, synthetic_vocab: List[str], harmful_vocab: List[str], n_synthetic: int = 200) -> List[str]:
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


def calibrate_guardrail_entry(entry: Dict[str, Any], cfg: Dict[str, Any], benign_texts: List[str]) -> Dict[str, Any]:
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
            "range_lo": 0.04,
            "range_hi": 0.06,
        }

    scores_sorted = sorted(scores, reverse=True)
    desired_lo = max(0.0, min(alpha_cap, 0.04))
    desired_hi = max(desired_lo, min(alpha_cap, 0.06))

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

    if final_fpr > alpha_cap:
        guardrail.set_threshold(min(1.0, chosen_threshold + 1e-3))
        chosen_threshold = guardrail.get_threshold()
        final_fpr = guardrail.false_positive_rate(benign_texts)

    params["threshold"] = chosen_threshold
    return {
        "name": entry.get("name"),
        "threshold": chosen_threshold,
        "fpr": final_fpr,
        "n_texts": n,
        "target": target,
        "range_lo": desired_lo,
        "range_hi": desired_hi,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate toy guardrail thresholds")
    parser.add_argument("--config", required=True, help="Path to Week 5 YAML config")
    parser.add_argument("--dataset", default="datasets/benign", help="Directory/file with benign texts")
    parser.add_argument("--summary", default="results/week5_scan/calibration_summary.json", help="Write calibration summary JSON here")
    parser.add_argument("--audit", default="runs/audit_week5.jsonl", help="Append calibration event to this JSONL chain")
    parser.add_argument("--output-config", help="Optional path to write calibrated config copy")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    dataset_path = Path(args.dataset)
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = load_config(cfg_path)
    attacker = cfg.get("attacker", {}) or {}
    params = attacker.get("params", {})
    benign_vocab = params.get("vocab_benign", [])
    harmful_vocab = params.get("vocab_harmful", [])
    benign_texts = load_benign_texts(dataset_path, benign_vocab, harmful_vocab)

    guardrail_summaries = []
    for entry in cfg.get("guardrails", []) or []:
        if entry.get("name") != "toy_threshold":
            continue
        summary = calibrate_guardrail_entry(entry, cfg, benign_texts)
        guardrail_summaries.append(summary)
        print(
            f"Calibrated {entry.get('name')} → threshold={summary['threshold']:.4f}, "
            f"FPR={summary['fpr']:.4f} on {summary['n_texts']} benign prompts"
        )

    summary_payload = {
        "timestamp": time.time(),
        "config": str(cfg_path),
        "dataset": str(dataset_path),
        "alpha_cap": cfg.get("alpha_cap", 0.05),
        "target_window": [0.04, 0.06],
        "guardrails": guardrail_summaries,
    }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    if args.output_config:
        save_config(cfg, Path(args.output_config))

    logger = ChainedJSONLLogger(str(args.audit))
    logger.log({
        "event": "guardrail_calibration",
        "config": str(cfg_path),
        "dataset": str(dataset_path),
        "summary": summary_payload,
    })
 
 
if __name__ == "__main__":
    main()
