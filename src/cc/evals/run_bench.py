# src/cc/evals/run_bench.py
"""Benchmark runner for real guardrail adapters."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from cc import __version__
from cc.adapters import ADAPTER_REGISTRY, Decision, GuardrailAdapter
from cc.cartographer.audit import append_jsonl
from cc.core.manifest import RunManifest, emit_run_manifest
from cc.core.metrics import (
    cc_max,
    cc_rel,
    delta_add,
    delta_mult,
    confusion_from_labels,
    rates_from_confusion,
    youden_j,
)
from cc.io.seeds import set_seed


@dataclass(frozen=True)
class BenchConfig:
    dataset: str
    adapters: Sequence[str]
    composition: str
    composition_parameter: Optional[float]
    prompt_field: str
    response_field: Optional[str]
    label_field: str
    review_policy: str
    seed: int


def main(argv: Optional[Iterable[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Run CC guardrail adapter benchmark.")
    ap.add_argument("--dataset", required=True, help="Dataset path (CSV or JSONL).")
    ap.add_argument("--adapters", required=True, help="Comma-separated adapter names.")
    ap.add_argument("--composition", default="OR", choices=["AND", "OR", "COND_OR"])
    ap.add_argument("--composition-parameter", type=float, default=None)
    ap.add_argument("--adapter-config", help="Optional JSON config for adapters.")
    ap.add_argument("--prompt-field", default="prompt")
    ap.add_argument("--response-field", default=None)
    ap.add_argument("--label-field", default="label")
    ap.add_argument("--review-policy", choices=["block", "allow"], default="block")
    ap.add_argument("--out", default="runs/bench.jsonl", help="JSONL output path.")
    ap.add_argument("--audit-out", default=None, help="Optional JSONL audit log path.")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args(argv)

    seed = set_seed(args.seed)
    adapters, adapter_configs = _init_adapters(args.adapters, args.adapter_config)
    dataset = load_dataset(Path(args.dataset))

    adapter_config_hashes = {
        name: _hash_adapter_config(cfg) for name, cfg in adapter_configs.items()
    }
    cfg = BenchConfig(
        dataset=str(args.dataset),
        adapters=[a.name for a in adapters],
        composition=args.composition,
        composition_parameter=args.composition_parameter,
        prompt_field=args.prompt_field,
        response_field=args.response_field,
        label_field=args.label_field,
        review_policy=args.review_policy,
        seed=seed,
    )
    config_hash = _hash_config(cfg)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_meta = {
        "run_id": run_id,
        "config": asdict(cfg),
        "config_hash": config_hash,
        "code_version": __version__,
        "git_sha": _git_sha(),
        "adapter_versions": {a.name: a.version for a in adapters},
        "adapter_config_hashes": adapter_config_hashes,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    run_manifest = RunManifest(
        run_id=run_id,
        config_hashes={
            "bench_config_sha256": config_hash,
            **{f"adapter_config_sha256:{name}": h for name, h in adapter_config_hashes.items()},
        },
        dataset_ids=[str(args.dataset)],
        guardrail_versions={a.name: a.version for a in adapters},
        git_sha=run_meta["git_sha"],
    )
    emit_run_manifest(run_manifest)
    audit_out_path = Path(args.audit_out) if args.audit_out else Path(args.out).with_suffix(".audit.jsonl")

    results = run_benchmark(
        dataset=dataset,
        adapters=adapters,
        composition=args.composition,
        prompt_field=args.prompt_field,
        response_field=args.response_field,
        label_field=args.label_field,
        review_policy=args.review_policy,
        run_meta=run_meta,
        out_path=Path(args.out),
        audit_out_path=audit_out_path,
    )
    print(json.dumps(results["summary"], indent=2))


def _hash_config(cfg: BenchConfig) -> str:
    payload = json.dumps(asdict(cfg), sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _hash_adapter_config(config: Dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _git_sha() -> Optional[str]:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode("utf-8")
            .strip()
        )
    except Exception:
        return None


def _init_adapters(
    adapters_arg: str,
    config_path: Optional[str],
) -> Tuple[List[GuardrailAdapter], Dict[str, Dict[str, Any]]]:
    config = {}
    if config_path:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    adapters = []
    adapter_configs: Dict[str, Dict[str, Any]] = {}
    for name in [a.strip() for a in adapters_arg.split(",") if a.strip()]:
        cls = ADAPTER_REGISTRY.get(name)
        if cls is None:
            raise ValueError(f"Unknown adapter: {name}")
        kwargs = config.get(name, {})
        adapter_configs[name] = dict(kwargs)
        adapters.append(cls(**kwargs))
    return adapters, adapter_configs


def load_dataset(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path).to_dict(orient="records")
    if path.suffix.lower() in {".jsonl", ".json"}:
        records = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        return records
    raise ValueError(f"Unsupported dataset format: {path}")


def run_benchmark(
    dataset: Sequence[Dict[str, Any]],
    adapters: Sequence[GuardrailAdapter],
    composition: str,
    prompt_field: str,
    response_field: Optional[str],
    label_field: str,
    review_policy: str,
    run_meta: Dict[str, Any],
    out_path: Path,
    audit_out_path: Optional[Path],
) -> Dict[str, Any]:
    y_true: List[int] = []
    per_adapter_preds: Dict[str, List[int]] = {a.name: [] for a in adapters}
    composed_preds: List[int] = []

    for idx, item in enumerate(dataset):
        prompt = str(item.get(prompt_field, ""))
        response = str(item.get(response_field)) if response_field and item.get(response_field) is not None else None
        label = _normalize_label(item.get(label_field))
        y_true.append(label)

        decisions: Dict[str, Decision] = {}
        blocked_flags: List[bool] = []
        for adapter in adapters:
            decision = adapter.check(prompt, response, metadata=item)
            decisions[adapter.name] = decision
            blocked = _decision_to_blocked(decision, review_policy)
            per_adapter_preds[adapter.name].append(int(blocked))
            blocked_flags.append(blocked)
            if audit_out_path and decision.audit:
                append_jsonl(
                    str(audit_out_path),
                    {
                        "record_type": "guardrail_adapter_audit",
                        "index": idx,
                        "adapter": adapter.name,
                        "audit": decision.audit,
                        "run_meta": run_meta,
                    },
                )

        composed = _compose(blocked_flags, composition)
        composed_preds.append(int(composed))

        record = {
            "record_type": "item",
            "index": idx,
            "prompt": prompt,
            "response": response,
            "label": label,
            "decisions": {k: asdict(v) for k, v in decisions.items()},
            "composed": composed,
            "composition": composition,
            "run_meta": run_meta,
        }
        append_jsonl(str(out_path), record)

    summary = _summarize(y_true, per_adapter_preds, composed_preds, composition)
    append_jsonl(
        str(out_path),
        {"record_type": "summary", "summary": summary, "run_meta": run_meta},
    )
    return {"summary": summary}


def _normalize_label(label: Any) -> int:
    if isinstance(label, (int, float)):
        return int(label)
    if isinstance(label, str):
        lowered = label.strip().lower()
        if lowered in {"1", "unsafe", "harmful", "bad"}:
            return 1
        if lowered in {"0", "safe", "benign", "good"}:
            return 0
    raise ValueError(f"Unsupported label value: {label!r}")


def _decision_to_blocked(decision: Decision, review_policy: str) -> bool:
    if decision.verdict == "block":
        return True
    if decision.verdict == "allow":
        return False
    return review_policy == "block"


def _compose(flags: Sequence[bool], composition: str) -> bool:
    if not flags:
        return False
    if composition == "AND":
        return all(flags)
    if composition == "OR":
        return any(flags)
    if composition == "COND_OR":
        blocked = False
        for flag in flags:
            blocked = blocked or flag
            if blocked:
                return True
        return False
    raise ValueError(f"Unknown composition rule: {composition}")


def _summarize(
    y_true: Sequence[int],
    per_adapter_preds: Dict[str, Sequence[int]],
    composed_preds: Sequence[int],
    composition: str,
) -> Dict[str, Any]:
    adapter_metrics: Dict[str, Any] = {}
    for name, preds in per_adapter_preds.items():
        conf = confusion_from_labels(y_true, preds, pos_label=1)
        rates = rates_from_confusion(conf)
        adapter_metrics[name] = {
            "confusion": conf.__dict__,
            "rates": rates.__dict__,
            "j": youden_j(rates.tpr, rates.fpr),
        }

    comp_conf = confusion_from_labels(y_true, composed_preds, pos_label=1)
    comp_rates = rates_from_confusion(comp_conf)
    comp_j = youden_j(comp_rates.tpr, comp_rates.fpr)

    j_values = [m["j"] for m in adapter_metrics.values()] or [0.0]
    best_j = max(j_values)
    metrics = {
        "composition": composition,
        "composed": {"confusion": comp_conf.__dict__, "rates": comp_rates.__dict__, "j": comp_j},
        "cc_max": cc_max(comp_j, best_j, best_j) if best_j > 0 else 0.0,
    }
    if len(j_values) >= 2:
        j_a, j_b = j_values[:2]
        metrics.update(
            {
                "delta_add": delta_add(comp_j, j_a, j_b),
                "cc_rel": cc_rel(comp_j, j_a, j_b),
                "delta_mult": delta_mult(comp_j, j_a, j_b),
            }
        )
    metrics["adapter_metrics"] = adapter_metrics
    metrics["disagreement_rates"] = _disagreement_rates(per_adapter_preds)
    return metrics


def _disagreement_rates(per_adapter_preds: Dict[str, Sequence[int]]) -> Dict[str, float]:
    names = list(per_adapter_preds.keys())
    rates: Dict[str, float] = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            preds_a = per_adapter_preds[a]
            preds_b = per_adapter_preds[b]
            if not preds_a:
                rates[f"{a}__{b}"] = 0.0
                continue
            diff = sum(int(pa != pb) for pa, pb in zip(preds_a, preds_b))
            rates[f"{a}__{b}"] = diff / len(preds_a)
    return rates


if __name__ == "__main__":  # pragma: no cover
    main()
