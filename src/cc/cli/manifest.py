# src/cc/cli/manifest.py
"""CLI helpers for querying run manifests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from cc.cartographer import audit as audit_chain


def _load_manifest(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _manifest_paths(manifest_dir: Path) -> List[Path]:
    if not manifest_dir.exists():
        return []
    return sorted(p for p in manifest_dir.glob("*.json") if not p.name.endswith(".jsonl"))


def _chain_head_for(run_id: str, manifest_dir: Path) -> Optional[str]:
    chain_path = manifest_dir / f"{run_id}.jsonl"
    if not chain_path.exists():
        return None
    return audit_chain.tail_sha(str(chain_path))


def _manifest_entry(path: Path, manifest_dir: Path) -> Dict[str, Any]:
    payload = _load_manifest(path)
    run_id = payload.get("run_id") or path.stem
    chain_path = manifest_dir / f"{run_id}.jsonl"
    return {
        "run_id": run_id,
        "manifest_path": str(path),
        "chain_path": str(chain_path),
        "chain_head_sha256": _chain_head_for(run_id, manifest_dir),
    }


def _resolve_manifest(
    *,
    manifest_dir: Path,
    run_id: Optional[str],
    chain_hash: Optional[str],
) -> Tuple[Optional[Path], Optional[Dict[str, Any]]]:
    if run_id:
        candidate = manifest_dir / f"{run_id}.json"
        if candidate.exists():
            return candidate, _manifest_entry(candidate, manifest_dir)
        return None, None

    for path in _manifest_paths(manifest_dir):
        payload = _load_manifest(path)
        candidate_run_id = payload.get("run_id") or path.stem
        if chain_hash and _chain_head_for(candidate_run_id, manifest_dir) == chain_hash:
            return path, _manifest_entry(path, manifest_dir)
    return None, None


def _cmd_list(args: argparse.Namespace) -> int:
    manifest_dir = Path(args.dir)
    entries = [_manifest_entry(path, manifest_dir) for path in _manifest_paths(manifest_dir)]
    print(json.dumps(entries, indent=2, sort_keys=True))
    return 0


def _cmd_show(args: argparse.Namespace) -> int:
    manifest_dir = Path(args.dir)
    path, entry = _resolve_manifest(
        manifest_dir=manifest_dir,
        run_id=args.run_id,
        chain_hash=args.hash,
    )
    if not path or not entry:
        print("Manifest not found.", file=sys.stderr)
        return 1

    payload = _load_manifest(path)
    output = {
        "manifest": payload,
        "lineage": entry,
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Query run manifests.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List available run manifests.")
    list_parser.add_argument("--dir", default="runs/manifest", help="Manifest directory.")
    list_parser.set_defaults(func=_cmd_list)

    show_parser = subparsers.add_parser("show", help="Show a manifest by run ID or hash.")
    show_parser.add_argument("--dir", default="runs/manifest", help="Manifest directory.")
    show_parser.add_argument("--run-id", help="Run ID to display.")
    show_parser.add_argument("--hash", help="Chain head hash to search for.")
    show_parser.set_defaults(func=_cmd_show)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "show" and not (args.run_id or args.hash):
        parser.error("show requires --run-id or --hash")
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
