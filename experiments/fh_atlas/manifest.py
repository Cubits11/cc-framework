from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import hashlib
import json


@dataclass
class ManifestEntry:
    path: str
    sha256: str
    size_bytes: int
    description: str


def hash_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def build_manifest(
    run_id: str,
    output_dir: Path,
    config_hash: str,
    entries: List[ManifestEntry],
) -> Dict[str, object]:
    manifest = {
        "run_id": run_id,
        "output_dir": str(output_dir),
        "config_hash": config_hash,
        "artifacts": [entry.__dict__ for entry in entries],
    }
    return manifest


def write_manifest(path: Path, manifest: Dict[str, object]) -> None:
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
