"""Storage backends for experiment artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


def _stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _hash_text(text: str) -> str:
    return _sha256_hex(text.encode("utf-8"))


def _hash_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _hash_directory(path: Path) -> str:
    entries = []
    for file_path in sorted(p for p in path.rglob("*") if p.is_file()):
        entries.append({"path": str(file_path.relative_to(path)), "sha256": _hash_file(file_path)})
    return _hash_text(_stable_json(entries))


def _iter_dataset_candidates(config: Mapping[str, Any]) -> Iterable[str]:
    keys = {
        "dataset",
        "datasets",
        "prompt_file",
        "benign_glob",
        "benign_source",
        "benign_corpus",
    }

    def walk(value: Any) -> Iterable[str]:
        if isinstance(value, Mapping):
            for k, v in value.items():
                if k in keys:
                    if isinstance(v, str):
                        yield v
                    elif isinstance(v, Sequence) and not isinstance(v, (str, bytes)):
                        for item in v:
                            if isinstance(item, str):
                                yield item
                yield from walk(v)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for item in value:
                yield from walk(item)

    return list(dict.fromkeys(walk(config)))


def dataset_hash_from_config(config: Mapping[str, Any], base_dir: Path | None = None) -> str:
    """Derive a deterministic dataset hash from config-referenced paths."""
    base = base_dir or Path.cwd()
    candidates = list(_iter_dataset_candidates(config))
    if not candidates:
        return "unknown"

    records = []
    for entry in candidates:
        candidate_path = Path(entry)
        if not candidate_path.is_absolute():
            candidate_path = base / candidate_path

        if any(ch in entry for ch in "*?["):
            matches = sorted({p for p in candidate_path.parent.glob(candidate_path.name) if p.exists()})
        else:
            matches = [candidate_path]

        if not matches:
            records.append({"path": str(candidate_path), "status": "missing"})
            continue

        for path in matches:
            if not path.exists():
                records.append({"path": str(path), "status": "missing"})
            elif path.is_dir():
                records.append({"path": str(path), "sha256": _hash_directory(path), "type": "dir"})
            else:
                records.append({"path": str(path), "sha256": _hash_file(path), "type": "file"})

    return _hash_text(_stable_json(records))


class StorageBackend(ABC):
    """Interface for saving run artifacts."""

    @abstractmethod
    def resolve_path(self, category: str, content_hash: str, filename: str) -> Path:
        raise NotImplementedError

    @abstractmethod
    def save_json(self, payload: Mapping[str, Any], *, category: str = "runs", filename: str = "manifest.json") -> Path:
        raise NotImplementedError

    @abstractmethod
    def save_csv(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        category: str = "results",
        filename: str = "results.csv",
        header: Sequence[str] | None = None,
    ) -> Path:
        raise NotImplementedError

    @abstractmethod
    def save_artifact(self, source_path: Path, *, category: str = "figures", filename: str | None = None) -> Path:
        raise NotImplementedError


@dataclass(frozen=True)
class LocalStorageBackend(StorageBackend):
    base_dir: Path = Path.cwd()

    def resolve_path(self, category: str, content_hash: str, filename: str) -> Path:
        shard = content_hash[:2]
        target_dir = self.base_dir / category / shard / content_hash
        target_dir.mkdir(parents=True, exist_ok=True)
        return target_dir / filename

    def save_json(self, payload: Mapping[str, Any], *, category: str = "runs", filename: str = "manifest.json") -> Path:
        data = _stable_json(payload)
        content_hash = _hash_text(data)
        path = self.resolve_path(category, content_hash, filename)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        return path

    def save_csv(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        category: str = "results",
        filename: str = "results.csv",
        header: Sequence[str] | None = None,
    ) -> Path:
        header_list = list(header) if header else sorted({k for row in rows for k in row.keys()})
        content_hash = _hash_text(_stable_json({"header": header_list, "rows": list(rows)}))
        path = self.resolve_path(category, content_hash, filename)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=header_list)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k, "") for k in header_list})
        return path

    def save_artifact(self, source_path: Path, *, category: str = "figures", filename: str | None = None) -> Path:
        if not source_path.exists():
            raise FileNotFoundError(f"Artifact does not exist: {source_path}")
        content_hash = _hash_file(source_path)
        name = filename or source_path.name
        path = self.resolve_path(category, content_hash, name)
        shutil.copy2(source_path, path)
        return path
