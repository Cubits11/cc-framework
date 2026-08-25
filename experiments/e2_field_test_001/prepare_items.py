#!/usr/bin/env python3
"""Materialize the frozen E2 Field Test 001 item selection.

This program is deliberately a *pre-score* tool.  It reads only the pinned
source CSV files, applies the selection rule in
``docs/research/E2_FIELD_TEST_001_PROTOCOL.md``, and writes a private input
file plus a non-text manifest.  It imports no model and performs no inference.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


STUDY_ID = "E2-FIELD-TEST-001"
SELECTION_VERSION = "v1"
SOURCE_FILES = ("data/test.csv", "data/train.csv", "data/validation.csv")
TARGET_ITEMS = 1_097


class SelectionError(ValueError):
    """The frozen input-selection contract could not be satisfied."""


@dataclass(frozen=True)
class Candidate:
    source_file: str
    source_row: int
    text: str
    item_hash: str

    @property
    def item_id(self) -> str:
        return f"{STUDY_ID}:{self.source_file}:{self.source_row}"


@dataclass(frozen=True)
class SelectedItem:
    candidate: Candidate
    rank: str

    def private_record(self) -> dict[str, object]:
        return {
            "item_id": self.candidate.item_id,
            "source_file": self.candidate.source_file,
            "source_row": self.candidate.source_row,
            "source_label": 1,
            "item_hash": self.candidate.item_hash,
            "rank": self.rank,
            "text": self.candidate.text,
        }

    def manifest_record(self) -> dict[str, object]:
        return {
            "item_id": self.candidate.item_id,
            "source_file": self.candidate.source_file,
            "source_row": self.candidate.source_row,
            "source_label": 1,
            "item_hash": self.candidate.item_hash,
            "rank": self.rank,
        }


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rank(*, dataset_revision: str, candidate: Candidate) -> str:
    material = (
        f"{STUDY_ID}/{SELECTION_VERSION}\0{dataset_revision}\0{candidate.source_file}\0"
        f"{candidate.source_row}\0{candidate.item_hash}"
    ).encode("utf-8")
    return _sha256_bytes(material)


def _read_candidates(dataset_root: Path) -> tuple[list[Candidate], dict[str, str]]:
    candidates: list[Candidate] = []
    source_hashes: dict[str, str] = {}
    for source_file in SOURCE_FILES:
        path = dataset_root / source_file
        if not path.is_file():
            raise SelectionError(f"required frozen source file is absent: {path}")
        source_hashes[source_file] = _sha256_file(path)
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or not {"text", "label"} <= set(reader.fieldnames):
                raise SelectionError(
                    f"{source_file} must expose UTF-8 CSV fields 'text' and 'label'; "
                    f"found {reader.fieldnames!r}"
                )
            for source_row, row in enumerate(reader):
                label = row.get("label")
                text = row.get("text")
                if label is None or text is None:
                    raise SelectionError(f"{source_file}:{source_row}: missing text or label field")
                try:
                    numeric_label = int(label)
                except ValueError as exc:
                    raise SelectionError(
                        f"{source_file}:{source_row}: non-integer label {label!r}"
                    ) from exc
                if numeric_label not in {0, 1}:
                    raise SelectionError(
                        f"{source_file}:{source_row}: binary label required, got {numeric_label!r}"
                    )
                if numeric_label != 1 or not text:
                    continue
                candidates.append(
                    Candidate(
                        source_file=source_file,
                        source_row=source_row,
                        text=text,
                        item_hash=_sha256_bytes(text.encode("utf-8")),
                    )
                )
    return candidates, source_hashes


def select_items(
    candidates: Iterable[Candidate], *, dataset_revision: str, target_items: int = TARGET_ITEMS
) -> tuple[list[SelectedItem], int]:
    """Deduplicate and deterministically choose the first frozen target items."""

    if target_items <= 0:
        raise SelectionError("target_items must be positive")
    earliest_by_hash: dict[str, Candidate] = {}
    for candidate in candidates:
        incumbent = earliest_by_hash.get(candidate.item_hash)
        if incumbent is None or (candidate.source_file, candidate.source_row) < (
            incumbent.source_file,
            incumbent.source_row,
        ):
            earliest_by_hash[candidate.item_hash] = candidate
    deduplicated = list(earliest_by_hash.values())
    selected = [
        SelectedItem(candidate=candidate, rank=_rank(dataset_revision=dataset_revision, candidate=candidate))
        for candidate in deduplicated
    ]
    selected.sort(key=lambda item: (bytes.fromhex(item.rank), item.candidate.source_file, item.candidate.source_row))
    if len(selected) < target_items:
        raise SelectionError(
            f"NOT ACTIVATED: {len(selected)} unique eligible INJECTION records remain; "
            f"the frozen target is {target_items}"
        )
    return selected[:target_items], len(deduplicated)


def _write_jsonl(path: Path, records: Iterable[dict[str, object]]) -> str:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
            handle.write("\n")
    return _sha256_file(path)


def materialize(
    *, dataset_root: Path, dataset_revision: str, out_dir: Path, target_items: int
) -> dict[str, object]:
    """Write the private item input and the hash-only public selection manifest."""

    if not dataset_revision or "\x00" in dataset_revision:
        raise SelectionError("dataset_revision must be a nonempty NUL-free string")
    candidates, source_hashes = _read_candidates(dataset_root)
    selected, deduplicated_count = select_items(
        candidates, dataset_revision=dataset_revision, target_items=target_items
    )
    if out_dir.exists():
        raise SelectionError(f"refusing to overwrite output directory: {out_dir}")
    out_dir.mkdir(parents=True)
    private_path = out_dir / "items.private.jsonl"
    manifest_path = out_dir / "items.jsonl"
    private_hash = _write_jsonl(private_path, (item.private_record() for item in selected))
    manifest_hash = _write_jsonl(manifest_path, (item.manifest_record() for item in selected))
    selection = {
        "study_id": STUDY_ID,
        "selection_version": SELECTION_VERSION,
        "dataset_revision": dataset_revision,
        "source_files": source_hashes,
        "eligible_records_before_deduplication": len(candidates),
        "unique_eligible_records": deduplicated_count,
        "selected_items": len(selected),
        "items_jsonl_sha256": manifest_hash,
        "items_private_jsonl_sha256": private_hash,
    }
    selection_path = out_dir / "selection_manifest.json"
    selection_path.write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(selection_path, 0o444)
    return selection


def _write_source_csv(path: Path, rows: list[tuple[str, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["text", "label"])
        writer.writeheader()
        writer.writerows({"text": text, "label": label} for text, label in rows)


def self_test() -> None:
    """Exercise selection/deduplication only; no model or external data is involved."""

    root = Path(tempfile.mkdtemp(prefix="e2-field-test-selection."))
    try:
        _write_source_csv(root / "data/test.csv", [(f"prompt {index}", 1) for index in range(600)])
        _write_source_csv(
            root / "data/train.csv",
            [("prompt 1", 1)] + [(f"prompt {index}", 1) for index in range(600, 1_200)],
        )
        _write_source_csv(root / "data/validation.csv", [("benign", 0)])
        out = root / "out"
        result = materialize(
            dataset_root=root,
            dataset_revision="0" * 40,
            out_dir=out,
            target_items=1_097,
        )
        assert result["selected_items"] == 1_097
        assert result["eligible_records_before_deduplication"] == 1_201
        assert result["unique_eligible_records"] == 1_200
        first = (out / "items.jsonl").read_text(encoding="utf-8")
        assert "\"text\"" not in first
        assert len(first.splitlines()) == 1_097
    finally:
        shutil.rmtree(root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--dataset-revision")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--target-items", type=int, default=TARGET_ITEMS)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if not args.self_test and (args.dataset_root is None or args.dataset_revision is None or args.out_dir is None):
        parser.error("--dataset-root, --dataset-revision, and --out-dir are required unless --self-test")
    return args


def main() -> int:
    args = parse_args()
    if args.self_test:
        self_test()
        print("prepare_items self-test: PASS")
        return 0
    try:
        result = materialize(
            dataset_root=args.dataset_root,
            dataset_revision=args.dataset_revision,
            out_dir=args.out_dir,
            target_items=args.target_items,
        )
    except SelectionError as exc:
        print(f"prepare_items: {exc}")
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
