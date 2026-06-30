#!/usr/bin/env python3
"""Check the repository's generated artifact boundary."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

PAPER_ARTIFACT_FILES = {
    "benchmark_example_summary.json",
    "environment.json",
    "figure_1_fh_interval.png",
    "figure_2_independence_regret.png",
    "figure_3_correlation_cliff_toy.png",
    "figure_4_runtime_scaling.png",
    "manifest.json",
    "minimal_bounds.json",
    "minimal_bundle.json",
    "minimal_witnesses.json",
    "table_1_classical_frechet_bounds.csv",
    "table_2_metric_examples.csv",
    "table_3_witness_verification.csv",
    "table_4_sample_complexity.csv",
    "table_5_runtime_scaling.csv",
}

PAPER_FIGURE_FILES = {
    "paper/figures/cc_convergence.pdf",
    "paper/figures/fig_week3_power_curve.png",
    "paper/figures/phase_diagram.pdf",
    "paper/figures/roc_comparison.pdf",
    "paper/figures/theorem1_visual.html",
}

REQUIRED_FIXTURES = {
    "tests/fixtures/week5_scan/scan.csv",
    "tests/fixtures/reporting/artifact.txt",
    "tests/fixtures/reporting/audit.jsonl",
    "tests/fixtures/reporting/calibration_summary.json",
    "tests/fixtures/reporting/figure_manifest.json",
    "tests/fixtures/reporting/measurement.json",
}

RUNTIME_MARKERS = {
    "checkpoints/README.md",
    "figs/README.md",
    "figures/README.md",
    "results/README.md",
    "runs/README.md",
}

ARCHIVE_MARKERS = {
    "docs/archive/generated-checkpoints/README.md",
    "docs/archive/generated-results/README.md",
    "docs/archive/generated-results/smoke/README.md",
    "docs/archive/generated-results/week5_scan/README.md",
}

RUNTIME_ROOTS = (
    "checkpoints/",
    "figs/",
    "figures/",
    "results/",
    "runs/",
)

ALLOWED_ARCHIVE_PREFIXES = (
    "docs/archive/",
    "theory/icse/lesson1/runs/q1/",
)


@dataclass(frozen=True)
class Finding:
    path: str
    message: str
    hint: str | None = None

    def format(self) -> str:
        if self.hint:
            return f"- {self.path}: {self.message} Hint: {self.hint}"
        return f"- {self.path}: {self.message}"


@dataclass(frozen=True)
class StatusEntry:
    code: str
    path: str
    raw: str
    original_path: str | None = None

    @property
    def is_untracked(self) -> bool:
        return self.code == "??"


def normalize_path(path: str | Path) -> str:
    normalized = str(path).replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized.strip()


def is_under(path: str, prefix: str) -> bool:
    normalized = normalize_path(path)
    normalized_prefix = normalize_path(prefix)
    if not normalized_prefix.endswith("/"):
        normalized_prefix = f"{normalized_prefix}/"
    return normalized.startswith(normalized_prefix)


def is_runtime_path(path: str) -> bool:
    normalized = normalize_path(path)
    return any(is_under(normalized, root) for root in RUNTIME_ROOTS)


def classify_tracked_path(path: str) -> str:
    normalized = normalize_path(path)

    if normalized in RUNTIME_MARKERS:
        return "runtime_marker"
    if is_runtime_path(normalized):
        return "forbidden_runtime_output"

    if is_under(normalized, "artifacts/paper/"):
        filename = normalized.removeprefix("artifacts/paper/")
        if filename in PAPER_ARTIFACT_FILES:
            return "paper_release_artifact"
        return "unexpected_paper_artifact"

    if is_under(normalized, "artifacts/"):
        return "unexpected_artifact"

    if is_under(normalized, "paper/figures/"):
        if normalized in PAPER_FIGURE_FILES:
            return "paper_source_figure"
        return "unexpected_paper_figure"

    if is_under(normalized, "tests/fixtures/"):
        return "test_fixture"

    if any(is_under(normalized, prefix) for prefix in ALLOWED_ARCHIVE_PREFIXES):
        return "historical_archive"

    if is_under(normalized, "examples/"):
        return "example_source"

    return "source"


def validate_tracked_paths(tracked_paths: Iterable[str]) -> list[Finding]:
    findings: list[Finding] = []
    for raw_path in tracked_paths:
        path = normalize_path(raw_path)
        classification = classify_tracked_path(path)

        if classification == "forbidden_runtime_output":
            findings.append(
                Finding(
                    path,
                    "tracked file is under a runtime-only generated directory",
                    "move it to tests/fixtures, docs/archive, artifacts/paper, or remove it from tracking",
                )
            )
        elif classification == "unexpected_paper_artifact":
            findings.append(
                Finding(
                    path,
                    "tracked file under artifacts/paper is not in the release artifact allowlist",
                    "update PAPER_ARTIFACT_FILES only after updating the paper artifact manifest",
                )
            )
        elif classification == "unexpected_artifact":
            findings.append(
                Finding(
                    path,
                    "tracked file under artifacts/ is outside artifacts/paper",
                    "move it to a fixture/archive path or document a new release-artifact root",
                )
            )
        elif classification == "unexpected_paper_figure":
            findings.append(
                Finding(
                    path,
                    "tracked paper figure is not in the manuscript figure allowlist",
                    "promote it deliberately by updating PAPER_FIGURE_FILES and docs",
                )
            )
    return findings


def validate_required_files(repo_root: Path = REPO_ROOT) -> list[Finding]:
    findings: list[Finding] = []
    required_paths = (
        {f"artifacts/paper/{filename}" for filename in PAPER_ARTIFACT_FILES}
        | PAPER_FIGURE_FILES
        | REQUIRED_FIXTURES
        | RUNTIME_MARKERS
        | ARCHIVE_MARKERS
    )

    for path in sorted(required_paths):
        if not (repo_root / path).is_file():
            findings.append(
                Finding(
                    path,
                    "required artifact boundary file is missing",
                    "restore the file or update scripts/check_artifact_boundary.py intentionally",
                )
            )

    findings.extend(validate_paper_manifest(repo_root))
    return findings


def validate_paper_manifest(repo_root: Path = REPO_ROOT) -> list[Finding]:
    manifest_path = repo_root / "artifacts/paper/manifest.json"
    if not manifest_path.is_file():
        return [
            Finding(
                "artifacts/paper/manifest.json",
                "missing required paper artifact manifest",
                "run make reproduce-paper and review the resulting manifest",
            )
        ]

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [
            Finding(
                "artifacts/paper/manifest.json",
                f"paper artifact manifest is not valid JSON ({exc})",
                "regenerate or repair the manifest",
            )
        ]

    required_files = set(manifest.get("required_files", []))
    manifest_entries = {
        entry.get("filename")
        for entry in manifest.get("files", [])
        if isinstance(entry, dict)
    }
    expected_hashed = PAPER_ARTIFACT_FILES - {"manifest.json"}
    findings: list[Finding] = []

    missing_required = PAPER_ARTIFACT_FILES - required_files
    if missing_required:
        findings.append(
            Finding(
                "artifacts/paper/manifest.json",
                "manifest required_files omits expected paper artifacts: "
                + ", ".join(sorted(missing_required)),
                "regenerate artifacts or update the boundary allowlist deliberately",
            )
        )

    missing_entries = expected_hashed - manifest_entries
    if missing_entries:
        findings.append(
            Finding(
                "artifacts/paper/manifest.json",
                "manifest files omits expected hashed artifacts: "
                + ", ".join(sorted(missing_entries)),
                "regenerate artifacts or update the boundary allowlist deliberately",
            )
        )

    return findings


def parse_status_line(line: str) -> StatusEntry | None:
    raw = line.rstrip("\n")
    if not raw:
        return None
    if len(raw) < 3:
        return StatusEntry(code=raw, path="", raw=raw)

    code = raw[:2]
    payload = raw[3:]
    original_path: str | None = None
    path = payload
    if " -> " in payload:
        original_path, path = payload.split(" -> ", 1)

    return StatusEntry(
        code=code,
        path=normalize_path(path),
        raw=raw,
        original_path=normalize_path(original_path) if original_path else None,
    )


def parse_status_lines(lines: Iterable[str]) -> list[StatusEntry]:
    entries: list[StatusEntry] = []
    for line in lines:
        entry = parse_status_line(line)
        if entry is not None:
            entries.append(entry)
    return entries


def validate_after_run_status(
    current_status_lines: Iterable[str],
    baseline_status_lines: Iterable[str] = (),
) -> list[Finding]:
    baseline = {line.rstrip("\n") for line in baseline_status_lines if line.rstrip("\n")}
    findings: list[Finding] = []

    for entry in parse_status_lines(current_status_lines):
        if entry.raw in baseline:
            continue

        if entry.is_untracked and is_runtime_path(entry.path):
            continue

        if entry.is_untracked:
            findings.append(
                Finding(
                    entry.path,
                    "new untracked file appeared outside ignored runtime output roots",
                    "write generated outputs under results/, checkpoints/, runs/, figs/, or figures/",
                )
            )
        else:
            findings.append(
                Finding(
                    entry.path,
                    f"new tracked git diff appeared after the reproducibility command (status {entry.code})",
                    "commands in this lane should write to temp dirs or ignored runtime roots",
                )
            )

    return findings


def git_lines(args: Sequence[str], repo_root: Path) -> list[str]:
    result = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.splitlines()


def read_lines(path: Path | None) -> list[str]:
    if path is None:
        return []
    return path.read_text(encoding="utf-8").splitlines()


def run_static(repo_root: Path) -> list[Finding]:
    tracked_paths = git_lines(["ls-files"], repo_root)
    return validate_tracked_paths(tracked_paths) + validate_required_files(repo_root)


def run_after(repo_root: Path, baseline_path: Path | None) -> list[Finding]:
    current_status = git_lines(["status", "--porcelain=v1", "--untracked-files=all"], repo_root)
    baseline_status = read_lines(baseline_path)
    return validate_after_run_status(current_status, baseline_status)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--static", action="store_true", help="validate tracked paths and markers")
    parser.add_argument(
        "--after-run",
        action="store_true",
        help="validate that a command sequence did not create new diffs",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        help="baseline git status file for --after-run comparisons",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = args.repo_root.resolve()

    run_static_mode = args.static or not args.after_run
    findings: list[Finding] = []
    if run_static_mode:
        findings.extend(run_static(repo_root))
    if args.after_run:
        findings.extend(run_after(repo_root, args.baseline))

    if findings:
        print("Artifact boundary check failed:", file=sys.stderr)
        for finding in findings:
            print(finding.format(), file=sys.stderr)
        return 1

    modes = []
    if run_static_mode:
        modes.append("static")
    if args.after_run:
        modes.append("after-run")
    print(f"Artifact boundary check passed ({', '.join(modes)}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
