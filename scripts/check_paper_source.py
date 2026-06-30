#!/usr/bin/env python
"""Static checks for the Paper 1 LaTeX source tree."""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

DEFAULT_PAPER_DIR = Path("paper")
FORBIDDEN_PATTERNS = (
    "CC = 0.87",
    "CC_max",
    "J-statistic",
    "J statistic",
    "Youden",
    "\\mathrm{CC}_{\\max}",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-dir", type=Path, default=DEFAULT_PAPER_DIR)
    parser.add_argument(
        "--latex",
        action="store_true",
        help="Run latexmk when available after static checks pass.",
    )
    args = parser.parse_args()

    errors = check_paper_source(args.paper_dir)
    if errors:
        print("Paper source checks failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    if args.latex:
        latexmk = shutil.which("latexmk")
        if latexmk is None:
            print("latexmk not found; skipped LaTeX compile.")
        else:
            result = subprocess.run(
                [latexmk, "-pdf", "-interaction=nonstopmode", "main.tex"],
                cwd=args.paper_dir,
                check=False,
            )
            if result.returncode != 0:
                return result.returncode
    print(f"Paper source checks passed in {args.paper_dir}")
    return 0


def check_paper_source(paper_dir: Path) -> list[str]:
    errors: list[str] = []
    main_path = paper_dir / "main.tex"
    if not main_path.is_file():
        return [f"missing {main_path}"]

    main_text = main_path.read_text(encoding="utf-8")
    if "Sharp Composition Bounds for AI Guardrails Under Unknown Dependence" not in main_text:
        errors.append("main.tex title does not match Paper 1 title")

    for input_path in _input_paths(main_text, paper_dir):
        if not input_path.is_file():
            errors.append(f"missing input file: {input_path}")

    source_files = [main_path, *sorted((paper_dir / "sections").glob("*.tex"))]
    for path in source_files:
        text = path.read_text(encoding="utf-8")
        if path.parent.name == "sections":
            for token in ("\\documentclass", "\\begin{document}", "\\end{document}"):
                if token in text:
                    errors.append(f"{path} contains section-forbidden token {token}")
        for pattern in FORBIDDEN_PATTERNS:
            if pattern in text:
                errors.append(f"{path} contains forbidden Paper 1 phrase: {pattern}")
    return errors


def _input_paths(main_text: str, paper_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for match in re.finditer(r"\\input\{([^}]+)\}", main_text):
        raw = match.group(1)
        path = paper_dir / raw
        if path.suffix != ".tex":
            path = path.with_suffix(".tex")
        paths.append(path)
    return paths


if __name__ == "__main__":
    raise SystemExit(main())
