"""
Pytest bootstrap for src/ layout.

Why:
- Repo uses ./src for packages.
- Some tests spawn pytest in a subprocess without PYTHONPATH=src.
- Without this, imports like `from cc.adapters...` can fail during collection.

This ensures ./src is always on sys.path for any pytest invocation.
"""
from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
src = repo_root / "src"
if src.is_dir():
    src_str = str(src)
    if src_str not in sys.path:
        # Put first so local src wins over any installed package named `cc`.
        sys.path.insert(0, src_str)
