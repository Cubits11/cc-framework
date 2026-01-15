"""Development convenience package to enable `python -m cc...` from repo root.

This injects the `src` directory into sys.path so the real package at
`src/cc` is importable without an editable install.
"""

from __future__ import annotations

import sys
from pathlib import Path
from pkgutil import extend_path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

__path__ = extend_path(__path__, __name__)
