# src/cc/io/seeds.py
from __future__ import annotations
import os, random
from typing import Optional
try:
    import numpy as np
except ImportError:
    np = None  # optional dependency

def set_seed(seed: Optional[int]) -> int:
    """Pin all RNGs we use; returns the resolved seed."""
    s = int(seed if seed is not None else os.environ.get("CC_SEED", "1337"))
    random.seed(s)
    os.environ["PYTHONHASHSEED"] = str(s)
    if np is not None:
        np.random.seed(s)
    return s
