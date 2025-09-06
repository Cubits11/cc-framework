# src/cc/guardrails/regex_filter.py
"""Regular-expression–based guardrail with calibration, explainability, and safety niceties.

Features
--------
- Supports one or many patterns.
- Per-pattern and global flags (e.g., re.IGNORECASE).
- Optional 'regex' module (if installed) with timeouts to mitigate catastrophic backtracking;
  falls back to stdlib 're' if unavailable.
- Binary or count-based scoring; configurable 'min_hits' threshold.
- Calibration: chooses 'min_hits' that meets/undershoots target FPR on benign data.
- Explain: returns which pattern(s) matched and where.

Notes
-----
This class remains API-compatible with existing Guardrail interface: blocks(), score(), calibrate().
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Tuple, Dict, Any, Union

import logging
import re

try:
    # Third-party 'regex' supports timeouts; use if available
    import regex as reX  # type: ignore
    _HAS_REGEX = True
except Exception:  # pragma: no cover
    reX = None
    _HAS_REGEX = False

from .base import Guardrail


logger = logging.getLogger(__name__)


# Small helper to normalize flags from strings or ints
_FLAG_MAP: Dict[str, int] = {
    "I": re.IGNORECASE,
    "IGNORECASE": re.IGNORECASE,
    "M": re.MULTILINE,
    "MULTILINE": re.MULTILINE,
    "S": re.DOTALL,
    "DOTALL": re.DOTALL,
    "X": re.VERBOSE,
    "VERBOSE": re.VERBOSE,
    "A": re.ASCII,
    "ASCII": re.ASCII,
    "U": re.UNICODE,
    "UNICODE": re.UNICODE,
}


def _parse_flags(flags: Union[int, str, Iterable[str], None]) -> int:
    if flags is None:
        return 0
    if isinstance(flags, int):
        return flags
    if isinstance(flags, str):
        flags = [flags]
    acc = 0
    for f in flags:
        acc |= _FLAG_MAP.get(f, 0)
    return acc


@dataclass
class CompiledPattern:
    raw: str
    flags: int
    engine: str  # "regex" or "re"
    compiled: Any  # Pattern object


@dataclass
class RegexFilter(Guardrail):
    """
    Regular expression based content filter.

    Parameters
    ----------
    patterns : str | list[str]
        One or more regex patterns (Python 're' syntax).
    flags : int | str | list[str], optional
        Global flags applied to all patterns (e.g., re.IGNORECASE or "I").
    per_pattern_flags : dict[str, (int|str|list[str])], optional
        Optional map of pattern -> flags (overrides global for that pattern).
    min_hits : int, default=1
        Number of distinct pattern matches required to block.
    use_regex_backend : bool, default=True
        If True and 'regex' is installed, use it with timeouts to reduce risk
        of catastrophic backtracking; otherwise fallback to stdlib 're'.
    match_timeout_ms : int, default=15
        Per-pattern search timeout (ms) when using 'regex' backend.
    binary_score : bool, default=True
        If True, score() returns 1.0 if blocks() is True else 0.0.
        If False, score() returns (#matches / #patterns), clipped to [0,1].
    name : str, optional
        Friendly name for reporting; defaults to "regex_filter".
    """

    patterns: Union[str, List[str]]
    flags: Union[int, str, Iterable[str], None] = None
    per_pattern_flags: Optional[Dict[str, Union[int, str, Iterable[str]]]] = None
    min_hits: int = 1
    use_regex_backend: bool = True
    match_timeout_ms: int = 15
    binary_score: bool = True
    name: str = "regex_filter"

    compiled: List[CompiledPattern] = field(default_factory=list, init=False)
    threshold: float = field(default=0.5, init=False)  # preserved for compatibility
    calibrated_fpr_: Optional[float] = field(default=None, init=False)
    calib_hist_: Optional[Dict[str, Any]] = field(default=None, init=False)

    # -------------------------------
    # Guardrail lifecycle
    # -------------------------------
    def __post_init__(self) -> None:
        self._compile_all()

    # -------------------------------
    # Public API
    # -------------------------------
    def blocks(self, text: str) -> bool:
        """Return True if at least 'min_hits' patterns match the text."""
        hits, _ = self._match_details(text, want_spans=False)
        return hits >= self.min_hits

    def score(self, text: str) -> float:
        """
        Binary or count-based score.
        - If binary_score=True: 1.0 if blocked else 0.0
        - Else: (#hits / #patterns) in [0,1]
        """
        hits, _ = self._match_details(text, want_spans=False)
        if self.binary_score:
            return 1.0 if hits >= self.min_hits else 0.0
        denom = max(1, len(self.compiled))
        return min(1.0, hits / denom)

    def explain(self, text: str, max_examples: int = 3) -> Dict[str, Any]:
        """
        Return a structured explanation of which patterns matched.
        {
          "blocked": bool,
          "hits": int,
          "min_hits": int,
          "matches": [
            {"pattern": "...", "engine":"regex|re", "examples": ["match1","match2", ...]},
            ...
          ]
        }
        """
        hits, spans = self._match_details(text, want_spans=True)
        examples: List[Dict[str, Any]] = []
        for pat, mlist in spans:
            ex = []
            for m in mlist[:max_examples]:
                # m can be a Match object (regex or re); extract the matched substring safely
                try:
                    ex.append(m.group(0))
                except Exception:  # pragma: no cover
                    ex.append(str(m))
            examples.append({"pattern": pat.raw, "engine": pat.engine, "examples": ex})
        return {"blocked": hits >= self.min_hits, "hits": hits, "min_hits": self.min_hits, "matches": examples}

    def calibrate(self, benign_texts: List[str], target_fpr: float = 0.05) -> None:
        """
        Choose the smallest 'min_hits' that achieves FPR <= target on benign_texts.

        Strategy:
          - For k = 1..K (K = #patterns), compute empirical FPR_k = mean[ hits(text) >= k ].
          - Set min_hits = smallest k with FPR_k <= target_fpr (if any); otherwise set K (most strict).
        Logs a short summary and stores calibration history in 'calib_hist_'.
        """
        if not benign_texts:
            logger.warning("RegexFilter.calibrate: no benign_texts provided; skipping calibration.")
            self.calibrated_fpr_ = None
            self.calib_hist_ = None
            return

        # Pre-compute hits per text
        hit_counts = [self._match_details(t, want_spans=False)[0] for t in benign_texts]
        K = max(1, len(self.compiled))
        fprs = []
        chosen_k = K
        for k in range(1, K + 1):
            fpr_k = sum(1 for h in hit_counts if h >= k) / len(hit_counts)
            fprs.append((k, fpr_k))
            if fpr_k <= target_fpr and k < chosen_k:
                chosen_k = k

        # If no k meets target, choose K (strictest)
        self.min_hits = chosen_k
        # The achieved FPR at chosen_k:
        self.calibrated_fpr_ = next(f for k, f in fprs if k == self.min_hits)

        self.calib_hist_ = {
            "target_fpr": target_fpr,
            "fprs": fprs,
            "chosen_min_hits": self.min_hits,
            "achieved_fpr": self.calibrated_fpr_,
            "n_benign": len(benign_texts),
        }

        logger.info(
            "RegexFilter calibration — target FPR=%.3f, chosen min_hits=%d, achieved FPR=%.3f (K=%d patterns)",
            target_fpr,
            self.min_hits,
            self.calibrated_fpr_,
            K,
        )

    # -------------------------------
    # Config helpers
    # -------------------------------
    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "RegexFilter":
        """
        Construct from a dict config; useful when loading YAML/JSON.
        Expected keys (all optional except 'patterns'):
          - patterns: str | list[str]
          - flags: int | str | list[str]
          - per_pattern_flags: {pattern: flags}
          - min_hits: int
          - use_regex_backend: bool
          - match_timeout_ms: int
          - binary_score: bool
          - name: str
        """
        return cls(
            patterns=cfg["patterns"],
            flags=cfg.get("flags"),
            per_pattern_flags=cfg.get("per_pattern_flags"),
            min_hits=int(cfg.get("min_hits", 1)),
            use_regex_backend=bool(cfg.get("use_regex_backend", True)),
            match_timeout_ms=int(cfg.get("match_timeout_ms", 15)),
            binary_score=bool(cfg.get("binary_score", True)),
            name=str(cfg.get("name", "regex_filter")),
        )

    def to_config(self) -> Dict[str, Any]:
        """Serialize to a simple config dict."""
        return {
            "patterns": [p.raw for p in self.compiled] if self.compiled else self.patterns,
            "flags": self.flags if isinstance(self.flags, int) else None,
            "min_hits": self.min_hits,
            "use_regex_backend": self.use_regex_backend,
            "match_timeout_ms": self.match_timeout_ms,
            "binary_score": self.binary_score,
            "name": self.name,
        }

    # -------------------------------
    # Internals
    # -------------------------------
    def _compile_all(self) -> None:
        pats = self.patterns if isinstance(self.patterns, list) else [self.patterns]
        if not pats:
            raise ValueError("RegexFilter requires at least one pattern.")
        global_flags = _parse_flags(self.flags)
        pp_flags = self.per_pattern_flags or {}

        compiled: List[CompiledPattern] = []
        for raw in pats:
            f = global_flags | _parse_flags(pp_flags.get(raw))
            try:
                if self.use_regex_backend and _HAS_REGEX:
                    c = reX.compile(raw, flags=f)
                    compiled.append(CompiledPattern(raw=raw, flags=f, engine="regex", compiled=c))
                else:
                    c = re.compile(raw, flags=f)
                    compiled.append(CompiledPattern(raw=raw, flags=f, engine="re", compiled=c))
            except Exception as e:
                logger.error("Failed to compile regex pattern %r: %s", raw, e)
                raise
        self.compiled = compiled

    def _search_once(self, pat: CompiledPattern, text: str) -> Optional[Any]:
        """Search with optional timeout if using 'regex' backend."""
        if pat.engine == "regex" and _HAS_REGEX:
            try:
                return pat.compiled.search(text, timeout=self.match_timeout_ms / 1000.0)
            except reX.TimeoutError:  # type: ignore
                logger.warning("Regex timeout for pattern %r (%.1f ms).", pat.raw, self.match_timeout_ms)
                return None
            except Exception as e:  # pragma: no cover
                logger.error("Regex engine error for %r: %s", pat.raw, e)
                return None
        else:
            # stdlib 're' has no timeout
            return pat.compiled.search(text)

    def _finditer(self, pat: CompiledPattern, text: str, max_matches: int = 16) -> List[Any]:
        """Return up to 'max_matches' match objects for explanation."""
        out: List[Any] = []
        try:
            if pat.engine == "regex" and _HAS_REGEX:
                for m in pat.compiled.finditer(text, timeout=self.match_timeout_ms / 1000.0):
                    out.append(m)
                    if len(out) >= max_matches:
                        break
            else:
                for m in pat.compiled.finditer(text):
                    out.append(m)
                    if len(out) >= max_matches:
                        break
        except Exception:  # pragma: no cover
            pass
        return out

    def _match_details(self, text: str, want_spans: bool) -> Tuple[int, List[Tuple[CompiledPattern, List[Any]]]]:
        """Return (#patterns that match, [(pattern, [matches]) ...])"""
        hits = 0
        spans: List[Tuple[CompiledPattern, List[Any]]] = []
        for pat in self.compiled:
            m = self._search_once(pat, text)
            if m:
                hits += 1
                if want_spans:
                    spans.append((pat, self._finditer(pat, text)))
        return hits, spans
