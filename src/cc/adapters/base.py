# src/cc/adapters/base.py
"""Base adapter interfaces for third-party guardrail systems."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional

Verdict = Literal["allow", "block", "review"]


@dataclass(frozen=True)
class Decision:
    """Normalized guardrail decision returned by adapter checks.

    Attributes
    ----------
    verdict:
        One of {"allow", "block", "review"}.
    category:
        Optional policy category or violation label when available.
    score:
        Optional confidence or risk score (adapter-specific scale).
    rationale:
        Optional short explanation for the decision.
    raw:
        Raw adapter payload (dict or string) for reproducibility/debugging.
    adapter_name:
        Stable adapter name (e.g., "llama_guard").
    adapter_version:
        Adapter or model version (e.g., model id or library version).
    """

    verdict: Verdict
    category: Optional[str]
    score: Optional[float]
    rationale: Optional[str]
    raw: Dict[str, Any] | str
    adapter_name: str
    adapter_version: str

    def __post_init__(self) -> None:
        if self.verdict not in {"allow", "block", "review"}:
            raise ValueError(f"Invalid verdict: {self.verdict}")
        if not isinstance(self.raw, (dict, str)):
            raise TypeError("Decision.raw must be a dict or string")
        if self.score is not None and not isinstance(self.score, (int, float)):
            raise TypeError("Decision.score must be float-like when provided")


class GuardrailAdapter(ABC):
    """Abstract base class for guardrail adapters."""

    name: str
    version: str
    supports_input_check: bool
    supports_output_check: bool

    @abstractmethod
    def check(self, prompt: str, response: Optional[str], metadata: Dict[str, Any]) -> Decision:
        """Check a prompt/response pair and return a Decision."""
        raise NotImplementedError

