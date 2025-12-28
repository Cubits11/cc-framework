# src/cc/adapters/registry.py
"""Adapter registry helpers."""

from __future__ import annotations

from typing import Dict, Type

from .base import GuardrailAdapter
from .guardrails_ai import GuardrailsAIAdapter
from .llama_guard import LlamaGuardAdapter
from .nemo_guardrails import NeMoGuardrailsAdapter

ADAPTER_REGISTRY: Dict[str, Type[GuardrailAdapter]] = {
    "llama_guard": LlamaGuardAdapter,
    "nemo_guardrails": NeMoGuardrailsAdapter,
    "guardrails_ai": GuardrailsAIAdapter,
}

