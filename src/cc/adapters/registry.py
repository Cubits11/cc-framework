# src/cc/adapters/registry.py
"""Adapter registry helpers."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Type

from .base import GuardrailAdapter
from .guardrails_ai import GuardrailsAIAdapter
from .llama_guard import LlamaGuardAdapter
from .nemo_guardrails import NeMoGuardrailsAdapter

ADAPTER_REGISTRY: Dict[str, Type[GuardrailAdapter]] = {
    "llama_guard": LlamaGuardAdapter,
    "nemo_guardrails": NeMoGuardrailsAdapter,
    "guardrails_ai": GuardrailsAIAdapter,
}

def list_adapters() -> Dict[str, Type[GuardrailAdapter]]:
    return dict(ADAPTER_REGISTRY)

def get_adapter_class(name: str) -> Type[GuardrailAdapter]:
    key = (name or "").strip()
    if key not in ADAPTER_REGISTRY:
        raise KeyError(f"Unknown adapter '{name}'. Available: {sorted(ADAPTER_REGISTRY)}")
    return ADAPTER_REGISTRY[key]


def create_adapter(name: str, **kwargs: Any) -> GuardrailAdapter:
    cls = get_adapter_class(name)
    return cls(**kwargs)  # type: ignore[misc]


def create_adapter_from_config(config: Mapping[str, Any]) -> GuardrailAdapter:
    """Instantiate an adapter from a config mapping with validation."""
    if "name" not in config:
        raise KeyError("Adapter config missing required 'name' field.")
    name = str(config["name"])
    params = dict(config)
    params.pop("name", None)
    return create_adapter(name, **params)
