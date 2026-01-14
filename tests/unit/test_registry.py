# tests/unit/test_registry.py
"""Registry validation tests."""
from __future__ import annotations

import pytest

from cc.adapters.registry import create_adapter_from_config, get_adapter_class


def test_unknown_adapter_name_fails() -> None:
    with pytest.raises(KeyError) as exc:
        get_adapter_class("nope")
    assert "Unknown adapter" in str(exc.value)


def test_create_adapter_from_config_requires_name() -> None:
    with pytest.raises(KeyError) as exc:
        create_adapter_from_config({})
    assert "missing required 'name'" in str(exc.value)
