# tests/unit/core/models/test_models_base.py

"""
Core ModelBase / ORM / serialization behaviour.

Scope:
- ModelBase defaults, immutability, hashing and hash exclusion
- Timestamp normalization on construction
- Thread-safety of blake3_hash cache
- Avro / Protobuf export (feature-gated)
- OpenAPI schema generation
- migrate() best-effort behaviour
- AuditColumnsMixin + OrmBase SQLAlchemy wiring
"""

import concurrent.futures
import json
from typing import Any, Dict, Optional

import pytest
from hypothesis import given, strategies as st
from pydantic import Field, ValidationError, computed_field

from cc.core.models import (
    AVRO_AVAILABLE,
    PROTO_AVAILABLE,
    SQLALCHEMY_AVAILABLE,
    AuditColumnsMixin,
    MAX_REASONABLE_UNIX_TIMESTAMP,
    ModelBase,
    OrmBase,
)


# ---------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------


def assert_immutable(instance: Any, field: str, new_value: Any) -> None:
    """
    Ensure models are effectively immutable from the outside.

    We treat any attempt to reassign a field as an error, regardless of the
    exact exception type (AttributeError, ValidationError, or TypeError).
    """
    with pytest.raises((AttributeError, ValidationError, TypeError)):
        setattr(instance, field, new_value)


# ---------------------------------------------------------------------
# ModelBase core semantics
# ---------------------------------------------------------------------


def test_modelbase_defaults_and_hash_cache_and_salt():
    """
    Basic invariants for a minimal ModelBase subclass:

    - schema_version default
    - creator_id default
    - updated_at > 0
    - blake3_hash caching
    - exclude + salt actually perturb the hash
    """

    class TestModel(ModelBase):
        field: int = 1

    m = TestModel()

    # Core defaults
    assert isinstance(m.schema_version, str)
    assert m.schema_version == "4.2"
    assert m.creator_id is None
    assert m.updated_at > 0

    # Cached hash is stable across calls
    h1 = m.blake3_hash()
    h2 = m.blake3_hash()
    assert isinstance(h1, str) and len(h1) == 64
    assert h1 == h2

    # Excluding a field changes the hash (semantic view changes)
    h_excl = m.blake3_hash(exclude=["field"], use_cache=False)
    assert h_excl != h1

    # Salt changes the hash
    h_salt = m.blake3_hash(use_cache=False, salt=b"x")
    assert h_salt != h1


@given(
    ts=st.floats(
        min_value=-MAX_REASONABLE_UNIX_TIMESTAMP,
        max_value=MAX_REASONABLE_UNIX_TIMESTAMP,
        allow_nan=False,
        allow_infinity=False,
    )
)
def test_modelbase_updated_at_roundtrips_for_finite_ts(ts: float):
    """
    updated_at semantics for finite range:

    - For -MAX_REASONABLE_UNIX_TIMESTAMP <= ts <= MAX_REASONABLE_UNIX_TIMESTAMP,
      the value is honoured verbatim (no normalization), regardless of sign.
    - None is handled separately and mapped to "now" (covered by defaults test).
    """

    class TestModel(ModelBase):
        field: int = 1

    m = TestModel(updated_at=ts)
    assert m.updated_at == ts


def test_modelbase_updated_at_rejects_too_large_timestamp():
    """
    Timestamps strictly larger than MAX_REASONABLE_UNIX_TIMESTAMP must be rejected.

    This protects against platform-dependent overflow and absurdly large values.
    """

    class TestModel(ModelBase):
        field: int = 1

    too_large = MAX_REASONABLE_UNIX_TIMESTAMP + 1.0

    with pytest.raises(ValidationError):
        TestModel(updated_at=too_large)


def test_modelbase_immutability_after_creation():
    """
    ModelBase instances should be effectively immutable for callers.
    """

    class TestModel(ModelBase):
        field: int = 1

    m = TestModel()
    assert m.field == 1
    assert_immutable(m, "field", 2)


def test_modelbase_blake3_hash_thread_safety():
    """
    Calling blake3_hash concurrently on the same instance should be race-free
    and consistent (all threads see the same cached value).
    """

    class TestModel(ModelBase):
        field: int = 1

    m = TestModel()

    def worker() -> str:
        return m.blake3_hash()

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as ex:
        results = list(ex.map(lambda _: worker(), range(128)))

    assert len(set(results)) == 1


def test_modelbase_hash_exclude_fields_subclass_safety():
    """
    HASH_EXCLUDE_FIELDS must behave correctly across subclass hierarchies.

    Parent excludes {"a"}; Child excludes {"a", "b"}.
    The hash should ignore the union specified by the most-derived class.

    IMPORTANT:
    We fix updated_at to a shared value so that only the intended fields
    (a, b) differ between instances. Otherwise, default timestamps would
    cause hash differences unrelated to HASH_EXCLUDE_FIELDS semantics.
    """

    class Parent(ModelBase):
        HASH_EXCLUDE_FIELDS = {"a"}
        a: int = 1
        b: int = 2

    class Child(Parent):
        HASH_EXCLUDE_FIELDS = {"a", "b"}

    shared_ts = 1234.5

    # In Parent, 'a' is excluded, 'b' is included.
    p1 = Parent(a=1, b=2, updated_at=shared_ts)
    p2 = Parent(a=999, b=2, updated_at=shared_ts)  # 'a' differs, but excluded
    assert p1.blake3_hash() == p2.blake3_hash()

    # In Child, both 'a' and 'b' are excluded.
    c1 = Child(a=1, b=2, updated_at=shared_ts)
    c2 = Child(a=999, b=999, updated_at=shared_ts)  # 'a' and 'b' differ, both excluded
    assert c1.blake3_hash() == c2.blake3_hash()


def test_modelbase_computed_fields_auto_excluded_from_hash():
    """
    @computed_field properties must not affect the hash.

    We check that:

    - computed fields appear in model_dump() by default, but
    - ModelBase.blake3_hash ignores them under the hood, so explicitly
      excluding them is a no-op for the hash.
    """

    class CompModel(ModelBase):
        base: int = 1

        @computed_field
        @property
        def double(self) -> int:
            return self.base * 2

    m = CompModel(base=1)

    # Sanity: computed field is part of the normal JSON view
    dump = m.model_dump(mode="json")
    assert "double" in dump
    assert dump["double"] == 2

    # Default hash vs. explicitly excluding the computed field
    h_default = m.blake3_hash(use_cache=False)
    h_exclude_double = m.blake3_hash(exclude=["double"], use_cache=False)

    # If computed fields are auto-excluded by the hashing logic,
    # these must be identical.
    assert h_default == h_exclude_double


# ---------------------------------------------------------------------
# Avro / Protobuf / OpenAPI
# ---------------------------------------------------------------------


def test_modelbase_avro_record_json_safe():
    """
    to_avro_record should return a JSON-serializable dict that includes
    schema_version and all scalar fields.
    """

    class TestModel(ModelBase):
        field: int = 42

    m = TestModel()
    rec = m.to_avro_record()

    assert rec["field"] == 42
    assert rec["schema_version"] == "4.2"

    # Must be JSON-serializable
    json_str = json.dumps(rec)
    assert isinstance(json_str, str)


@pytest.mark.skipif(not AVRO_AVAILABLE, reason="fastavro not available")
def test_modelbase_avro_bytes_roundtrip_nonempty():
    """
    to_avro_bytes should emit a non-empty Avro record compatible with the
    provided schema.
    """

    class TestModel(ModelBase):
        field: int = 42

    m = TestModel()
    schema = {
        "type": "record",
        "name": "TestModel",
        "fields": [
            {"name": "schema_version", "type": "string"},
            {"name": "creator_id", "type": ["null", "string"]},
            {"name": "updated_at", "type": "double"},
            {"name": "field", "type": "int"},
        ],
    }

    b = m.to_avro_bytes(schema)
    assert isinstance(b, (bytes, bytearray))
    assert len(b) > 0


@pytest.mark.skipif(not PROTO_AVAILABLE, reason="protobuf not available")
def test_modelbase_protobuf_basic_fields_preserved():
    """
    to_protobuf must produce a Message with scalar fields preserved.
    """
    from google.protobuf.message import Message as PBMessage  # type: ignore[import]

    class TestModel(ModelBase):
        field: int = 42
        extra_field: str = "hello"

    m = TestModel()
    msg = m.to_protobuf()

    assert isinstance(msg, PBMessage)
    assert getattr(msg, "field") == 42
    assert getattr(msg, "extra_field") == "hello"
    assert getattr(msg, "schema_version") == "4.2"


@pytest.mark.skipif(not PROTO_AVAILABLE, reason="protobuf not available")
def test_modelbase_protobuf_strict_does_not_raise_for_scalar_model():
    """
    When all fields are scalar, strict=True should not raise,
    because no fields need to be dropped during dynamic schema creation.
    """

    class TestModel(ModelBase):
        field: int = 1

    m = TestModel()
    # Should not raise, even with strict=True
    m.to_protobuf(strict=True)


def test_modelbase_openapi_schema_includes_field_description():
    """
    openapi_schema should include descriptions passed via Field().
    """

    class TestModel(ModelBase):
        field: int = Field(description="test field")

    schema = TestModel.openapi_schema()
    assert schema["title"] == "TestModel"
    assert "field" in schema["properties"]
    assert schema["properties"]["field"]["description"] == "test field"


# ---------------------------------------------------------------------
# migrate() behaviour
# ---------------------------------------------------------------------

from cc.core.schema import SCHEMA_VERSION

@given(st.dictionaries(st.text(min_size=1, max_size=10), st.text()))
def test_modelbase_migrate_best_effort(old_data: Dict[str, str]):
    """
    migrate() should be best-effort and never throw on arbitrary old dicts.

    We don't assert exact semantics; we only require that:
    - an instance is created
    - schema_version falls back to current default (for missing key)
    """

    class TestModel(ModelBase):
        optional_field: Optional[str] = None

    # Ensure schema_version is not present to exercise defaulting behaviour
    old_data = {k: v for k, v in old_data.items() if k != "schema_version"}

    m = TestModel.migrate(old_data)
    assert isinstance(m, TestModel)
    assert m.schema_version == "4.2"
    # The optional field may or may not be populated, depending on keys.


def test_modelbase_migrate_preserves_known_fields_ignores_extra():
    """
    migrate() should map known fields through and silently ignore extras,
    respecting the model_config(extra="ignore") behaviour.
    """

    class TestModel(ModelBase):
        optional_field: Optional[str] = None

    old_data = {
        "optional_field": "hello",
        "some_extra": "ignore-me",
        "schema_version": "0.1",  # should be overridden by current default
    }

    m = TestModel.migrate(old_data)
    assert isinstance(m, TestModel)
    assert m.optional_field == "hello"
    assert m.schema_version == "4.2"


# ---------------------------------------------------------------------
# SQLAlchemy AuditColumnsMixin / OrmBase wiring
# ---------------------------------------------------------------------


@pytest.mark.skipif(not SQLALCHEMY_AVAILABLE, reason="SQLAlchemy not available")
def test_audit_columns_mixin_schema():
    """
    AuditColumnsMixin + OrmBase must attach creator_id and updated_at columns
    with the expected SQLAlchemy types.
    """
    from sqlalchemy import Column, DateTime, Integer, String  # type: ignore[import]

    class TestORM(OrmBase, AuditColumnsMixin):  # type: ignore[misc]
        __tablename__ = "test_table"
        id = Column(Integer, primary_key=True)

    assert hasattr(TestORM, "creator_id")
    assert isinstance(TestORM.creator_id.type, String)  # type: ignore[attr-defined]

    assert hasattr(TestORM, "updated_at")
    assert isinstance(TestORM.updated_at.type, DateTime)  # type: ignore[attr-defined]
