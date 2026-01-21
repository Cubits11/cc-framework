from __future__ import annotations

"""
Module: core.models (data layer for CC framework)
Purpose: Strongly-typed, versioned, and serializable data models used across the
         two-world protocol, calibration, analysis, and reporting.

Author: Pranav Bhave
Schema versions:
  - 2025-08-31: original
  - 2025-09-28: v3.x - validation, slots, hashing helpers, immutability
  - 2025-11-13: v4.0 - Pydantic v2, enums, BLAKE3, basic interop hooks
  - 2025-11-19: v4.1 - hardened invariants, unified timestamp handling,
                safer hashing, stricter config normalization, lossy
                Protobuf export explicitly documented, back-compat alias
                for AttackStrategySpec.
  - 2025-11-24: v4.2 - tighten migration semantics, centralize invariants,
                clarify hashing surface, stabilize dynamic Protobuf
                reflection, and normalize bootstrap cleaning semantics.

Design notes
------------
This module is the "data spine" for the probabilistic core:

- Pydantic BaseModel (v2) for runtime validation, serialization, schemas.
- Frozen/immutable instances for safety in evaluations (no in-place mutation).
- Enums for fixed values (WorldBit, CiMethod, RiskLevel).
- BLAKE3 hashing for integrity and deduplication, with:
    - Stable JSON canonicalization.
    - Optional salt for per-run/domain separation.
    - Per-class hash-exclusion sets for derived / non-semantic fields.
- Schema versioning via `schema_version` and a migration entrypoint.
- Optional, best-effort interop helpers:
    - Avro:
        * `to_avro_record()` - JSON-safe dict for external writers.
        * `to_avro_bytes(schema)` - schemaless write if `fastavro` is installed.
    - Protobuf:
        * Either hydrate a provided Message subclass, or
        * Build a LOSSY dynamic scalar-only message type (documented below).

Out of scope for this file (lives elsewhere):
- ORM models and full DB integration details (this module only defines
  SQLAlchemy mixins / base type if installed).
- Higher-level stats (J/CC computation), experiment engines, analysis code.

Dependencies:
- Required: pydantic >= 2.0, blake3
- Optional: numpy (for CCResult.bootstrap_samples),
            sqlalchemy (for ORM mixins),
            fastavro (for Avro bytes),
            google.protobuf (for Protobuf export).

Important safety notes
----------------------
- We explicitly disallow NaN/inf in core numeric fields (J, CC, deltas,
  utility scores, timestamps).
- We normalize timestamps defensively to avoid platform-dependent overflows.
- Dynamic Protobuf export is intentionally LOSSY:
    * Only scalar and repeated-scalar fields are included.
    * Nested models / mappings are skipped.
    * Unknown fields in the JSON are ignored during ParseDict.
  Use JSON/Avro for canonical archival; use Protobuf only for lightweight
  integration with scalar subsets.
"""

import importlib.util
import io
import json
import math
import threading
import time
import unicodedata
import uuid
import warnings
from collections.abc import Mapping, Sequence
from datetime import datetime
from enum import Enum, IntEnum
from functools import cached_property
from typing import (
    Any,
    ClassVar,
    TypeVar,
    Union,
    get_args,
    get_origin,
)

if importlib.util.find_spec("blake3") is None:
    import hashlib

    class _Blake3Fallback:
        def __init__(self) -> None:
            self._hasher = hashlib.blake2b(digest_size=32)

        def update(self, data: bytes) -> None:
            self._hasher.update(data)

        def hexdigest(self) -> str:
            return self._hasher.hexdigest()

    class _Blake3Module:
        @staticmethod
        def blake3() -> _Blake3Fallback:
            return _Blake3Fallback()

    blake3 = _Blake3Module()
else:
    import blake3  # type: ignore[no-redef]
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    computed_field,
    field_serializer,
    field_validator,
)

# ---------------------------------------------------------------------------
# Global schema & type aliases
# ---------------------------------------------------------------------------
from cc.core.schema import SCHEMA_VERSION as _SCHEMA_VERSION

JsonDict = dict[str, Any]
FloatSeq = Sequence[float]
TModel = TypeVar("TModel", bound="ModelBase")

# Magic-number constants
BLAKE3_HEX_LENGTH: int = 64  # BLAKE3 hex digest length
REQUEST_ID_LENGTH: int = 12  # Short, human-ish request id
MAX_REASONABLE_UNIX_TIMESTAMP: float = 1e12  # ~year 33658, avoids overflow


# ---------------------------------------------------------------------------
# Optional deps
# ---------------------------------------------------------------------------

try:  # numpy (optional)
    import numpy as np  # type: ignore[import]

    NUMPY_AVAILABLE = True
except ImportError:  # pragma: no cover
    NUMPY_AVAILABLE = False
    np = None  # type: ignore[assignment]

try:  # SQLAlchemy (optional)
    from sqlalchemy import Column, DateTime, Integer, String  # type: ignore[import]
    from sqlalchemy.orm import DeclarativeBase, declared_attr  # type: ignore[import]

    SQLALCHEMY_AVAILABLE = True
except ImportError:  # pragma: no cover
    SQLALCHEMY_AVAILABLE = False
    DeclarativeBase = object  # type: ignore[assignment]

try:  # Avro (optional)
    import fastavro  # type: ignore[import]

    AVRO_AVAILABLE = True
except ImportError:  # pragma: no cover
    AVRO_AVAILABLE = False

try:  # Protobuf (optional)
    from google.protobuf.json_format import ParseDict  # type: ignore[import]
    from google.protobuf.message import Message  # type: ignore[import]

    PROTO_AVAILABLE = True
except ImportError:  # pragma: no cover
    PROTO_AVAILABLE = False
    Message = object  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Time & hashing helpers
# ---------------------------------------------------------------------------


def _now_unix() -> float:
    """Unix timestamp in seconds (UTC)."""
    return float(time.time())


def _normalize_unix_timestamp(
    v: Any,
    *,
    allow_none: bool = True,
    max_ts: float | None = MAX_REASONABLE_UNIX_TIMESTAMP,
) -> float:
    """
    Normalize a value into a sane Unix timestamp (UTC seconds).

    Philosophy
    ----------
    - Strict on bad *values* (type, NaN/inf, absurdly large).
    - Mildly forgiving on None when allow_none=True (snap to "now").
    - 0 and negative timestamps are allowed (epoch and pre-epoch logs).

    Rules:
    - If v is None:
        * allow_none=True  -> use "now"
        * allow_none=False -> raise (timestamp required)
    - Else: must be numeric, finite, and <= max_ts (if provided).
    """
    if v is None:
        if allow_none:
            return _now_unix()
        raise ValueError("timestamp is required and cannot be None")

    try:
        f = float(v)
    except (TypeError, ValueError):
        raise ValueError(f"timestamp must be numeric unix seconds (got {type(v).__name__}: {v!r})")

    if not math.isfinite(f):
        raise ValueError(f"timestamp must be finite (got {f!r})")

    if max_ts is not None and f > max_ts:
        raise ValueError(f"timestamp {f!r} exceeds MAX_REASONABLE_UNIX_TIMESTAMP={max_ts!r}")

    return f


def _iso_from_unix(ts: float) -> str:
    """
    ISO-8601 in UTC with millisecond precision, suffixed with 'Z'.

    Assumes `ts` is a sane Unix timestamp; upstream validators ensure this does
    not overflow time.gmtime on supported platforms.
    """
    tm = time.gmtime(ts)
    ms = int((ts - int(ts)) * 1000)
    return (
        f"{tm.tm_year:04d}-{tm.tm_mon:02d}-{tm.tm_mday:02d}"
        f"T{tm.tm_hour:02d}:{tm.tm_min:02d}:{tm.tm_sec:02d}.{ms:03d}Z"
    )


def _normalize_unicode_for_hash(obj: Any) -> Any:
    """
    Recursively normalize all text to NFC for hashing stability.

    Handles:
    - dicts: normalize both keys and values if they are str
    - lists/tuples: normalize elements
    - str: normalized with unicodedata.normalize("NFC", s)
    All other JSON-safe primitives are returned unchanged.
    """
    # Normalize plain strings
    if isinstance(obj, str):
        return unicodedata.normalize("NFC", obj)

    # Normalize dict keys and values
    if isinstance(obj, dict):
        return {
            _normalize_unicode_for_hash(k): _normalize_unicode_for_hash(v) for k, v in obj.items()
        }

    # Normalize lists
    if isinstance(obj, list):
        return [_normalize_unicode_for_hash(v) for v in obj]

    # Normalize tuples
    if isinstance(obj, tuple):
        return tuple(_normalize_unicode_for_hash(v) for v in obj)

    # Everything else (numbers, bools, None) we leave as-is
    return obj


def _hash_json(obj: Any, *, salt: bytes | None = None) -> str:
    """
    Canonical JSON -> BLAKE3 hex digest.

    Invariants
    ----------
    - Keys are sorted and separators are compact ("," and ":").
    - All Unicode text is normalized to NFC so that composed vs decomposed
      forms hash identically (e.g. "café" NFC vs NFD).
    - An optional salt can be used to separate hash namespaces.

    Parameters
    ----------
    obj:
        Any JSON-serializable object.
    salt:
        Optional bytes to prepend into the hash state (for adversarial resistance).
    """
    # Normalize Unicode so NFC and NFD variants collapse to the same representation
    normalized = _normalize_unicode_for_hash(obj)

    # Canonical JSON: sorted keys, compact separators
    data = json.dumps(normalized, sort_keys=True, separators=(",", ":"))

    hasher = blake3.blake3()
    if salt is not None:
        hasher.update(salt)
    hasher.update(data.encode("utf-8"))
    return hasher.hexdigest()


def _hash_text(text: str | bytes, *, salt: bytes | None = None) -> str:
    """
    Hash raw transcript text/bytes with BLAKE3.

    Parameters
    ----------
    text:
        Input text or bytes.
    salt:
        Optional bytes to prepend into the hash state.
    """
    data = text if isinstance(text, bytes) else str(text).encode("utf-8", errors="replace")
    hasher = blake3.blake3()
    if salt is not None:
        hasher.update(salt)
    hasher.update(data)
    return hasher.hexdigest()


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class WorldBit(IntEnum):
    """World indicator for the two-world protocol."""

    BASELINE = 0
    PROTECTED = 1


class CiMethod(str, Enum):
    """Methods for constructing confidence intervals."""

    BOOTSTRAP = "bootstrap"
    WILSON = "wilson"
    BAYES = "bayes"  # reserved for future Bayesian CIs


class RiskLevel(str, Enum):
    """Coarse qualitative risk tags for guardrail specs."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# ---------------------------------------------------------------------------
# SQLAlchemy helpers (optional, separate from Pydantic models)
# ---------------------------------------------------------------------------

if SQLALCHEMY_AVAILABLE:

    class OrmBase(DeclarativeBase):
        """Base for SQLAlchemy ORM models in the CC framework."""

    class AuditColumnsMixin:
        """
        Audit columns mixin for SQLAlchemy ORM models.

        Typical usage (in a separate ORM module):

            class AttackResultORM(OrmBase, AuditColumnsMixin):
                __tablename__ = "attack_results"
                id = Column(Integer, primary_key=True)
                ...

        These columns are intentionally generic; mapping back-and-forth
        between ORM rows and Pydantic models should be handled externally.
        """

        __allow_unmapped__ = True

        @declared_attr
        def creator_id(cls):  # type: ignore[no-untyped-def]
            return Column(String(64), nullable=True)

        @declared_attr
        def updated_at(cls):  # type: ignore[no-untyped-def]
            return Column(
                DateTime(timezone=True),
                default=datetime.utcnow,
                onupdate=datetime.utcnow,
                nullable=False,
            )

else:  # pragma: no cover

    class OrmBase:  # type: ignore[too-many-ancestors]
        """Fallback stub when SQLAlchemy is not installed."""

    class AuditColumnsMixin:  # type: ignore[too-many-ancestors]
        """Fallback stub when SQLAlchemy is not installed."""


# ---------------------------------------------------------------------------
# Base Pydantic model
# ---------------------------------------------------------------------------


class ModelBase(BaseModel):
    """
    Shared BaseModel config for all CC core models.

    Features
    --------
    - Frozen/immutable instances.
    - Extra fields ignored on decode (backwards-compatible).
    - schema_version and creator_id for audit / lineage.
    - updated_at as a snapshot timestamp.
    - Stable BLAKE3 hashing with per-class exclusion sets.

    Hashing semantics
    -----------------
    - All models automatically exclude:
        * manual HASH_EXCLUDE_FIELDS, and
        * ALL @computed_field properties
      from the hash input.
    - Passing `exclude=[...]` to blake3_hash() adds *extra* exclusions on top.
    - This guarantees that adding a computed field cannot silently change
      deduplication hashes.

    Protobuf semantics
    ------------------
    - Dynamic Protobuf export is intentionally LOSSY and scalar-only.
      For full fidelity, prefer JSON/Avro.
    - In dynamic mode (no proto_cls supplied):
        * Only scalar and repeated-scalar fields are included.
        * Nested models / mappings are skipped.
        * The set of dropped fields is cached per (model class, schema_version),
          and `strict=True` will reliably raise if any were dropped.
    """

    schema_version: str = Field(default=_SCHEMA_VERSION, frozen=True)
    creator_id: str | None = Field(
        default=None,
        description="Optional id of human/system that created this record.",
    )
    updated_at: float = Field(
        default_factory=_now_unix,
        description="Unix timestamp (UTC seconds) when this snapshot was last updated.",
    )

    model_config = ConfigDict(
        frozen=True,
        extra="ignore",
        validate_assignment=False,
        populate_by_name=True,
        arbitrary_types_allowed=True,
        ser_json_timedelta="iso8601",
        ser_json_bytes="utf8",
    )

    # ------------------------------------------------------------------
    # Protobuf message cache with lock
    #  - Keyed by (model class, schema_version)
    #  - Value is (Message subclass, frozenset of dropped field names)
    #    so `strict=True` behaviour is consistent across calls.
    # ------------------------------------------------------------------
    _PROTO_MESSAGE_CACHE: ClassVar[
        dict[tuple[type[ModelBase], str], tuple[type[Message], frozenset[str]]]
    ] = {}
    _PROTO_MESSAGE_CACHE_LOCK: ClassVar[threading.Lock] = threading.Lock()

    # Manual hash-exclusion set; computed fields are handled at hash-time
    HASH_EXCLUDE_FIELDS: ClassVar[frozenset[str]] = frozenset()

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """
        Normalize HASH_EXCLUDE_FIELDS for subclasses.

        Computed fields (@computed_field) are not baked into HASH_EXCLUDE_FIELDS
        here because Pydantic wires them up after __init_subclass__. Instead,
        the hash helpers inspect `cls.model_computed_fields` at call-time.

        NOTE
        ----
        Subclasses override (not union) HASH_EXCLUDE_FIELDS; if you want to
        extend the parent set you should write, e.g.:

            class Child(Parent):
                HASH_EXCLUDE_FIELDS = Parent.HASH_EXCLUDE_FIELDS | {"extra_field"}
        """
        super().__init_subclass__(**kwargs)
        manual = getattr(cls, "HASH_EXCLUDE_FIELDS", frozenset())
        if not isinstance(manual, frozenset):
            manual = frozenset(manual)
        cls.HASH_EXCLUDE_FIELDS = manual

    # ------------------------------------------------------------------
    # Core timestamp normalization
    # ------------------------------------------------------------------

    @field_validator("updated_at", mode="before")
    @classmethod
    def _normalize_updated_at(cls, v: Any) -> float:
        # snapshot-style: None -> now, bad types / absurdly large -> error
        return _normalize_unix_timestamp(v, allow_none=True)

    # ------------------------------------------------------------------
    # Hash helpers with computed-field auto-exclusion
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Backward-compatible dict serialization helper."""
        return self.model_dump()

    def _hash_exclude_keys(self, extra: Sequence[str] | None = None) -> set[str]:
        """
        Compute the full set of keys to exclude from hashing:

            manual HASH_EXCLUDE_FIELDS
          U names of @computed_field properties
          U any additional `extra` keys supplied by caller.

        This ensures that all computed fields are always ignored by the hash,
        so adding/removing a computed field never changes hashes.
        """
        cls = self.__class__
        exclude: set[str] = set(extra or [])

        # Manual excludes (class-level)
        exclude |= set(getattr(cls, "HASH_EXCLUDE_FIELDS", frozenset()))

        # Auto-exclude computed fields from Pydantic (if present)
        computed = getattr(cls, "model_computed_fields", None)
        if computed:
            exclude |= set(computed.keys())

        return exclude

    @cached_property
    def _default_hash(self) -> str:
        """
        Cached BLAKE3 hash of the JSON representation:

        - Excludes manual HASH_EXCLUDE_FIELDS
        - Excludes all @computed_field properties
        - No salt

        This is used when blake3_hash() is called with all default knobs
        (exclude=None, use_cache=True, salt=None).
        """
        data = self.model_dump(mode="json", exclude=self._hash_exclude_keys())
        return _hash_json(data)

    def blake3_hash(
        self,
        *,
        exclude: Sequence[str] | None = None,
        use_cache: bool = True,
        salt: bytes | None = None,
    ) -> str:
        """
        Stable BLAKE3 hash of the JSON representation.

        Parameters
        ----------
        exclude:
            Optional extra field names to exclude from hashing, in addition to
            HASH_EXCLUDE_FIELDS and all @computed_field properties.
        use_cache:
            When True (default) and when `exclude is None` and `salt is None`,
            use the cached `_default_hash`. Any non-default knob forces a fresh
            hash computation.
        salt:
            Optional bytes to prepend into the hash state (for per-run/domain
            separation or adversarial hardening).

        SECURITY / COLLISION NOTE
        -------------------------
        - BLAKE3 is collision-resistant for practical purposes, but not magic.
        - If you use hashes as primary keys in a large store, you SHOULD:
            * store the full object keyed by hash, and
            * on insert, check for hash collisions by comparing payloads.
        """
        # Only use the cached default hash when *all* knobs are at defaults.
        if exclude is None and use_cache and salt is None:
            return self._default_hash

        # For any custom call, still union:
        #   HASH_EXCLUDE_FIELDS U model_computed_fields U exclude
        exclude_set = self._hash_exclude_keys(exclude)
        data = self.model_dump(mode="json", exclude=exclude_set)
        return _hash_json(data, salt=salt)

    # ------------------------------------------------------------------
    # Avro helpers
    # ------------------------------------------------------------------

    def to_avro_record(self) -> JsonDict:
        """
        JSON-serializable dict suitable for Avro writers.

        This is intentionally just the JSON view of the model; any Avro schema
        evolution must be handled by Avro tooling, not this helper.
        """
        return self.model_dump(mode="json")

    def to_avro_bytes(self, schema: Mapping[str, Any]) -> bytes:
        """
        Serialize this model to Avro binary using a provided schema.

        Requires fastavro to be installed.
        """
        if not AVRO_AVAILABLE:  # pragma: no cover
            raise ImportError("fastavro required for Avro export")
        rec = self.to_avro_record()
        buf = io.BytesIO()
        fastavro.schemaless_writer(buf, schema, rec)  # type: ignore[call-arg]
        return buf.getvalue()

    # ------------------------------------------------------------------
    # Protobuf helpers (LOSSY dynamic export)
    # ------------------------------------------------------------------

    def to_protobuf(
        self,
        proto_cls: type[Message] | None = None,
        *,
        strict: bool = False,
    ) -> Message:
        """
        Export to a protobuf message instance.

        Parameters
        ----------
        proto_cls:
            Optional protobuf Message subclass whose field names match this model.
            If provided, we hydrate that type via ParseDict (unknown fields ignored)
            and do not perform any lossy-detection logic.
        strict:
            When False (default), lossy dynamic export emits a warning if fields
            are skipped. When True, lossy export raises ValueError listing the
            dropped fields.

        Dynamic type behavior (LOSSY)
        -----------------------------
        - Only basic scalar fields are mapped:
            * int / IntEnum  -> int32
            * float          -> double
            * bool           -> bool
            * str / Enum     -> string
        - Optionals are unwrapped to their inner type.
        - Sequence[T] becomes repeated scalar if T is scalar.
        - Mappings, nested models, and arbitrary objects are SKIPPED.

        Use JSON/Avro for canonical archival; use Protobuf only for lightweight
        integration with scalar subsets.
        """
        if not PROTO_AVAILABLE:  # pragma: no cover
            raise ImportError("google.protobuf required for protobuf export")

        # Local imports avoid hard dependency when PROTO_AVAILABLE is False
        from google.protobuf import (  # type: ignore[import]
            descriptor_pb2,
            descriptor_pool,
            message_factory,
        )

        data = self.model_dump(mode="json")

        # If explicit proto type given, just hydrate; strict applies only to dynamic mode.
        if proto_cls is not None:
            msg = proto_cls()  # type: ignore[call-arg]
            ParseDict(data, msg, ignore_unknown_fields=True)
            return msg

        cls: type[ModelBase] = self.__class__
        cache_key = (cls, _SCHEMA_VERSION)

        cached = ModelBase._PROTO_MESSAGE_CACHE.get(cache_key)
        if cached is None:
            with ModelBase._PROTO_MESSAGE_CACHE_LOCK:
                cached = ModelBase._PROTO_MESSAGE_CACHE.get(cache_key)
                if cached is None:
                    dropped_fields: list[str] = []

                    fd_proto = descriptor_pb2.FileDescriptorProto()
                    fd_proto.name = f"{cls.__module__}.{cls.__name__}.dynamic.proto"
                    fd_proto.package = cls.__module__

                    msg_proto = descriptor_pb2.DescriptorProto()
                    msg_proto.name = cls.__name__

                    def _unwrap_optional(t: Any) -> Any:
                        origin = get_origin(t)
                        if origin is Union:
                            args = get_args(t)
                            non_none = [a for a in args if a is not type(None)]
                            if len(non_none) == 1:
                                return _unwrap_optional(non_none[0])
                        return t

                    def _scalar_pb_type(t: Any) -> int | None:
                        if isinstance(t, type) and issubclass(t, IntEnum):
                            return descriptor_pb2.FieldDescriptorProto.TYPE_INT32  # type: ignore[attr-defined]
                        if isinstance(t, type) and issubclass(t, Enum):
                            return descriptor_pb2.FieldDescriptorProto.TYPE_STRING  # type: ignore[attr-defined]
                        if t is int:
                            return descriptor_pb2.FieldDescriptorProto.TYPE_INT32  # type: ignore[attr-defined]
                        if t is float:
                            return descriptor_pb2.FieldDescriptorProto.TYPE_DOUBLE  # type: ignore[attr-defined]
                        if t is bool:
                            return descriptor_pb2.FieldDescriptorProto.TYPE_BOOL  # type: ignore[attr-defined]
                        if t is str:
                            return descriptor_pb2.FieldDescriptorProto.TYPE_STRING  # type: ignore[attr-defined]
                        return None

                    field_number = 1
                    for field_name, field_info in cls.model_fields.items():
                        py_type = field_info.annotation
                        if py_type is Any or py_type is None:
                            dropped_fields.append(field_name)
                            continue

                        core_type = _unwrap_optional(py_type)
                        origin = get_origin(core_type)
                        label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL  # type: ignore[attr-defined]
                        pb_type: int | None = None

                        if origin in (list, list, Sequence, tuple, tuple):
                            args = get_args(core_type)
                            if not args:
                                dropped_fields.append(field_name)
                                continue
                            elem_core = _unwrap_optional(args[0])
                            scalar_type = _scalar_pb_type(elem_core)
                            if scalar_type is None:
                                dropped_fields.append(field_name)
                                continue
                            pb_type = scalar_type
                            label = descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED  # type: ignore[attr-defined]
                        else:
                            scalar_type = _scalar_pb_type(core_type)
                            if scalar_type is None:
                                dropped_fields.append(field_name)
                                continue
                            pb_type = scalar_type

                        field_proto = msg_proto.field.add()
                        field_proto.name = field_name
                        field_proto.number = field_number
                        field_proto.type = pb_type
                        field_proto.label = label
                        field_number += 1

                    fd_proto.message_type.add().CopyFrom(msg_proto)
                    pool = descriptor_pool.DescriptorPool()
                    fd = pool.AddSerializedFile(fd_proto.SerializeToString())
                    factory = message_factory.MessageFactory(pool=pool)
                    msg_cls = factory.GetPrototype(fd.message_types_by_name[cls.__name__])

                    # Cache both the Message class and the dropped fields.
                    ModelBase._PROTO_MESSAGE_CACHE[cache_key] = (
                        msg_cls,
                        frozenset(dropped_fields),
                    )
                    cached = ModelBase._PROTO_MESSAGE_CACHE[cache_key]

        msg_cls, dropped_fields_cached = cached

        if strict and dropped_fields_cached:
            raise ValueError(
                f"Protobuf export from {cls.__name__} is lossy; "
                f"dropped fields: {sorted(dropped_fields_cached)}"
            )
        elif dropped_fields_cached:
            warnings.warn(
                f"Protobuf export from {cls.__name__} skipped non-scalar fields: "
                f"{sorted(dropped_fields_cached)}",
                UserWarning,
                stacklevel=2,
            )

        msg = msg_cls()  # type: ignore[call-arg]
        ParseDict(data, msg, ignore_unknown_fields=True)
        return msg

    # ------------------------------------------------------------------
    # Schema & migration helpers
    # ------------------------------------------------------------------

    @classmethod
    def openapi_schema(cls) -> JsonDict:
        """Return the OpenAPI-compatible JSON schema for this model type."""
        return cls.model_json_schema()

    @classmethod
    def migrate(cls: type[TModel], old_data: Mapping[str, Any]) -> TModel:
        """
        Best-effort migration entrypoint for schema upgrades.

        Default behavior:
        - Accept arbitrary mapping (e.g. legacy JSON payload).
        - Extra keys ignored per model_config.
        - Any embedded `schema_version` is discarded in favour of current.

        Override in subclasses when:
        - You introduce breaking schema changes.
        - You need to inspect old_data["schema_version"] and adapt fields.
        """
        clean = dict(old_data)
        clean.pop("schema_version", None)
        return cls.model_validate(clean)


# ---------------------------------------------------------------------------
# AttackResult
# ---------------------------------------------------------------------------


class AttackResult(ModelBase):
    """
    Result of a single attack session in the two-world protocol.

    Semantics
    ---------
    - world_bit:
        * 0 = baseline (WorldBit.BASELINE)
        * 1 = guardrail-enabled (WorldBit.PROTECTED)

    - success (STRICT BOOL):
        * Typically, "harmful output passed guardrail" (or analogous binary
          outcome). The exact semantics must be consistent with how J is
          computed in the stats layer.
        * This field is a StrictBool:
            - Accepts only True / False.
            - Rejects strings like "true"/"false", integers, etc.
          This is enforced so API misuse is caught early instead of silently
          coerced.

    - transcript_hash:
        * BLAKE3 hash of the full transcript (not stored inline).

    - guardrails_applied:
        * Human-readable label of the guardrail config / stack applied.
          For reproducibility, prefer to correlate this with a stable config
          key elsewhere (e.g., ExperimentConfig.guardrail_configs).

    Hashing / identity
    ------------------
    - `iso_time` is a computed field and is excluded from hashes:
        * via HASH_EXCLUDE_FIELDS, and
        * via ModelBase's automatic exclusion of @computed_field members.
    - For deduplication / integrity checks, use `blake3_hash()` or
      the compatibility alias `model_hash()`.
    """

    # iso_time is computed; we explicitly exclude it from hashes.
    # (ModelBase also auto-excludes all @computed_field properties.)
    HASH_EXCLUDE_FIELDS: ClassVar[frozenset[str]] = frozenset({"iso_time"})

    world_bit: WorldBit = Field(
        description="0 (baseline) or 1 (guardrail-enabled).",
    )
    # STRICT BOOL semantics: only True/False allowed; no coercion from strings/ints.
    success: StrictBool = Field(
        description=(
            "Did the attack succeed according to the experiment's semantics. "
            "StrictBool: only True/False accepted; strings like 'true' are rejected."
        ),
    )
    attack_id: str
    transcript_hash: str
    guardrails_applied: str
    rng_seed: int
    timestamp: float = Field(
        default_factory=_now_unix,
        description="Unix timestamp in seconds (UTC) for when the attack was evaluated.",
    )
    # --- optional / metadata
    session_id: str = ""
    attack_strategy: str = ""
    utility_score: float | None = None
    request_id: str = Field(
        default_factory=lambda: uuid.uuid4().hex[:REQUEST_ID_LENGTH],
        description=(f"Short request identifier (first {REQUEST_ID_LENGTH} hex chars of uuid4)."),
    )

    # -----------------------
    # Field validators
    # -----------------------

    @field_validator("transcript_hash", mode="before")
    @classmethod
    def _validate_transcript_hash(cls, v: Any) -> str:
        """
        transcript_hash must be a full BLAKE3 hex digest.

        Invariants:
        - Required (cannot be None).
        - Exactly BLAKE3_HEX_LENGTH hex characters.
        - Normalized to lowercase for stability.
        """
        if v is None:
            raise ValueError("transcript_hash is required")
        s = str(v).strip()
        if len(s) != BLAKE3_HEX_LENGTH:
            if s.startswith("hash-") and s[5:].isdigit():
                return _hash_text(s)
            raise ValueError(
                f"transcript_hash must be a {BLAKE3_HEX_LENGTH}-character hex string "
                f"(got length {len(s)}: {s!r})"
            )
        try:
            int(s, 16)
        except ValueError as exc:  # pragma: no cover - defensive
            raise ValueError("transcript_hash must be hex-encoded") from exc
        return s.lower()

    @field_validator("utility_score", mode="before")
    @classmethod
    def _normalize_utility(cls, v: Any) -> float | None:
        """
        Normalize optional utility_score.

        Rules:
        - None / "" → None
        - Else: must be numeric and finite.
        """
        if v is None or v == "":
            return None
        try:
            f = float(v)
        except (TypeError, ValueError):
            raise ValueError(
                f"utility_score must be numeric or None (got {type(v).__name__}: {v!r})"
            )
        if math.isnan(f) or math.isinf(f):
            raise ValueError("utility_score cannot be NaN or infinite")
        return f

    @field_validator("timestamp", mode="before")
    @classmethod
    def _normalize_timestamp(cls, v: Any) -> float:
        """
        Timestamp normalization.

        - None → now (snapshot-style, via allow_none=True).
        - Else: must be numeric, finite, and not absurdly large.
        """
        return _normalize_unix_timestamp(v, allow_none=True)

    @field_validator("rng_seed", mode="before")
    @classmethod
    def _normalize_seed(cls, v: Any) -> int:
        """
        rng_seed must be an integer.

        We accept things that cleanly cast to int, but reject type errors
        so configs like "abc" fail fast.
        """
        try:
            iv = int(v)
        except (TypeError, ValueError):
            raise ValueError(f"rng_seed must be an integer (got {type(v).__name__}: {v!r})")
        return iv

    @field_validator(
        "attack_id",
        "guardrails_applied",
        "session_id",
        "attack_strategy",
        mode="before",
    )
    @classmethod
    def _coerce_str(cls, v: Any) -> str:
        """
        Normalize metadata-ish string fields.

        For AttackResult, these are *metadata*:
        - None is normalized to "" (empty string).
        - Other types are stringified.

        Contrast with ExperimentConfig.attack_strategies, which are required
        and thus reject None. That difference is intentional: session-level
        metadata is optional; experiment-level config is not.
        """
        if v is None:
            return ""
        return str(v)

    # -----------------------
    # Computed properties
    # -----------------------

    @computed_field
    @property
    def iso_time(self) -> str:
        """
        ISO-8601 representation of `timestamp` in UTC.

        This is a *view* field:
        - Included in `model_dump()` by default.
        - Automatically excluded from hashes via ModelBase.
        """
        return _iso_from_unix(self.timestamp)

    # -----------------------
    # Hashing / helpers
    # -----------------------

    def model_hash(self) -> str:
        """
        Backwards-compatible alias for blake3_hash().

        Use this in older call-sites that predate the unified hashing API.
        """
        return self.blake3_hash()

    # -----------------------
    # High-level constructor
    # -----------------------

    @classmethod
    def from_transcript(
        cls,
        *,
        world_bit: WorldBit,
        success: bool,
        attack_id: str,
        transcript: str | bytes,
        guardrails_applied: str,
        rng_seed: int,
        timestamp: float | None = None,
        session_id: str = "",
        attack_strategy: str = "",
        utility_score: float | None = None,
        creator_id: str | None = None,
        salt: bytes | None = None,
    ) -> AttackResult:
        """
        Convenience constructor that takes the raw transcript and hashes it.

        IMPORTANT
        ---------
        - The transcript is *not* stored on the model, only a BLAKE3 hash.
        - If `salt` is provided, reproducibility of the hash requires that the
          caller store the same salt alongside experiment metadata.
        - `success` must be a real boolean; StrictBool on the model ensures
          that passing strings like "true" will raise a ValidationError.

        Timestamp semantics
        -------------------
        - If `timestamp` is None: we capture "now" via `_now_unix()`.
        - Else: we normalize via `_normalize_unix_timestamp()` for safety.
        """
        ts = _normalize_unix_timestamp(timestamp) if timestamp is not None else _now_unix()
        thash = _hash_text(transcript, salt=salt)
        return cls(
            world_bit=world_bit,
            success=success,  # StrictBool on the field enforces strictness
            attack_id=str(attack_id),
            transcript_hash=thash,
            guardrails_applied=str(guardrails_applied),
            rng_seed=int(rng_seed),
            timestamp=ts,
            session_id=str(session_id),
            attack_strategy=str(attack_strategy),
            utility_score=utility_score,
            creator_id=creator_id,
        )


# ---------------------------------------------------------------------------
# GuardrailSpec
# ---------------------------------------------------------------------------


class GuardrailSpec(ModelBase):
    """Specification for a guardrail configuration."""

    name: str
    params: dict[str, Any] = Field(default_factory=dict)
    calibration_fpr_target: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Target false positive rate for calibration.",
    )
    calibration_data_hash: str = ""
    version: str = "1.0"
    id: str = Field(
        default_factory=lambda: uuid.uuid4().hex[:REQUEST_ID_LENGTH],
        description=f"Stable guardrail spec id (first {REQUEST_ID_LENGTH} hex chars of uuid4).",
    )
    risk_level: RiskLevel = RiskLevel.MEDIUM  # Safety tagging

    @field_validator("params", mode="before")
    @classmethod
    def _coerce_params(cls, v: Any) -> dict[str, Any]:
        if v is None:
            return {}
        if isinstance(v, dict):
            return v
        if isinstance(v, Mapping):
            return dict(v)
        raise TypeError("params must be a mapping-like object")

    @computed_field
    @property
    def config_hash(self) -> str:
        """
        Stable configuration hash over (name, version, params).

        Ignores calibration_data_hash, risk_level and id so that two specs with
        identical configuration but different calibration datasets or ids
        share a hash.
        """
        payload = {
            "name": self.name,
            "version": self.version,
            "params": self.params,
        }
        return _hash_json(payload)


# ---------------------------------------------------------------------------
# WorldConfig
# ---------------------------------------------------------------------------


class WorldConfig(ModelBase):
    """Configuration for a world in the two-world protocol."""

    world_id: WorldBit
    guardrail_stack: list[GuardrailSpec] = Field(default_factory=list)
    utility_profile: dict[str, Any] = Field(default_factory=dict)
    baseline_success_rate: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="P(attack succeeds | no guardrail block) in this world.",
    )
    description: str = ""

    @field_validator("guardrail_stack", mode="before")
    @classmethod
    def _normalize_stack(cls, v: Any) -> list[GuardrailSpec]:
        if v is None:
            return []
        if isinstance(v, str):
            raise TypeError(
                "guardrail_stack must not be a bare string. "
                "Provide a GuardrailSpec, a mapping, or an iterable of those."
            )
        # Accept single spec / mapping or iterable of specs
        if isinstance(v, (GuardrailSpec, Mapping)):
            raw_list = [v]
        else:
            try:
                raw_list = list(v)
            except TypeError:
                raw_list = [v]

        out: list[GuardrailSpec] = []
        for gr in raw_list:
            if isinstance(gr, GuardrailSpec):
                out.append(gr)
            else:
                out.append(GuardrailSpec.model_validate(gr))
        return out

    @computed_field
    @property
    def env_hash(self) -> str:
        """Fingerprint of this world configuration (world_id + guardrail stack)."""
        stack_hashes = [spec.config_hash for spec in self.guardrail_stack]
        payload = {"world_id": int(self.world_id), "stack": stack_hashes}
        return _hash_json(payload)


# ---------------------------------------------------------------------------
# ExperimentConfig
# ---------------------------------------------------------------------------


class ExperimentConfig(ModelBase):
    """Complete experiment configuration for a two-world CC experiment."""

    # Exclude derived iso_time from hashes
    HASH_EXCLUDE_FIELDS: ClassVar[frozenset[str]] = frozenset({"iso_time"})

    experiment_id: str
    n_sessions: int = Field(gt=0)
    attack_strategies: list[str] = Field(min_length=1)
    guardrail_configs: dict[str, list[GuardrailSpec]] = Field(
        min_length=1,
        description="Mapping from label -> list[GuardrailSpec] composing that configuration.",
    )
    utility_target: float = Field(default=0.9, ge=0.0, le=1.0)
    utility_tolerance: float = Field(default=0.02, ge=0.0, le=0.5)
    random_seed: int = 42
    created_at: float = Field(default_factory=_now_unix)

    @field_validator("guardrail_configs", mode="before")
    @classmethod
    def _normalize_guardrail_configs(
        cls,
        v: Any,
    ) -> dict[str, list[GuardrailSpec]]:
        """
        Normalize guardrail_configs: mapping[str, list[GuardrailSpec]].

        Rules:
        - The mapping itself must be non-empty.
        - For each key:
            * None -> [] (no guardrails for that label)
            * GuardrailSpec -> [spec]
            * mapping -> [GuardrailSpec(**mapping)]
            * iterable -> list of GuardrailSpec / coercible mappings
        - An empty list for a key is allowed (e.g. 'baseline': []) to represent
          a no-guardrail configuration.
        """
        if v is None:
            raise ValueError("guardrail_configs must be provided and non-empty")
        if not isinstance(v, Mapping):
            raise TypeError("guardrail_configs must be a mapping name -> list[GuardrailSpec]")
        if not v:
            raise ValueError("guardrail_configs cannot be empty")

        result: dict[str, list[GuardrailSpec]] = {}
        for k, raw_val in v.items():
            if not isinstance(k, str):
                raise TypeError(f"guardrail_configs keys must be strings, got {type(k)}")

            # Interpret None as "no guardrails for this config label"
            if raw_val is None:
                result[k] = []
                continue

            if isinstance(raw_val, str):
                raise TypeError(
                    f"guardrail_configs['{k}'] must not be a bare string. "
                    "Provide a GuardrailSpec, a mapping, or an iterable of those."
                )

            # Accept single spec / mapping or iterable of specs
            if isinstance(raw_val, (GuardrailSpec, Mapping)):
                raw_list = [raw_val]
            else:
                try:
                    raw_list = list(raw_val)
                except TypeError:
                    raw_list = [raw_val]

            specs: list[GuardrailSpec] = []
            for gr in raw_list:
                if isinstance(gr, GuardrailSpec):
                    specs.append(gr)
                else:
                    specs.append(GuardrailSpec.model_validate(gr))

            # IMPORTANT: empty list is allowed. This is how we represent
            # a baseline / no-guardrail configuration for that label.
            result[k] = specs

        return result

    @field_validator("attack_strategies", mode="before")
    @classmethod
    def _normalize_attack_strategies(cls, v: Any) -> list[str]:
        if v is None:
            raise ValueError("attack_strategies cannot be empty")
        if isinstance(v, str):
            val = v.strip()
            return [val] if val else []
        try:
            seq = list(v)
        except TypeError:
            raise TypeError("attack_strategies must be an iterable of strings or a single string")
        if not seq:
            raise ValueError("attack_strategies cannot be empty")
        out: list[str] = []
        for x in seq:
            s = str(x).strip()
            if s:
                out.append(s)
        if not out:
            raise ValueError("attack_strategies cannot be all empty strings")
        return out

    @field_validator("random_seed", mode="before")
    @classmethod
    def _normalize_seed(cls, v: Any) -> int:
        try:
            iv = int(v)
        except (TypeError, ValueError):
            raise ValueError("random_seed must be an integer")
        return iv

    @field_validator("created_at", mode="before")
    @classmethod
    def _normalize_created_at(cls, v: Any) -> float:
        return _normalize_unix_timestamp(v)

    @computed_field
    @property
    def iso_time(self) -> str:
        """ISO-8601 representation of `created_at` in UTC."""
        return _iso_from_unix(self.created_at)


# ---------------------------------------------------------------------------
# CCResult
# ---------------------------------------------------------------------------


class CCResult(ModelBase):
    """
    Results of CC analysis for a particular composition.

    j_empirical:
        Empirical J statistic for the composition (can be negative if destructive).
    cc_max:
        Maximal CC statistic for the composition (ratio vs single-rail best J).
    delta_add:
        Additive delta in J vs best single rail (J_comp - max(J_i)).
    cc_multiplicative:
        Optional multiplicative CC definition (e.g., J_comp / product(J_i)).
    confidence_interval:
        Optional CI over cc_max (lo, hi) at `ci_level`, using `ci_method`.
    bootstrap_samples:
        Optional raw bootstrap sample values for cc_max (post-cleaning).
    n_sessions:
        Number of attack sessions used to estimate the statistics.
    """

    j_empirical: float
    cc_max: float
    delta_add: float
    cc_multiplicative: float | None = None
    confidence_interval: tuple[float, float] | None = None
    bootstrap_samples: FloatSeq | None = None
    n_sessions: int = 0
    ci_method: CiMethod = CiMethod.BOOTSTRAP
    ci_level: float = Field(default=0.95, gt=0.5, lt=1.0)

    @field_validator("j_empirical", "cc_max", "delta_add", "cc_multiplicative", mode="before")
    @classmethod
    def _validate_finite_float(cls, v: Any) -> Any:
        if v is None:
            return None
        try:
            f = float(v)
        except (TypeError, ValueError):
            raise ValueError("CCResult numeric fields must be numeric")
        if math.isnan(f) or math.isinf(f):
            raise ValueError("CCResult numeric fields cannot be NaN or infinite")
        return f

    @field_validator("n_sessions", mode="before")
    @classmethod
    def _validate_n_sessions(cls, v: Any) -> int:
        try:
            iv = int(v)
        except (TypeError, ValueError):
            raise ValueError("n_sessions must be an integer")
        if iv < 0:
            raise ValueError("n_sessions must be >= 0")
        return iv

    @field_validator("confidence_interval", mode="before")
    @classmethod
    def _normalize_ci(cls, v: Any) -> tuple[float, float] | None:
        if v is None:
            return None
        if isinstance(v, Sequence) and len(v) == 2:
            lo, hi = float(v[0]), float(v[1])
            if math.isnan(lo) or math.isinf(lo) or math.isnan(hi) or math.isinf(hi):
                raise ValueError("confidence_interval bounds cannot be NaN or infinite")
            if lo > hi:
                raise ValueError("confidence_interval must satisfy lo <= hi")
            return (lo, hi)
        raise ValueError("confidence_interval must be a 2-tuple (lo, hi) or None")

    @field_validator("bootstrap_samples", mode="before")
    @classmethod
    def _normalize_bootstrap(cls, v: Any) -> FloatSeq | None:
        """
        Normalize bootstrap samples.

        - Reject bare strings outright.
        - Drop NaN/inf but:
            * warn about how many were dropped,
            * raise if *all* values are non-finite.
        """
        if v is None:
            return None

        # Reject bare strings to avoid character-level splitting
        if isinstance(v, str):
            raise ValueError("bootstrap_samples must be a sequence of numerics, not a string")

        # Numpy array → flattened list
        if NUMPY_AVAILABLE and isinstance(v, np.ndarray):  # type: ignore[attr-defined]
            seq = v.flatten().tolist()
        else:
            seq = v

        out: list[float] = []
        dropped = 0

        try:
            for x in seq:
                f = float(x)
                if math.isnan(f) or math.isinf(f):
                    dropped += 1
                    continue
                out.append(f)
        except (TypeError, ValueError):
            raise ValueError("bootstrap_samples must be a sequence of numerics")

        if dropped:
            warnings.warn(
                f"CCResult.bootstrap_samples: dropped {dropped} non-finite samples",
                UserWarning,
                stacklevel=2,
            )

        if not out:
            raise ValueError("bootstrap_samples contained no finite values")

        return out

    @field_serializer("bootstrap_samples")
    def _serialize_bootstrap(self, v: FloatSeq | None) -> list[float] | None:
        if v is None:
            return None
        return [float(x) for x in v]


# ---------------------------------------------------------------------------
# AttackStrategySpec (declarative attacker configuration)
# ---------------------------------------------------------------------------


class AttackStrategySpec(ModelBase):
    """
    Configuration for an attack strategy (declarative spec).

    NOTE
    ----
    This is a *specification* for building a runtime attacker, not the attacker
    implementation itself. Runtime attacker classes in `core.attackers` or
    similar modules should accept this spec and implement the actual attack
    behavior.

    Backwards-compatibility
    -----------------------
    A deprecated alias `AttackStrategy` is exported via __getattr__ so existing
    code referring to AttackStrategy continues to work, but new code should use
    AttackStrategySpec explicitly.
    """

    name: str
    params: dict[str, Any] = Field(default_factory=dict)
    vocabulary: list[str] = Field(default_factory=list)
    success_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    description: str = ""

    @field_validator("params", mode="before")
    @classmethod
    def _coerce_params(cls, v: Any) -> dict[str, Any]:
        if v is None:
            return {}
        if isinstance(v, dict):
            return v
        if isinstance(v, Mapping):
            return dict(v)
        raise TypeError("params must be a mapping-like object")

    @field_validator("vocabulary", mode="before")
    @classmethod
    def _normalize_vocab(cls, v: Any) -> list[str]:
        if v is None:
            return []
        if isinstance(v, str):
            s = v.strip()
            return [s] if s else []
        try:
            seq = list(v)
        except TypeError:
            raise TypeError("vocabulary must be an iterable of strings or a single string")
        out: list[str] = []
        for x in seq:
            s = str(x).strip()
            if s:
                out.append(s)
        return out


__all__ = [
    "AVRO_AVAILABLE",
    "MAX_REASONABLE_UNIX_TIMESTAMP",
    "NUMPY_AVAILABLE",
    "PROTO_AVAILABLE",
    "SQLALCHEMY_AVAILABLE",
    "AttackResult",
    "AttackStrategy",
    "AttackStrategySpec",
    "AuditColumnsMixin",
    "CCResult",
    "CiMethod",
    "ExperimentConfig",
    "GuardrailSpec",
    # models
    "ModelBase",
    # ORM
    "OrmBase",
    "RiskLevel",
    # enums
    "WorldBit",
    "WorldConfig",
    "_hash_json",
    "_hash_text",
    "_iso_from_unix",
    "_normalize_unix_timestamp",
    # helpers
    "_now_unix",
]


def __getattr__(name: str) -> Any:
    """
    Backwards-compat attribute hook.

    Accessing `cc.core.models.AttackStrategy` emits a DeprecationWarning
    and returns AttackStrategySpec.
    """
    if name == "AttackStrategy":
        warnings.warn(
            "AttackStrategy is deprecated; use AttackStrategySpec instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return AttackStrategySpec
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
