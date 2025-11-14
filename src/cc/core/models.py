# src/cc/core/models.py
"""
Module: core.models (data layer for CC framework)
Purpose: Strongly-typed, versioned, and serializable data models used across the
         two-world protocol, calibration, analysis, and reporting.

Author: Pranav Bhave
Dates:
  - 2025-08-31: original
  - 2025-09-28: ultimate upgrade (validation, slots, versioned schemas, robust
                (de)serialization, hashing helpers, immutability patterns, numpy
                safety, convenience constructors)
  - 2025-11-13: 10/10 polish - Pydantic for validation/serialization/schemas,
                enums, blake3 hashing, frozen immutability, recursive validation,
                risk enums for safety, full tests/doctests per research.
  - 2025-11-13 v4.0: Ultimate hardening - SQLAlchemy integration hooks for DB
                persistence, Avro/Protobuf export for big data, hash caching,
                type aliases, lightweight migration helpers, audit metadata,
                OpenAPI schemas, property-based tests & benchmarks (out of band).

Design notes
------------
- Pydantic BaseModel (v2) for runtime validation, serialization, schemas.
- Frozen/immutable instances for safety in evaluations.
- Enums for fixed values (WorldBit, CiMethod, RiskLevel).
- BLAKE3 hashing for integrity (faster/secure than sha256) with cached default hash.
- Auto JSON schema generation + OpenAPI export via Pydantic.
- Backwards-compatible field names for persisted payloads.
- Optional SQLAlchemy helpers (separate ORM base + audit mixin).
- Optional Avro/Protobuf serializers for big-data pipelines.
- Audit metadata field (creator_id) and updated_at for traceability snapshots.
- Migration helper entrypoint for schema bumps.
- Tests: doctests + pytest + hypothesis (property-based) live in tests/.
- Benchmarks: hashing/serialization perf live in benchmarks/.

Dependencies:
- Required: pydantic >= 2.0, blake3
- Optional: numpy (for CCResult.bootstrap_samples),
            sqlalchemy (for ORM mixins),
            fastavro (for Avro bytes),
            google.protobuf (for Protobuf export).
"""

from __future__ import annotations

import io
import json
import math
import time
import uuid
from enum import Enum, IntEnum
from functools import cached_property
from typing import (
    Any,
    ClassVar,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import blake3
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_serializer,
    field_validator,
)

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
    from datetime import datetime
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


_SCHEMA_VERSION: str = "4.0"

# Type aliases for clarity
JsonDict = Dict[str, Any]
FloatSeq = Sequence[float]
TModel = TypeVar("TModel", bound="ModelBase")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _now_unix() -> float:
    """Unix timestamp in seconds (UTC)."""
    return float(time.time())


def _iso_from_unix(ts: float) -> str:
    """ISO-8601 in UTC with millisecond precision, suffixed with 'Z'."""
    tm = time.gmtime(ts)
    ms = int((ts - int(ts)) * 1000)
    return (
        f"{tm.tm_year:04d}-{tm.tm_mon:02d}-{tm.tm_mday:02d}"
        f"T{tm.tm_hour:02d}:{tm.tm_min:02d}:{tm.tm_sec:02d}.{ms:03d}Z"
    )


def _hash_json(obj: Any) -> str:
    """Canonical JSON -> BLAKE3 hex digest."""
    data = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return blake3.blake3(data.encode("utf-8")).hexdigest()


def _hash_text(text: Union[str, bytes]) -> str:
    """Hash raw transcript text/bytes with BLAKE3."""
    if isinstance(text, bytes):
        data = text
    else:
        data = str(text).encode("utf-8", errors="replace")
    return blake3.blake3(data).hexdigest()


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class WorldBit(IntEnum):
    """World indicator for the two-world protocol."""
    BASELINE = 0
    PROTECTED = 1


class CiMethod(str, Enum):
    BOOTSTRAP = "bootstrap"
    WILSON = "wilson"
    BAYES = "bayes"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# ---------------------------------------------------------------------------
# SQLAlchemy helpers (optional, separate from Pydantic models)
# ---------------------------------------------------------------------------
if SQLALCHEMY_AVAILABLE:

    class OrmBase(DeclarativeBase):
        """Base for SQLAlchemy ORM models in the CC framework."""
        pass

    class AuditColumnsMixin:
        """
        Audit columns mixin for SQLAlchemy ORM models.

        Typical usage (in a separate ORM module):

            class AttackResultORM(OrmBase, AuditColumnsMixin):
                __tablename__ = "attack_results"
                id = Column(Integer, primary_key=True)
                ...
        """

        @declared_attr
        def creator_id(cls) -> Column:  # type: ignore[type-arg]
            return Column(String(64), nullable=True)

        @declared_attr
        def updated_at(cls) -> Column:  # type: ignore[type-arg]
            return Column(
                DateTime(timezone=True),
                default=datetime.utcnow,
                onupdate=datetime.utcnow,
                nullable=False,
            )
else:  # pragma: no cover

    class OrmBase:
        """Fallback stub when SQLAlchemy is not installed."""
        pass

    class AuditColumnsMixin:
        """Fallback stub when SQLAlchemy is not installed."""
        pass


# ---------------------------------------------------------------------------
# Base Pydantic model
# ---------------------------------------------------------------------------
class ModelBase(BaseModel):
    """
    Shared BaseModel config for all CC core models.

    - frozen/immutable instances (safety for evaluations)
    - extra fields ignored (backwards-compatible decoding)
    - schema_version attached to every instance
    - optional creator_id for audit metadata
    - updated_at as a snapshot timestamp for this representation
    """
    schema_version: str = Field(default=_SCHEMA_VERSION, frozen=True)
    creator_id: Optional[str] = Field(
        default=None,
        description="Optional id of human/system that created this record.",
    )
    updated_at: float = Field(
        default_factory=_now_unix,
        description="Unix timestamp (UTC seconds) when this snapshot was last updated.",
    )

    # Pydantic v2 config
    model_config = ConfigDict(
        frozen=True,
        extra="ignore",
        validate_assignment=False,  # immutability via new instance, not in-place
        populate_by_name=True,
        arbitrary_types_allowed=True,
        ser_json_timedelta="iso8601",
        ser_json_bytes="utf8",
    )

    @field_validator("updated_at", mode="before")
    @classmethod
    def _normalize_updated_at(cls, v: Any) -> float:
        if v in (None, ""):
            return _now_unix()
        try:
            f = float(v)
        except (TypeError, ValueError):
            raise ValueError("updated_at must be numeric unix seconds")
        if f <= 0 or math.isinf(f):
            return _now_unix()
        return f

    @cached_property
    def _default_hash(self) -> str:
        """Cached BLAKE3 hash of the full JSON representation (no excludes)."""
        data = self.model_dump(mode="json")
        return _hash_json(data)

    def blake3_hash(
        self,
        *,
        exclude: Optional[Sequence[str]] = None,
        use_cache: bool = True,
    ) -> str:
        """
        Stable BLAKE3 hash of the JSON representation of the model.

        Parameters
        ----------
        exclude:
            Optional iterable of field names to drop before hashing
            (useful to ignore ids, timestamps, etc.).
        use_cache:
            If True and `exclude` is None, reuse cached hash of the full object.
        """
        if exclude is None and use_cache:
            return self._default_hash
        exclude_set = set(exclude or [])
        data = self.model_dump(mode="json", exclude=exclude_set)
        return _hash_json(data)

    # ---------- Big-data / interoperability helpers ----------

    def to_avro_record(self) -> JsonDict:
        """
        Export as an Avro-compatible record (plain JSON dict).

        Note
        ----
        - This method does not require fastavro. It returns a JSON-safe dict
          that can be written with fastavro or any Avro writer using a schema.
        """
        return self.model_dump(mode="json")

    def to_avro_bytes(self, schema: JsonDict) -> bytes:
        """
        Export as Avro bytes using fastavro (if installed).

        Parameters
        ----------
        schema:
            Avro schema dict compatible with this model's fields.

        Raises
        ------
        ImportError
            If fastavro is not installed.
        """
        if not AVRO_AVAILABLE:  # pragma: no cover
            raise ImportError("fastavro is required for Avro byte serialization")
        buf = io.BytesIO()
        fastavro.schemaless_writer(buf, schema, self.to_avro_record())  # type: ignore[arg-type]
        return buf.getvalue()

    def to_protobuf(self, proto_cls: Type[Message]) -> Message:
        """
        Export to a protobuf message instance.

        Parameters
        ----------
        proto_cls:
            A protobuf Message subclass whose field names match this model.

        Raises
        ------
        ImportError
            If `google.protobuf` is not installed.
        """
        if not PROTO_AVAILABLE:  # pragma: no cover
            raise ImportError("google.protobuf is required for protobuf export")
        data = self.model_dump(mode="json")
        msg = proto_cls()  # type: ignore[call-arg]
        ParseDict(data, msg, ignore_unknown_fields=True)  # type: ignore[arg-type]
        return msg

    def openapi_schema(self) -> JsonDict:
        """Return the OpenAPI-compatible JSON schema for this model type."""
        # This is instance-independent; we delegate to Pydantic's JSON schema.
        return self.model_json_schema()

    # ---------- Migration helper ----------

    @classmethod
    def migrate(cls: Type[TModel], old_data: Mapping[str, Any]) -> TModel:
        """
        Best-effort migration entrypoint for schema upgrades.

        Default implementation:
            - Accepts a mapping (e.g., legacy JSON-decoded payload).
            - Validates/coerces it into the current model.

        Override in subclasses if you need schema_version-aware transforms.
        """
        return cls.model_validate(old_data)


# ---------------------------------------------------------------------------
# AttackResult
# ---------------------------------------------------------------------------
class AttackResult(ModelBase):
    """
    Result of a single attack session in the two-world protocol.

    Backwards-compatible fields with additional optional metadata.

    - world_bit: 0 = baseline, 1 = guardrail-enabled (WorldBit enum)
    - transcript_hash: BLAKE3 hash of the full transcript (not stored inline)
    - utility_score: optional scalar for multi-objective evaluation
    """
    world_bit: WorldBit = Field(
        description="0 (baseline) or 1 (guardrail-enabled).",
    )
    success: bool
    attack_id: str
    transcript_hash: str
    guardrails_applied: str
    rng_seed: int
    timestamp: float = Field(
        description="Unix timestamp in seconds (UTC) for when the attack was evaluated.",
    )
    # --- optional / metadata
    session_id: str = ""
    attack_strategy: str = ""
    utility_score: Optional[float] = None
    request_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])

    @field_validator("utility_score", mode="before")
    @classmethod
    def _normalize_utility(cls, v: Any) -> Optional[float]:
        if v is None or v == "":
            return None
        try:
            f = float(v)
        except (TypeError, ValueError):
            raise ValueError("utility_score must be numeric or None")
        if math.isnan(f) or math.isinf(f):
            raise ValueError("utility_score cannot be NaN or infinite")
        return f

    @field_validator("timestamp", mode="before")
    @classmethod
    def _normalize_timestamp(cls, v: Any) -> float:
        if v in (None, "", 0):
            return _now_unix()
        try:
            f = float(v)
        except (TypeError, ValueError):
            raise ValueError("timestamp must be numeric")
        if f <= 0 or math.isinf(f):
            return _now_unix()
        return f

    @field_validator("rng_seed", mode="before")
    @classmethod
    def _normalize_seed(cls, v: Any) -> int:
        try:
            return int(v)
        except (TypeError, ValueError):
            raise ValueError("rng_seed must be an integer")

    @field_validator("attack_id", "guardrails_applied", "session_id", "attack_strategy", mode="before")
    @classmethod
    def _coerce_str(cls, v: Any) -> str:
        if v is None:
            return ""
        return str(v)

    @computed_field
    @property
    def iso_time(self) -> str:
        """ISO-8601 representation of `timestamp` in UTC."""
        return _iso_from_unix(self.timestamp)

    def model_hash(self) -> str:
        """Alias for blake3_hash for backwards compatibility."""
        return self.blake3_hash()

    @classmethod
    def from_transcript(
        cls,
        *,
        world_bit: WorldBit,
        success: bool,
        attack_id: str,
        transcript: Union[str, bytes],
        guardrails_applied: str,
        rng_seed: int,
        timestamp: Optional[float] = None,
        session_id: str = "",
        attack_strategy: str = "",
        utility_score: Optional[float] = None,
        creator_id: Optional[str] = None,
    ) -> "AttackResult":
        """
        Convenience constructor that takes the raw transcript and hashes it.

        This avoids accidentally persisting the full transcript in the data layer.
        """
        ts = _now_unix() if timestamp is None else float(timestamp)
        thash = _hash_text(transcript)
        return cls(
            world_bit=world_bit,
            success=success,
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
    params: Dict[str, Any] = Field(default_factory=dict)
    calibration_fpr_target: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Target false positive rate for calibration.",
    )
    calibration_data_hash: str = ""
    version: str = "1.0"
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    risk_level: RiskLevel = RiskLevel.MEDIUM  # Safety tagging

    @field_validator("params", mode="before")
    @classmethod
    def _coerce_params(cls, v: Any) -> Dict[str, Any]:
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

        This deliberately ignores calibration_data_hash so that two specs with
        identical configuration but different calibration datasets share a hash.
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
    guardrail_stack: List[GuardrailSpec] = Field(default_factory=list)
    utility_profile: Dict[str, Any] = Field(default_factory=dict)
    baseline_success_rate: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="P(attack succeeds | no guardrail block) in this world.",
    )
    description: str = ""

    @field_validator("guardrail_stack", mode="before")
    @classmethod
    def _normalize_stack(cls, v: Any) -> List[GuardrailSpec]:
        if v is None:
            return []
        # Accept single spec / mapping or iterable of specs
        if isinstance(v, GuardrailSpec):
            raw_list = [v]
        elif isinstance(v, Mapping):
            raw_list = [v]
        else:
            try:
                raw_list = list(v)
            except TypeError:
                raw_list = [v]

        out: List[GuardrailSpec] = []
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
    """Complete experiment configuration."""
    experiment_id: str
    n_sessions: int = Field(gt=0)
    attack_strategies: List[str] = Field(min_length=1)
    guardrail_configs: Dict[str, List[GuardrailSpec]] = Field(
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
    ) -> Dict[str, List[GuardrailSpec]]:
        if v is None:
            raise ValueError("guardrail_configs must be provided and non-empty")
        if not isinstance(v, Mapping):
            raise TypeError("guardrail_configs must be a mapping name -> list[GuardrailSpec]")
        if not v:
            raise ValueError("guardrail_configs cannot be empty")

        result: Dict[str, List[GuardrailSpec]] = {}
        for k, raw_val in v.items():
            if not isinstance(k, str):
                raise TypeError(f"guardrail_configs keys must be strings, got {type(k)}")

            # Accept single spec, mapping, or iterable of specs
            if isinstance(raw_val, GuardrailSpec):
                raw_list = [raw_val]
            elif isinstance(raw_val, Mapping):
                raw_list = [raw_val]
            else:
                try:
                    raw_list = list(raw_val)
                except TypeError:
                    raw_list = [raw_val]

            specs: List[GuardrailSpec] = []
            for gr in raw_list:
                if isinstance(gr, GuardrailSpec):
                    specs.append(gr)
                else:
                    specs.append(GuardrailSpec.model_validate(gr))
            result[k] = specs

        return result

    @field_validator("attack_strategies", mode="before")
    @classmethod
    def _normalize_attack_strategies(cls, v: Any) -> List[str]:
        if v is None:
            raise ValueError("attack_strategies cannot be empty")
        if isinstance(v, str):
            return [v]
        try:
            seq = list(v)
        except TypeError:
            raise TypeError("attack_strategies must be an iterable of strings or a single string")
        if not seq:
            raise ValueError("attack_strategies cannot be empty")
        return [str(x) for x in seq]

    @field_validator("random_seed", mode="before")
    @classmethod
    def _normalize_seed(cls, v: Any) -> int:
        try:
            return int(v)
        except (TypeError, ValueError):
            raise ValueError("random_seed must be an integer")

    @computed_field
    @property
    def iso_time(self) -> str:
        """ISO-8601 representation of `created_at` in UTC."""
        return _iso_from_unix(self.created_at)


# ---------------------------------------------------------------------------
# CCResult
# ---------------------------------------------------------------------------
class CCResult(ModelBase):
    """Results of CC analysis."""
    j_empirical: float
    cc_max: float
    delta_add: float
    cc_multiplicative: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    bootstrap_samples: Optional[FloatSeq] = None
    n_sessions: int = 0
    ci_method: CiMethod = CiMethod.BOOTSTRAP
    ci_level: float = Field(default=0.95, gt=0.5, lt=1.0)

    @field_validator("confidence_interval", mode="before")
    @classmethod
    def _normalize_ci(cls, v: Any) -> Optional[Tuple[float, float]]:
        if v is None:
            return None
        if isinstance(v, Sequence) and len(v) == 2:
            lo, hi = float(v[0]), float(v[1])
            if lo > hi:
                raise ValueError("confidence_interval must satisfy lo <= hi")
            return (lo, hi)
        raise ValueError("confidence_interval must be a 2-tuple (lo, hi) or None")

    @field_validator("bootstrap_samples", mode="before")
    @classmethod
    def _normalize_bootstrap(cls, v: Any) -> Optional[FloatSeq]:
        if v is None:
            return None
        # Explicitly reject bare strings to avoid character-level splitting
        if isinstance(v, str):
            raise ValueError("bootstrap_samples must be a sequence of numerics, not a string")

        # numpy array -> flattened list
        if NUMPY_AVAILABLE and isinstance(v, np.ndarray):  # type: ignore[attr-defined]
            flat_list = v.flatten().tolist()
            return [float(x) for x in flat_list if not (math.isnan(float(x)) or math.isinf(float(x)))]

        # generic iterable, filter invalid
        try:
            out: List[float] = []
            for x in v:
                f = float(x)
                if not (math.isnan(f) or math.isinf(f)):
                    out.append(f)
            return out
        except (TypeError, ValueError):
            raise ValueError("bootstrap_samples must be a sequence of numerics")

    @field_serializer("bootstrap_samples")
    def _serialize_bootstrap(self, v: Optional[FloatSeq]) -> Optional[List[float]]:
        if v is None:
            return None
        return [float(x) for x in v]


# ---------------------------------------------------------------------------
# AttackStrategy
# ---------------------------------------------------------------------------
class AttackStrategy(ModelBase):
    """Configuration for an attack strategy (declarative)."""
    name: str
    params: Dict[str, Any] = Field(default_factory=dict)
    vocabulary: List[str] = Field(default_factory=list)
    success_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    description: str = ""

    @field_validator("params", mode="before")
    @classmethod
    def _coerce_params(cls, v: Any) -> Dict[str, Any]:
        if v is None:
            return {}
        if isinstance(v, dict):
            return v
        if isinstance(v, Mapping):
            return dict(v)
        raise TypeError("params must be a mapping-like object")

    @field_validator("vocabulary", mode="before")
    @classmethod
    def _normalize_vocab(cls, v: Any) -> List[str]:
        if v is None:
            return []
        if isinstance(v, str):
            # Treat a single string as a single token list, not char-split
            return [v]
        try:
            seq = list(v)
        except TypeError:
            raise TypeError("vocabulary must be an iterable of strings or a single string")
        return [str(x) for x in seq]


__all__ = [
    "WorldBit",
    "CiMethod",
    "RiskLevel",
    "AttackResult",
    "GuardrailSpec",
    "WorldConfig",
    "ExperimentConfig",
    "CCResult",
    "AttackStrategy",
    "OrmBase",
    "AuditColumnsMixin",
]

# ---------------------------------------------------------------------------
# CLI / smoke utilities
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Example: print AttackResult schema (OpenAPI-compatible)
    print(json.dumps(AttackResult.model_json_schema(), indent=2))

    # Run doctests (if any)
    import doctest

    doctest.testmod()

    # Simple hash benchmark if numpy is available
    if NUMPY_AVAILABLE:  # type: ignore[truthy-bool]
        import timeit

        samples = np.random.rand(1000)  # type: ignore[attr-defined]
        res = CCResult(j_empirical=0.5, cc_max=0.6, delta_add=0.1, bootstrap_samples=samples)
        dur = timeit.timeit(res.blake3_hash, number=1000)
        print(f"blake3_hash x1000: {dur:.6f}s")
