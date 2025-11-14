import json
import math
import time
from typing import Any, Dict, Optional, Sequence, Mapping
import pytest
from hypothesis import given, strategies as st
from pydantic import ValidationError, BaseModel, Field

from cc.core.models import (
    _now_unix,
    _iso_from_unix,
    _hash_json,
    _hash_text,
    AVRO_AVAILABLE,
    PROTO_AVAILABLE,
    SQLALCHEMY_AVAILABLE,
    NUMPY_AVAILABLE,
    WorldBit,
    CiMethod,
    RiskLevel,
    ModelBase,
    OrmBase,
    AuditColumnsMixin,
    AttackResult,
    GuardrailSpec,
    WorldConfig,
    ExperimentConfig,
    CCResult,
    AttackStrategy,
)

if NUMPY_AVAILABLE:  # type: ignore[truthy-bool]
    import numpy as np  # type: ignore[import]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def assert_immutable(instance: Any, field: str, new_value: Any) -> None:
    """Ensure models are effectively immutable."""
    with pytest.raises((AttributeError, ValidationError, TypeError)):
        setattr(instance, field, new_value)


# ---------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------


@pytest.mark.parametrize("enum_cls", [WorldBit, CiMethod, RiskLevel])
def test_enums_roundtrip(enum_cls):
    # value roundtrip and JSON representation
    for member in enum_cls:
        assert enum_cls(member.value) == member
        assert json.dumps(member.value) == json.dumps(member)


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------


def test_now_unix():
    ts = _now_unix()
    assert isinstance(ts, float)
    # should be roughly "now"
    assert ts > time.time() - 1.0


def test_iso_from_unix_epoch():
    iso = _iso_from_unix(0.0)
    assert iso == "1970-01-01T00:00:00.000Z"
    # shape check: YYYY-MM-DDTHH:MM:SS.mmmZ
    assert len(_iso_from_unix(time.time())) == 24


@given(
    obj=st.dictionaries(st.text(min_size=0, max_size=10), st.integers()),
    salt=st.binary(min_size=1, max_size=16),  # <-- min_size = 1
)
def test_hash_json_determinism_and_salt(obj: Dict[str, int], salt: bytes):
    # deterministic without salt
    h1 = _hash_json(obj)
    h2 = _hash_json(obj)
    assert h1 == h2

    # salted hash should differ from unsalted with very high probability
    hs = _hash_json(obj, salt=salt)
    assert hs != h1

@given(
    text=st.text(min_size=0, max_size=50),
    salt=st.binary(min_size=1, max_size=16),
)
def test_hash_text_determinism_and_salt(text: str, salt: bytes):
    h_plain = _hash_text(text)
    h_salted = _hash_text(text, salt=salt)
    assert h_plain != h_salted


@given(
    obj=st.dictionaries(st.text(min_size=0, max_size=10), st.integers()),
    salt=st.binary(min_size=0, max_size=16),
)
def test_hash_json_determinism_and_salt(obj: Dict[str, int], salt: bytes):
    # deterministic without salt
    h1 = _hash_json(obj)
    h2 = _hash_json(obj)
    assert h1 == h2

    # salted hash differs from unsalted (with overwhelmingly high probability)
    hs = _hash_json(obj, salt=salt)
    if salt:
        assert hs != h1


# ---------------------------------------------------------------------
# ModelBase
# ---------------------------------------------------------------------


def test_modelbase_defaults_and_hash():
    class TestModel(ModelBase):
        field: int = 1

    m = TestModel()
    assert m.schema_version == "4.0"
    assert m.creator_id is None
    assert m.updated_at > 0

    h1 = m.blake3_hash()
    h2 = m.blake3_hash()  # uses cached hash
    assert isinstance(h1, str) and len(h1) == 64
    assert h1 == h2

    # exclude / salt change the hash
    h_excl = m.blake3_hash(exclude=["field"], use_cache=False)
    assert h_excl != h1
    h_salt = m.blake3_hash(use_cache=False, salt=b"x")
    assert h_salt != h1


@given(st.floats(allow_nan=False, allow_infinity=False))
def test_modelbase_updated_at_normalization(ts: float):
    class TestModel(ModelBase):
        field: int = 1

    m = TestModel(updated_at=ts)
    if ts <= 0:
        assert m.updated_at > 0
    else:
        assert m.updated_at == ts


def test_modelbase_immutability_and_assignment():
    class TestModel(ModelBase):
        field: int = 1

    m = TestModel()
    assert m.field == 1
    assert_immutable(m, "field", 2)


def test_modelbase_avro_record():
    class TestModel(ModelBase):
        field: int = 42

    m = TestModel()
    rec = m.to_avro_record()
    assert rec["field"] == 42
    assert rec["schema_version"] == "4.0"
    # record should be JSON-safe
    json.dumps(rec)


@pytest.mark.skipif(not AVRO_AVAILABLE, reason="fastavro not available")
def test_modelbase_avro_bytes_roundtrip():
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
def test_modelbase_protobuf_basic():
    # Only run if google.protobuf is actually present
    from google.protobuf.message import Message as PBMessage  # type: ignore[import]

    class TestModel(ModelBase):
        field: int = 42
        extra_field: str = "hello"

    m = TestModel()
    msg = m.to_protobuf()  # dynamic Message
    assert isinstance(msg, PBMessage)
    # The parsed fields should exist
    assert getattr(msg, "field") == 42
    assert getattr(msg, "extra_field") == "hello"
    assert getattr(msg, "schema_version") == "4.0"


def test_modelbase_openapi_schema():
    class TestModel(ModelBase):
        field: int = Field(description="test field")

    schema = TestModel.openapi_schema()
    assert schema["title"] == "TestModel"
    assert "field" in schema["properties"]
    assert schema["properties"]["field"]["description"] == "test field"


@given(st.dictionaries(st.text(min_size=1, max_size=10), st.text()))
def test_modelbase_migrate(old_data: Dict[str, str]):
    class TestModel(ModelBase):
        optional_field: Optional[str] = None

    # Should not crash; just best-effort validate
    m = TestModel.migrate(old_data)
    assert isinstance(m, TestModel)
    assert m.schema_version == "4.0"


@pytest.mark.skipif(not SQLALCHEMY_AVAILABLE, reason="SQLAlchemy not available")
def test_audit_columns_mixin():
    # Use OrmBase + AuditColumnsMixin from models
    from sqlalchemy import Column, Integer, String, DateTime  # type: ignore[import]

    class TestORM(OrmBase, AuditColumnsMixin):  # type: ignore[misc]
        __tablename__ = "test"
        id = Column(Integer, primary_key=True)

    assert hasattr(TestORM, "creator_id")
    assert isinstance(TestORM.creator_id.type, String)  # type: ignore[attr-defined]
    assert hasattr(TestORM, "updated_at")
    assert isinstance(TestORM.updated_at.type, DateTime)  # type: ignore[attr-defined]


# ---------------------------------------------------------------------
# AttackResult
# ---------------------------------------------------------------------


@given(
    success=st.booleans(),
    attack_id=st.text(min_size=1),
    guardrails_applied=st.text(min_size=1),
    rng_seed=st.integers(),
    timestamp=st.floats(allow_nan=False, allow_infinity=False),
)
def test_attack_result_valid_and_iso(
    success: bool,
    attack_id: str,
    guardrails_applied: str,
    rng_seed: int,
    timestamp: float,
):
    if timestamp <= 0:
        timestamp = time.time()
    # transcript_hash must be 64-char hex
    th = _hash_text("dummy")
    ar = AttackResult(
        world_bit=WorldBit.BASELINE,
        success=success,
        attack_id=attack_id,
        transcript_hash=th,
        guardrails_applied=guardrails_applied,
        rng_seed=rng_seed,
        timestamp=timestamp,
    )
    assert ar.world_bit is WorldBit.BASELINE
    assert ar.iso_time == _iso_from_unix(ar.timestamp)


def test_attack_result_transcript_hash_length_enforced():
    # Too short
    with pytest.raises(ValidationError):
        AttackResult(
            world_bit=WorldBit.BASELINE,
            success=True,
            attack_id="id",
            transcript_hash="short",
            guardrails_applied="ga",
            rng_seed=42,
        )


def test_attack_result_invalid_utility_nan():
    with pytest.raises(ValidationError):
        AttackResult(
            world_bit=WorldBit.BASELINE,
            success=True,
            attack_id="id",
            transcript_hash=_hash_text("x"),
            guardrails_applied="ga",
            rng_seed=42,
            utility_score=math.nan,
        )


def test_attack_result_invalid_utility_inf():
    with pytest.raises(ValidationError):
        AttackResult(
            world_bit=WorldBit.BASELINE,
            success=True,
            attack_id="id",
            transcript_hash=_hash_text("x"),
            guardrails_applied="ga",
            rng_seed=42,
            utility_score=float("inf"),
        )


def test_attack_result_from_transcript_and_salt():
    ar1 = AttackResult.from_transcript(
        world_bit=WorldBit.PROTECTED,
        success=False,
        attack_id="id",
        transcript=b"binary",
        guardrails_applied="ga",
        rng_seed=42,
    )
    ar2 = AttackResult.from_transcript(
        world_bit=WorldBit.PROTECTED,
        success=False,
        attack_id="id",
        transcript=b"binary",
        guardrails_applied="ga",
        rng_seed=42,
        salt=b"x",
    )
    assert ar1.transcript_hash == _hash_text(b"binary")
    assert ar2.transcript_hash == _hash_text(b"binary", salt=b"x")
    assert ar1.transcript_hash != ar2.transcript_hash
    assert ar1.world_bit is WorldBit.PROTECTED
    assert ar1.session_id == ""


def test_attack_result_model_hash_alias_and_immutability():
    ar = AttackResult.from_transcript(
        world_bit=WorldBit.BASELINE,
        success=True,
        attack_id="id",
        transcript="hello",
        guardrails_applied="none",
        rng_seed=0,
    )
    h = ar.model_hash()
    assert h == ar.blake3_hash()
    assert_immutable(ar, "success", False)


# ---------------------------------------------------------------------
# GuardrailSpec
# ---------------------------------------------------------------------


def test_guardrail_spec_params_normalization_and_config_hash():
    spec1 = GuardrailSpec(name="g1", params={"a": 1}, calibration_fpr_target=0.1)
    spec2 = GuardrailSpec(name="g1", params={"a": 1}, calibration_fpr_target=0.5)
    spec3 = GuardrailSpec(name="g1", params={"a": 2})

    # mapping coercion
    class MyMap(Mapping[str, Any]):
        def __init__(self):
            self._data = {"a": 1}

        def __getitem__(self, k: str) -> Any:
            return self._data[k]

        def __iter__(self):
            return iter(self._data)

        def __len__(self) -> int:
            return len(self._data)

    spec4 = GuardrailSpec(name="g1", params=MyMap())

    assert spec1.params == {"a": 1}
    assert spec4.params == {"a": 1}
    # config_hash ignores calibration_data_hash and calibration_fpr_target
    assert spec1.config_hash == spec2.config_hash
    assert spec1.config_hash != spec3.config_hash


def test_guardrail_spec_invalid_params_type():
    with pytest.raises(TypeError):
        GuardrailSpec(name="g", params=123)  # type: ignore[arg-type]


def test_guardrail_spec_fpr_bounds():
    GuardrailSpec(name="g", params={}, calibration_fpr_target=0.0)
    GuardrailSpec(name="g", params={}, calibration_fpr_target=1.0)
    with pytest.raises(ValidationError):
        GuardrailSpec(name="g", params={}, calibration_fpr_target=-0.1)
    with pytest.raises(ValidationError):
        GuardrailSpec(name="g", params={}, calibration_fpr_target=1.1)


# ---------------------------------------------------------------------
# WorldConfig
# ---------------------------------------------------------------------


def test_world_config_guardrail_stack_normalization():
    spec = GuardrailSpec(name="g", params={})

    # None -> []
    wc_none = WorldConfig(world_id=WorldBit.BASELINE, guardrail_stack=None)  # type: ignore[arg-type]
    assert wc_none.guardrail_stack == []

    # single spec
    wc_single = WorldConfig(world_id=WorldBit.BASELINE, guardrail_stack=spec)  # type: ignore[arg-type]
    assert wc_single.guardrail_stack == [spec]

    # mapping
    wc_map = WorldConfig(world_id=WorldBit.BASELINE, guardrail_stack={"name": "g2", "params": {}})  # type: ignore[arg-type]
    assert isinstance(wc_map.guardrail_stack[0], GuardrailSpec)

    # list of mixed entries
    wc_list = WorldConfig(
        world_id=WorldBit.BASELINE,
        guardrail_stack=[spec, {"name": "g3", "params": {}}],
    )
    assert len(wc_list.guardrail_stack) == 2
    assert isinstance(wc_list.guardrail_stack[0], GuardrailSpec)
    assert isinstance(wc_list.guardrail_stack[1], GuardrailSpec)


def test_world_config_env_hash_changes_with_stack_or_world():
    spec = GuardrailSpec(name="g", params={})
    wc1 = WorldConfig(world_id=WorldBit.BASELINE, guardrail_stack=[spec])
    wc2 = WorldConfig(world_id=WorldBit.PROTECTED, guardrail_stack=[spec])
    wc3 = WorldConfig(world_id=WorldBit.BASELINE, guardrail_stack=[])

    assert wc1.env_hash != wc2.env_hash
    assert wc1.env_hash != wc3.env_hash


def test_world_config_baseline_success_rate_bounds():
    WorldConfig(world_id=WorldBit.BASELINE, guardrail_stack=[], baseline_success_rate=0.0)
    WorldConfig(world_id=WorldBit.BASELINE, guardrail_stack=[], baseline_success_rate=1.0)
    with pytest.raises(ValidationError):
        WorldConfig(world_id=WorldBit.BASELINE, guardrail_stack=[], baseline_success_rate=-0.1)
    with pytest.raises(ValidationError):
        WorldConfig(world_id=WorldBit.BASELINE, guardrail_stack=[], baseline_success_rate=1.1)


# ---------------------------------------------------------------------
# ExperimentConfig
# ---------------------------------------------------------------------


def _minimal_guardrail_cfg() -> Dict[str, Sequence[GuardrailSpec]]:
    return {"cfg": [GuardrailSpec(name="g", params={})]}


def test_experiment_config_basic_and_iso():
    cfg = ExperimentConfig(
        experiment_id="exp1",
        n_sessions=10,
        attack_strategies=["a1", "a2"],
        guardrail_configs=_minimal_guardrail_cfg(),
        random_seed="123",
    )
    assert cfg.n_sessions == 10
    assert cfg.attack_strategies == ["a1", "a2"]
    assert list(cfg.guardrail_configs.keys()) == ["cfg"]
    assert isinstance(cfg.guardrail_configs["cfg"][0], GuardrailSpec)
    assert cfg.iso_time == _iso_from_unix(cfg.created_at)
    assert cfg.random_seed == 123


def test_experiment_config_attack_strategies_normalization_and_errors():
    cfg = ExperimentConfig(
        experiment_id="exp2",
        n_sessions=5,
        attack_strategies="single",
        guardrail_configs=_minimal_guardrail_cfg(),
    )
    assert cfg.attack_strategies == ["single"]

    with pytest.raises(ValueError):
        ExperimentConfig(
            experiment_id="exp3",
            n_sessions=5,
            attack_strategies=[],  # type: ignore[arg-type]
            guardrail_configs=_minimal_guardrail_cfg(),
        )


def test_experiment_config_guardrail_configs_errors():
    with pytest.raises(ValueError):
        ExperimentConfig(
            experiment_id="exp",
            n_sessions=1,
            attack_strategies=["a"],
            guardrail_configs=None,  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError):
        ExperimentConfig(
            experiment_id="exp",
            n_sessions=1,
            attack_strategies=["a"],
            guardrail_configs=["not", "mapping"],  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError):
        ExperimentConfig(
            experiment_id="exp",
            n_sessions=1,
            attack_strategies=["a"],
            guardrail_configs={},
        )


# ---------------------------------------------------------------------
# CCResult
# ---------------------------------------------------------------------


def test_ccresult_ci_and_level_validation():
    CCResult(j_empirical=0.1, cc_max=0.2, delta_add=0.1, confidence_interval=(0.0, 0.5))
    with pytest.raises(ValueError):
        CCResult(j_empirical=0.1, cc_max=0.2, delta_add=0.1, confidence_interval=(0.6, 0.2))
    with pytest.raises(ValidationError):
        CCResult(j_empirical=0.1, cc_max=0.2, delta_add=0.1, ci_level=0.5)
    with pytest.raises(ValidationError):
        CCResult(j_empirical=0.1, cc_max=0.2, delta_add=0.1, ci_level=1.0)


def test_ccresult_bootstrap_normalization_list_and_string_error():
    r = CCResult(
        j_empirical=0.1,
        cc_max=0.2,
        delta_add=0.1,
        bootstrap_samples=[0.1, float("nan"), 0.2, float("inf")],
    )
    assert r.bootstrap_samples == [0.1, 0.2]
    # strings are rejected explicitly
    with pytest.raises(ValidationError):
        CCResult(j_empirical=0.1, cc_max=0.2, delta_add=0.1, bootstrap_samples="abc")  # type: ignore[arg-type]


@pytest.mark.skipif(not NUMPY_AVAILABLE, reason="numpy not available")
def test_ccresult_bootstrap_normalization_numpy():
    arr = np.array([0.1, 0.2, float("nan")], dtype=float)  # type: ignore[attr-defined]
    r = CCResult(j_empirical=0.1, cc_max=0.2, delta_add=0.1, bootstrap_samples=arr)
    assert r.bootstrap_samples == [0.1, 0.2]
    # serializer returns list of floats
    dumped = r.model_dump(mode="json")
    assert isinstance(dumped["bootstrap_samples"], list)


# ---------------------------------------------------------------------
# AttackStrategy
# ---------------------------------------------------------------------


def test_attack_strategy_params_and_vocab_normalization():
    s1 = AttackStrategy(name="s", params=None, vocabulary=None)  # type: ignore[arg-type]
    assert s1.params == {}
    assert s1.vocabulary == []

    s2 = AttackStrategy(name="s", params={"k": 1}, vocabulary="token")
    assert s2.vocabulary == ["token"]

    class MyMap(Mapping[str, Any]):
        def __getitem__(self, k: str) -> Any:
            return {"a": 1}[k]

        def __iter__(self):
            return iter({"a": 1})

        def __len__(self) -> int:
            return 1

    s3 = AttackStrategy(name="s", params=MyMap())
    assert s3.params == {"a": 1}


def test_attack_strategy_params_invalid_type():
    with pytest.raises(TypeError):
        AttackStrategy(name="s", params=123)  # type: ignore[arg-type]


def test_attack_strategy_success_threshold_bounds():
    AttackStrategy(name="s", params={}, success_threshold=0.0)
    AttackStrategy(name="s", params={}, success_threshold=1.0)
    with pytest.raises(ValidationError):
        AttackStrategy(name="s", params={}, success_threshold=-0.1)
    with pytest.raises(ValidationError):
        AttackStrategy(name="s", params={}, success_threshold=1.1)


# ---------------------------------------------------------------------
# Roundtrip smoke tests for all models
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "model_cls, data",
    [
        (
            AttackResult,
            {
                "world_bit": WorldBit.BASELINE,
                "success": True,
                "attack_id": "id",
                "transcript_hash": _hash_text("x"),
                "guardrails_applied": "ga",
                "rng_seed": 42,
            },
        ),
        (
            GuardrailSpec,
            {
                "name": "spec",
                "params": {"k": "v"},
            },
        ),
        (
            WorldConfig,
            {
                "world_id": WorldBit.BASELINE,
                "guardrail_stack": [],
            },
        ),
        (
            ExperimentConfig,
            {
                "experiment_id": "exp1",
                "n_sessions": 10,
                "attack_strategies": ["basic"],
                "guardrail_configs": {"cfg": []},
            },
        ),
        (
            CCResult,
            {
                "j_empirical": 0.4,
                "cc_max": 0.6,
                "delta_add": 0.2,
            },
        ),
        (
            AttackStrategy,
            {
                "name": "strat",
                "params": {"foo": "bar"},
            },
        ),
    ],
)
def test_roundtrip_all_models(model_cls, data):
    inst = model_cls(**data)
    json_str = inst.model_dump_json()
    inst2 = model_cls.model_validate_json(json_str)
    assert inst.model_dump() == inst2.model_dump()
