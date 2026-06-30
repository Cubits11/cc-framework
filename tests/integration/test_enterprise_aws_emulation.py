from __future__ import annotations

import importlib
import importlib.util
import json
import os
from pathlib import Path
from types import ModuleType

import pytest

from cc.enterprise.aws_reference import (
    deploy_emulated_reference,
    generate_smoke_evidence_bundle,
    upload_evidence_bundle,
    verify_bundle,
)

_STRICT_ENTERPRISE_ENV = "CC_ENTERPRISE_STRICT"


def _enterprise_dependency(module_name: str) -> ModuleType:
    message = (
        f"{module_name} is required for enterprise validation. "
        "Run `make enterprise-smoke` or install `.[enterprise,test]`."
    )
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        if os.environ.get(_STRICT_ENTERPRISE_ENV) == "1":
            pytest.fail(message, pytrace=False)
        pytest.skip(message, allow_module_level=False)
        raise AssertionError("unreachable") from exc


def test_enterprise_reference_uses_real_emulated_aws_api_surface(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    boto3 = _enterprise_dependency("boto3")
    botocore_exceptions = _enterprise_dependency("botocore.exceptions")
    moto = _enterprise_dependency("moto")
    ClientError = botocore_exceptions.ClientError

    with moto.mock_aws():
        _run_emulated_reference_test(tmp_path, monkeypatch, boto3, ClientError)


def _run_emulated_reference_test(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    boto3: object,
    ClientError: type[Exception],
) -> None:
    session = boto3.Session(region_name="us-east-1")
    resources = deploy_emulated_reference(
        boto3_session=session,
        prefix="cc-test",
        region_name="us-east-1",
    )

    s3 = session.client("s3", region_name="us-east-1")
    lock_config = s3.get_object_lock_configuration(Bucket=resources.bucket_name)
    assert lock_config["ObjectLockConfiguration"]["ObjectLockEnabled"] == "Enabled"

    generated = generate_smoke_evidence_bundle(tmp_path)
    uploaded = upload_evidence_bundle(
        boto3_session=session,
        resources=resources,
        dashboard_bundle_path=Path(generated["dashboard_bundle_path"]),
        sequence_number=1,
    )

    verification = verify_bundle(
        boto3_session=session,
        resources=resources,
        bundle_id=uploaded["bundle_id"],
    )
    assert verification["ok"] is True
    assert verification["inclusion_ok"] is True
    assert verification["consistency_ok"] is True
    assert verification["kms_signature_ok"] is True

    with pytest.raises(ClientError, match="ConditionalCheckFailed"):
        upload_evidence_bundle(
            boto3_session=session,
            resources=resources,
            dashboard_bundle_path=Path(generated["dashboard_bundle_path"]),
            sequence_number=1,
        )

    monkeypatch.setenv("RUN_METADATA_TABLE", resources.metadata_table_name)
    monkeypatch.setenv("EVIDENCE_BUCKET", resources.bucket_name)
    monkeypatch.setenv("ATTESTATION_KEY_ID", resources.kms_key_id)
    monkeypatch.setenv("AWS_REGION", resources.region_name)
    handler = _load_lambda_handler()
    response = handler({"pathParameters": {"bundle_id": uploaded["bundle_id"]}}, None)
    body = json.loads(response["body"])

    assert response["statusCode"] == 200
    assert body["ok"] is True
    assert body["bundle_id"] == uploaded["bundle_id"]


def _load_lambda_handler():
    handler_path = Path(__file__).resolve().parents[2] / "infra" / "lambda" / "verify_handler.py"
    spec = importlib.util.spec_from_file_location("cc_enterprise_verify_handler", handler_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.handler
