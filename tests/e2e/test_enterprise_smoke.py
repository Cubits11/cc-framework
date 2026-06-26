from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from cc.enterprise.aws_reference import (
    deploy_emulated_reference,
    generate_smoke_evidence_bundle,
    upload_evidence_bundle,
    verify_bundle,
)


def test_enterprise_smoke_pipeline(tmp_path: Path) -> None:
    boto3 = pytest.importorskip("boto3")
    moto = pytest.importorskip("moto")

    with moto.mock_aws():
        _run_enterprise_smoke_pipeline(tmp_path, boto3)


def _run_enterprise_smoke_pipeline(tmp_path: Path, boto3: object) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    dashboard_dir = repo_root / "apps" / "dashboard"
    if not (dashboard_dir / "node_modules").exists():
        pytest.skip("apps/dashboard dependencies are not installed")

    session = boto3.Session(region_name="us-east-1")
    resources = deploy_emulated_reference(
        boto3_session=session,
        prefix="cc-smoke",
        region_name="us-east-1",
    )
    generated = generate_smoke_evidence_bundle(tmp_path)
    uploaded = upload_evidence_bundle(
        boto3_session=session,
        resources=resources,
        dashboard_bundle_path=Path(generated["dashboard_bundle_path"]),
        sequence_number=1,
    )

    backend_result = verify_bundle(
        boto3_session=session,
        resources=resources,
        bundle_id=uploaded["bundle_id"],
    )
    assert backend_result["ok"] is True

    s3 = session.client("s3", region_name="us-east-1")
    uploaded_bundle_path = tmp_path / "uploaded_enterprise_bundle.json"
    uploaded_bundle_path.write_bytes(
        s3.get_object(Bucket=resources.bucket_name, Key=uploaded["object_key"])["Body"].read()
    )

    env = {**os.environ, "ENTERPRISE_BUNDLE_PATH": str(uploaded_bundle_path)}
    subprocess.run(
        ["npm", "run", "smoke"],
        cwd=dashboard_dir,
        env=env,
        check=True,
        text=True,
    )
