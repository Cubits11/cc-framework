from cc.adapters.base import build_audit_payload


def test_audit_payload_event_id_override_is_stable():
    payload_a = build_audit_payload(
        prompt="sensitive prompt",
        response=None,
        adapter_name="TestAdapter",
        adapter_version="1.0",
        parameters={"mode": "test"},
        decision="allow",
        category=None,
        rationale=None,
        started_at=100.0,
        completed_at=101.0,
        event_id="fixed-event-id",
    )
    payload_b = build_audit_payload(
        prompt="sensitive prompt",
        response=None,
        adapter_name="TestAdapter",
        adapter_version="1.0",
        parameters={"mode": "test"},
        decision="allow",
        category=None,
        rationale=None,
        started_at=200.0,
        completed_at=201.0,
        event_id="fixed-event-id",
    )

    assert payload_a["event_id"] == "fixed-event-id"
    assert payload_b["event_id"] == "fixed-event-id"
    assert payload_a["event_hash"] != ""
    assert payload_b["event_hash"] != ""
