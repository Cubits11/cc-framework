"""Tests for ``cc-guard``: the inference guards, and their decision table.

The table is data a caller in another language reimplements from. If it drifts
from the code, every such caller silently enforces the wrong rule -- so the
agreement between the two is asserted here, case by case.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from cc.cli.guard import DECISION_TABLE, MIN_N_FOR_PROPORTION_INTERVAL, check, main
from cc.kernel.cliff import CliffCertificate, cliff_certificate


def _run(payload: dict, *args: str) -> tuple[int, dict]:
    proc = subprocess.run(
        [sys.executable, "-m", "cc.cli.guard", *args],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode, json.loads(proc.stdout)


# --- the guard refuses what it must -------------------------------------------


def test_post_selection_interval_is_refused():
    verdict = check(
        {
            "guard": "post_selection_interval",
            "provenance": "post-selection",
            "estimate": 0.42,
            "ci_low": 0.21,
            "ci_high": 0.63,
        }
    )
    assert verdict["verdict"] == "discovery-only"
    assert verdict["confidence_claim_permitted"] is False
    assert verdict["remedy"]


def test_confirmatory_interval_is_permitted():
    verdict = check(
        {
            "guard": "post_selection_interval",
            "provenance": "confirmatory",
            "estimate": 0.42,
            "ci_low": 0.21,
            "ci_high": 0.63,
        }
    )
    assert verdict["verdict"] == "confirmatory"
    assert verdict["confidence_claim_permitted"] is True


@pytest.mark.parametrize("provenance", [None, "", "maybe", "CONFIRMATORY", 7])
def test_unknown_provenance_fails_closed(provenance):
    """A guard that defaults to permitting a claim is not a guard."""
    verdict = check({"guard": "post_selection_interval", "provenance": provenance})
    assert verdict["verdict"] == "refused"
    assert verdict["confidence_claim_permitted"] is False


def test_census_interval_is_refused():
    verdict = check({"guard": "census_interval", "provenance": "census"})
    assert verdict["verdict"] == "no-interval"
    assert verdict["confidence_claim_permitted"] is False


@pytest.mark.parametrize("n", [0, 1, 2, 29])
def test_small_sample_interval_is_refused(n):
    verdict = check({"guard": "census_interval", "provenance": "sampled", "n": n})
    assert verdict["verdict"] == "no-interval"
    assert verdict["confidence_claim_permitted"] is False


@pytest.mark.parametrize("n", [30, 31, 1000])
def test_adequate_sample_interval_is_permitted(n):
    verdict = check({"guard": "census_interval", "provenance": "sampled", "n": n})
    assert verdict["verdict"] == "interval-permitted"
    assert verdict["confidence_claim_permitted"] is True


@pytest.mark.parametrize("n", [None, -1, 3.5, "30", True])
def test_sampled_proportion_without_a_valid_denominator_is_refused(n):
    """No proportion without a denominator."""
    verdict = check({"guard": "census_interval", "provenance": "sampled", "n": n})
    assert verdict["verdict"] == "refused"


def test_unknown_guard_is_refused():
    verdict = check({"guard": "vibes"})
    assert verdict["verdict"] == "refused"
    assert verdict["confidence_claim_permitted"] is False


# --- the table and the code must agree ----------------------------------------


def test_decision_table_matches_the_implementation():
    """The table is what a non-Python caller enforces. It must not drift."""
    for guard in DECISION_TABLE["guards"]:
        guard_id = guard["id"]
        for rule in guard["rules"]:
            request = {"guard": guard_id, **rule["when"]}
            # The census guard's threshold rule is expressed as `n_below`.
            n_below = request.pop("n_below", None)
            if n_below is not None:
                request["n"] = n_below - 1
            elif guard_id == "census_interval" and request.get("provenance") == "sampled":
                request["n"] = MIN_N_FOR_PROPORTION_INTERVAL
            verdict = check(request)
            assert verdict["verdict"] == rule["verdict"], (guard_id, rule["when"])
            assert verdict["confidence_claim_permitted"] == rule["confidence_claim_permitted"], (
                guard_id,
                rule["when"],
            )


def test_decision_table_threshold_matches_the_constant():
    census = next(g for g in DECISION_TABLE["guards"] if g["id"] == "census_interval")
    thresholds = [r["when"]["n_below"] for r in census["rules"] if "n_below" in r["when"]]
    assert thresholds == [MIN_N_FOR_PROPORTION_INTERVAL]


def test_decision_table_is_json_native():
    assert json.loads(json.dumps(DECISION_TABLE)) == DECISION_TABLE


def test_decision_table_carries_non_claims():
    assert DECISION_TABLE["non_claims"]
    assert any("does not mean" in c for c in DECISION_TABLE["non_claims"])


# --- agreement with the original in-kernel guard ------------------------------


def test_guard_agrees_with_cliff_certificate_on_post_selection():
    """`cc-guard` must reach the same verdict as the guard it exposes.

    `cc.kernel.cliff.cliff_certificate` is the original refusal, reachable only
    from Python. If the CLI ever permitted what the kernel refuses, every
    cross-language caller would be worse off than a Python one.
    """
    certificate = cliff_certificate(0.42, (0.21, 0.63), provenance="post-selection")
    assert isinstance(certificate, CliffCertificate)
    assert certificate.regime == "discovery-only"

    verdict = check(
        {
            "guard": "post_selection_interval",
            "provenance": "post-selection",
            "estimate": 0.42,
            "ci_low": 0.21,
            "ci_high": 0.63,
        }
    )
    assert verdict["verdict"] == certificate.regime
    assert verdict["confidence_claim_permitted"] is False


def test_guard_and_cliff_certificate_agree_that_confirmatory_is_not_refused():
    """The other direction: the CLI must not refuse what the kernel allows."""
    certificate = cliff_certificate(0.42, (0.21, 0.63), provenance="confirmatory")
    assert certificate.regime != "discovery-only"

    verdict = check(
        {
            "guard": "post_selection_interval",
            "provenance": "confirmatory",
            "estimate": 0.42,
            "ci_low": 0.21,
            "ci_high": 0.63,
        }
    )
    assert verdict["confidence_claim_permitted"] is True


# --- the process boundary -----------------------------------------------------


def test_cli_check_refuses_and_can_gate_a_shell_caller():
    code, verdict = _run(
        {"guard": "post_selection_interval", "provenance": "post-selection"},
        "check",
        "--strict-exit",
    )
    assert code == 1
    assert verdict["confidence_claim_permitted"] is False


def test_cli_check_permits_with_exit_zero():
    code, verdict = _run(
        {"guard": "post_selection_interval", "provenance": "confirmatory"},
        "check",
        "--strict-exit",
    )
    assert code == 0
    assert verdict["confidence_claim_permitted"] is True


def test_cli_rejects_malformed_json():
    proc = subprocess.run(
        [sys.executable, "-m", "cc.cli.guard", "check"],
        input="not json",
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2
    assert json.loads(proc.stdout)["confidence_claim_permitted"] is False


def test_cli_rejects_a_non_object_request():
    proc = subprocess.run(
        [sys.executable, "-m", "cc.cli.guard", "check"],
        input="[1, 2, 3]",
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2
    assert json.loads(proc.stdout)["verdict"] == "refused"


def test_cli_table_emits_the_decision_data():
    proc = subprocess.run(
        [sys.executable, "-m", "cc.cli.guard", "table"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert json.loads(proc.stdout) == DECISION_TABLE


def test_main_requires_a_subcommand():
    with pytest.raises(SystemExit):
        main([])


# --- in-process CLI coverage --------------------------------------------------
#
# The subprocess tests above exercise the real process boundary, which is what a
# non-Python caller actually uses. Coverage cannot see into a subprocess, so
# these drive main() in-process as well: without them cc/cli/guard.py reports
# ~64% while its CLI body is in fact fully exercised. Both kinds are kept --
# dropping the subprocess tests would stop testing the boundary that matters.


def test_main_check_refuses_in_process(monkeypatch, capsys):
    monkeypatch.setattr(
        "sys.stdin",
        __import__("io").StringIO(
            json.dumps({"guard": "post_selection_interval", "provenance": "post-selection"})
        ),
    )
    assert main(["check", "--strict-exit"]) == 1
    verdict = json.loads(capsys.readouterr().out)
    assert verdict["verdict"] == "discovery-only"


def test_main_check_permits_in_process(monkeypatch, capsys):
    monkeypatch.setattr(
        "sys.stdin",
        __import__("io").StringIO(
            json.dumps({"guard": "post_selection_interval", "provenance": "confirmatory"})
        ),
    )
    assert main(["check", "--strict-exit"]) == 0
    assert json.loads(capsys.readouterr().out)["confidence_claim_permitted"] is True


def test_main_check_without_strict_exit_returns_zero_even_when_refusing(monkeypatch, capsys):
    """The exit code is opt-in; the verdict is always in the payload."""
    monkeypatch.setattr(
        "sys.stdin",
        __import__("io").StringIO(json.dumps({"guard": "census_interval", "provenance": "census"})),
    )
    assert main(["check"]) == 0
    assert json.loads(capsys.readouterr().out)["confidence_claim_permitted"] is False


def test_main_check_on_empty_stdin_refuses(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", __import__("io").StringIO(""))
    assert main(["check", "--strict-exit"]) == 1
    assert json.loads(capsys.readouterr().out)["verdict"] == "refused"


def test_main_check_on_malformed_json_in_process(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", __import__("io").StringIO("{not json"))
    assert main(["check"]) == 2
    assert json.loads(capsys.readouterr().out)["confidence_claim_permitted"] is False


def test_main_check_on_non_object_in_process(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", __import__("io").StringIO("[1,2,3]"))
    assert main(["check"]) == 2
    assert json.loads(capsys.readouterr().out)["verdict"] == "refused"


def test_main_table_in_process(capsys):
    assert main(["table"]) == 0
    assert json.loads(capsys.readouterr().out) == DECISION_TABLE
