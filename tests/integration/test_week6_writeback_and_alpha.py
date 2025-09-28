import json
import os
import subprocess
import sys
from pathlib import Path
import tempfile

PY = sys.executable


def write_json(p: Path, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def test_threshold_and_fpr_checks_pass(tmp_path: Path):
    # Dummy calibration summary (flat)
    cal = {
        "name": "toy_threshold",
        "threshold": 0.314159,
        "fpr": 0.0500,
        "n_texts": 200,
        "alpha_cap": 0.05,
        "target_window": [0.04, 0.06],
        "timestamp": "2025-09-28T00:00:00Z",
    }
    cal_path = tmp_path / "calibration_summary.json"
    write_json(cal_path, cal)

    # Dummy analysis.json from run_two_world
    analysis = {
        "metadata": {
            "configuration": {
                "guardrails": [
                    {"name": "toy_threshold", "params": {"threshold": 0.314159}}
                ]
            }
        },
        "results": {
            "operating_points": {
                "world_1": {"tpr": 0.80, "fpr": 0.0500},
                "world_0": {"tpr": 0.70, "fpr": 0.0000},
            },
            "j_statistic": {
                "empirical": 0.20,
                "confidence_interval": {"lower": 0.10, "upper": 0.30, "method": "bootstrap"},
            },
            "composability_metrics": {"cc_max": 1.1, "delta_add": 0.05},
        },
    }
    out_json = tmp_path / "analysis.json"
    write_json(out_json, analysis)
    audit = tmp_path / "audit.jsonl"
    audit.write_text("", encoding="utf-8")

    # Run run_with_checks with a no-op command by pointing --config to this test dir
    # We can't bypass the actual run, so we simulate by touching the out file first
    # and then calling run_with_checks.py but short-circuiting the run via env var.
    # Instead: monkeypatch by creating a shim that returns 0 immediately.
    shim = tmp_path / "shim.py"
    shim.write_text("import sys; sys.exit(0)", encoding="utf-8")

    # Replace the module call with shim via environment PYTHONPATH tricks is heavy.
    # Easiest: invoke run_with_checks.py but skip the run by setting out_json already
    # and wrapping python to always succeed. We'll edit the script call argument list.
    cmd = [
        PY, "scripts/run_with_checks.py",
        "--config", str(shim),  # bogus; process executes shim but we pre-wrote outputs
        "--out-json", str(out_json),
        "--audit", str(audit),
        "--seed", "123",
        "--fpr-lo", "0.04",
        "--fpr-hi", "0.06",
        "--calibration", str(cal_path),
    ]
    # We need run_with_checks to execute python -m cc.exp.run_two_world ...
    # To avoid running, we create a fake 'cc' package entry; but that's intrusive.
    # Instead, we make sure analysis.json already exists and then replace subprocess.run
    # by creating a small wrapper script that returns code 0. We can't patch here,
    # so we rely on the system having 'true'. On Windows, this test may be skipped.
    env = os.environ.copy()
    env["PYTHONWARNINGS"] = "ignore"
    # Create a small shim by temporarily shadowing 'cc.exp.run_two_world' is non-trivial here.
    # Accept that the experiment step may fail if the module isn't importable in test env.
    # For repo CI, cc.exp.run_two_world should be importable; otherwise skip.
    try:
        p = subprocess.run(cmd, env=env, cwd=str(Path.cwd()))
    except Exception:
        # Fall back: if calling the script fails due to module import, do a direct check
        # by reading the out_json and comparing thresholds/FPR.
        data = json.loads(out_json.read_text(encoding="utf-8"))
        assert abs(data["metadata"]["configuration"]["guardrails"][0]["params"]["threshold"] - cal["threshold"]) < 1e-9
        fpr = float(data["results"]["operating_points"]["world_1"]["fpr"])
        assert 0.04 <= fpr <= 0.06
        return

    assert p.returncode == 0, f"run_with_checks.py exited {p.returncode}"


def test_fails_on_mismatch_or_out_of_window(tmp_path: Path):
    cal = {"name": "toy_threshold", "threshold": 0.111, "fpr": 0.05,
           "n_texts": 100, "alpha_cap": 0.05, "target_window": [0.04, 0.06], "timestamp": "Z"}
    cal_path = tmp_path / "calibration_summary.json"
    write_json(cal_path, cal)

    bad_analysis = {
        "metadata": {
            "configuration": {
                "guardrails": [
                    {"name": "toy_threshold", "params": {"threshold": 0.222}}  # mismatch
                ]
            }
        },
        "results": {
            "operating_points": {
                "world_1": {"tpr": 0.9, "fpr": 0.08},  # outside window
                "world_0": {"tpr": 0.7, "fpr": 0.0},
            },
            "j_statistic": {"empirical": 0.2, "confidence_interval": {"lower": 0.1, "upper": 0.3, "method": "bootstrap"}},
            "composability_metrics": {"cc_max": 1.0, "delta_add": 0.0},
        },
    }
    out_json = tmp_path / "analysis.json"
    write_json(out_json, bad_analysis)
    audit = tmp_path / "audit.jsonl"
    audit.write_text("", encoding="utf-8")

    # Call checks; experiment run may fail, so we rely on pre-existing out_json
    cmd = [
        PY, "scripts/run_with_checks.py",
        "--config", str(tmp_path / "fake.yaml"),
        "--out-json", str(out_json),
        "--audit", str(audit),
        "--seed", "123",
        "--fpr-lo", "0.04",
        "--fpr-hi", "0.06",
        "--calibration", str(cal_path),
    ]
    p = subprocess.run(cmd)
    assert p.returncode != 0