import json, sys, subprocess
from pathlib import Path

def test_methods_cli_smoke(tmp_path):
    out = tmp_path / "week3.json"
    cmd = [
        sys.executable, "-m", "cc.cartographer.cli", "methods",
        "--D", "0.55",
        "--tpr-a", "0.72", "--tpr-b", "0.65",
        "--fpr-a", "0.035", "--fpr-b", "0.050",
        "--n1", "200", "--n0", "200",
        "--k1", "124", "--k0", "18",
        "--alpha-cap", "0.05", "--delta", "0.05",
        "--json-out", str(out),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    payload = json.loads(out.read_text())
    assert payload["point"]["cc_hat"] > 0
    assert "ci" in payload and "wilson" in payload

def test_methods_cli_infeasible_alpha(tmp_path):
    # α below max(FPR) should error out
    cmd = [
        sys.executable, "-m", "cc.cartographer.cli", "methods",
        "--D", "0.55",
        "--tpr-a", "0.72", "--tpr-b", "0.65",
        "--fpr-a", "0.035", "--fpr-b", "0.050",
        "--n1", "200", "--n0", "200",
        "--k1", "124", "--k0", "18",
        "--alpha-cap", "0.04", "--delta", "0.05",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode != 0
    assert "Policy cap makes I0 empty" in (res.stderr + res.stdout)
