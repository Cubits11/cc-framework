import json
from pathlib import Path

from cc.analysis.generate_figures import main as gen_main


def _make_audit_jsonl(tmp_path: Path, n=12):
    p = tmp_path / "audit.jsonl"
    with p.open("w", encoding="utf-8") as f:
        for i in range(n):
            rec = {
                "cfg": {"epsilon": 0.1 + 0.02 * i, "T": 5 + i, "samples": 100 + i},
                "metrics": {"CC_max": 1.0 + 0.01 * i},
            }
            f.write(json.dumps(rec) + "\n")
    return p


def test_generate_figures_legacy_end_to_end(tmp_path):
    hist = _make_audit_jsonl(tmp_path, n=15)
    fig_dir = tmp_path / "figs"
    out_dir = tmp_path / "out"
    gen_main(["--history", str(hist), "--fig-dir", str(fig_dir), "--out-dir", str(out_dir)])

    # Legacy artifacts should exist
    assert (fig_dir / "phase_diagram.pdf").exists()
    assert (fig_dir / "cc_convergence.pdf").exists()
    assert (fig_dir / "roc_comparison.pdf").exists()
    assert (out_dir / "summary.csv").exists()
