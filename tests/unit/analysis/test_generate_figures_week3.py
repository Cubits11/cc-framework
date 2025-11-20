from pathlib import Path
from cc.analysis.generate_figures import main as gen_main

def test_week3_figure_optional_path(tmp_path):
    # Minimal legacy audit to satisfy main()
    hist = tmp_path / "audit.jsonl"
    hist.write_text('{"cfg": {"epsilon": 0.1, "T": 5, "samples": 120}, "metrics": {"CC_max": 1.02}}\n')

    fig_dir = tmp_path / "figs"
    out_dir = tmp_path / "out"

    args = [
        "--history", str(hist),
        "--fig-dir", str(fig_dir),
        "--out-dir", str(out_dir),
        # Week-3 extras
        "--week3-figure",
        "--week3-theta", "0.72,0.035,0.65,0.050",
        "--week3-I1", "0.37,0.65",
        "--week3-I0", "0.04,0.08",
        "--week3-D", "0.55",
        "--week3-n1", "200",
        "--week3-n0", "220",
        "--week3-delta", "0.05",
        "--week3-alpha-cap", "0.05",
    ]
    gen_main(args)
    assert (fig_dir / "fig_week3_roc_fh.png").exists()
