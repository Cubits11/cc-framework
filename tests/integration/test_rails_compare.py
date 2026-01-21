import subprocess
import sys

import pandas as pd


def test_rails_demo_runs(tmp_path):
    csv = tmp_path / "toy.csv"
    csv.write_text("id,label,rail_a_score,rail_b_score\nx,1,0.9,0.2\ny,0,0.1,0.8\n")
    out = tmp_path / "out.csv"
    subprocess.check_call(
        [sys.executable, "scripts/rails_compare.py", "--csv", str(csv), "--out", str(out)]
    )
    df = pd.read_csv(out)
    assert {"A_J", "B_J", "Combo_any_J", "Indep_any_J"}.issubset(df.columns)
