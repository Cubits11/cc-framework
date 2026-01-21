# tests/unit/test_week3_ci_and_audit.py
from pathlib import Path

import numpy as np

from cc.analysis.cc_estimation import cc_confint_newcombe
from cc.analysis.generate_figures import plot_cc_ci_comparison, plot_fh_heatmap
from cc.cartographer.audit import audit_fh_ceiling_by_index
from cc.cartographer.bounds import cc_confint, envelope_over_rocs, fh_intervals


def _roc_line(n=20, bias=1.8, seed=1):
    np.random.default_rng(seed)
    fpr = np.linspace(0, 1, n)
    tpr = 1 - (1 - fpr) ** bias
    return np.column_stack([fpr, tpr])


def test_bernstein_width_increases_with_tighter_delta():
    # fixed FH intervals
    tpr_a, tpr_b, fpr_a, fpr_b = 0.8, 0.7, 0.12, 0.10
    I1, I0 = fh_intervals(tpr_a, tpr_b, fpr_a, fpr_b, alpha_cap=None)
    # empirical
    p1_hat, p0_hat = 0.42, 0.11
    n1 = n0 = 600
    D = 0.8
    lo1, hi1 = cc_confint(n1, n0, p1_hat, p0_hat, D, I1, I0, delta=0.10)
    lo2, hi2 = cc_confint(n1, n0, p1_hat, p0_hat, D, I1, I0, delta=0.01)
    w1 = hi1 - lo1
    w2 = hi2 - lo2
    assert w2 >= w1  # smaller delta => wider interval (more conservative)


def test_newcombe_and_bernstein_contain_point():
    p1_hat, p0_hat = 0.42, 0.11
    n1 = n0 = 600
    D = 0.8
    tpr_a, tpr_b, fpr_a, fpr_b = 0.8, 0.7, 0.12, 0.10
    I1, I0 = fh_intervals(tpr_a, tpr_b, fpr_a, fpr_b)
    loB, hiB = cc_confint(n1, n0, p1_hat, p0_hat, D, I1, I0, delta=0.05)
    x1, x0 = round(p1_hat * n1), round(p0_hat * n0)
    loN, hiN = cc_confint_newcombe(x1, n1, x0, n0, D, alpha=0.05)
    cc_hat = (1 - (p1_hat - p0_hat)) / D
    assert loB <= cc_hat <= hiB
    assert loN <= cc_hat <= hiN


def test_fh_auditor_no_violations_on_argmax():
    A = _roc_line(30, bias=1.7, seed=3)
    B = _roc_line(25, bias=2.3, seed=5)
    _, J = envelope_over_rocs(A, B, comp="AND", add_anchors=True)
    ia, ib = np.unravel_index(int(np.argmax(J)), J.shape)
    j_cap = float(J[ia, ib])
    violations = audit_fh_ceiling_by_index(
        A, B, [(ia, ib, j_cap - 1e-15)], comp="AND", add_anchors=True
    )
    assert violations == []


def test_figure_helpers(tmp_path: Path):
    A = _roc_line(22, bias=1.6, seed=7)
    B = _roc_line(18, bias=2.0, seed=11)
    out1 = tmp_path / "fh_heat.png"
    out2 = tmp_path / "cc_cis.png"
    # heatmap
    info = plot_fh_heatmap(A, B, outpath=str(out1))
    assert out1.exists()
    assert "J_max" in info and "argmax" in info
    # cc cis
    res = plot_cc_ci_comparison(
        p1_hat=0.41,
        n1=500,
        p0_hat=0.12,
        n0=500,
        D=0.8,
        tpr_a=0.78,
        tpr_b=0.70,
        fpr_a=0.10,
        fpr_b=0.12,
        outpath=str(out2),
    )
    assert out2.exists()
    assert res["bernstein"][0] <= res["cc_hat"] <= res["bernstein"][1]
    assert res["newcombe"][0] <= res["cc_hat"] <= res["newcombe"][1]
