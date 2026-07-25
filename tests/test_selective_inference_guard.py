"""Selective-inference guard: a searched maximum may never carry a confidence claim.

WHY THIS TEST EXISTS
--------------------
`run_dependence_search` uses simulated annealing to find the prompt perturbation
that MAXIMISES the joint tail co-failure rate, then bootstrapped a CI from that
winner's own failures and handed both to `cliff_certificate`. The certificate
then said, verbatim:

    "At 95.0% confidence, max(lambda_L, lambda_U) exceeds 0.200; the data
     support super-critical co-failure."

That sentence is false by construction. The estimate is the maximum over many
candidates, so it is biased upward, and a bootstrap around it does not attain
its nominal level for the true parameter (winner's curse / selective inference).
For a framework whose entire value proposition is refusing false confidence,
that was the philosophy eating itself.

The fix is a tier that post-selection data cannot escape: provenance
"post-selection" forces regime "discovery-only", which names a hypothesis and
explicitly disclaims any regime claim. Only data held out from the search can
certify. These tests pin that behaviour in both directions -- the guard must
fire on searched data, and must NOT fire on genuinely held-out data (a guard
that always refuses is useless, not safe).
"""

from __future__ import annotations

import pytest

from cc.kernel.cliff import cliff_certificate


CRITICAL = 0.20


def test_post_selection_ci_cannot_yield_a_regime_claim() -> None:
    """An interval entirely above the threshold still gets no confidence claim."""
    # This CI would read "super-critical" under the old code path.
    cert = cliff_certificate(
        {"lambda_any": 0.63},
        {"lambda_any": (0.55, 0.71), "confidence_level": 0.95},
        critical_value=CRITICAL,
        provenance="post-selection",
    )
    assert cert.regime == "discovery-only"
    assert cert.provenance == "post-selection"
    # The statement must not assert support for a regime.
    assert "support super-critical" not in cert.statement
    assert "DISCOVERY ONLY" in cert.statement
    assert "no nominal coverage" in cert.statement
    # It must tell the reader how to actually establish the claim.
    assert "held out" in cert.falsifier


def test_confirmatory_ci_still_certifies_normally() -> None:
    """The guard must not neuter legitimate held-out inference."""
    cert = cliff_certificate(
        {"lambda_any": 0.63},
        {"lambda_any": (0.55, 0.71), "confidence_level": 0.95},
        critical_value=CRITICAL,
        provenance="confirmatory",
    )
    assert cert.regime == "super-critical"
    assert cert.provenance == "confirmatory"
    assert "support super-critical" in cert.statement


def test_confirmatory_is_the_default_but_sub_critical_still_works() -> None:
    cert = cliff_certificate(
        {"lambda_any": 0.02},
        {"lambda_any": (0.00, 0.05), "confidence_level": 0.95},
        critical_value=CRITICAL,
    )
    assert cert.regime == "sub-critical"
    assert cert.provenance == "confirmatory"


def test_unknown_provenance_is_rejected() -> None:
    with pytest.raises(ValueError, match="provenance"):
        cliff_certificate(
            {"lambda_any": 0.5},
            {"lambda_any": (0.4, 0.6), "confidence_level": 0.95},
            critical_value=CRITICAL,
            provenance="probably-fine",  # type: ignore[arg-type]
        )


def test_search_result_declares_post_selection_provenance() -> None:
    """The defect site itself: the search's own certificate must be discovery-only."""
    import inspect

    from cc.redteam import dependence_search

    src = inspect.getsource(dependence_search.run_dependence_search)
    assert 'provenance="post-selection"' in src, (
        "run_dependence_search must declare that its certificate is built on the "
        "argmax of its own search; without this the certificate emits a confidence "
        "claim the design cannot support."
    )
