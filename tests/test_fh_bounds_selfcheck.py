import pytest

from theory.fh_bounds import verify_fh_bound_properties


def test_fh_bounds_internal_selfcheck():
    """
    Wraps verify_fh_bound_properties() so that if any internal
    mathematical sanity check fails, pytest will surface it clearly.
    """
    results = verify_fh_bound_properties()

    # Ignore the aggregate key when listing failures.
    failing = {
        name: ok
        for name, ok in results.items()
        if name != "all_passed" and not ok
    }

    # If "all_passed" is present, require it to be True as well.
    if "all_passed" in results:
        assert results["all_passed"] is True, (
            f"FH self-check reports overall failure; "
            f"failing tests: {sorted(failing)}"
        )

    # Also assert no individual test is marked False.
    assert not failing, f"FH self-check failed for: {sorted(failing)}"
