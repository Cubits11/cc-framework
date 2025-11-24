# tests/unit/models/test_models_time.py

"""
Time helper tests for cc.core.models.

Focus:
- _now_unix: shape, recency, monotonic behaviour
- _iso_from_unix: ISO-8601 format, precision, UTC semantics, ordering

These tests are deliberately narrow: they do NOT touch model classes
(AttackResult, ExperimentConfig, etc.) – those live in their own files.
"""

import calendar
import time

import pytest
from hypothesis import assume, given, strategies as st

from cc.core.models import _iso_from_unix, _now_unix


# ---------------------------------------------------------------------
# _now_unix semantics
# ---------------------------------------------------------------------


def test_now_unix_is_recent_and_monotone():
    """
    _now_unix returns a float timestamp near 'now' and is monotone
    non-decreasing over short intervals.

    WHAT:
        Call _now_unix twice with a tiny sleep in between.
    WHY:
        Many timestamps (created_at, updated_at) use this as their source.
    THREAT:
        If this regresses to something non-time-based or non-monotone,
        ordering and recency assumptions across the codebase break.
    """
    t1 = _now_unix()
    time.sleep(0.001)
    t2 = _now_unix()

    assert isinstance(t1, float)
    assert isinstance(t2, float)

    # Not absurdly old – within a few seconds of the system clock.
    assert t1 > time.time() - 5.0

    # Non-decreasing in normal operation.
    assert t2 >= t1


def test_now_unix_close_to_system_time():
    """
    _now_unix should track the system wall-clock time (Unix epoch seconds).

    WHAT:
        Compare _now_unix() to time.time() at a single instant.
    WHY:
        Downstream code assumes timestamps are Unix epoch seconds in UTC.
    THREAT:
        If _now_unix switched to a monotonic/perf_counter-style clock,
        persisted timestamps would no longer be interpretable as real times.
    """
    t_model = _now_unix()
    t_system = time.time()

    # Allow generous margin for scheduling jitter and clock resolution.
    assert abs(t_model - t_system) < 5.0


# ---------------------------------------------------------------------
# _iso_from_unix: format and basic bounds
# ---------------------------------------------------------------------


@given(
    ts=st.floats(
        min_value=0.0,
        max_value=2**31 - 2,  # avoid platform-specific overflow issues
        allow_nan=False,
        allow_infinity=False,
    )
)
def test_iso_from_unix_shape_and_year_bounds(ts: float):
    """
    _iso_from_unix outputs well-formed ISO timestamps for reasonable inputs.

    WHAT:
        Generate a wide range of timestamps and check format + basic bounds.
    WHY:
        These strings are logged, persisted, and compared lexicographically.
    THREAT:
        A format change or locale-dependent behaviour would break parsing,
        dashboards, and ordering assumptions.
    """
    iso = _iso_from_unix(ts)

    # Shape: YYYY-MM-DDTHH:MM:SS.mmmZ (24 chars, 3 decimal places + Z)
    assert isinstance(iso, str)
    assert len(iso) == 24
    assert iso.endswith("Z")

    # Basic year sanity check.
    year = int(iso[:4])
    assert 1900 <= year <= 2200


def test_iso_from_unix_epoch_exact():
    """
    Epoch 0.0 must roundtrip to the canonical ISO string.

    WHAT:
        Explicitly check the exact representation of the Unix epoch.
    WHY:
        This is a fixed point used in docs, debugging, and regression tests.
    """
    assert _iso_from_unix(0.0) == "1970-01-01T00:00:00.000Z"


def test_iso_from_unix_millisecond_precision():
    """
    _iso_from_unix must preserve millisecond precision (3 decimal places).

    WHAT:
        Use a timestamp with a fractional part and inspect the output.
    WHY:
        Many logs and plots assume millisecond-level precision; we want
        a clear, documented contract.
    THREAT:
        Accidentally dropping to second-level precision would silently
        degrade temporal resolution.
    """
    ts = 1.234  # 1 second + 234 ms after epoch
    iso = _iso_from_unix(ts)

    assert iso.startswith("1970-01-01T00:00:01.")
    # Split "1970-01-01T00:00:01.234Z" → ["1970-...:01", "234Z"]
    frac = iso.split(".")[1]
    assert frac.endswith("Z")
    ms = frac[:-1]  # "234"
    assert len(ms) == 3
    assert ms == "234"


def test_iso_from_unix_leap_second_boundary_utc():
    """
    We do NOT handle leap seconds specially; we assume POSIX/UTC time.

    WHAT:
        Timestamp at 2016-12-31 23:59:59 UTC, right before a leap second.
    WHY:
        Documents that we follow standard POSIX semantics (no :60 seconds).
    THREAT:
        If someone assumed leap-second awareness, they'd misinterpret
        logs around leap-second events.
    """
    tm_struct = time.strptime("2016-12-31 23:59:59", "%Y-%m-%d %H:%M:%S")
    leap_ts = calendar.timegm(tm_struct)  # UTC-safe conversion
    iso = _iso_from_unix(leap_ts)
    assert iso == "2016-12-31T23:59:59.000Z"


# ---------------------------------------------------------------------
# _iso_from_unix vs UTC semantics & ordering
# ---------------------------------------------------------------------


@given(
    ts=st.floats(
        min_value=0.0,
        max_value=2**31 - 2,
        allow_nan=False,
        allow_infinity=False,
    )
)
def test_iso_from_unix_matches_gmtime_seconds(ts: float):
    """
    The second-level part of _iso_from_unix must match UTC gmtime().

    WHAT:
        Compare the "YYYY-MM-DDTHH:MM:SS" prefix to time.gmtime().
    WHY:
        Ensures we're using UTC, not local time, and that the date/time
        portion is consistent with standard library semantics.
    THREAT:
        Local-time formatting or timezone drift would break cross-machine
        reproducibility of timestamps.
    """
    iso = _iso_from_unix(ts)
    prefix = iso.split(".")[0]  # drop fractional seconds

    expected = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(ts))
    assert prefix == expected


@given(
    ts1=st.floats(
        min_value=0.0,
        max_value=2**31 - 3,
        allow_nan=False,
        allow_infinity=False,
    ),
    ts2=st.floats(
        min_value=0.0,
        max_value=2**31 - 2,
        allow_nan=False,
        allow_infinity=False,
    ),
)
def test_iso_from_unix_lexicographic_order_monotone(ts1: float, ts2: float):
    """
    ISO strings must be lexicographically non-decreasing with time.

    WHAT:
        For ts1 < ts2, ensure iso1 <= iso2 lexicographically.
    WHY:
        Many systems rely on sorting ISO strings as a proxy for time order.
    THREAT:
        If format changes or timezone offsets slip in, lex ordering could
        diverge from chronological ordering.
    """
    assume(ts1 < ts2)

    iso1 = _iso_from_unix(ts1)
    iso2 = _iso_from_unix(ts2)

    # With fixed-format UTC ISO strings, lex order matches time order.
    # Rounding can make iso1 == iso2 for very close timestamps, so we
    # only require <=, not strict <.
    assert iso1 <= iso2
