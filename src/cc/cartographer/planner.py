"""Sample size planning utilities.

This module houses helpers for determining per-class sample sizes needed to
control Bernstein tail bounds for class-conditional rates.
"""

from __future__ import annotations

from math import ceil, log

from .bounds import fh_var_envelope

__all__ = ["needed_n_bernstein"]


def needed_n_bernstein(
    t: float,
    D: float,
    delta: float,
    I1: tuple[float, float],
    I0: tuple[float, float],
) -> tuple[int, int]:
    """Return sample sizes ensuring Bernstein tails below ``delta``.

    Parameters
    ----------
    t : float
        Target deviation in the performance metric. Must be ``> 0``.
    D : float
        Denominator scaling the deviation. Must be ``> 0``.
    delta : float
        Total risk budget in ``(0, 1)``.
    I1, I0 : tuple of float
        Fréchet-Hoeffding intervals ``[a, b]`` for class-conditional rates.

    Returns
    -------
    n1 : int
        Required sample size for the positive class.
    n0 : int
        Required sample size for the negative class.

    Raises
    ------
    ValueError
        If any argument is outside its valid range.

    Notes
    -----
    The required ``n_y`` satisfies

    .. math::

        n_y \\ge \frac{2 \bar v_y + \frac{2}{3} tD}{(tD)^2}\\log\frac{4}{\\delta},

    where ``\bar v_y`` is the Fréchet-Hoeffding variance envelope of the
    corresponding interval.
    """

    if t <= 0 or D <= 0:
        raise ValueError("t and D must be > 0.")
    if not (0 < delta < 1):
        raise ValueError("delta must be in (0,1).")

    for interval in (I1, I0):
        a, b = interval
        if not (0.0 <= a <= b <= 1.0):
            raise ValueError("intervals must satisfy 0 <= a <= b <= 1.")

    v1 = fh_var_envelope(I1)
    v0 = fh_var_envelope(I0)

    tD = t * D
    coeff = log(4.0 / delta)

    def n_req(vbar: float) -> int:
        num = 2.0 * vbar + (2.0 / 3.0) * tD
        den = tD**2
        return max(1, ceil((num / den) * coeff))

    return n_req(v1), n_req(v0)
