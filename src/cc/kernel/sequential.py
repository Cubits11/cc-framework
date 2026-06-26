"""Anytime-valid Bernoulli sequential tests via e-processes.

Null hypothesis
---------------
Let ``X_t in {0, 1}`` be the composed-system miss indicator at time ``t``.  In
the guardrail setting, ``X_t = 1`` means the composed guardrail stack missed an
unsafe example.  The operational null is

```
H0: P(X_t = 1 | F_{t-1}) <= p0   for every t,
```

where ``p0`` is a pre-registered benchmark such as the worst-case-independent
baseline miss rate.  For two guardrails with marginal miss rates ``p_A`` and
``p_B`` under an independence benchmark, ``p0 = p_A * p_B`` for an OR-style
stack that misses only when both rails miss.  The code accepts ``p0`` directly
because production users must define the appropriate benchmark for their
composition rule and threat model.

E-process construction
----------------------
For any predictable betting parameter ``lambda_t in [0, 1 / p0]``, define

```
M_t = product_{i=1}^t (1 + lambda_i (X_i - p0)).
```

The factor is nonnegative.  Under ``H0``,

```
E[1 + lambda_t (X_t - p0) | F_{t-1}]
  = 1 + lambda_t (E[X_t | F_{t-1}] - p0)
  <= 1.
```

Thus ``(M_t)`` is a nonnegative supermartingale with ``M_0 = 1``.  This module
uses a finite uniform mixture over constant betting fractions
``lambda = fraction / p0``.  A convex mixture of nonnegative supermartingales is
again a nonnegative supermartingale, hence the reported wealth ``E_t`` is an
e-process under ``H0``.

Anytime Type-I error theorem
----------------------------
Ville's inequality: if ``(E_t)`` is a nonnegative supermartingale with
``E_0 <= 1``, then for every ``c > 0``,

```
P_H0(sup_t E_t >= c) <= 1 / c.
```

Therefore the stopping rule

```
tau = inf{t: E_t >= 1 / alpha}
```

satisfies

```
sup_{tau stopping time} P_H0(tau < infinity) <= alpha.
```

This guarantee is uniform over all stopping times adapted to the observed data:
monitoring after every sample, stopping early after looking at the curve, or
continuing longer after inconclusive evidence does not inflate Type-I error.

Conditions
----------
The guarantee requires:

1. outcomes are binary miss indicators;
2. the benchmark ``p0`` is fixed before monitoring or is computed from data
   independent of the monitored sequence;
3. each betting parameter is predictable, meaning chosen using only past data;
4. under the null, ``E[X_t | F_{t-1}] <= p0`` for every monitored sample.

No global random state is used.  Randomness appears only in simulation helpers
and must be supplied through an explicit ``numpy.random.Generator``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray: TypeAlias = NDArray[np.float64]
StoppingDecision: TypeAlias = Literal["continue", "reject_null"]

__all__ = [
    "AnytimeBernoulliResult",
    "AnytimeBernoulliTester",
    "CalibrationResult",
    "PowerResult",
    "betting_fractions",
    "calibrate_false_stop_rate",
    "fixed_sample_size_one_sided",
    "independent_joint_miss_baseline",
    "simulate_power_curve",
]


@dataclass(frozen=True)
class AnytimeBernoulliResult:
    """Snapshot of an anytime-valid Bernoulli e-process."""

    n: int
    successes: int
    null_rate: float
    alpha: float
    e_value: float
    log_e_value: float
    max_e_value: float
    threshold: float
    decision: StoppingDecision
    stopped: bool
    stop_time: int | None
    sample_mean: float


@dataclass(frozen=True)
class CalibrationResult:
    """Empirical false-stop calibration under the null."""

    n_trials: int
    n_max: int
    null_rate: float
    alpha: float
    false_stops: int
    false_stop_rate: float
    monte_carlo_se: float
    within_two_se: bool
    seed_note: str


@dataclass(frozen=True)
class PowerResult:
    """Expected stopping-time summary for one alternative miss rate."""

    true_rate: float
    effect_size: float
    n_trials: int
    n_max: int
    stop_probability: float
    expected_stop_time: float
    median_stop_time: float
    fixed_sample_n_approx: int | None


def betting_fractions(n: int = 32, *, max_fraction: float = 0.95) -> FloatArray:
    """Return deterministic betting fractions in ``(0, max_fraction]``."""

    if n < 1:
        raise ValueError("n must be at least 1.")
    max_f = float(max_fraction)
    if not math.isfinite(max_f) or not 0.0 < max_f < 1.0:
        raise ValueError("max_fraction must be finite and in (0, 1).")
    return np.linspace(max_f / n, max_f, n, dtype=np.float64)


def independent_joint_miss_baseline(p_a_miss: float, p_b_miss: float) -> float:
    """Return the independent co-miss benchmark ``p_A * p_B``."""

    p_a = _probability(p_a_miss, "p_a_miss")
    p_b = _probability(p_b_miss, "p_b_miss")
    return float(p_a * p_b)


class AnytimeBernoulliTester:
    """Anytime-valid one-sided Bernoulli test for ``H0: miss_rate <= p0``."""

    def __init__(
        self,
        *,
        null_rate: float,
        alpha: float = 0.05,
        fractions: ArrayLike | None = None,
        min_samples: int = 1,
    ) -> None:
        self.null_rate = _open_probability(null_rate, "null_rate")
        self.alpha = _open_probability(alpha, "alpha")
        self.threshold = 1.0 / self.alpha
        self.min_samples = _positive_int(min_samples, "min_samples")

        frac = betting_fractions() if fractions is None else np.asarray(fractions, dtype=np.float64)
        if frac.ndim != 1 or frac.size == 0:
            raise ValueError("fractions must be a non-empty one-dimensional array.")
        if not np.all(np.isfinite(frac)) or np.any(frac <= 0.0) or np.any(frac >= 1.0):
            raise ValueError("fractions must be finite and lie in (0, 1).")

        self.fractions = frac.astype(np.float64)
        self.lambdas = self.fractions / self.null_rate
        self.log_weights = np.full(
            self.fractions.size,
            -math.log(float(self.fractions.size)),
            dtype=np.float64,
        )
        self.reset()

    def reset(self) -> None:
        """Reset wealth and counts to their initial state."""

        self.n = 0
        self.successes = 0
        self.log_wealths = np.zeros(self.fractions.size, dtype=np.float64)
        self.log_e_value = 0.0
        self.e_value = 1.0
        self.max_e_value = 1.0
        self.stop_time: int | None = None

    def update(self, outcome: bool | int | float) -> AnytimeBernoulliResult:
        """Consume one binary outcome and return the current anytime result."""

        x = _binary(outcome, "outcome")
        self.n += 1
        self.successes += int(x)
        increment = 1.0 + self.lambdas * (float(x) - self.null_rate)
        if np.any(increment < -1.0e-12):
            raise FloatingPointError("Betting increment became negative; invalid bet grid.")
        increment = np.maximum(increment, 0.0)
        with np.errstate(divide="ignore"):
            self.log_wealths += np.log(increment)

        self.log_e_value = _logsumexp(self.log_weights + self.log_wealths)
        self.e_value = _safe_exp(self.log_e_value)
        self.max_e_value = max(self.max_e_value, self.e_value)
        if self.stop_time is None and self.n >= self.min_samples and self.e_value >= self.threshold:
            self.stop_time = self.n
        return self.result()

    def update_many(self, outcomes: ArrayLike) -> AnytimeBernoulliResult:
        """Consume a sequence of binary outcomes."""

        arr = np.asarray(outcomes, dtype=np.float64).reshape(-1)
        if arr.size == 0:
            return self.result()
        if not np.all((arr == 0.0) | (arr == 1.0)):
            raise ValueError("outcomes must be binary 0/1.")

        start_n = self.n
        increments = 1.0 + (arr[:, None] - self.null_rate) * self.lambdas[None, :]
        if np.any(increments < -1.0e-12):
            raise FloatingPointError("Betting increment became negative; invalid bet grid.")
        increments = np.maximum(increments, 0.0)
        with np.errstate(divide="ignore"):
            log_increment_cumsum = np.cumsum(np.log(increments), axis=0)
        path_log_wealths = self.log_wealths[None, :] + log_increment_cumsum
        path_log_e = _logsumexp_axis1(self.log_weights[None, :] + path_log_wealths)
        path_e = np.exp(np.minimum(path_log_e, math.log(float(np.finfo(np.float64).max))))

        if self.stop_time is None:
            sample_numbers = start_n + np.arange(1, arr.size + 1)
            crossed = (sample_numbers >= self.min_samples) & (path_e >= self.threshold)
            if np.any(crossed):
                first = int(np.argmax(crossed))
                self.stop_time = int(sample_numbers[first])

        self.n += int(arr.size)
        self.successes += int(np.sum(arr))
        self.log_wealths = path_log_wealths[-1, :].astype(np.float64)
        self.log_e_value = float(path_log_e[-1])
        self.e_value = _safe_exp(self.log_e_value)
        self.max_e_value = max(self.max_e_value, float(np.max(path_e)))
        return self.result()

    def result(self) -> AnytimeBernoulliResult:
        """Return the current e-process snapshot."""

        stopped = self.stop_time is not None
        mean = float(self.successes / self.n) if self.n else 0.0
        return AnytimeBernoulliResult(
            n=int(self.n),
            successes=int(self.successes),
            null_rate=float(self.null_rate),
            alpha=float(self.alpha),
            e_value=float(self.e_value),
            log_e_value=float(self.log_e_value),
            max_e_value=float(self.max_e_value),
            threshold=float(self.threshold),
            decision="reject_null" if stopped else "continue",
            stopped=bool(stopped),
            stop_time=self.stop_time,
            sample_mean=mean,
        )


def calibrate_false_stop_rate(
    *,
    rng: np.random.Generator,
    null_rate: float,
    alpha: float = 0.05,
    n_trials: int = 5_000,
    n_max: int = 1_000,
    min_samples: int = 1,
) -> CalibrationResult:
    """Simulate null trials and estimate the false-stop rate."""

    _require_rng(rng)
    p0 = _open_probability(null_rate, "null_rate")
    a = _open_probability(alpha, "alpha")
    trials = _positive_int(n_trials, "n_trials")
    horizon = _positive_int(n_max, "n_max")
    false_stops = 0

    for _ in range(trials):
        tester = AnytimeBernoulliTester(null_rate=p0, alpha=a, min_samples=min_samples)
        outcomes = rng.binomial(1, p0, size=horizon)
        result = tester.update_many(outcomes)
        false_stops += int(result.stopped)

    rate = false_stops / trials
    mc_se = math.sqrt(max(rate * (1.0 - rate), 0.0) / trials)
    return CalibrationResult(
        n_trials=trials,
        n_max=horizon,
        null_rate=p0,
        alpha=a,
        false_stops=false_stops,
        false_stop_rate=float(rate),
        monte_carlo_se=float(mc_se),
        within_two_se=bool(rate <= a + 2.0 * mc_se),
        seed_note="rng supplied by caller; no global RNG state used",
    )


def simulate_power_curve(
    *,
    rng: np.random.Generator,
    null_rate: float,
    true_rates: ArrayLike,
    alpha: float = 0.05,
    n_trials: int = 1_000,
    n_max: int = 2_000,
    fixed_power: float = 0.80,
    min_samples: int = 1,
) -> tuple[PowerResult, ...]:
    """Simulate expected stopping time across alternatives."""

    _require_rng(rng)
    p0 = _open_probability(null_rate, "null_rate")
    rates = np.asarray(true_rates, dtype=np.float64).reshape(-1)
    if rates.size == 0:
        raise ValueError("true_rates must be non-empty.")
    trials = _positive_int(n_trials, "n_trials")
    horizon = _positive_int(n_max, "n_max")

    out: list[PowerResult] = []
    for rate_value in rates:
        p = _open_probability(float(rate_value), "true_rate")
        stop_times: list[int] = []
        stopped_count = 0
        for _ in range(trials):
            tester = AnytimeBernoulliTester(null_rate=p0, alpha=alpha, min_samples=min_samples)
            outcomes = rng.binomial(1, p, size=horizon)
            result = tester.update_many(outcomes)
            if result.stopped and result.stop_time is not None:
                stopped_count += 1
                stop_times.append(result.stop_time)
            else:
                stop_times.append(horizon)

        stops = np.asarray(stop_times, dtype=np.float64)
        out.append(
            PowerResult(
                true_rate=p,
                effect_size=p - p0,
                n_trials=trials,
                n_max=horizon,
                stop_probability=float(stopped_count / trials),
                expected_stop_time=float(np.mean(stops)),
                median_stop_time=float(np.median(stops)),
                fixed_sample_n_approx=fixed_sample_size_one_sided(
                    null_rate=p0,
                    true_rate=p,
                    alpha=alpha,
                    power=fixed_power,
                ),
            )
        )
    return tuple(out)


def fixed_sample_size_one_sided(
    *,
    null_rate: float,
    true_rate: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> int | None:
    """Normal-approximate fixed-sample size for one-sample one-sided Bernoulli test."""

    p0 = _open_probability(null_rate, "null_rate")
    p1 = _open_probability(true_rate, "true_rate")
    if p1 <= p0:
        return None
    _open_probability(alpha, "alpha")
    _open_probability(power, "power")
    z_alpha = _normal_quantile(1.0 - alpha)
    z_power = _normal_quantile(power)
    numerator = z_alpha * math.sqrt(p0 * (1.0 - p0)) + z_power * math.sqrt(
        p1 * (1.0 - p1)
    )
    n = (numerator / (p1 - p0)) ** 2
    return math.ceil(n)


def _normal_quantile(probability: float) -> float:
    if abs(probability - 0.95) < 1.0e-12:
        return 1.6448536269514722
    if abs(probability - 0.80) < 1.0e-12:
        return 0.8416212335729143
    lo, hi = -8.0, 8.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        cdf = 0.5 * (1.0 + math.erf(mid / math.sqrt(2.0)))
        if cdf < probability:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _probability(value: float, name: str) -> float:
    x = float(value)
    if not math.isfinite(x) or not 0.0 <= x <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1], got {value!r}.")
    return x


def _open_probability(value: float, name: str) -> float:
    x = _probability(value, name)
    if not 0.0 < x < 1.0:
        raise ValueError(f"{name} must lie strictly between 0 and 1, got {value!r}.")
    return x


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a positive integer, got bool.")
    x = int(value)
    if x < 1 or x != value:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    return x


def _binary(value: bool | int | float, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    x = float(value)
    if x not in (0.0, 1.0):
        raise ValueError(f"{name} must be binary 0/1, got {value!r}.")
    return int(x)


def _require_rng(rng: np.random.Generator) -> None:
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be an explicit numpy.random.Generator.")


def _logsumexp(values: FloatArray) -> float:
    m = float(np.max(values))
    if not math.isfinite(m):
        return m
    return float(m + math.log(float(np.sum(np.exp(values - m)))))


def _logsumexp_axis1(values: FloatArray) -> FloatArray:
    max_values = np.max(values, axis=1)
    finite = np.isfinite(max_values)
    out = np.array(max_values, dtype=np.float64)
    if np.any(finite):
        centered = values[finite, :] - max_values[finite, None]
        out[finite] = max_values[finite] + np.log(np.sum(np.exp(centered), axis=1))
    return out


def _safe_exp(value: float) -> float:
    if value >= math.log(float(np.finfo(np.float64).max)):
        return float("inf")
    return float(math.exp(value))
