"""Copula tail-dependence kernels for guardrail co-failure cliffs.

The functions in this module treat a guardrail miss pair as a bivariate
rare-event problem.  Pearson correlation describes average linear association;
tail-dependence coefficients describe whether simultaneous extreme misses stay
first-order likely as the miss marginal goes to zero.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize, minimize_scalar  # type: ignore[import-untyped]
from scipy.special import gammaln  # type: ignore[import-untyped]
from scipy.stats import kendalltau, norm, rankdata  # type: ignore[import-untyped]
from scipy.stats import t as student_t_dist

FloatArray: TypeAlias = NDArray[np.float64]
CopulaFamily: TypeAlias = Literal["gaussian", "clayton", "gumbel", "student_t"]
Criterion: TypeAlias = Literal["aic", "bic"]
Regime: TypeAlias = Literal["sub-critical", "critical", "super-critical"]

_EPS = 1.0e-12
_DEFAULT_CANDIDATES: tuple[CopulaFamily, ...] = (
    "gaussian",
    "clayton",
    "gumbel",
    "student_t",
)

__all__ = [
    "CliffCertificate",
    "CopulaCandidateFit",
    "CopulaFamily",
    "CopulaFitResult",
    "Regime",
    "TailDependenceCI",
    "TailDependenceEstimate",
    "bootstrap_tail_dependence_ci",
    "cliff_certificate",
    "estimate_tail_dependence",
    "fit_copula_family",
    "sample_copula",
    "theoretical_tail_dependence",
]


@dataclass(frozen=True)
class TailDependenceCI:
    """Percentile bootstrap confidence interval for lower and upper tail dependence."""

    lambda_lower: tuple[float, float]
    lambda_upper: tuple[float, float]
    confidence_level: float
    n_bootstrap: int
    method: str = "percentile_bootstrap"

    @property
    def lambda_any(self) -> tuple[float, float]:
        """Conservative interval for ``max(lambda_lower, lambda_upper)``."""

        lower = max(float(self.lambda_lower[0]), float(self.lambda_upper[0]))
        upper = max(float(self.lambda_lower[1]), float(self.lambda_upper[1]))
        return (_clip01(lower), _clip01(upper))


@dataclass(frozen=True)
class TailDependenceEstimate:
    """Estimated copula tail-dependence coefficients for a bivariate sample."""

    lambda_lower: float
    lambda_upper: float
    n_samples: int
    tail_fraction: float
    lower_joint_count: int
    upper_joint_count: int
    method: str
    ci: TailDependenceCI | None = None

    @property
    def lambda_any(self) -> float:
        """Return the larger of lower- and upper-tail dependence."""

        return max(float(self.lambda_lower), float(self.lambda_upper))


@dataclass(frozen=True)
class CopulaCandidateFit:
    """Likelihood fit for one copula candidate family."""

    family: CopulaFamily
    parameters: Mapping[str, float]
    log_likelihood: float
    aic: float
    bic: float
    converged: bool
    message: str
    tail_dependence: TailDependenceEstimate


@dataclass(frozen=True)
class CopulaFitResult:
    """Selected copula model and full AIC/BIC ranking."""

    selected: CopulaCandidateFit
    ranking: tuple[CopulaCandidateFit, ...]
    criterion: Criterion
    n_samples: int

    @property
    def family(self) -> CopulaFamily:
        return self.selected.family

    @property
    def parameters(self) -> Mapping[str, float]:
        return self.selected.parameters

    @property
    def tail_dependence(self) -> TailDependenceEstimate:
        return self.selected.tail_dependence


@dataclass(frozen=True)
class CliffCertificate:
    """Falsifiable statement about the supported co-failure regime."""

    regime: Regime
    lambda_hat: float
    ci: tuple[float, float]
    critical_value: float
    confidence_level: float
    statement: str
    falsifier: str


def estimate_tail_dependence(
    joint_samples: ArrayLike,
    *,
    tail_fraction: float = 0.05,
    marginal_samples: ArrayLike | Sequence[ArrayLike] | None = None,
    assume_uniform: bool = False,
    n_bootstrap: int = 0,
    confidence_level: float = 0.95,
    random_state: int | np.random.Generator | None = None,
) -> TailDependenceEstimate:
    """Estimate lower and upper copula tail-dependence coefficients.

    ``joint_samples`` must contain two columns.  By default, columns are
    converted to pseudo-observations using empirical ranks.  Pass
    ``assume_uniform=True`` when the rows are already draws from a copula on
    ``(0,1)^2``.
    """

    q = _tail_fraction(tail_fraction)
    u = _pseudo_observations(
        joint_samples,
        marginal_samples=marginal_samples,
        assume_uniform=assume_uniform,
    )
    lower_count = int(np.sum((u[:, 0] <= q) & (u[:, 1] <= q)))
    upper_count = int(np.sum((u[:, 0] >= 1.0 - q) & (u[:, 1] >= 1.0 - q)))
    denom = float(u.shape[0]) * q
    lam_l = _clip01(float(lower_count) / denom)
    lam_u = _clip01(float(upper_count) / denom)

    ci = None
    if n_bootstrap > 0:
        ci = bootstrap_tail_dependence_ci(
            joint_samples,
            tail_fraction=q,
            marginal_samples=marginal_samples,
            assume_uniform=assume_uniform,
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level,
            random_state=random_state,
        )

    return TailDependenceEstimate(
        lambda_lower=lam_l,
        lambda_upper=lam_u,
        n_samples=int(u.shape[0]),
        tail_fraction=q,
        lower_joint_count=lower_count,
        upper_joint_count=upper_count,
        method="empirical_threshold",
        ci=ci,
    )


def bootstrap_tail_dependence_ci(
    joint_samples: ArrayLike,
    *,
    tail_fraction: float = 0.05,
    marginal_samples: ArrayLike | Sequence[ArrayLike] | None = None,
    assume_uniform: bool = False,
    n_bootstrap: int = 500,
    confidence_level: float = 0.95,
    random_state: int | np.random.Generator | None = None,
) -> TailDependenceCI:
    """Bootstrap percentile CI for empirical lower and upper tail dependence."""

    q = _tail_fraction(tail_fraction)
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be at least 1.")
    level = _confidence_level(confidence_level)
    arr = _as_2d_float_array(joint_samples)
    n = int(arr.shape[0])
    rng = _rng(random_state)
    lower = np.empty(n_bootstrap, dtype=np.float64)
    upper = np.empty(n_bootstrap, dtype=np.float64)

    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        sample = arr[idx, :]
        est = estimate_tail_dependence(
            sample,
            tail_fraction=q,
            marginal_samples=marginal_samples,
            assume_uniform=assume_uniform,
            n_bootstrap=0,
        )
        lower[b] = est.lambda_lower
        upper[b] = est.lambda_upper

    alpha = 1.0 - level
    lo_q = 100.0 * alpha / 2.0
    hi_q = 100.0 * (1.0 - alpha / 2.0)
    return TailDependenceCI(
        lambda_lower=(
            _clip01(float(np.percentile(lower, lo_q))),
            _clip01(float(np.percentile(lower, hi_q))),
        ),
        lambda_upper=(
            _clip01(float(np.percentile(upper, lo_q))),
            _clip01(float(np.percentile(upper, hi_q))),
        ),
        confidence_level=level,
        n_bootstrap=int(n_bootstrap),
    )


def fit_copula_family(
    marginal_samples: ArrayLike | Sequence[ArrayLike] | None,
    joint_samples: ArrayLike,
    *,
    candidate_families: Iterable[CopulaFamily] = _DEFAULT_CANDIDATES,
    criterion: Criterion = "bic",
    assume_uniform: bool = False,
) -> CopulaFitResult:
    """Fit candidate copula families by pseudo-likelihood and select by AIC/BIC."""

    crit = _criterion(criterion)
    u = _pseudo_observations(
        joint_samples,
        marginal_samples=marginal_samples,
        assume_uniform=assume_uniform,
    )
    candidates = tuple(_family(f) for f in candidate_families)
    if not candidates:
        raise ValueError("candidate_families must contain at least one family.")

    fits: list[CopulaCandidateFit] = []
    for family in candidates:
        fits.append(_fit_one_family(family, u))

    ranking = tuple(sorted(fits, key=lambda fit: fit.aic if crit == "aic" else fit.bic))
    return CopulaFitResult(
        selected=ranking[0],
        ranking=ranking,
        criterion=crit,
        n_samples=int(u.shape[0]),
    )


def cliff_certificate(
    estimate: TailDependenceEstimate | CopulaFitResult | float | Mapping[str, Any],
    ci: TailDependenceCI | tuple[float, float] | Mapping[str, Any],
    *,
    critical_value: float = 0.20,
) -> CliffCertificate:
    """Return a falsifiable co-failure regime statement from an estimate and CI.

    The default regime threshold is intentionally operational, not universal:
    callers should pre-register a domain-specific ``critical_value``.  The
    decision rule is conservative: a sub-critical or super-critical claim is
    made only when the whole CI is on that side of the threshold.  If the CI
    intersects the threshold, the result is labeled critical.
    """

    threshold = _critical_value(critical_value)
    lam_hat = _extract_lambda_hat(estimate)
    ci_low, ci_high, confidence = _extract_ci(ci)
    if ci_low > ci_high:
        raise ValueError(f"CI lower endpoint exceeds upper endpoint: {(ci_low, ci_high)!r}")

    if ci_high < threshold:
        regime: Regime = "sub-critical"
        statement = (
            f"At {confidence:.1%} confidence, max(lambda_L, lambda_U) is below "
            f"{threshold:.3f}; the data support sub-critical co-failure."
        )
        falsifier = (
            f"A replication whose bootstrap CI has lower endpoint >= {threshold:.3f} "
            "would falsify the sub-critical certificate."
        )
    elif ci_low > threshold:
        regime = "super-critical"
        statement = (
            f"At {confidence:.1%} confidence, max(lambda_L, lambda_U) exceeds "
            f"{threshold:.3f}; the data support super-critical co-failure."
        )
        falsifier = (
            f"A replication whose bootstrap CI has upper endpoint <= {threshold:.3f} "
            "would falsify the super-critical certificate."
        )
    else:
        regime = "critical"
        statement = (
            f"At {confidence:.1%} confidence, max(lambda_L, lambda_U) is not separated "
            f"from the critical threshold {threshold:.3f}; the data support only the "
            "transition-compatible critical regime."
        )
        falsifier = (
            f"A tighter replication CI entirely below or above {threshold:.3f} would "
            "move the certificate out of the critical regime."
        )

    return CliffCertificate(
        regime=regime,
        lambda_hat=_clip01(lam_hat),
        ci=(_clip01(ci_low), _clip01(ci_high)),
        critical_value=threshold,
        confidence_level=confidence,
        statement=statement,
        falsifier=falsifier,
    )


def theoretical_tail_dependence(
    family: CopulaFamily,
    parameters: Mapping[str, float],
) -> TailDependenceEstimate:
    """Return closed-form tail-dependence coefficients for common copulas."""

    fam = _family(family)
    params = {str(k): float(v) for k, v in parameters.items()}

    if fam == "gaussian":
        rho = _rho(params.get("rho", 0.0))
        if rho >= 1.0 - 1.0e-12:
            lam_l = lam_u = 1.0
        else:
            lam_l = lam_u = 0.0
    elif fam == "clayton":
        theta = max(0.0, float(params.get("theta", 0.0)))
        lam_l = 0.0 if theta <= 0.0 else 2.0 ** (-1.0 / theta)
        lam_u = 0.0
    elif fam == "gumbel":
        theta = max(1.0, float(params.get("theta", 1.0)))
        lam_l = 0.0
        lam_u = 2.0 - 2.0 ** (1.0 / theta)
    else:
        rho = _rho(params.get("rho", 0.0))
        df = _df(params.get("df", 4.0))
        if rho >= 1.0 - 1.0e-12:
            lam_l = lam_u = 1.0
        else:
            arg = -math.sqrt((df + 1.0) * (1.0 - rho) / max(1.0 + rho, _EPS))
            lam_l = lam_u = _clip01(2.0 * float(student_t_dist.cdf(arg, df + 1.0)))

    return TailDependenceEstimate(
        lambda_lower=_clip01(lam_l),
        lambda_upper=_clip01(lam_u),
        n_samples=0,
        tail_fraction=float("nan"),
        lower_joint_count=0,
        upper_joint_count=0,
        method=f"theoretical_{fam}",
    )


def sample_copula(
    family: CopulaFamily,
    parameters: Mapping[str, float],
    n: int,
    *,
    random_state: int | np.random.Generator | None = None,
) -> FloatArray:
    """Draw ``n`` samples from a fitted bivariate copula on ``(0,1)^2``."""

    if n < 1:
        raise ValueError("n must be positive.")
    fam = _family(family)
    params = {str(k): float(v) for k, v in parameters.items()}
    rng = _rng(random_state)

    if fam == "gaussian":
        rho = _rho(params.get("rho", 0.0))
        z = _sample_correlated_normals(rho, n, rng)
        return _clip_unit_interval(norm.cdf(z))

    if fam == "student_t":
        rho = _rho(params.get("rho", 0.0))
        df = _df(params.get("df", 4.0))
        z = _sample_correlated_normals(rho, n, rng)
        w = rng.chisquare(df=df, size=n)
        t_draws = z / np.sqrt(w[:, None] / df)
        return _clip_unit_interval(student_t_dist.cdf(t_draws, df))

    if fam == "clayton":
        theta = max(float(params.get("theta", 0.0)), 0.0)
        if theta <= 1.0e-10:
            return rng.random((n, 2), dtype=np.float64)
        s = rng.gamma(shape=1.0 / theta, scale=1.0, size=n)
        e = rng.exponential(scale=1.0, size=(n, 2))
        return _clip_unit_interval((1.0 + e / s[:, None]) ** (-1.0 / theta))

    theta = max(float(params.get("theta", 1.0)), 1.0)
    if theta <= 1.0 + 1.0e-10:
        return rng.random((n, 2), dtype=np.float64)
    alpha = 1.0 / theta
    stable = _sample_positive_stable(alpha, n, rng)
    e = rng.exponential(scale=1.0, size=(n, 2))
    return _clip_unit_interval(np.exp(-((e / stable[:, None]) ** alpha)))


def _fit_one_family(family: CopulaFamily, u: FloatArray) -> CopulaCandidateFit:
    if family == "gaussian":
        params, ll, ok, msg = _fit_gaussian(u)
    elif family == "clayton":
        params, ll, ok, msg = _fit_clayton(u)
    elif family == "gumbel":
        params, ll, ok, msg = _fit_gumbel(u)
    else:
        params, ll, ok, msg = _fit_student_t(u)

    k = 2 if family == "student_t" else 1
    n = int(u.shape[0])
    aic = 2.0 * k - 2.0 * ll
    bic = math.log(float(n)) * k - 2.0 * ll
    return CopulaCandidateFit(
        family=family,
        parameters=params,
        log_likelihood=float(ll),
        aic=float(aic),
        bic=float(bic),
        converged=bool(ok),
        message=msg,
        tail_dependence=theoretical_tail_dependence(family, params),
    )


def _fit_gaussian(u: FloatArray) -> tuple[Mapping[str, float], float, bool, str]:
    tau_rho = _rho_from_kendall(u)

    def objective(rho_value: float) -> float:
        return -_sum_finite(_log_density_gaussian(u, rho_value))

    result = minimize_scalar(objective, bounds=(-0.995, 0.995), method="bounded")
    rho = _rho(float(result.x if result.success else tau_rho))
    ll = _sum_finite(_log_density_gaussian(u, rho))
    return ({"rho": rho}, ll, bool(result.success), str(result.message))


def _fit_clayton(u: FloatArray) -> tuple[Mapping[str, float], float, bool, str]:
    def objective(theta_value: float) -> float:
        return -_sum_finite(_log_density_clayton(u, theta_value))

    result = minimize_scalar(objective, bounds=(1.0e-6, 30.0), method="bounded")
    theta = max(0.0, float(result.x if result.success else 0.0))
    ll = _sum_finite(_log_density_clayton(u, theta))
    return ({"theta": theta}, ll, bool(result.success), str(result.message))


def _fit_gumbel(u: FloatArray) -> tuple[Mapping[str, float], float, bool, str]:
    def objective(theta_value: float) -> float:
        return -_sum_finite(_log_density_gumbel(u, theta_value))

    result = minimize_scalar(objective, bounds=(1.0, 30.0), method="bounded")
    theta = max(1.0, float(result.x if result.success else 1.0))
    ll = _sum_finite(_log_density_gumbel(u, theta))
    return ({"theta": theta}, ll, bool(result.success), str(result.message))


def _fit_student_t(u: FloatArray) -> tuple[Mapping[str, float], float, bool, str]:
    rho0 = _rho_from_kendall(u)

    def objective(x: FloatArray) -> float:
        rho = math.tanh(float(x[0]))
        df = 2.01 + math.exp(float(x[1]))
        return -_sum_finite(_log_density_student_t(u, rho, df))

    x0 = np.array([math.atanh(max(min(rho0, 0.98), -0.98)), math.log(8.0 - 2.01)])
    bounds = [(-3.0, 3.0), (math.log(0.05), math.log(98.0))]
    result = minimize(objective, x0=x0, method="L-BFGS-B", bounds=bounds)
    x = cast(FloatArray, result.x if result.success else x0)
    rho = _rho(math.tanh(float(x[0])))
    df = _df(2.01 + math.exp(float(x[1])))
    ll = _sum_finite(_log_density_student_t(u, rho, df))
    return ({"rho": rho, "df": df}, ll, bool(result.success), str(result.message))


def _log_density_gaussian(u: FloatArray, rho: float) -> FloatArray:
    r = _rho(rho)
    z = norm.ppf(_clip_unit_interval(u))
    z1 = z[:, 0]
    z2 = z[:, 1]
    one_minus = max(1.0 - r * r, _EPS)
    log_c = -0.5 * math.log(one_minus)
    log_c += (2.0 * r * z1 * z2 - r * r * (z1 * z1 + z2 * z2)) / (2.0 * one_minus)
    return cast(FloatArray, log_c)


def _log_density_student_t(u: FloatArray, rho: float, df: float) -> FloatArray:
    r = _rho(rho)
    nu = _df(df)
    x = student_t_dist.ppf(_clip_unit_interval(u), nu)
    x1 = cast(FloatArray, x[:, 0])
    x2 = cast(FloatArray, x[:, 1])
    det = max(1.0 - r * r, _EPS)
    quad = (x1 * x1 - 2.0 * r * x1 * x2 + x2 * x2) / det
    log_multi = (
        gammaln((nu + 2.0) / 2.0)
        - gammaln(nu / 2.0)
        - math.log(nu * math.pi)
        - 0.5 * math.log(det)
        - ((nu + 2.0) / 2.0) * np.log1p(quad / nu)
    )
    log_uni = student_t_dist.logpdf(x1, nu) + student_t_dist.logpdf(x2, nu)
    return cast(FloatArray, log_multi - log_uni)


def _log_density_clayton(u: FloatArray, theta: float) -> FloatArray:
    th = max(float(theta), 0.0)
    if th <= 1.0e-10:
        return np.zeros(u.shape[0], dtype=np.float64)
    x = _clip_unit_interval(u[:, 0])
    y = _clip_unit_interval(u[:, 1])
    s = np.power(x, -th) + np.power(y, -th) - 1.0
    log_c = math.log1p(th) + (-th - 1.0) * (np.log(x) + np.log(y))
    log_c += (-2.0 - 1.0 / th) * np.log(np.maximum(s, _EPS))
    return log_c


def _log_density_gumbel(u: FloatArray, theta: float) -> FloatArray:
    th = max(float(theta), 1.0)
    if th <= 1.0 + 1.0e-10:
        return np.zeros(u.shape[0], dtype=np.float64)
    x = -np.log(_clip_unit_interval(u[:, 0]))
    y = -np.log(_clip_unit_interval(u[:, 1]))
    s = np.power(x, th) + np.power(y, th)
    a = np.power(s, 1.0 / th)
    log_c = -a
    log_c += (th - 1.0) * (np.log(x) + np.log(y))
    log_c -= np.log(_clip_unit_interval(u[:, 0])) + np.log(_clip_unit_interval(u[:, 1]))
    log_c += (2.0 / th - 2.0) * np.log(np.maximum(s, _EPS))
    log_c += np.log1p((th - 1.0) * np.power(np.maximum(s, _EPS), -1.0 / th))
    return cast(FloatArray, log_c)


def _sum_finite(values: FloatArray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(finite)):
        return -float("inf")
    return float(np.sum(finite))


def _pseudo_observations(
    joint_samples: ArrayLike,
    *,
    marginal_samples: ArrayLike | Sequence[ArrayLike] | None,
    assume_uniform: bool,
) -> FloatArray:
    arr = _as_2d_float_array(joint_samples)
    if assume_uniform:
        return _clip_unit_interval(arr)

    if marginal_samples is None:
        out = np.empty_like(arr, dtype=np.float64)
        out[:, 0] = rankdata(arr[:, 0], method="average") / float(arr.shape[0] + 1)
        out[:, 1] = rankdata(arr[:, 1], method="average") / float(arr.shape[0] + 1)
        return _clip_unit_interval(out)

    refs = _marginal_reference_arrays(marginal_samples)
    out = np.empty_like(arr, dtype=np.float64)
    for j, ref in enumerate(refs):
        sorted_ref = np.sort(ref)
        ranks = np.searchsorted(sorted_ref, arr[:, j], side="right")
        out[:, j] = (ranks + 0.5) / float(sorted_ref.size + 1)
    return _clip_unit_interval(out)


def _marginal_reference_arrays(
    samples: ArrayLike | Sequence[ArrayLike],
) -> tuple[FloatArray, FloatArray]:
    if isinstance(samples, np.ndarray):
        arr = _as_2d_float_array(samples)
        return (arr[:, 0], arr[:, 1])

    if isinstance(samples, Sequence) and len(samples) == 2:
        first = _as_1d_float_array(samples[0])
        second = _as_1d_float_array(samples[1])
        return (first, second)

    arr = _as_2d_float_array(cast(ArrayLike, samples))
    return (arr[:, 0], arr[:, 1])


def _as_2d_float_array(samples: ArrayLike) -> FloatArray:
    arr = np.asarray(samples, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"samples must have shape (n, 2), got {arr.shape!r}.")
    if arr.shape[0] < 2:
        raise ValueError("samples must contain at least two rows.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("samples must be finite.")
    return arr


def _as_1d_float_array(samples: ArrayLike) -> FloatArray:
    arr = np.asarray(samples, dtype=np.float64).reshape(-1)
    if arr.size < 2:
        raise ValueError("marginal reference samples must contain at least two values.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("marginal reference samples must be finite.")
    return arr


def _clip_unit_interval(u: ArrayLike) -> FloatArray:
    arr = np.asarray(u, dtype=np.float64)
    return np.clip(arr, _EPS, 1.0 - _EPS)


def _sample_correlated_normals(rho: float, n: int, rng: np.random.Generator) -> FloatArray:
    r = _rho(rho)
    z1 = rng.standard_normal(size=n)
    z2 = r * z1 + math.sqrt(max(1.0 - r * r, 0.0)) * rng.standard_normal(size=n)
    return np.column_stack([z1, z2]).astype(np.float64)


def _sample_positive_stable(alpha: float, n: int, rng: np.random.Generator) -> FloatArray:
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1).")
    v = rng.uniform(0.0, math.pi, size=n)
    w = rng.exponential(scale=1.0, size=n)
    numerator = np.sin(alpha * v)
    denominator = np.power(np.sin(v), 1.0 / alpha)
    factor = np.power(np.sin((1.0 - alpha) * v) / w, (1.0 - alpha) / alpha)
    return numerator / denominator * factor


def _rho_from_kendall(u: FloatArray) -> float:
    tau = float(kendalltau(u[:, 0], u[:, 1], nan_policy="omit").statistic)
    if not math.isfinite(tau):
        tau = 0.0
    return _rho(math.sin(math.pi * max(min(tau, 1.0), -1.0) / 2.0))


def _extract_lambda_hat(
    estimate: TailDependenceEstimate | CopulaFitResult | float | Mapping[str, Any],
) -> float:
    if isinstance(estimate, TailDependenceEstimate):
        return estimate.lambda_any
    if isinstance(estimate, CopulaFitResult):
        return estimate.tail_dependence.lambda_any
    if isinstance(estimate, (float, int, np.floating)):
        return float(estimate)
    if isinstance(estimate, Mapping):
        if "lambda_any" in estimate:
            return float(estimate["lambda_any"])
        if "lambda" in estimate:
            return float(estimate["lambda"])
        lower = float(estimate.get("lambda_lower", 0.0))
        upper = float(estimate.get("lambda_upper", 0.0))
        return max(lower, upper)
    raise TypeError(f"Unsupported estimate type: {type(estimate).__name__}")


def _extract_ci(
    ci: TailDependenceCI | tuple[float, float] | Mapping[str, Any],
) -> tuple[float, float, float]:
    if isinstance(ci, TailDependenceCI):
        low, high = ci.lambda_any
        return (float(low), float(high), float(ci.confidence_level))
    if isinstance(ci, tuple) and len(ci) == 2:
        return (float(ci[0]), float(ci[1]), 0.95)
    if isinstance(ci, Mapping):
        confidence = float(ci.get("confidence_level", 0.95))
        if "lambda_any" in ci:
            pair = ci["lambda_any"]
            return (float(pair[0]), float(pair[1]), confidence)
        if "lambda" in ci:
            pair = ci["lambda"]
            return (float(pair[0]), float(pair[1]), confidence)
        if "lower" in ci and "upper" in ci:
            return (float(ci["lower"]), float(ci["upper"]), confidence)
    raise TypeError(f"Unsupported CI type: {type(ci).__name__}")


def _rng(random_state: int | np.random.Generator | None) -> np.random.Generator:
    if isinstance(random_state, np.random.Generator):
        return random_state
    return np.random.default_rng(random_state)


def _tail_fraction(value: float) -> float:
    q = float(value)
    if not math.isfinite(q) or not 0.0 < q < 0.5:
        raise ValueError("tail_fraction must be finite and in (0, 0.5).")
    return q


def _confidence_level(value: float) -> float:
    level = float(value)
    if not math.isfinite(level) or not 0.0 < level < 1.0:
        raise ValueError("confidence_level must be finite and in (0, 1).")
    return level


def _critical_value(value: float) -> float:
    threshold = float(value)
    if not math.isfinite(threshold) or not 0.0 < threshold < 1.0:
        raise ValueError("critical_value must be finite and in (0, 1).")
    return threshold


def _criterion(value: str) -> Criterion:
    crit = str(value).strip().lower()
    if crit not in {"aic", "bic"}:
        raise ValueError("criterion must be 'aic' or 'bic'.")
    return cast(Criterion, crit)


def _family(value: str) -> CopulaFamily:
    fam = str(value).strip().lower().replace("-", "_")
    if fam not in set(_DEFAULT_CANDIDATES):
        raise ValueError(f"Unknown copula family: {value!r}.")
    return fam


def _rho(value: float) -> float:
    rho = float(value)
    if not math.isfinite(rho) or not -1.0 < rho < 1.0:
        if rho >= 1.0:
            return 1.0 - _EPS
        if rho <= -1.0:
            return -1.0 + _EPS
        raise ValueError(f"rho must be finite in (-1, 1), got {value!r}.")
    return rho


def _df(value: float) -> float:
    df = float(value)
    if not math.isfinite(df) or df <= 2.0:
        raise ValueError(f"Student-t df must be finite and > 2, got {value!r}.")
    return df


def _clip01(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)
