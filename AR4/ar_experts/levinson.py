"""Positive forecast convention: xhat_t = sum_k a_k x_(t-k)."""

from dataclasses import dataclass
import numpy as np


@dataclass
class FitResult:
    coefficients: np.ndarray  # (L,L), row p-1, lag k-1; zeros beyond p
    variances: np.ndarray  # (L,), standardized squared units online
    valid: np.ndarray  # fresh this pass; false means retained prior estimate
    reflections: np.ndarray  # attempted current pass; NaN beyond first failure
    working_covariances: np.ndarray
    failed_order: int | None
    failure_reason: str | None


def regularized_covariances(covariances, epsilon=1e-6):
    r = np.array(covariances, dtype=float, copy=True)
    if r.ndim != 1 or len(r) < 2:
        raise ValueError("Need a vector of covariances for lags 0 through L")
    if not np.isfinite(epsilon) or epsilon < 0:
        raise ValueError("epsilon must be finite and nonnegative")
    # Scalar assignment regularization: R(0) += epsilon*R(0).
    r[0] += epsilon * r[0]
    return r


def levinson_durbin(covariances, *, epsilon=1e-6,
                    previous_coefficients=None, previous_variances=None,
                    variance_floor=1e-12, reflection_margin=1e-10):
    """All orders in one O(L^2) recursion, stopping at first invalid order.

    Failed order AND all higher orders keep their previous coefficients and
    variances. Defaults are zero forecasts and unit prior variances. These are
    stale fallbacks, never claimed as solutions to the current Yule-Walker system.
    """
    if not np.isfinite(variance_floor) or variance_floor <= 0:
        raise ValueError("variance_floor must be positive and finite")
    if not np.isfinite(reflection_margin) or not 0 <= reflection_margin < 1:
        raise ValueError("reflection_margin must be in [0,1)")
    r = regularized_covariances(covariances, epsilon)
    L = len(r) - 1
    coefficients = (np.zeros((L, L)) if previous_coefficients is None
                    else np.array(previous_coefficients, dtype=float, copy=True))
    variances = (np.ones(L) if previous_variances is None
                 else np.array(previous_variances, dtype=float, copy=True))
    if coefficients.shape != (L, L) or variances.shape != (L,):
        raise ValueError("Prior estimate shapes must be (L,L) and (L,)")
    if not np.isfinite(coefficients).all() or not np.isfinite(variances).all():
        raise ValueError("Prior estimates must be finite")
    if np.any(variances <= 0):
        raise ValueError("Prior variances must be positive")
    valid = np.zeros(L, dtype=bool)
    reflections = np.full(L, np.nan)
    a = np.empty(0)
    variance = r[0]
    failure = reason = None
    for p in range(1, L + 1):
        if not np.isfinite(variance) or variance <= variance_floor:
            failure, reason = p, "nonpositive, tiny, or nonfinite preceding variance"
            break
        # Copy before reversing/revising; do not update old entries in place.
        old = a.copy()
        reflection = (r[p] - old @ r[1:p][::-1]) / variance
        reflections[p - 1] = reflection
        next_variance = variance * (1 - reflection * reflection)
        if not np.isfinite(reflection) or abs(reflection) >= 1 - reflection_margin:
            failure, reason = p, "reflection coefficient outside allowed unit interval"
            break
        if not np.isfinite(next_variance) or next_variance <= variance_floor:
            failure, reason = p, "nonpositive, tiny, or nonfinite updated variance"
            break
        a = np.r_[old - reflection * old[::-1], reflection]
        coefficients[p - 1] = 0
        coefficients[p - 1, :p] = a
        variance = next_variance
        variances[p - 1] = variance
        valid[p - 1] = True
    return FitResult(coefficients, variances, valid, reflections, r, failure, reason)


def direct_yule_walker(covariances, order, *, epsilon=1e-6):
    """Independent dense solve of the SAME regularized Toeplitz equations.

    Return coefficients and Schur-complement variance; no recursion or fallback.
    Caller compares only orders accepted by the guarded recursion.
    """
    r = regularized_covariances(covariances, epsilon)
    if not 1 <= order < len(r):
        raise ValueError("order outside supplied covariance range")
    indices = np.arange(order)
    matrix = r[np.abs(indices[:, None] - indices[None, :])]
    rhs = r[1:order + 1]
    coefficients = np.linalg.solve(matrix, rhs)
    return coefficients, float(r[0] - coefficients @ rhs)
