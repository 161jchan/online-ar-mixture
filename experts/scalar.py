"""Calculate the coefficients for AR(1) through AR(8).

Levinson-Durbin solves the Yule-Walker equations for all eight orders
together. A direct NumPy solve checks that the calculated coefficients
and estimated error variances agree. The maximum order is configurable.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class FitResult:
    # L is the maximum AR order: L=8 by default.

    # coefficients is one 8x8 table when L=8.
    # Row 1 holds AR(1)'s one coefficient, followed by seven zeros.
    # Row 2 holds AR(2)'s two coefficients, followed by six zeros.
    # Row 8 holds AR(8)'s eight coefficients.
    # Each column corresponds to a past lag.
    # Python counts from zero, so row p is accessed as coefficients[p-1].
    coefficients: np.ndarray

    # variances contains L estimated prediction-error variances:
    # one for AR(1), one for AR(2), ..., one for AR(L).
    variances: np.ndarray

    # valid[p-1] says whether AR(p)'s coefficients were successfully
    # recalculated during the current observation update.
    valid: np.ndarray

    # reflections contains the internal numbers used to check each AR order.
    # If AR(5) fails, stop before trying AR(6), AR(7), and AR(8).
    # Their uncalculated reflection numbers remain NaN.
    # Their coefficients keep their previous values; they do not become NaN.
    reflections: np.ndarray
    working_covariances: np.ndarray
    failed_order: int | None
    failure_reason: str | None


def regularized_covariances(covariances, epsilon=1e-6):
    """Before solving, add epsilon*r[0] to a copy of r[0].

    r[0] estimates the average squared standardized observation. It appears
    on the diagonal of the Yule-Walker matrix. Assignment Section 1.3
    requires this small increase to help the coefficient calculation behave
    better numerically.

    Do not change the stored r[0], or repeated fitting would repeatedly
    add the adjustment to the running estimate.
    """
    r = np.array(covariances, dtype=float, copy=True)
    if r.ndim != 1 or len(r) < 2:
        raise ValueError("Need a vector of covariances for lags 0 through L")
    if not np.isfinite(epsilon) or epsilon < 0:
        raise ValueError("epsilon must be finite and nonnegative")
    r[0] += epsilon * r[0]
    return r


def levinson_durbin(covariances, *, epsilon=1e-6,
                    previous_coefficients=None, previous_variances=None,
                    variance_floor=1e-12, reflection_margin=1e-10):
    """Calculate AR(1), save its coefficients, and use that calculation to
    build AR(2). Save AR(2)'s coefficients and continue through AR(L).

    AR(1) and AR(2) keep separate coefficient rows. Building AR(2) does not
    overwrite AR(1).

    If AR(5)'s variance or reflection check fails, stop the recursion.
    AR(1)..AR(4) keep their newly calculated coefficients and variances.
    AR(5)..AR(L) keep their coefficients and variances from the previous
    observation update.

    Before the first successful fit, use zero coefficients and an assumed
    error variance of one. Mark these starting values as not refreshed.
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
        # Copy AR(p-1)'s coefficients before calculating AR(p)'s coefficients.
        # Every calculation for AR(p) must use the unchanged AR(p-1) values.
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
    """Calculate AR(p)'s coefficients a second way, using np.linalg.solve
    instead of Levinson-Durbin. Section 1.4 requires this check.

    Both methods receive the same covariance estimates and regularization.
    After solving for the coefficients, calculate the estimated error variance:
        r[0] - coefficients @ r[1:p+1]

    Compare both the coefficients and error variance with Levinson-Durbin.
    This checks agreement between the two solvers, not agreement with the
    real coefficients used to generate the data.
    """
    r = regularized_covariances(covariances, epsilon)
    if not 1 <= order < len(r):
        raise ValueError("order outside supplied covariance range")
    indices = np.arange(order)
    matrix = r[np.abs(indices[:, None] - indices[None, :])]
    rhs = r[1:order + 1]
    coefficients = np.linalg.solve(matrix, rhs)
    return coefficients, float(r[0] - coefficients @ rhs)
