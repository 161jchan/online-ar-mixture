"""Whittle forward/backward order recursion; matrices multiply column vectors.

R[k] = E[z_t z_{t-k}.T]. Coefficients use (order, lag, output, input).
See README for the derivation, sources inspected, and pending teammate review.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class MultichannelFit:
    coefficients: np.ndarray
    covariances: np.ndarray
    backward_covariances: np.ndarray
    valid: np.ndarray
    reflections: np.ndarray
    normalized_reflection_norms: np.ndarray
    working_covariances: np.ndarray
    failed_order: int | None
    failure_reason: str | None


def regularized_covariances(covariances, epsilon=1e-6):
    r = np.array(covariances, dtype=float, copy=True)
    if r.ndim != 3 or len(r) < 2 or r.shape[1] != r.shape[2] or r.shape[1] < 1:
        raise ValueError("Need covariance matrices with shape (L+1,n,n)")
    if not np.isfinite(epsilon) or epsilon < 0:
        raise ValueError("epsilon must be finite and nonnegative")
    if not np.allclose(r[0], r[0].T, atol=1e-12, rtol=1e-12, equal_nan=True):
        raise ValueError("Lag-zero covariance must be symmetric")
    n = r.shape[1]
    r[0] += epsilon * np.trace(r[0]) / n * np.eye(n)
    return r


def _symmetric(a):
    return (a + a.T) * .5


def whittle(covariances, *, epsilon=1e-6, previous_coefficients=None,
            previous_covariances=None, previous_backward_covariances=None,
            variance_floor=1e-12, reflection_margin=1e-10):
    """Fit every valid order once, retaining failed/higher orders unchanged.

    Delta=R[p]-sum(A[j-1]@R[p-j]), Kf=Delta@Pb^-1,
    Kb=Delta.T@Pf^-1. Update A[j] with Kf@B[p-j] and vice versa,
    always using the preceding order's unmodified A, B, Pf and Pb.
    The Cholesky-whitened Delta has the same singular values as the
    symmetric-square-root normalization required by Section 1.3.
    """
    if not np.isfinite(variance_floor) or variance_floor <= 0:
        raise ValueError("variance_floor must be positive and finite")
    if not np.isfinite(reflection_margin) or not 0 <= reflection_margin < 1:
        raise ValueError("reflection_margin must be in [0,1)")
    r = regularized_covariances(covariances, epsilon)
    L, n = len(r)-1, r.shape[1]
    a_all = (np.zeros((L, L, n, n)) if previous_coefficients is None
             else np.array(previous_coefficients, dtype=float, copy=True))
    pf_all = (np.tile(np.eye(n), (L, 1, 1)) if previous_covariances is None
              else np.array(previous_covariances, dtype=float, copy=True))
    pb_all = (np.tile(np.eye(n), (L, 1, 1)) if previous_backward_covariances is None
              else np.array(previous_backward_covariances, dtype=float, copy=True))
    if a_all.shape != (L, L, n, n) or pf_all.shape != (L, n, n) or pb_all.shape != (L, n, n):
        raise ValueError("Prior coefficients/covariances have incompatible shapes")
    if not all(np.isfinite(x).all() for x in (a_all, pf_all, pb_all)):
        raise ValueError("Prior estimates must be finite")
    for prior in (pf_all, pb_all):
        if not np.allclose(prior, prior.swapaxes(-1, -2)):
            raise ValueError("Prior covariances must be symmetric")
        np.linalg.cholesky(prior)
    valid = np.zeros(L, bool)
    reflections = np.full((L, n, n), np.nan)
    norms = np.full(L, np.nan)
    a = b = np.empty((0, n, n))
    pf = pb = r[0].copy()
    failed = reason = None
    for p in range(1, L+1):
        try:
            if not all(np.isfinite(x).all() for x in (pf, pb, r[p])):
                raise np.linalg.LinAlgError("nonfinite covariance")
            # Require strictly PD covariances, including the configured floor.
            np.linalg.cholesky(pf - variance_floor*np.eye(n))
            np.linalg.cholesky(pb - variance_floor*np.eye(n))
            lf, lb = np.linalg.cholesky(pf), np.linalg.cholesky(pb)
            delta = r[p] - np.sum(a @ r[p-1:0:-1], axis=0)
            if not np.isfinite(delta).all():
                raise np.linalg.LinAlgError("nonfinite residual cross-covariance")
            kf = np.linalg.solve(pb, delta.T).T
            kb = np.linalg.solve(pf, delta).T
            reflections[p-1] = kf
            whitened = np.linalg.solve(lb, np.linalg.solve(lf, delta).T).T
            norms[p-1] = np.linalg.svd(whitened, compute_uv=False)[0]
            if norms[p-1] >= 1-reflection_margin:
                raise np.linalg.LinAlgError("normalized reflection singular value >= 1-margin")
            next_pf = _symmetric(pf-kf @ delta.T)
            next_pb = _symmetric(pb-kb @ delta)
            np.linalg.cholesky(next_pf-variance_floor*np.eye(n))
            np.linalg.cholesky(next_pb-variance_floor*np.eye(n))
            next_a = np.concatenate((a-kf @ b[::-1], kf[None]), axis=0)
            next_b = np.concatenate((b-kb @ a[::-1], kb[None]), axis=0)
            if not np.isfinite(next_a).all() or not np.isfinite(next_b).all():
                raise np.linalg.LinAlgError("nonfinite coefficients")
        except np.linalg.LinAlgError as exc:
            failed, reason = p, str(exc)
            break
        a, b, pf, pb = next_a, next_b, next_pf, next_pb
        a_all[p-1] = 0
        a_all[p-1, :p] = a
        pf_all[p-1], pb_all[p-1] = pf, pb
        valid[p-1] = True
    return MultichannelFit(a_all, pf_all, pb_all, valid, reflections, norms, r, failed, reason)


def direct_yule_walker(covariances, order, *, epsilon=1e-6):
    """Independent dense block solve with newest-to-oldest regressors.

    T[i,j]=Cov(z_{t-i-1},z_{t-j-1})=R[j-i] for j>=i,
    otherwise R[i-j].T. A T = [R[1] ... R[p]].
    """
    r = regularized_covariances(covariances, epsilon)
    if not 1 <= order < len(r):
        raise ValueError("order outside supplied covariance range")
    n = r.shape[1]
    matrix = np.block([[r[j-i] if j >= i else r[i-j].T
                        for j in range(order)] for i in range(order)])
    rhs = np.concatenate(r[1:order+1], axis=1)
    a = np.linalg.solve(matrix.T, rhs.T).T
    covariance = _symmetric(r[0] - a @ rhs.T)
    return a.reshape(n, order, n).transpose(1, 0, 2), covariance
