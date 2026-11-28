"""Section 1.3 FPE, evaluated in log space before observing the target."""

from dataclasses import dataclass
import numpy as np


@dataclass
class FPEChoice:
    order: int
    log_scores: np.ndarray
    eligible: np.ndarray
    used_fallback: bool


def select_fpe(covariances, forgetting_factor, *, fresh=None, previous_order=1):
    """Exclude stale, nonfinite, asymmetric, non-PD, or invalid-denominator fits.

    Ties choose the smallest order. If none is eligible, retain previous_order
    (AR(1) initially); the forecast still uses that expert's retained fit.
    Fallback is explicitly flagged, never described as a finite FPE optimum.
    """
    sigma = np.asarray(covariances, dtype=float)
    if sigma.ndim != 3 or sigma.shape[1] != sigma.shape[2] or min(sigma.shape) < 1:
        raise ValueError("covariances must have shape (L,n,n)")
    if not np.isfinite(forgetting_factor) or not 0 < forgetting_factor < 1:
        raise ValueError("forgetting_factor must be in (0,1)")
    L, n, _ = sigma.shape
    if not isinstance(previous_order, (int, np.integer)) or not 1 <= previous_order <= L:
        raise ValueError("previous_order must be in 1..L")
    fresh = np.ones(L, bool) if fresh is None else np.asarray(fresh, dtype=bool)
    if fresh.shape != (L,):
        raise ValueError("fresh must have shape (L,)")
    effective_n = 1/(1-forgetting_factor)
    scores = np.full(L, np.inf)
    for i, s in enumerate(sigma):
        p = i+1
        denominator = effective_n-p*n-1
        if denominator <= 8*np.finfo(float).eps*effective_n or not fresh[i] or not np.isfinite(s).all():
            continue
        if not np.allclose(s, s.T, atol=1e-12, rtol=1e-12):
            continue
        try:
            chol = np.linalg.cholesky(s)
        except np.linalg.LinAlgError:
            continue
        scores[i] = (2*np.log(np.diag(chol)).sum()
                     + n*(np.log(effective_n+p*n+1)-np.log(denominator)))
    eligible = np.isfinite(scores)
    return FPEChoice(int(np.argmin(scores))+1 if eligible.any() else int(previous_order),
                     scores, eligible, not bool(eligible.any()))
