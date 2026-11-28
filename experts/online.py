"""Causal scalar/vector AR banks with frozen warm-up normalization."""

from dataclasses import asdict, dataclass
from copy import deepcopy
import numpy as np
from .scalar import levinson_durbin
from .multichannel import whittle


@dataclass(frozen=True)
class ExpertConfig:
    max_order: int = 8
    forgetting_factor: float = .99
    epsilon: float = 1e-6
    variance_floor: float = 1e-12
    reflection_margin: float = 1e-10
    normalization_floor: float = 1e-12

    def __post_init__(self):
        if not isinstance(self.max_order, int) or isinstance(self.max_order, bool) or self.max_order < 1:
            raise ValueError("max_order must be a positive integer")
        if not np.isfinite(self.forgetting_factor) or not 0 < self.forgetting_factor < 1:
            raise ValueError("forgetting_factor must be in (0,1)")
        if not np.isfinite(self.epsilon) or self.epsilon < 0:
            raise ValueError("epsilon must be nonnegative and finite")
        if not np.isfinite(self.reflection_margin) or not 0 <= self.reflection_margin < 1:
            raise ValueError("reflection_margin must be in [0,1)")
        for floor in (self.variance_floor, self.normalization_floor):
            if not np.isfinite(floor) or floor <= 0:
                raise ValueError("floors must be finite and positive")

    @property
    def warmup_length(self):
        return 5*self.max_order


class OnlineARExperts:
    """initialize(first 5L raw rows); predict(); then update(raw target).

    predict() returns standardized (L,n) forecasts. snapshot() returns
    copied pre-next-target estimates and covariances (L,n,n). Scalar legacy
    coefficient/covariance fields keep their original one-dimensional form;
    explicitly named coefficient fields always use (L,L,n,n).
    """

    def __init__(self, config=ExpertConfig()):
        self.config = config
        self._initialized = False

    def initialize(self, warmup_values):
        x = np.asarray(warmup_values, dtype=float)
        if x.ndim == 1:
            x = x[:, None]
        if x.ndim != 2 or x.shape[0] != self.config.warmup_length or x.shape[1] < 1:
            raise ValueError("initialize needs exactly 5L raw rows with n>=1 channels")
        if not np.isfinite(x).all():
            raise ValueError("Warm-up contains nonfinite values")
        self.channels = n = x.shape[1]
        mean, scale = x.mean(axis=0), x.std(axis=0, ddof=0)
        constant = scale <= self.config.normalization_floor
        scale = np.where(constant, 1., scale)
        self._mean = float(mean[0]) if n == 1 else mean
        self._scale = float(scale[0]) if n == 1 else scale
        self._constant_warmup = bool(constant[0]) if n == 1 else constant
        L = self.config.max_order
        self._r = np.zeros(L+1 if n == 1 else (L+1, n, n))
        self._history = np.zeros(L if n == 1 else (L, n))
        self._coefficients = np.zeros((L, L) if n == 1 else (L, L, n, n))
        self._variances = np.ones(L)
        self._covariances = np.tile(np.eye(n), (L, 1, 1))
        self._backward_covariances = self._covariances.copy()
        self._stale_steps = np.zeros(L, dtype=int)
        self._ever_valid = np.zeros(L, dtype=bool)
        self._guard_events = self._seen = 0
        for row in (x-mean)/scale:
            self._ingest(float(row[0]) if n == 1 else row)
        self._fit()
        self._initialized = True
        return self

    def _require_initialized(self):
        if not self._initialized:
            raise RuntimeError("Call initialize(warmup_values) first")

    def standardize(self, raw_values):
        self._require_initialized()
        return (np.asarray(raw_values, dtype=float)-self._mean)/self._scale

    def inverse_transform(self, standardized_values):
        self._require_initialized()
        return self._mean + self._scale*np.asarray(standardized_values, dtype=float)

    def predict(self):
        self._require_initialized()
        if self.channels == 1:
            return (self._coefficients @ self._history)[:, None]
        return np.einsum('pkij,kj->pi', self._coefficients, self._history)

    def _ingest(self, z):
        if self.channels == 1:
            products = z*np.r_[z, self._history]
        else:
            products = z[None, :, None]*np.vstack((z, self._history))[:, None, :]
        lam = self.config.forgetting_factor
        self._r = lam*self._r + (1-lam)*products
        self._history[1:] = self._history[:-1].copy()
        self._history[0] = z
        self._seen += 1

    def _fit(self):
        kwargs = dict(epsilon=self.config.epsilon, previous_coefficients=self._coefficients,
                      variance_floor=self.config.variance_floor,
                      reflection_margin=self.config.reflection_margin)
        if self.channels == 1:
            fit = levinson_durbin(self._r, previous_variances=self._variances, **kwargs)
            self._variances = fit.variances
            self._covariances = self._variances[:, None, None]
            self._backward_covariances = self._covariances.copy()
        else:
            fit = whittle(self._r, previous_covariances=self._covariances,
                          previous_backward_covariances=self._backward_covariances, **kwargs)
            self._covariances = fit.covariances
            self._backward_covariances = fit.backward_covariances
        self._coefficients = fit.coefficients
        self._stale_steps = np.where(fit.valid, 0, self._stale_steps+1)
        self._ever_valid |= fit.valid
        self._guard_events += int(fit.failed_order is not None)
        self._fit_result = fit

    def update(self, new_observation):
        self._require_initialized()
        x = np.asarray(new_observation, dtype=float)
        n = self.channels
        allowed = ((), (1,), (1, 1)) if n == 1 else ((n,), (1, n))
        if x.shape not in allowed or not np.isfinite(x).all():
            raise ValueError("update needs one finite raw n-channel observation")
        z = (x.reshape(n)-self._mean)/self._scale
        if not np.isfinite(z).all():
            raise ValueError("Observation overflows frozen normalization")
        self._ingest(float(z[0]) if n == 1 else z)
        self._fit()

    def snapshot(self):
        self._require_initialized()
        f, n = self._fit_result, self.channels
        a = self._coefficients[:, :, None, None] if n == 1 else self._coefficients
        scale = np.asarray(self._scale).reshape(n)
        mean = np.asarray(self._mean).reshape(n)
        # x = mu + D*z: A_raw=D*A_z*D^-1; b_raw=mu-sum(A_raw)*mu.
        raw_a = a*scale[None, None, :, None]/scale[None, None, None, :]
        raw_sigma = self._covariances*scale[None, :, None]*scale[None, None, :]
        result = {
            "configuration": asdict(self.config), "channels": n,
            "observations_seen": self._seen, "next_target_index": self._seen,
            "coefficients": self._coefficients,
            "coefficients_standardized": a, "coefficients_raw": raw_a,
            "intercepts_raw": mean-np.sum(raw_a, axis=1) @ mean,
            "error_covariances_standardized": self._covariances,
            "error_covariances_raw": raw_sigma,
            "backward_error_covariances_standardized": self._backward_covariances,
            "covariances_standardized": self._r,
            "working_covariances_standardized": f.working_covariances,
            "history_standardized_newest_first": self._history,
            "normalization": {"mean": self._mean, "scale": self._scale, "ddof": 0,
                              "frozen": True, "warmup_length": self.config.warmup_length,
                              "constant_warmup": self._constant_warmup},
            "diagnostics": {"fresh": f.valid, "ever_valid": self._ever_valid,
                            "stale_steps": self._stale_steps,
                            "reflections_current_attempt": f.reflections,
                            "normalized_reflection_norms": (np.abs(f.reflections) if n == 1
                                                           else f.normalized_reflection_norms),
                            "failed_order": f.failed_order, "failure_reason": f.failure_reason,
                            "guard_events": self._guard_events},
        }
        if n == 1:
            result["error_variances_standardized"] = self._variances
        return deepcopy(result)
