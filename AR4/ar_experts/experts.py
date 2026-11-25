"""Small stateful interface for Tasnuba's causal prediction loop."""

from dataclasses import asdict, dataclass
from copy import deepcopy
import numpy as np
from .levinson import levinson_durbin


@dataclass(frozen=True)
class ExpertConfig:
    max_order: int = 8
    forgetting_factor: float = 0.99  # implementation choice, not assignment default
    epsilon: float = 1e-6
    variance_floor: float = 1e-12
    reflection_margin: float = 1e-10
    normalization_floor: float = 1e-12

    def __post_init__(self):
        if not isinstance(self.max_order, int) or self.max_order < 1:
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
        return 5 * self.max_order


class OnlineARExperts:
    """predict returns STANDARDIZED forecasts (L,1); update consumes RAW data.

    initialize consumes exactly 5L raw warm-up values and fits once at the end.
    The first subsequent prediction targets retained observation 5L (zero based).
    No ground-truth metadata, future data, or full-series summaries enter here.
    """

    def __init__(self, config=ExpertConfig()):
        self.config = config
        self._initialized = False

    def initialize(self, warmup_values):
        x = np.asarray(warmup_values, dtype=float)
        if x.shape not in ((self.config.warmup_length,), (self.config.warmup_length, 1)):
            raise ValueError("initialize needs exactly 5L scalar raw observations")
        if not np.isfinite(x).all():
            raise ValueError("Warm-up contains nonfinite values")
        x = x.reshape(-1)
        self._mean = float(x.mean())
        measured_scale = float(x.std(ddof=0))
        self._constant_warmup = measured_scale <= self.config.normalization_floor
        self._scale = 1.0 if self._constant_warmup else measured_scale
        L = self.config.max_order
        self._r = np.zeros(L + 1)
        self._history = np.zeros(L)  # newest first; missing pre-retention lags are zero
        self._coefficients = np.zeros((L, L))
        self._variances = np.ones(L)  # finite conservative prior, stale until fitted
        self._stale_steps = np.zeros(L, dtype=int)
        self._guard_events = 0
        self._seen = 0
        for value in (x - self._mean) / self._scale:
            self._ingest(float(value))
        self._fit()
        self._initialized = True
        return self

    def _require_initialized(self):
        if not self._initialized:
            raise RuntimeError("Call initialize(warmup_values) first")

    def standardize(self, raw_values):
        self._require_initialized()
        return (np.asarray(raw_values, dtype=float) - self._mean) / self._scale

    def inverse_transform(self, standardized_values):
        self._require_initialized()
        return self._mean + self._scale * np.asarray(standardized_values, dtype=float)

    def predict(self):
        self._require_initialized()
        return (self._coefficients @ self._history)[:, None]

    def _ingest(self, z):
        products = z * np.r_[z, self._history]
        lam = self.config.forgetting_factor
        self._r = lam * self._r + (1 - lam) * products
        self._history[1:] = self._history[:-1].copy()
        self._history[0] = z
        self._seen += 1

    def _fit(self):
        fit = levinson_durbin(
            self._r, epsilon=self.config.epsilon,
            previous_coefficients=self._coefficients,
            previous_variances=self._variances,
            variance_floor=self.config.variance_floor,
            reflection_margin=self.config.reflection_margin,
        )
        self._coefficients, self._variances = fit.coefficients, fit.variances
        self._stale_steps = np.where(fit.valid, 0, self._stale_steps + 1)
        self._guard_events += int(fit.failed_order is not None)
        self._fit_result = fit

    def update(self, new_observation):
        self._require_initialized()
        x = np.asarray(new_observation, dtype=float)
        if x.shape not in ((), (1,), (1, 1)) or not np.isfinite(x).all():
            raise ValueError("update needs one finite raw scalar observation")
        z = float((x.item() - self._mean) / self._scale)
        if not np.isfinite(z):
            raise ValueError("Observation overflows frozen normalization")
        self._ingest(z)
        self._fit()

    def snapshot(self):
        self._require_initialized()
        f = self._fit_result
        # deepcopy includes every ndarray: callers can safely retain/mutate copies.
        return deepcopy({
            "configuration": asdict(self.config), "observations_seen": self._seen,
            "next_target_index": self._seen, "coefficients": self._coefficients,
            "error_variances_standardized": self._variances,
            "error_covariances_standardized": self._variances[:, None, None],
            "error_covariances_raw": (self._scale**2 * self._variances)[:, None, None],
            "covariances_standardized": self._r,
            "working_covariances_standardized": f.working_covariances,
            "history_standardized_newest_first": self._history,
            "normalization": {"mean": self._mean, "scale": self._scale,
                              "ddof": 0, "frozen": True,
                              "warmup_length": self.config.warmup_length,
                              "constant_warmup": self._constant_warmup},
            "diagnostics": {"fresh": f.valid, "stale_steps": self._stale_steps,
                            "reflections_current_attempt": f.reflections,
                            "failed_order": f.failed_order, "failure_reason": f.failure_reason,
                            "guard_events": self._guard_events},
        })
