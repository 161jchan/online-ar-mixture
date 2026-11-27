"""Keep track of the predictors' current coefficients, recent observations,
and running covariance estimates.

Tasnuba can use OnlineARExperts to initialize the eight predictors,
get their forecasts, and update them after each new observation.
OnlineARExperts calls the Levinson-Durbin fitting code itself.
"""

from dataclasses import asdict, dataclass
from copy import deepcopy
import numpy as np
from .scalar import levinson_durbin


@dataclass(frozen=True)
class ExpertConfig:
    max_order: int = 8
    forgetting_factor: float = 0.99  # Start with 0.99; the assignment does not specify a default.
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
    """initialize() receives the first 40 raw observations. It calculates
    their mean and standard deviation, standardizes the 40 observations,
    and fits AR(1)..AR(8). Keep that mean and standard deviation fixed.

    predict() returns eight standardized forecasts in an (8,1) NumPy array.
    Row 1 is AR(1)'s forecast; row 8 is AR(8)'s forecast.

    update() receives the next actual observation in raw units, after
    its forecast has been scored. It updates the covariance estimates
    and recalculates AR(1)..AR(8) for the following observation.

    These sizes use the default maximum order L=8. For other choices of L,
    initialize() takes 5L observations and predict() returns shape (L,1).
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
        # Store recent standardized observations with the newest first.
        # Before an earlier observation is available, use zero for that lag.
        self._history = np.zeros(L)
        self._coefficients = np.zeros((L, L))
        # Start each expert's estimated error variance at one.
        # Replace it when that AR order fits successfully.
        self._variances = np.ones(L)
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
        """Return a copied dictionary containing:
        - AR(1)..AR(8)'s coefficients in one (8,8) table;
        - eight estimated error variances;
        - recent observations and running covariance estimates;
        - the warm-up mean and standard deviation;
        - flags showing which AR orders fitted successfully.

        predict() returns forecasts; snapshot() returns the quantities listed above.

        The caller receives the copy with:
            before = experts.snapshot()

        The example passes before to the optional gate function. Editing before
        cannot change the coefficients or arrays stored inside the actual experts.
        The sizes above use the default L=8 and change with the configured order.
        """
        self._require_initialized()
        f = self._fit_result
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
