"""Synthetic truth lives here; experts receive observations only."""

from dataclasses import asdict, dataclass
import numpy as np


@dataclass(frozen=True)
class DataConfig:
    seed: int = 685
    n_observations: int = 10_000
    burn_in: int = 200
    radius_cap: float = 0.9

    def __post_init__(self):
        if self.n_observations < 1 or self.burn_in < 0:
            raise ValueError("Require positive length and nonnegative burn-in")
        if not 0 < self.radius_cap < 1:
            raise ValueError("radius_cap must be between zero and one")


def companion(coefficients):
    a = np.asarray(coefficients, dtype=float)
    if a.ndim != 1 or len(a) == 0 or not np.isfinite(a).all():
        raise ValueError("coefficients must be a nonempty finite vector")
    matrix = np.zeros((len(a), len(a)))
    matrix[0] = a
    matrix[1:, :-1] = np.eye(len(a) - 1)
    return matrix


def spectral_radius(coefficients):
    return float(np.max(np.abs(np.linalg.eigvals(companion(coefficients)))))


def stabilize(coefficients, radius_cap=0.9):
    """a[k-1] *= s**k maps every companion eigenvalue lambda to s*lambda."""
    if not 0 < radius_cap < 1:
        raise ValueError("radius_cap must be in (0,1)")
    a = np.asarray(coefficients, dtype=float)
    radius = spectral_radius(a)
    # Tiny inward margin protects the <= cap check against eigensolver rounding.
    scale = min(1.0, radius_cap * (1 - 1e-12) / radius) if radius else 1.0
    result = a * scale ** np.arange(1, len(a) + 1)
    if spectral_radius(result) > radius_cap:
        raise ArithmeticError("Companion stabilization failed")
    return result, scale


def generate_stationary_ar4(config=DataConfig()):
    """Return observations (N,1) and separate JSON-compatible ground truth.

    Independent SeedSequence child streams separate coefficient and innovation
    draws. Start at four zero lags, simulate burn_in + N steps, discard burn_in.
    """
    coefficient_seed, noise_seed = np.random.SeedSequence(config.seed).spawn(2)
    original = np.random.default_rng(coefficient_seed).normal(size=4)
    a, scale = stabilize(original, config.radius_cap)
    rng = np.random.default_rng(noise_seed)
    innovations = rng.normal(size=config.n_observations + config.burn_in)
    history = np.zeros(4)
    values = np.empty(len(innovations))
    for t, innovation in enumerate(innovations):
        values[t] = a @ history + innovation
        history[1:] = history[:-1].copy()
        history[0] = values[t]
    truth = {
        "configuration": asdict(config), "order": 4, "channels": 1,
        "coefficients": a.tolist(), "gaussian_draw": original.tolist(),
        "stabilization_scale": scale, "spectral_radius": spectral_radius(a),
        "original_spectral_radius": spectral_radius(original),
        "noise_variance_raw": 1.0, "initial_lags": [0.0] * 4,
        "rng": "NumPy PCG64; SeedSequence(seed).spawn(2): coefficients, innovations",
    }
    return values[config.burn_in:, None].copy(), truth
