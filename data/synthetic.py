"""We generate data by creating each new value from the previous four values,
weighted by four coefficients, plus random noise.

The coefficients are random draws controlled by a fixed seed. We adjust
them if necessary to make the AR(4) process stable, then keep them fixed.

We save the real coefficients so we can check the estimates later.
AR(1) through AR(8) receive the observations but not the real coefficients,
and must estimate their own coefficients.
"""

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
    """Multiply the lag-k coefficient by scale**k:
    a1 x scale, a2 x scale**2, a3 x scale**3, a4 x scale**4.

    We raise the scale factor to the lag power, not the coefficient itself.
    This adjustment shrinks the companion matrix's eigenvalues by scale.
    We choose scale so the largest absolute eigenvalue is at most radius_cap,
    which defaults to the assignment's stability limit of 0.9.
    """
    if not 0 < radius_cap < 1:
        raise ValueError("radius_cap must be in (0,1)")
    a = np.asarray(coefficients, dtype=float)
    radius = spectral_radius(a)
    # Aim just below 0.9 so computer rounding does not put the result above 0.9.
    scale = min(1.0, radius_cap * (1 - 1e-12) / radius) if radius else 1.0
    result = a * scale ** np.arange(1, len(a) + 1)
    if spectral_radius(result) > radius_cap:
        raise ArithmeticError("Companion stabilization failed")
    return result, scale


def generate_stationary_ar4(config=DataConfig()):
    """By default, start with four previous values set to zero. Generate 10,200
    observations, discard the first 200, and return the remaining 10,000.

    N counts the observations we keep, so the returned array has shape
    (N,1) = (10000,1). The four starting zeros are not included in that count.
    N and the burn-in length can be changed in DataConfig.

    The same seed and settings reproduce the same coefficients and noise.
    Use separate random streams for the coefficient draws and noise draws.

    Return the observations and a separate dictionary containing the real
    coefficients, seed, and generation settings.
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
