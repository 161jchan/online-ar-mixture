"""Generate stationary, switching and drifting AR observations.

For one channel, each coefficient weights a past value of that channel. For
three channels, each lag has a 3x3 coefficient matrix: entry [i,j] weights past
channel j when generating current channel i. Add independent N(0,1) noise to
each channel. The fixed seed makes both the coefficients and noise repeatable.

Keep coefficients fixed within each stationary or switching regime. The drifting
AR(1) instead follows its prescribed coefficient path. Return the observations
separately from the real coefficients so experts can fit without seeing them.
"""

from dataclasses import asdict, dataclass
import numpy as np


@dataclass(frozen=True)
class DataConfig:
    seed: int = 685
    n_observations: int = 10_000
    burn_in: int = 200
    radius_cap: float = 0.9
    channels: int = 1

    def __post_init__(self):
        for name, value, minimum in (("seed", self.seed, 0),
                                     ("n_observations", self.n_observations, 1),
                                     ("burn_in", self.burn_in, 0),
                                     ("channels", self.channels, 1)):
            if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if not np.isfinite(self.radius_cap) or not 0 < self.radius_cap < 1:
            raise ValueError("radius_cap must be between zero and one")


def companion(coefficients):
    """Build the matrix whose eigenvalues we use to check AR stability.

    Scalar coefficients have shape (p,). Multichannel coefficients have shape
    (p,n,n), ordered by lag, output channel, input channel. The first block row
    contains all lag matrices; lower rows shift the previous observations back.
    """
    a = np.asarray(coefficients, dtype=float)
    if a.ndim == 1:
        a = a[:, None, None]
    if (a.ndim != 3 or min(a.shape) < 1 or a.shape[1] != a.shape[2]
            or not np.isfinite(a).all()):
        raise ValueError("coefficients must be finite with shape (p,) or (p,n,n)")
    p, n, _ = a.shape
    matrix = np.zeros((p*n, p*n))
    matrix[:n] = np.concatenate(a, axis=1)
    matrix[n:, :-n] = np.eye((p-1)*n)
    return matrix


def spectral_radius(coefficients):
    return float(np.max(np.abs(np.linalg.eigvals(companion(coefficients)))))


def stabilize(coefficients, radius_cap=0.9):
    """Multiply the lag-k coefficient or coefficient matrix by scale**k.
    For AR(4): a1 x scale, a2 x scale**2, a3 x scale**3, a4 x scale**4.

    We raise the scale factor to the lag power, not the coefficient itself.
    This adjustment shrinks the companion matrix's eigenvalues by scale.
    We choose scale so the largest absolute eigenvalue is at most radius_cap,
    which defaults to the assignment's stability limit of 0.9.
    """
    if not np.isfinite(radius_cap) or not 0 < radius_cap < 1:
        raise ValueError("radius_cap must be in (0,1)")
    a = np.asarray(coefficients, dtype=float)
    radius = spectral_radius(a)
    # Aim just below the cap so computer rounding does not put the result above it.
    scale = min(1.0, radius_cap * (1 - 1e-12) / radius) if radius else 1.0
    lag_scales = scale ** np.arange(1, len(a) + 1)
    result = a * (lag_scales if a.ndim == 1 else lag_scales[:, None, None])
    if spectral_radius(result) > radius_cap:
        raise ArithmeticError("Companion stabilization failed")
    return result, scale


def _simulate_regime(coefficients, noise, burn_in):
    """Start this regime at zero, discard burn-in, and keep the remaining values.

    Also return the past p values just before the first kept observation, newest
    first. They let tests check the AR equation even at a retained boundary.
    """
    a = np.asarray(coefficients)
    if a.ndim == 1:
        # Preserve the scalar calculation and seeded series from the first handoff.
        history = np.zeros(len(a))
        values = np.empty(len(noise))
        for t, innovation in enumerate(np.asarray(noise).reshape(-1)):
            if t == burn_in:
                retained_lags = history.copy()
            values[t] = a @ history + innovation
            history[1:] = history[:-1].copy()
            history[0] = values[t]
        return values[burn_in:, None].copy(), retained_lags
    p, n, _ = a.shape
    history = np.zeros((p, n))
    lag_matrices = np.concatenate(a, axis=1)
    values = np.empty((len(noise), n))
    for t, innovation in enumerate(noise):
        if t == burn_in:
            retained_lags = history.copy()
        values[t] = lag_matrices @ history.reshape(-1) + innovation
        history[1:] = history[:-1].copy()
        history[0] = values[t]
    return values[burn_in:].copy(), retained_lags


def _draw_regime(config, order, length, coefficient_seed, noise_seed):
    shape = (order,) if config.channels == 1 else (order, config.channels, config.channels)
    original = np.random.default_rng(coefficient_seed).normal(size=shape)
    a, scale = stabilize(original, config.radius_cap)
    noise_shape = (length + config.burn_in,) if config.channels == 1 else (
        length + config.burn_in, config.channels)
    noise = np.random.default_rng(noise_seed).normal(size=noise_shape)
    values, retained_lags = _simulate_regime(a, noise, config.burn_in)
    metadata = {
        "order": order, "coefficients": a.tolist(), "gaussian_draw": original.tolist(),
        "stabilization_scale": scale, "spectral_radius": spectral_radius(a),
        "original_spectral_radius": spectral_radius(original),
        "initial_lags": np.zeros_like(retained_lags).tolist(),
        "retained_initial_lags": retained_lags.tolist(),
        "coefficient_spawn_key": list(coefficient_seed.spawn_key),
        "noise_spawn_key": list(noise_seed.spawn_key),
        "burn_in_discarded": config.burn_in,
        "simulated_observations": length + config.burn_in,
    }
    return values, metadata


def generate_stationary_ar4(config=DataConfig()):
    """By default, start with four previous values set to zero. Generate 10,200
    observations, discard the first 200, and return the remaining 10,000.

    N counts the observations we keep, so the returned array has shape
    (N,n), where n=config.channels. For n=3, start with four zero vectors.
    N, channel count and burn-in length can be changed in DataConfig.

    The same seed and settings reproduce the same coefficients and noise.
    Use separate random streams for the coefficient draws and noise draws.

    Return the observations and a separate dictionary containing the real
    coefficients, seed, and generation settings.
    """
    coefficient_seed, noise_seed = np.random.SeedSequence(config.seed).spawn(2)
    values, metadata = _draw_regime(config, 4, config.n_observations, coefficient_seed, noise_seed)
    truth = {
        "configuration": asdict(config), "kind": "stationary", "channels": config.channels,
        **metadata, "noise_variance_raw": 1.0,
        "noise_covariance_raw": np.eye(config.channels).tolist(),
        "rng": "NumPy PCG64; SeedSequence(seed).spawn(2): coefficients, innovations",
    }
    return values, truth


ORDER_CYCLE = (2, 6, 3, 8)


def generate_switching_ar(config=DataConfig(), regime_length=500):
    """Generate equal-length regimes with orders 2,6,3,8,2,... .

    regime_length counts kept observations, not burn-in. With 10,000 kept values,
    T=500 gives 20 regimes and T=2,000 gives five. Redraw the Gaussian coefficients
    at every boundary. Start each regime at zero and discard its own 200 burn-in
    values before joining it to the preceding regime. Do not carry observations
    from the previous regime into the new regime's simulation.

    Return only observations as the first result. The second result records each
    regime's coefficients, random streams and [start_index,stop_index) boundaries.
    These indices refer to the concatenated retained series and start at zero.
    """
    if (not isinstance(regime_length, int) or isinstance(regime_length, bool)
            or regime_length < 1 or config.n_observations % regime_length):
        raise ValueError("regime_length must be a positive integer dividing n_observations")
    count = config.n_observations // regime_length
    streams = np.random.SeedSequence(config.seed).spawn(count)
    segments, regimes = [], []
    for index, stream in enumerate(streams):
        order = ORDER_CYCLE[index % len(ORDER_CYCLE)]
        coefficient_seed, noise_seed = stream.spawn(2)
        values, metadata = _draw_regime(config, order, regime_length, coefficient_seed, noise_seed)
        segments.append(values)
        regimes.append({"regime_index": index, "start_index": index*regime_length,
                        "stop_index": (index+1)*regime_length, **metadata})
    truth = {
        "configuration": asdict(config), "kind": "switching", "channels": config.channels,
        "regime_length": regime_length, "order_cycle": list(ORDER_CYCLE),
        "n_regimes": count, "switch_indices": [i*regime_length for i in range(1, count)],
        "regimes": regimes, "noise_variance_raw": 1.0,
        "noise_covariance_raw": np.eye(config.channels).tolist(),
        "initialization": "Independent zero start and discarded burn-in for each regime",
        "rng": "PCG64; SeedSequence(seed).spawn(n_regimes); each child spawns coefficient and noise streams",
    }
    return np.concatenate(segments, axis=0), truth


def generate_drifting_ar1(config=DataConfig()):
    """Generate the scalar AR(1) whose coefficient drifts, then jumps.

    For 10,000 kept observations, coefficients at indices 0..4999 go linearly
    from 0.9 to -0.9, including both endpoints. At index 5000 the coefficient
    jumps to 0.8 and stays there. Continue from the preceding observation at
    the jump: do not restart the process or discard another burn-in block.

    Initialization choice: start at zero and discard config.burn_in observations
    at coefficient 0.9 BEFORE the retained drift begins. The assignment fixes the
    retained coefficient path but does not specify a separate drifting burn-in.
    """
    if config.channels != 1:
        raise ValueError("The specified drifting AR(1) example is scalar; use channels=1")
    if config.n_observations < 4 or config.n_observations % 2:
        raise ValueError("Drifting AR(1) needs an even n_observations >= 4")
    half = config.n_observations // 2
    coefficients = np.r_[np.linspace(0.9, -0.9, half), np.full(half, 0.8)]
    noise_seed = np.random.SeedSequence(config.seed).spawn(2)[1]
    noise = np.random.default_rng(noise_seed).normal(size=config.burn_in + config.n_observations)
    previous = 0.0
    for value in noise[:config.burn_in]:
        previous = 0.9 * previous + value
    retained_initial_lag = float(previous)
    values = np.empty(config.n_observations)
    for t, coefficient in enumerate(coefficients):
        values[t] = coefficient * previous + noise[config.burn_in+t]
        previous = values[t]
    truth = {
        "configuration": asdict(config), "kind": "drifting", "channels": 1, "order": 1,
        "coefficients": coefficients.tolist(), "jump_index": half,
        "noise_variance_raw": 1.0, "noise_covariance_raw": [[1.0]],
        "noise_spawn_key": list(noise_seed.spawn_key), "initial_lag": 0.0,
        "retained_initial_lag": retained_initial_lag,
        "burn_in_coefficient": 0.9, "burn_in_discarded": config.burn_in,
        "simulated_observations": config.n_observations + config.burn_in,
        "initialization": "Discard initial burn-in at coefficient 0.9; no reset at the jump",
        "radius_cap_applies_to": "Random stationary/switching coefficients; drifting path is fixed by assignment",
        "source": "Assignment Section 1.2; Ding, Zhou and Tarokh Example 4; 10,000 retained observations",
        "rng": "PCG64; second child of SeedSequence(seed).spawn(2) for noise",
    }
    return values[:, None], truth
