"""Check lengths, coefficient changes, noise alignment and reproducible draws."""

import unittest
import numpy as np
from data import DataConfig, generate_stationary_ar4, generate_switching_ar, generate_drifting_ar1
from data.synthetic import companion, spectral_radius, stabilize, _simulate_regime


def replay_with_companion(coefficients, noise, burn_in):
    """Rebuild the whole AR state independently, including the discarded values."""
    a = np.asarray(coefficients)
    if a.ndim == 1:
        a = a[:, None, None]
    p, n, _ = a.shape
    first_row = np.hstack([a[k] for k in range(p)])
    if p == 1:
        matrix = first_row
    else:
        matrix = np.vstack([first_row, np.hstack([np.eye((p-1)*n), np.zeros(((p-1)*n, n))])])
    state = np.zeros(p*n)
    values = []
    for t, innovation in enumerate(np.asarray(noise).reshape(-1, n)):
        if t == burn_in:
            initial = state.copy().reshape(p, n)
        state = matrix @ state + np.r_[innovation, np.zeros((p-1)*n)]
        if t >= burn_in:
            values.append(state[:n].copy())
    return np.array(values), initial


class SyntheticTests(unittest.TestCase):
    def test_block_companion_and_cross_channel_orientation(self):
        a = np.array([[[.4, .2, 0], [-.1, .3, .15], [.1, 0, .2]],
                      [[-.12, 0, .05], [.1, -.1, 0], [0, .08, -.15]]])
        expected = np.block([[a[0], a[1]], [np.eye(3), np.zeros((3, 3))]])
        np.testing.assert_array_equal(companion(a), expected)
        noise = np.zeros((8, 3))
        noise[0] = [1, 0, 0]
        actual, _ = _simulate_regime(a, noise, 0)
        np.testing.assert_allclose(actual[1], a[0, :, 0], atol=1e-15)
        np.testing.assert_allclose(actual[2], a[0] @ a[0, :, 0]+a[1, :, 0], atol=1e-15)

    def test_matrix_lag_scaling_and_radius_across_seeds(self):
        for seed in (17, 685, 687):
            for n in (1, 3):
                for p in (2, 3, 4, 6, 8):
                    draw = np.random.default_rng(seed+p).normal(size=(p, n, n))
                    a, scale = stabilize(draw)
                    # A separate matrix identity checks that all eigenvalues scale.
                    diagonal = np.repeat(scale**np.arange(p), n)
                    expected = (scale*companion(draw))*diagonal[None, :]/diagonal[:, None]
                    np.testing.assert_allclose(companion(a), expected, atol=2e-14, rtol=2e-14)
                    self.assertLessEqual(spectral_radius(a), .9)

    def test_stationary_shapes_fixed_coefficients_and_independent_replay(self):
        for n in (1, 3):
            cfg = DataConfig(seed=685, channels=n, n_observations=300)
            values, truth = generate_stationary_ar4(cfg)
            self.assertEqual(values.shape, (300, n))
            self.assertEqual(np.asarray(truth['coefficients']).shape, (4,) if n == 1 else (4,n,n))
            self.assertEqual(truth['simulated_observations'], 500)
            noise_seed = np.random.SeedSequence(cfg.seed, spawn_key=tuple(truth['noise_spawn_key']))
            noise = np.random.default_rng(noise_seed).normal(size=(500, n))
            replay, lags = replay_with_companion(truth['coefficients'], noise, 200)
            np.testing.assert_allclose(values, replay, atol=2e-13, rtol=2e-13)
            np.testing.assert_allclose(np.asarray(truth['retained_initial_lags']).reshape(4,n), lags, atol=2e-13)
            if n == 3:
                a = np.array(truth['coefficients'])
                self.assertGreater(np.abs(a[:, ~np.eye(n, dtype=bool)]).max(), .01)

    def test_required_switch_counts_orders_and_boundaries(self):
        for n in (1, 3):
            for T in (500, 2000):
                values, truth = generate_switching_ar(DataConfig(channels=n), T)
                self.assertEqual(values.shape, (10000, n))
                self.assertTrue(np.isfinite(values).all())
                count = 10000//T
                self.assertEqual(truth['n_regimes'], count)
                self.assertEqual(truth['switch_indices'], list(range(T, 10000, T)))
                for i, regime in enumerate(truth['regimes']):
                    self.assertEqual(regime['order'], (2,6,3,8)[i%4])
                    self.assertEqual((regime['start_index'], regime['stop_index']), (i*T, (i+1)*T))
                    self.assertEqual(regime['simulated_observations'], T+200)
                    self.assertEqual(regime['burn_in_discarded'], 200)
                    self.assertLessEqual(spectral_radius(regime['coefficients']), .9)
                    np.testing.assert_array_equal(regime['initial_lags'], 0)
                if count > 4:
                    self.assertFalse(np.array_equal(truth['regimes'][0]['gaussian_draw'],
                                                    truth['regimes'][4]['gaussian_draw']))

    def test_each_switching_regime_has_its_own_burn_in_and_noise(self):
        for seed in (685, 686, 687):
            for n in (1, 3):
                cfg = DataConfig(seed=seed, channels=n, n_observations=40, burn_in=9)
                values, truth = generate_switching_ar(cfg, regime_length=5)
                noise_keys = []
                for regime in truth['regimes']:
                    key = tuple(regime['noise_spawn_key'])
                    noise_keys.append(key)
                    noise = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=key)).normal(size=(14,n))
                    expected, initial = replay_with_companion(regime['coefficients'], noise, 9)
                    start, stop = regime['start_index'], regime['stop_index']
                    np.testing.assert_allclose(values[start:stop], expected, rtol=5e-13, atol=5e-13)
                    np.testing.assert_allclose(np.array(regime['retained_initial_lags']).reshape(-1,n), initial,
                                               rtol=5e-13, atol=5e-13)
                self.assertEqual(len(set(noise_keys)), len(noise_keys))

    def test_reproducibility_without_changing_global_random_state(self):
        global_before = np.random.get_state()
        for seed in (685, 686, 687):
            for n in (1, 3):
                cfg = DataConfig(seed=seed, channels=n, n_observations=80)
                for generate in (generate_stationary_ar4, lambda c: generate_switching_ar(c, 20)):
                    x, truth = generate(cfg)
                    x2, truth2 = generate(cfg)
                    np.testing.assert_array_equal(x, x2)
                    self.assertEqual(truth, truth2)
                    changed, _ = generate(DataConfig(seed=seed+10, channels=n, n_observations=80))
                    self.assertFalse(np.array_equal(x, changed))
            x, truth = generate_drifting_ar1(DataConfig(seed=seed, n_observations=80))
            x2, truth2 = generate_drifting_ar1(DataConfig(seed=seed, n_observations=80))
            np.testing.assert_array_equal(x, x2)
            self.assertEqual(truth, truth2)
        global_after = np.random.get_state()
        self.assertEqual(global_before[0], global_after[0])
        np.testing.assert_array_equal(global_before[1], global_after[1])
        self.assertEqual(global_before[2:], global_after[2:])

    def test_drift_endpoints_jump_and_continuous_history(self):
        cfg = DataConfig(seed=687)
        values, truth = generate_drifting_ar1(cfg)
        a = np.array(truth['coefficients'])
        self.assertEqual(values.shape, (10000,1))
        self.assertEqual((a[0], a[4999], a[5000], a[-1]), (.9, -.9, .8, .8))
        np.testing.assert_allclose(np.diff(a[:5000]), -1.8/4999, atol=2e-16)
        self.assertEqual(truth['jump_index'], 5000)
        noise = np.random.default_rng(np.random.SeedSequence(cfg.seed, spawn_key=(1,))).normal(size=10200)
        previous = 0.0
        for innovation in noise[:200]:
            previous = .9*previous + innovation
        self.assertEqual(previous, truth['retained_initial_lag'])
        expected = []
        for coefficient, innovation in zip(a, noise[200:]):
            previous = coefficient*previous + innovation
            expected.append(previous)
        np.testing.assert_array_equal(values[:,0], expected)
        self.assertAlmostEqual(values[5000,0], .8*values[4999,0]+noise[5200], delta=1e-14)

    def test_noise_has_unit_variance_and_no_cross_channel_correlation(self):
        for seed in (685, 686, 687):
            cfg = DataConfig(seed=seed, channels=3)
            x, truth = generate_stationary_ar4(cfg)
            a = np.array(truth['coefficients'])
            errors = np.array([x[t] - sum((a[k] @ x[t-k-1] for k in range(4)))
                               for t in range(4,len(x))])
            self.assertLess(np.abs(errors.mean(axis=0)).max(), .06)
            self.assertLess(np.abs(np.cov(errors, rowvar=False, bias=True)-np.eye(3)).max(), .08)

    def test_invalid_configs_and_coefficient_shapes(self):
        for kwargs in ({'channels':0}, {'channels':1.5}, {'seed':-1}, {'n_observations':True},
                       {'n_observations':2.5}, {'burn_in':-1}, {'radius_cap':np.nan}):
            with self.assertRaises(ValueError):
                DataConfig(**kwargs)
        for a in (np.zeros((3,2)), np.zeros((4,2,3)), np.array([np.inf]), np.zeros(0)):
            with self.assertRaises(ValueError):
                companion(a)
        for T in (0, -1, 3000, 500.0, True):
            with self.assertRaises(ValueError):
                generate_switching_ar(DataConfig(), T)
        for cfg in (DataConfig(channels=3), DataConfig(n_observations=3), DataConfig(n_observations=5)):
            with self.assertRaises(ValueError):
                generate_drifting_ar1(cfg)


if __name__ == '__main__':
    unittest.main()
