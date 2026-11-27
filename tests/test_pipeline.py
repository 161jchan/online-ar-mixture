"""Check coefficient calculations, forecast timing, and handling of failed fits.

Use tests with mathematically known answers and tests with generated data.
Repeating the same seed must reproduce the same data. Different seeds
produce different data, but the calculations must still pass the checks.
"""

from copy import deepcopy
import unittest
import numpy as np
from data import DataConfig, generate_stationary_ar4
from experts import ExpertConfig, OnlineARExperts
from data.synthetic import companion, spectral_radius, stabilize
from experts.scalar import direct_yule_walker, levinson_durbin


def population_covariances(a, max_lag=8, noise_variance=1.0):
    """Calculate theoretical covariances from known AR coefficients using
    a discrete Lyapunov equation.

    We chose this additional test method so the expected covariance values
    do not come from the Levinson-Durbin code we are testing.
    The assignment does not specifically require the Lyapunov method.
    """
    a = np.asarray(a, dtype=float)
    p = len(a)
    F = np.vstack([a, np.eye(p)[:-1]])
    Q = np.zeros((p, p))
    Q[0, 0] = noise_variance
    P = np.linalg.solve(np.eye(p*p) - np.kron(F, F), Q.ravel()).reshape(p, p)
    np.testing.assert_allclose(P, F @ P @ F.T + Q, atol=2e-14)
    return np.array([(np.linalg.matrix_power(F, h) @ P)[0, 0]
                     for h in range(max_lag + 1)])


class LevinsonTests(unittest.TestCase):
    def assert_direct_agreement(self, r, epsilon):
        saved = r.copy()
        fit = levinson_durbin(r, epsilon=epsilon)
        self.assertTrue(fit.valid.all())
        for p in range(1, len(r)):
            a, variance = direct_yule_walker(r, p, epsilon=epsilon)
            np.testing.assert_allclose(fit.coefficients[p-1, :p], a, rtol=2e-12, atol=2e-12)
            self.assertAlmostEqual(fit.variances[p-1], variance, delta=2e-12)
        np.testing.assert_array_equal(r, saved)
        return fit

    def test_white_noise_population(self):
        r = population_covariances([0.0], noise_variance=2.0)
        fit = self.assert_direct_agreement(r, 0)
        np.testing.assert_array_equal(fit.coefficients, 0)
        np.testing.assert_allclose(fit.variances, 2, atol=1e-14)

    def test_ar1_positive_and_negative_population(self):
        for a in (0.7, -0.7):
            r = population_covariances([a])
            np.testing.assert_allclose(r, a**np.arange(9)/(1-a*a), atol=1e-14)
            fit = self.assert_direct_agreement(r, 0)
            np.testing.assert_allclose(fit.coefficients[:, 0], a, atol=2e-14)
            np.testing.assert_allclose(fit.coefficients[:, 1:], 0, atol=2e-14)
            np.testing.assert_allclose(fit.variances, 1, atol=2e-14)

    def test_nontrivial_ar4_population_and_plateau(self):
        a = np.array([0.55, -0.35, 0.22, 0.10])
        self.assertLess(spectral_radius(a), 1)
        fit = self.assert_direct_agreement(population_covariances(a), 0)
        np.testing.assert_allclose(fit.coefficients[3:, :4], np.tile(a, (5, 1)), atol=3e-14)
        np.testing.assert_allclose(fit.coefficients[3:, 4:], 0, atol=3e-14)
        np.testing.assert_allclose(fit.variances[3:], 1, atol=3e-14)
        self.assertTrue(np.all(np.diff(fit.variances[:4]) < 0))

    def test_loaded_reference_identical_at_every_order(self):
        r = population_covariances([0.55, -0.35, 0.22, 0.10])
        self.assert_direct_agreement(r, 1e-6)

    def test_failed_order_and_higher_preserve_previous(self):
        prior = levinson_durbin(population_covariances([0.55, -0.35, 0.22, 0.10]))
        prior_copy = deepcopy(prior)
        broken = np.array([1, .4, 1.1, 0, 0, 0, 0, 0, 0.])
        fit = levinson_durbin(broken, epsilon=0,
                              previous_coefficients=prior.coefficients,
                              previous_variances=prior.variances)
        self.assertEqual(fit.failed_order, 2)
        np.testing.assert_array_equal(fit.valid, [True] + [False]*7)
        np.testing.assert_array_equal(fit.coefficients[1:], prior.coefficients[1:])
        np.testing.assert_array_equal(fit.variances[1:], prior.variances[1:])
        np.testing.assert_array_equal(prior.coefficients, prior_copy.coefficients)
        self.assertTrue(np.isnan(fit.reflections[2:]).all())

    def test_zero_nonfinite_and_boundary_guards(self):
        for r in (np.zeros(9), np.r_[1., 1., np.zeros(7)],
                  np.r_[1., np.nan, np.zeros(7)], np.r_[-1., np.zeros(8)]):
            fit = levinson_durbin(r, epsilon=0)
            self.assertEqual(fit.failed_order, 1)
            self.assertFalse(fit.valid.any())
            self.assertTrue(np.isfinite(fit.coefficients).all())


class GeneratorTests(unittest.TestCase):
    def test_reproducibility_and_several_seeds(self):
        samples = []
        for seed in (11, 685, 686, 687):
            x, truth = generate_stationary_ar4(DataConfig(seed=seed))
            x2, truth2 = generate_stationary_ar4(DataConfig(seed=seed))
            np.testing.assert_array_equal(x, x2)
            self.assertEqual(truth, truth2)
            self.assertEqual(x.shape, (10000, 1))
            self.assertTrue(np.isfinite(x).all())
            self.assertLessEqual(spectral_radius(truth['coefficients']), .9)
            samples.append(x)
        self.assertFalse(np.array_equal(samples[0], samples[1]))

    def test_eigenvalue_scaling_and_lag_powers(self):
        original = np.array([1.1, -.7, .4, 1.2])
        a, s = stabilize(original)
        np.testing.assert_allclose(a, original*s**np.arange(1, 5), atol=1e-15)
        # Check that multiplying the lag-k coefficient by scale**k changes
        # every companion eigenvalue to its original value multiplied by scale.
        np.testing.assert_allclose(np.sort_complex(np.linalg.eigvals(companion(a))),
                                   np.sort_complex(s*np.linalg.eigvals(companion(original))),
                                   atol=2e-14)

    def test_exact_burn_in_and_innovations(self):
        cfg = DataConfig(seed=23, n_observations=100, burn_in=200)
        x, truth = generate_stationary_ar4(cfg)
        _, innovation_seed = np.random.SeedSequence(cfg.seed).spawn(2)
        noise = np.random.default_rng(innovation_seed).normal(size=300)
        a = np.array(truth['coefficients'])
        for t in range(4, 100):
            self.assertAlmostEqual(x[t, 0] - a @ x[t-4:t, 0][::-1], noise[200+t], delta=2e-14)


class OnlineTests(unittest.TestCase):
    def setUp(self):
        self.values, _ = generate_stationary_ar4(DataConfig(n_observations=400))
        self.bank = OnlineARExperts().initialize(self.values[:40])

    def test_predict_and_snapshot_are_pure(self):
        import pickle
        before = pickle.dumps(self.bank.__dict__)
        p = self.bank.predict()
        self.assertEqual(p.shape, (8, 1))
        np.testing.assert_array_equal(p, self.bank.predict())
        s = self.bank.snapshot()
        s['coefficients'][:] = 999
        s['diagnostics']['fresh'][:] = False
        s['normalization']['mean'] = 999
        self.assertEqual(before, pickle.dumps(self.bank.__dict__))

    def test_lag_order_and_positive_forecast_sign(self):
        # Supply fixed coefficients and different observations at each lag.
        # Compare predict() with a forecast calculated by hand.
        # This catches reversed lags and incorrectly applied coefficient signs.
        self.bank._history = np.arange(1., 9.)
        self.bank._coefficients[:] = 0
        self.bank._coefficients[3, :4] = [.5, -.2, .1, -.05]
        self.assertAlmostEqual(self.bank.predict()[3, 0], .5*1-.2*2+.1*3-.05*4)

    def test_exact_ew_covariance_initialization_and_no_loading_accumulation(self):
        z = (self.values[:40, 0]-self.values[:40, 0].mean())/self.values[:40, 0].std()
        r = np.zeros(9)
        for t, target in enumerate(z):
            for k in range(9):
                product = target*z[t-k] if t >= k else 0
                r[k] = .99*r[k] + .01*product
        np.testing.assert_allclose(self.bank.snapshot()['covariances_standardized'], r, atol=2e-16)
        stored = self.bank.snapshot()['covariances_standardized'].copy()
        for _ in range(5):
            self.bank._fit()
        np.testing.assert_array_equal(self.bank.snapshot()['covariances_standardized'], stored)

    def test_strict_causality_and_frozen_normalization(self):
        def run(x):
            bank = OnlineARExperts().initialize(x[:40])
            forecasts, normalizations = [], []
            for value in x[40:]:
                forecasts.append(bank.predict())
                normalizations.append(bank.snapshot()['normalization'])
                bank.update(value)
            return np.array(forecasts), normalizations
        changed = self.values.copy()
        changed[150:] += 100
        p, norms = run(self.values)
        q, changed_norms = run(changed)
        # Change observations at index 150 and later.
        # Forecasts for index 150 and earlier must stay unchanged:
        # those changed observations had not arrived when the forecasts were made.
        np.testing.assert_array_equal(p[:111], q[:111])
        self.assertEqual(norms, changed_norms)
        self.assertTrue(all(n == norms[0] for n in norms))

    def test_constant_warmup_and_input_validation(self):
        bank = OnlineARExperts().initialize(np.ones(40)*7)
        snap = bank.snapshot()
        self.assertEqual(snap['normalization']['scale'], 1)
        self.assertTrue(snap['normalization']['constant_warmup'])
        self.assertEqual(snap['diagnostics']['failed_order'], 1)
        np.testing.assert_array_equal(bank.predict(), np.zeros((8, 1)))
        np.testing.assert_array_equal(bank.inverse_transform(bank.predict()), np.ones((8, 1))*7)
        with self.assertRaises(ValueError):
            bank.update(np.nan)
        with self.assertRaises(ValueError):
            bank.initialize(np.zeros(39))
        with self.assertRaises(RuntimeError):
            OnlineARExperts().predict()

    def test_online_guard_exposes_stale_estimates(self):
        prior = self.bank.snapshot()
        self.bank._r = np.r_[1., .4, 1.1, np.zeros(6)]
        # Set covariance estimates that deliberately make fitting fail.
        # Then supply an observation whose standardized value is zero, which
        # preserves those invalid covariance relationships.
        # Check that the failed and higher AR orders retain their old coefficients.
        self.bank.update(prior['normalization']['mean'])
        after = self.bank.snapshot()
        self.assertEqual(after['diagnostics']['failed_order'], 2)
        np.testing.assert_array_equal(after['coefficients'][1:], prior['coefficients'][1:])
        np.testing.assert_array_equal(after['error_variances_standardized'][1:],
                                      prior['error_variances_standardized'][1:])
        self.assertTrue(np.all(after['diagnostics']['stale_steps'][1:] >= 1))

    def test_configurable_order(self):
        bank = OnlineARExperts(ExpertConfig(max_order=4)).initialize(self.values[:20])
        self.assertEqual(bank.predict().shape, (4, 1))
        self.assertEqual(bank.snapshot()['error_covariances_standardized'].shape, (4, 1, 1))

    def test_seeded_online_sanity_and_dense_agreement(self):
        for seed in (685, 686, 687):
            x, _ = generate_stationary_ar4(DataConfig(seed=seed, n_observations=2000))
            bank = OnlineARExperts().initialize(x[:40])
            errors = []
            for value in x[40:]:
                prediction = bank.predict()
                self.assertTrue(np.isfinite(prediction).all())
                errors.append((bank.inverse_transform(prediction)[3, 0]-value[0])**2)
                bank.update(value)
                s = bank.snapshot()
                for p in np.flatnonzero(s['diagnostics']['fresh']) + 1:
                    a, v = direct_yule_walker(s['covariances_standardized'], p)
                    np.testing.assert_allclose(s['coefficients'][p-1, :p], a,
                                               atol=2e-10, rtol=2e-10)
                    self.assertAlmostEqual(s['error_variances_standardized'][p-1], v, delta=2e-10)
            # Check that forecast errors remain reasonably sized across several seeds.
            # Do not require exact coefficient recovery or AR(4) to win every run.
            self.assertLess(np.mean(errors[460:]), 3.0)


if __name__ == '__main__':
    unittest.main()
