"""Deterministic independent population and online vector checks."""

import unittest
import numpy as np
from experts.multichannel import whittle, direct_yule_walker
from experts.scalar import levinson_durbin
from experts.online import OnlineARExperts, ExpertConfig


def population(a, L=12, q=None):
    """Solve a companion Lyapunov equation, independently of either AR solver."""
    a = np.asarray(a)
    p, n, _ = a.shape
    f = np.zeros((p*n, p*n))
    f[:n] = np.concatenate(a, axis=1)
    f[n:, :-n] = np.eye((p-1)*n)
    noise = np.zeros_like(f)
    noise[:n, :n] = np.eye(n) if q is None else q
    assert max(abs(np.linalg.eigvals(f))) < 1
    size = len(f)
    state = np.linalg.solve(np.eye(size*size)-np.kron(f, f), noise.ravel()).reshape(size, size)
    np.testing.assert_allclose(state, f@state@f.T+noise, atol=1e-12)
    return np.array([(np.linalg.matrix_power(f, k)@state)[:n, :n] for k in range(L+1)])


class MultichannelTests(unittest.TestCase):
    def test_nonsymmetric_population_all_orders_and_plateau(self):
        a = np.zeros((4, 3, 3))
        a[0] = [[.35, .4, -.1], [0, .2, .15], [.1, 0, -.3]]
        a[1] = [[-.12, .03, 0], [.07, .05, .02], [0, -.08, .1]]
        a[2] = np.diag([.08, -.03, .04])
        a[3] = [[.05, .01, -.02], [.03, .02, 0], [.01, .02, -.03]]
        r = population(a)
        self.assertGreater(np.max(abs(r[1]-r[1].T)), .1)
        saved = r.copy()
        for epsilon in (0, 1e-6):
            fit = whittle(r, epsilon=epsilon)
            self.assertTrue(fit.valid.all())
            for p in range(1, 13):
                direct, covariance = direct_yule_walker(r, p, epsilon=epsilon)
                np.testing.assert_allclose(fit.coefficients[p-1, :p], direct, atol=2e-12)
                np.testing.assert_allclose(fit.covariances[p-1], covariance, atol=2e-12)
            if epsilon == 0:
                np.testing.assert_allclose(fit.coefficients[3:, :4],
                                           np.broadcast_to(a, (9, 4, 3, 3)), atol=2e-12)
                np.testing.assert_allclose(fit.coefficients[3:, 4:], 0, atol=2e-12)
                np.testing.assert_allclose(fit.covariances[3:],
                                           np.broadcast_to(np.eye(3), (9, 3, 3)), atol=2e-12)
        np.testing.assert_array_equal(saved, r)

    def test_scalar_reduction(self):
        r = .7**np.arange(13)/(1-.7**2)
        for epsilon in (0, 1e-6):
            scalar = levinson_durbin(r, epsilon=epsilon)
            vector = whittle(r[:, None, None], epsilon=epsilon)
            np.testing.assert_allclose(scalar.coefficients, vector.coefficients[:, :, 0, 0], atol=2e-14)
            np.testing.assert_allclose(scalar.variances, vector.covariances[:, 0, 0], atol=2e-14)

    def test_raw_reflection_norm_can_exceed_one(self):
        a = np.array([[[.4, 2.], [0, .2]]])
        r = population(a, 4, q=np.diag([.01, 1.]))
        fit = whittle(r, epsilon=0)
        self.assertTrue(fit.valid.all())
        self.assertGreater(np.linalg.norm(fit.reflections[0], 2), 1)
        self.assertTrue(np.all(fit.normalized_reflection_norms < 1))

    def test_failure_retains_failed_and_higher_orders(self):
        r = np.zeros((5, 3, 3))
        r[0], r[1], r[2] = np.eye(3), .4*np.eye(3), 1.1*np.eye(3)
        prior = whittle(population(np.array([.3*np.eye(3)]), 4))
        fit = whittle(r, epsilon=0, previous_coefficients=prior.coefficients,
                      previous_covariances=prior.covariances,
                      previous_backward_covariances=prior.backward_covariances)
        self.assertEqual(fit.failed_order, 2)
        np.testing.assert_array_equal(fit.valid, [True, False, False, False])
        for key in ("coefficients", "covariances", "backward_covariances"):
            np.testing.assert_array_equal(getattr(fit, key)[1:], getattr(prior, key)[1:])
        for bad in (np.zeros_like(r), np.full_like(r, np.nan)):
            self.assertEqual(whittle(bad).failed_order, 1)
        negative = r.copy()
        negative[0] = np.diag([1., -1., 1.])
        self.assertEqual(whittle(negative).failed_order, 1)

    def test_seven_channels(self):
        a = np.array([.4*np.eye(7)+np.diag(np.full(6, .15), 1)])
        r = population(a, 8)
        fit = whittle(r)
        self.assertTrue(fit.valid.all())
        for p in range(1, 9):
            expected, covariance = direct_yule_walker(r, p)
            np.testing.assert_allclose(fit.coefficients[p-1, :p], expected, atol=2e-12)
            np.testing.assert_allclose(fit.covariances[p-1], covariance, atol=2e-12)


class VectorOnlineTests(unittest.TestCase):
    def setUp(self):
        self.x = np.random.default_rng(685).normal(size=(100, 3))*[2, 5, 9]+[1, 20, -7]

    def test_normalization_raw_conversion_snapshot_and_lags(self):
        bank = OnlineARExperts().initialize(self.x[:40])
        bank._coefficients[:] = 0
        bank._coefficients[1, 0] = np.arange(9).reshape(3, 3)/10
        bank._coefficients[1, 1] = np.diag([.2, -.3, .4])
        bank._history = np.arange(24).reshape(8, 3)/3
        p = bank.predict()
        expected = (bank._coefficients[1, 0]@bank._history[0]
                    + bank._coefficients[1, 1]@bank._history[1])
        np.testing.assert_allclose(p[1], expected)
        snap = bank.snapshot()
        raw_history = bank.inverse_transform(bank._history)
        raw_prediction = snap["intercepts_raw"][1] + sum(
            snap["coefficients_raw"][1, k]@raw_history[k] for k in range(8))
        np.testing.assert_allclose(raw_prediction, bank.inverse_transform(p)[1])
        scale = self.x[:40].std(axis=0)
        np.testing.assert_allclose(snap["error_covariances_raw"],
            snap["error_covariances_standardized"]*scale[None, :, None]*scale[None, None, :])
        snap["coefficients_raw"][:] = 999
        snap["diagnostics"]["fresh"][:] = False
        snap["normalization"]["scale"][:] = 999
        np.testing.assert_array_equal(p, bank.predict())
        np.testing.assert_allclose(bank.snapshot()["normalization"]["scale"], scale)

    def test_nonsymmetric_ew_products_causality_and_all_valid_solves(self):
        def run(x):
            bank = OnlineARExperts().initialize(x[:40])
            forecasts = []
            r = np.zeros((9, 3, 3))
            z = bank.standardize(x[:40])
            for t in range(40):
                for k in range(9):
                    r[k] = .99*r[k]+.01*(np.outer(z[t], z[t-k]) if t >= k else 0)
            np.testing.assert_allclose(bank.snapshot()["covariances_standardized"], r, atol=2e-16)
            for row in x[40:]:
                forecasts.append(bank.predict())
                s = bank.snapshot()
                for p in np.flatnonzero(s["diagnostics"]["fresh"])+1:
                    a, covariance = direct_yule_walker(s["covariances_standardized"], p)
                    np.testing.assert_allclose(s["coefficients_standardized"][p-1, :p], a, atol=2e-10)
                    np.testing.assert_allclose(s["error_covariances_standardized"][p-1], covariance, atol=2e-10)
                bank.update(row)
            return np.array(forecasts), bank.snapshot()
        a, final_a = run(self.x)
        changed = self.x.copy()
        changed[70:] += 500
        b, final_b = run(changed)
        np.testing.assert_array_equal(a[:31], b[:31])
        np.testing.assert_array_equal(final_a["normalization"]["mean"], final_b["normalization"]["mean"])
        np.testing.assert_array_equal(final_a["normalization"]["scale"], final_b["normalization"]["scale"])

    def test_stale_retention_and_invalid_inputs(self):
        bank = OnlineARExperts().initialize(self.x[:40])
        before = bank.snapshot()
        bank._r[:] = 0
        bank._r[0] = np.eye(3)
        bank._r[1] = .4*np.eye(3)
        bank._r[2] = 1.1*np.eye(3)
        bank.update(before["normalization"]["mean"])
        after = bank.snapshot()
        self.assertEqual(after["diagnostics"]["failed_order"], 2)
        np.testing.assert_array_equal(after["coefficients"][1:], before["coefficients"][1:])
        np.testing.assert_array_equal(after["error_covariances_standardized"][1:],
                                      before["error_covariances_standardized"][1:])
        self.assertTrue((after["diagnostics"]["stale_steps"][1:] > 0).all())
        with self.assertRaises(ValueError):
            bank.update([1, 2])
        with self.assertRaises(ValueError):
            bank.update([1, np.nan, 3])
        constant = OnlineARExperts(ExpertConfig(max_order=4)).initialize(np.ones((20, 7)))
        self.assertEqual(constant.predict().shape, (4, 7))
        self.assertEqual(constant.snapshot()["error_covariances_raw"].shape, (4, 7, 7))


if __name__ == "__main__":
    unittest.main()
