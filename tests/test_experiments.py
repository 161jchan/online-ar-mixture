"""Check the saved handoff timing, FPE choices and observed-target masks."""

import unittest
import numpy as np
from data.real import parse_dataset
from experiments.experts_fpe import run_stream
from experts.fpe import select_fpe
from experts.online import ExpertConfig


class HandoffTests(unittest.TestCase):
    def test_saved_vector_forecasts_fpe_and_future_invariance(self):
        x = np.random.default_rng(91).normal(size=(85, 3))
        cfg = ExpertConfig(max_order=4)
        a, summary = run_stream(x, cfg)
        changed = x.copy()
        changed[55:] += [20, -30, 40]
        b, _ = run_stream(changed, cfg)
        np.testing.assert_array_equal(a["forecasts_standardized"][:36],
                                      b["forecasts_standardized"][:36])
        np.testing.assert_array_equal(a["fpe_order"][:36], b["fpe_order"][:36])
        np.testing.assert_array_equal(a["target_indices"], np.arange(20, 85))
        self.assertEqual(a["error_covariances_raw"].shape, (65, 4, 3, 3))
        previous = 1
        for j in range(65):
            choice = select_fpe(a["error_covariances_standardized"][j], .99,
                                fresh=a["fresh"][j], previous_order=previous)
            previous = choice.order
            self.assertEqual(choice.order, a["fpe_order"][j])
            np.testing.assert_array_equal(a["fpe_forecasts_raw"][j],
                                          a["forecasts_raw"][j, choice.order-1])
        self.assertGreater(summary["direct_solve"]["valid_order_comparisons"], 0)


    def test_common_interval_matches_saved_losses_for_each_L(self):
        from pathlib import Path
        import tempfile
        from experiments.experts_fpe import common_interval_loss
        x = np.random.default_rng(93).normal(size=(100, 3))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"observations.npz"
            np.savez_compressed(path, observations_raw=x)
            for L in (4, 8, 12):
                arrays, summary = run_stream(x, ExpertConfig(max_order=L))
                summary["source_file"] = str(path)
                expert, fpe = common_interval_loss(summary)
                mask = arrays["target_indices"] >= 60
                expected = ((arrays["forecasts_raw"]-arrays["targets_raw"][:, None])**2).sum(axis=2)
                expected_fpe = ((arrays["fpe_forecasts_raw"]-arrays["targets_raw"])**2).sum(axis=1)
                np.testing.assert_allclose(expert, expected[mask].mean(axis=0), atol=1e-14)
                self.assertAlmostEqual(fpe, expected_fpe[mask].mean(), delta=1e-14)


    def test_incomplete_targets_excluded_and_upstream_interpolation_masked(self):
        x = np.random.default_rng(92).normal(size=(50, 1))
        observed = np.ones_like(x, bool)
        observed[30] = False
        _, summary = run_stream(x, ExpertConfig(max_order=4), observed=observed)
        self.assertEqual(summary["scored_observations"], 29)
        self.assertEqual(summary["excluded_incomplete_targets"], 1)
        text = "1824;1;1824.0;10;0;1;1\n1824;2;1824.1;99;0;0;1\n1824;3;1824.2;20;0;1;1\n"
        values, _, mask, _ = parse_dataset("sunspots", text)
        np.testing.assert_array_equal(values[:, 0], [10, 10, 20])
        self.assertFalse(mask[1, 0])


if __name__ == "__main__":
    unittest.main()
