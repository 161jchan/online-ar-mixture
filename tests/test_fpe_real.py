"""FPE denominator/stale rules, causal missing cells, timestamps and cache."""

import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from data.real import causal_fill, parse_dataset, load_real, sha256
from experts.fpe import select_fpe


class FPETests(unittest.TestCase):
    def test_assignment_formula_scalar_and_vector(self):
        for n in (1, 3, 7):
            cov = np.array([np.eye(n)*(.9**p) for p in range(1, 9)])
            choice = select_fpe(cov, .99)
            p = np.arange(1, 9)
            expected = np.linalg.det(cov)*((100+p*n+1)/(100-p*n-1))**n
            np.testing.assert_allclose(np.exp(choice.log_scores), expected, rtol=2e-14)
            self.assertEqual(choice.order, np.argmin(expected)+1)

    def test_stale_invalid_denominator_pd_and_fallback(self):
        cov = np.tile(np.eye(3), (8, 1, 1))
        choice = select_fpe(cov, .9)
        np.testing.assert_array_equal(choice.eligible, [True, True]+[False]*6)
        choice = select_fpe(cov, .99, fresh=np.zeros(8, bool), previous_order=5)
        self.assertEqual(choice.order, 5)
        self.assertTrue(choice.used_fallback)
        self.assertTrue(np.isinf(choice.log_scores).all())
        cov[0] = np.diag([-1, -1, 1])  # Positive determinant alone is insufficient.
        cov[1, 0, 1] = 1
        cov[2] = np.nan
        choice = select_fpe(cov, .99)
        self.assertEqual(choice.order, 4)
        self.assertFalse(choice.eligible[:3].any())
        self.assertEqual(select_fpe(np.ones((3, 1, 1)), .99).order, 1)


class RealDataTests(unittest.TestCase):
    def test_causal_fill_and_leading_missing(self):
        x = [[np.nan, 2], [4, np.nan], [np.nan, 7], [100, 10]]
        a, observed = causal_fill(x)
        np.testing.assert_array_equal(a, [[0, 2], [4, 2], [4, 7], [100, 10]])
        b, _ = causal_fill(x[:3]+[[-999, -999]])
        np.testing.assert_array_equal(a[:3], b[:3])
        self.assertFalse(observed[0, 0])

    def test_monthly_fourth_column_missing_month_and_minus_one(self):
        text = "2000;1;2000.04;12;99;1;1\n2000;3;2000.2;-1;99;1;1\n2000;4;2000.3;17;99;1;1\n"
        x, times, mask, names = parse_dataset("sunspots", text)
        np.testing.assert_array_equal(x[:, 0], [12, 12, 12, 17])
        np.testing.assert_array_equal(mask[:, 0], [True, False, False, True])
        self.assertEqual(times[1], "2000-02-01T00:00:00")
        self.assertEqual(len(names), 1)
        with self.assertRaises(ValueError):
            parse_dataset("sunspots", text+text)

    def test_seven_numeric_channels_and_hourly_gap(self):
        text = ("date,HUFL,HULL,MUFL,MULL,LUFL,LULL,OT\n"
                "2016-07-01 00:00:00,1,2,3,4,5,6,7\n"
                "2016-07-01 02:00:00,8,,10,11,12,13,14\n")
        x, times, mask, names = parse_dataset("etth1", text)
        self.assertEqual(x.shape, (3, 7))
        self.assertEqual(len(names), 7)
        np.testing.assert_array_equal(x[1], x[0])
        self.assertEqual(x[2, 1], 2)
        self.assertFalse(mask[1].any())
        self.assertFalse(mask[2, 1])
        self.assertEqual(times[1], "2016-07-01T01:00:00")

    def test_cache_hash_verified_without_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root/"sunspots.csv"
            path.write_text("2000;1;2000.0;12;0;1;1\n", encoding="utf-8")
            (root/"sunspots.json").write_text(json.dumps({"sha256": sha256(path)}))
            self.assertEqual(load_real("sunspots", root)[0][0, 0], 12)
            path.write_text("changed")
            with self.assertRaises(ValueError):
                load_real("sunspots", root)


if __name__ == "__main__":
    unittest.main()
