import itertools
import math
import unittest

import numpy as np
from scipy.stats import norm

import betting
import robust_studentized_dp as robust_dp
import robust_studentized_experiment as experiment


class RobustStudentizedTests(unittest.TestCase):
    def test_binary_bellman_value_matches_exact_enumeration(self):
        n = 5
        m = 0.35
        z = 1.1
        correction = 0.2
        value = robust_dp.grid_robust_tail_probability(
            n=n,
            m=m,
            z=z,
            correction=correction,
            grid_intervals=1,
        )
        exact = 0.0
        for bits in itertools.product((0.0, 1.0), repeat=n):
            X = np.asarray(bits)
            centered = X - m
            if np.sum(centered) >= (
                z * math.sqrt(float(np.dot(centered, centered)))
                + correction
            ):
                successes = int(np.sum(X))
                exact += m**successes * (1.0 - m) ** (n - successes)
        self.assertAlmostEqual(value, exact, places=12)

    def test_bellman_price_decreases_with_correction(self):
        layers = robust_dp.reachable_state_layers(5, 3)
        prices = [
            robust_dp.grid_robust_tail_probability(
                n=5,
                m=0.4,
                z=1.2,
                correction=correction,
                grid_intervals=3,
                layers=layers,
            )
            for correction in (0.0, 0.2, 0.5)
        ]
        self.assertGreaterEqual(prices[0], prices[1])
        self.assertGreaterEqual(prices[1], prices[2])

    def test_rare_path_forces_the_claimed_correction(self):
        n = 50
        m = 0.4
        alpha = 0.005
        z = float(norm.isf(alpha))
        cutoff = m * (alpha ** (-1.0 / n) - 1.0)
        d = 0.999 * min(1.0 - m, cutoff)
        high_probability = m / (m + d)
        margin = d * (n - z * math.sqrt(n))
        lower_bound = robust_dp.rare_event_lower_bound(
            n, m, alpha, z
        )
        self.assertGreater(high_probability**n, alpha)
        self.assertLess(margin, lower_bound)
        self.assertAlmostEqual(
            lower_bound,
            (n - z * math.sqrt(n)) * min(1.0 - m, cutoff),
            places=12,
        )

    def test_unbiased_quantization_is_grid_valued_and_reproducible(self):
        X = np.array([0.0, 0.13, 0.51, 0.99, 1.0])
        first = robust_dp.unbiased_grid_quantize(
            X, 8, np.random.default_rng(17)
        )
        second = robust_dp.unbiased_grid_quantize(
            X, 8, np.random.default_rng(17)
        )
        np.testing.assert_array_equal(first, second)
        np.testing.assert_allclose(8.0 * first, np.round(8.0 * first))
        self.assertEqual(first[0], 0.0)
        self.assertEqual(first[-1], 1.0)

    def test_bernoulli_rounding_supports_exact_binomial_inversion(self):
        rng = np.random.default_rng(19)
        X = np.linspace(0.05, 0.95, 40)
        rounded = robust_dp.unbiased_grid_quantize(X, 1, rng)
        self.assertTrue(np.all((rounded == 0.0) | (rounded == 1.0)))
        lower, upper = betting.bernoulli_dp_ci_endpoints(
            rounded,
            0.01,
            upper_randomizer=0.4,
            lower_randomizer=0.7,
        )
        self.assertLessEqual(0.0, lower)
        self.assertLessEqual(lower, upper)
        self.assertLessEqual(upper, 1.0)

    def test_optimistic_inversion_has_interval_geometry(self):
        rng = np.random.default_rng(23)
        X = rng.beta(2.0, 4.0, 80)
        lower, upper, empty = (
            robust_dp.optimistic_studentized_ci_endpoints(X, 0.01)
        )
        self.assertFalse(empty)
        self.assertLessEqual(0.0, lower)
        self.assertLessEqual(lower, float(np.mean(X)))
        self.assertLessEqual(float(np.mean(X)), upper)
        self.assertLessEqual(upper, 1.0)

    def test_symmetric_empirical_variance_interval_matches_formula(self):
        X = np.array([0.1, 0.2, 0.4, 0.7, 0.9] * 20)
        delta = 0.01
        lower, upper, empty = (
            experiment.symmetric_empirical_variance_ci_endpoints(X, delta)
        )
        tail_log = math.log(3.0 / delta)
        denominator = 1.0 - math.sqrt(2.0 * tail_log / X.size)
        center = float(np.mean(X))
        radius = (
            math.sqrt(
                2.0 * float(np.sum((X - center) ** 2)) * tail_log
            ) / X.size
            + 3.15 * tail_log / X.size
        ) / denominator
        self.assertFalse(empty)
        self.assertAlmostEqual(lower, max(0.0, center - radius))
        self.assertAlmostEqual(upper, min(1.0, center + radius))


if __name__ == "__main__":
    unittest.main()
