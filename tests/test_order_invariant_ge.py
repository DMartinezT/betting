import itertools
import unittest

import numpy as np

import betting
from experiments import order_invariant_ge as symmetric_ge


class OrderInvariantGETests(unittest.TestCase):
    def test_stopped_overshoot_differs_only_at_a_crossing(self):
        X = np.linspace(0.2, 0.8, 20)
        delta = 0.01
        target = 2.0 / delta
        for m in (0.3, 0.5, 0.7):
            stopped = symmetric_ge.compute_M_probit_common_clock_stopped_arms(
                X, m, delta
            )
            capped = betting.compute_M_probit_common_clock_arms(X, m, delta)
            for raw_arm, capped_arm in zip(stopped, capped):
                if capped_arm < target - 1e-12:
                    self.assertAlmostEqual(raw_arm, capped_arm, places=12)
                else:
                    self.assertGreaterEqual(raw_arm, target)

    def test_permutation_average_is_exactly_symmetric_for_full_orbit(self):
        X = np.array([0.1, 0.4, 0.9])
        orbit = np.asarray([
            X[list(permutation)]
            for permutation in (
                (0, 1, 2), (0, 2, 1), (1, 0, 2),
                (1, 2, 0), (2, 0, 1), (2, 1, 0),
            )
        ])
        reordered = orbit[:, [2, 0, 1]]
        first = symmetric_ge.permutation_average_arms(
            orbit, 0.45, 0.1, False
        )
        second = symmetric_ge.permutation_average_arms(
            reordered, 0.45, 0.1, False
        )
        np.testing.assert_allclose(first, second, rtol=1e-12, atol=1e-12)

    def test_average_preserves_one_step_null_mean(self):
        m = 0.3
        delta = 0.1
        for retain_overshoot in (False, True):
            zero = np.zeros((5, 1))
            one = np.ones((5, 1))
            zero_arms = symmetric_ge.permutation_average_arms(
                zero, m, delta, retain_overshoot
            )
            one_arms = symmetric_ge.permutation_average_arms(
                one, m, delta, retain_overshoot
            )
            for arm in range(2):
                expectation = (
                    (1.0 - m) * zero_arms[arm] + m * one_arms[arm]
                )
                self.assertAlmostEqual(expectation, 1.0, places=12)

    def test_full_permutation_average_has_valid_exact_bernoulli_mean(self):
        n = 4
        m = 0.35
        delta = 0.1
        permutations = list(itertools.permutations(range(n)))
        for retain_overshoot in (False, True):
            expectation = np.zeros(2)
            for bits in itertools.product((0.0, 1.0), repeat=n):
                X = np.asarray(bits)
                orbit = np.asarray([X[list(order)] for order in permutations])
                arms = symmetric_ge.permutation_average_arms(
                    orbit, m, delta, retain_overshoot
                )
                successes = int(np.sum(X))
                probability = m**successes * (1.0 - m)**(n - successes)
                expectation += probability * np.asarray(arms)
            if retain_overshoot:
                np.testing.assert_allclose(
                    expectation, np.ones(2), rtol=1e-11, atol=1e-11
                )
            else:
                self.assertTrue(np.all(expectation <= 1.0 + 1e-11))

    def test_capped_average_arms_are_ordered_in_candidate_mean(self):
        rng = np.random.default_rng(17)
        X = rng.beta(2.0, 5.0, 40)
        permutations = symmetric_ge.draw_permuted_samples(X, 8, rng)
        audit = symmetric_ge.audit_arm_monotonicity(
            permutations,
            0.05,
            retain_overshoot=False,
            grid_size=65,
        )
        self.assertEqual(audit["plus_violations"], 0)
        self.assertEqual(audit["minus_violations"], 0)

    def test_endpoint_output_is_well_formed(self):
        rng = np.random.default_rng(22)
        X = rng.uniform(0.2, 0.8, 50)
        permutations = symmetric_ge.draw_permuted_samples(X, 6, rng)
        lower, upper, empty = symmetric_ge.permutation_integrated_ci_endpoints(
            permutations,
            0.05,
            randomizers=(0.6, 0.4),
        )
        self.assertFalse(empty)
        self.assertLessEqual(0.0, lower)
        self.assertLessEqual(lower, np.mean(X))
        self.assertLessEqual(np.mean(X), upper)
        self.assertLessEqual(upper, 1.0)

    def test_overshoot_is_not_silently_inverted_as_monotone(self):
        rng = np.random.default_rng(23)
        X = rng.uniform(0.2, 0.8, 20)
        permutations = symmetric_ge.draw_permuted_samples(X, 4, rng)
        with self.assertRaisesRegex(ValueError, "not guaranteed to be monotone"):
            symmetric_ge.permutation_integrated_ci_endpoints(
                permutations,
                0.05,
                retain_overshoot=True,
            )


if __name__ == "__main__":
    unittest.main()
