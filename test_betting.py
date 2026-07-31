import itertools
import unittest
from unittest import mock

import numpy as np
from scipy.optimize import brentq
from scipy.stats import binom, norm

import betting


class HeatFlowConstructionTests(unittest.TestCase):
    def test_time_zero_value_is_I2(self):
        strike, initial_wealth = betting.get_optimal_lambda(0.005)
        value = betting.heat_payoff_value(0.0, 1.0, strike)
        self.assertAlmostEqual(value, initial_wealth, places=13)

    def test_delta_is_derivative_of_continuation_value(self):
        eps = 1e-5
        for x, variance, strike in (
            (-1.0, 1.0, 2.0),
            (0.5, 0.7, 0.2),
            (2.0, 0.1, 1.5),
        ):
            finite_difference = (
                betting.heat_payoff_value(x + eps, variance, strike)
                - betting.heat_payoff_value(x - eps, variance, strike)
            ) / (2.0 * eps)
            delta = betting.heat_payoff_delta(x, variance, strike)
            self.assertAlmostEqual(delta, finite_difference, places=8)

    def test_digital_delta_is_derivative_of_continuation_value(self):
        eps = 1e-5
        for x, variance, boundary in (
            (-1.0, 1.0, 2.0),
            (0.5, 0.7, 0.2),
            (2.0, 0.1, 1.5),
        ):
            finite_difference = (
                betting.digital_payoff_value(
                    x + eps, variance, boundary
                )
                - betting.digital_payoff_value(
                    x - eps, variance, boundary
                )
            ) / (2.0 * eps)
            delta = betting.digital_payoff_delta(
                x, variance, boundary
            )
            self.assertAlmostEqual(delta, finite_difference, places=8)

    def test_finite_sample_wealth_is_nonnegative(self):
        rng = np.random.default_rng(7)
        strike, initial_wealth = betting.get_optimal_lambda(0.005)
        paths = [
            np.zeros(200),
            np.ones(200),
            np.tile(np.array([0.0, 1.0]), 100),
            rng.beta(1, 5, 200),
        ]
        for X in paths:
            for m in (0.05, 0.5, 0.95):
                wealth = betting.compute_M_heat_path(
                    X, m, strike, initial_wealth
                )
                self.assertGreaterEqual(wealth, -1e-12)

                plus, minus = betting.compute_M_heat_trajectory(
                    X, m, strike, initial_wealth
                )
                self.assertGreaterEqual(float(np.min(plus)), -1e-12)
                self.assertGreaterEqual(float(np.min(minus)), -1e-12)

    def test_target_aware_wealth_is_nonnegative(self):
        rng = np.random.default_rng(11)
        delta = 0.01
        boundary = betting.asymptotic_limit_digital(delta)
        paths = [
            np.zeros(100),
            np.ones(100),
            np.tile(np.array([0.0, 1.0]), 50),
            rng.beta(1, 5, 100),
        ]
        for X in paths:
            for m in (0.05, 0.5, 0.95):
                self.assertGreaterEqual(
                    betting.compute_M_star(X, m, delta), -1e-12
                )
                self.assertGreaterEqual(
                    betting.compute_M_hinge_feedback_star(X, m, delta),
                    -1e-12,
                )
                self.assertGreaterEqual(
                    betting.compute_M_capped_feedback_star(X, m, delta),
                    -1e-12,
                )
                self.assertGreaterEqual(
                    betting.compute_M_capped_exponential_feedback_star(
                        X, m, delta
                    ),
                    -1e-12,
                )
                self.assertGreaterEqual(
                    betting.compute_M_bets(X, m, delta), -1e-12
                )
                self.assertGreaterEqual(
                    betting.compute_M_probit_star(
                        X,
                        m,
                        delta,
                        buffer_rounds=float(len(X)) ** (2.0 / 3.0),
                    ),
                    -1e-12,
                )
                self.assertGreaterEqual(
                    betting.compute_M_digital_dp(
                        X, m, delta, boundary
                    ),
                    -1e-12,
                )
                _, initial_wealth = betting.get_optimal_lambda(
                    delta / 2.0
                )
                self.assertGreaterEqual(
                    betting.compute_M_heat_star_path(
                        X, m, delta, initial_wealth
                    ),
                    -1e-12,
                )
                self.assertGreaterEqual(
                    betting.compute_M_heat_capped_star_path(
                        X, m, delta, initial_wealth
                    ),
                    -1e-12,
                )

    def test_target_aware_one_step_null_means(self):
        delta = 0.1
        m = 0.3
        boundary = betting.asymptotic_limit_digital(delta)
        for statistic in (
            lambda x: betting.compute_M_bets(x, m, delta),
            lambda x: betting.compute_M_star(x, m, delta),
            lambda x: betting.compute_M_hinge_feedback_star(x, m, delta),
            lambda x: betting.compute_M_capped_feedback_star(x, m, delta),
            lambda x: betting.compute_M_capped_exponential_feedback_star(
                x, m, delta
            ),
            lambda x: betting.compute_M_probit_star(
                x, m, delta, buffer_rounds=1.0
            ),
            lambda x: betting.compute_M_digital_dp(
                x, m, delta, boundary
            ),
        ):
            zero = statistic(np.array([0.0]))
            one = statistic(np.array([1.0]))
            expectation = (1.0 - m) * zero + m * one
            self.assertLessEqual(expectation, 1.0 + 1e-12)
            self.assertGreaterEqual(expectation, 1.0 - 1e-12)

        _, initial_wealth = betting.get_optimal_lambda(delta / 2.0)
        zero = betting.compute_M_heat_star_path(
            np.array([0.0]), m, delta, initial_wealth
        )
        one = betting.compute_M_heat_star_path(
            np.array([1.0]), m, delta, initial_wealth
        )
        expectation = (1.0 - m) * zero + m * one
        self.assertAlmostEqual(expectation, initial_wealth, places=12)

        capped_zero = betting.compute_M_heat_capped_star_path(
            np.array([0.0]), m, delta, initial_wealth
        )
        capped_one = betting.compute_M_heat_capped_star_path(
            np.array([1.0]), m, delta, initial_wealth
        )
        capped_expectation = (
            (1.0 - m) * capped_zero + m * capped_one
        )
        self.assertAlmostEqual(
            capped_expectation, initial_wealth, places=12
        )

    def test_probit_leverage_matches_inverse_mills_ratio(self):
        for probability in (0.001, 0.005, 0.05, 0.5, 0.95):
            expected = (
                norm.pdf(norm.ppf(probability)) / probability
            )
            observed = betting.probit_target_leverage(probability)
            self.assertAlmostEqual(observed, expected, places=7)

    def test_machine_precision_normal_quantile(self):
        for probability in (
            1e-100,
            1e-20,
            1e-12,
            0.001,
            0.5,
            0.999,
            1.0 - 1e-12,
        ):
            self.assertAlmostEqual(
                betting._normal_ppf(probability),
                norm.ppf(probability),
                places=9,
            )

    def test_randomized_probit_endpoint_tracks_buffered_gaussian(self):
        n = 500
        delta = 0.01
        X = np.tile(np.array([0.0, 1.0]), n // 2)
        buffer_rounds = float(n) ** (2.0 / 3.0)
        endpoints = betting.probit_star_ci_endpoints(
            X,
            delta,
            buffer_rounds=buffer_rounds,
            randomizers=(0.5, 0.5),
        )
        repeated = betting.probit_star_ci_endpoints(
            X,
            delta,
            buffer_rounds=buffer_rounds,
            randomizers=(0.5, 0.5),
        )
        np.testing.assert_allclose(endpoints, repeated, atol=1e-12)

        observed = np.sqrt(n) * (endpoints[1] - endpoints[0])
        buffered_gaussian = (
            norm.isf(delta / 2.0)
            * np.sqrt(1.0 + buffer_rounds / n)
        )
        self.assertLess(abs(observed - buffered_gaussian), 0.01)

        with self.assertRaises(ValueError):
            betting.probit_star_ci_endpoints(
                X,
                delta,
                buffer_rounds=buffer_rounds,
                randomizers=(0.0, 0.5),
            )

    def test_one_step_probit_terminal_randomization_is_valid(self):
        delta = 0.1
        alpha = delta / 2.0
        m = 0.3
        rejection_probability = 0.0
        for x, mass in ((0.0, 1.0 - m), (1.0, m)):
            plus, minus = betting.compute_M_probit_star_arms(
                np.array([x]),
                m,
                delta,
                buffer_rounds=1.0,
            )
            p_plus = min(alpha * plus, 1.0)
            p_minus = min(alpha * minus, 1.0)
            rejection_probability += mass * (
                p_plus + p_minus - p_plus * p_minus
            )
        self.assertLessEqual(rejection_probability, delta + 1e-12)

    def test_experiment_horizon_is_capped_at_one_million(self):
        self.assertEqual(max(betting.DEFAULT_N_VALUES), 1_000_000)
        self.assertEqual(
            betting._validated_n_values((10, 1_000_000)),
            [10, 1_000_000],
        )
        with self.assertRaisesRegex(ValueError, "may not exceed 1000000"):
            betting._validated_n_values((10, 1_000_001))
        counts = betting._simulation_counts(
            [10, 1_000_000], {10: 50, 1_000_000: 20}
        )
        self.assertEqual(counts, {10: 50, 1_000_000: 20})
        self.assertEqual(
            betting.PUBLICATION_SIMULATION_COUNTS[1_000_000], 20
        )
        with self.assertRaisesRegex(ValueError, "no entry"):
            betting._simulation_counts([10, 100], {10: 2})

    def test_rejected_probit_center_is_recorded_as_empty(self):
        X = np.array([0.1, 0.2, 0.3])
        rejection = ValueError(
            "the supplied center is not in the confidence set"
        )
        with mock.patch.object(
            betting,
            "probit_star_randomized_ci_endpoints",
            side_effect=rejection,
        ) as endpoint:
            lower, upper, empty = (
                betting._probit_star_experiment_component(
                    X, 0.01, np.random.default_rng(1)
                )
            )
        self.assertTrue(empty)
        self.assertAlmostEqual(lower, np.mean(X), places=14)
        self.assertEqual(lower, upper)
        endpoint.assert_called_once()

    def test_fixed_and_recalculating_product_bets_agree_initially(self):
        delta = 0.05
        for x in (0.0, 0.2, 1.0):
            X = np.array([x])
            fixed = betting.compute_M_bets(X, 0.4, delta)
            recalculated = betting.compute_M_star(X, 0.4, delta)
            self.assertAlmostEqual(fixed, recalculated, places=13)

    def test_matched_exponential_feedback_is_exact_product_star(self):
        rng = np.random.default_rng(123)
        for X in (rng.beta(2, 2, 100), rng.binomial(1, 0.3, 100)):
            expected = betting.compute_M_star(X, 0.4, 0.01)
            observed = betting._compute_M_recalculating_feedback(
                X, 0.4, 0.01, 1.0, 1.0, 0
            )
            self.assertAlmostEqual(observed, expected, places=12)

    def test_parallel_exponential_feedback_scores_match_product_star(self):
        rng = np.random.default_rng(124)
        X = rng.beta(2, 3, 120)
        means = np.array([0.2, 0.35, 0.5, 0.65])
        observed = betting._recalculating_feedback_scores(
            X, means, 0.01, 0
        )
        expected = np.array(
            [betting.compute_M_star(X, m, 0.01) for m in means]
        )
        np.testing.assert_allclose(observed, expected, rtol=0.0, atol=1e-12)

        with self.assertRaisesRegex(ValueError, "feedback_kind"):
            betting._compute_M_recalculating_feedback(
                X, 0.4, 0.01, 1.0, 1.0, 4
            )

    def test_target_recalculating_hinge_starts_at_original_strike(self):
        for alpha in (0.005, 0.025, 0.1):
            original, _ = betting.get_optimal_lambda(alpha)
            recalculated = betting.target_recalculating_strike(alpha)
            self.assertAlmostEqual(recalculated, original, places=5)

    def test_target_capped_hinge_feedback_is_nearly_digital(self):
        alpha = 0.005
        capped = betting.capped_hinge_target_leverage(alpha)
        digital = betting.probit_target_leverage(alpha)
        product = betting.exponential_target_leverage(alpha)
        self.assertAlmostEqual(capped, 2.8031, delta=5e-4)
        self.assertLess(capped, product)
        self.assertLess(abs(capped - digital), abs(product - digital))

    def test_target_capped_exponential_lookup_matches_direct_inversion(self):
        for probability in (0.005, 0.05, 0.5, 0.9):
            slope = np.sqrt(2.0 * np.log(1.0 / probability))
            strike = brentq(
                lambda value: betting._capped_exponential_price_delta(
                    value, slope
                )[0] - probability,
                -16.0,
                16.0,
            )
            price, delta = betting._capped_exponential_price_delta(
                strike, slope
            )
            observed = betting.capped_exponential_target_leverage(
                probability
            )
            self.assertAlmostEqual(observed, delta / price, delta=2e-5)

        alpha = 0.005
        capped = betting.capped_exponential_target_leverage(alpha)
        self.assertAlmostEqual(capped, 2.5525242, delta=2e-5)
        self.assertLess(capped, betting.exponential_target_leverage(alpha))

    def test_target_capped_lookup_matches_direct_gaussian_inversion(self):
        eta = betting._CAPPED_HINGE_RAMP

        def price_delta(strike):
            upper = strike + eta
            interval = norm.cdf(upper) - norm.cdf(strike)
            phi_lower = norm.pdf(strike)
            phi_upper = norm.pdf(upper)
            first = phi_lower - phi_upper - strike * interval
            second = (
                (1.0 + strike * strike) * interval
                - strike * phi_lower
                + (strike - eta) * phi_upper
            )
            price = norm.sf(upper) + second / (eta * eta)
            delta = 2.0 * first / (eta * eta)
            return price, delta

        for probability in (0.005, 0.05, 0.5, 0.9):
            strike = brentq(
                lambda value: price_delta(value)[0] - probability,
                -9.0,
                12.0,
            )
            price, delta = price_delta(strike)
            expected = delta / price
            observed = betting.capped_hinge_target_leverage(probability)
            self.assertAlmostEqual(observed, expected, delta=2e-5)

    def test_parallel_probit_scores_match_scalar_scores(self):
        rng = np.random.default_rng(987)
        X = rng.beta(2, 5, 80)
        means = np.array([0.15, 0.25, 0.35, 0.45])
        delta = 0.01
        buffer_rounds = float(len(X)) ** (2.0 / 3.0)
        u_plus, u_minus = 0.3, 0.7
        observed = betting._probit_randomized_scores(
            X,
            means,
            delta,
            buffer_rounds,
            u_plus,
            u_minus,
        )
        alpha = delta / 2.0
        expected = []
        for m in means:
            plus, minus = betting.compute_M_probit_star_arms(
                X, m, delta, buffer_rounds=buffer_rounds
            )
            expected.append(
                max(alpha * plus / u_plus, alpha * minus / u_minus)
            )
        np.testing.assert_allclose(observed, expected, rtol=0.0, atol=0.0)

    def test_common_clock_probit_arms_are_ordered_in_candidate_mean(self):
        rng = np.random.default_rng(20260730)
        X = rng.beta(2.0, 5.0, 250)
        means = np.linspace(0.02, 0.98, 97)
        arms = np.asarray([
            betting.compute_M_probit_common_clock_arms(
                X, mean, 0.01, buffer_rounds=0.0
            )
            for mean in means
        ])
        self.assertTrue(np.all(np.diff(arms[:, 0]) <= 1e-11))
        self.assertTrue(np.all(np.diff(arms[:, 1]) >= -1e-11))

    def test_common_clock_hinge_star_arms_are_ordered(self):
        rng = np.random.default_rng(20260802)
        X = rng.beta(2.0, 5.0, 250)
        _, initial_wealth = betting.get_optimal_lambda(0.005)
        means = np.linspace(0.02, 0.98, 97)
        arms = np.asarray([
            betting.compute_M_heat_star_arms(
                X, mean, 0.01, initial_wealth
            )
            for mean in means
        ])
        self.assertTrue(np.all(np.diff(arms[:, 0]) <= 1e-11))
        self.assertTrue(np.all(np.diff(arms[:, 1]) >= -1e-11))

    def test_common_clock_hinge_star_direct_inversion_matches_mesh(self):
        rng = np.random.default_rng(20260803)
        X = rng.beta(2.0, 3.0, 200)
        delta = 0.01
        _, initial_wealth = betting.get_optimal_lambda(delta / 2.0)
        lower, upper, empty = (
            betting.heat_star_common_clock_ci_endpoints(
                X, delta, initial_wealth
            )
        )
        self.assertFalse(empty)
        target = 2.0 * initial_wealth / delta

        def score(mean):
            plus, minus = betting.compute_M_heat_star_arms(
                X, mean, delta, initial_wealth
            )
            return max(plus, minus) / target

        components = betting._confidence_set_components(
            score,
            threshold=1.0,
            scan_points=1001,
        )
        self.assertEqual(len(components), 1)
        np.testing.assert_allclose(
            (lower, upper), components[0], atol=2e-8, rtol=0.0
        )

    def test_parallel_common_clock_scores_match_scalar_scores(self):
        rng = np.random.default_rng(20260731)
        X = rng.uniform(0.0, 1.0, 120)
        means = np.array([0.15, 0.3, 0.5, 0.7, 0.85])
        delta = 0.01
        u_plus, u_minus = 0.4, 0.8
        observed = betting._probit_common_clock_randomized_scores(
            X, means, delta, 0.0, u_plus, u_minus
        )
        alpha = delta / 2.0
        expected = []
        for mean in means:
            plus, minus = betting.compute_M_probit_common_clock_arms(
                X, mean, delta, buffer_rounds=0.0
            )
            expected.append(
                max(alpha * plus / u_plus, alpha * minus / u_minus)
            )
        np.testing.assert_allclose(observed, expected, rtol=0.0, atol=0.0)

    def test_common_clock_direct_inversion_matches_global_mesh(self):
        rng = np.random.default_rng(20260801)
        X = rng.beta(2.0, 3.0, 200)
        delta = 0.01
        for randomizers in ((1.0, 1.0), (0.35, 0.65)):
            lower, upper, empty = betting.probit_common_clock_ci_endpoints(
                X, delta, randomizers=randomizers
            )
            self.assertFalse(empty)
            self.assertLess(upper - lower, 1.0)
            batched = betting.probit_common_clock_batched_ci_endpoints(
                X, delta, randomizers=randomizers
            )
            self.assertFalse(batched[2])
            np.testing.assert_allclose(
                batched[:2], (lower, upper), atol=1e-6, rtol=0.0
            )
        randomizers = (0.35, 0.65)
        scores = lambda means: (
            betting._probit_common_clock_randomized_scores(
                X,
                np.asarray(means),
                delta,
                0.0,
                randomizers[0],
                randomizers[1],
            )
        )
        components = betting._confidence_set_components(
            lambda mean: float(scores(np.asarray([mean]))[0]),
            threshold=1.0,
            scan_points=1001,
            batch_statistic=scores,
        )
        self.assertEqual(len(components), 1)
        np.testing.assert_allclose(
            (lower, upper), components[0], atol=2e-8, rtol=0.0
        )

    def test_geometric_scan_finds_first_rejection_component(self):
        def narrow_rejection_band(m):
            distance = abs(m - 0.5)
            return 2.0 - ((distance - 0.015) / 0.005) ** 2

        lo, hi = betting._interval_component(
            narrow_rejection_band,
            threshold=1.0,
            center=0.5,
            scan_points=56,
            geometric_scan=True,
        )
        self.assertAlmostEqual(lo, 0.49, places=7)
        self.assertAlmostEqual(hi, 0.51, places=7)

    def test_exact_bernoulli_dp_interval_and_size(self):
        X = np.array([0.0, 1.0, 1.0, 0.0, 1.0])
        lo, hi = betting.bernoulli_dp_ci_endpoints(
            X,
            0.1,
            upper_randomizer=0.4,
            lower_randomizer=0.7,
        )
        self.assertGreaterEqual(lo, 0.0)
        self.assertLessEqual(hi, 1.0)
        self.assertLessEqual(lo, hi)
        zero_lo, zero_hi = betting.bernoulli_dp_ci_endpoints(
            np.zeros(5),
            0.1,
            upper_randomizer=0.4,
            lower_randomizer=0.01,
        )
        self.assertEqual(zero_lo, 0.0)
        self.assertEqual(zero_hi, 0.0)
        one_lo, one_hi = betting.bernoulli_dp_ci_endpoints(
            np.ones(5),
            0.1,
            upper_randomizer=0.01,
            lower_randomizer=0.4,
        )
        self.assertEqual(one_lo, 1.0)
        self.assertEqual(one_hi, 1.0)
        with self.assertRaises(ValueError):
            betting.bernoulli_dp_ci_endpoints(
                np.array([0.0, 0.5, 1.0]), 0.1
            )

        n = 8
        m = 0.3
        alpha = 0.05
        rejection_probability = 0.0
        for successes in range(n + 1):
            mass = binom.pmf(successes, n, m)
            strict_upper = binom.sf(successes, n, m)
            strict_lower = binom.cdf(successes - 1, n, m)
            at_successes = binom.pmf(successes, n, m)
            upper_reject = np.clip(
                (alpha - strict_upper) / at_successes, 0.0, 1.0
            )
            lower_reject = np.clip(
                (alpha - strict_lower) / at_successes, 0.0, 1.0
            )
            either_reject = (
                upper_reject
                + lower_reject
                - upper_reject * lower_reject
            )
            rejection_probability += mass * either_reject
        self.assertLessEqual(rejection_probability, 2.0 * alpha + 1e-12)

    def test_one_step_null_and_one_sided_supermartingale_means(self):
        delta = 0.1
        m = 0.3
        strike, initial_wealth = betting.get_optimal_lambda(delta / 2.0)

        def expected_arm_wealth(success_probability, arm):
            plus_zero, minus_zero = betting.compute_M_heat_trajectory(
                np.array([0.0]), m, strike, initial_wealth
            )
            plus_one, minus_one = betting.compute_M_heat_trajectory(
                np.array([1.0]), m, strike, initial_wealth
            )
            zero = plus_zero[-1] if arm == "plus" else minus_zero[-1]
            one = plus_one[-1] if arm == "plus" else minus_one[-1]
            return (1.0 - success_probability) * zero + success_probability * one

        self.assertAlmostEqual(
            expected_arm_wealth(m, "plus"), initial_wealth, places=13
        )
        self.assertAlmostEqual(
            expected_arm_wealth(m, "minus"), initial_wealth, places=13
        )
        self.assertLessEqual(
            expected_arm_wealth(0.2, "plus"), initial_wealth + 1e-14
        )
        self.assertLessEqual(
            expected_arm_wealth(0.4, "minus"), initial_wealth + 1e-14
        )

    def test_exact_bernoulli_tree_satisfies_ville_bound(self):
        n = 8
        m = 0.5
        delta = 0.2
        strike, initial_wealth = betting.get_optimal_lambda(delta / 2.0)
        crossing_probability = 0.0

        for bits in itertools.product((0.0, 1.0), repeat=n):
            X = np.asarray(bits)
            plus, minus = betting.compute_M_heat_trajectory(
                X, m, strike, initial_wealth
            )
            two_sided = 0.5 * (plus + minus)
            crossed = np.max(two_sided) >= initial_wealth / delta
            successes = int(np.sum(X))
            path_probability = (
                m**successes * (1.0 - m) ** (n - successes)
            )
            crossing_probability += path_probability * crossed

        self.assertLessEqual(crossing_probability, delta + 1e-14)

    def test_star_statistic_has_a_quasiconvexity_counterexample(self):
        X = np.r_[1.0, np.zeros(19)]
        means = (0.047125, 0.997, 0.997875)
        values = [betting.compute_M_star(X, m, 0.01) for m in means]
        self.assertGreater(values[1], max(values[0], values[2]))

    def test_full_set_inversion_reports_disconnected_components(self):
        statistic = lambda m: min(abs(m - 0.25) - 0.05, abs(m - 0.75) - 0.05)
        components = betting._confidence_set_components(
            statistic,
            threshold=0.0,
            scan_points=1001,
        )
        self.assertEqual(len(components), 2)
        np.testing.assert_allclose(
            components,
            ((0.2, 0.3), (0.7, 0.8)),
            atol=1e-9,
            rtol=0.0,
        )
        widths = betting._confidence_set_widths(components, center=0.25)
        self.assertAlmostEqual(widths["total_length"], 0.2, places=8)
        self.assertAlmostEqual(widths["hull_width"], 0.6, places=8)
        self.assertAlmostEqual(
            widths["largest_component_width"], 0.1, places=8
        )
        self.assertAlmostEqual(
            widths["center_component_width"], 0.1, places=8
        )

    def test_adaptive_inversion_recovers_disconnected_components(self):
        statistic = lambda m: min(
            abs(m - 0.25) - 0.05,
            abs(m - 0.75) - 0.05,
        )
        components, diagnostics = (
            betting._adaptive_confidence_set_components(
                statistic,
                threshold=0.0,
                center=0.5,
                standard_error=0.05,
                batch_statistic=lambda means: np.minimum(
                    np.abs(means - 0.25) - 0.05,
                    np.abs(means - 0.75) - 0.05,
                ),
                verification_scan_points=257,
            )
        )
        np.testing.assert_allclose(
            components,
            ((0.2, 0.3), (0.7, 0.8)),
            atol=1e-8,
            rtol=0.0,
        )
        self.assertEqual(diagnostics["component_count"], 2)
        self.assertAlmostEqual(
            diagnostics["hull_width"], 0.6, places=8
        )
        self.assertAlmostEqual(
            diagnostics["largest_component_width"], 0.1, places=8
        )
        self.assertTrue(diagnostics["fragmentation_detected"])

    def test_clipped_heat_confidence_set_can_be_disconnected(self):
        delta = 0.01
        strike, initial_wealth = betting.get_optimal_lambda(delta / 2.0)
        X = np.zeros(100)
        X[[2, 12, 17, 23, 28, 30, 33, 34, 68]] = 1.0
        statistic = lambda m: betting.compute_M_heat_path(
            X, m, strike, initial_wealth
        )
        components = betting._confidence_set_components(
            statistic,
            threshold=initial_wealth / delta,
            scan_points=4097,
        )
        self.assertEqual(len(components), 2)
        np.testing.assert_allclose(
            components,
            ((0.0, 0.1958495052), (0.2000967565, 0.3151697627)),
            atol=2e-9,
            rtol=0.0,
        )
        widths = betting._confidence_set_widths(
            components, center=np.mean(X)
        )
        self.assertAlmostEqual(widths["total_length"], 0.3109225114, places=8)
        self.assertAlmostEqual(widths["hull_width"], 0.3151697627, places=8)
        self.assertAlmostEqual(
            widths["largest_component_width"], 0.1958495052, places=8
        )
        self.assertAlmostEqual(
            widths["center_component_width"], 0.1958495052, places=8
        )
        lower, upper, empty, hull_widths = (
            betting._confidence_set_hull_endpoints(
                statistic,
                threshold=initial_wealth / delta,
                center=np.mean(X),
                scan_points=4097,
            )
        )
        self.assertFalse(empty)
        self.assertAlmostEqual(lower, 0.0, places=12)
        self.assertAlmostEqual(upper, 0.3151697627, places=8)
        self.assertEqual(hull_widths["component_count"], 2)
        plus, minus = betting.compute_M_heat_trajectory(
            X, 0.21, strike, initial_wealth
        )
        self.assertTrue(np.all(plus >= 0.0))
        self.assertTrue(np.all(minus >= 0.0))

    def test_clipped_heat_upper_arm_need_not_be_ordered(self):
        delta = 0.01
        strike, initial_wealth = betting.get_optimal_lambda(delta / 2.0)
        X = np.array([
            0.4, 0.4, 0.0, 0.0, 0.5, 0.5, 1.0, 0.0,
            0.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0,
        ])

        def upper_arm(mean):
            plus_path, _ = betting.compute_M_heat_trajectory(
                X, mean, strike, initial_wealth
            )
            return plus_path[-1]

        components = betting._confidence_set_components(
            upper_arm,
            threshold=initial_wealth / (delta / 2.0),
            scan_points=4097,
        )
        self.assertEqual(len(components), 2)
        np.testing.assert_allclose(
            components,
            ((0.1516269869, 0.2284890051), (0.2316502000, 1.0)),
            atol=2e-9,
            rtol=0.0,
        )

    def test_exposed_arms_preserve_existing_two_sided_statistics(self):
        rng = np.random.default_rng(20260730)
        X = rng.beta(2.0, 2.0, 80)
        mean = 0.43
        delta = 0.01
        strike, initial_wealth = betting.get_optimal_lambda(delta / 2.0)
        comparisons = (
            (
                betting.compute_M_inf(X, mean, delta),
                betting.compute_M_inf_arms(X, mean, delta),
            ),
            (
                betting.compute_M_star(X, mean, delta),
                betting.compute_M_star_arms(X, mean, delta),
            ),
            (
                betting.compute_M_hinge_feedback_star(X, mean, delta),
                betting.compute_M_recalculating_feedback_arms(
                    X, mean, delta, 1
                ),
            ),
            (
                betting.compute_M_heat_path(
                    X, mean, strike, initial_wealth
                ),
                betting.compute_M_heat_path_arms(
                    X, mean, strike, initial_wealth
                ),
            ),
        )
        for combined, arms in comparisons:
            self.assertAlmostEqual(combined, 0.5 * sum(arms), places=12)

    def test_randomized_markov_thresholds_shrink_ordered_intervals(self):
        rng = np.random.default_rng(11)
        X = rng.beta(2.0, 2.0, 250)
        delta = 0.01
        _, initial_wealth = betting.get_optimal_lambda(delta / 2.0)

        heat_det = betting.heat_star_common_clock_ci_endpoints(
            X, delta, initial_wealth, randomizers=(1.0, 1.0)
        )
        heat_rand = betting.heat_star_common_clock_ci_endpoints(
            X, delta, initial_wealth, randomizers=(0.4, 0.7)
        )
        self.assertFalse(heat_det[2])
        self.assertFalse(heat_rand[2])
        self.assertGreaterEqual(heat_rand[0], heat_det[0])
        self.assertLessEqual(heat_rand[1], heat_det[1])

        efficient_det = betting.probit_common_clock_ci_endpoints(
            X, delta, randomizers=(1.0, 1.0)
        )
        efficient_rand = betting.probit_common_clock_ci_endpoints(
            X, delta, randomizers=(0.4, 0.7)
        )
        self.assertFalse(efficient_det[2])
        self.assertFalse(efficient_rand[2])
        self.assertGreaterEqual(efficient_rand[0], efficient_det[0])
        self.assertLessEqual(efficient_rand[1], efficient_det[1])

    def test_randomized_product_orthant_gaffke_endpoints(self):
        from compare_markov_calibrations_large import (
            fast_gaffke_ci,
            randomized_product_orthant_gaffke_ci,
        )

        delta = 0.01
        tail_probability = delta / 2.0
        X = np.full(8, 0.4)
        deterministic = fast_gaffke_ci(
            X,
            delta,
            binary=False,
            exact_cutoff=3000,
        )[:2]
        randomizers = (0.25, 0.75)
        observed = randomized_product_orthant_gaffke_ci(
            X,
            delta,
            deterministic,
            randomizers,
        )
        expected = (
            0.4 * (tail_probability / randomizers[0]) ** (1.0 / len(X)),
            1.0
            - 0.6
            * (tail_probability / randomizers[1]) ** (1.0 / len(X)),
        )
        np.testing.assert_allclose(observed, expected, atol=1e-14, rtol=0.0)
        self.assertGreaterEqual(observed[0], deterministic[0])
        self.assertLessEqual(observed[1], deterministic[1])

        mixed = np.asarray([0.1, 0.4, 0.8])
        unchanged = randomized_product_orthant_gaffke_ci(
            mixed,
            delta,
            (0.2, 0.7),
            randomizers,
        )
        self.assertEqual(unchanged, (0.2, 0.7))

    def test_scaled_width_matches_bentkus_limit(self):
        delta = 0.01
        strike, initial_wealth = betting.get_optimal_lambda(delta / 2.0)
        X = np.tile(np.array([0.0, 1.0]), 2500)
        lo, hi = betting.heat_ci_endpoints(
            X,
            delta,
            strike,
            initial_wealth,
        )
        observed = np.sqrt(len(X)) * (hi - lo)
        target = betting.asymptotic_limit_heat(delta)
        self.assertAlmostEqual(observed, target, delta=0.015)


if __name__ == "__main__":
    unittest.main()
