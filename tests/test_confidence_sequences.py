#!/usr/bin/env python3
"""Tests for the chronological confidence-sequence constructions."""

import itertools
import json
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

try:
    import betting
    betting.get_optimal_lambda
except (ImportError, AttributeError):
    from betting import betting
import confidence_sequences as cs


class ConfidenceSequenceProcessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.delta = 0.1
        cls.horizons, cls.strikes, cls.weights, cls.cash = (
            cs.bentkus_horizon_schedule(32, cls.delta)
        )
        cls.heat_config = {
            "horizons": cls.horizons,
            "strikes": cls.strikes,
            "weights": cls.weights,
            "cash_weight": cls.cash,
        }
        (
            cls.scale_fractions,
            cls.scale_weights,
            cls.scale_cash,
        ) = cs.product_scale_schedule(32)
        cls.scale_config = {
            "fractions": cls.scale_fractions,
            "weights": cls.scale_weights,
            "cash_weight": cls.scale_cash,
        }
        # Warm all scalar and parallel kernels once.
        warm = np.array([0.0, 1.0, 0.25, 0.75])
        means = np.array([0.25, 0.5, 0.75])
        times = np.array([1, 2, 4])
        cs.hgkelly_log_e_path(warm, 0.5, G=4)
        cs.product_scale_mixture_log_e_path(
            warm,
            0.5,
            cls.scale_fractions,
            cls.scale_weights,
            cls.scale_cash,
        )
        cs.agrapa_log_e_path(warm, 0.5)
        cs.bentkus_mixture_log_e_path(
            warm,
            0.5,
            cls.horizons,
            cls.strikes,
            cls.weights,
            cls.cash,
        )
        cs.heat_constrained_agrapa_log_e_path(
            warm,
            0.5,
            cls.horizons,
            cls.strikes,
            cls.weights,
            cls.cash,
        )
        for method, config in (
            ("hgkelly", {"G": 4}),
            ("product_scale_mixture", cls.scale_config),
            ("agrapa", {}),
            ("bentkus_mixture", cls.heat_config),
            ("heat_constrained_agrapa", cls.heat_config),
        ):
            cs.running_log_e_at_times(
                warm, means, times, method, method_config=config
            )

    def test_product_scale_schedule_is_normalized_with_cash_tail(self):
        fractions, weights, cash = cs.product_scale_schedule(
            32,
            weight_power=2.0,
            horizon_overshoot=2.0,
            scale_ratio=2.0,
        )
        expected_fractions = 2.0 ** (
            -0.5 * np.arange(1, len(fractions) + 1)
        )
        expected_weights = (
            1.0
            / ((np.pi**2 / 6.0) * np.arange(1, len(weights) + 1) ** 2)
        )
        np.testing.assert_allclose(fractions, expected_fractions, rtol=0.0)
        np.testing.assert_allclose(weights, expected_weights, rtol=1e-14)
        self.assertLessEqual(fractions[-1], 1.0 / np.sqrt(64.0))
        if len(fractions) > 1:
            self.assertGreater(fractions[-2], 1.0 / np.sqrt(64.0))
        self.assertGreater(cash, 0.0)
        self.assertAlmostEqual(float(np.sum(weights)) + cash, 1.0, places=14)

    def test_horizon_schedule_is_normalized_and_calibrated(self):
        self.assertEqual(self.horizons[0], 1)
        self.assertGreaterEqual(self.horizons[-1], 64)
        self.assertTrue(np.all(np.diff(self.horizons) > 0))
        self.assertTrue(np.all(self.weights > 0.0))
        self.assertAlmostEqual(
            float(np.sum(self.weights)) + self.cash, 1.0, places=14
        )
        self.assertGreater(self.cash, 0.0)

        for strike, weight in zip(self.strikes, self.weights):
            expected, _ = betting.get_optimal_lambda(
                self.delta * weight / 2.0
            )
            self.assertAlmostEqual(strike, expected, places=12)

    def test_hgkelly_matches_manual_component_average(self):
        X = np.array([0.0, 0.3, 1.0, 0.6])
        m = 0.4
        G = 5
        observed = np.exp(cs.hgkelly_log_e_path(X, m, G=G))
        plus = np.ones(G)
        minus = np.ones(G)
        expected = [1.0]
        for x in X:
            for g_index in range(G):
                fraction = (g_index + 1.0) / (G + 1.0)
                plus[g_index] *= 1.0 + fraction * (x - m) / m
                minus[g_index] *= (
                    1.0 - fraction * (x - m) / (1.0 - m)
                )
            expected.append(0.5 * (np.mean(plus) + np.mean(minus)))
        np.testing.assert_allclose(observed, expected, rtol=1e-13, atol=1e-14)

    def test_product_scale_mixture_matches_exact_weighted_components(self):
        X = np.array([0.0, 0.3, 1.0, 0.6])
        m = 0.4
        fractions = np.array([0.5, 0.25, 0.125])
        weights = np.array([0.4, 0.2, 0.1])
        cash = 0.3
        observed = np.exp(
            cs.product_scale_mixture_log_e_path(
                X, m, fractions, weights, cash
            )
        )
        plus = np.ones(len(fractions))
        minus = np.ones(len(fractions))
        expected = [1.0]
        for x in X:
            plus *= 1.0 + fractions * (x - m) / m
            minus *= 1.0 - fractions * (x - m) / (1.0 - m)
            expected.append(
                cash + 0.5 * np.sum(weights * (plus + minus))
            )
        np.testing.assert_allclose(observed, expected, rtol=1e-13, atol=1e-14)

    def test_agrapa_matches_manual_regularized_recursion(self):
        X = np.array([0.1, 0.9, 0.2, 0.7])
        m = 0.35
        c = 0.5
        observed = np.exp(cs.agrapa_log_e_path(X, m, c=c))
        wealth = 1.0
        sum_x = 0.0
        variance_numerator = 0.0
        expected = [wealth]
        for i, x in enumerate(X):
            mean_hat = (0.5 + sum_x) / (1.0 + i)
            variance_hat = (0.25 + variance_numerator) / (1.0 + i)
            difference = mean_hat - m
            stake = difference / (
                variance_hat + difference * difference
            )
            stake = np.clip(stake, -c / (1.0 - m), c / m)
            wealth *= 1.0 + stake * (x - m)
            expected.append(wealth)
            sum_x += x
            updated_mean = (0.5 + sum_x) / (2.0 + i)
            variance_numerator += (x - updated_mean) ** 2
        np.testing.assert_allclose(observed, expected, rtol=1e-13, atol=1e-14)

    def test_all_processes_have_exact_one_step_null_mean(self):
        m = 0.3

        def process_values(path_function):
            zero = np.exp(path_function(np.array([0.0]), m)[-1])
            one = np.exp(path_function(np.array([1.0]), m)[-1])
            return zero, one

        functions = (
            lambda X, candidate: cs.hgkelly_log_e_path(X, candidate),
            lambda X, candidate: cs.product_scale_mixture_log_e_path(
                X,
                candidate,
                self.scale_fractions,
                self.scale_weights,
                self.scale_cash,
            ),
            lambda X, candidate: cs.agrapa_log_e_path(X, candidate),
            lambda X, candidate: cs.bentkus_mixture_log_e_path(
                X,
                candidate,
                self.horizons,
                self.strikes,
                self.weights,
                self.cash,
            ),
            lambda X, candidate: cs.heat_constrained_agrapa_log_e_path(
                X,
                candidate,
                self.horizons,
                self.strikes,
                self.weights,
                self.cash,
            ),
        )
        for function in functions:
            zero, one = process_values(function)
            expectation = (1.0 - m) * zero + m * one
            self.assertAlmostEqual(expectation, 1.0, places=12)

    def test_single_heat_expert_matches_existing_trajectory(self):
        rng = np.random.default_rng(12)
        X = rng.beta(2.0, 3.0, 40)
        m = 0.37
        strike, initial = betting.get_optimal_lambda(0.01)
        plus, minus = betting.compute_M_heat_trajectory(
            X, m, strike, initial
        )
        expected = 0.5 * (plus + minus) / initial
        observed = np.exp(
            cs.bentkus_mixture_log_e_path(
                X,
                m,
                np.array([len(X)]),
                np.array([strike]),
                np.array([1.0]),
                0.0,
            )
        )
        np.testing.assert_allclose(observed, expected, rtol=1e-12, atol=1e-13)

    def test_heat_experts_stop_and_freeze_at_maturity(self):
        X = np.array([0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0])
        strike, _ = betting.get_optimal_lambda(0.02)
        path = cs.bentkus_mixture_log_e_path(
            X,
            0.4,
            np.array([3]),
            np.array([strike]),
            np.array([1.0]),
            0.0,
        )
        np.testing.assert_allclose(path[3:], path[3], rtol=0.0, atol=0.0)

    def test_unit_solvency_cap_is_exact_at_worst_endpoint(self):
        # At H=1, m=1/2, and strike zero, the raw heat delta exceeds
        # M_0/m.  A zero observation therefore binds the plus cap and its
        # theoretically zero wealth is formed without unstable subtraction.
        observed = cs.bentkus_mixture_log_e_path(
            np.array([0.0]),
            0.5,
            np.array([1]),
            np.array([0.0]),
            np.array([1.0]),
            0.0,
            solvency_fraction=1.0,
        )
        self.assertTrue(np.all(np.isfinite(observed)))
        self.assertEqual(observed[1], 0.0)

    def test_heat_mixture_equals_weighted_single_experts(self):
        X = np.array([0.1, 0.7, 0.2, 0.8, 0.4, 0.9])
        horizons = np.array([3, 7])
        strikes = np.array([1.2, 2.0])
        weights = np.array([0.4, 0.35])
        cash = 0.25
        mixture = np.exp(
            cs.bentkus_mixture_log_e_path(
                X, 0.45, horizons, strikes, weights, cash
            )
        )
        expected = np.full(len(X) + 1, cash)
        for horizon, strike, weight in zip(horizons, strikes, weights):
            expert = np.exp(
                cs.bentkus_mixture_log_e_path(
                    X,
                    0.45,
                    np.array([horizon]),
                    np.array([strike]),
                    np.array([1.0]),
                    0.0,
                )
            )
            expected += weight * expert
        np.testing.assert_allclose(mixture, expected, rtol=2e-12, atol=2e-13)

    def test_adversarial_paths_remain_nonnegative_and_finite(self):
        rng = np.random.default_rng(91)
        paths = (
            np.zeros(100),
            np.ones(100),
            np.tile(np.array([0.0, 1.0]), 50),
            rng.beta(0.5, 0.5, 100),
        )
        for X in paths:
            for m in (0.01, 0.5, 0.99):
                heat = cs.bentkus_mixture_log_e_path(
                    X,
                    m,
                    self.horizons,
                    self.strikes,
                    self.weights,
                    self.cash,
                )
                constrained = cs.heat_constrained_agrapa_log_e_path(
                    X,
                    m,
                    self.horizons,
                    self.strikes,
                    self.weights,
                    self.cash,
                )
                self.assertTrue(np.all(np.isfinite(heat)))
                self.assertTrue(np.all(np.isfinite(constrained)))
                self.assertTrue(np.all(np.exp(heat) >= 0.0))
                self.assertTrue(np.all(np.exp(constrained) >= 0.0))

    def test_constrained_heat_is_predictable_and_initially_capped(self):
        X = np.array([1.0, 0.0, 1.0, 0.0, 0.2, 0.8])
        full = cs.heat_constrained_agrapa_log_e_path(
            X,
            0.5,
            self.horizons,
            self.strikes,
            self.weights,
            self.cash,
        )
        # At m=prior_mean the aGRAPA cap is zero in round one.
        self.assertAlmostEqual(full[1], 0.0, places=15)
        for prefix in range(1, len(X) + 1):
            truncated = cs.heat_constrained_agrapa_log_e_path(
                X[:prefix],
                0.5,
                self.horizons,
                self.strikes,
                self.weights,
                self.cash,
            )
            np.testing.assert_allclose(
                truncated, full[: prefix + 1], rtol=0.0, atol=0.0
            )

    def test_parallel_running_surfaces_match_scalar_paths(self):
        rng = np.random.default_rng(4)
        X = rng.beta(1.0, 4.0, 30)
        means = np.array([0.1, 0.25, 0.5])
        times = np.array([0, 3, 11, 30])
        methods = (
            ("hgkelly", {"G": 6}, cs.hgkelly_log_e_path),
            (
                "product_scale_mixture",
                self.scale_config,
                cs.product_scale_mixture_log_e_path,
            ),
            ("agrapa", {"c": 0.5}, cs.agrapa_log_e_path),
            ("bentkus_mixture", self.heat_config, None),
            ("heat_constrained_agrapa", self.heat_config, None),
        )
        for method, config, scalar in methods:
            observed = cs.running_log_e_at_times(
                X, means, times, method, method_config=config
            )
            expected = np.empty_like(observed)
            for row, m in enumerate(means):
                if method == "hgkelly":
                    path = scalar(X, m, G=6)
                elif method == "product_scale_mixture":
                    path = scalar(X, m, **self.scale_config)
                elif method == "agrapa":
                    path = scalar(X, m, c=0.5)
                elif method == "bentkus_mixture":
                    path = cs.bentkus_mixture_log_e_path(
                        X, m, **self.heat_config
                    )
                else:
                    path = cs.heat_constrained_agrapa_log_e_path(
                        X, m, **self.heat_config
                    )
                running = np.maximum.accumulate(path)
                expected[row] = running[times]
            np.testing.assert_allclose(observed, expected, rtol=0.0, atol=0.0)

    def test_exact_binary_tree_respects_ville_bound(self):
        n = 8
        m = 0.4
        delta = 0.2
        horizons, strikes, weights, cash = cs.bentkus_horizon_schedule(
            n, delta
        )
        functions = (
            lambda X: cs.hgkelly_log_e_path(X, m, G=8),
            lambda X: cs.product_scale_mixture_log_e_path(X, m),
            lambda X: cs.agrapa_log_e_path(X, m),
            lambda X: cs.bentkus_mixture_log_e_path(
                X, m, horizons, strikes, weights, cash
            ),
            lambda X: cs.heat_constrained_agrapa_log_e_path(
                X, m, horizons, strikes, weights, cash
            ),
        )
        threshold = np.log(1.0 / delta)
        for function in functions:
            crossing_probability = 0.0
            for bits in itertools.product((0.0, 1.0), repeat=n):
                X = np.asarray(bits)
                successes = int(np.sum(X))
                probability = m**successes * (1.0 - m) ** (n - successes)
                crossing_probability += probability * (
                    np.max(function(X)) >= threshold
                )
            self.assertLessEqual(crossing_probability, delta + 2e-12)


class ConfidenceSequenceInversionTests(unittest.TestCase):
    def setUp(self):
        self.X = np.tile(np.array([0.0, 1.0]), 30)
        self.delta = 0.2
        self.times = np.array([5, 10, 20, 40, 60])

    def test_convex_hull_endpoints_are_nested(self):
        for method, config in (
            ("hgkelly", {"G": 8}),
            ("product_scale_mixture", {}),
            ("agrapa", {"c": 0.5}),
        ):
            result = cs.confidence_sequence_endpoints(
                self.X,
                self.delta,
                self.times,
                method,
                method_config=config,
                topology_grid_size=41,
            )
            self.assertFalse(np.any(result["empty"]))
            self.assertTrue(np.all((0.0 <= result["lower"])))
            self.assertTrue(np.all((result["upper"] <= 1.0)))
            self.assertTrue(np.all(result["lower"] <= result["upper"]))
            self.assertTrue(np.all(np.diff(result["lower"]) >= -2e-8))
            self.assertTrue(np.all(np.diff(result["upper"]) <= 2e-8))

    def test_detected_empty_set_has_zero_width_and_nan_endpoints(self):
        X = np.r_[np.ones(20), np.zeros(20)]
        result = cs.confidence_sequence_endpoints(
            X,
            0.2,
            np.array([len(X)]),
            "product_scale_mixture",
            topology_grid_size=65,
        )
        self.assertTrue(result["empty"][0])
        self.assertEqual(result["width"][0], 0.0)
        self.assertTrue(np.isnan(result["lower"][0]))
        self.assertTrue(np.isnan(result["upper"][0]))

    def test_pairwise_summaries_mask_nonfinite_and_zero_denominators(self):
        numerator = np.array([1.0, np.nan, 0.0, np.inf, 3.0])
        denominator = np.array([2.0, 3.0, 0.0, 4.0, np.inf])
        ratio, mask = cs._finite_positive_denominator_ratio(
            numerator, denominator
        )
        np.testing.assert_array_equal(
            mask, np.array([True, False, False, False, False])
        )
        np.testing.assert_allclose(ratio, np.array([0.5]))
        self.assertEqual(
            cs._summary_or_none(np.array([])),
            {"mean": None, "median": None, "lo": None, "hi": None},
        )

    def test_refined_hgkelly_endpoints_agree_with_dense_grid(self):
        result = cs.confidence_sequence_endpoints(
            self.X,
            self.delta,
            np.array([60]),
            "hgkelly",
            method_config={"G": 8},
            topology_grid_size=33,
        )
        dense = np.linspace(0.0, 1.0, 4001)
        values = cs.running_log_e_at_times(
            self.X,
            dense,
            np.array([60]),
            "hgkelly",
            method_config={"G": 8},
        )[:, 0]
        accepted = values < np.log(1.0 / self.delta)
        indices = np.flatnonzero(accepted)
        spacing = dense[1] - dense[0]
        self.assertTrue(len(indices))
        self.assertLessEqual(
            abs(result["lower"][0] - dense[indices[0]]), spacing * 1.1
        )
        self.assertLessEqual(
            abs(result["upper"][0] - dense[indices[-1]]), spacing * 1.1
        )

    def test_running_intersection_uses_unreported_times(self):
        X = np.array([1.0] * 6 + [0.0] * 20)
        m = 0.5
        full_path = cs.agrapa_log_e_path(X, m)
        expected = np.max(full_path)
        observed = cs.running_log_e_at_times(
            X,
            np.array([m]),
            np.array([len(X)]),
            "agrapa",
        )[0, 0]
        self.assertAlmostEqual(observed, expected, places=14)

    def test_default_times_and_smoke_driver_schema(self):
        times = cs.default_cs_times(100, count=6)
        self.assertEqual(times[0], 1)
        self.assertEqual(times[-1], 100)
        self.assertTrue(np.all(np.diff(times) > 0))
        output = cs.run_confidence_sequence_smoke_experiment(
            delta=0.2,
            max_time=12,
            num_sims=1,
            seed=3,
            topology_grid_size=9,
        )
        self.assertEqual(output["max_time"], 12)
        self.assertEqual(output["num_sims"], 1)
        self.assertEqual(len(output["results"]), 6)
        for methods in output["results"].values():
            self.assertEqual(
                set(methods),
                {
                    "hgkelly",
                    "product_scale_mixture",
                    "agrapa",
                    "bentkus_mixture",
                    "heat_constrained_agrapa",
                },
            )

        benchmark = cs.run_confidence_sequence_experiment(
            delta=0.2,
            max_time=8,
            times=np.array([4, 8]),
            num_width_sims=1,
            coverage_max_time=6,
            num_coverage_sims=2,
            seed=4,
            product_grid_size=4,
            topology_grid_size=7,
        )
        self.assertEqual(benchmark["times"], [4, 8])
        self.assertEqual(benchmark["num_coverage_sims"], 2)
        self.assertIn("paired", benchmark["results"]["Beta(2,2)"])
        beta_results = benchmark["results"]["Beta(2,2)"]
        self.assertIn("product_scale_mixture", beta_results)
        for method in benchmark["methods"]:
            self.assertEqual(len(beta_results[method]["per_path_widths"]), 1)
            self.assertEqual(len(beta_results[method]["per_path_empty"]), 1)
        for comparison in beta_results["paired"].values():
            self.assertIn("terminal_finite_pair_count", comparison)
            self.assertIn("log_time_auc_finite_pair_count", comparison)
        # The promised schema is directly serializable without custom hooks.
        json.dumps(benchmark, allow_nan=False)

    def test_input_validation(self):
        with self.assertRaises(ValueError):
            cs.hgkelly_log_e_path(np.array([1.2]), 0.5)
        with self.assertRaises(ValueError):
            cs.agrapa_log_e_path(np.array([0.2]), -0.1)
        with self.assertRaises(ValueError):
            cs.product_scale_schedule(0)
        with self.assertRaises(ValueError):
            cs.product_scale_schedule(10, weight_power=1.0)
        with self.assertRaises(ValueError):
            cs.product_scale_schedule(10, horizon_overshoot=0.5)
        with self.assertRaises(ValueError):
            cs.product_scale_schedule(10, scale_ratio=1.0)
        with self.assertRaises(ValueError):
            cs.product_scale_mixture_log_e_path(
                np.array([0.2]), 0.5, fractions=np.array([0.5])
            )
        with self.assertRaises(ValueError):
            cs.product_scale_mixture_log_e_path(
                np.array([0.2]),
                0.5,
                np.array([0.25, 0.5]),
                np.array([0.4, 0.3]),
                0.3,
            )
        with self.assertRaises(ValueError):
            cs.running_log_e_at_times(
                np.array([0.2]),
                np.array([0.5]),
                np.array([1]),
                "product_scale_mixture",
                method_config={"fractions": np.array([0.5])},
            )
        with self.assertRaises(ValueError):
            cs.bentkus_horizon_schedule(10, 0.1, weight_power=1.0)
        with self.assertRaises(ValueError):
            cs.running_log_e_at_times(
                np.array([0.2]),
                np.array([0.5]),
                np.array([1]),
                "not_a_method",
            )


if __name__ == "__main__":
    unittest.main()
