import unittest

import numpy as np

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

    def test_scaled_width_matches_bentkus_limit(self):
        delta = 0.01
        strike, initial_wealth = betting.get_optimal_lambda(delta / 2.0)
        X = np.tile(np.array([0.0, 1.0]), 2500)
        lo, hi = betting.heat_ci_endpoints(
            X,
            delta,
            strike,
            initial_wealth,
            B=5,
            rng=np.random.default_rng(123),
        )
        observed = np.sqrt(len(X)) * (hi - lo)
        target = betting.asymptotic_limit_mt2(delta)
        self.assertAlmostEqual(observed, target, delta=0.01)


if __name__ == "__main__":
    unittest.main()
