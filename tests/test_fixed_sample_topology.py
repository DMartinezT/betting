import unittest

import numpy as np

import augment_fixed_sample_topology as topology
import betting


class FixedSampleMethodDispatchTests(unittest.TestCase):
    def test_explicit_dispatch_matches_direct_arm_scores(self):
        x = np.asarray([0.08, 0.61, 0.27, 0.93, 0.44])
        means = np.asarray([0.2, 0.5, 0.8])
        delta = 0.01
        u_plus, u_minus = 0.37, 0.71
        strike, initial_wealth = betting.get_optimal_lambda(delta / 2.0)

        cases = (
            (
                "heat_original",
                lambda mean: (
                    betting.compute_M_heat_path_arms(
                        x, mean, strike, initial_wealth
                    ),
                    initial_wealth,
                ),
            ),
            (
                "product_original",
                lambda mean: (
                    betting.compute_M_inf_arms(x, mean, delta),
                    1.0,
                ),
            ),
        )
        alpha = delta / 2.0
        for method, direct in cases:
            with self.subTest(method=method):
                observed = topology._scores(
                    x,
                    means,
                    delta,
                    strike,
                    initial_wealth,
                    topology.SCORE_KIND_BY_METHOD[method],
                    u_plus,
                    u_minus,
                )
                expected = []
                for mean in means:
                    (plus, minus), scale = direct(mean)
                    expected.append(max(
                        alpha * plus / (scale * u_plus),
                        alpha * minus / (scale * u_minus),
                    ))
                np.testing.assert_allclose(observed, expected, rtol=1e-13)


if __name__ == "__main__":
    unittest.main()
