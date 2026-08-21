import math
import unittest

import numpy as np

from experiments import survival_fixed_event as survival


class FixedEventSurvivalTests(unittest.TestCase):
    def test_risk_set_sequence_without_censoring(self):
        treatment = np.array([0, 1, 0, 1], dtype=np.int8)
        event_times = np.array([1.0, 2.0, 3.0, 4.0])
        censor_times = np.full(4, np.inf)
        (
            risk_control,
            risk_treatment,
            failing_arm,
            terminal_time,
            censor_fraction,
        ) = survival._risk_set_sequence(
            treatment,
            event_times,
            censor_times,
            3,
        )
        np.testing.assert_array_equal(risk_control, [2, 1, 1])
        np.testing.assert_array_equal(risk_treatment, [2, 2, 1])
        np.testing.assert_array_equal(failing_arm, [0, 1, 0])
        self.assertEqual(terminal_time, 3.0)
        self.assertEqual(censor_fraction, 0.0)

    def test_piecewise_exponential_inverse(self):
        early = 0.5
        late = 2.0
        change = 1.0
        self.assertAlmostEqual(
            survival._piecewise_exponential_time(0.25, early, late, change),
            0.5,
        )
        self.assertAlmostEqual(
            survival._piecewise_exponential_time(1.5, early, late, change),
            1.5,
        )

    def test_ge_one_step_has_null_mean_one(self):
        risk_control = np.array([3], dtype=np.int64)
        risk_treatment = np.array([2], dtype=np.int64)
        candidate = 0.2
        probability = survival._risk_probability(3, 2, candidate)
        upper_zero, lower_zero = survival._ge_wealths(
            risk_control,
            risk_treatment,
            np.array([0], dtype=np.int8),
            candidate,
            0.05,
            0.5,
        )
        upper_one, lower_one = survival._ge_wealths(
            risk_control,
            risk_treatment,
            np.array([1], dtype=np.int8),
            candidate,
            0.05,
            0.5,
        )
        self.assertAlmostEqual(
            (1.0 - probability) * upper_zero + probability * upper_one,
            1.0,
            places=12,
        )
        self.assertAlmostEqual(
            (1.0 - probability) * lower_zero + probability * lower_one,
            1.0,
            places=12,
        )

    def test_grow_one_step_is_an_e_variable(self):
        risk_control = np.array([7], dtype=np.int64)
        risk_treatment = np.array([5], dtype=np.int64)
        candidate = -0.1
        offset = abs(math.log(0.7))
        probability = survival._risk_probability(7, 5, candidate)
        scores = []
        for outcome in (0, 1):
            score = survival._av_grow_score(
                risk_control,
                risk_treatment,
                np.array([outcome], dtype=np.int8),
                candidate,
                0.05,
                offset,
            )
            log_e_value = score + math.log(1.0 / 0.05)
            scores.append(math.exp(log_e_value))
        expectation = (1.0 - probability) * scores[0] + probability * scores[1]
        self.assertAlmostEqual(expectation, 1.0, places=12)

    def test_conditional_grow_multiplies_eventwise_mixtures(self):
        risk_control = np.array([10, 9], dtype=np.int64)
        risk_treatment = np.array([10, 10], dtype=np.int64)
        failing_arm = np.array([1, 1], dtype=np.int8)
        candidate = 0.0
        offset = abs(math.log(0.7))
        observed = survival._av_grow_score(
            risk_control,
            risk_treatment,
            failing_arm,
            candidate,
            0.05,
            offset,
        ) + math.log(1.0 / 0.05)
        log_expected = 0.0
        for event_index in range(2):
            ratios = []
            for alternative in (-offset, offset):
                null_probability = survival._risk_probability(
                    risk_control[event_index],
                    risk_treatment[event_index],
                    candidate,
                )
                alternative_probability = survival._risk_probability(
                    risk_control[event_index],
                    risk_treatment[event_index],
                    alternative,
                )
                ratio = (
                    survival._bernoulli_log_probability(
                        failing_arm[event_index],
                        alternative_probability,
                    )
                    - survival._bernoulli_log_probability(
                        failing_arm[event_index],
                        null_probability,
                    )
                )
                ratios.append(math.exp(ratio))
            log_expected += math.log(0.5 * ratios[0] + 0.5 * ratios[1])
        self.assertAlmostEqual(observed, log_expected, places=12)

    def test_path_mixture_mixes_full_likelihood_ratios(self):
        risk_control = np.array([10, 9], dtype=np.int64)
        risk_treatment = np.array([10, 10], dtype=np.int64)
        failing_arm = np.array([1, 1], dtype=np.int8)
        candidate = 0.0
        offset = abs(math.log(0.7))
        observed = survival._av_path_mixture_score(
            risk_control,
            risk_treatment,
            failing_arm,
            candidate,
            0.05,
            offset,
        ) + math.log(1.0 / 0.05)
        ratios = []
        for alternative in (-offset, offset):
            ratio = 0.0
            for event_index in range(2):
                null_probability = survival._risk_probability(
                    risk_control[event_index],
                    risk_treatment[event_index],
                    candidate,
                )
                alternative_probability = survival._risk_probability(
                    risk_control[event_index],
                    risk_treatment[event_index],
                    alternative,
                )
                ratio += (
                    survival._bernoulli_log_probability(
                        failing_arm[event_index],
                        alternative_probability,
                    )
                    - survival._bernoulli_log_probability(
                        failing_arm[event_index],
                        null_probability,
                    )
                )
            ratios.append(math.exp(ratio))
        expected = math.log(0.5 * ratios[0] + 0.5 * ratios[1])
        self.assertAlmostEqual(observed, expected, places=12)

    def test_small_trial_produces_ordered_intervals(self):
        rng = np.random.default_rng(19)
        hazard = survival.HAZARD_SCENARIOS[0]
        censor = survival.CENSOR_SCENARIOS[0]
        result = survival._run_one_scenario(
            rng,
            repetitions=4,
            subject_count=90,
            event_horizon=30,
            hazard=hazard,
            censor=censor,
            delta=0.05,
            normal_cutoff=1.959963984540054,
            solvency_constant=0.5,
            grow_offset=abs(math.log(0.7)),
            candidate_bound=3.0,
            grid_size=25,
        )
        lower, upper = result[0], result[1]
        self.assertTrue(np.all(np.isfinite(lower)))
        self.assertTrue(np.all(np.isfinite(upper)))
        self.assertTrue(np.all(lower < upper))


if __name__ == "__main__":
    unittest.main()
