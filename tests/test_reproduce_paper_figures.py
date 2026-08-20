import tempfile
import unittest
from pathlib import Path

import reproduce_paper_figures as reproduction


class PaperFigureReproductionTests(unittest.TestCase):
    def test_manifest_is_unique_and_complete(self):
        names = [figure.filename for figure in reproduction.FIGURES]
        self.assertEqual(len(names), 11)
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(set(reproduction.GENERATORS), {
            figure.group for figure in reproduction.FIGURES
        })

    def test_all_cached_inputs_exist(self):
        reproduction.validate_inputs(tuple(reproduction.GENERATORS))

    def test_single_analytic_figure_can_be_generated(self):
        with tempfile.TemporaryDirectory() as temporary:
            outputs = reproduction.reproduce(
                Path(temporary),
                ["efficient_betting_function.pdf"],
            )
            self.assertEqual(len(outputs), 1)
            self.assertGreater(outputs[0].stat().st_size, 1_000)
            self.assertEqual(outputs[0].read_bytes()[:4], b"%PDF")


if __name__ == "__main__":
    unittest.main()
