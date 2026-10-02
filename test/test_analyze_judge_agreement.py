import tempfile
import unittest
from pathlib import Path

import numpy as np
from openpyxl import Workbook

from eval.src.analyze_judge_agreement import (
    intraclass_correlations,
    kendalls_w,
    load_score_workbook,
    rankdata,
)


class AgreementMetricTests(unittest.TestCase):
    def test_identical_variable_scores_have_perfect_icc(self):
        scores = np.column_stack(([1.0, 2.0, 4.0, 7.0], [1.0, 2.0, 4.0, 7.0]))
        absolute, consistency = intraclass_correlations(scores)
        self.assertAlmostEqual(absolute, 1.0)
        self.assertAlmostEqual(consistency, 1.0)

    def test_constant_severity_shift_separates_absolute_and_consistency_icc(self):
        scores = np.column_stack(([1.0, 2.0, 4.0, 7.0], [11.0, 12.0, 14.0, 17.0]))
        absolute, consistency = intraclass_correlations(scores)
        self.assertLess(absolute, 0.5)
        self.assertAlmostEqual(consistency, 1.0)

    def test_rankdata_uses_average_rank_for_ties(self):
        np.testing.assert_allclose(rankdata([30, 10, 10, 20]), [4.0, 1.5, 1.5, 3.0])

    def test_identical_rankings_have_perfect_kendall_w(self):
        scores = np.column_stack(([1.0, 2.0, 2.0, 5.0], [10.0, 20.0, 20.0, 50.0]))
        self.assertAlmostEqual(kendalls_w(scores), 1.0)


class WorkbookLoadingTests(unittest.TestCase):
    def test_loader_rejects_duplicate_keys(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "scores.xlsx"
            workbook = Workbook()
            sheet = workbook.active
            sheet.title = "Task_Stage_Scores"
            sheet.append(["Model Name", "Task ID", "score"])
            sheet.append(["model", "task", 1])
            sheet.append(["model", "task", 2])
            workbook.save(path)

            with self.assertRaisesRegex(ValueError, "重复 key"):
                load_score_workbook(
                    path,
                    "Task_Stage_Scores",
                    ["Model Name", "Task ID"],
                    ["score"],
                )

    def test_loader_reads_valid_rows(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "scores.xlsx"
            workbook = Workbook()
            sheet = workbook.active
            sheet.title = "Task_Stage_Scores"
            sheet.append(["Model Name", "Task ID", "score"])
            sheet.append(["model", "task", 7.5])
            workbook.save(path)

            rows = load_score_workbook(
                path,
                "Task_Stage_Scores",
                ["Model Name", "Task ID"],
                ["score"],
            )
            self.assertEqual(rows[("model", "task")]["score"], 7.5)


if __name__ == "__main__":
    unittest.main()
