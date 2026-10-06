from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from analyze_per_report_subtask_stage import (  # noqa: E402
    build_distribution_summary,
    calculate_report_metrics,
    extract_cell_scores,
    load_subtasks,
    write_analysis_outputs,
)


class ExtractCellScoresTest(unittest.TestCase):
    def test_normalizes_each_subtask_stage_to_ten_point_scale(self) -> None:
        raw = {
            "1": {
                "问题识别": [{"score": 35}, {"score": 30}, {"score": 25}],
                "模型构建": [{"score": 60}, {"score": 20}],
            },
            "2": {
                "问题识别": [{"score": 10}, {"score": 40}],
                "模型构建": [{"score": 25}, {"score": 25}],
            },
        }

        cells = extract_cell_scores(
            raw,
            subtask_ids=["1", "2"],
            stages=["问题识别", "模型构建"],
        )

        self.assertEqual(
            cells,
            {
                ("1", "问题识别"): 9.0,
                ("1", "模型构建"): 8.0,
                ("2", "问题识别"): 5.0,
                ("2", "模型构建"): 5.0,
            },
        )

    def test_rejects_a_missing_subtask_stage_cell(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing cell: subtask=1, stage=模型构建"):
            extract_cell_scores(
                {"1": {"问题识别": [{"score": 100}]}},
                subtask_ids=["1"],
                stages=["问题识别", "模型构建"],
            )


class LoadSubtasksTest(unittest.TestCase):
    def test_accepts_legacy_root_and_wrapped_result_layouts(self) -> None:
        legacy = {"1": {"问题识别": [{"score": 100}]}}
        wrapped = {"status": "completed", "subtasks": legacy}
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            legacy_path = root / "legacy.json"
            wrapped_path = root / "wrapped.json"
            legacy_path.write_text(json.dumps(legacy, ensure_ascii=False), encoding="utf-8")
            wrapped_path.write_text(json.dumps(wrapped, ensure_ascii=False), encoding="utf-8")

            self.assertEqual(load_subtasks(legacy_path), legacy)
            self.assertEqual(load_subtasks(wrapped_path), legacy)


class CalculateReportMetricsTest(unittest.TestCase):
    def test_perfectly_identical_judges_have_unit_icc_and_zero_error(self) -> None:
        matrix = np.asarray(
            [
                [1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0],
                [4.0, 4.0, 4.0],
                [8.0, 8.0, 8.0],
            ]
        )

        metrics = calculate_report_metrics("synthetic", matrix, ["gpt", "gemini", "deepseek"])

        self.assertAlmostEqual(metrics["icc_2_1"], 1.0)
        self.assertAlmostEqual(metrics["icc_c_1"], 1.0)
        self.assertAlmostEqual(metrics["mean_pairwise_mae"], 0.0)
        self.assertAlmostEqual(metrics["max_cell_range"], 0.0)
        self.assertAlmostEqual(metrics["target_mean_sd"], math.sqrt(115 / 12))
        self.assertAlmostEqual(metrics["target_mean_range"], 7.0)
        self.assertEqual(metrics["agreement_level"], "excellent")

    def test_absolute_icc_penalizes_systematic_judge_bias(self) -> None:
        matrix = np.asarray(
            [
                [1.0, 2.0, 1.0],
                [3.0, 4.0, 3.0],
                [5.0, 6.0, 5.0],
                [7.0, 8.0, 7.0],
            ]
        )

        metrics = calculate_report_metrics("biased", matrix, ["gpt", "gemini", "deepseek"])

        self.assertLess(metrics["icc_2_1"], metrics["icc_c_1"])
        self.assertAlmostEqual(metrics["gpt_vs_gemini_mae"], 1.0)
        self.assertAlmostEqual(metrics["gpt_vs_deepseek_mae"], 0.0)
        self.assertAlmostEqual(metrics["gemini_vs_deepseek_mae"], 1.0)
        self.assertAlmostEqual(metrics["max_cell_range"], 1.0)


class DistributionSummaryTest(unittest.TestCase):
    def test_summarizes_report_iccs_and_uses_documented_bins(self) -> None:
        rows = [
            {"task_id": "A", "icc_2_1": 0.20, "mean_pairwise_mae": 0.10, "max_cell_range": 1.0},
            {"task_id": "B", "icc_2_1": 0.50, "mean_pairwise_mae": 0.20, "max_cell_range": 2.0},
            {"task_id": "C", "icc_2_1": 0.75, "mean_pairwise_mae": 0.30, "max_cell_range": 3.0},
            {"task_id": "D", "icc_2_1": 0.90, "mean_pairwise_mae": 0.40, "max_cell_range": 4.0},
            {"task_id": "E", "icc_2_1": 1.00, "mean_pairwise_mae": 0.50, "max_cell_range": 5.0},
            {"task_id": "F", "icc_2_1": math.nan, "mean_pairwise_mae": 0.60, "max_cell_range": 6.0},
        ]

        summary = build_distribution_summary(rows)

        self.assertEqual(summary["report_count"], 6)
        self.assertEqual(summary["finite_icc_count"], 5)
        self.assertEqual(summary["undefined_icc_count"], 1)
        self.assertAlmostEqual(summary["median"], 0.75)
        self.assertAlmostEqual(summary["minimum"], 0.20)
        self.assertAlmostEqual(summary["maximum"], 1.00)
        self.assertEqual(
            summary["bins"],
            {
                "<0.50": 1,
                "0.50-<0.75": 1,
                "0.75-<0.90": 1,
                ">=0.90": 2,
            },
        )
        self.assertEqual(summary["lowest_reports"][0]["task_id"], "A")
        self.assertAlmostEqual(summary["mean_pairwise_mae"]["median"], 0.35)
        self.assertAlmostEqual(summary["mean_pairwise_mae"]["q1"], 0.225)
        self.assertAlmostEqual(summary["max_cell_range"]["maximum"], 6.0)

    def test_writes_machine_readable_report_and_distribution_plot(self) -> None:
        rows = [
            {
                "task_id": "A",
                "cell_count": 28,
                "icc_2_1": 0.80,
                "icc_c_1": 0.90,
                "agreement_level": "good",
                "mean_pairwise_mae": 0.40,
                "max_pairwise_mae": 0.50,
                "max_cell_range": 1.25,
                "target_mean_sd": 2.0,
                "target_mean_range": 6.0,
                "gpt_vs_gemini_mae": 0.30,
                "gpt_vs_deepseek_mae": 0.40,
                "gemini_vs_deepseek_mae": 0.50,
            },
            {
                "task_id": "B",
                "cell_count": 28,
                "icc_2_1": 0.95,
                "icc_c_1": 0.97,
                "agreement_level": "excellent",
                "mean_pairwise_mae": 0.20,
                "max_pairwise_mae": 0.30,
                "max_cell_range": 0.75,
                "target_mean_sd": 2.5,
                "target_mean_range": 7.0,
                "gpt_vs_gemini_mae": 0.10,
                "gpt_vs_deepseek_mae": 0.20,
                "gemini_vs_deepseek_mae": 0.30,
            },
        ]
        summary = build_distribution_summary(rows)
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)

            write_analysis_outputs(rows, summary, output_dir)

            csv_text = (output_dir / "per_report_metrics.csv").read_text(encoding="utf-8-sig")
            markdown = (output_dir / "report.md").read_text(encoding="utf-8")
            png = (output_dir / "icc_distribution.png").read_bytes()
            saved_summary = json.loads(
                (output_dir / "distribution_summary.json").read_text(encoding="utf-8")
            )
            self.assertIn("task_id,cell_count,icc_2_1", csv_text)
            self.assertIn("A,28,0.8", csv_text)
            self.assertIn("Per-report subtask-by-stage agreement", markdown)
            self.assertEqual(png[:8], b"\x89PNG\r\n\x1a\n")
            self.assertAlmostEqual(saved_summary["median"], 0.875)


if __name__ == "__main__":
    unittest.main()
