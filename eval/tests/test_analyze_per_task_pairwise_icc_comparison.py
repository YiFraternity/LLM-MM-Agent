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

from analyze_per_task_pairwise_icc_comparison import (  # noqa: E402
    build_overlap_rows,
    calculate_pairwise_rows,
    load_static_task_matrices,
    summarize_distributions,
    write_outputs,
)


class CalculatePairwiseRowsTest(unittest.TestCase):
    def test_computes_all_three_pairwise_iccs_within_one_task(self) -> None:
        matrix = np.asarray(
            [
                [1.0, 1.0, 2.0],
                [3.0, 3.0, 4.0],
                [5.0, 5.0, 6.0],
                [7.0, 7.0, 8.0],
            ]
        )

        rows = calculate_pairwise_rows(
            method="proposed",
            task_id="T1",
            report_model="report-model",
            score_matrix=matrix,
            judge_labels=["gpt", "gemini", "deepseek"],
        )

        by_pair = {row["judge_pair"]: row for row in rows}
        self.assertEqual(set(by_pair), {"gpt_vs_gemini", "gpt_vs_deepseek", "gemini_vs_deepseek"})
        self.assertAlmostEqual(by_pair["gpt_vs_gemini"]["icc_2_1"], 1.0)
        self.assertAlmostEqual(by_pair["gpt_vs_gemini"]["mae"], 0.0)
        self.assertLess(by_pair["gpt_vs_deepseek"]["icc_2_1"], 1.0)
        self.assertAlmostEqual(by_pair["gpt_vs_deepseek"]["mae"], 1.0)
        self.assertEqual(by_pair["gpt_vs_deepseek"]["target_count"], 4)


class LoadStaticTaskMatricesTest(unittest.TestCase):
    def test_loads_eight_atomic_static_scores_in_declared_order(self) -> None:
        cell_names = [
            ("problem_analysis", "problem_definition_and_goals"),
            ("problem_analysis", "scope_and_coverage"),
            ("modeling_rigor", "assumptions"),
            ("modeling_rigor", "model_rationality"),
            ("practicality_scientificity", "practicality"),
            ("practicality_scientificity", "scientificity"),
            ("result_bias", "result_analysis"),
            ("result_bias", "bias_analysis"),
        ]
        with tempfile.TemporaryDirectory() as temporary_directory:
            experiment_dir = Path(temporary_directory)
            manifest = {
                "samples": [{"task_id": "2015_A", "report_model": "model-a"}]
            }
            (experiment_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            judges = ["gpt-5-mini", "gemini-3.1-p", "ali-deepseek-v4-pro"]
            columns = [range(1, 9), range(2, 10), range(3, 11)]
            for judge, values in zip(judges, columns):
                payload: dict[str, dict[str, dict[str, int]]] = {}
                for (stage, item), value in zip(cell_names, values):
                    payload.setdefault(stage, {})[item] = {"score": value}
                path = experiment_dir / "runs" / judge / "results" / "2015_A" / "model-a.json"
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps(payload), encoding="utf-8")

            loaded = load_static_task_matrices(experiment_dir)

        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0]["task_id"], "2015_A")
        self.assertEqual(loaded[0]["report_model"], "model-a")
        np.testing.assert_array_equal(
            loaded[0]["score_matrix"],
            np.asarray(
                [
                    [1, 2, 3],
                    [2, 3, 4],
                    [3, 4, 5],
                    [4, 5, 6],
                    [5, 6, 7],
                    [6, 7, 8],
                    [7, 8, 9],
                    [8, 9, 10],
                ],
                dtype=float,
            ),
        )


class DistributionAndOverlapTest(unittest.TestCase):
    def test_summarizes_each_method_and_judge_pair_separately(self) -> None:
        rows = [
            {"method": "proposed", "judge_pair": "gpt_vs_gemini", "icc_2_1": 0.8},
            {"method": "proposed", "judge_pair": "gpt_vs_gemini", "icc_2_1": 1.0},
            {"method": "static", "judge_pair": "gpt_vs_gemini", "icc_2_1": 0.1},
            {"method": "static", "judge_pair": "gpt_vs_gemini", "icc_2_1": math.nan},
        ]

        summaries = summarize_distributions(rows)
        indexed = {(row["method"], row["judge_pair"]): row for row in summaries}

        proposed = indexed[("proposed", "gpt_vs_gemini")]
        static = indexed[("static", "gpt_vs_gemini")]
        self.assertEqual(proposed["task_count"], 2)
        self.assertAlmostEqual(proposed["median"], 0.9)
        self.assertEqual(proposed["at_least_good_count"], 2)
        self.assertEqual(static["task_count"], 2)
        self.assertEqual(static["finite_count"], 1)
        self.assertEqual(static["undefined_count"], 1)
        self.assertAlmostEqual(static["median"], 0.1)

    def test_overlap_rows_include_only_shared_task_and_pair_keys(self) -> None:
        rows = [
            {
                "method": "proposed",
                "task_id": "A",
                "report_model": "proposed-report",
                "target_count": 28,
                "judge_pair": "gpt_vs_gemini",
                "icc_2_1": 0.9,
            },
            {
                "method": "static",
                "task_id": "A",
                "report_model": "static-report",
                "target_count": 8,
                "judge_pair": "gpt_vs_gemini",
                "icc_2_1": 0.2,
            },
            {
                "method": "proposed",
                "task_id": "B",
                "report_model": "proposed-report",
                "target_count": 28,
                "judge_pair": "gpt_vs_gemini",
                "icc_2_1": 0.8,
            },
        ]

        overlap = build_overlap_rows(rows)

        self.assertEqual(len(overlap), 1)
        self.assertEqual(overlap[0]["task_id"], "A")
        self.assertAlmostEqual(overlap[0]["icc_delta_proposed_minus_static"], 0.7)
        self.assertEqual(overlap[0]["proposed_target_count"], 28)
        self.assertEqual(overlap[0]["static_target_count"], 8)
        self.assertEqual(overlap[0]["proposed_report_model"], "proposed-report")
        self.assertEqual(overlap[0]["static_report_model"], "static-report")

    def test_writes_pairwise_tables_report_and_plot(self) -> None:
        rows = [
            {
                "method": "proposed",
                "task_id": "A",
                "report_model": "proposed-report",
                "target_count": 28,
                "judge_pair": "gpt_vs_gemini",
                "icc_2_1": 0.9,
                "icc_c_1": 0.92,
                "agreement_level": "excellent",
                "mae": 0.5,
                "mean_bias": 0.1,
            },
            {
                "method": "static",
                "task_id": "A",
                "report_model": "static-report",
                "target_count": 8,
                "judge_pair": "gpt_vs_gemini",
                "icc_2_1": 0.2,
                "icc_c_1": 0.3,
                "agreement_level": "poor",
                "mae": 2.0,
                "mean_bias": 1.5,
            },
        ]
        summaries = summarize_distributions(rows)
        overlap = build_overlap_rows(rows)
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)

            write_outputs(rows, summaries, overlap, output_dir)

            long_csv = (output_dir / "per_task_pairwise_icc.csv").read_text(encoding="utf-8-sig")
            wide_csv = (output_dir / "per_task_pairwise_icc_wide.csv").read_text(encoding="utf-8-sig")
            overlap_csv = (output_dir / "overlap_pairwise_comparison.csv").read_text(encoding="utf-8-sig")
            report = (output_dir / "report.md").read_text(encoding="utf-8")
            png = (output_dir / "pairwise_icc_distributions.png").read_bytes()
            self.assertIn("method,task_id,report_model,target_count,judge_pair,icc_2_1", long_csv)
            self.assertIn(
                "method,task_id,report_model,target_count,gpt_vs_gemini,gpt_vs_deepseek,gemini_vs_deepseek",
                wide_csv,
            )
            self.assertIn("proposed,A,proposed-report,28,0.9", wide_csv)
            self.assertIn("A,gpt_vs_gemini", overlap_csv)
            self.assertIn("Per-task pairwise ICC(2,1) comparison", report)
            self.assertEqual(png[:8], b"\x89PNG\r\n\x1a\n")


if __name__ == "__main__":
    unittest.main()
