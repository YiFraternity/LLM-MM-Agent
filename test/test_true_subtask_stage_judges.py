import unittest

from eval.src.run_true_subtask_stage_judges import (
    STAGES,
    compute_report_scores,
    validate_response,
)


class TrueSubtaskStageJudgeTests(unittest.TestCase):
    def test_validation_accepts_nfkc_equivalent_dimension_names(self):
        subtask = {
            "criteria": {
                "evaluation_criteria": {
                    stage: [] for stage in STAGES
                }
            }
        }
        subtask["criteria"]["evaluation_criteria"]["问题复述"] = [
            {"sub_criteria": "口径(A vs B)", "score": 100}
        ]
        response = {stage: [] for stage in STAGES}
        response["问题复述"] = [
            {
                "dimension": "口径（A vs B）",
                "comment": "原文给出了定义。",
                "score": 80,
            }
        ]
        validated = validate_response(response, subtask)
        self.assertEqual(validated["问题复述"][0]["score"], 80.0)

    def test_validation_rejects_score_above_maximum(self):
        subtask = {
            "criteria": {
                "evaluation_criteria": {
                    stage: [] for stage in STAGES
                }
            }
        }
        subtask["criteria"]["evaluation_criteria"]["问题识别"] = [
            {"sub_criteria": "目标", "score": 30}
        ]
        response = {stage: [] for stage in STAGES}
        response["问题识别"] = [
            {"dimension": "目标", "comment": "证据", "score": 31}
        ]
        with self.assertRaisesRegex(ValueError, "score out of range"):
            validate_response(response, subtask)

    def test_report_score_averages_subtasks_and_stages(self):
        results = {
            "1": {stage: [{"score": 50.0}] for stage in STAGES},
            "2": {stage: [{"score": 100.0}] for stage in STAGES},
        }
        stages, overall = compute_report_scores(results)
        for stage in STAGES:
            self.assertAlmostEqual(stages[stage], 7.5)
        self.assertAlmostEqual(overall, 7.5)


if __name__ == "__main__":
    unittest.main()
