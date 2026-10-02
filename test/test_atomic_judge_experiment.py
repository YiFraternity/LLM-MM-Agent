import unittest

from eval.src.run_atomic_judge_experiment import (
    compute_scores,
    flatten_criteria,
    validate_evaluations,
)


class AtomicJudgeTests(unittest.TestCase):
    def test_flatten_and_deterministic_score(self):
        subtask = {
            "criteria": {
                "evaluation_criteria": {
                    "问题识别": [
                        {"sub_criteria": "A", "score": 60},
                        {"sub_criteria": "B", "score": 40},
                    ],
                    "结果分析": [
                        {"sub_criteria": "C", "score": 100},
                    ],
                }
            }
        }
        criteria = flatten_criteria(subtask)
        results = {
            "1": {
                "evaluations": [
                    {"criterion_id": "问题识别::1", "level": "FULL"},
                    {"criterion_id": "问题识别::2", "level": "PARTIAL"},
                    {"criterion_id": "结果分析::1", "level": "ALMOST"},
                ]
            }
        }
        stage_scores, overall = compute_scores(results, {"1": criteria})
        self.assertAlmostEqual(stage_scores["problem_analysis"], 8.0)
        self.assertAlmostEqual(stage_scores["result_bias"], 8.0)
        self.assertAlmostEqual(overall, 8.0)

    def test_validation_restores_criteria_order(self):
        criteria = [
            {"criterion_id": "A"},
            {"criterion_id": "B"},
        ]
        response = {
            "evaluations": [
                {"criterion_id": "B", "level": "partial", "evidence": "b", "reason": "b"},
                {"criterion_id": "A", "level": "full", "evidence": "a", "reason": "a"},
            ]
        }
        validated = validate_evaluations(response, criteria)
        self.assertEqual([item["criterion_id"] for item in validated], ["A", "B"])
        self.assertEqual([item["level"] for item in validated], ["FULL", "PARTIAL"])

    def test_validation_rejects_missing_item(self):
        with self.assertRaisesRegex(ValueError, "criterion ids mismatch"):
            validate_evaluations(
                {"evaluations": []},
                [{"criterion_id": "A"}],
            )


if __name__ == "__main__":
    unittest.main()
