from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from run_bestpaper_judges import (  # noqa: E402
    STAGES,
    build_subtask_inputs,
    clean_bestpaper_tex,
    evaluate_report,
    freeze_inputs,
    load_judge_configs,
    select_bestpapers,
    write_summaries,
)


class SelectBestpapersTest(unittest.TestCase):
    def test_selects_one_paper_per_task_deterministically_and_records_exclusions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            bestpaper_root = root / "BestPaper"
            criteria_root = root / "criteria"
            for task_id, paper_names in {
                "2015_A": ["1.tex", "2.tex"],
                "2015_B": ["1.tex", "2.tex"],
                "2015_C": ["1.tex"],
            }.items():
                task_dir = bestpaper_root / task_id
                task_dir.mkdir(parents=True)
                for paper_name in paper_names:
                    (task_dir / paper_name).write_text(task_id + paper_name, encoding="utf-8")
            criteria_root.mkdir()
            for task_id in ("2015_A", "2015_B"):
                (criteria_root / f"{task_id}.json").write_text(
                    json.dumps({"subtask": {"1": {}}}), encoding="utf-8"
                )

            first, exclusions = select_bestpapers(
                bestpaper_root=bestpaper_root,
                criteria_root=criteria_root,
                seed=20261006,
            )
            second, _ = select_bestpapers(
                bestpaper_root=bestpaper_root,
                criteria_root=criteria_root,
                seed=20261006,
            )

        self.assertEqual(first, second)
        self.assertEqual([row["task_id"] for row in first], ["2015_A", "2015_B"])
        self.assertEqual(len({row["task_id"] for row in first}), 2)
        self.assertTrue(all(row["paper_number"] in {1, 2} for row in first))
        self.assertEqual(
            exclusions,
            [
                {
                    "task_id": "2015_C",
                    "reason": "missing task-specific criteria",
                    "candidate_count": 1,
                }
            ],
        )


class CleanBestpaperTexTest(unittest.TestCase):
    def test_removes_conversion_noise_but_preserves_paper_evidence(self) -> None:
        source = (
            "\\tableofcontents问\\tableofcontents题\\tableofcontents一\n"
            "% conversion comment\n"
            "\\section{模型建立}\n证据正文\n"
        )

        cleaned = clean_bestpaper_tex(source)

        self.assertNotIn("\\tableofcontents", cleaned)
        self.assertNotIn("conversion comment", cleaned)
        self.assertIn("问题一", cleaned)
        self.assertIn("\\section{模型建立}", cleaned)
        self.assertIn("证据正文", cleaned)


class BuildSubtaskInputsTest(unittest.TestCase):
    def test_supplies_complete_cleaned_paper_to_every_declared_subtask(self) -> None:
        criteria = {
            "subtask": {
                "2": {"subtask": "second", "criteria": {"evaluation_criteria": {}}},
                "1": {"subtask": "first", "criteria": {"evaluation_criteria": {}}},
            }
        }

        inputs = build_subtask_inputs("complete paper", criteria)

        self.assertEqual([row["subtask_id"] for row in inputs], ["1", "2"])
        self.assertEqual([row["subproblem"] for row in inputs], ["first", "second"])
        self.assertTrue(all(row["report_content"] == "complete paper" for row in inputs))

    def test_rejects_criteria_without_subtasks(self) -> None:
        with self.assertRaisesRegex(ValueError, "subtask"):
            build_subtask_inputs("paper", {"subtask": {}})


class FreezeInputsTest(unittest.TestCase):
    def test_freezes_cleaned_reports_criteria_prompt_and_exclusions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            bestpaper_root = root / "BestPaper"
            criteria_root = root / "criteria"
            output_dir = root / "experiment"
            prompt_path = root / "prompt.yaml"
            prompt_path.write_text("math_modeling_report_eval: {}\n", encoding="utf-8")
            for task_id in ("2015_A", "2015_B"):
                task_dir = bestpaper_root / task_id
                task_dir.mkdir(parents=True)
                (task_dir / "1.tex").write_text(
                    rf"\tableofcontents {task_id} evidence", encoding="utf-8"
                )
            criteria_root.mkdir()
            (criteria_root / "2015_A.json").write_text(
                json.dumps({"subtask": {"1": {}}}), encoding="utf-8"
            )

            manifest = freeze_inputs(
                bestpaper_root=bestpaper_root,
                criteria_root=criteria_root,
                prompt_path=prompt_path,
                output_dir=output_dir,
                seed=20261006,
                public_judges=[{"config_name": "judge", "model_name": "model"}],
                git_commit="abc123",
            )

            frozen_report = output_dir / manifest["reports"][0]["frozen_report"]
            frozen_criteria = output_dir / manifest["reports"][0]["frozen_criteria"]
            self.assertEqual(len(manifest["reports"]), 1)
            self.assertEqual(manifest["excluded_tasks"][0]["task_id"], "2015_B")
            self.assertNotIn("\\tableofcontents", frozen_report.read_text(encoding="utf-8"))
            self.assertTrue(frozen_criteria.is_file())
            self.assertEqual(manifest["configuration"]["seed"], 20261006)
            self.assertEqual(manifest["configuration"]["git_commit"], "abc123")


class EvaluateReportTest(unittest.TestCase):
    def test_scores_each_subtask_and_reuses_completed_result(self) -> None:
        class FakeJudge:
            calls_made = 0

            def __init__(self, **kwargs):
                self.calls = []

            def generate(self, prompt: str, system: str = "") -> str:
                del prompt, system
                type(self).calls_made += 1
                response = {
                    stage: [
                        {
                            "dimension": f"{stage}-criterion",
                            "score": 5,
                            "comment": "evidence",
                        }
                    ]
                    for stage in STAGES
                }
                self.calls.append({"status": "completed"})
                return json.dumps(response)

            def get_total_usage(self):
                return {"total_tokens": 10}

        criteria = {
            "subtask": {
                str(subtask_id): {
                    "subtask": f"subproblem {subtask_id}",
                    "criteria": {
                        "evaluation_criteria": {
                            stage: [
                                {
                                    "sub_criteria": f"{stage}-criterion",
                                    "score": 10,
                                }
                            ]
                            for stage in STAGES
                        }
                    },
                }
                for subtask_id in (1, 2)
            }
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            report_path = root / "paper.tex"
            criteria_path = root / "criteria.json"
            report_path.write_text("complete evidence", encoding="utf-8")
            criteria_path.write_text(json.dumps(criteria), encoding="utf-8")
            report = {
                "task_id": "2015_A",
                "sample_id": "S001",
                "paper_number": 1,
                "frozen_report": "paper.tex",
                "frozen_criteria": "criteria.json",
            }
            judge_config = {
                "provider": "fake",
                "model_name": "fake-model",
                "api_key": "unused",
                "base_url": "https://unused.invalid",
                "api_version": "unused",
            }

            first = evaluate_report(
                report=report,
                judge_name="fake-judge",
                judge_config=judge_config,
                experiment_dir=root,
                template={"zh": "{{report_content}}", "system": "system"},
                temperature=0.0,
                max_tokens=100,
                max_attempts=2,
                judge_factory=FakeJudge,
            )
            second = evaluate_report(
                report=report,
                judge_name="fake-judge",
                judge_config=judge_config,
                experiment_dir=root,
                template={"zh": "{{report_content}}", "system": "system"},
                temperature=0.0,
                max_tokens=100,
                max_attempts=2,
                judge_factory=FakeJudge,
            )

        self.assertEqual(first["status"], "completed")
        self.assertEqual(first, second)
        self.assertEqual(FakeJudge.calls_made, 2)
        self.assertEqual(first["overall_mean"], 0.5)
        self.assertEqual(set(first["subtasks"]), {"1", "2"})


class LoadJudgeConfigsTest(unittest.TestCase):
    def test_combines_json_and_env_configs_without_exposing_keys(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            api_config = root / "api.json"
            api_config.write_text(
                json.dumps(
                    {
                        "gemini": {
                            "provider": "modelhub",
                            "model_name": "gemini-model",
                            "base_url": "https://gemini.invalid",
                            "api_keys": ["gemini-secret"],
                        }
                    }
                ),
                encoding="utf-8",
            )
            env_path = root / "gpt.env"
            env_path.write_text(
                "MODEL_NAME=gpt-model\n"
                "OPENAI_API_BASE=https://gpt.invalid\n"
                "OPENAI_API_KEY=gpt-secret\n",
                encoding="utf-8",
            )

            configs, public = load_judge_configs(
                api_config=api_config,
                judge_names=["gpt", "gemini"],
                env_judges=[f"gpt={env_path}"],
            )

        self.assertEqual(configs["gpt"]["model_name"], "gpt-model")
        self.assertEqual(configs["gemini"]["provider"], "modelhub")
        self.assertEqual(configs["gpt"]["api_key"], "gpt-secret")
        self.assertNotIn("api_key", public[0])
        self.assertNotIn("gpt-secret", json.dumps(public))
        self.assertNotIn("gemini-secret", json.dumps(public))


class WriteSummariesTest(unittest.TestCase):
    def test_writes_combined_csv_and_per_judge_workbooks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            reports = [{"task_id": "2015_A", "paper_number": 2}]
            for judge_name, model_name, overall in (
                ("gpt", "gpt-model", 8.0),
                ("gemini", "gemini-model", 7.5),
            ):
                path = root / "runs" / judge_name / "results" / "2015_A.json"
                path.parent.mkdir(parents=True)
                path.write_text(
                    json.dumps(
                        {
                            "judge_model": model_name,
                            "stage_scores": {stage: overall for stage in STAGES},
                            "overall_mean": overall,
                        }
                    ),
                    encoding="utf-8",
                )

            write_summaries(root, reports, ["gpt", "gemini"])

            csv_text = (root / "summaries" / "all_judges.csv").read_text(
                encoding="utf-8-sig"
            )
            self.assertIn("gpt,gpt-model,2015_A,2", csv_text)
            self.assertIn("gemini,gemini-model,2015_A,2", csv_text)
            self.assertTrue((root / "summaries" / "gpt.xlsx").is_file())
            self.assertTrue((root / "summaries" / "gemini.xlsx").is_file())


if __name__ == "__main__":
    unittest.main()
