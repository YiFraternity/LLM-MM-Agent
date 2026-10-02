import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "MMAgent"))
sys.path.insert(0, str(REPO_ROOT / "eval" / "src"))

SPEC = importlib.util.spec_from_file_location(
    "report_evaluator_for_test",
    REPO_ROOT / "eval" / "src" / "3_eval_report_using_mmagent.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class TaskUnderstandingExtractionTests(unittest.TestCase):
    def test_reads_nested_current_schema(self):
        criteria = {
            "subtask": {
                "1": {
                    "criteria": {
                        "task_understanding": {
                            "core_goal": "核心目标 A",
                            "expected_output": "输出 B",
                            "key_inputs_constraints": "约束 C",
                            "modeling_type": "优化",
                            "role_in_pipeline": "前置环节",
                            "assumptions": "假设 D",
                        }
                    }
                }
            }
        }
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "2004_A.json"
            path.write_text(json.dumps(criteria, ensure_ascii=False), encoding="utf-8")
            text = MODULE.extract_task_understanding_all({}, path.parent, "2004_A")

        self.assertIn("核心目标 A", text)
        self.assertIn("输出 B", text)
        self.assertIn("约束 C", text)
        self.assertIn("假设 D", text)
        self.assertNotIn("No task understanding", text)

    def test_accepts_legacy_direct_schema(self):
        criteria = {
            "subtask": {
                "1": {
                    "task_understanding": {
                        "core_goal": "legacy goal",
                    }
                }
            }
        }
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "2004_A.json"
            path.write_text(json.dumps(criteria), encoding="utf-8")
            text = MODULE.extract_task_understanding_all({}, path.parent, "2004_A")

        self.assertIn("legacy goal", text)

    def test_fails_fast_when_schema_contains_no_understanding(self):
        criteria = {"subtask": {"1": {"criteria": {}}}}
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "2004_A.json"
            path.write_text(json.dumps(criteria), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "Failed to load"):
                MODULE.extract_task_understanding_all({}, path.parent, "2004_A")


if __name__ == "__main__":
    unittest.main()
