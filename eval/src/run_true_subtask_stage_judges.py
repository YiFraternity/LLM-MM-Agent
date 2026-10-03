#!/usr/bin/env python3
"""Run the repository's true subtask-by-stage judging prompt across judges.

This runner preserves the semantics of ``3_eval_subtasks_openai.py``:
``solution.tex`` is parsed by subtask, the task-specific criteria JSON is
provided in full, and one judge call scores all seven stages for one subtask.
It adds frozen inputs, strict output validation, provenance, retry/resume, and
multi-judge execution without changing the evaluation prompt.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import shutil
import subprocess
import sys
import threading
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from openpyxl import Workbook

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "MMAgent"))
sys.path.insert(0, str(REPO_ROOT / "eval" / "src"))

from eval_utils import clean_json_txt, load_yaml, populate_template  # noqa: E402
from parser_latex import parse_latex  # noqa: E402
from run_multijudge_experiment import (  # noqa: E402
    AuditedJudge,
    sha256_file,
    write_json_atomic,
)


STAGES = ["问题识别", "问题复述", "假设建立", "模型构建", "模型求解", "代码实现", "结果分析"]
TASK_PATTERN = re.compile(r"(\d{4}_[A-F])")
PROGRESS_LOCK = threading.Lock()


def load_actual_eval_module():
    path = REPO_ROOT / "eval" / "src" / "3_eval_subtasks_openai.py"
    spec = importlib.util.spec_from_file_location("actual_subtask_stage_eval", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def git_show(relative_path: str) -> bytes:
    result = subprocess.run(
        ["git", "show", f"HEAD:{relative_path}"],
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise FileNotFoundError(result.stderr.decode("utf-8", errors="replace"))
    return result.stdout


def discover_reports() -> list[dict[str, str]]:
    paths = sorted(
        REPO_ROOT.glob(
            "output/gemini-2.5-flash-priority/CPMCM/MM-Agent-criteria/*/latex/solution.tex"
        )
    )
    reports: list[dict[str, str]] = []
    for index, tex_path in enumerate(paths, start=1):
        match = TASK_PATTERN.search(str(tex_path))
        if match is None:
            continue
        task_id = match.group(1)
        verification = tex_path.parent.parent / "json" / "code_verification.json"
        if not verification.is_file():
            raise FileNotFoundError(f"missing code verification: {verification}")
        reports.append(
            {
                "sample_id": f"S{index:03d}",
                "task_id": task_id,
                "report_model": "gemini-2.5-flash-priority_MM-Agent-criteria",
                "source_tex": str(tex_path.relative_to(REPO_ROOT)),
                "source_code_verification": str(verification.relative_to(REPO_ROOT)),
            }
        )
    return reports


def freeze_inputs(args: argparse.Namespace, public_judges: list[dict[str, str]]) -> dict[str, Any]:
    output_dir = args.output_dir
    manifest_path = output_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["judge_models"] != public_judges:
            raise ValueError("existing manifest judge models differ")
        return manifest

    frozen = output_dir / "frozen_inputs"
    prompt_target = frozen / "prompt" / "criterial_generate.yaml"
    prompt_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.prompt_path, prompt_target)

    reports = discover_reports()
    evaluator = load_actual_eval_module()
    frozen_reports: list[dict[str, Any]] = []
    for report in reports:
        task_id = report["task_id"]
        report_dir = frozen / "reports" / report["sample_id"]
        report_dir.mkdir(parents=True, exist_ok=True)
        tex_target = report_dir / "solution.tex"
        verification_target = report_dir / "code_verification.json"
        criteria_target = frozen / "criteria" / f"{task_id}.json"
        criteria_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO_ROOT / report["source_tex"], tex_target)
        shutil.copy2(REPO_ROOT / report["source_code_verification"], verification_target)
        criteria_target.write_bytes(git_show(f"MMBench/CPMCM/criteria/{task_id}.json"))

        criteria = json.loads(criteria_target.read_text(encoding="utf-8"))
        tree = parse_latex(tex_target.read_text(encoding="utf-8"), target_level=2).to_dict()
        empty_subtasks = [
            sid
            for sid in sorted(criteria.get("subtask", {}), key=int)
            if not evaluator.extract_sections_and_parents(tree, sid)
        ]
        if empty_subtasks:
            raise ValueError(f"{task_id} has empty extracted subtasks: {empty_subtasks}")
        frozen_reports.append(
            {
                **report,
                "frozen_tex": str(tex_target.relative_to(REPO_ROOT)),
                "tex_sha256": sha256_file(tex_target),
                "frozen_code_verification": str(verification_target.relative_to(REPO_ROOT)),
                "code_verification_sha256": sha256_file(verification_target),
                "frozen_criteria": str(criteria_target.relative_to(REPO_ROOT)),
                "criteria_sha256": sha256_file(criteria_target),
                "subtask_count": len(criteria.get("subtask", {})),
            }
        )

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "3_eval_subtasks_openai.py + criterial_generate.yaml",
        "prompt_sha256": sha256_file(prompt_target),
        "judge_models": public_judges,
        "temperature": args.temperature,
        "reports": frozen_reports,
    }
    write_json_atomic(manifest, manifest_path)
    return manifest


def append_code_verification(content: str, verification: dict[str, Any], subtask_id: str) -> str:
    code_file = f"main{subtask_id}.py"
    if code_file not in verification:
        return content
    result = verification[code_file]
    parts = [content, f"\n\n[附录：沙盒代码执行结果 ({code_file})]", "Status: " + ("Success" if result.get("success") else "Failed")]
    if result.get("stdout"):
        parts.append(f"Stdout:\n```\n{result['stdout']}\n```")
    if result.get("stderr"):
        parts.append(f"Stderr:\n```\n{result['stderr']}\n```")
    return "\n".join(parts)


def validate_response(response: Any, subtask_info: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(response, dict):
        raise ValueError("response is not a JSON object")
    criteria = subtask_info.get("criteria", {}).get("evaluation_criteria", {})
    validated: dict[str, Any] = {}
    for stage in STAGES:
        expected = criteria.get(stage, [])
        if not isinstance(expected, list):
            expected = []
        actual = response.get(stage)
        if not isinstance(actual, list):
            raise ValueError(f"{stage} must be a list")
        def normalize_name(value: Any) -> str:
            return "".join(unicodedata.normalize("NFKC", str(value)).split())

        expected_by_name = {
            normalize_name(item.get("sub_criteria")): item for item in expected
        }
        actual_by_name: dict[str, dict[str, Any]] = {}
        for item in actual:
            if not isinstance(item, dict):
                raise ValueError(f"{stage} contains non-object item")
            name = normalize_name(item.get("dimension"))
            if name in actual_by_name:
                raise ValueError(f"duplicate dimension: {stage}/{name}")
            actual_by_name[name] = item
        missing = [name for name in expected_by_name if name not in actual_by_name]
        extra = [name for name in actual_by_name if name not in expected_by_name]
        if missing or extra:
            raise ValueError(f"{stage} dimensions mismatch; missing={missing}, extra={extra}")
        ordered: list[dict[str, Any]] = []
        for name, rubric_item in expected_by_name.items():
            item = actual_by_name[name]
            score = item.get("score")
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise ValueError(f"non-numeric score: {stage}/{name}")
            maximum = float(rubric_item.get("score", 0))
            if not 0 <= float(score) <= maximum:
                raise ValueError(f"score out of range: {stage}/{name}={score}, max={maximum}")
            comment = str(item.get("comment", "")).strip()
            if not comment:
                raise ValueError(f"missing comment: {stage}/{name}")
            ordered.append({**item, "score": float(score)})
        validated[stage] = ordered
    return validated


def compute_report_scores(results: dict[str, Any]) -> tuple[dict[str, float], float]:
    stage_values: dict[str, list[float]] = {stage: [] for stage in STAGES}
    for subtask_result in results.values():
        for stage in STAGES:
            items = subtask_result.get(stage, [])
            if items:
                stage_values[stage].append(sum(float(item["score"]) for item in items) / 10.0)
    stage_scores = {
        stage: sum(values) / len(values)
        for stage, values in stage_values.items()
        if values
    }
    overall = sum(stage_scores.values()) / len(stage_scores)
    return stage_scores, overall


def evaluate_report(
    report: dict[str, Any],
    judge_name: str,
    judge_config: dict[str, Any],
    args: argparse.Namespace,
    semaphore: threading.Semaphore,
) -> dict[str, Any]:
    output_dir = args.output_dir
    result_path = output_dir / "runs" / judge_name / "results" / f"{report['task_id']}.json"
    if result_path.is_file():
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if existing.get("status") == "completed":
            return existing

    evaluator = load_actual_eval_module()
    template = load_yaml(output_dir / "frozen_inputs" / "prompt" / "criterial_generate.yaml")["math_modeling_report_eval"]
    tex_path = REPO_ROOT / report["frozen_tex"]
    criteria = json.loads((REPO_ROOT / report["frozen_criteria"]).read_text(encoding="utf-8"))
    verification = json.loads((REPO_ROOT / report["frozen_code_verification"]).read_text(encoding="utf-8"))
    tree = parse_latex(tex_path.read_text(encoding="utf-8"), target_level=2).to_dict()
    judge = AuditedJudge(
        config_name=judge_name,
        model=judge_config["model_name"],
        provider=judge_config["provider"],
        api_key=judge_config["api_key"],
        base_url=judge_config["base_url"],
        api_version=judge_config["api_version"],
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        artifact_dir=output_dir / "runs" / judge_name / "artifacts" / report["task_id"],
    )

    subtask_results: dict[str, Any] = {}
    for subtask_id, subtask_info in sorted(criteria["subtask"].items(), key=lambda item: int(item[0])):
        sections = evaluator.extract_sections_and_parents(tree, subtask_id)
        content = ""
        for section in sections:
            content += f"\\section{{{section['section_title']}}}\n{section['section_content']}\n"
            content += f"\\subsection{{{section['subsection_title']}}}\n{section['subsection_content']}\n"
        content = append_code_verification(content, verification, subtask_id)
        prompt = populate_template(
            template["zh"],
            {
                "subproblem": subtask_info.get("subtask", ""),
                "report_content": content,
                "report_criteria": json.dumps(subtask_info.get("criteria", {}), ensure_ascii=False, indent=2),
            },
        )
        previous_error = None
        for attempt in range(1, args.max_attempts + 1):
            call_prompt = prompt
            if previous_error:
                call_prompt += f"\n\n上一次输出未通过结构校验：{previous_error}\n请保持评分依据不变，完整重新输出合法 JSON。"
            try:
                with semaphore:
                    raw = judge.generate(prompt=call_prompt, system=template.get("system", ""))
                parsed = clean_json_txt(raw)
                subtask_results[subtask_id] = validate_response(parsed, subtask_info)
                break
            except Exception as exc:  # noqa: BLE001 - bounded output repair
                previous_error = f"{type(exc).__name__}: {exc}"
                if attempt == args.max_attempts:
                    raise
                time.sleep(min(5 * attempt, 20))

    stage_scores, overall = compute_report_scores(subtask_results)
    result = {
        "status": "completed",
        "task_id": report["task_id"],
        "sample_id": report["sample_id"],
        "report_model": report["report_model"],
        "judge_config_name": judge_name,
        "judge_model": judge_config["model_name"],
        "subtasks": subtask_results,
        "stage_scores": stage_scores,
        "overall_mean": overall,
        "calls": judge.calls,
        "usage": judge.get_total_usage(),
    }
    write_json_atomic(result, result_path)
    return result


def write_summaries(output_dir: Path, manifest: dict[str, Any], judge_names: list[str]) -> None:
    summary_dir = output_dir / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    for judge_name in judge_names:
        workbook = Workbook()
        sheet = workbook.active
        sheet.title = "Task_Stage_Scores"
        sheet.append(["Model Name", "Task ID", *STAGES, "Average Stage Score"])
        for report in manifest["reports"]:
            result = json.loads(
                (output_dir / "runs" / judge_name / "results" / f"{report['task_id']}.json").read_text(encoding="utf-8")
            )
            sheet.append(
                [
                    report["report_model"],
                    report["task_id"],
                    *(result["stage_scores"].get(stage) for stage in STAGES),
                    result["overall_mean"],
                ]
            )
        workbook.save(summary_dir / f"{judge_name}.xlsx")


def append_progress(path: Path, record: dict[str, Any]) -> None:
    with PROGRESS_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def run(args: argparse.Namespace) -> int:
    args.output_dir = args.output_dir.resolve()
    args.prompt_path = args.prompt_path.resolve()
    args.api_config = args.api_config.resolve()
    raw_api = json.loads(args.api_config.read_text(encoding="utf-8"))
    judge_configs: dict[str, dict[str, Any]] = {}
    public_judges: list[dict[str, str]] = []
    for name in args.judge_model:
        raw = raw_api[name]
        judge_configs[name] = {
            "provider": raw.get("provider", "openai"),
            "base_url": raw["base_url"],
            "api_version": raw.get("api_version", "2024-03-01-preview"),
            "model_name": raw["model_name"],
            "api_key": raw["api_keys"][0],
        }
        public_judges.append(
            {"config_name": name, "provider": judge_configs[name]["provider"], "model_name": judge_configs[name]["model_name"]}
        )

    manifest = freeze_inputs(args, public_judges)
    reports = (
        manifest["reports"][: args.limit_reports]
        if args.limit_reports is not None
        else manifest["reports"]
    )
    semaphores = {name: threading.Semaphore(args.per_judge_concurrency) for name in args.judge_model}
    jobs = [(report, name) for report in reports for name in args.judge_model]
    completed = 0
    failures: list[dict[str, Any]] = []
    counter_lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(evaluate_report, report, name, judge_configs[name], args, semaphores[name]): (report, name)
            for report, name in jobs
        }
        for future in as_completed(futures):
            report, name = futures[future]
            try:
                result = future.result()
                with counter_lock:
                    completed += 1
                    done = completed
                record = {
                    "status": "completed",
                    "completed": done,
                    "total": len(jobs),
                    "task_id": report["task_id"],
                    "judge": name,
                    "overall_mean": result["overall_mean"],
                }
            except Exception as exc:  # noqa: BLE001 - finish independent jobs
                record = {
                    "status": "failed",
                    "task_id": report["task_id"],
                    "judge": name,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                failures.append(record)
            append_progress(args.output_dir / "progress.jsonl", record)
            print(json.dumps(record, ensure_ascii=False), flush=True)

    if not failures:
        write_summaries(args.output_dir, {"reports": reports}, args.judge_model)
    write_json_atomic(
        {"total": len(jobs), "completed": completed, "failed": len(failures), "failures": failures},
        args.output_dir / "run_summary.json",
    )
    return 1 if failures else 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="真实 subtask×stage 双裁判实验")
    parser.add_argument("--judge-model", action="append", required=True)
    parser.add_argument("--api-config", type=Path, default=REPO_ROOT / "api-available.json")
    parser.add_argument("--prompt-path", type=Path, default=REPO_ROOT / "eval/prompts/criterial_generate.yaml")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--max-attempts", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--per-judge-concurrency", type=int, default=2)
    parser.add_argument("--limit-reports", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if len(args.judge_model) < 2 or len(args.judge_model) != len(set(args.judge_model)):
        raise ValueError("need at least two distinct judge models")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
