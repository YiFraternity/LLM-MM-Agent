#!/usr/bin/env python3
"""Evaluate reports with atomic rubric decisions and deterministic scoring.

Judges classify each task-specific criterion into a fixed ordinal level and
quote supporting evidence. Python, rather than the judge model, converts those
levels and criterion weights into stage scores. This reduces model-specific
interpretations of a free-form 1--10 scale.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from openpyxl import Workbook

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "eval" / "src"))

from run_multijudge_experiment import (  # noqa: E402
    AuditedJudge,
    sha256_file,
    write_json_atomic,
)


LEVEL_MULTIPLIERS = {
    "FULL": 1.0,
    "ALMOST": 0.8,
    "PARTIAL": 0.5,
    "BARELY": 0.25,
    "NOT_MET": 0.0,
    "ABSENT": 0.0,
}

STAGE_SECTIONS = {
    "problem_analysis": ["问题识别", "问题复述"],
    "modeling_rigor": ["假设建立", "模型构建"],
    "practicality_scientificity": ["模型求解", "代码实现"],
    "result_bias": ["结果分析"],
}

PROGRESS_LOCK = threading.Lock()


SYSTEM_PROMPT = """你是一名严格的数学建模竞赛评审专家。你的唯一任务是逐项判断给定评分点是否被报告中的明确证据满足。不得根据常识补写报告内容，不得直接给出1--10总分，不得因为术语多、方法新或篇幅长而提高判定。"""


def parse_json_response(text: str) -> dict[str, Any]:
    stripped = text.strip()
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", stripped, re.IGNORECASE)
    if fenced:
        stripped = fenced.group(1).strip()
    return json.loads(stripped)


def report_subtask_content(report: dict[str, Any], subtask_index: int) -> str:
    tasks = report.get("tasks", [])
    if not isinstance(tasks, list) or subtask_index >= len(tasks):
        return "报告中没有与该子任务对应的结构化内容。"
    task = tasks[subtask_index]
    if not isinstance(task, dict):
        return "报告中该子任务内容格式无效。"
    selected = {
        key: task.get(key)
        for key in [
            "task_description",
            "task_analysis",
            "mathematical_modeling_process",
            "subtask_outcome_analysis",
            "answer",
        ]
        if task.get(key)
    }
    return json.dumps(selected, ensure_ascii=False, indent=2)


def flatten_criteria(subtask: dict[str, Any]) -> list[dict[str, Any]]:
    criteria = subtask.get("criteria", {}).get("evaluation_criteria", {})
    flattened: list[dict[str, Any]] = []
    for section, entries in criteria.items():
        if not isinstance(entries, list):
            continue
        for index, item in enumerate(entries, start=1):
            if not isinstance(item, dict):
                continue
            flattened.append(
                {
                    "criterion_id": f"{section}::{index}",
                    "section": section,
                    "criterion": item.get("sub_criteria", ""),
                    "description": item.get("description", ""),
                    "scoring_hint": item.get("scoring_hint", ""),
                    "max_weight": float(item.get("score", 0)),
                }
            )
    return flattened


def build_prompt(
    subtask_id: str,
    subtask: dict[str, Any],
    report_content: str,
    criteria: list[dict[str, Any]],
    previous_error: str | None = None,
) -> str:
    payload = [
        f"【子任务 {subtask_id}】",
        str(subtask.get("subtask", "")),
        "",
        "【报告中该子任务的内容】",
        report_content,
        "",
        "【必须逐项判断的评分点】",
        json.dumps(criteria, ensure_ascii=False, indent=2),
        "",
        "对每个 criterion_id 恰好输出一次。level 只能是以下六个值之一：",
        "FULL, ALMOST, PARTIAL, BARELY, NOT_MET, ABSENT。",
        "判定含义：FULL=完整且有可复核证据；ALMOST=核心齐全但次要细节不足；",
        "PARTIAL=提及但不完整；BARELY=只有很浅的相关描述；NOT_MET=内容偏离要求；",
        "ABSENT=报告完全没有相关内容。",
        "evidence 必须是报告原文短引或 NONE；reason 用一句话说明判定。",
        "不得输出分数，分数由程序根据 level 和 max_weight 计算。",
        "严格输出 JSON，不要 Markdown，不要额外文字：",
        '{"evaluations":[{"criterion_id":"...","level":"FULL|ALMOST|PARTIAL|BARELY|NOT_MET|ABSENT","evidence":"...或NONE","reason":"..."}]}',
    ]
    if previous_error:
        payload.extend(["", f"上一次输出校验失败：{previous_error}。请完整重做。"])
    return "\n".join(payload)


def validate_evaluations(
    response: dict[str, Any], criteria: list[dict[str, Any]]
) -> list[dict[str, str]]:
    evaluations = response.get("evaluations")
    if not isinstance(evaluations, list):
        raise ValueError("evaluations must be a list")
    expected_ids = [item["criterion_id"] for item in criteria]
    by_id: dict[str, dict[str, Any]] = {}
    for item in evaluations:
        if not isinstance(item, dict):
            raise ValueError("each evaluation must be an object")
        criterion_id = item.get("criterion_id")
        if criterion_id in by_id:
            raise ValueError(f"duplicate criterion_id: {criterion_id}")
        by_id[criterion_id] = item
    missing = [item for item in expected_ids if item not in by_id]
    extra = [item for item in by_id if item not in expected_ids]
    if missing or extra:
        raise ValueError(f"criterion ids mismatch; missing={missing}, extra={extra}")

    validated: list[dict[str, str]] = []
    for criterion_id in expected_ids:
        item = by_id[criterion_id]
        level = str(item.get("level", "")).upper()
        if level not in LEVEL_MULTIPLIERS:
            raise ValueError(f"invalid level for {criterion_id}: {level}")
        evidence = str(item.get("evidence", "")).strip()
        reason = str(item.get("reason", "")).strip()
        if not evidence or not reason:
            raise ValueError(f"missing evidence/reason for {criterion_id}")
        validated.append(
            {
                "criterion_id": criterion_id,
                "level": level,
                "evidence": evidence,
                "reason": reason,
            }
        )
    return validated


def compute_scores(
    subtask_results: dict[str, Any], criteria_by_subtask: dict[str, list[dict[str, Any]]]
) -> tuple[dict[str, float], float]:
    section_scores: dict[str, list[float]] = {}
    for subtask_id, result in subtask_results.items():
        criteria = criteria_by_subtask[subtask_id]
        decisions = {
            item["criterion_id"]: item for item in result["evaluations"]
        }
        section_weighted: dict[str, float] = {}
        section_max: dict[str, float] = {}
        for criterion in criteria:
            section = criterion["section"]
            weight = criterion["max_weight"]
            level = decisions[criterion["criterion_id"]]["level"]
            section_weighted[section] = section_weighted.get(section, 0.0) + (
                weight * LEVEL_MULTIPLIERS[level]
            )
            section_max[section] = section_max.get(section, 0.0) + weight
        for section, weighted in section_weighted.items():
            if section_max[section] > 0:
                section_scores.setdefault(section, []).append(
                    10.0 * weighted / section_max[section]
                )

    stage_scores: dict[str, float] = {}
    for stage, sections in STAGE_SECTIONS.items():
        values = [score for section in sections for score in section_scores.get(section, [])]
        if values:
            stage_scores[stage] = sum(values) / len(values)
    overall = sum(stage_scores.values()) / len(stage_scores)
    return stage_scores, overall


def evaluate_sample(
    sample: dict[str, Any],
    judge_name: str,
    judge_config: dict[str, Any],
    source_root: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    result_path = output_dir / "runs" / judge_name / "results" / f"{sample['sample_id']}.json"
    if result_path.is_file():
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if existing.get("status") == "completed":
            return existing

    report = json.loads((source_root / sample["frozen_report_path"]).read_text(encoding="utf-8"))
    criteria_data = json.loads((source_root / sample["frozen_criteria_path"]).read_text(encoding="utf-8"))
    subtasks = criteria_data.get("subtask", {})
    artifact_dir = output_dir / "runs" / judge_name / "artifacts" / sample["sample_id"]
    judge = AuditedJudge(
        config_name=judge_name,
        model=judge_config["model_name"],
        provider=judge_config["provider"],
        api_key=judge_config["api_key"],
        base_url=judge_config["base_url"],
        api_version=judge_config["api_version"],
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        artifact_dir=artifact_dir,
    )

    subtask_results: dict[str, Any] = {}
    criteria_by_subtask: dict[str, list[dict[str, Any]]] = {}
    for ordinal, (subtask_id, subtask) in enumerate(
        sorted(subtasks.items(), key=lambda item: int(item[0]))
    ):
        criteria = flatten_criteria(subtask)
        criteria_by_subtask[subtask_id] = criteria
        previous_error = None
        for attempt in range(1, args.max_attempts + 1):
            prompt = build_prompt(
                subtask_id,
                subtask,
                report_subtask_content(report, ordinal),
                criteria,
                previous_error,
            )
            try:
                raw = judge.generate(prompt=prompt, system=SYSTEM_PROMPT)
                parsed = parse_json_response(raw)
                evaluations = validate_evaluations(parsed, criteria)
                subtask_results[subtask_id] = {"evaluations": evaluations}
                break
            except Exception as exc:  # noqa: BLE001 - bounded repair retry
                previous_error = f"{type(exc).__name__}: {exc}"
                if attempt == args.max_attempts:
                    raise
                time.sleep(min(2**attempt, 10))

    stage_scores, overall = compute_scores(subtask_results, criteria_by_subtask)
    result = {
        "status": "completed",
        "sample_id": sample["sample_id"],
        "task_id": sample["task_id"],
        "report_model": sample["report_model"],
        "judge_config_name": judge_name,
        "judge_model": judge_config["model_name"],
        "level_multipliers": LEVEL_MULTIPLIERS,
        "subtasks": subtask_results,
        "stage_scores": stage_scores,
        "overall_mean": overall,
        "calls": judge.calls,
        "usage": judge.get_total_usage(),
    }
    write_json_atomic(result, result_path)
    return result


def write_summary(
    output_dir: Path, manifest: dict[str, Any], judge_names: list[str]
) -> None:
    summaries = output_dir / "summaries"
    summaries.mkdir(parents=True, exist_ok=True)
    for judge_name in judge_names:
        workbook = Workbook()
        sheet = workbook.active
        sheet.title = "Task_Stage_Scores"
        header = ["Model Name", "Task ID", *STAGE_SECTIONS, "Average Stage Score"]
        sheet.append(header)
        for sample in manifest["samples"]:
            result_path = output_dir / "runs" / judge_name / "results" / f"{sample['sample_id']}.json"
            result = json.loads(result_path.read_text(encoding="utf-8"))
            row = [sample["report_model"], sample["task_id"]]
            row.extend(result["stage_scores"].get(stage) for stage in STAGE_SECTIONS)
            row.append(result["overall_mean"])
            sheet.append(row)
        workbook.save(summaries / f"{judge_name}.xlsx")


def append_progress(path: Path, record: dict[str, Any]) -> None:
    with PROGRESS_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def run(args: argparse.Namespace) -> int:
    source_manifest_path = args.source_manifest.resolve()
    source_root = REPO_ROOT
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    samples = source_manifest["samples"][: args.sample_size]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    api_config = json.loads(args.api_config.read_text(encoding="utf-8"))

    judge_configs: dict[str, dict[str, Any]] = {}
    for judge_name in args.judge_model:
        raw = api_config[judge_name]
        judge_configs[judge_name] = {
            "provider": raw.get("provider", "openai"),
            "base_url": raw["base_url"],
            "api_version": raw.get("api_version", "2024-03-01-preview"),
            "model_name": raw["model_name"],
            "api_key": raw["api_keys"][0],
        }

    public_manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_manifest": str(source_manifest_path.relative_to(REPO_ROOT)),
        "sample_size": len(samples),
        "samples": samples,
        "judge_models": [
            {
                "config_name": name,
                "provider": judge_configs[name]["provider"],
                "model_name": judge_configs[name]["model_name"],
            }
            for name in args.judge_model
        ],
        "level_multipliers": LEVEL_MULTIPLIERS,
        "temperature": args.temperature,
        "runner_sha256": sha256_file(Path(__file__)),
    }
    write_json_atomic(public_manifest, output_dir / "manifest.json")

    completed = 0
    failures: list[dict[str, Any]] = []
    counter_lock = threading.Lock()

    def run_judge(judge_name: str) -> list[dict[str, Any]]:
        nonlocal completed
        local_failures: list[dict[str, Any]] = []
        for sample in samples:
            try:
                result = evaluate_sample(
                    sample,
                    judge_name,
                    judge_configs[judge_name],
                    source_root,
                    output_dir,
                    args,
                )
                with counter_lock:
                    completed += 1
                    done = completed
                record = {
                    "status": "completed",
                    "completed": done,
                    "total": len(samples) * len(args.judge_model),
                    "sample_id": sample["sample_id"],
                    "judge": judge_name,
                    "overall_mean": result["overall_mean"],
                }
            except Exception as exc:  # noqa: BLE001 - finish remaining samples
                record = {
                    "status": "failed",
                    "sample_id": sample["sample_id"],
                    "judge": judge_name,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                local_failures.append(record)
            append_progress(output_dir / "progress.jsonl", record)
            print(json.dumps(record, ensure_ascii=False), flush=True)
        return local_failures

    with ThreadPoolExecutor(max_workers=len(args.judge_model)) as pool:
        futures = [pool.submit(run_judge, name) for name in args.judge_model]
        for future in as_completed(futures):
            failures.extend(future.result())

    if not failures:
        write_summary(output_dir, public_manifest, args.judge_model)
    write_json_atomic(
        {
            "total": len(samples) * len(args.judge_model),
            "completed": completed,
            "failed": len(failures),
            "failures": failures,
        },
        output_dir / "run_summary.json",
    )
    return 1 if failures else 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="原子评分点多裁判实验")
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--judge-model", action="append", required=True)
    parser.add_argument("--sample-size", type=int, default=10)
    parser.add_argument("--api-config", type=Path, default=REPO_ROOT / "api-available.json")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--max-attempts", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if len(args.judge_model) < 2 or len(args.judge_model) != len(set(args.judge_model)):
        raise ValueError("需要至少两个互不相同的裁判模型")
    args.api_config = args.api_config.resolve()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
