#!/usr/bin/env python3
"""Score one deterministic Best Paper per task with task-specific rubrics.

The runner freezes a reproducible paper selection, cleaned LaTeX, criteria,
and prompt before making model calls. Each task-specific subtask receives the
complete cleaned paper as evidence because the source papers' numbered
questions do not necessarily match the evaluation subtasks.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from openpyxl import Workbook

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "MMAgent"))
sys.path.insert(0, str(REPO_ROOT / "eval" / "src"))

from eval_utils import clean_json_txt, load_yaml, populate_template
from run_multijudge_experiment import AuditedJudge, sha256_file, write_json_atomic
from run_true_subtask_stage_judges import (
    STAGES,
    compute_report_scores,
    validate_response,
)


PROGRESS_LOCK = threading.Lock()


def select_bestpapers(
    *, bestpaper_root: Path, criteria_root: Path, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Select one paper per eligible Task ID with a reproducible RNG stream."""
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    for task_dir in sorted(path for path in bestpaper_root.iterdir() if path.is_dir()):
        candidates = sorted(task_dir.glob("*.tex"))
        if not candidates:
            continue
        if not (criteria_root / f"{task_dir.name}.json").is_file():
            exclusions.append(
                {
                    "task_id": task_dir.name,
                    "reason": "missing task-specific criteria",
                    "candidate_count": len(candidates),
                }
            )
            continue
        chosen = rng.choice(candidates)
        selected.append(
            {
                "task_id": task_dir.name,
                "paper_number": int(chosen.stem),
                "source_path": str(chosen),
            }
        )
    return selected, exclusions


def clean_bestpaper_tex(source: str) -> str:
    """Remove known conversion noise without summarizing paper evidence."""
    cleaned = source.replace(r"\tableofcontents", "")
    cleaned = re.sub(r"(?m)^\s*%.*$", "", cleaned)
    return cleaned.strip() + "\n"


def build_subtask_inputs(
    report_content: str, criteria: dict[str, Any]
) -> list[dict[str, Any]]:
    """Use the complete cleaned paper as evidence for every rubric subtask."""
    subtasks = criteria.get("subtask")
    if not isinstance(subtasks, dict) or not subtasks:
        raise ValueError("criteria must contain a non-empty subtask mapping")
    return [
        {
            "subtask_id": subtask_id,
            "subproblem": subtask_info.get("subtask", ""),
            "criteria": subtask_info.get("criteria", {}),
            "report_content": report_content,
        }
        for subtask_id, subtask_info in sorted(
            subtasks.items(), key=lambda item: int(item[0])
        )
    ]


def exact_output_contract(subtask_info: Mapping[str, Any]) -> str:
    """Describe the only accepted stage/dimension keys for one subtask."""
    evaluation_criteria = (
        subtask_info.get("criteria", {}).get("evaluation_criteria", {})
    )
    schema = {
        stage: [
            str(item.get("sub_criteria", ""))
            for item in evaluation_criteria.get(stage, [])
            if isinstance(item, Mapping)
        ]
        for stage in STAGES
    }
    return (
        "\n\n=====================================\n"
        "🔐 精确输出键约束（机器校验，必须遵守）\n"
        "=====================================\n"
        "下列对象给出了每个 stage 唯一允许的 dimension 名称。"
        "不得改写、增删或合并 dimension 名称；名称必须逐字复制。"
        "若某个 stage 对应空数组，该 stage 必须原样输出 []，不得自行补充评分项。\n"
        + json.dumps(schema, ensure_ascii=False)
    )


def prune_not_applicable_stages(
    response: Any, subtask_info: Mapping[str, Any]
) -> Any:
    """Make rubric-declared empty stages authoritative without mutating raw output."""
    if not isinstance(response, dict):
        return response
    evaluation_criteria = (
        subtask_info.get("criteria", {}).get("evaluation_criteria", {})
    )
    pruned = dict(response)
    for stage in STAGES:
        if evaluation_criteria.get(stage, []) == []:
            pruned[stage] = []
    return pruned


def freeze_inputs(
    *,
    bestpaper_root: Path,
    criteria_root: Path,
    prompt_path: Path,
    output_dir: Path,
    seed: int,
    public_judges: list[dict[str, str]],
    git_commit: str,
) -> dict[str, Any]:
    """Freeze selected papers and all scoring inputs into an audit directory."""
    manifest_path = output_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        configuration = manifest.get("configuration", {})
        if configuration.get("seed") != seed:
            raise ValueError("existing manifest seed differs")
        if configuration.get("judge_models") != public_judges:
            raise ValueError("existing manifest judge models differ")
        return manifest

    selected, exclusions = select_bestpapers(
        bestpaper_root=bestpaper_root,
        criteria_root=criteria_root,
        seed=seed,
    )
    frozen_root = output_dir / "frozen_inputs"
    prompt_target = frozen_root / "prompt" / prompt_path.name
    prompt_target.parent.mkdir(parents=True, exist_ok=True)
    prompt_target.write_bytes(prompt_path.read_bytes())

    reports: list[dict[str, Any]] = []
    for index, sample in enumerate(selected, start=1):
        task_id = str(sample["task_id"])
        sample_id = f"S{index:03d}"
        source_path = Path(str(sample["source_path"]))
        report_target = (
            frozen_root
            / "reports"
            / f"{sample_id}__{task_id}__paper{sample['paper_number']}.tex"
        )
        report_target.parent.mkdir(parents=True, exist_ok=True)
        raw_report = source_path.read_text(encoding="utf-8", errors="replace")
        report_target.write_text(clean_bestpaper_tex(raw_report), encoding="utf-8")

        criteria_source = criteria_root / f"{task_id}.json"
        criteria_target = frozen_root / "criteria" / f"{task_id}.json"
        criteria_target.parent.mkdir(parents=True, exist_ok=True)
        criteria_target.write_bytes(criteria_source.read_bytes())
        criteria = json.loads(criteria_target.read_text(encoding="utf-8"))
        subtask_count = len(criteria.get("subtask", {}))
        if subtask_count <= 0:
            raise ValueError(f"{task_id} has no criteria subtasks")

        reports.append(
            {
                "sample_id": sample_id,
                "task_id": task_id,
                "paper_number": sample["paper_number"],
                "report_model": f"BestPaper-{sample['paper_number']}",
                "source_path": str(source_path),
                "source_sha256": sha256_file(source_path),
                "frozen_report": str(report_target.relative_to(output_dir)),
                "frozen_report_sha256": sha256_file(report_target),
                "frozen_criteria": str(criteria_target.relative_to(output_dir)),
                "criteria_sha256": sha256_file(criteria_target),
                "subtask_count": subtask_count,
                "cleaned_character_count": len(
                    report_target.read_text(encoding="utf-8")
                ),
            }
        )

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "seed": seed,
            "selection": "one uniform random choice per Task ID from sorted .tex candidates",
            "evidence_scope": "complete cleaned paper supplied independently to each task-specific subtask rubric",
            "judge_models": public_judges,
            "prompt": str(prompt_target.relative_to(output_dir)),
            "prompt_sha256": sha256_file(prompt_target),
            "git_commit": git_commit,
        },
        "reports": reports,
        "excluded_tasks": exclusions,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(manifest, manifest_path)
    return manifest


def _sum_usage(total: dict[str, int], addition: Mapping[str, Any]) -> None:
    for key, value in addition.items():
        if isinstance(value, int):
            total[key] = total.get(key, 0) + value


def evaluate_report(
    *,
    report: dict[str, Any],
    judge_name: str,
    judge_config: dict[str, str],
    experiment_dir: Path,
    template: dict[str, str],
    temperature: float,
    max_tokens: int,
    max_attempts: int,
    judge_factory: Callable[..., Any] = AuditedJudge,
) -> dict[str, Any]:
    """Score one frozen paper for one judge with subtask-level checkpoints."""
    result_path = (
        experiment_dir / "runs" / judge_name / "results" / f"{report['task_id']}.json"
    )
    if result_path.is_file():
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if existing.get("status") == "completed":
            return existing

    checkpoint_path = (
        experiment_dir
        / "runs"
        / judge_name
        / "checkpoints"
        / f"{report['task_id']}.json"
    )
    if checkpoint_path.is_file():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    else:
        checkpoint = {"subtasks": {}, "calls": [], "usage": {}}

    report_content = (
        experiment_dir / str(report["frozen_report"])
    ).read_text(encoding="utf-8")
    criteria = json.loads(
        (experiment_dir / str(report["frozen_criteria"])).read_text(encoding="utf-8")
    )
    inputs = build_subtask_inputs(report_content, criteria)

    for subtask_input in inputs:
        subtask_id = str(subtask_input["subtask_id"])
        if subtask_id in checkpoint["subtasks"]:
            continue
        prompt = populate_template(
            template["zh"],
            {
                "subproblem": subtask_input["subproblem"],
                "report_content": subtask_input["report_content"],
                "report_criteria": json.dumps(
                    subtask_input["criteria"], ensure_ascii=False, indent=2
                ),
            },
        )
        prompt += exact_output_contract(subtask_input)
        previous_error: str | None = None
        for attempt in range(1, max_attempts + 1):
            judge = judge_factory(
                config_name=judge_name,
                model=judge_config["model_name"],
                provider=judge_config["provider"],
                api_key=judge_config["api_key"],
                base_url=judge_config["base_url"],
                api_version=judge_config["api_version"],
                temperature=temperature,
                max_tokens=max_tokens,
                artifact_dir=(
                    experiment_dir
                    / "runs"
                    / judge_name
                    / "artifacts"
                    / report["task_id"]
                    / f"subtask_{subtask_id}"
                    / f"attempt_{attempt}"
                ),
            )
            call_prompt = prompt
            if previous_error:
                call_prompt += (
                    "\n\n上一次输出未通过结构校验："
                    + previous_error
                    + "\n请保持评分依据不变，完整重新输出合法 JSON。"
                )
            try:
                raw = judge.generate(
                    prompt=call_prompt, system=template.get("system", "")
                )
                parsed = clean_json_txt(raw)
                parsed = prune_not_applicable_stages(parsed, subtask_input)
                subtask_info = criteria["subtask"][subtask_id]
                checkpoint["subtasks"][subtask_id] = validate_response(
                    parsed, subtask_info
                )
                checkpoint["calls"].extend(judge.calls)
                _sum_usage(checkpoint["usage"], judge.get_total_usage())
                write_json_atomic(checkpoint, checkpoint_path)
                break
            except Exception as error:  # noqa: BLE001 - bounded output repair
                checkpoint["calls"].extend(judge.calls)
                _sum_usage(checkpoint["usage"], judge.get_total_usage())
                write_json_atomic(checkpoint, checkpoint_path)
                previous_error = f"{type(error).__name__}: {error}"
                if attempt == max_attempts:
                    raise
                time.sleep(min(2**attempt, 10))

    stage_scores, overall = compute_report_scores(checkpoint["subtasks"])
    result = {
        "status": "completed",
        "sample_id": report["sample_id"],
        "task_id": report["task_id"],
        "paper_number": report["paper_number"],
        "report_model": report.get("report_model", f"BestPaper-{report['paper_number']}"),
        "judge_config_name": judge_name,
        "judge_model": judge_config["model_name"],
        "subtasks": checkpoint["subtasks"],
        "stage_scores": stage_scores,
        "overall_mean": overall,
        "calls": checkpoint["calls"],
        "usage": checkpoint["usage"],
    }
    write_json_atomic(result, result_path)
    return result


def _read_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def load_judge_configs(
    *, api_config: Path, judge_names: Sequence[str], env_judges: Sequence[str]
) -> tuple[dict[str, dict[str, str]], list[dict[str, str]]]:
    """Load JSON and optional dotenv-backed judge configs without persisting keys."""
    raw_api = json.loads(api_config.read_text(encoding="utf-8"))
    env_paths: dict[str, Path] = {}
    for specification in env_judges:
        if "=" not in specification:
            raise ValueError("--env-judge must use NAME=PATH")
        name, raw_path = specification.split("=", 1)
        env_paths[name] = Path(raw_path).resolve()

    configs: dict[str, dict[str, str]] = {}
    public: list[dict[str, str]] = []
    for name in judge_names:
        if name in env_paths:
            values = _read_env_file(env_paths[name])
            config = {
                "provider": "openai",
                "base_url": values.get("OPENAI_API_BASE", "https://api.openai.com/v1"),
                "api_version": values.get("API_VERSION", "2024-03-01-preview"),
                "model_name": values.get("MODEL_NAME", name),
                "api_key": values.get("OPENAI_API_KEY", ""),
            }
        else:
            raw = raw_api.get(name)
            if not isinstance(raw, dict):
                raise KeyError(f"missing judge config: {name}")
            keys = raw.get("api_keys")
            config = {
                "provider": str(raw.get("provider", "openai")),
                "base_url": str(raw.get("base_url", "")),
                "api_version": str(raw.get("api_version", "2024-03-01-preview")),
                "model_name": str(raw.get("model_name", name)),
                "api_key": str(keys[0] if isinstance(keys, list) and keys else ""),
            }
        if not config["api_key"] or not config["base_url"] or not config["model_name"]:
            raise ValueError(f"incomplete judge config: {name}")
        configs[name] = config
        public.append(
            {
                "config_name": name,
                "provider": config["provider"],
                "model_name": config["model_name"],
            }
        )
    return configs, public


def append_progress(path: Path, record: dict[str, Any]) -> None:
    with PROGRESS_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_summaries(
    output_dir: Path, reports: Sequence[Mapping[str, Any]], judge_names: Sequence[str]
) -> None:
    summary_dir = output_dir / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    csv_rows: list[dict[str, Any]] = []
    for judge_name in judge_names:
        workbook = Workbook()
        sheet = workbook.active
        sheet.title = "Task_Stage_Scores"
        sheet.append(["Task ID", "Paper Number", *STAGES, "Average Stage Score"])
        for report in reports:
            result_path = (
                output_dir
                / "runs"
                / judge_name
                / "results"
                / f"{report['task_id']}.json"
            )
            if not result_path.is_file():
                continue
            result = json.loads(result_path.read_text(encoding="utf-8"))
            row = {
                "judge": judge_name,
                "judge_model": result["judge_model"],
                "task_id": report["task_id"],
                "paper_number": report["paper_number"],
                **{stage: result["stage_scores"].get(stage) for stage in STAGES},
                "overall_mean": result["overall_mean"],
            }
            csv_rows.append(row)
            sheet.append(
                [
                    report["task_id"],
                    report["paper_number"],
                    *(row[stage] for stage in STAGES),
                    row["overall_mean"],
                ]
            )
        workbook.save(summary_dir / f"{judge_name}.xlsx")

    fields = [
        "judge",
        "judge_model",
        "task_id",
        "paper_number",
        *STAGES,
        "overall_mean",
    ]
    with (summary_dir / "all_judges.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(csv_rows)


def run(args: argparse.Namespace) -> int:
    output_dir = args.output_dir.resolve()
    judge_configs, public_judges = load_judge_configs(
        api_config=args.api_config.resolve(),
        judge_names=args.judge_model,
        env_judges=args.env_judge,
    )
    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    manifest = freeze_inputs(
        bestpaper_root=args.bestpaper_root.resolve(),
        criteria_root=args.criteria_root.resolve(),
        prompt_path=args.prompt_path.resolve(),
        output_dir=output_dir,
        seed=args.seed,
        public_judges=public_judges,
        git_commit=git_commit,
    )
    print(
        f"Prepared {len(manifest['reports'])} reports; "
        f"excluded {len(manifest['excluded_tasks'])} tasks",
        flush=True,
    )
    if args.prepare_only:
        return 0

    frozen_prompt = output_dir / manifest["configuration"]["prompt"]
    template = load_yaml(frozen_prompt)["math_modeling_report_eval"]
    reports = (
        manifest["reports"][: args.limit_reports]
        if args.limit_reports is not None
        else manifest["reports"]
    )
    jobs = [(report, judge_name) for report in reports for judge_name in args.judge_model]
    semaphores = {
        name: threading.Semaphore(args.per_judge_concurrency)
        for name in args.judge_model
    }
    completed = 0
    failures: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {}
        for report, judge_name in jobs:
            def job(
                current_report: dict[str, Any] = report,
                current_judge: str = judge_name,
            ) -> dict[str, Any]:
                with semaphores[current_judge]:
                    return evaluate_report(
                        report=current_report,
                        judge_name=current_judge,
                        judge_config=judge_configs[current_judge],
                        experiment_dir=output_dir,
                        template=template,
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                        max_attempts=args.max_attempts,
                    )

            futures[pool.submit(job)] = (report, judge_name)

        for future in as_completed(futures):
            report, judge_name = futures[future]
            try:
                result = future.result()
                completed += 1
                record = {
                    "time_utc": datetime.now(timezone.utc).isoformat(),
                    "status": "completed",
                    "completed": completed,
                    "total": len(jobs),
                    "task_id": report["task_id"],
                    "paper_number": report["paper_number"],
                    "judge": judge_name,
                    "overall_mean": result["overall_mean"],
                    "usage": result["usage"],
                }
            except Exception as error:  # noqa: BLE001 - finish independent jobs
                record = {
                    "time_utc": datetime.now(timezone.utc).isoformat(),
                    "status": "failed",
                    "task_id": report["task_id"],
                    "paper_number": report["paper_number"],
                    "judge": judge_name,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
                failures.append(record)
            append_progress(output_dir / "progress.jsonl", record)
            print(json.dumps(record, ensure_ascii=False), flush=True)

    write_summaries(output_dir, reports, args.judge_model)
    write_json_atomic(
        {
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "total_jobs": len(jobs),
            "completed_jobs": completed,
            "failed_jobs": len(failures),
            "failures": failures,
        },
        output_dir / "run_summary.json",
    )
    return 1 if failures else 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score one deterministic Best Paper per task with two judges"
    )
    parser.add_argument("--judge-model", action="append", required=True)
    parser.add_argument(
        "--env-judge",
        action="append",
        default=[],
        help="optional NAME=PATH dotenv source for a judge",
    )
    parser.add_argument("--seed", type=int, default=20261006)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--max-attempts", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--per-judge-concurrency", type=int, default=2)
    parser.add_argument("--limit-reports", type=int, default=None)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--api-config", type=Path, default=REPO_ROOT / "api-available.json"
    )
    parser.add_argument(
        "--bestpaper-root",
        type=Path,
        default=REPO_ROOT / "MMBench/CPMCM/BestPaper",
    )
    parser.add_argument(
        "--criteria-root",
        type=Path,
        default=REPO_ROOT / "MMBench/CPMCM/criteria",
    )
    parser.add_argument(
        "--prompt-path",
        type=Path,
        default=REPO_ROOT / "eval/prompts/criterial_generate.yaml",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if len(args.judge_model) < 2 or len(args.judge_model) != len(
        set(args.judge_model)
    ):
        raise ValueError("need at least two distinct judge models")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
