#!/usr/bin/env python3
"""Run an auditable cross-model judge agreement experiment.

Unlike the legacy batch script, this runner freezes every report, rubric, and
prompt used by the experiment and writes the exact judge model plus request
metadata next to each result. It is resumable: completed stage results are not
called again.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import os
import random
import re
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from openai import AzureOpenAI, OpenAI


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "MMAgent"))
sys.path.insert(0, str(REPO_ROOT / "eval" / "src"))


REPORT_MODELS = [
    "DeepSeek-V3.2-Instruct",
    "DeepSeek-V3.2-Thinking",
    "Qwen3-235B-A22B-Instruct-2507",
    "o4-mini",
]
TASK_PATTERN = re.compile(r"^(\d{4}_[A-F])(?:_|$)")
PROGRESS_LOCK = threading.Lock()
EXPECTED_STAGES = {
    "problem_analysis",
    "modeling_rigor",
    "practicality_scientificity",
    "result_bias",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def invalid_result_stages(result: dict[str, Any]) -> list[str]:
    invalid: list[str] = []
    for stage in EXPECTED_STAGES:
        stage_result = result.get(stage)
        if not isinstance(stage_result, dict) or not stage_result:
            invalid.append(stage)
            continue
        for item in stage_result.values():
            if (
                not isinstance(item, dict)
                or isinstance(item.get("score"), bool)
                or not isinstance(item.get("score"), (int, float))
            ):
                invalid.append(stage)
                break
    return sorted(set(invalid))


def load_legacy_evaluator():
    path = REPO_ROOT / "eval" / "src" / "3_eval_report_using_mmagent.py"
    spec = importlib.util.spec_from_file_location("legacy_report_evaluator", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载评测模块: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def git_show(relative_path: str) -> bytes:
    result = subprocess.run(
        ["git", "show", f"HEAD:{relative_path}"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        message = result.stderr.decode("utf-8", errors="replace").strip()
        raise FileNotFoundError(f"无法从 Git HEAD 读取 {relative_path}: {message}")
    return result.stdout


def allocate_quotas(total: int, categories: Sequence[str]) -> dict[str, int]:
    base, remainder = divmod(total, len(categories))
    return {
        category: base + (1 if index < remainder else 0)
        for index, category in enumerate(categories)
    }


def report_candidates(report_model: str) -> dict[str, list[Path]]:
    root = REPO_ROOT / "output" / report_model / "CPMCM" / "MM-Agent"
    grouped: dict[str, list[Path]] = {}
    if not root.is_dir():
        return grouped
    for report_path in sorted(root.glob("*/json/*.json")):
        match = TASK_PATTERN.match(report_path.parent.parent.name)
        if not match or report_path.stem != match.group(1):
            continue
        grouped.setdefault(match.group(1), []).append(report_path)
    return grouped


def choose_samples(
    sample_size: int,
    seed: int,
    baseline_results_dir: Path,
) -> list[dict[str, Any]]:
    """Choose balanced report models and unique Task IDs without score-based sampling."""
    quotas = allocate_quotas(sample_size, REPORT_MODELS)
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    used_tasks: set[str] = set()

    for report_model in REPORT_MODELS:
        candidates = [
            (task_id, paths[0])
            for task_id, paths in report_candidates(report_model).items()
            if len(paths) == 1
            and (baseline_results_dir / task_id / f"{report_model}.json").is_file()
        ]
        rng.shuffle(candidates)
        # Stable year-spread: hashed order first, then greedily favour unused years.
        chosen_for_model: list[tuple[str, Path]] = []
        used_years_for_model: set[str] = set()
        for prefer_new_year in (True, False):
            for task_id, report_path in candidates:
                if len(chosen_for_model) >= quotas[report_model]:
                    break
                year = task_id[:4]
                if task_id in used_tasks or (prefer_new_year and year in used_years_for_model):
                    continue
                criteria_path = f"MMBench/CPMCM/criteria/{task_id}.json"
                try:
                    git_show(criteria_path)
                except FileNotFoundError:
                    continue
                chosen_for_model.append((task_id, report_path))
                used_tasks.add(task_id)
                used_years_for_model.add(year)
            if len(chosen_for_model) >= quotas[report_model]:
                break
        if len(chosen_for_model) != quotas[report_model]:
            raise RuntimeError(
                f"{report_model} 只能选出 {len(chosen_for_model)} 份，目标为 {quotas[report_model]}"
            )
        for task_id, report_path in chosen_for_model:
            selected.append(
                {
                    "task_id": task_id,
                    "report_model": report_model,
                    "original_report_path": str(report_path.relative_to(REPO_ROOT)),
                }
            )

    rng.shuffle(selected)
    for index, sample in enumerate(selected, start=1):
        sample["sample_id"] = f"S{index:03d}"
    return selected


def freeze_inputs(
    args: argparse.Namespace,
    public_judge_configs: list[dict[str, str]],
) -> dict[str, Any]:
    manifest_path = args.output_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["configuration"]["sample_size"] != args.sample_size:
            raise ValueError("已有 manifest 的 sample_size 与本次参数不一致")
        if manifest["configuration"]["judge_models"] != public_judge_configs:
            raise ValueError("已有 manifest 的 judge_models 与本次参数不一致")
        if manifest["configuration"]["baseline_judge_model"] != args.baseline_model:
            raise ValueError("已有 manifest 的 baseline_judge_model 与本次参数不一致")
        return manifest

    frozen_root = args.output_dir / "frozen_inputs"
    prompt_target = frozen_root / "prompt" / args.prompt_path.name
    prompt_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.prompt_path, prompt_target)

    if args.source_manifest is not None:
        source_manifest_data = json.loads(
            args.source_manifest.read_text(encoding="utf-8")
        )
        source_samples = source_manifest_data.get("samples", [])
        if len(source_samples) < args.sample_size:
            raise ValueError(
                f"source manifest 只有 {len(source_samples)} 个样本，少于 {args.sample_size}"
            )
        samples = [
            {
                "sample_id": source["sample_id"],
                "task_id": source["task_id"],
                "report_model": source["report_model"],
                "original_report_path": source["original_report_path"],
            }
            for source in source_samples[: args.sample_size]
        ]
        sampling_description = (
            f"first {args.sample_size} frozen samples reused from "
            f"{args.source_manifest.relative_to(REPO_ROOT)}"
        )
    else:
        samples = choose_samples(args.sample_size, args.seed, args.baseline_results_dir)
        sampling_description = (
            "balanced across report models; unique Task IDs; unique source run; "
            "no score-based selection"
        )
    for sample in samples:
        task_id = sample["task_id"]
        report_model = sample["report_model"]
        source = REPO_ROOT / sample["original_report_path"]
        report_target = frozen_root / "reports" / f"{sample['sample_id']}__{report_model}__{task_id}.json"
        report_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, report_target)

        criteria_relative = f"MMBench/CPMCM/criteria/{task_id}.json"
        criteria_target = frozen_root / "criteria" / f"{task_id}.json"
        criteria_target.parent.mkdir(parents=True, exist_ok=True)
        criteria_target.write_bytes(git_show(criteria_relative))

        baseline_source = (
            args.baseline_results_dir / task_id / f"{report_model}.json"
        )
        safe_baseline = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.baseline_model)
        baseline_target = (
            args.output_dir
            / "runs"
            / safe_baseline
            / "results"
            / task_id
            / f"{report_model}.json"
        )
        baseline_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(baseline_source, baseline_target)

        sample.update(
            {
                "frozen_report_path": str(report_target.relative_to(REPO_ROOT)),
                "report_sha256": sha256_file(report_target),
                "frozen_criteria_path": str(criteria_target.relative_to(REPO_ROOT)),
                "criteria_sha256": sha256_file(criteria_target),
                "original_baseline_score_path": str(baseline_source.relative_to(REPO_ROOT)),
                "frozen_baseline_score_path": str(baseline_target.relative_to(REPO_ROOT)),
                "baseline_score_sha256": sha256_file(baseline_target),
            }
        )

    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "sample_size": args.sample_size,
            "seed": args.seed,
            "sampling": sampling_description,
            "report_models": REPORT_MODELS,
            "judge_models": public_judge_configs,
            "baseline_judge_model": args.baseline_model,
            "baseline_provenance": "user-confirmed model identity; frozen from existing results",
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "prompt_path": str(prompt_target.relative_to(REPO_ROOT)),
            "prompt_sha256": sha256_file(prompt_target),
            "criteria_source": "git HEAD:MMBench/CPMCM/criteria/{Task ID}.json",
            "git_commit": git_commit,
            "source_manifest": (
                str(args.source_manifest.relative_to(REPO_ROOT))
                if args.source_manifest is not None
                else None
            ),
        },
        "samples": samples,
    }
    write_json_atomic(manifest, manifest_path)
    return manifest


@dataclass
class AuditedJudge:
    config_name: str
    model: str
    provider: str
    api_key: str
    base_url: str
    api_version: str
    temperature: float
    max_tokens: int
    artifact_dir: Path

    def __post_init__(self) -> None:
        if self.provider == "modelhub":
            self.client = AzureOpenAI(
                api_key=self.api_key,
                azure_endpoint=self.base_url.split("?")[0],
                api_version=self.api_version,
                timeout=1200,
                max_retries=0,
            )
        else:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=1200,
                max_retries=0,
            )
        self.usages: list[dict[str, int]] = []
        self.calls: list[dict[str, Any]] = []
        self.call_index = 0

    def generate(self, prompt: str, system: str = "", usage: bool = True, **_: Any) -> str:
        self.call_index += 1
        call_name = f"call_{self.call_index:02d}"
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        prompt_path = self.artifact_dir / f"{call_name}.prompt.txt"
        prompt_path.write_text(f"SYSTEM\n{system}\n\nUSER\n{prompt}", encoding="utf-8")
        started = datetime.now(timezone.utc)
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        except Exception as exc:
            ended = datetime.now(timezone.utc)
            failure = {
                "call_index": self.call_index,
                "judge_config_name": self.config_name,
                "provider": self.provider,
                "model_requested": self.model,
                "started_at_utc": started.isoformat(),
                "ended_at_utc": ended.isoformat(),
                "duration_seconds": (ended - started).total_seconds(),
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "prompt_path": str(prompt_path.relative_to(REPO_ROOT)),
                "prompt_sha256": sha256_file(prompt_path),
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            self.calls.append(failure)
            write_json_atomic(failure, self.artifact_dir / f"{call_name}.error.json")
            raise
        ended = datetime.now(timezone.utc)
        content = response.choices[0].message.content or ""
        response_path = self.artifact_dir / f"{call_name}.response.txt"
        response_path.write_text(content, encoding="utf-8")

        response_usage = response.usage
        usage_data = {
            "prompt_tokens": int(getattr(response_usage, "prompt_tokens", 0) or 0),
            "completion_tokens": int(getattr(response_usage, "completion_tokens", 0) or 0),
            "total_tokens": int(getattr(response_usage, "total_tokens", 0) or 0),
        }
        if usage:
            self.usages.append(usage_data)
        self.calls.append(
            {
                "call_index": self.call_index,
                "request_id": getattr(response, "id", None),
                "judge_config_name": self.config_name,
                "provider": self.provider,
                "model_requested": self.model,
                "model_returned": getattr(response, "model", None),
                "started_at_utc": started.isoformat(),
                "ended_at_utc": ended.isoformat(),
                "duration_seconds": (ended - started).total_seconds(),
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "prompt_path": str(prompt_path.relative_to(REPO_ROOT)),
                "prompt_sha256": sha256_file(prompt_path),
                "response_path": str(response_path.relative_to(REPO_ROOT)),
                "response_sha256": sha256_file(response_path),
                "usage": usage_data,
            }
        )
        return content

    def get_total_usage(self) -> dict[str, int]:
        keys = {key for item in self.usages for key in item}
        return {key: sum(item.get(key, 0) for item in self.usages) for key in keys}

    def clear_usage(self) -> None:
        self.usages.clear()


def append_progress(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False)
    with PROGRESS_LOCK:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def evaluate_one(
    evaluator,
    sample: dict[str, Any],
    judge_config_name: str,
    judge_config: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    sample_id = sample["sample_id"]
    task_id = sample["task_id"]
    report_model = sample["report_model"]
    judge_model = judge_config["model_name"]
    safe_judge = re.sub(r"[^A-Za-z0-9_.-]+", "_", judge_config_name)
    result_path = args.output_dir / "runs" / safe_judge / "results" / task_id / f"{report_model}.json"
    tmp_path = args.output_dir / "runs" / safe_judge / "tmp" / task_id / f"{report_model}.json"
    metadata_path = args.output_dir / "runs" / safe_judge / "metadata" / task_id / f"{report_model}.json"

    if metadata_path.is_file() and result_path.is_file():
        existing_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        existing_result = json.loads(result_path.read_text(encoding="utf-8"))
        if (
            existing_metadata.get("status") == "completed"
            and existing_metadata.get("result_sha256") == sha256_file(result_path)
            and not invalid_result_stages(existing_result)
        ):
            return existing_metadata

    last_error: Exception | None = None
    for attempt in range(1, args.max_attempts + 1):
        artifact_dir = (
            args.output_dir
            / "runs"
            / safe_judge
            / "artifacts"
            / task_id
            / report_model
            / f"attempt_{attempt}"
        )
        judge = AuditedJudge(
            config_name=judge_config_name,
            model=judge_model,
            provider=judge_config["provider"],
            api_key=judge_config["api_key"],
            base_url=judge_config["base_url"],
            api_version=judge_config["api_version"],
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            artifact_dir=artifact_dir,
        )
        started = datetime.now(timezone.utc)
        try:
            result, usage = evaluator.evaluate_math_modeling(
                judge,
                solution_path=REPO_ROOT / sample["frozen_report_path"],
                prompt_templates_dict=evaluator.load_yaml(
                    args.output_dir / "frozen_inputs" / "prompt" / args.prompt_path.name
                ),
                final_path=result_path,
                tmp_path=tmp_path,
                criteria_dir=args.output_dir / "frozen_inputs" / "criteria",
                system_prompt=evaluator.load_yaml(
                    args.output_dir / "frozen_inputs" / "prompt" / args.prompt_path.name
                )["system_prompt"],
            )
            invalid_stages = invalid_result_stages(result)
            if invalid_stages:
                for cache_path in (result_path, tmp_path):
                    if not cache_path.is_file():
                        continue
                    cached = json.loads(cache_path.read_text(encoding="utf-8"))
                    for stage in invalid_stages:
                        cached.pop(stage, None)
                    write_json_atomic(cached, cache_path)
                raise ValueError(f"阶段结果不是有效评分 JSON: {invalid_stages}")
            metadata = {
                "status": "completed",
                "sample_id": sample_id,
                "task_id": task_id,
                "report_model": report_model,
                "judge_model": judge_model,
                "judge_config_name": judge_config_name,
                "provider": judge_config["provider"],
                "attempt": attempt,
                "started_at_utc": started.isoformat(),
                "ended_at_utc": datetime.now(timezone.utc).isoformat(),
                "source": sample,
                "prompt_sha256": sha256_file(
                    args.output_dir / "frozen_inputs" / "prompt" / args.prompt_path.name
                ),
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
                "usage": usage,
                "calls": judge.calls,
                "result_path": str(result_path.relative_to(REPO_ROOT)),
                "result_sha256": sha256_file(result_path),
            }
            write_json_atomic(metadata, metadata_path)
            return metadata
        except Exception as exc:  # noqa: BLE001 - persisted and retried by design
            last_error = exc
            failure = {
                "status": "retrying" if attempt < args.max_attempts else "failed",
                "sample_id": sample_id,
                "task_id": task_id,
                "report_model": report_model,
                "judge_model": judge_model,
                "judge_config_name": judge_config_name,
                "attempt": attempt,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "calls": judge.calls,
            }
            write_json_atomic(failure, metadata_path)
            if attempt < args.max_attempts:
                time.sleep(min(2**attempt, 10))

    assert last_error is not None
    raise last_error


def run(args: argparse.Namespace) -> int:
    args.output_dir = args.output_dir.resolve()
    args.prompt_path = args.prompt_path.resolve()
    args.api_config = args.api_config.resolve()
    args.baseline_results_dir = args.baseline_results_dir.resolve()
    if args.source_manifest is not None:
        args.source_manifest = args.source_manifest.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw_api_config = json.loads(args.api_config.read_text(encoding="utf-8"))
    judge_configs: dict[str, dict[str, Any]] = {}
    public_judge_configs: list[dict[str, str]] = []
    for config_name in args.judge_model:
        if config_name not in raw_api_config:
            raise KeyError(f"api config 中不存在裁判模型: {config_name}")
        raw = raw_api_config[config_name]
        api_keys = raw.get("api_keys")
        if not isinstance(api_keys, list) or not api_keys or not api_keys[0]:
            raise ValueError(f"{config_name} 没有可用 api_keys")
        if not raw.get("base_url") or not raw.get("model_name"):
            raise ValueError(f"{config_name} 缺少 base_url 或 model_name")
        judge_configs[config_name] = {
            "provider": raw.get("provider", "openai"),
            "base_url": raw["base_url"],
            "api_version": raw.get("api_version", "2024-03-01-preview"),
            "model_name": raw["model_name"],
            "api_key": api_keys[0],
        }
        public_judge_configs.append(
            {
                "config_name": config_name,
                "provider": raw.get("provider", "openai"),
                "model_name": raw["model_name"],
            }
        )

    manifest = freeze_inputs(args, public_judge_configs)
    if args.prepare_only:
        print(f"Prepared {len(manifest['samples'])} samples at {args.output_dir / 'manifest.json'}")
        return 0

    evaluator = load_legacy_evaluator()
    progress_path = args.output_dir / "progress.jsonl"
    selected_samples = (
        manifest["samples"][: args.limit_samples]
        if args.limit_samples is not None
        else manifest["samples"]
    )
    total_jobs = len(selected_samples) * len(args.judge_model)
    completed = 0
    failures: list[dict[str, str]] = []
    counter_lock = threading.Lock()

    def run_one_judge(judge_config_name: str) -> list[dict[str, str]]:
        nonlocal completed
        judge_failures: list[dict[str, str]] = []
        judge_model = judge_configs[judge_config_name]["model_name"]
        for sample in selected_samples:
            try:
                metadata = evaluate_one(
                    evaluator,
                    sample,
                    judge_config_name,
                    judge_configs[judge_config_name],
                    args,
                )
                with counter_lock:
                    completed += 1
                    completed_snapshot = completed
                record = {
                    "time_utc": datetime.now(timezone.utc).isoformat(),
                    "status": "completed",
                    "completed": completed_snapshot,
                    "total": total_jobs,
                    "sample_id": sample["sample_id"],
                    "task_id": sample["task_id"],
                    "report_model": sample["report_model"],
                    "judge_model": judge_model,
                    "judge_config_name": judge_config_name,
                    "usage": metadata["usage"],
                }
            except Exception as exc:  # noqa: BLE001 - batch must finish other jobs
                record = {
                    "time_utc": datetime.now(timezone.utc).isoformat(),
                    "status": "failed",
                    "completed": completed,
                    "total": total_jobs,
                    "sample_id": sample["sample_id"],
                    "task_id": sample["task_id"],
                    "report_model": sample["report_model"],
                    "judge_model": judge_model,
                    "judge_config_name": judge_config_name,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                judge_failures.append(record)
            append_progress(progress_path, record)
            print(json.dumps(record, ensure_ascii=False), flush=True)
        return judge_failures

    with ThreadPoolExecutor(max_workers=min(args.workers, len(args.judge_model))) as pool:
        futures = {
            pool.submit(run_one_judge, judge_config_name): judge_config_name
            for judge_config_name in args.judge_model
        }
        for future in as_completed(futures):
            failures.extend(future.result())

    summary = {
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_jobs": total_jobs,
        "completed_jobs": completed,
        "failed_jobs": len(failures),
        "failures": failures,
    }
    write_json_atomic(summary, args.output_dir / "run_summary.json")
    return 1 if failures else 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行带完整溯源的多裁判一致性实验")
    parser.add_argument("--judge-model", action="append", required=True)
    parser.add_argument("--baseline-model", default="gpt-5-mini")
    parser.add_argument(
        "--baseline-results-dir",
        type=Path,
        default=REPO_ROOT / "eval" / "eval_using_understanding" / "results",
    )
    parser.add_argument("--sample-size", type=int, default=30)
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=None,
        help="复用既有实验 manifest 的前 N 个样本，保证修复前后完全配对",
    )
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument(
        "--limit-samples",
        type=int,
        default=None,
        help="仅运行 manifest 前 N 个样本，用于 smoke test；manifest 本身仍保持完整",
    )
    parser.add_argument(
        "--api-config", type=Path, default=REPO_ROOT / "api-available.json"
    )
    parser.add_argument(
        "--prompt-path",
        type=Path,
        default=REPO_ROOT / "eval" / "prompts" / "mmagent_eval_with_task_understanding.yaml",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)
    if len(args.judge_model) < 2:
        raise ValueError("至少需要两个不同裁判模型")
    if len(args.judge_model) != len(set(args.judge_model)):
        raise ValueError("裁判模型名称必须唯一")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
