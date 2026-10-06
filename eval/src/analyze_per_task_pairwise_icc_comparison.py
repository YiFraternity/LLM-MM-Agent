#!/usr/bin/env python3
"""Compare per-task pairwise ICC(2,1) distributions across two rubrics.

The proposed method uses 28 matched targets per task (four subtasks by seven
stages). The legacy static rubric uses eight atomic scores from four generic
rubric calls per task. The source experiments contain different reports and
only partially overlap, so the resulting method comparison is descriptive
rather than paired causal evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from analyze_judge_agreement import agreement_level, intraclass_correlations
from analyze_per_report_subtask_stage import STAGES, extract_cell_scores, load_subtasks


SHORT_JUDGES = ["gpt", "gemini", "deepseek"]
STATIC_JUDGES = {
    "gpt": "gpt-5-mini",
    "gemini": "gemini-3.1-p",
    "deepseek": "ali-deepseek-v4-pro",
}
STATIC_CELLS = [
    ("problem_analysis", "problem_definition_and_goals"),
    ("problem_analysis", "scope_and_coverage"),
    ("modeling_rigor", "assumptions"),
    ("modeling_rigor", "model_rationality"),
    ("practicality_scientificity", "practicality"),
    ("practicality_scientificity", "scientificity"),
    ("result_bias", "result_analysis"),
    ("result_bias", "bias_analysis"),
]
PAIR_ORDER = ["gpt_vs_gemini", "gpt_vs_deepseek", "gemini_vs_deepseek"]
PAIR_LABELS = {
    "gpt_vs_gemini": "GPT--Gemini",
    "gpt_vs_deepseek": "GPT--DeepSeek",
    "gemini_vs_deepseek": "Gemini--DeepSeek",
}
METHOD_ORDER = ["proposed", "static"]
LONG_FIELDS = [
    "method",
    "task_id",
    "report_model",
    "target_count",
    "judge_pair",
    "icc_2_1",
    "icc_c_1",
    "agreement_level",
    "mae",
    "mean_bias",
    "target_mean_sd",
]
SUMMARY_FIELDS = [
    "method",
    "judge_pair",
    "task_count",
    "finite_count",
    "undefined_count",
    "mean",
    "median",
    "q1",
    "q3",
    "minimum",
    "maximum",
    "at_least_good_count",
    "at_least_good_rate",
    "excellent_count",
    "excellent_rate",
]
OVERLAP_FIELDS = [
    "task_id",
    "judge_pair",
    "proposed_report_model",
    "static_report_model",
    "same_report_model",
    "proposed_target_count",
    "static_target_count",
    "proposed_icc_2_1",
    "static_icc_2_1",
    "icc_delta_proposed_minus_static",
    "proposed_mae",
    "static_mae",
]
WIDE_FIELDS = ["method", "task_id", "report_model", "target_count", *PAIR_ORDER]


def calculate_pairwise_rows(
    *,
    method: str,
    task_id: str,
    report_model: str,
    score_matrix: np.ndarray,
    judge_labels: Sequence[str],
) -> list[dict[str, Any]]:
    """Compute the three pairwise ICC(2,1) values for one task."""
    matrix = np.asarray(score_matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 2:
        raise ValueError("score_matrix must contain at least two matched targets")
    if matrix.shape[1] != len(judge_labels) or len(judge_labels) < 2:
        raise ValueError("judge_labels must match score_matrix columns")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("score_matrix contains non-finite values")

    rows: list[dict[str, Any]] = []
    for first, second in combinations(range(len(judge_labels)), 2):
        pair_matrix = matrix[:, [first, second]]
        icc_absolute, icc_consistency = intraclass_correlations(pair_matrix)
        difference = pair_matrix[:, 1] - pair_matrix[:, 0]
        rows.append(
            {
                "method": method,
                "task_id": task_id,
                "report_model": report_model,
                "target_count": int(matrix.shape[0]),
                "judge_pair": f"{judge_labels[first]}_vs_{judge_labels[second]}",
                "icc_2_1": icc_absolute,
                "icc_c_1": icc_consistency,
                "agreement_level": agreement_level(icc_absolute),
                "mae": float(np.mean(np.abs(difference))),
                "mean_bias": float(np.mean(difference)),
                "target_mean_sd": float(np.std(np.mean(pair_matrix, axis=1), ddof=1)),
            }
        )
    return rows


def load_static_task_matrices(experiment_dir: Path) -> list[dict[str, Any]]:
    """Load eight atomic static-rubric scores for all three judges per task."""
    manifest_path = experiment_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    samples = manifest.get("samples")
    if not isinstance(samples, list):
        raise ValueError(f"missing samples list in {manifest_path}")

    loaded: list[dict[str, Any]] = []
    for sample in samples:
        if not isinstance(sample, Mapping):
            raise ValueError(f"invalid sample in {manifest_path}")
        task_id = str(sample["task_id"])
        report_model = str(sample["report_model"])
        judge_columns: dict[str, list[float]] = {}
        for short_judge in SHORT_JUDGES:
            judge = STATIC_JUDGES[short_judge]
            result_path = (
                experiment_dir
                / "runs"
                / judge
                / "results"
                / task_id
                / f"{report_model}.json"
            )
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            column: list[float] = []
            for call_name, criterion_name in STATIC_CELLS:
                try:
                    score = float(payload[call_name][criterion_name]["score"])
                except (KeyError, TypeError, ValueError) as error:
                    raise ValueError(
                        f"missing or invalid static score {call_name}.{criterion_name} "
                        f"in {result_path}"
                    ) from error
                if not math.isfinite(score) or not 1.0 <= score <= 10.0:
                    raise ValueError(
                        f"static score outside [1, 10] for "
                        f"{call_name}.{criterion_name} in {result_path}: {score}"
                    )
                column.append(score)
            judge_columns[short_judge] = column
        matrix = np.asarray(
            [
                [judge_columns[judge][cell_index] for judge in SHORT_JUDGES]
                for cell_index in range(len(STATIC_CELLS))
            ],
            dtype=float,
        )
        loaded.append(
            {
                "task_id": task_id,
                "report_model": report_model,
                "score_matrix": matrix,
            }
        )
    return loaded


def summarize_distributions(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Summarize per-task ICC distributions separately by method and judge pair."""
    summaries: list[dict[str, Any]] = []
    present_groups = {(str(row["method"]), str(row["judge_pair"])) for row in rows}
    for method in METHOD_ORDER:
        for judge_pair in PAIR_ORDER:
            if (method, judge_pair) not in present_groups:
                continue
            group = [
                row
                for row in rows
                if row["method"] == method and row["judge_pair"] == judge_pair
            ]
            finite = np.asarray(
                [
                    float(row["icc_2_1"])
                    for row in group
                    if row.get("icc_2_1") is not None
                    and math.isfinite(float(row["icc_2_1"]))
                ],
                dtype=float,
            )
            finite_count = int(finite.size)
            summary: dict[str, Any] = {
                "method": method,
                "judge_pair": judge_pair,
                "task_count": len(group),
                "finite_count": finite_count,
                "undefined_count": len(group) - finite_count,
                "mean": float(np.mean(finite)) if finite_count else None,
                "median": float(np.median(finite)) if finite_count else None,
                "q1": float(np.quantile(finite, 0.25)) if finite_count else None,
                "q3": float(np.quantile(finite, 0.75)) if finite_count else None,
                "minimum": float(np.min(finite)) if finite_count else None,
                "maximum": float(np.max(finite)) if finite_count else None,
                "at_least_good_count": int(np.sum(finite >= 0.75)),
                "at_least_good_rate": (
                    float(np.mean(finite >= 0.75)) if finite_count else None
                ),
                "excellent_count": int(np.sum(finite >= 0.90)),
                "excellent_rate": (
                    float(np.mean(finite >= 0.90)) if finite_count else None
                ),
            }
            summaries.append(summary)
    return summaries


def build_overlap_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Align method results on shared Task-ID and judge-pair keys."""
    index = {
        (str(row["method"]), str(row["task_id"]), str(row["judge_pair"])): row
        for row in rows
    }
    proposed_keys = {
        (task_id, judge_pair)
        for method, task_id, judge_pair in index
        if method == "proposed"
    }
    static_keys = {
        (task_id, judge_pair)
        for method, task_id, judge_pair in index
        if method == "static"
    }
    overlap: list[dict[str, Any]] = []
    for task_id, judge_pair in sorted(
        proposed_keys & static_keys,
        key=lambda key: (key[0], PAIR_ORDER.index(key[1])),
    ):
        proposed = index[("proposed", task_id, judge_pair)]
        static = index[("static", task_id, judge_pair)]
        proposed_icc = float(proposed["icc_2_1"])
        static_icc = float(static["icc_2_1"])
        delta = (
            proposed_icc - static_icc
            if math.isfinite(proposed_icc) and math.isfinite(static_icc)
            else math.nan
        )
        overlap.append(
            {
                "task_id": task_id,
                "judge_pair": judge_pair,
                "proposed_report_model": proposed["report_model"],
                "static_report_model": static["report_model"],
                "same_report_model": proposed["report_model"] == static["report_model"],
                "proposed_target_count": proposed["target_count"],
                "static_target_count": static["target_count"],
                "proposed_icc_2_1": proposed_icc,
                "static_icc_2_1": static_icc,
                "icc_delta_proposed_minus_static": delta,
                "proposed_mae": proposed.get("mae"),
                "static_mae": static.get("mae"),
            }
        )
    return overlap


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _wide_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Put all three pairwise ICC(2,1) estimates for a task on one row."""
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["method"]), str(row["task_id"]))
        wide = indexed.setdefault(
            key,
            {
                "method": row["method"],
                "task_id": row["task_id"],
                "report_model": row["report_model"],
                "target_count": row["target_count"],
            },
        )
        wide[str(row["judge_pair"])] = row["icc_2_1"]
    return sorted(
        indexed.values(),
        key=lambda row: (METHOD_ORDER.index(str(row["method"])), str(row["task_id"])),
    )


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _write_distribution_plot(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    finite_values = [
        float(row["icc_2_1"])
        for row in rows
        if row.get("icc_2_1") is not None and math.isfinite(float(row["icc_2_1"]))
    ]
    observed_min = min(finite_values, default=0.0)
    y_min = min(0.0, math.floor((observed_min - 0.05) * 2.0) / 2.0)
    y_max = 1.0
    width, height = 1600, 900
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = _font(34, bold=True)
    label_font = _font(21)
    small_font = _font(17)
    draw.text((55, 28), "Per-task pairwise ICC(2,1) distributions", fill="#172033", font=title_font)
    draw.text(
        (55, 76),
        "Proposed: 28 subtask-stage targets/task; Static: 8 atomic rubric scores/task",
        fill="#526079",
        font=label_font,
    )

    left, top, right, bottom = 110, 145, width - 70, 720

    def y_position(value: float) -> float:
        return bottom - (value - y_min) * (bottom - top) / (y_max - y_min)

    for threshold, color in [(0.0, "#94a3b8"), (0.75, "#2563eb"), (0.90, "#059669")]:
        y = y_position(threshold)
        draw.line((left, y, right, y), fill=color, width=2)
        draw.text((right + 8, y - 10), f"{threshold:.2f}", fill=color, font=small_font)
    draw.line((left, top, left, bottom), fill="#334155", width=2)
    draw.line((left, bottom, right, bottom), fill="#334155", width=2)

    group_positions: dict[tuple[str, str], float] = {}
    panel_width = (right - left) / len(PAIR_ORDER)
    for pair_index, judge_pair in enumerate(PAIR_ORDER):
        center = left + panel_width * (pair_index + 0.5)
        proposed_x = center - panel_width * 0.17
        static_x = center + panel_width * 0.17
        group_positions[("proposed", judge_pair)] = proposed_x
        group_positions[("static", judge_pair)] = static_x
        pair_label = PAIR_LABELS[judge_pair]
        label_box = draw.textbbox((0, 0), pair_label, font=label_font)
        draw.text(
            (center - (label_box[2] - label_box[0]) / 2, bottom + 75),
            pair_label,
            fill="#172033",
            font=label_font,
        )
        draw.text((proposed_x - 45, bottom + 26), "Proposed", fill="#2563eb", font=small_font)
        draw.text((static_x - 28, bottom + 26), "Static", fill="#d97706", font=small_font)
        if pair_index:
            divider_x = left + panel_width * pair_index
            draw.line((divider_x, top, divider_x, bottom + 115), fill="#e2e8f0", width=2)

    for method in METHOD_ORDER:
        for judge_pair in PAIR_ORDER:
            group = [
                row
                for row in rows
                if row["method"] == method
                and row["judge_pair"] == judge_pair
                and row.get("icc_2_1") is not None
                and math.isfinite(float(row["icc_2_1"]))
            ]
            values = [float(row["icc_2_1"]) for row in group]
            x_center = group_positions[(method, judge_pair)]
            color = "#2563eb" if method == "proposed" else "#d97706"
            for index, value in enumerate(sorted(values)):
                jitter = ((index % 7) - 3) * 6
                y = y_position(value)
                draw.ellipse(
                    (x_center + jitter - 5, y - 5, x_center + jitter + 5, y + 5),
                    fill=color,
                    outline="white",
                    width=1,
                )
            if values:
                median = float(np.median(values))
                y = y_position(median)
                draw.line((x_center - 40, y, x_center + 40, y), fill="#111827", width=5)
                draw.text((x_center - 38, y - 27), f"{median:.2f}", fill="#111827", font=small_font)

    draw.text((left, bottom + 135), "Each dot is one Task ID; black bars show medians.", fill="#526079", font=label_font)
    image.save(path, format="PNG")


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None or not math.isfinite(float(value)):
        return "NA"
    return f"{float(value):.{digits}f}"


def write_outputs(
    rows: Sequence[Mapping[str, Any]],
    summaries: Sequence[Mapping[str, Any]],
    overlap_rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "per_task_pairwise_icc.csv", rows, LONG_FIELDS)
    _write_csv(output_dir / "per_task_pairwise_icc_wide.csv", _wide_rows(rows), WIDE_FIELDS)
    _write_csv(output_dir / "distribution_summary.csv", summaries, SUMMARY_FIELDS)
    _write_csv(output_dir / "overlap_pairwise_comparison.csv", overlap_rows, OVERLAP_FIELDS)
    (output_dir / "distribution_summary.json").write_text(
        json.dumps(list(summaries), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Per-task pairwise ICC(2,1) comparison",
        "",
        "For each Task ID, ICC(2,1) is computed separately for GPT--Gemini, "
        "GPT--DeepSeek, and Gemini--DeepSeek. The proposed method uses 28 "
        "subtask-by-stage targets per task; the static rubric uses eight atomic "
        "scores returned by four generic-rubric calls per task.",
        "",
        "## Distribution summary",
        "",
        "| Method | Judge pair | Tasks | Median | IQR | Range | ICC>=0.75 | ICC>=0.90 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        lines.append(
            f"| {summary['method']} | {summary['judge_pair']} | {summary['task_count']} | "
            f"{_fmt(summary['median'])} | [{_fmt(summary['q1'])}, {_fmt(summary['q3'])}] | "
            f"[{_fmt(summary['minimum'])}, {_fmt(summary['maximum'])}] | "
            f"{summary['at_least_good_count']}/{summary['finite_count']} | "
            f"{summary['excellent_count']}/{summary['finite_count']} |"
        )

    overlap_task_count = len({row["task_id"] for row in overlap_rows})
    same_report_count = len(
        {
            row["task_id"]
            for row in overlap_rows
            if bool(row["same_report_model"])
        }
    )
    lines.extend(
        [
            "",
            "## Comparison scope",
            "",
            f"The proposed experiment contains 23 tasks and the static experiment "
            f"contains 30 tasks. They share {overlap_task_count} Task IDs. Among "
            f"those shared IDs, {same_report_count} use the same report-generation "
            "model; the report contents and scoring targets are not matched. The "
            "method comparison is therefore descriptive and must not be interpreted "
            "as a paired causal estimate of rubric quality.",
            "",
            "The static per-task ICC values are based on only eight targets, so they "
            "are intrinsically more variable than the proposed estimates based on "
            "28 targets. This target-count difference is part of the evaluation "
            "design and also limits direct numerical comparability.",
            "",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    _write_distribution_plot(rows, output_dir / "pairwise_icc_distributions.png")


def _load_proposed_rows(repo_root: Path, experiment_dir: Path) -> list[dict[str, Any]]:
    manifest = json.loads((experiment_dir / "manifest.json").read_text(encoding="utf-8"))
    judge_roots = {
        "gpt": repo_root / "eval/eval_using_ourcriteria/results/gemini-2.5-flash-priority_MM-Agent-criteria",
        "gemini": experiment_dir / "runs/gemini-3.1-p/results",
        "deepseek": experiment_dir / "runs/ali-deepseek-v4-pro/results",
    }
    rows: list[dict[str, Any]] = []
    for report in manifest["reports"]:
        task_id = str(report["task_id"])
        subtask_ids = [str(index) for index in range(1, int(report["subtask_count"]) + 1)]
        paths = {
            "gpt": judge_roots["gpt"] / task_id / "gemini-2.5-flash-priority_MM-Agent-criteria.json",
            "gemini": judge_roots["gemini"] / f"{task_id}.json",
            "deepseek": judge_roots["deepseek"] / f"{task_id}.json",
        }
        judge_cells = {
            judge: extract_cell_scores(
                load_subtasks(path), subtask_ids=subtask_ids, stages=STAGES
            )
            for judge, path in paths.items()
        }
        cell_keys = [(subtask_id, stage) for subtask_id in subtask_ids for stage in STAGES]
        matrix = np.asarray(
            [[judge_cells[judge][cell] for judge in SHORT_JUDGES] for cell in cell_keys],
            dtype=float,
        )
        rows.extend(
            calculate_pairwise_rows(
                method="proposed",
                task_id=task_id,
                report_model=str(report["report_model"]),
                score_matrix=matrix,
                judge_labels=SHORT_JUDGES,
            )
        )
    return rows


def run_analysis(
    *,
    repo_root: Path,
    proposed_experiment_dir: Path,
    static_experiment_dir: Path,
    output_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rows = _load_proposed_rows(repo_root, proposed_experiment_dir)
    for task in load_static_task_matrices(static_experiment_dir):
        rows.extend(
            calculate_pairwise_rows(
                method="static",
                task_id=task["task_id"],
                report_model=task["report_model"],
                score_matrix=task["score_matrix"],
                judge_labels=SHORT_JUDGES,
            )
        )
    summaries = summarize_distributions(rows)
    overlap_rows = build_overlap_rows(rows)
    write_outputs(rows, summaries, overlap_rows, output_dir)
    return rows, summaries, overlap_rows


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    proposed = repo_root / "eval/judge_agreement/cross_judge_true_method_23"
    static = repo_root / "eval/judge_agreement/cross_judge_30_gemini31_deepseekv4"
    parser = argparse.ArgumentParser(
        description="Compare per-task pairwise ICC(2,1) distributions for proposed and static rubrics"
    )
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--proposed-experiment-dir", type=Path, default=proposed)
    parser.add_argument(
        "--static-experiment-dir",
        type=Path,
        default=static,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=proposed / "per_task_pairwise_static_comparison",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    rows, summaries, overlap = run_analysis(
        repo_root=args.repo_root.resolve(),
        proposed_experiment_dir=args.proposed_experiment_dir.resolve(),
        static_experiment_dir=args.static_experiment_dir.resolve(),
        output_dir=args.output_dir.resolve(),
    )
    print(f"Per-task pairwise rows: {len(rows)}")
    print(f"Distribution groups: {len(summaries)}")
    print(f"Overlapping task-pair rows: {len(overlap)}")
    print(f"Output: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
