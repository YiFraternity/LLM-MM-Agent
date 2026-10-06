#!/usr/bin/env python3
"""Measure cross-judge agreement separately within each report.

Each report contributes 28 matched targets (four subtasks by seven stages).
The script computes ICC(2,1) across the three judges for each report, then
summarizes the distribution of the 23 report-level ICC estimates. It does not
pool the 644 cells into one ICC estimate.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from analyze_judge_agreement import agreement_level, intraclass_correlations, kendalls_w


STAGES = ["问题识别", "问题复述", "假设建立", "模型构建", "模型求解", "代码实现", "结果分析"]
JUDGE_LABELS = ["gpt", "gemini", "deepseek"]
CSV_FIELDS = [
    "task_id",
    "cell_count",
    "icc_2_1",
    "icc_c_1",
    "kendall_w",
    "agreement_level",
    "mean_pairwise_mae",
    "max_pairwise_mae",
    "target_mean_sd",
    "target_mean_range",
    "mean_cell_range",
    "max_cell_range",
    "gpt_vs_gemini_mae",
    "gpt_vs_deepseek_mae",
    "gemini_vs_deepseek_mae",
    "gpt_vs_gemini_bias",
    "gpt_vs_deepseek_bias",
    "gemini_vs_deepseek_bias",
]


def load_subtasks(path: Path) -> dict[str, Any]:
    """Load either a legacy root-level subtask mapping or a wrapped run result."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"result is not an object: {path}")
    subtasks = payload.get("subtasks", payload)
    if not isinstance(subtasks, dict):
        raise ValueError(f"subtasks is not an object: {path}")
    return subtasks


def extract_cell_scores(
    subtasks: Mapping[str, Any],
    *,
    subtask_ids: Sequence[str],
    stages: Sequence[str],
) -> dict[tuple[str, str], float]:
    """Return normalized 0--10 scores for every requested subtask-stage cell."""
    cells: dict[tuple[str, str], float] = {}
    for subtask_id in subtask_ids:
        subtask = subtasks.get(subtask_id)
        if not isinstance(subtask, dict):
            raise ValueError(f"missing subtask: {subtask_id}")
        for stage in stages:
            items = subtask.get(stage)
            if not isinstance(items, list) or not items:
                raise ValueError(f"missing cell: subtask={subtask_id}, stage={stage}")
            scores: list[float] = []
            for item in items:
                if not isinstance(item, dict) or isinstance(item.get("score"), bool):
                    raise ValueError(
                        f"invalid score item: subtask={subtask_id}, stage={stage}"
                    )
                try:
                    score = float(item["score"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(
                        f"invalid score item: subtask={subtask_id}, stage={stage}"
                    ) from exc
                if not math.isfinite(score):
                    raise ValueError(
                        f"non-finite score: subtask={subtask_id}, stage={stage}"
                    )
                scores.append(score)
            cells[(subtask_id, stage)] = sum(scores) / 10.0
    return cells


def _safe_label(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def calculate_report_metrics(
    task_id: str,
    score_matrix: np.ndarray,
    judge_labels: Sequence[str],
) -> dict[str, Any]:
    """Calculate ICC and complementary error diagnostics for one report."""
    matrix = np.asarray(score_matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 2:
        raise ValueError("score_matrix must contain at least two matched targets")
    if matrix.shape[1] != len(judge_labels) or len(judge_labels) < 2:
        raise ValueError("judge_labels must match score_matrix columns")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("score_matrix contains non-finite values")

    icc_absolute, icc_consistency = intraclass_correlations(matrix)
    pair_maes: list[float] = []
    metrics: dict[str, Any] = {
        "task_id": task_id,
        "cell_count": int(matrix.shape[0]),
        "icc_2_1": icc_absolute,
        "icc_c_1": icc_consistency,
        "kendall_w": kendalls_w(matrix),
        "agreement_level": agreement_level(icc_absolute),
    }
    for first, second in combinations(range(len(judge_labels)), 2):
        difference = matrix[:, second] - matrix[:, first]
        pair_name = f"{_safe_label(judge_labels[first])}_vs_{_safe_label(judge_labels[second])}"
        mae = float(np.mean(np.abs(difference)))
        metrics[f"{pair_name}_mae"] = mae
        metrics[f"{pair_name}_bias"] = float(np.mean(difference))
        pair_maes.append(mae)

    cell_ranges = np.ptp(matrix, axis=1)
    target_means = np.mean(matrix, axis=1)
    metrics["mean_pairwise_mae"] = float(np.mean(pair_maes))
    metrics["max_pairwise_mae"] = float(np.max(pair_maes))
    metrics["target_mean_sd"] = float(np.std(target_means, ddof=1))
    metrics["target_mean_range"] = float(np.ptp(target_means))
    metrics["mean_cell_range"] = float(np.mean(cell_ranges))
    metrics["max_cell_range"] = float(np.max(cell_ranges))
    return metrics


def build_distribution_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize the distribution of per-report ICC(2,1) point estimates."""
    def summarize_field(field: str) -> dict[str, float | int | None]:
        field_values = np.asarray(
            [
                float(row[field])
                for row in rows
                if row.get(field) is not None and math.isfinite(float(row[field]))
            ],
            dtype=float,
        )
        if field_values.size == 0:
            return {
                "count": 0,
                "mean": None,
                "median": None,
                "q1": None,
                "q3": None,
                "minimum": None,
                "maximum": None,
            }
        return {
            "count": int(field_values.size),
            "mean": float(np.mean(field_values)),
            "median": float(np.median(field_values)),
            "q1": float(np.quantile(field_values, 0.25)),
            "q3": float(np.quantile(field_values, 0.75)),
            "minimum": float(np.min(field_values)),
            "maximum": float(np.max(field_values)),
        }

    finite_rows = [
        row
        for row in rows
        if row.get("icc_2_1") is not None
        and math.isfinite(float(row["icc_2_1"]))
    ]
    values = np.asarray([float(row["icc_2_1"]) for row in finite_rows], dtype=float)
    bins = {
        "<0.50": int(np.sum(values < 0.50)),
        "0.50-<0.75": int(np.sum((values >= 0.50) & (values < 0.75))),
        "0.75-<0.90": int(np.sum((values >= 0.75) & (values < 0.90))),
        ">=0.90": int(np.sum(values >= 0.90)),
    }
    summary: dict[str, Any] = {
        "report_count": len(rows),
        "finite_icc_count": len(finite_rows),
        "undefined_icc_count": len(rows) - len(finite_rows),
        "bins": bins,
        "bin_proportions": {
            key: (count / len(finite_rows) if finite_rows else None)
            for key, count in bins.items()
        },
        "mean_pairwise_mae": summarize_field("mean_pairwise_mae"),
        "max_cell_range": summarize_field("max_cell_range"),
    }
    if not finite_rows:
        summary.update(
            {
                "mean": None,
                "median": None,
                "q1": None,
                "q3": None,
                "minimum": None,
                "maximum": None,
                "lowest_reports": [],
            }
        )
        return summary

    summary.update(
        {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "q1": float(np.quantile(values, 0.25)),
            "q3": float(np.quantile(values, 0.75)),
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
            "lowest_reports": [
                {
                    "task_id": str(row["task_id"]),
                    "icc_2_1": float(row["icc_2_1"]),
                    "mean_pairwise_mae": (
                        float(row["mean_pairwise_mae"])
                        if row.get("mean_pairwise_mae") is not None
                        else None
                    ),
                    "max_cell_range": (
                        float(row["max_cell_range"])
                        if row.get("max_cell_range") is not None
                        else None
                    ),
                }
                for row in sorted(finite_rows, key=lambda item: float(item["icc_2_1"]))[:5]
            ],
        }
    )
    return summary


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    names = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    for name in names:
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _write_distribution_plot(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    finite_rows = sorted(
        (
            row
            for row in rows
            if row.get("icc_2_1") is not None
            and math.isfinite(float(row["icc_2_1"]))
        ),
        key=lambda row: float(row["icc_2_1"]),
    )
    width, height = 1400, 820
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = _font(34, bold=True)
    label_font = _font(22)
    small_font = _font(18)
    draw.text((60, 30), "Per-report cross-judge ICC(2,1)", fill="#172033", font=title_font)
    draw.text(
        (60, 78),
        "Each point uses 28 matched subtask-by-stage cells within one report",
        fill="#526079",
        font=label_font,
    )

    left, top, right, bottom = 105, 145, width - 65, 650
    values = [float(row["icc_2_1"]) for row in finite_rows]
    observed_min = min(values, default=0.0)
    y_min = min(0.0, math.floor((observed_min - 0.05) * 10.0) / 10.0)
    y_max = 1.0
    if y_min == y_max:
        y_min = y_max - 1.0

    def x_position(index: int) -> float:
        if len(values) <= 1:
            return (left + right) / 2
        return left + index * (right - left) / (len(values) - 1)

    def y_position(value: float) -> float:
        return bottom - (value - y_min) * (bottom - top) / (y_max - y_min)

    for threshold, color, label in [
        (0.50, "#d97706", "0.50"),
        (0.75, "#2563eb", "0.75"),
        (0.90, "#059669", "0.90"),
    ]:
        y = y_position(threshold)
        draw.line((left, y, right, y), fill=color, width=2)
        draw.text((right + 8, y - 11), label, fill=color, font=small_font)

    draw.line((left, top, left, bottom), fill="#334155", width=2)
    draw.line((left, bottom, right, bottom), fill="#334155", width=2)
    tick_start = math.floor(y_min * 10) / 10
    for tick in np.arange(tick_start, 1.01, 0.10):
        y = y_position(float(tick))
        draw.line((left - 6, y, left, y), fill="#334155", width=2)
        draw.text((35, y - 10), f"{tick:.1f}", fill="#334155", font=small_font)

    for index, row in enumerate(finite_rows):
        value = float(row["icc_2_1"])
        x = x_position(index)
        y = y_position(value)
        color = "#059669" if value >= 0.90 else "#2563eb" if value >= 0.75 else "#d97706" if value >= 0.50 else "#dc2626"
        draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=color, outline="white", width=2)
        draw.text((x - 18, bottom + 14), str(index + 1), fill="#64748b", font=small_font)

    draw.text((left, bottom + 55), "Reports sorted by ICC (lowest to highest)", fill="#334155", font=label_font)
    bin_text = "   ".join(
        [
            f"Poor <0.50: {sum(value < 0.50 for value in values)}",
            f"Moderate: {sum(0.50 <= value < 0.75 for value in values)}",
            f"Good: {sum(0.75 <= value < 0.90 for value in values)}",
            f"Excellent >=0.90: {sum(value >= 0.90 for value in values)}",
        ]
    )
    draw.text((left, bottom + 105), bin_text, fill="#172033", font=label_font)
    image.save(path, format="PNG")


def _format_number(value: Any, digits: int = 3) -> str:
    if value is None or not math.isfinite(float(value)):
        return "NA"
    return f"{float(value):.{digits}f}"


def write_analysis_outputs(
    rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    output_dir: Path,
) -> None:
    """Write the per-report table, distribution summary, report, and PNG plot."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "per_report_metrics.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    (output_dir / "distribution_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    bin_proportions = summary["bin_proportions"]
    lines = [
        "# Per-report subtask-by-stage agreement",
        "",
        "Each report is analyzed separately using its 28 matched targets "
        "(4 subtasks $\\times$ 7 stages) and 3 judge columns. The analysis "
        "summarizes the distribution of report-specific ICC(2,1) estimates; "
        "it does not pool all 644 cells into a single ICC.",
        "",
        "## ICC(2,1) distribution",
        "",
        f"- Reports: {summary['report_count']}",
        f"- Finite ICC estimates: {summary['finite_icc_count']}",
        f"- Median: {_format_number(summary.get('median'))}",
        f"- IQR: [{_format_number(summary.get('q1'))}, {_format_number(summary.get('q3'))}]",
        f"- Range: [{_format_number(summary.get('minimum'))}, {_format_number(summary.get('maximum'))}]",
        f"- Median mean-pairwise MAE: {_format_number(summary['mean_pairwise_mae']['median'])} "
        f"(IQR [{_format_number(summary['mean_pairwise_mae']['q1'])}, "
        f"{_format_number(summary['mean_pairwise_mae']['q3'])}])",
        f"- Median maximum cell range: {_format_number(summary['max_cell_range']['median'])}",
        "",
        "| ICC band | Reports | Share of finite estimates |",
        "|---|---:|---:|",
    ]
    for label in ["<0.50", "0.50-<0.75", "0.75-<0.90", ">=0.90"]:
        proportion = bin_proportions[label]
        lines.append(
            f"| {label} | {summary['bins'][label]} | "
            f"{(100.0 * proportion if proportion is not None else math.nan):.1f}\\% |"
        )

    lines.extend(
        [
            "",
            "## Per-report results",
            "",
            "| Task ID | ICC(2,1) | ICC(C,1) | Level | Mean pairwise MAE | Max cell range |",
            "|---|---:|---:|---|---:|---:|",
        ]
    )
    for row in sorted(
        rows,
        key=lambda item: (
            not math.isfinite(float(item["icc_2_1"])),
            float(item["icc_2_1"]) if math.isfinite(float(item["icc_2_1"])) else math.inf,
        ),
    ):
        lines.append(
            f"| {row['task_id']} | {_format_number(row['icc_2_1'])} | "
            f"{_format_number(row['icc_c_1'])} | {row['agreement_level']} | "
            f"{_format_number(row['mean_pairwise_mae'])} | "
            f"{_format_number(row['max_cell_range'])} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation note",
            "",
            "A report-specific ICC depends on the spread of the 28 target scores. "
            "When a report receives a compressed score range, ICC can be low even "
            "when absolute score differences are modest. The ICC distribution "
            "should therefore be interpreted together with pairwise MAE and the "
            "maximum within-cell judge range.",
            "",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    _write_distribution_plot(rows, output_dir / "icc_distribution.png")


def _default_paths(repo_root: Path, experiment_dir: Path) -> dict[str, Path]:
    return {
        "gpt": repo_root / "eval/eval_using_ourcriteria/results/gemini-2.5-flash-priority_MM-Agent-criteria",
        "gemini": experiment_dir / "runs/gemini-3.1-p/results",
        "deepseek": experiment_dir / "runs/ali-deepseek-v4-pro/results",
    }


def run_analysis(
    *,
    repo_root: Path,
    experiment_dir: Path,
    output_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = json.loads((experiment_dir / "manifest.json").read_text(encoding="utf-8"))
    judge_roots = _default_paths(repo_root, experiment_dir)
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
            label: extract_cell_scores(
                load_subtasks(path), subtask_ids=subtask_ids, stages=STAGES
            )
            for label, path in paths.items()
        }
        cell_keys = [(subtask_id, stage) for subtask_id in subtask_ids for stage in STAGES]
        matrix = np.asarray(
            [
                [judge_cells[label][cell_key] for label in JUDGE_LABELS]
                for cell_key in cell_keys
            ],
            dtype=float,
        )
        rows.append(calculate_report_metrics(task_id, matrix, JUDGE_LABELS))

    summary = build_distribution_summary(rows)
    write_analysis_outputs(rows, summary, output_dir)
    return rows, summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    default_experiment = repo_root / "eval/judge_agreement/cross_judge_true_method_23"
    parser = argparse.ArgumentParser(
        description="Compute a distribution of report-specific subtask-by-stage ICC(2,1) estimates"
    )
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--experiment-dir", type=Path, default=default_experiment)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_experiment / "per_report_subtask_stage",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    rows, summary = run_analysis(
        repo_root=args.repo_root.resolve(),
        experiment_dir=args.experiment_dir.resolve(),
        output_dir=args.output_dir.resolve(),
    )
    print(f"Reports analyzed: {len(rows)}")
    print(f"Median per-report ICC(2,1): {_format_number(summary['median'])}")
    print(
        "IQR: "
        f"[{_format_number(summary['q1'])}, {_format_number(summary['q3'])}]"
    )
    print(f"Output: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
