import argparse
from pathlib import Path

import pandas as pd

from eval_utils import (
    find_task_id_from_path,
    load_json,
)


def _process_file(json_path: Path | str, output_excel: Path | str | None = None):
    if json_path is None:
        raise ValueError("必须提供 json_path 参数。")

    data = load_json(json_path)
    sections = ["问题识别", "问题复述", "假设建立", "模型构建", "模型求解", "代码实现", "结果分析"]

    results = {}
    try:
        for sub_id, content in data.items():
            results[sub_id] = {}
            for section in sections:
                entries = content.get(section, [])
                total_score = sum(entry.get("score", 0) for entry in entries)
                results[sub_id][section] = total_score

        df = pd.DataFrame(results).T
        df.index.name = "子问题"

        if output_excel is None:
            output_excel = Path(json_path).with_suffix(".xlsx")
        else:
            output_excel = Path(output_excel)

        output_excel.parent.mkdir(parents=True, exist_ok=True)
        df.to_excel(output_excel, sheet_name="scores")

        for sub_id, section_scores in results.items():
            print(f"子问题{sub_id}:")
            for section, score in section_scores.items():
                print(f"  {section}: {score}")
            print()

        print(f"评分结果已写入 Excel 文件: {output_excel}")
    except:
        task_id = find_task_id_from_path(json_path)
        return task_id


def main(input_path: Path | str | None = None, output_path: Path | str | None = None):
    if input_path is None:
        raise ValueError("必须提供 input_path 参数。")

    input_path = Path(input_path)

    errors = []
    if input_path.is_dir():
        if output_path is None:
            raise ValueError("处理目录时必须提供输出目录 output_path。")
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        for task_id in input_path.iterdir():
            for json_file in sorted(task_id.glob("*.json")):
                target_excel = output_dir / task_id.name / f"{json_file.stem}.xlsx"
                error = _process_file(json_file, target_excel)
                errors.append(error)
    else:
        _process_file(input_path, output_path)
    errors = [_ for _ in errors if _]
    print(f'⚠️  errors task ids: {errors}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="汇总评分并导出为 Excel。")
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="输入 JSON 文件或包含多个 JSON 的目录。",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="输出 Excel 文件路径；若输入为目录，则应提供输出目录。",
    )

    args = parser.parse_args()
    main(args.input, args.output)
