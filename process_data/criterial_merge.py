import argparse
import json
from pathlib import Path

from utils import (
    load_jsonl_file,
    clean_json_txt,
)


def merge_outputs(file_path: str, output_file: str) -> None:
    """
    合并多个 JSONL 文件中的 output 内容，生成合并后的 JSON 文件。

    Args:
        file_path (str): 包含 JSONL 文件的目录路径。
        output_file (str): 合并后的 JSON 文件路径。
    """
    jsonl_data = load_jsonl_file(file_path)
    criteria = {}
    for entry in jsonl_data:
        subtask = entry.get("subtask", "")
        output = entry.get("output", "")
        number_of_subtask = entry.get("number_of_subtask", 0)
        dimension = entry.get("eval_dimension", "")
        subtask_id = f"subtask_{number_of_subtask}"
        if subtask_id not in criteria:
            criteria[subtask_id] = {
                "subtask": subtask,
                "criteria": {}
            }
        output = clean_json_txt(output)
        criteria[subtask_id]["criteria"][dimension] = output.get("evaluation_criteria", {})

    # 保存合并后的 JSON 文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(criteria, f, ensure_ascii=False, indent=2)
    print(f"合并完成，结果已保存到 {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate criteria and ensure placeholders for problems.')
    parser.add_argument('--criteria2', type=Path, default=Path('MMBench/CPMCM/criteria_2'), help='Output directory for phase 2 results')
    parser.add_argument('--criteria-root', type=Path, default=Path('MMBench/CPMCM/criteria'), help='Directory to ensure criteria placeholders exist')
    args = parser.parse_args()
    # 合并输出
    task_ids = args.criteria2.iterdir()
    for file in task_ids:
        task_id = file.stem
        merge_outputs(file, args.criteria_root/f"{task_id}.json")
