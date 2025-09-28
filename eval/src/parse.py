import json
import re
from typing import Dict, List, Set
from collections import defaultdict
from prettytable import PrettyTable
import os

def extract_scores(text: str) -> List[float]:
    """Extract scores from \fbox{} commands in the text."""
    # Look for \fbox{number} patterns
    pattern = r'\\fbox\{([\d.]+)\}'
    scores = re.findall(pattern, text)
    return [float(score) for score in scores]

def parse_jsonl_file(file_path: str) -> Dict[str, Dict[str, List[float]]]:
    """Parse the JSONL file and extract scores by task and dimension."""
    task_scores = defaultdict(dict)

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                task_id = data.get('task_id', '')
                dimension = data.get('eval_dimension', '')
                output = data.get('output', '')

                if task_id and dimension and output:
                    scores = extract_scores(output)
                    if scores:
                        task_scores[task_id][dimension] = scores
            except json.JSONDecodeError:
                continue

    return task_scores

def generate_table(task_scores: Dict[str, Dict[str, List[float]]]) -> PrettyTable:
    """Generate a formatted table from the scores."""
    # Collect all unique dimensions
    all_dimensions = set()
    for scores in task_scores.values():
        all_dimensions.update(scores.keys())
    all_dimensions = sorted(all_dimensions)

    # Create table
    table = PrettyTable()
    table.field_names = ["任务/维度"] + list(all_dimensions)

    # Add rows for each task
    for task_id in sorted(task_scores.keys()):
        row = [task_id]
        for dimension in all_dimensions:
            scores = task_scores[task_id].get(dimension, [])
            score_str = ", ".join(f"{s:.2f}" for s in scores) if scores else "N/A"
            row.append(score_str)
        table.add_row(row)

    return table

def main():
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level to the 'eval' directory
    eval_dir = os.path.dirname(script_dir)
    # Construct the path to the output file
    file_path = os.path.join(eval_dir, 'output/2010_D/1.output_openai_4.1_mini.jsonl')

    # Parse the file
    task_scores = parse_jsonl_file(file_path)

    # Generate and print the table
    if task_scores:
        table = generate_table(task_scores)
        print("各子任务在不同评估维度上的得分：")
        print(table)

        # Also save to a text file
        output_file = os.path.join(eval_dir, 'output/2010_D/evaluation_scores.txt')
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("各子任务在不同评估维度上的得分：\n")
            f.write(str(table))
        print(f"\n结果已保存到: {output_file}")
    else:
        print("未找到有效的评分数据。")

if __name__ == "__main__":
    main()