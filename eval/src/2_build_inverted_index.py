# Description: 构建倒排索引
#

import json
from collections import defaultdict
from typing import Dict, List, Union
import re

from utils import(
    clean_json_txt,
)

def extract_subtasks_dimensions(task_dimension: Union[Dict[str, str], List[Dict[str, str]]]) -> tuple:
    """Extract subtasks and dimensions from the task_dimension."""
    if isinstance(task_dimension, list):
        for dim in task_dimension:
            subtask = dim.get('subtask', '')
            if '子任务' in subtask or 'subtask' in subtask.lower() or re.search(r'\d', subtask):
                return (
                    subtask,
                    dim.get('dimensions', '')
                )
    elif isinstance(task_dimension, dict):
        subtask = task_dimension.get('subtask', '')
        if '子任务' in subtask or 'subtask' in subtask.lower() or re.search(r'\d', subtask):
            return (
                subtask,
                task_dimension.get('dimensions', '')
            )
    return None


def build_inverted_index(input_file: str) -> Dict[str, List[Dict]]:
    """
    Build an inverted index from the section classification output.

    Args:
        input_file: Path to the JSONL file containing section classification output

    Returns:
        Dict where keys are subtasks/dimensions and values are lists of documents
        containing that subtask/dimension
    """
    inverted_index = defaultdict(list)

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                title_path = data.get('title_path', [])
                content = data.get('content', '')
                subtasks_dim = data.get('subtasks_dimensions', '')

                output = clean_json_txt(data['raw_output'])
                if not subtasks_dim:
                    continue

                # Extract subtasks and dimensions
                keys = extract_subtasks_dimensions(output)
                if not keys:
                    continue
                doc_info = {
                    'title_path': title_path,
                    'content': content,
                }
                if keys not in inverted_index:
                    inverted_index[keys] = []
                inverted_index[keys].append(doc_info)

            except json.JSONDecodeError:
                print(f"Warning: Could not parse line: {line[:100]}...")
                continue

    return dict(inverted_index)

def save_inverted_index(index: Dict[tuple, List[Dict]], output_file: str):
    """Save the inverted index to a JSON file."""
    serializable_index = {
        str(key): value
        for key, value in index.items()
    }

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_index, f, indent=4, ensure_ascii=False)
        print(f"倒排索引已成功保存到: {output_file}")
    except IOError as e:
        print(f"写入文件时发生错误: {e}")

def search_index(index: Dict[tuple, List[Dict]], query: tuple) -> List[Dict]:
    """
    Search the inverted index for documents containing the query term.

    Args:
        index: The inverted index
        query: Search term (subtask or dimension)

    Returns:
        List of matching documents
    """
    return index.get(query, [])

if __name__ == "__main__":
    # Paths
    input_file = "eval/output/2010_D/2.model_generate_section_classification_output.jsonl"
    output_file = "eval/output/2010_D/2.model_generate_inverted_index.json"

    # Build and save the index
    print(f"Building inverted index from {input_file}...")
    index = build_inverted_index(input_file)
    save_inverted_index(index, output_file)
    print(f"Inverted index saved to {output_file}")
    print(f"Total unique keys in index: {len(index)}")
