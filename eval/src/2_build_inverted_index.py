# Description: 构建倒排索引
#
import os
import json
from pathlib import Path
import argparse
from collections import defaultdict
from typing import Dict, List, Union
import re
from tqdm import tqdm

from eval_utils import(
    clean_json_txt,
)

def extract_subtasks_dimensions(task_dimension: Union[Dict[str, str], List[Dict[str, str]]]) -> Union[tuple, None]:
    """Extract subtasks and dimensions from the task_dimension."""
    if isinstance(task_dimension, list):
        for dim in task_dimension:
            subtask = dim.get('subtask', '')
            dimension = dim.get('dimension', '')
            match = re.search(r'\d', subtask)
            if match:
                subtask_id = match.group()
            else:
                subtask_id = ''
            return (f"子任务{subtask_id}", dimension) if subtask and dimension else None
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

    task_id = Path(input_file).stem
    with open(input_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            try:
                data = json.loads(line.strip())
                title_path = data.get('title_path', [])
                model_output = data.get('model_output', '')
                content = data.get('content', '')
                output = clean_json_txt(model_output)

                if isinstance(output, str):
                    print(f"Warning: {task_id} line {i} Could not parse \n{output}")
                    continue
                if not output:
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
                print(f"Warning: {task_id} line {i} Could not parse")
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
    parser = argparse.ArgumentParser(description='Build inverted index from section classification output.')
    parser.add_argument('--input-dir', default='eval/output/Qwen2.5-32B-Instruct/1_section_classification',
                      help='Input JSONL dir containing section classification output')
    parser.add_argument('--input-file', default=None,
                      help='Input JSONL file containing section classification output')
    parser.add_argument('--input-files', nargs='+', default=None,
                      help='Input JSONL files containing section classification output')
    parser.add_argument('--output-dir', default='eval/output/Qwen2.5-32B-Instruct/2_inverted_index',
                      help='Output JSON dir to save the inverted index')

    # Parse arguments
    args = parser.parse_args()

    input_files = []
    if args.input_file is not None:
        input_files.append(args.input_file)
    elif args.input_files is not None:
        input_files = args.input_files
    else:
        input_files = [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir) if f.endswith('.jsonl')]
    input_files = sorted(input_files)

    os.makedirs(args.output_dir, exist_ok=True)
    try:
        for input_file in tqdm(input_files):
            index = build_inverted_index(input_file)
            file_name = Path(input_file).stem
            output_file = os.path.join(args.output_dir, f"{file_name}.json")
            save_inverted_index(index, output_file)
        print(f"Inverted index saved to {args.output_dir}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        exit(1)
    except Exception as e:
        print(f"An error occurred: {e}")
        exit(1)
