"""
DevRead: A Versatile File Processing Library for Text, PDFs, DOCX, JSON,
XML, YAML, HTML, Markdown, LaTeX, PPTX, Excel, Images, and Videos, etc.
"""

import os
import re
import json
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Any
from rich.logging import RichHandler
from rich.console import Console
from vllm import LLM, SamplingParams
from jinja2 import Template, StrictUndefined
import yaml
from process_read_content import DevRead

console = Console()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[RichHandler()],
)
logger = logging.getLogger(__name__)

def load_yaml(yaml_path: str) -> dict[str, Any]:
    with open(yaml_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def write_json(data: dict[str, Any], json_path: str) -> None:
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def populate_template(template: str, variables: dict[str, Any]) -> str:
    """
    Populate a Jinja template with variables.
    """
    compiled_template = Template(template, undefined=StrictUndefined)
    try:
        return compiled_template.render(**variables)
    except Exception as e:
        raise Exception(f"Error during jinja template rendering: {type(e).__name__}: {e}")


def load_llm(model_name_or_path, tokenizer_name_or_path=None, gpu_num=1, lora_model_name_or_path=None):
    """
    Load a VLLM model.
    """
    kw_args = {
        "model": model_name_or_path,
        "tokenizer": tokenizer_name_or_path,
        "tokenizer_mode": "slow",
        "tensor_parallel_size" : gpu_num,
        "enable_lora": bool(lora_model_name_or_path)
    }
    llm = LLM(**kw_args)
    kwargs={
        "n":1,
        "max_tokens": 8192,
        "top_p":1.0,
        # sampling
        "temperature":0,
        'top_k': 1,
    }
    sampling_params = SamplingParams(**kwargs)
    return llm, sampling_params


def prepare_batch_prompts(prompts_kwargs: List[dict[str, Any]], prompt_template: str, system_prompt='') -> List[str]:
    """
    Prepare a batch of prompts for inference.
    """
    prompts = [populate_template(prompt_template, prompt_kwarg) for prompt_kwarg in prompts_kwargs]
    if system_prompt == '':
        sys_prompt = 'You are a helpful AI assistant.'
    else:
        sys_prompt = system_prompt
    system_prompts = [sys_prompt for _ in prompts]
    message_list = []
    for prompt, sys_prompt in zip(prompts, system_prompts):
        message_list.append([
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt}
            ])
    return message_list


def postprocess_output(inputs:List[dict], outputs):
    """
    Postprocess the output of the model.
    """
    assert len(inputs) == len(outputs)

    for _input, _output in zip(inputs, outputs):
        _input['output'] = _output
    return inputs

def extract_year_and_problem_id(filename: str) -> Tuple[Optional[int], Optional[str]]:
    """Extract year and problem ID from filename."""
    patterns = [
        r'(\d{4}).*?[（(]([A-Da-d])[)）]',  # 2004年A题
        r'[A-Da-d]题',                       # A题
        r'[A-Da-d]\s*题',                    # A 题
    ]

    year = None
    problem_id = None

    # Try to extract year and problem ID from filename
    for pattern in patterns:
        match = re.search(pattern, filename, re.IGNORECASE)
        if match:
            if len(match.groups()) >= 2:
                year = int(match.group(1))
                problem_id = match.group(2).upper()
            else:
                problem_id = match.group(0)[0].upper()
            break

    # If year not found in filename, try to get it from directory name
    if year is None:
        year_match = re.search(r'(\d{4})', str(Path(filename).parent.name))
        if year_match:
            year = int(year_match.group(1))

    return year, problem_id

def extract_sections(text: str) -> Dict[str, str]:
    """Extract sections from problem text."""
    result = {
        "background": "",
        "problem_requirement": "",
        "dataset_path": "",
        "dataset_description": {},
        "variable_description": "",
        "addendum": ""
    }

    # Common section headers in Chinese
    sections = {
        "background": ["背景", "问题背景", "问题重述", "问题提出"],
        "problem_requirement": ["问题要求", "问题描述", "问题分析", "问题"],
        "dataset_description": ["数据说明", "数据描述", "数据集描述", "数据"],
        "variable_description": ["变量说明", "参数说明", "符号说明"],
        "addendum": ["附件", "补充说明", "注意事项", "注"]
    }

    # Initialize current section
    current_section = None

    # Split text into lines and process each line
    lines = text.split('\n')

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Check if this line is a section header
        found_section = False
        for section, headers in sections.items():
            for header in headers:
                if line.startswith(header) or header in line:
                    current_section = section
                    result[current_section] += f"{line}\n"
                    found_section = True
                    break
            if found_section:
                break
        else:
            # If not a section header, add to current section
            if current_section:
                result[current_section] += f"{line}\n"
            else:
                # If no section detected yet, add to background
                result["background"] += f"{line}\n"

    # Clean up the results
    for key in result:
        result[key] = result[key].strip()

    return result

def process_problem_file(filepath: Path) -> Dict:
    """Process a single problem file and return structured data."""
    # Read file content based on file extension
    reader = DevRead()
    content = reader.read(filepath, '')

    # Extract year and problem ID from filename
    filename = filepath.name
    year, problem_id = extract_year_and_problem_id(filename)

    # If problem_id is still None, try to get it from the filename
    if problem_id is None:
        # Look for A, B, C, D in the filename
        match = re.search(r'[A-Da-d]', filename)
        if match:
            problem_id = match.group(0).upper()

    # Extract sections from content
    sections = extract_sections(content)

    # Find dataset files in the same directory
    dataset_files = []
    data_dir = filepath.parent
    for data_file in data_dir.glob("*"):
        if data_file.is_file() and data_file.suffix.lower() in ['.xls', '.xlsx', '.csv', '.txt', '.dat']:
            dataset_files.append(data_file.name)

    # Structure the problem data
    problem_data = {
        "year": year,
        "problem_id": problem_id,
        "title": f"{year}年中国研究生数学建模竞赛{problem_id}题",
        "background": sections["background"],
        "problem_requirement": sections["problem_requirement"],
        "dataset_path": ", ".join(dataset_files) if dataset_files else "",
        "dataset_description": sections["dataset_description"],
        "variable_description": sections["variable_description"],
        "addendum": sections["addendum"],
        "source": "中国研究生数学建模竞赛",
        "original_filename": filename
    }

    return problem_data

def process_directory(source_dir: str, target_dir: str) -> None:
    """Process all problem files in the source directory and save as JSON."""
    source_path = Path(source_dir)
    target_path = Path(target_dir)

    # Create target directory if it doesn't exist
    target_path.mkdir(parents=True, exist_ok=True)

    # Process each year's directory
    for year_dir in source_path.glob("*年研究生数学建模竞赛*"):
        year_match = re.search(r'(\d{4})', year_dir.name)
        if not year_match:
            continue

        year = year_match.group(1)
        print(f"Processing year: {year}")

        # Process each problem file in the year directory
        for problem_file in year_dir.glob("*"):
            if problem_file.is_file() and problem_file.suffix.lower() in ['.docx', '.pdf', '.txt']:
                problem_data = process_problem_file(problem_file)
                if not problem_data:
                    continue

                # Skip if we couldn't determine the problem ID
                if not problem_data.get('problem_id'):
                    print(f"  - Warning: Could not determine problem ID for {problem_file.name}")
                    continue

                # Create output filename
                output_filename = f"{year}_{problem_data['problem_id']}.json"
                output_path = target_path / output_filename

                # Save as JSON
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(problem_data, f, ensure_ascii=False, indent=2)

                print(f"  - Processed: {problem_file.name} -> {output_filename}")


def scan_problem_dir(root_dir):
    """
    Scan a directory containing math modeling problems and return a list of dictionaries.

    Args:
        root_dir (str): Path to the directory containing problem files

    Returns:
        list: List of dictionaries with 'task' and 'content' fields
    """
    result = []
    for year_dir in os.listdir(root_dir):
        if not os.path.isdir(os.path.join(root_dir, year_dir)):
            continue
        # Extract year from directory name (e.g., '2023年研究生数学建模竞赛试题' -> '2023')
        year_match = re.search(r'(\d{4})', os.path.basename(year_dir))
        year = year_match.group(1) if year_match else 'unknown_year'

        yeardir = os.path.join(root_dir, year_dir)
        for fname in os.listdir(yeardir):
            fpath = os.path.join(yeardir, fname)
            # Process direct problem files
            if os.path.isfile(fpath) and (fname.endswith('.docx') or fname.endswith('.pdf')):
                # Extract problem number (e.g., 'A题' from 'A题.pdf')
                problem_match = re.match(r'^([A-Fa-f])[题]?', fname)
                if problem_match:
                    problem_num = problem_match.group(1).upper()
                    task_id = f"{year}_{problem_num}"

                    fpath = Path(fpath)
                    # Read file content
                    try:
                        dr = DevRead()
                        content = dr.read(fpath, task=task_id)[0][0]
                        result.append({
                            'task': task_id,
                            'problem_text': content
                        })
                    except Exception as e:
                        logger.error(f"Error reading file {fpath}: {str(e)}")

            # Process problem directories
            elif os.path.isdir(fpath):
                subdir = fpath
                main_path = None

                # Find main problem file (prefer .docx over .pdf)
                for fname2 in os.listdir(subdir):
                    if re.match(r"^[A-Fa-f][题]?.*\.docx$", fname2, re.IGNORECASE):
                        main_path = os.path.join(subdir, fname2)
                        break
                    elif main_path is None and re.match(r"^[A-Fa-f][题]?.*\.pdf$", fname2, re.IGNORECASE):
                        main_path = os.path.join(subdir, fname2)

                if main_path:
                    # Extract problem number from filename or directory name
                    problem_match = re.search(r'([A-Fa-f])[题]?', os.path.basename(main_path)) or \
                                re.search(r'([A-Fa-f])[题]?', os.path.basename(subdir))

                    if problem_match:
                        problem_num = problem_match.group(1).upper()
                        task_id = f"{year}_{problem_num}"

                        main_path = Path(main_path)
                        try:
                            # Read main problem file content
                            dr = DevRead()
                            content = dr.read(main_path, task=task_id)[0][0]

                            # Add to results
                            result.append({
                                'task': task_id,
                                'problem_text': content
                            })
                        except Exception as e:
                            logger.error(f"Error reading file {main_path}: {str(e)}")
    return result

def main():
    # Define source and target directories
    base_dir = Path(__file__).parent
    source_dir = base_dir / "data" / "中国研究生数学建模"
    contents = scan_problem_dir(source_dir)

    # Prepare prompts
    templates = load_yaml('quest_extract_strc_prompt.yaml')
    prompt_template = templates['user_prompt']
    system_prompt = templates['system_prompt']
    all_prompts = prepare_batch_prompts(contents, prompt_template, system_prompt)

    model_name_or_path = '/home/share/models/modelscope/Qwen/Qwen2.5-32B-Instruct/'
    model, sampling_params = load_llm(model_name_or_path, gpu_num=4)

    outputs_t = model.chat(all_prompts, sampling_params, use_tqdm=True)

    pred_lst = []
    for o_t in outputs_t:
        pred_lst.append(o_t.outputs[0].text)

    results = postprocess_output(contents, pred_lst)
    for result in results:
        write_json(result, base_dir / "MMBench" / "CPMCM" / "problem" / f"{result['task']}.json")

    # target_dir = base_dir / "MMBench" / "MCMICM" / "problem"

    # # Create target directory if it doesn't exist
    # target_dir.mkdir(parents=True, exist_ok=True)

    # # Process all problems
    # print("Starting to process mathematical modeling problems...")
    # process_directory(source_dir, target_dir)
    # print("\nProcessing complete!")

if __name__ == "__main__":
    main()
