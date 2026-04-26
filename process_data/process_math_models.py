"""
DevRead: A Versatile File Processing Library for Text, PDFs, DOCX, JSON,
XML, YAML, HTML, Markdown, LaTeX, PPTX, Excel, Images, and Videos, etc.
"""

import os
import re
import json
import logging
from pathlib import Path
from typing import List, Any
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


def scan_problem_latex_json_dir(root_dir):
    results = []
    for fname in os.listdir(root_dir):
        if fname.endswith('.json'):
            fpath = os.path.join(root_dir, fname)
            with open(fpath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            results.append({
                'task': fname.replace('.json', ''),
                'problem_text': data['content']
            })
    return results


def main():
    # Define source and target directories
    base_dir = Path('.')
    source_dir = 'latexs_json_str'
    # contents = scan_problem_dir(source_dir)
    contents = scan_problem_latex_json_dir(source_dir)

    # Prepare prompts
    templates = load_yaml('process_data/quest_extract_strc_prompt.yaml')
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
        write_json(result, base_dir / "MMBench" / "CPMCM" / "problem_1" / f"{result['task']}.json")

if __name__ == "__main__":
    main()
