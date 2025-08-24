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

def read_file(filepath: Path) -> List[dict]:
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    contents = []
    for idx,line in enumerate(lines):
        contents.append({'text': line.strip(), 'idx': idx})
    return contents


def main():
    file = 'MMAgent/HMML/HMML_en.md'
    contents = read_file(file)

    # Prepare prompts
    templates = load_yaml('translated_prompt.yaml')
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

if __name__ == "__main__":
    main()
