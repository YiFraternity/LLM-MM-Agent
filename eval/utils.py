import os
import re
from pathlib import Path
import json
from typing import Dict, Any, List, Union
from vllm import LLM, SamplingParams
from jinja2 import Template, StrictUndefined
import yaml


os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'


def load_tex_content(latex_file):
    """
    Load LaTeX file content.
    """
    with open(latex_file, 'r', encoding='utf-8') as file:
        return file.read()

def load_json(file_path):
    """
    Load JSON file content.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def load_yaml(config_file):
    """
    Load YAML configuration file.
    """
    with open(config_file, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)


def batch_standardize_json(texts: List[str], llm) -> List[Dict]:
    """
    批量标准化JSON文本

    参数:
        texts: 需要标准化的JSON文本列表
        llm: 用于标准化的语言模型实例

    返回:
        标准化后的JSON对象列表
    """
    if not texts:
        return []

    try:
        current_dir = Path(__file__).parent
        prompt_path = current_dir / 'json_standardization.yaml'

        with open(prompt_path, 'r', encoding='utf-8') as f:
            prompt_template = yaml.safe_load(f).get('json_standardization', '')

        if not prompt_template:
            return [{}] * len(texts)

        # 为每个文本准备提示
        template = Template(prompt_template)
        prompts = [template.render(input_text=text) for text in texts]

        # 批量生成响应
        responses = llm.generate(prompts)

        # 处理响应
        results = []
        for response in responses:
            try:
                standardized_json = response.outputs[0].text.strip()
                results.append(json.loads(standardized_json))
            except (json.JSONDecodeError, IndexError, AttributeError):
                results.append({})

        return results

    except Exception as e:
        print(f"Error during batch JSON standardization: {str(e)}")
        return [{}] * len(texts)


def clean_json_txt(json_txt: str, standardize: bool = True, llm=None) -> Union[str, Dict[str, Any]]:
    """
    从可能包含 ```json ... ``` 或 ``` ... ``` 的文本中提取 JSON 并解析为 dict。

    优先级：
      1) 第一个标记为 ```json 的代码块（忽略大小写）
      2) 第一个任意 ``` ... ``` 代码块
      3) 整个输入字符串

    参数:
        json_txt: 要处理的JSON文本
        standardize: 是否尝试标准化非标准JSON
        llm: 可选的LLM实例，用于标准化

    返回:
        解析后的JSON对象，如果standardize为True且标准化失败，则返回空字典
    """
    if not isinstance(json_txt, str):
        raise TypeError("json_txt must be a str")

    # 1) 优先查找 ```json ... ```（不区分大小写）
    m = re.search(r'```(?:\s*json\b)[\r\n]*([\s\S]*?)```', json_txt, re.IGNORECASE)
    # 2) 若没有带 json 标记的，再查找任意 ``` ... ``` 代码块
    if not m:
        m = re.search(r'```[\r\n]*([\s\S]*?)```', json_txt)

    if m:
        payload = m.group(1).strip()
    else:
        # 没有 code fence，就用原始字符串
        payload = json_txt.strip()

    # 首先尝试直接解析
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        if not standardize or not llm:
            return {}

    try:
        current_dir = Path(__file__).parent
        prompt_path = current_dir / 'json_standardization.yaml'

        with open(prompt_path, 'r', encoding='utf-8') as f:
            prompt_template = yaml.safe_load(f).get('json_standardization', '')

        if not prompt_template:
            return {}

        template = Template(prompt_template)
        prompt = template.render(input_text=payload)

        response = llm.generate([prompt])
        standardized_json = response[0].outputs[0].text.strip()

        try:
            return json.loads(standardized_json)
        except json.JSONDecodeError:
            return {}
    except Exception as e:
        print(f"Error during JSON standardization: {str(e)}")
        return {}

def populate_template(template: str, variables: dict) -> str:
    """
    Populate a Jinja template with variables.
    """
    compiled_template = Template(template, undefined=StrictUndefined)
    try:
        return compiled_template.render(**variables)
    except Exception as e:
        raise Exception(f"Error during jinja template rendering: {type(e).__name__}: {e}")


def prepare_batch_prompts(prompts_kwargs: List[dict[str, Any]], prompt_template: str, system_prompt='') -> List[List[dict]]:
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

def write_json(data, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def write_jsonl(data, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'a', encoding='utf-8') as f:
        if isinstance(data, list):
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        else:
            f.write(json.dumps(data, ensure_ascii=False)+ '\n')

def load_llm(model_name_or_path, tokenizer_name_or_path=None, gpu_num=1, lora_model_name_or_path=None):
    """
    Load a VLLM model.
    """
    kw_args = {
        "model": model_name_or_path,
        "tokenizer": tokenizer_name_or_path,
        "tokenizer_mode": "slow",
        "tensor_parallel_size" : gpu_num,
        "enable_lora": bool(lora_model_name_or_path),
        'max_model_len': 8192,
    }
    llm = LLM(**kw_args)
    kwargs={
        "n":1,
        "max_tokens": 4096,
        "top_p":1.0,
        # sampling
        "temperature":0,
        'top_k': 1,
    }
    sampling_params = SamplingParams(**kwargs)
    return llm, sampling_params
