import os
import re
from pathlib import Path
import json
from typing import Dict, Any, List, Union
import logging
import time

from jinja2 import Template, StrictUndefined
import yaml
from openai import OpenAI


os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

DIMENSION_MAPPING = {
    '问题识别': 'problem_identify',
    '问题复述': 'problem_formulation',
    '假设建立': 'assumption_develop',
    '模型构建': 'model_construction',
    '模型求解': 'model_solving',
    '代码实现': 'code_implementation',
    '结果分析': 'result_analysis',
}

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

def load_yaml(config_file: Union[str, Path]) -> Dict[str, Any]:
    """
    Load YAML configuration file.
    """
    with open(config_file, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)

def load_subtasks_from_criteria(criteria_file: str) -> tuple[list, dict]:
    """
    Load subtasks from criteria JSON file

    Args:
        criteria_file: Path to the criteria JSON file

    Returns:
        tuple: (subtasks_list, subtask_mapping)
    """
    try:
        criteria = load_json(criteria_file)
        # Extract subtask keys that start with 'subtask_'
        subtask_keys = [key for key in criteria.keys() if key.startswith('subtask_')]
        subtask_keys.sort()  # Ensure consistent ordering

        # Create subtasks list and mapping
        subtasks = [f'子任务{int(key.split("_")[1])}' for key in subtask_keys]
        subtask_ens = {f'子任务{int(key.split("_")[1])}': key for key in subtask_keys}

        print(f"Loaded {len(subtasks)} subtasks from criteria file")
        return subtasks, subtask_ens

    except Exception as e:
        print(f"Error loading subtasks from criteria file: {e}")
        raise

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

def dict_to_latex_table(data: List[Dict[str, Any]]) -> str:
    """Converts the list of dictionaries into a simple LaTeX tabular environment."""
    if not data:
        return ""

    headers = list(data[0].keys())
    latex_str = "\\begin{tabular}{|" + "|".join(["l"] * len(headers)) + "|}\n"
    latex_str += "\\hline\n"
    latex_str += " & ".join([h.replace("_", "\\_") for h in headers]) + " \\\\\n"
    latex_str += "\\hline\n"
    for row in data:
        escaped_row = [str(v).replace("\\", "\\textbackslash ").replace("_", "\\_").replace("&", "\\&") for v in row.values()]
        latex_str += " & ".join(escaped_row) + " \\\\\n"
    latex_str += "\\hline\n"
    latex_str += "\\end{tabular}\n"

    return latex_str

def clean_json_txt(json_txt: str, standardize: bool = False) -> Union[str, Dict[str, Any], List[Dict[str, Any]]]:
    """
    从可能包含 ```json ... ``` 或 ``` ... ``` 的文本中提取 JSON 并解析为 dict。

    优先级：
      1) 第一个标记为 ```json 的代码块（忽略大小写）
      2) 第一个任意 ``` ... ``` 代码块
      3) 解析包含latex公式的json
      3) 整个输入字符串

    参数:
        json_txt: 要处理的JSON文本
        standardize: 是否尝试标准化非标准JSON

    返回:
        解析后的JSON对象，如果standardize为True且标准化失败，则返回空字典，否则返回json_txt
    """
    if not isinstance(json_txt, str):
        raise TypeError("json_txt must be a str")

    m = re.search(r'```(?:\s*json\b)[\r\n]*([\s\S]*?)```', json_txt, re.IGNORECASE)
    if not m:
        m = re.search(r'```[\r\n]*([\s\S]*?)```', json_txt)

    if m:
        payload = m.group(1).strip()
    else:
        payload = json_txt.strip()

    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        # If it fails, check if the error is likely due to unescaped backslashes
        corrected_payload = re.sub(r'\\', r'\\\\', payload)
        try:
            return json.loads(corrected_payload)
        except json.JSONDecodeError:
            if standardize:
                return {}
            else:
                return json_txt

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
def write_json(data: Any, file_path: Union[str, Path]) -> None:
    """
    将数据写入 JSON 文件，自动创建父目录。

    Args:
        data: 要写入的 JSON 可序列化数据。
        file_path: 输出文件路径（字符串或 Path 对象）。
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with file_path.open('w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except (TypeError, ValueError) as e:
        raise ValueError(f"数据无法序列化为 JSON: {e}") from e
    except OSError as e:
        raise OSError(f"无法写入文件 {file_path}: {e}") from e

def write_jsonl(data, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'a', encoding='utf-8') as f:
        if isinstance(data, list):
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        else:
            f.write(json.dumps(data, ensure_ascii=False)+ '\n')


def load_jsonl_file(file_path: str) -> List[Dict]:
    """加载 JSONL 文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def load_llm(model_name_or_path, tokenizer_name_or_path=None, gpu_num=1, lora_model_name_or_path=None, max_model_len=32768):
    """
    Load a VLLM model.
    """
    from vllm import LLM, SamplingParams
    kw_args = {
        "model": model_name_or_path,
        "tokenizer": tokenizer_name_or_path,
        "tokenizer_mode": "slow",
        "tensor_parallel_size" : gpu_num,
        "enable_lora": bool(lora_model_name_or_path),
        'max_model_len': max_model_len,
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

def latex_to_json(latex_content):
    """
    解析LaTeX文件内容，并将其转换为树状结构的字典列表。

    Args:
        latex_content: 包含LaTeX源码的字符串。

    Returns:
        一个列表，其中每个元素都是一个代表section的字典。
    """

    # 定义LaTeX命令的层级
    hierarchy = {
        'section': 1,
        'subsection': 2,
        'subsubsection': 3,
        'paragraph': 4,
        'subparagraph': 5,
    }

    command_pattern = '|'.join(hierarchy.keys())
    pattern = re.compile(r'\\(' + command_pattern + r')\*?\s*\{([^}]+)\}')

    doc_match = re.search(r'\\begin\{document\}(.*?)\\end\{document\}', latex_content, re.DOTALL)
    if not doc_match:
        print("警告: 未找到 \\begin{document} 环境。将尝试解析整个文件。")
        doc_content = latex_content
    else:
        doc_content = doc_match.group(1)

    parts = pattern.split(doc_content)
    result_tree = []
    level_parents = {0: {"children": result_tree}}
    preamble_content = parts[0].strip()
    if preamble_content:
        preamble_node = {
            "title": "前言",
            "content": preamble_content,
            "children": []
        }
        result_tree.append(preamble_node)
        level_parents[1] = preamble_node

    for i in range(1, len(parts), 3):
        command = parts[i]
        title = parts[i+1].strip()
        content = parts[i+2].strip()

        level = hierarchy.get(command)
        if not level:
            continue

        new_node = {
            "title": title,
            "content": content,
            "children": []
        }

        parent_node = level_parents.get(level - 1)
        if not parent_node:
            parent_level = max(k for k in level_parents if k < level)
            parent_node = level_parents[parent_level]

        parent_node["children"].append(new_node)

        level_parents[level] = new_node
        keys_to_delete = [k for k in level_parents if k > level]
        for k in keys_to_delete:
            del level_parents[k]

    return result_tree

def find_task_id_from_path(path: Path) -> Union[None, str]:
    task_id_pattern = re.compile(r'(\d{4}_[A-F])')
    for part in path.parts:
        match = task_id_pattern.search(part)
        if match:
            return match.group(0)

    match = task_id_pattern.search(str(path))
    if match:
        return match.group(0)

    return None

def call_openai_api(
    user_prompt: str,
    system_prompt: str = "你是一个专业的文本分类助手。",
    model: str = "gpt-4-turbo",
    max_retries: int = 3,
    max_tokens: int = 32768,
    temperature: float = 0.3,
    top_p: float = 1.0,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    **kwargs: Any
) -> str:
    """
    Call OpenAI API with retry logic using v1.0+ client.

    Args:
        user_prompt: The prompt to send to the model.
        system_prompt: The system prompt to send to the model.
        model: The model to use (default: gpt-4-turbo).
        max_retries: Maximum number of retry attempts.
        max_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature.
        top_p: Nucleus sampling threshold.
        frequency_penalty: Frequency penalty.
        presence_penalty: Presence penalty.
        **kwargs: Additional arguments to pass to chat.completions.create.

    Returns:
        The model's response as a string.

    Raises:
        Exception: If all retry attempts fail.
    """
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE") or os.getenv("OPENAI_BASE_URL"),
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                **kwargs,  # 支持额外参数（如 timeout, response_format 等）
            )
            content = response.choices[0].message.content
            if content is None:
                raise ValueError("Empty response from OpenAI API")
            return content

        except Exception as e:  # 捕获网络/限流等异常
            if attempt == max_retries - 1:
                logger.error(f"OpenAI API failed after {max_retries} attempts: {e}")
                raise Exception(f"Failed to get response from OpenAI API: {e}") from e

            wait_time = (2 ** attempt) * 1.0  # 指数退避：1s, 2s, 4s...
            logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
            time.sleep(wait_time)

    # 理论上不会执行到这里
    raise RuntimeError("Unexpected fall-through in retry loop")
