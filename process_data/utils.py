import re
import os
import json
from typing import List, Dict, Any, Union
from pathlib import Path

import yaml
from jinja2 import Template



def clean_json_txt(json_txt: str, standardize: bool = True, llm=None) -> Union[str, Dict[str, Any], List[Dict[str, Any]]]:
    """
    从可能包含 ```json ... ``` 或 ``` ... ``` 的文本中提取 JSON 并解析为 dict。

    优先级：
      1) 第一个标记为 ```json 的代码块（忽略大小写）
      2) 第一个任意 ``` ... ``` 代码块
      3) 整个输入字符串


    返回:
        解析后的JSON对象，如果standardize为True且标准化失败，则返回空字典
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
            if not standardize or not llm:
                print("Failed to parse JSON even after backslash correction.")
                return json_txt
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
            return json_txt
    except Exception as e:
        print(f"Error during JSON standardization: {str(e)}")
        return json_txt


def read_json_file(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        json_txt = f.read()
    return json.loads(json_txt)



def load_jsonl_file(file_path: str) -> List[Dict]:
    """加载 JSONL 文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def load_yaml(yaml_path: str) -> dict:
    with open(yaml_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


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
