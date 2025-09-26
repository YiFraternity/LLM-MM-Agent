import os
import copy
import json
import re
from pathlib import Path
from typing import List, Dict, Any
from vllm import LLM, SamplingParams
from jinja2 import Template, StrictUndefined
import yaml


os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'


def clean_json_txt(json_txt: str) -> Dict[str, Any]:
    """
    从可能包含 ```json ... ``` 或 ``` ... ``` 的文本中提取 JSON 并解析为 dict。

    优先级：
      1) 第一个标记为 ```json 的代码块（忽略大小写）
      2) 第一个任意 ``` ... ``` 代码块
      3) 整个输入字符串

    解析失败时打印错误信息和出错行的上下文，返回 {} 。
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

    try:
        return json.loads(payload)
    except json.JSONDecodeError as e:
        # 打印调试信息：错误信息 + 错误附近上下文，便于定位
        print("JSONDecodeError:", e)
        lines = payload.splitlines()
        # JSONDecodeError 有 lineno, colno 属性（从1开始）
        lineno = getattr(e, "lineno", None)
        colno = getattr(e, "colno", None)
        if lineno is not None and lineno > 0:
            err_i = lineno - 1
            start = max(0, err_i - 3)
            end = min(len(lines), err_i + 3)
            print("---- payload context (lines {}..{}) ----".format(start+1, end))
            for i in range(start, end):
                prefix = ">>" if i == err_i else "  "
                print(f"{prefix} {i+1:4d}: {lines[i]}")
            if colno is not None and err_i < len(lines):
                # 在出错行下方画指针
                pointer = " " * (6 + colno) + "^"
                print(pointer)
            print("---- end context ----")
        else:
            # 如果没有行号信息，打印前 2000 字符以便检查
            print("Payload (first 2000 chars):")
            print(payload[:2000])
        return {}


def read_json_file(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        json_txt = f.read()
    return json.loads(json_txt)


def read_jsonl_file(jsonl_file) -> list:
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        jsonl = f.readlines()
    return [json.loads(json_txt) for json_txt in jsonl]

def load_yaml(yaml_path: str) -> dict:
    with open(yaml_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def populate_template(template: str, variables: dict) -> str:
    """
    Populate a Jinja template with variables.
    """
    compiled_template = Template(template, undefined=StrictUndefined)
    try:
        return compiled_template.render(**variables)
    except Exception as e:
        raise Exception(f"Error during jinja template rendering: {type(e).__name__}: {e}")

def prepare_batch_prompts(prompts_kwargs: List[dict], prompt_template: str, system_prompt='') -> List[str]:
    """
    Prepare a batch of prompts for inference.
    """
    prompts = [populate_template(prompt_template, prompt_kwarg) for prompt_kwarg in prompts_kwargs]
    message_list = [[{"role": "user", "content": prompt}] for prompt in prompts]
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

def get_dataset_path(root_dir):
    dataset_path = []
    if not os.path.exists(root_dir):
        return dataset_path
    dataset_path = os.listdir(root_dir)
    return dataset_path

def postprocess_output(inputs:List[dict], outputs):
    """
    Postprocess the output of the model.
    """
    assert len(inputs) == len(outputs)
    for _input, _output in zip(inputs, outputs):
        _input['output'] = _output
    return inputs

def get_phase1_variable(root_dir:Path) -> List[dict]:
    result = []
    if os.path.isfile(root_dir):
        data = read_json_file(root_dir)
        data['question'] = data['problem_requirement']
        data['task_id'] = root_dir.stem.split('.')[0]
        result.append(data)
        return result

    tasks = os.listdir(root_dir)
    tasks.sort()
    for task in tasks:
        task_id = task.split('.')[0]
        data = read_json_file(os.path.join(root_dir, task))
        data['question'] = data['problem_requirement']
        data['task_id'] = task_id
        result.append(data)
    return result

def get_phase2_variable(root_dir: Path) -> List[dict]:
    result = []
    tasks = os.listdir(root_dir)
    tasks.sort()
    for task in tasks:
        data = read_json_file(os.path.join(root_dir, task))
        _data = {'task_id': data['task_id']}
        all_subtask = data['output']
        all_subtask = clean_json_txt(all_subtask)
        for key, value in all_subtask.items():
            if key.lower().startswith('subtask'):
                __data = copy.deepcopy(_data)
                __data['subtask'] = value
                __data['number_of_subtask'] = int(key.split('_')[1])
                result.append(__data)
    return result

if __name__ == '__main__':
    base_dir = Path('MMBench/CPMCM/problem/2010_D.json')
    output_dir = Path('MMBench/CPMCM/criteria_1')
    output_dir_2 = Path('MMBench/CPMCM/criteria_2')
    contents = get_phase1_variable(base_dir)

    dimensions = {
        '问题识别': 'problem_identify',
        '问题复述': 'problem_formulation',
        '假设建立': 'assumption_develop',
        '模型构建': 'model_construction',
        '模型求解': 'model_solving',
        '代码实现': 'code_implement',
        '结果分析': 'result_analysis',
    }
    # 阶段一：1. 生成子问题数量和描述
    templates = load_yaml('process_data/criterial_generate.yaml')
    phase1_templates = templates['math_modeling_criteria_generator']['zh']['number_of_questions']
    all_prompts = prepare_batch_prompts(contents, phase1_templates, '')

    print(all_prompts[0][0]['content'])

    model_name_or_path = '/home/share/models/modelscope/Qwen/Qwen2.5-32B-Instruct'
    model, sampling_params = load_llm(model_name_or_path, gpu_num=4)

    outputs_t = model.chat(all_prompts, sampling_params, use_tqdm=True)

    """
    pred_lst = []
    for o_t in outputs_t:
        pred_lst.append(o_t.outputs[0].text)

    results = postprocess_output(contents, pred_lst)
    for result in results:
        task_id = result['task_id']
        output_path = output_dir / f"{task_id}.json"
        write_json(result, output_path)
    """

    # 阶段二：2. 生成评分细则
    contents = get_phase2_variable(output_dir)
    dimension_references = templates['dimension_reference']
    dimes = [
        {
            'eval_dimension': dim,
            'eval_dimension_ref': dimension_references.get(dim_en, '')}
        for dim, dim_en in dimensions.items()
    ]
    content_dimes = [
        {**content, **dime}
        for content in contents
        for dime in dimes
    ]
    phase2_templates = templates['math_modeling_criteria_generator']['zh']['eval_dimension']
    all_prompts = prepare_batch_prompts(content_dimes, phase2_templates, '')

    outputs_t = model.chat(all_prompts, sampling_params, use_tqdm=True)

    pred_lst = []
    for o_t in outputs_t:
        pred_lst.append(o_t.outputs[0].text)

    results = postprocess_output(content_dimes, pred_lst)
    # del output dir 2
    if output_dir_2.exists():
        for file in output_dir_2.iterdir():
            if file.is_file():
                file.unlink()
    else:
        output_dir_2.mkdir(parents=True, exist_ok=True)
    for result in results:
        task_id = result['task_id']
        output_path = output_dir_2 / f"{task_id}.jsonl"
        write_jsonl(result, output_path)
