import os
import json
from pathlib import Path
from typing import List
from vllm import LLM, SamplingParams
from jinja2 import Template, StrictUndefined
import yaml


os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

def clean_json_txt(json_txt) -> dict:
    if '```json' in json_txt:
        json_txt = json_txt.replace('```json', '')
    elif '```' in json_txt:
        json_txt = json_txt.replace('```', '')
    json_txt = json_txt.replace('```', '').strip()
    try:
        json_txt = json.loads(json_txt)
    except json.decoder.JSONDecodeError:
        print(json_txt)
        return {}
    return json_txt


def read_json_file(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        json_txt = f.read()
    return json.loads(json_txt)


def load_jsonl_file(jsonl_file) -> list:
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
    with open(file_path, 'a', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)
        f.write('\n')

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
        'max_model_len': 32768,
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

def scan_all_dataset_path(root_dir) -> List[dict]:
    result = []
    tasks = os.listdir(root_dir)
    for task in tasks:
        if '2017_D' not in task:
            continue
        datas = load_jsonl_file(os.path.join(root_dir, task))
        for data in datas:
            file_path = data.get('dataset_path') or data.get('file_path')
            file_path = file_path.split('/')[-1]
            content = data.get('output')[:16384]
            result.append({
                'task_id': task.split('.')[0],
                'file_path': file_path,
                'content': content,
            })
    return result

if __name__ == '__main__':
    base_dir = Path('MMBench/CPMCM/problem_2')
    output_dir = Path('MMBench/CPMCM/problem_3')
    contents = scan_all_dataset_path(base_dir)
    contents_lens = [len(content['content']) for content in contents]
    # Prepare prompts
    templates = load_yaml('process_data/rewrite_json.yaml')
    templates = templates['system_prompt']
    all_prompts = prepare_batch_prompts(contents, templates, '')

    model_name_or_path = '/group_homes/our_llm_domain/home/share/open_models/Qwen/Qwen2.5-14B-Instruct'
    model, sampling_params = load_llm(model_name_or_path, gpu_num=1)

    outputs_t = model.chat(all_prompts, sampling_params, use_tqdm=True)

    pred_lst = []
    for o_t in outputs_t:
        pred_lst.append(o_t.outputs[0].text)

    results = postprocess_output(contents, pred_lst)
    for result in results:
        task_id = result['task_id']
        output_path = output_dir / f"{task_id}.jsonl"
        write_json(result, output_path)
