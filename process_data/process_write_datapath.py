import os
import json
from pathlib import Path
from typing import List
from tqdm import tqdm
from vllm import LLM, SamplingParams
from jinja2 import Template, StrictUndefined
import yaml
from process_read_content import DevRead

def read(file_path: Path) -> str:
    dr = DevRead()
    if file_path.suffix in ['.bmp', '.jpg', '.jpeg', '.png', '.mp4', '.mpg', '.mpeg', '.avi', '.wmv', '.flv', '.webm']:
        return ''
    else:
        return dr.read(file_path)[0][:32768]


def load_yaml(yaml_path: str) -> dict:
    with open(yaml_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def get_data_content(root_dir, task_id) -> dict:
    """
    我应该是要
    return
        {
            'task_id': task_id,
            'dataset_path': dataset_path,
            'content': content,
            'sub_files': sub_files,
        }
    """
    if os.path.isfile(root_dir):
        path = Path(root_dir)
        return {
            'task_id': task_id,
            'dataset_path': root_dir,
            'content': read(path),
            'sub_files': [],
        }
    if os.path.isdir(root_dir):
        files = os.listdir(root_dir)
        return {
            'task_id': task_id,
            'dataset_path': root_dir,
            'content': '',
            'sub_files': files,
        }

def populate_template(template: str, variables: dict) -> str:
    """
    Populate a Jinja template with variables.
    """
    compiled_template = Template(template, undefined=StrictUndefined)
    try:
        return compiled_template.render(**variables)
    except Exception as e:
        raise Exception(f"Error during jinja template rendering: {type(e).__name__}: {e}")

def prepare_batch_prompts(prompts_kwargs: List[dict], prompt_template: dict, system_prompt='') -> List[str]:
    """
    Prepare a batch of prompts for inference.
    """
    prompts = []
    for prompt_kwarg in prompts_kwargs:
        if prompt_kwarg['sub_files']:
            prompt = populate_template(prompt_template['folder_desp'], prompt_kwarg)
        if prompt_kwarg['content']:
            prompt = populate_template(prompt_template['file_desp'], prompt_kwarg)
        prompts.append(prompt)
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

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

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
    dataset_root_dir = 'MMBench/CPMCM/dataset'
    for task in tqdm(tasks):
        year = int(task.split('_')[0])
        if year < 2015 or year > 2018:
            continue
        task_id = task.split('.')[0]
        data = load_json(os.path.join(root_dir, task))
        dataset_paths = data['dataset_path']
        if len(dataset_paths) > 10:
            content = get_data_content(os.path.join(dataset_root_dir, task_id), task_id)
            result.append(content)
        else:
            for dataset_path in dataset_paths:
                content = get_data_content(os.path.join(dataset_root_dir, task_id, dataset_path), task_id)
                result.append(content)

    return result

if __name__ == '__main__':
    base_dir = Path('MMBench/CPMCM/problem')
    output_dir = Path('MMBench/CPMCM/problem_2')
    contents = scan_all_dataset_path(base_dir)
    contents = [_ for _ in contents if _['sub_files'] or _['content']]

    # Prepare prompts
    templates = load_yaml('process_data/process_write_datapath.yaml')
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
        task_id = result['task_id']
        output_path = output_dir / f"{task_id}.jsonl"
        write_json(result, output_path)
