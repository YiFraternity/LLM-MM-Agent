import os
import copy
import shutil
import argparse
from pathlib import Path
from typing import List
from vllm import LLM, SamplingParams
from jinja2 import Template, StrictUndefined

from utils import (
    clean_json_txt,
    load_yaml,
    read_json_file,
    write_json,
    write_jsonl,
)

os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'



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
        "max_tokens": 16384,
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

def get_phase1_variable(root_dir: List[Path]) -> List[dict]:
    result = []
    for task in root_dir:
        task_id = task.stem
        data = read_json_file(task)
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
    parser = argparse.ArgumentParser(description='Generate criteria and ensure placeholders for problems.')
    parser.add_argument('--criteria1', type=Path, default=Path('MMBench/CPMCM/criteria_1'), help='Output directory for phase 1 results')
    parser.add_argument('--criteria2', type=Path, default=Path('MMBench/CPMCM/criteria_2'), help='Output directory for phase 2 results')
    parser.add_argument('--problem-root', type=Path, default=Path('MMBench/CPMCM/problem'), help='Root directory containing problem JSON files')
    parser.add_argument('--criteria-root', type=Path, default=Path('MMBench/CPMCM/criteria'), help='Directory to ensure criteria placeholders exist')
    parser.add_argument('--model-name-or-path', type=str, default='/group_homes/our_llm_domain/home/share/open_models/Qwen/Qwen2.5-32B-Instruct', help='Path to the model')
    parser.add_argument('--gpu-num', type=int, default=2, help='Number of GPUs to use')
    args = parser.parse_args()

    task_ids = []
    finished = os.listdir(args.criteria_root)
    finished = [f.split('.')[0] for f in finished]

    problem_root = args.problem_root
    for file in problem_root.iterdir():
        task_id = file.stem
        if task_id in finished:
            continue
        task_ids.append(file)

    phase1_contents = get_phase1_variable(task_ids)

    criteria1 = args.criteria1
    criteria2 = args.criteria2
    shutil.rmtree(criteria1, ignore_errors=True)
    os.makedirs(criteria1, exist_ok=True)
    shutil.rmtree(criteria2, ignore_errors=True)
    os.makedirs(criteria2, exist_ok=True)

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
    all_prompts = prepare_batch_prompts(phase1_contents, phase1_templates, '')

    print(all_prompts[0][0]['content'])

    model, sampling_params = load_llm(args.model_name_or_path, gpu_num=args.gpu_num)
    outputs_t = model.chat(all_prompts, sampling_params, use_tqdm=True)

    pred_lst = []
    for o_t in outputs_t:
        pred_lst.append(o_t.outputs[0].text)
    results = postprocess_output(phase1_contents, pred_lst)
    for result in results:
        task_id = result['task_id']
        output_path = criteria1 / f"{task_id}.json"
        write_json(result, output_path)

    # 阶段二：2. 生成评分细则
    phase2_contents = get_phase2_variable(criteria1)
    dimension_references = templates['dimension_reference']
    dimes = [
        {
            'eval_dimension': dim,
            'eval_dimension_ref': dimension_references.get(dim_en, '')}
        for dim, dim_en in dimensions.items()
    ]
    content_dimes = [
        {**content, **dime}
        for content in phase2_contents
        for dime in dimes
    ]
    phase2_templates = templates['math_modeling_criteria_generator']['zh']['eval_dimension']
    all_prompts = prepare_batch_prompts(content_dimes, phase2_templates, '')

    outputs_t = model.chat(all_prompts, sampling_params, use_tqdm=True)

    pred_lst = []
    for o_t in outputs_t:
        pred_lst.append(o_t.outputs[0].text)

    results = postprocess_output(content_dimes, pred_lst)
    for result in results:
        task_id = result['task_id']
        output_path = criteria2 / f"{task_id}.jsonl"
        write_jsonl(result, output_path)
