import os
import json
import sys
import itertools

from typing import List, Dict, Union
from utils import (
    load_tex_content,
    load_json,
    load_yaml,
    clean_json_txt,
    populate_template,
    load_llm,
    write_jsonl,
)

os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

dimensions = {
    '问题识别': 'problem_identify',
    '问题复述': 'problem_formulation',
    '假设建立': 'assumption_develop',
    '模型构建': 'model_construction',
    '模型求解': 'model_solving',
    '代码实现': 'code_implement',
    '结果分析': 'result_analysis',
}

subtasks = ['子任务1', '子任务2', '子任务3', '子任务4']
subtask_ens = {
    '子任务1': 'subtask_1',
    '子任务2': 'subtask_2',
    '子任务3': 'subtask_3',
    '子任务4': 'subtask_4',
}

evaluation_prompts = load_yaml('eval/eval_prompt.yaml')['evaluation_prompts']

def get_evaluation_prompt(step_chinese_name: str) -> str:
    """
    根据建模步骤的中文名称，返回对应的评估提示词模板。

    Args:
        step_chinese_name: 建模步骤的中文名称（如 '问题识别'）。

    Returns:
        对应的评估提示词字符串，如果名称不存在则返回错误信息。
    """
    # 1. 查找中文名对应的英文 key
    step_english_key = dimensions.get(step_chinese_name)

    if step_english_key is None:
        return f"错误：未找到中文名 '{step_chinese_name}' 对应的建模步骤或提示词。"

    # 2. 根据英文 key 查找提示词模板
    prompt = evaluation_prompts.get(step_english_key)

    if prompt is None:
        return f"错误：步骤 key '{step_english_key}' 存在，但未找到对应的提示词模板。"

    return prompt.strip()


def get_str(modelgenerate_lst):
    model_gene_str = ''
    for model_gen in modelgenerate_lst:
        cur_section_path = '->'.join(model_gen['title_path'])
        model_gene_str += f'当前章节路径：{cur_section_path}\n'
        model_gene_str += model_gen['content'][:2000]
        model_gene_str += '\n\n'
    return model_gene_str

def get_batch_prompt(prompt_dict: List[dict]) -> List[List[dict]]:
    message_list = []
    for _prompt_ in prompt_dict:
        template = _prompt_.pop('prompt_template')
        prompt = populate_template(template, _prompt_)
        message_list.append([
            {"role": "user", "content": prompt}
        ])
    return message_list


if __name__ == '__main__':
    model_generate_inverted_index = 'eval/model_generate_inverted_index.json'
    model_generate_inverted_index = load_json(model_generate_inverted_index)

    bestpaper_inverted_index = 'eval/bestpaper_inverted_index.json'
    bestpaper_inverted_index = load_json(bestpaper_inverted_index)

    """
    prompt_dict = [
        {
            'subtask': '',
            'bestpaper': 'content',
            'model_generate': 'content',
            'criteria': 'criteria',
            'prompt': 'prompt',
        }
    ]
    """

    criteria_file = 'MMBench/CPMCM/criteria/2010_D.json'
    criteria_dict = load_json(criteria_file)
    prompt_dict = []
    for dimension in dimensions.keys():
        for subtask in subtasks:
            subtask_en = subtask_ens[subtask]
            task_dimens_str = str((subtask, dimension))
            bestpaper_lst = bestpaper_inverted_index.get(task_dimens_str, [])
            modelgenerate_lst = model_generate_inverted_index.get(task_dimens_str, [])
            bestpaper_str = get_str(bestpaper_lst)
            modelgenerate_str = get_str(modelgenerate_lst)

            _criteria_dimension = criteria_dict[subtask_en]['criteria'][dimension]
            criteria_dimension = []
            for _ in _criteria_dimension:
                _.pop('dimension', None)
                criteria_dimension.append(_)

            _prompt_ = {
                'subtask': criteria_dict[subtask_en]['subtask'],
                'bestpaper': bestpaper_str,
                'model_generate': modelgenerate_str,
                'criteria': criteria_dimension,
                'prompt_template': get_evaluation_prompt(dimension)
            }
            prompt_dict.append(_prompt_)

    batch_prompts = get_batch_prompt(prompt_dict)

    model_config = {
        'model_name': '/home/share/models/modelscope/Qwen/Qwen2.5-32B-Instruct',
        'gpu_num': 4,
    }
    model, sampling_params = load_llm(model_config['model_name'], gpu_num=model_config['gpu_num'])
    outputs = model.chat(batch_prompts, sampling_params, use_tqdm=True)

    # 收集所有需要标准化的输出
    raw_outputs = [output.outputs[0].text for output in outputs]
    for prompt, output in zip(prompt_dict, raw_outputs):
        prompt['output'] = output

    write_jsonl(prompt_dict, 'eval/output.jsonl')