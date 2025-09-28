"""
生成评估子任务的评估结果

评估子任务的评估结果将被保存到一个JSONL文件中，每个子任务对应一个JSON对象。

评估结果的JSON对象中将包含以下字段：

- task_id: 评估子任务的ID
- task_name: 评估子任务的名称
- task_type: 评估子任务的类型
- criteria: 评估子任务的评估维度
- score: 评估子任务的评估分数
- comments: 评估子任务的评估意见

使用方法：
    python eval/3_eval_subtasks.py --model_index <model_index> --bestpaper_index <bestpaper_index> --criteria <criteria> --output <output>

参数：
    --model_index <model_index>: 评估模型生成的倒排索引文件路径
    --bestpaper_index <bestpaper_index>: 最优论文的倒排索引文件路径
    --criteria <criteria>: 评估维度的JSON文件路径
    --output <output>: 评估结果的JSONL文件路径

示例：
    python eval/3_eval_subtasks.py --model_index eval/model_generate_inverted_index.json --bestpaper_index eval/bestpaper_inverted_index.json --criteria criteria/2020_D.json --output eval/output.jsonl
"""

import os
import argparse
from typing import List, Dict

from utils import (
    load_json,
    load_yaml,
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


def parse_arguments():
    parser = argparse.ArgumentParser(description='Evaluate subtasks based on criteria and generate outputs.')
    parser.add_argument('--model_index', default='eval/model_generate_inverted_index.json',
                      help='Path to the model generated inverted index JSON file')
    parser.add_argument('--bestpaper_index', default='eval/bestpaper_inverted_index.json',
                      help='Path to the best paper inverted index JSON file')
    parser.add_argument('--criteria', default='MMBench/CPMCM/criteria/2010_D.json',
                      help='Path to the criteria JSON file')
    parser.add_argument('--output', default='eval/output.jsonl',
                      help='Output JSONL file path')
    parser.add_argument('--model_name', default='/home/share/models/modelscope/Qwen/Qwen2.5-32B-Instruct',
                      help='Name or path of the model to use')
    parser.add_argument('--gpu_num', type=int, default=4,
                      help='Number of GPUs to use')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()

    try:
        # Load indices
        model_generate_inverted_index = load_json(args.model_index)
        bestpaper_inverted_index = load_json(args.bestpaper_index)
        criteria_dict = load_json(args.criteria)

        # Process subtasks
        subtask_keys = [key for key in criteria_dict.keys() if key.startswith('subtask_')]
        subtask_keys.sort()

        subtasks = [f'子任务{int(key.split("_")[1])}' for key in subtask_keys]
        subtask_ens = {f'子任务{int(key.split("_")[1])}': key for key in subtask_keys}

        print(f"Found {len(subtasks)} subtasks in criteria file: {subtasks}")

        # Generate prompts
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

        # Process prompts and get model outputs
        batch_prompts = get_batch_prompt(prompt_dict)
        model, sampling_params = load_llm(args.model_name, gpu_num=args.gpu_num)
        outputs = model.chat(batch_prompts, sampling_params, use_tqdm=True)

        # Process outputs
        raw_outputs = [output.outputs[0].text for output in outputs]
        for prompt, output in zip(prompt_dict, raw_outputs):
            prompt['output'] = output

        # Save results
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        write_jsonl(prompt_dict, args.output)
        print(f"Evaluation completed. Results saved to {args.output}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        exit(1)
    except Exception as e:
        print(f"An error occurred: {e}")
        exit(1)