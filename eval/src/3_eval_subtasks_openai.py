"""
本文件用于对子任务进行评估，使用 OpenAI API

参数：
    --model_generate_index: 模型生成的倒排索引 JSON 路径
    --bestpaper_index: 最优论文的倒排索引 JSON 路径
    --criteria_file: 评估标准 JSON 路径
    --eval_prompt_file: 评估提示词 YAML 路径
    --output: 输出 JSONL 文件路径（追加写入）

使用方法：
    python eval/3_eval_subtasks_openai.py --model_generate_index <model_generate_index> --bestpaper_index <bestpaper_index> --criteria_file <criteria_file> --eval_prompt_file <eval_prompt_file> --output <output>

"""

import os
import json
import argparse
from typing import List, Dict

from utils import (
    load_json,
    load_yaml,
    load_subtasks_from_criteria,
    populate_template,
)

from dotenv import load_dotenv
load_dotenv(override=True)
from openai import OpenAI


dimensions = {
    '问题识别': 'problem_identify',
    '问题复述': 'problem_formulation',
    '假设建立': 'assumption_develop',
    '模型构建': 'model_construction',
    '模型求解': 'model_solving',
    '代码实现': 'code_implementation',
    '结果分析': 'result_analysis',
}

def get_evaluation_prompt(step_chinese_name: str, evaluation_prompts: Dict[str, str]) -> str:
    """
    Get evaluation prompt for a specific step

    Args:
        step_chinese_name: Chinese name of the step
        evaluation_prompts: Dictionary of evaluation prompts

    Returns:
        str: Evaluation prompt for the step
    """
    step_english_key = dimensions.get(step_chinese_name)
    if step_english_key is None:
        raise ValueError(f"Unknown step name: {step_chinese_name}")
    return evaluation_prompts.get(step_english_key, "")

def get_str(section_contents: List[dict], max_length: int = 2000) -> str:
    """
    Convert a list of section contents into a formatted string.

    Args:
        section_contents: List of dictionaries containing section data
        max_length: Maximum length of content to include from each section

    Returns:
        Formatted string with section paths and contents
    """
    formatted_sections = []

    for section in section_contents:
        # Get section path and format it
        section_path = '->'.join(section.get('title_path', []))
        section_content = section.get('content', '')[:max_length]
        # Format the section with its path and content
        formatted_section = f'当前章节路径：{section_path}\n{section_content}\n'
        formatted_sections.append(formatted_section)

    return '\n'.join(formatted_sections)


def get_batch_messages(prompt_dict: List[dict]) -> List[List[Dict[str, str]]]:
    message_list = []
    for _prompt_ in prompt_dict:
        template = _prompt_.pop('prompt_template')
        prompt = populate_template(template, _prompt_)
        message_list.append([
            {"role": "user", "content": prompt}
        ])
    return message_list


def call_openai_single(messages: List[Dict[str, str]], model: str) -> str:
    """Call OpenAI API for a single message with error handling"""
    if OpenAI is None:
        raise ImportError("请先安装 openai 包：pip install openai>=1.0.0")

    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise EnvironmentError("未检测到 OPENAI_API_KEY 环境变量。请先设置 OpenAI API Key。")

    base_url = os.getenv('OPENAI_API_BASE') or os.getenv('OPENAI_BASE_URL')
    client_kwargs = {"api_key": api_key, **({"base_url": base_url} if base_url else {})}

    try:
        client = OpenAI(**client_kwargs)
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0,
            top_p=1.0,
            n=1,
            max_tokens=32768,
        )
        return resp.choices[0].message.get('content') if hasattr(resp.choices[0].message, 'get') else resp.choices[0].message.content or ""
    except Exception as e:
        print(f"Error calling OpenAI API: {str(e)}")
        raise


def main():
    parser = argparse.ArgumentParser(description='评估子任务')
    parser.add_argument('--model_generate_index', required=True, help='模型生成的倒排索引 JSON 路径')
    parser.add_argument('--bestpaper_index', required=True, help='最优论文的倒排索引 JSON 路径')
    parser.add_argument('--criteria_file', required=True, help='评估标准 JSON 路径')
    parser.add_argument('--eval_prompt_file', default='eval/prompts/eval_prompt.yaml', help='评估提示词 YAML 路径')
    parser.add_argument('--output', required=True, help='输出 JSONL 文件路径（追加写入）')

    parser.add_argument('--openai_model', default='gpt-5-mini', help='OpenAI 模型名称，例如 gpt-4o-mini')

    args = parser.parse_args()

    # Load subtasks from criteria file
    subtasks, subtask_ens = load_subtasks_from_criteria(args.criteria_file)
    model_generate_inverted_index = load_json(args.model_generate_index)
    bestpaper_inverted_index = load_json(args.bestpaper_index)

    evaluation_prompts = load_yaml(args.eval_prompt_file)['evaluation_prompts']
    criteria_dict = load_json(args.criteria_file)

    # 2) Assemble prompt_dict
    prompt_dict: List[dict] = []
    for subtask in subtasks:
        for dimension_cn in dimensions.keys():
            subtask_en = subtask_ens[subtask]
            task_dimens_str = str((subtask, dimension_cn))
            bestpaper_lst = bestpaper_inverted_index.get(task_dimens_str, [])
            modelgenerate_lst = model_generate_inverted_index.get(task_dimens_str, [])
            bestpaper_str = get_str(bestpaper_lst)
            modelgenerate_str = get_str(modelgenerate_lst)

            _criteria_dimension = criteria_dict[subtask_en]['criteria'][dimension_cn]
            criteria_dimension = []
            for _ in _criteria_dimension:
                _.pop('dimension', None)
                criteria_dimension.append(_)

            _prompt_ = {
                'task_id': subtask,
                'eval_dimension': dimension_cn,
                'subtask': criteria_dict[subtask_en]['subtask'],
                'bestpaper': bestpaper_str,
                'model_generate': modelgenerate_str,
                'criteria': criteria_dimension,
                'prompt_template': get_evaluation_prompt(dimension_cn, evaluation_prompts)
            }
            prompt_dict.append(_prompt_)

    # 3) 准备 messages 并逐个处理
    batch_messages = get_batch_messages(prompt_dict)

    # 检查输出文件是否已存在，如果存在则加载已处理的结果
    processed_indices = set()
    output_file = args.output

    if os.path.exists(output_file):
        print(f"检测到已存在的输出文件 {output_file}，将尝试从中恢复...")
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        task_id = data.get('task_id', '')
                        eval_dimension = data.get('eval_dimension', '')
                        subtask_content = data.get('subtask', '')
                        if task_id and eval_dimension:
                            processed_indices.add((task_id, eval_dimension))
                    except json.JSONDecodeError:
                        continue
            print(f"已加载 {len(processed_indices)} 条已处理记录")
        except Exception as e:
            print(f"加载已存在文件时出错: {e}，将创建新文件")

    # 4) 逐个处理并写入结果
    total = len(prompt_dict)
    processed_count = 0
    success_count = 0

    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

    with open(output_file, 'a', encoding='utf-8') as f:
        for i, (prompt, messages) in enumerate(zip(prompt_dict, batch_messages), 1):
            # 检查是否已经处理过
            task_id = prompt.get('task_id', '')
            subtask_content = prompt.get('subtask', '')
            eval_dimension = prompt.get('eval_dimension', '')

            if (task_id, eval_dimension) in processed_indices:
                print(f"[{i}/{total}] 跳过已处理项: [{task_id}]({subtask_content}) - 维度[{eval_dimension}]")
                processed_count += 1
                continue

            print(f"[{i}/{total}] 正在处理: [{task_id}]({subtask_content}) - 维度[{eval_dimension}]")
            # print(f"Prompt: {messages[0]['content']}")
            print(f"Model Generate: {prompt['model_generate'][:2000]}\n")

            model_generate_str = prompt.get('model_generate', '')
            if not model_generate_str:
                print(f"跳过 {task_id}({subtask_content}) - {eval_dimension}，因为 model_generate_str 为空")
                processed_count += 1
                continue

            try:
                output = call_openai_single(messages, model=args.openai_model)
                prompt['output'] = output

                f.write(json.dumps(prompt, ensure_ascii=False) + '\n')
                f.flush()

                success_count += 1
                print(f"成功处理: [{task_id}]({subtask_content}) - 维度[{eval_dimension}]")

            except Exception as e:
                print(f"处理 [{task_id}]({subtask_content}) - 维度[{eval_dimension}] 时出错: {str(e)}")
                continue

            processed_count += 1

    print(f"\n处理完成! 总计: {total}, 成功: {success_count}, 已跳过: {len(processed_indices)}")
    print(f"输出文件: {os.path.abspath(output_file)}")


if __name__ == '__main__':
    main()
