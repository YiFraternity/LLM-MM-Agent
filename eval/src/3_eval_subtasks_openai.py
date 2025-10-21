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
from pathlib import Path
from typing import List, Dict, Union, Set, Tuple
import logging

from dotenv import load_dotenv
load_dotenv(override=True)
from openai import OpenAI

from utils import (
    load_json,
    load_yaml,
    populate_template,
    DIMENSION_MAPPING,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


def get_openai_message(prompt_dict: dict) -> List[Dict[str, str]]:
    template = prompt_dict.pop('prompt_template')
    prompt = populate_template(template, prompt_dict)
    return [
        {"role": "user", "content": prompt}
    ]


def call_openai(messages: List[Dict[str, str]], model: str) -> str:
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

def assemble_prompt_dict(
    model_generate_inverted_index: Dict[str, List[Dict]],  # 模型生成的倒排索引，从 model_generate_index_file 中读取
    bestpaper_inverted_index: Dict[str, List[Dict]],  # 最优论文的倒排索引，从 bestpaper_index_file 中读取
    criteria_dict: Dict[str, Dict],  # 评估标准，从 criteria_file 中读取
    evaluation_prompts: Dict[str, str],  # 评估提示词，从 eval_prompt_file 中读取
) -> List[dict]:
    """
    Assemble prompt dict for evaluation
    """
    all_tasks = list(criteria_dict.keys())
    prompt_dict: List[dict] = []
    dimension_cn_lst = DIMENSION_MAPPING.keys()
    for subtask in all_tasks:
        task_id = subtask.split('_')[1]
        for dimension_cn in dimension_cn_lst:
            task_dimension_str = str((f"子任务{task_id}", dimension_cn))
            bestpaper_lst = bestpaper_inverted_index.get(task_dimension_str, [])
            modelgenerate_lst = model_generate_inverted_index.get(task_dimension_str, [])
            bestpaper_str = get_str(bestpaper_lst)
            modelgenerate_str = get_str(modelgenerate_lst)

            criteria_dimension = [{k: v for k, v in d.items() if k != 'dimension'}
                      for d in criteria_dict[subtask]['criteria'][dimension_cn]]

            dimension_english = DIMENSION_MAPPING.get(dimension_cn)
            _prompt_ = {
                'subtask_id': f"子任务{task_id}",
                'eval_dimension': dimension_cn,
                'subtask': criteria_dict[subtask]['subtask'],
                'bestpaper': bestpaper_str,
                'model_generate': modelgenerate_str,
                'criteria': criteria_dimension,
                'prompt_template': evaluation_prompts[dimension_english],
            }
            prompt_dict.append(_prompt_)
    return prompt_dict

def check_processed_indices(output_file):
    processed_indices = set()
    if os.path.exists(output_file):
        logger.info(f"检测到已存在的输出文件 {output_file}，将尝试从中恢复...")
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        subtask = data.get('subtask_id', '')
                        eval_dimension = data.get('eval_dimension', '')
                        if subtask and eval_dimension:
                            processed_indices.add((subtask, eval_dimension))
                    except json.JSONDecodeError:
                        continue
            logger.info(f"已加载 {len(processed_indices)} 条已处理记录")
        except Exception as e:
            logger.error(f"加载已存在文件时出错: {e}，将创建新文件")
    return processed_indices

def append_result(
    output_file: Path,
    prompt_dict_lst: List[dict],
    processed_indices: Set[Tuple[str, str]],
    openai_model: str = 'gpt-5-mini'
):
    """
    将结果追加到输出文件
    Args:
        output_file (str): 输出文件路径
        prompt_dict_lst (List[dict]):
            - subtask_id (str): 任务ID
            - eval_dimension (str): 评估维度
            - subtask (str): 任务描述
            - bestpaper (str): 最佳论文
            - model_generate (str): 模型生成
            - criteria (List[dict]): 评估标准
            - prompt_template (str): 提示词模板
        processed_indices (Set[Tuple[str, str]]): 已处理的索引集合
    """
    processed_count = 0
    success_count = 0
    total = len(prompt_dict_lst)
    with open(output_file, 'a', encoding='utf-8') as f:
        for i, prompt_dict in enumerate(prompt_dict_lst):
            subtask_id = prompt_dict.get('subtask_id', '')
            subtask = prompt_dict.get('subtask', '')
            eval_dimension = prompt_dict.get('eval_dimension', '')
            if (subtask_id, eval_dimension) in processed_indices:
                logger.info(f"[{i}/{total}] 跳过已处理项: [{subtask_id}]({subtask}) - 维度[{eval_dimension}]")
                processed_count += 1
                continue

            model_generate_str = prompt_dict.get('model_generate', '')
            if not model_generate_str:
                logger.info(f"跳过 {subtask_id}({subtask}) - 维度[{eval_dimension}]，因为 model_generate_str 为空")
                processed_count += 1
                continue

            logger.info(f"[{i}/{total}] 正在处理: [{subtask_id}]({subtask}) - 维度[{eval_dimension}]")
            openai_message = get_openai_message(prompt_dict)

            try:
                output = call_openai(openai_message, model=openai_model)
                prompt_dict['output'] = output

                f.write(json.dumps(prompt_dict, ensure_ascii=False) + '\n')
                f.flush()

                success_count += 1
                logger.info(f"成功处理: [{subtask_id}]({subtask}) - 维度[{eval_dimension}]")

            except Exception as e:
                logger.info(f"处理 [{subtask_id}]({subtask}) - 维度[{eval_dimension}] 时出错: {str(e)}")
                continue
    logger.info(f"总计{total}条，成功处理 {success_count} 条，跳过 {processed_count} 条，保存到 {output_file}")


def main():
    parser = argparse.ArgumentParser(description='评估子任务')
    parser.add_argument('--model-generate-index-file', default=None, type=Path, help='模型生成的倒排索引 JSON 路径')
    parser.add_argument('--bestpaper-index-file', default=None, type=Path, help='最优论文的倒排索引 JSON 路径')

    parser.add_argument('--model-generate-indexs', nargs='+', type=Path, help='模型生成的倒排索引 JSON 路径')
    parser.add_argument('--bestpaper-indexs', nargs='+', type=Path, help='最优论文的倒排索引 JSON 路径')

    parser.add_argument('--model-generate-index-dir', default=None, type=Path, help='模型生成的倒排索引 JSON 目录')
    parser.add_argument('--bestpaper-index-dir', default=None, type=Path, help='最优论文的倒排索引 JSON 目录')
    parser.add_argument('--criteria-dir', default=None, type=Path, help='评估标准 JSON 目录')

    parser.add_argument('--eval-prompt-file', default='eval/prompts/eval_prompt.yaml', type=Path, help='评估提示词 YAML 路径')
    parser.add_argument('--output-dir', default='eval/output', type=Path, help='输出 JSONL 文件目录（追加写入）')

    parser.add_argument('--openai-model', default='gpt-5-mini', help='OpenAI 模型名称，例如 gpt-4o-mini')

    args = parser.parse_args()

    # Load subtasks from criteria file
    model_generate_index = []
    if args.model_generate_index_file:
        model_generate_index.append(args.model_generate_index_file)
    elif args.model_generate_indexs:
        model_generate_index.extend(args.model_generate_indexs)
    elif args.model_generate_index_dir:
        for _ in args.model_generate_index_dir.iterdir():
            model_generate_index.append(_)
    else:
        raise ValueError("未提供模型生成的倒排索引 JSON 路径")
    model_generate_index = sorted(model_generate_index)
    print(f"模型生成的倒排索引文件数：{len(model_generate_index)}")

    evaluation_prompts = load_yaml(args.eval_prompt_file)['evaluation_prompts']

    for task in model_generate_index:
        task_id = task.stem

        criteria_dict = load_json(args.criteria_dir / f'{task_id}.json')
        model_generate_inverted_index = load_json(task)
        bestpaper_path = args.bestpaper_index_dir / f'{task_id}.json'
        if not os.path.exists(bestpaper_path):
            logger.warning(f"未找到最优论文的倒排索引文件：{bestpaper_path}")
            continue
        bestpaper_inverted_index = load_json(bestpaper_path)

        prompt_dict_lst = assemble_prompt_dict(
            model_generate_inverted_index=model_generate_inverted_index,
            bestpaper_inverted_index=bestpaper_inverted_index,
            criteria_dict=criteria_dict,
            evaluation_prompts=evaluation_prompts
        )

        os.makedirs(args.output_dir, exist_ok=True)
        output_file = args.output_dir / f'{task_id}.jsonl'
        processed_indices = check_processed_indices(output_file)

        append_result(
            output_file=output_file,
            prompt_dict_lst=prompt_dict_lst,
            processed_indices=processed_indices,
            openai_model=args.openai_model
        )


if __name__ == '__main__':
    main()
