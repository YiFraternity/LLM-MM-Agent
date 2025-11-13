"""
本文件用于对子任务进行评估，使用 OpenAI API
"""

import argparse
from pathlib import Path
import logging

from dotenv import load_dotenv
load_dotenv(override=True)

from eval_utils import (
    call_openai_api,
    find_task_id_from_path,
    load_tex_content,
    write_json,
    load_json,
    load_yaml,
    populate_template,
)
from parser_latex import parse_latex


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_sections_and_parents(report_dict: dict, subtask_id: str) -> list:
    """
    遍历 LaTeX 结构树字典，查找标题中包含subtask_id的子章节，

    Args:
        report_dict: 树的根字典 (Document Root)。
        subtask_id: 子任务 ID "1", "2"。

    Returns:
        list: 包含 {parent_title, parent_content, child_title} 的列表。
    """
    results = []

    # 遍历 Document Root 的直接子节点，这些是 level 1 的祖先节点
    for parent_node in report_dict.get('children', []):
        if parent_node['level'] != 1:
            # 仅处理 level 1 的节点作为祖先起点
            continue

        parent_title = parent_node['title']
        parent_content = parent_node.get('content', '').strip()

        def recursive_search(node):
            node_title = node['title']

            # 检查当前节点的标题是否包含 subtask_id
            if str(subtask_id) in node_title:
                results.append({
                    "section_title": parent_title,
                    "section_content": parent_content,
                    "subsection_title": node_title,
                    "subsection_level": node['level'],
                    "subsection_content": node.get('content', '').strip()
                })

            for child in node.get('children', []):
                recursive_search(child)
        recursive_search(parent_node)

    return results

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='评估子任务')
    parser.add_argument('--task-paths', nargs='+', type=Path,
                        help='指定子任务数学建模报告的latex文件')
    parser.add_argument('--task-dir', default='output/deepseek-chat-v3.1:free/CPMCM/MM-Agent', type=Path,
                        help='子任务目录')
    parser.add_argument('--criteria-dir', default='MMBench/CPMCM/criteria', type=Path,
                        help='评估标准 JSON 目录')
    parser.add_argument('--eval-prompt', default='eval/prompts/criterial_generate.yaml', type=Path,
                        help='评估提示词 YAML 路径')
    parser.add_argument('--output-dir', default='eval/output/deepseek-chat-v3.1:free', type=Path, help='输出 Json 文件目录')
    parser.add_argument('--tmp-dir', default='tmp/eval/deepseek-chat-v3.1:free', type=Path, help='输出 Json 文件目录')

    parser.add_argument('--openai-model', default='gpt-5-mini', help='OpenAI 模型名称，例如 gpt-4o-mini')

    args = parser.parse_args()
    return args

def main(args):
    # Load subtasks from criteria file
    template = load_yaml(args.eval_prompt)['math_modeling_report_eval']
    user_prompt_template = template['zh']
    system_prompt = template.get('system', '')

    task_paths = []
    if args.task_paths:
        for task_path in args.task_paths:
            task_paths.append(task_path)
    else:
        for task_dir in args.task_dir.iterdir():
            task_paths.append(task_dir/'latex'/'solution.tex')

    task_paths = sorted(task_paths)
    for task in task_paths:
        task_id = find_task_id_from_path(task)
        criteria_path = args.criteria_dir / f'{task_id}.json'

        if not task.exists():
            logger.warning(f"未找到任务文件 {task}，跳过任务 {task_id}")
            continue

        if not criteria_path.exists():
            logger.warning(f"未找到评估标准文件 {criteria_path}，跳过任务 {task_id}")
            continue

        ai_report_tex = load_tex_content(task)
        ai_report_dict = parse_latex(ai_report_tex, target_level=2).to_dict()
        criteria_dict = load_json(criteria_path)
        subtask_criterias = criteria_dict.get('subtask', {})
        subtask_criterias = sorted(subtask_criterias.items(), key=lambda x: int(x[0]))

        args.output_dir.mkdir(parents=True, exist_ok=True)
        args.tmp_dir.mkdir(parents=True, exist_ok=True)
        output_path = args.output_dir / f'{task_id}.json'
        tmp_path = args.tmp_dir / f'{task_id}.json'

        tmp_results = load_json(tmp_path)
        eval_results = load_json(output_path)
        assert isinstance(eval_results, dict)
        assert isinstance(tmp_results, dict)

        try:
            for subtask_criteria in subtask_criterias:
                subtask_id = subtask_criteria[0]

                eval_res = eval_results.get(str(subtask_id), None)
                _eval_res = tmp_results.get(str(subtask_id), {}).get('eval_res', None)
                if eval_res or _eval_res:
                    logger.info(f"✅  任务 {task_id} 子任务 {subtask_id} 已存在，跳过")
                    continue
                sections_and_parents = extract_sections_and_parents(ai_report_dict, subtask_id)

                subsection_content = ''
                for sect in sections_and_parents:
                    subsection_content += f"\\section{{{sect['section_title']}}}\n{sect['section_content']}\n"
                    subsection_content += f"\\subsection{{{sect['subsection_title']}}}\n{sect['subsection_content']}\n"
                subtask_info: dict = subtask_criteria[1]
                prompt_info = {
                    'subproblem': subtask_info.get('subtask', ''),
                    'report_content': subsection_content,
                    'report_criteria': subtask_info.get('criteria', ''),
                }
                user_prompt = populate_template(user_prompt_template, prompt_info)
                response = call_openai_api(user_prompt, system_prompt, model=args.openai_model)
                # response = '模拟的评估结果'
                eval_results[str(subtask_id)] = response
                tmp_results[str(subtask_id)] = {
                    **prompt_info,
                    'eval_res': response
                }
        except Exception as e:
            logger.error(f"❌  任务 {task_id} 子任务 {subtask_id} 评估出错: {e}")
        write_json(eval_results, output_path)
        write_json(tmp_results, tmp_path)


if __name__ == '__main__':
    args = parse_args()
    main(args)
