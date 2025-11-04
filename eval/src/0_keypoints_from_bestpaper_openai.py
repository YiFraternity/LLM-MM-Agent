"""
数学建模题目核心要点生成工具

该脚本用于分析优秀数学建模论文，并生成对应的数学建模题目核心要点。
"""
import os
import argparse
from typing import List
from pathlib import Path
import logging
from dotenv import load_dotenv

from utils import (
    load_yaml,
    load_json,
    write_json,
    load_tex_content,
    populate_template,
    clean_json_txt,
    call_openai_api,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# 避免重复添加 handler（在模块级别只执行一次）
if not logger.handlers:
    handler = logging.StreamHandler()  # 输出到控制台
    handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)

    logger.addHandler(handler)

load_dotenv(override=True)


def write_txt(content: str, file_path: Path):
    """
    将内容写入文件。
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

def main():
    parser = argparse.ArgumentParser(description='数学建模题目核心要点生成工具')
    parser.add_argument('--problem-file', type=Path, default=None,
                        help='')
    parser.add_argument('--problem-dir', type=Path, default='MMBench/CPMCM/problem',
                        help='')
    parser.add_argument('--paper-dir', type=Path, default='criteria/BestPaper2_only_model_stage',
                        help='论文文件路径')
    parser.add_argument('--prompt-template', type=Path, default='eval/prompts/criterial_generate.yaml',
                        help='Path to the prompt template')
    parser.add_argument('--model', type=str, default='gpt-5-mini', help='OpenAI 模型名称，例如 gpt-4o-mini')
    parser.add_argument('--output-dir', type=Path, default='criteria/key_points_2_gpt5', help='输出目录')
    parser.add_argument('--tmp-dir', type=Path, default='tmp/gpt5', help='输出目录')

    args = parser.parse_args()

    # 加载提示词模板
    template = load_yaml(args.prompt_template)
    system_prompt = template['model_key_insight']['system']
    user_template = template['model_key_insight']['task']
    problems = []
    if args.problem_file:
        problems.append(args.problem_file)
    elif args.problem_dir:
        for _ in args.problem_dir.iterdir():
            problems.append(_)

    problems = sorted(problems, key=lambda x: x.stem)
    contents: List[dict] = []

    args.output_dir.mkdir(exist_ok=True, parents=True)
    args.tmp_dir.mkdir(exist_ok=True, parents=True)

    for problem in problems:
        task_id = problem.stem
        if (args.output_dir / f"{task_id}.json").exists():
            logger.info(f"已存在生成结果，task_id={task_id}")
            continue
        problem_content = load_json(problem)
        question = problem_content['problem_requirement']
        report_path = args.paper_dir / f"{task_id}.tex"
        if not report_path.exists():
            logger.warning(f"未找到论文文件 {report_path}")
            continue
        excellent_report = load_tex_content(report_path)
        data = dict()
        data['excellent_report'] = excellent_report[:30000]
        data['question'] = question
        data['task_id'] = task_id
        contents.append(data)

        user_prompt = populate_template(user_template, data)
        write_txt(user_prompt, args.tmp_dir / f"{task_id}.txt")
        """
        pred = call_openai_api(system_prompt, user_prompt, model=args.model, max_tokens=32768)
        _pred = clean_json_txt(pred)
        if isinstance(_pred, dict):
            write_json(_pred, args.output_dir / f"{task_id}.json")
        else:
            write_json(pred, args.tmp_dir / f"{task_id}.json")
            logger.warning(f"未找到生成的核心要点，task_id={task_id}")
        """

if __name__ == "__main__":
    main()
