"""
数学建模题目核心要点生成工具

该脚本用于分析优秀数学建模论文，并生成对应的数学建模题目核心要点。
"""

import argparse
from typing import List
from pathlib import Path
import logging

from utils import (
    load_yaml,
    load_json,
    write_json,
    load_tex_content,
    load_llm,
    prepare_batch_prompts,
    clean_json_txt,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def postprocess_output(contents:List[dict], pred_lst:List[str]):
    assert len(contents) == len(pred_lst)
    results = []
    for content, pred in zip(contents, pred_lst):
        _pred = clean_json_txt(pred)
        content['pred'] = _pred
        results.append(content)
    return results


def main():
    parser = argparse.ArgumentParser(description='数学建模题目核心要点生成工具')
    parser.add_argument('--problem-file', type=Path, default=None,
                        help='')
    parser.add_argument('--problem-dir', type=Path, default='MMBench/CPMCM/problem',
                        help='')
    parser.add_argument('--paper-dir', type=Path, default='criteria/BestPaper_only_model_stage',
                        help='论文文件路径')
    parser.add_argument('--model-name-or-path', type=str, default='/group_homes/our_llm_domain/home/share/open_models/Qwen/Qwen2.5-32B-Instruct',
                        help='Path to the model')
    parser.add_argument('--gpu-num', type=int, default=2,
                        help='Number of GPUs to use')
    parser.add_argument('--prompt-template', type=Path, default='eval/prompts/criterial_generate.yaml',
                        help='Path to the prompt template')
    parser.add_argument('--output-dir', type=Path, default='criteria/key_points_Qwen2.5-32B', help='输出目录')
    parser.add_argument('--tmp-dir', type=Path, default='tmp', help='输出目录')

    args = parser.parse_args()

    # 加载提示词模板
    template = load_yaml(args.prompt_template)
    system_prompt = template['model_key_insight']['system']
    user_prompt = template['model_key_insight']['task']
    problems = []
    if args.problem_file:
        problems.append(args.problem_file)
    elif args.problem_dir:
        for _ in args.problem_dir.iterdir():
            problems.append(_)

    problems = sorted(problems, key=lambda x: x.stem)
    contents: List[dict] = []
    for problem in problems:
        task_id = problem.stem
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

    all_prompts = prepare_batch_prompts(contents, user_prompt, system_prompt)
    print(all_prompts[0][1]['content'])
    model, sampling_params = load_llm(args.model_name_or_path, gpu_num=args.gpu_num)
    outputs_t = model.chat(all_prompts, sampling_params, use_tqdm=True)

    pred_lst = []
    for o_t in outputs_t:
        pred_lst.append(o_t.outputs[0].text)

    args.output_dir.mkdir(exist_ok=True, parents=True)
    args.tmp_dir.mkdir(exist_ok=True, parents=True)

    results = postprocess_output(contents, pred_lst)
    for result in results:
        write_json(result, args.tmp_dir / f"{result['task_id']}.json")
        pred = result['pred']
        if isinstance(pred, dict):
            write_json(pred, args.output_dir / f"{result['task_id']}.json")
        else:
            logger.warning(f"未找到核心要点 {result['task_id']}")
            print(f'{result["task_id"]}: {pred}')
    # 创建输出目录


if __name__ == "__main__":
    main()
