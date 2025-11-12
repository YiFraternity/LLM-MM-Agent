import os
import re
from pathlib import Path
import json
import argparse
from typing import List

from eval_utils import (
    load_tex_content,
    load_json,
    load_yaml,
    prepare_batch_prompts,
    load_llm,
    write_jsonl,
    latex_to_json,
    find_task_id_from_path,
)


def match_task_and_dimension(subsection, criteria_file, config):
    """
    Match subsection to task, subtask, and dimension.
    """
    with open(criteria_file, 'r', encoding='utf-8') as file:
        criteria = json.load(file)

    for task in config['tasks']:
        for subtask in task['subtasks']:
            for dimension in subtask['dimensions']:
                # Example matching logic: check if subsection contains dimension keyword
                if dimension in subsection:
                    return {
                        "task": task['task_name'],
                        "subtask": subtask['subtask_name'],
                        "dimension": dimension
                    }
    return None

def extract_content_from_sections(sections: list) -> list:
    """
    遍历一系列 section：
    - 遇到包含参考文献的 section 就停止
    - 优先提取所有 subsubsection
    - 若没有 subsubsection，则提取 subsection
    - 若没有 subsection，则提取 section
    - 保留层级信息
    """
    results = []

    for section in sections:
        # 如果包含参考文献，停止遍历
        if "参考文献" in section.get("title", "") or section.get("reference"):
            break

        content_infos = extract_content(section)
        results.extend(content_infos)

    return results


def extract_content(section: dict) -> list:
    """
    单个 section 的提取规则：
    - 所有 subsubsection > subsection > section
    - 返回列表，保留层级关系
    """
    collected = []

    # 1. 遍历 subsection
    for subsection in section.get("children", []):
        subsub_collected = []
        # 2. 遍历所有 subsubsection
        for subsubsection in subsection.get("children", []):
            if "content" in subsubsection and subsubsection["content"].strip():
                subsub_collected.append({
                    "level": "subsubsection",
                    "title_path": [section.get("title", ""), subsection.get("title", ""), subsubsection.get("title", "")],
                    "content": subsubsection["content"]
                })
        # 如果有 subsubsection 内容，加入结果（注意这里不退回 subsection）
        if subsub_collected:
            collected.extend(subsub_collected)
        # 如果没有 subsubsection 内容，考虑 subsection
        elif "content" in subsection and subsection["content"].strip():
            collected.append({
                "level": "subsection",
                "title_path": [section.get("title", ""), subsection.get("title", "")],
                "content": subsection["content"]
            })

    # 3. 如果既没有 subsubsection 也没有 subsection，就取 section
    if not collected and "content" in section and section["content"].strip():
        collected.append({
            "level": "section",
            "title_path": [section.get("title", "")],
            "content": section["content"]
        })

    return collected

def get_criteria_str(criteria_file: str) -> str:
    criteria = load_json(criteria_file)

    criteria_temp_str = ''
    for key, value in criteria.items():
        id = key.split('_')[-1]
        criteria_temp_str += f'- [子任务{id}]：' + value['subtask'] + '\n'
        _criteria = value['criteria']
        # dimesonsions = '\n    * '.join(_criteria.keys())
        dimension_str = ''
        for dim_key, dim_value in _criteria.items():
            dimension_str += f'    * [{dim_key}]\n'
            for crit in dim_value:
                desp = crit["description"]
                dimension_str += f'        + {desp}\n'
        criteria_temp_str += '- [评估维度]\n' + dimension_str + '\n'
    return criteria_temp_str


def main():
    parser = argparse.ArgumentParser(description='Batch section classification for LaTeX files')
    parser.add_argument('--latex-file', default=None, type=Path,
                        help='LaTeX file to process')
    parser.add_argument('--latex-dir', default='MMBench/CPMCM/BestPaper', type=Path,
                        help='Directory to search for LaTeX files')
    parser.add_argument('--latex-file-name', default='1.tex',
                        help='LaTeX file name, if used latex-dir')
    parser.add_argument('--criteria-dir', default='MMBench/CPMCM/criteria', type=Path,
                        help='Criteria JSON directory')
    parser.add_argument('--latex-files', nargs='+', default=None, type=Path,
                        help='List of LaTeX files to process')
    parser.add_argument('--prompt-template-file', default='eval/prompts/section_classification.yaml',
                        help='YAML file containing prompt templates')
    parser.add_argument('--output', default='eval/output/1_bestpaper_section_classification', type=Path,
                        help='Output JSONL file path (appended)')
    parser.add_argument('--model-name', default='/group_homes/our_llm_domain/home/share/open_models/Qwen/Qwen2.5-32B-Instruct',
                        help='Model name or path for vLLM')
    parser.add_argument('--gpu-num', type=int, default=4, help='Number of GPUs for tensor parallelism')
    parser.add_argument('--max-content-chars', type=int, default=8192, help='Max chars of content to include in prompt')
    args = parser.parse_args()

    # 1) Resolve LaTeX files to process
    latex_files: List[Path] = []
    if args.latex_file:
        latex_files.append(args.latex_file)
    elif args.latex_files:
        latex_files.extend(args.latex_files)
    elif args.latex_dir and args.latex_file_name:
        latex_files = [(task_dir / args.latex_file_name) for task_dir in args.latex_dir.iterdir() if task_dir.is_dir()]

    prompts_yaml = load_yaml(args.prompt_template_file)
    classification_prompt = prompts_yaml['classification']

    total_items = 0
    batch_prompts, all_content_infos = [], []
    for latex_file in latex_files:
        task_id = find_task_id_from_path(latex_file)
        criteria_file = args.criteria_dir / f'{task_id}.json'
        if not os.path.exists(criteria_file):
            print(f"Criteria file {criteria_file} not found, skipping.")
            continue
        if not os.path.exists(latex_file):
            print(f"LaTeX file {latex_file} not found, skipping.")
            continue
        criteria_str = get_criteria_str(criteria_file)
        latex_content = load_tex_content(latex_file)
        sections = latex_to_json(latex_content)

        contents = extract_content_from_sections(sections)

        for content_info in contents:
            content_info['subtasks_dimensions'] = criteria_str
            cur_section_path = '->'.join(content_info['title_path'])
            content_info['section_content'] = '当前章节路径：' \
                + cur_section_path + '\n' \
                + '内容：' \
                + content_info['content'][:args.max_content_chars]
            # 附加文件元数据，便于追踪
            content_info['source_latex_file'] = Path(criteria_file).stem

        if not contents:
            continue

        one_latex_prompt = prepare_batch_prompts(contents, classification_prompt)
        batch_prompts.extend(one_latex_prompt)
        all_content_infos.extend(contents)

    model, sampling_params = load_llm(args.model_name, gpu_num=args.gpu_num)
    outputs = model.chat(batch_prompts, sampling_params, use_tqdm=True)

    raw_outputs = [output.outputs[0].text for output in outputs]

    assert len(all_content_infos) == len(raw_outputs), "Number of content infos and raw outputs must match."
    for content_info, raw_output in zip(all_content_infos, raw_outputs):
        content_info['model_output'] = raw_output
        del content_info['subtasks_dimensions']

    os.makedirs(args.output, exist_ok=True)
    cleared_outputs = set()
    for content_info in all_content_infos:
        output_path = os.path.join(args.output, content_info['source_latex_file'] + '.jsonl')
        if output_path not in cleared_outputs:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            open(output_path, 'w', encoding='utf-8').close()
            cleared_outputs.add(output_path)
        write_jsonl(content_info, output_path)
    total_items += len(all_content_infos)
    print(f"Processed total items: {len(all_content_infos)}")


if __name__ == "__main__":
    main()
