
import json
import argparse
import glob
from pathlib import Path

from .utils import (
    load_tex_content,
    load_json,
    load_yaml,
    clean_json_txt,
    batch_standardize_json,
    prepare_batch_prompts,
    load_llm,
    write_jsonl,
    latex_to_json,
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
    parser.add_argument('--latex_files', nargs='*', default=None,
                        help='Glob patterns for LaTeX files (e.g., output/**/latex/solution.tex)')
    parser.add_argument('--input_dir', default=None,
                        help='Directory to search for LaTeX files (will use pattern **/latex/solution.tex)')
    parser.add_argument('--criteria_file', default='MMBench/CPMCM/criteria/2010_D.json',
                        help='Criteria JSON file')
    parser.add_argument('--prompt_template_file', default='eval/prompts/section_classification.yaml',
                        help='YAML file containing prompt templates')
    parser.add_argument('--output', default='eval/output/2010_D/section_classification_output.jsonl',
                        help='Output JSONL file path (appended)')
    parser.add_argument('--model_name', default='/home/share/models/modelscope/Qwen/Qwen2.5-32B-Instruct',
                        help='Model name or path for vLLM')
    parser.add_argument('--gpu_num', type=int, default=4, help='Number of GPUs for tensor parallelism')
    parser.add_argument('--max_content_chars', type=int, default=2000, help='Max chars of content to include in prompt')
    args = parser.parse_args()

    # 1) Resolve LaTeX files to process
    latex_files: list[str] = []
    if args.latex_files:
        for pattern in args.latex_files:
            latex_files.extend(glob.glob(pattern, recursive=True))
    elif args.input_dir:
        search_pattern = str(Path(args.input_dir) / '**' / 'latex' / 'solution.tex')
        latex_files.extend(glob.glob(search_pattern, recursive=True))
    else:
        # Sensible default: search whole repo for typical solution.tex paths
        default_pattern = str(Path('output') / '**' / 'latex' / 'solution.tex')
        latex_files.extend(glob.glob(default_pattern, recursive=True))

    latex_files = sorted(set(latex_files))
    if not latex_files:
        print('No LaTeX files found. Please specify --latex_files or --input_dir.')
        return

    # 2) Load shared resources once
    criteria_str = get_criteria_str(args.criteria_file)
    prompts_yaml = load_yaml(args.prompt_template_file)
    classification_prompt = prompts_yaml['classification']

    model, sampling_params = load_llm(args.model_name, gpu_num=args.gpu_num)

    total_items = 0
    for latex_file in latex_files:
        try:
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
                content_info['source_latex_file'] = latex_file
                content_info['source_name'] = Path(latex_file).parent.parent.parent.name if len(Path(latex_file).parts) >= 3 else Path(latex_file).name

            if not contents:
                continue

            batch_prompts = prepare_batch_prompts(contents, classification_prompt)
            outputs = model.chat(batch_prompts, sampling_params, use_tqdm=True)

            raw_outputs = [output.outputs[0].text for output in outputs]
            cleaned_outputs = batch_standardize_json(raw_outputs, model)

            for content_info, raw_output, cleaned_output in zip(contents, raw_outputs, cleaned_outputs):
                content_info['model_output'] = cleaned_output if isinstance(cleaned_output, dict) else {}
                content_info['raw_output'] = raw_output

            write_jsonl(contents, args.output)
            total_items += len(contents)
            print(f"Processed {latex_file}: {len(contents)} items")
        except Exception as e:
            print(f"Error processing {latex_file}: {e}")

    print(f"Done. Total items written: {total_items}. Output: {args.output}")


if __name__ == "__main__":
    main()
