"""
评估子任务 OpenAI API

使用 OpenAI API 评估给定的 LaTeX 文件中的子任务

评估维度：
- 问题识别
- 问题复述
- 假设建立
- 模型构建
- 模型求解
- 代码实现
- 结果分析

评估结果：
- 评估结果的 JSONL 文件

评估过程：
1. 读取 LaTeX 文件的内容
2. 对每个 section，提取其内容并构建对应的评估 prompt
3. 使用 OpenAI API 评估每个 section 的内容
4. 将评估结果保存到 JSONL 文件中
"""

import json
import argparse
import glob
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

# Initialize OpenAI client

from eval_utils import (
    load_tex_content,
    load_json,
    load_yaml,
    write_jsonl,
    latex_to_json,
    populate_template,
    call_openai_api,
)

def extract_content_from_sections(sections: List[Dict]) -> List[Dict]:
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

def extract_content(section: Dict) -> List[Dict]:
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
        dimension_str = ''
        for dim_key, dim_value in value['criteria'].items():
            dimension_str += f'    * [{dim_key}]\n'
            for crit in dim_value:
                desp = crit["description"]
                dimension_str += f'        + {desp}\n'
        criteria_temp_str += '- [评估维度]\n' + dimension_str + '\n'
    return criteria_temp_str



def process_content_with_openai(content_info: Dict, prompt_template: str, model: str = "gpt-4-turbo") -> Dict:
    """Process a single content item with OpenAI API."""
    # Prepare the prompt using template variables
    try:
        prompt = populate_template(prompt_template, content_info)
    except Exception as e:
        print(f"Error populating template: {e}")
        raise

    # Call OpenAI API
    raw_output = call_openai_api(prompt, model=model)

    try:
        json_start = raw_output.find('[')
        json_end = raw_output.rfind(']') + 1
        json_str = raw_output[json_start:json_end]

        model_output = json.loads(json_str)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Warning: Failed to parse model output as JSON: {e}")
        print(f"Raw output: {raw_output}")
        model_output = []

    return {
        **content_info,
        "model_output": model_output,
        "raw_output": raw_output
    }

def main():
    parser = argparse.ArgumentParser(description='Batch section classification for LaTeX files using OpenAI API')
    parser.add_argument('--latex_files', nargs='*', default='MMBench/CPMCM/BestPaper/2010_D/2.tex',
                       help='Glob patterns for LaTeX files (e.g., output/**/latex/solution.tex)')
    parser.add_argument('--input_dir', default=None,
                       help='Directory to search for LaTeX files (will use pattern **/latex/solution.tex)')
    parser.add_argument('--criteria_file', default='MMBench/CPMCM/criteria/2010_D.json',
                       help='Criteria JSON file')
    parser.add_argument('--prompt_template_file', default='eval/prompts/section_classification.yaml',
                       help='YAML file containing prompt templates')
    parser.add_argument('--output', default='eval/output/2010_D/2.model_generate_section_classification_output.jsonl',
                       help='Output JSONL file path (appended)')
    parser.add_argument('--model', default='gpt-4.1-mini',
                       help='OpenAI model to use (default: gpt-4.1-mini)')
    parser.add_argument('--max_content_chars', type=int, default=2000,
                       help='Max chars of content to include in prompt')
    parser.add_argument('--batch_size', type=int, default=5,
                       help='Number of sections to process in parallel (default: 5)')
    parser.add_argument('--request_delay', type=float, default=1.0,
                       help='Delay between API requests in seconds (default: 1.0)')

    args = parser.parse_args()

    # 1) Resolve LaTeX files to process
    latex_files: List[str] = []
    if args.latex_files:
        if isinstance(args.latex_files, str):
            latex_files = [args.latex_files]
        else:
            for pattern in args.latex_files:
                latex_files.extend(glob.glob(pattern, recursive=True))
    elif args.input_dir:
        search_pattern = str(Path(args.input_dir) / '**' / 'latex' / 'solution.tex')
        latex_files.extend(glob.glob(search_pattern, recursive=True))
    else:
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

    # Keep track of total processed
    total_items = 0
    for latex_file in latex_files:
        try:
            print(f"Processing {latex_file}...")
            latex_content = load_tex_content(latex_file)
            sections = latex_to_json(latex_content)

            # Extract content from sections
            contents = extract_content_from_sections(sections)

            # Prepare content for classification
            for content_info in contents:
                content_info['subtasks_dimensions'] = criteria_str
                cur_section_path = '->'.join(content_info['title_path'])
                content_info['section_content'] = (
                    f"当前章节路径：{cur_section_path}\n"
                    f"内容：{content_info['content'][:args.max_content_chars]}"
                )
                # 附加文件元数据，便于追踪
                content_info['source_latex_file'] = latex_file
                content_info['source_name'] = Path(latex_file).parent.parent.parent.name if len(Path(latex_file).parts) >= 3 else Path(latex_file).name

            if not contents:
                print(f"No content extracted from {latex_file}")
                continue

            # Process contents in batches
            results = []
            seen_ids = set()  # 用于在本次运行中去重（避免重复写入相同 section）
            for i in tqdm(range(0, len(contents), args.batch_size), desc="Processing sections"):
                batch = contents[i:i + args.batch_size]

                for content_info in batch:
                    try:
                        # 构造一个简单的唯一 id（基于文件路径 + section 路径）
                        cur_section_path = '->'.join(content_info.get('title_path', []))
                        unique_id = f"{content_info.get('source_latex_file', '')}:::{cur_section_path}"
                        if unique_id in seen_ids:
                            # 如果已经处理过，跳过（本次运行内去重）
                            print(f"Skipping duplicate section: {unique_id}")
                            continue

                        result = process_content_with_openai(
                            content_info,
                            prompt_template=classification_prompt,
                            model=args.model
                        )

                        # 标记已见并写入单条结果，避免重复
                        seen_ids.add(unique_id)
                        results.append(result)

                        # —— 关键修改：每次只写入当前这条 result（而不是整个 results）
                        write_jsonl(result, args.output)

                        # Add delay between requests to avoid rate limiting
                        time.sleep(args.request_delay)

                    except Exception as e:
                        print(f"Error processing section {content_info.get('title_path')}: {e}")
                        # 遇到单条异常：跳过该条，继续下一条。已写入的结果不会丢失（因为是逐条写入）
                        continue

            total_items += len(results)
            print(f"Processed {latex_file}: {len(results)} items")

        except Exception as e:
            print(f"Error processing {latex_file}: {e}")
            continue

    print(f"Done. Total items written: {total_items}. Output: {args.output}")

if __name__ == "__main__":
    main()
