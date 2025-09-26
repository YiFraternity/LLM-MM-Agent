import json
import sys

from utils import (
    load_tex_content,
    load_json,
    load_yaml,
    clean_json_txt,
    batch_standardize_json,
    prepare_batch_prompts,
    load_llm,
    write_jsonl,
)

sys.path.append('/home/yhliu/MathModeling/LLM-MM-Agent')
from MMAgent.utils.convert_format import latex_to_json



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
    latex_file = 'output/deepseek-chat-v3.1:free/CPMCM/MM-Agent/2010_D_20250916-234916/latex/solution.tex'
    latex_content = load_tex_content(latex_file)
    sections = latex_to_json(latex_content)

    criteria_file = 'MMBench/CPMCM/criteria/2010_D.json'
    criteria_str = get_criteria_str(criteria_file)

    contents = extract_content_from_sections(sections)
    for content_info in contents:
        content_info['subtasks_dimensions'] = criteria_str
        cur_section_path = '->'.join(content_info['title_path'])
        content_info['section_content'] = '当前章节路径：' \
            + cur_section_path + '\n' \
            + '内容：' \
            + content_info['content'][:2000]  # 截断，防止过长

    prompt_template_file = 'eval/section_classification.yaml'
    prompts = load_yaml(prompt_template_file)
    classfication_prompt = prompts['classification']
    batch_prompts = prepare_batch_prompts(contents, classfication_prompt)

    model_config = {
        'model_name': '/home/share/models/modelscope/Qwen/Qwen2.5-32B-Instruct',
        'gpu_num': 4,
    }
    model, sampling_params = load_llm(model_config['model_name'], gpu_num=model_config['gpu_num'])
    outputs = model.chat(batch_prompts, sampling_params, use_tqdm=True)

    # 收集所有需要标准化的输出
    raw_outputs = [output.outputs[0].text for output in outputs]

    # 批量标准化JSON输出
    cleaned_outputs = batch_standardize_json(raw_outputs, model)

    # 更新结果
    for content_info, raw_output, cleaned_output in zip(contents, raw_outputs, cleaned_outputs):
        content_info['model_output'] = cleaned_output if isinstance(cleaned_output, dict) else {}
        content_info['raw_output'] = raw_output

    write_jsonl(contents, 'eval/section_classification_output.jsonl')
    print("Extracted Contents:")


if __name__ == "__main__":
    main()
