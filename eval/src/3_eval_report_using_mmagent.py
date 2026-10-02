"""
本文件用于对子任务进行评估，使用 OpenAI API
"""

import argparse
from pathlib import Path
import logging
import json
import re

from dotenv import load_dotenv
load_dotenv(override=True)

from llm.llm import LLM
from eval_utils import (
    clean_json_txt,
    find_task_id_from_path,
    load_tex_content,
    write_json,
    load_json,
    load_yaml,
    populate_template,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_task_understanding_all(solution_data: dict, criteria_dir: Path, task_id: str) -> str:
    """
    Extract and format 'task_understanding' from the corresponding criteria JSON file.
    Combines understanding from all subtasks into a single text block.
    """
    criteria_file = criteria_dir / f"{task_id}.json"
    if not criteria_file.exists():
        logger.warning(f"Criteria file not found for task {task_id} at {criteria_file}")
        return "No specific task understanding constraints provided."

    try:
        criteria_data = load_json(criteria_file)
        subtasks = criteria_data.get("subtask", {})

        understanding_texts = []
        def subtask_sort_key(item):
            try:
                return (0, int(item[0]))
            except (TypeError, ValueError):
                return (1, str(item[0]))

        for sub_id, sub_info in sorted(subtasks.items(), key=subtask_sort_key):
            if not isinstance(sub_info, dict):
                continue

            # 兼容两种 criteria schema。当前 CPMCM 文件使用嵌套结构：
            # subtask.<id>.criteria.task_understanding。
            criteria_block = sub_info.get("criteria", {})
            if not isinstance(criteria_block, dict):
                criteria_block = {}
            tu = (
                sub_info.get("task_understanding")
                or criteria_block.get("task_understanding")
                or {}
            )
            if not tu:
                continue

            text = f"【子任务 {sub_id} 约束】\n"
            text += f"- 核心目标: {tu.get('core_goal', 'N/A')}\n"
            text += f"- 预期输出: {tu.get('expected_output', 'N/A')}\n"
            text += f"- 关键输入与约束: {tu.get('key_inputs_constraints', 'N/A')}\n"
            text += f"- 建模类型: {tu.get('modeling_type', 'N/A')}\n"
            text += f"- 流程角色: {tu.get('role_in_pipeline', 'N/A')}\n"
            text += f"- 关键假设: {tu.get('assumptions', 'N/A')}\n"
            understanding_texts.append(text)

        if not understanding_texts:
            raise ValueError(
                f"Criteria file {criteria_file} contains subtasks but no readable "
                "task_understanding blocks."
            )

        return "\n\n".join(understanding_texts)
    except Exception as e:
        logger.error(f"Error loading criteria for {task_id}: {e}")
        raise RuntimeError(
            f"Failed to load task understanding constraints for {task_id}"
        ) from e

def extract_task_analysis(solution_data: dict) -> str:
    """
    Extract task descriptions for problem analysis stage.
    Equivalent to the logic in generate_problem_analysis_prompt.
    """
    task_analyses = []
    task_number = 1

    for _, task in solution_data.items():
        if not isinstance(task, dict):
            continue

        task_analysis = task.get("task_description", "").strip()
        if task_analysis:
            task_analyses.append(f"**Task {task_number}**: {task_analysis}")
        else:
            task_analyses.append(f"**Task {task_number}**: No task analysis content.")

        task_number += 1

    return "\n\n".join(task_analyses)

def extract_modeling_analysis(solution_data: dict) -> str:
    """
    Extract modeling analysis for rigor and rationality stage.
    Equivalent to the logic in generate_modeling_rigorousness_prompt.
    """
    task_analyses = []
    task_number = 1

    for _, task in solution_data.items():
        if not isinstance(task, dict):
            continue

        task_analysis = task.get("task_analysis", "").strip()
        if task_analysis:
            task_analyses.append(f"**Task {task_number}**: {task_analysis}")
        else:
            task_analyses.append(f"**Task {task_number}**: No task analysis content.")

        task_number += 1

    return "\n\n".join(task_analyses)

def extract_modeling_process(solution_data: dict) -> str:
    """
    Extract mathematical modeling process for practicality & scientificity stage.
    Equivalent to the logic in generate_practicality_and_scientificity_prompt.
    """
    task_analyses = []
    task_number = 1

    for _, task in solution_data.items():
        if not isinstance(task, dict):
            continue

        task_analysis = task.get("mathematical_modeling_process", "").strip()
        if task_analysis:
            task_analyses.append(f"**Task {task_number}**: {task_analysis}")
        else:
            task_analyses.append(f"**Task {task_number}**: No task analysis content.")

        task_number += 1

    return "\n\n".join(task_analyses)

def extract_result_analysis(solution_data: dict) -> str:
    """
    Extract result analysis and bias discussion.
    Equivalent to the logic in generate_result_and_bias_analysis_prompt.
    """
    task_analyses = []
    task_number = 1

    # 提取代码执行验证结果
    code_verification = solution_data.get("code_verification", {})

    for _, task in solution_data.items():
        if not isinstance(task, dict):
            continue

        answer_analysis = (
            task.get("subtask_outcome_analysis")
            or task.get("answer", "")
        ).strip()

        task_content = f"**Task {task_number}**: "
        if answer_analysis:
            task_content += answer_analysis
        else:
            task_content += "No task analysis content."

        # 如果有对应的代码执行结果，将其附加到分析中
        code_file = f"main{task_number}.py"
        if code_file in code_verification:
            v_res = code_verification[code_file]
            task_content += f"\n\n[Code Execution Result for {code_file}]\n"
            task_content += "Status: " + ("Success" if v_res["success"] else "Failed") + "\n"
            if v_res["stdout"]:
                task_content += f"Stdout:\n```\n{v_res['stdout']}\n```\n"
            if v_res["stderr"]:
                task_content += f"Stderr:\n```\n{v_res['stderr']}\n```\n"

        task_analyses.append(task_content)

        task_number += 1

    return "\n\n".join(task_analyses)


def load_solution_json(file_path):
    """
    Load solution data

    Args:
        file_path (str): the file path

    Returns:
        dict: solution data in json format
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        tasks = data.get('tasks', [])
        task_dict = {f"task{i+1}": task for i, task in enumerate(tasks)}

        result = data.get('problem', {})
        result.update(task_dict)

        return result
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return {}
    except json.JSONDecodeError:
        print(f"Error: File {file_path} not in valid JSON format.")
        return {}

def build_prompt_fields(solution_data: dict, field_extractors: dict) -> dict:
    """
    Build prompt fields by applying extractors to solution_data.
    """
    prompt_fields = {}
    for field_name, extractor in field_extractors.items():
        try:
            prompt_fields[field_name] = extractor(solution_data)
        except Exception as e:
            raise RuntimeError(
                f"Failed to extract field '{field_name}': {e}"
            )
    return prompt_fields


def evaluate_math_modeling(
    llm,
    solution_path: Path,
    prompt_templates_dict: dict,
    final_path: Path,
    tmp_path: Path,
    criteria_dir: Path,
    system_prompt: str = "You are a helpful AI assistant.",
):
    """
    Evaluate a mathematical modeling report using staged JSON-based prompts.

    Args:
        llm: initialized LLM instance
        solution_path (Path): path to solution.json
        prompt_templates_dict: prompt templates
        final_path: output final results
        tmp_path: temporary results cache
        criteria_dir (Path): path to criteria directory containing task_understanding
    """

    task_id = find_task_id_from_path(solution_path)

    final_results = load_json(final_path)
    tmp_results = load_json(tmp_path)
    assert isinstance(final_results, dict)
    assert isinstance(tmp_results, dict)

    solution_data = load_solution_json(solution_path)
    if not solution_data:
        raise RuntimeError(f"Cannot load solution file: {solution_path}")

    # 提前解析并提取所有的 task_understanding
    task_understanding_all = extract_task_understanding_all(solution_data, criteria_dir, task_id)

    # Load code verification results if they exist
    verification_path = solution_path.parent / "code_verification.json"
    if verification_path.exists():
        try:
            with open(verification_path, 'r', encoding='utf-8') as f:
                code_verification = json.load(f)
            solution_data["code_verification"] = code_verification
            logger.info(f"Loaded code verification results from {verification_path}")
        except Exception as e:
            logger.warning(f"Failed to load code verification: {e}")

    STAGES = {
        "problem_analysis": {
            "template_key": "problem_analysis_prompt",
            "field_extractors": {
                "background": lambda d: d.get("background", ""),
                "requirements": lambda d: d.get("problem_requirement", ""),
                "task_analysis": extract_task_analysis,
                "task_understanding_all": lambda d: task_understanding_all,
            },
        },
        "modeling_rigor": {
            "template_key": "modeling_rigor_prompt",
            "field_extractors": {
                "background": lambda d: d.get("background", ""),
                "requirements": lambda d: d.get("problem_requirement", ""),
                "modeling_analysis": extract_modeling_analysis,
                "task_understanding_all": lambda d: task_understanding_all,
            },
        },
        "practicality_scientificity": {
            "template_key": "practicality_scientificity_prompt",
            "field_extractors": {
                "background": lambda d: d.get("background", ""),
                "requirements": lambda d: d.get("problem_requirement", ""),
                "modeling_process": extract_modeling_process,
                "task_understanding_all": lambda d: task_understanding_all,
            },
        },
        "result_bias": {
            "template_key": "result_bias_prompt",
            "field_extractors": {
                "background": lambda d: d.get("background", ""),
                "requirements": lambda d: d.get("problem_requirement", ""),
                "modeling_report": extract_result_analysis,
                "task_understanding_all": lambda d: task_understanding_all,
            },
        },
    }

    for stage_name, stage_cfg in STAGES.items():

        if stage_name in final_results:
            continue

        if stage_name in tmp_results:
            final_results[stage_name] = tmp_results[stage_name]
            write_json(final_results, final_path)
            continue

        template = prompt_templates_dict[stage_cfg["template_key"]]

        prompt_fields = build_prompt_fields(
            solution_data,
            stage_cfg["field_extractors"]
        )

        # 这里做一个容错：只有当模板字符串中包含对应占位符时，才保留该字段。
        # 对于老的 prompt 模板，即使我们传了 task_understanding_all 进去，
        # populate_template 也可以正常工作（或者你可以用 jinja2 也能忽略多余变量）
        user_prompt = populate_template(template, prompt_fields)

        try:
            response = llm.generate(
                prompt=user_prompt,
                system=system_prompt,
            )
            response = clean_json_txt(response)

            tmp_results[stage_name] = response
            final_results[stage_name] = response

            write_json(tmp_results, tmp_path)
            write_json(final_results, final_path)

        except Exception as e:
            raise RuntimeError(f"[{task_id}] Stage {stage_name} failed: {e}")


    # ---------- usage logging ----------
    usage = llm.get_total_usage()
    llm.clear_usage()
    return final_results, usage

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='评估子任务')
    parser.add_argument('--task-path', default='output/Qwen2.5-7B-Instruct/CPMCM/MM-Agent/2005_B_20251127-070402/json/2005_B.json', type=Path,
                        help='子任务目录')
    parser.add_argument('--eval-prompt', default='eval/prompts/mmagent_eval_prompt.yaml', type=Path,
                        help='评估提示词 YAML 路径')
    parser.add_argument('--output-dir', default='eval/output/results', type=Path, help='输出 Json 文件目录')
    parser.add_argument('--tmp-dir', default='tmp/eval/results', type=Path, help='输出 Json 文件目录')
    parser.add_argument('--ai-model-name', default='Qwen2.5-7B-Instruct', help='AI 模型名称，例如 Qwen2.5-7B-Instruct')

    parser.add_argument('--openai-model', default='gpt-5-mini-ca', help='OpenAI 模型名称，例如 gpt-4o-mini')
    parser.add_argument('--openai-log-dir', default='eval/logs/openai', type=Path,
                        help='OpenAI API 日志目录')
    parser.add_argument('--criteria-dir', default='MMBench/CPMCM/criteria', type=Path,
                        help='评估标准 JSON 目录 (用于提取 task_understanding)')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    task_path = args.task_path
    task_id = find_task_id_from_path(task_path)
    llm = LLM(model_name=args.openai_model)
    prompts_dict = load_yaml(args.eval_prompt)
    system_prompt = prompts_dict['system_prompt']

    output_dir = args.output_dir
    tmp_dir = args.tmp_dir
    output_dir.mkdir(exist_ok=True, parents=True)
    tmp_dir.mkdir(exist_ok=True, parents=True)
    output_path = output_dir/task_id /f'{args.ai_model_name}.json'
    tmp_path = tmp_dir/task_id /f'{args.ai_model_name}.json'

    evaluate_math_modeling(
        llm,
        solution_path=task_path,
        prompt_templates_dict=prompts_dict,
        final_path=output_path,
        tmp_path=tmp_path,
        criteria_dir=args.criteria_dir,
        system_prompt=system_prompt,
    )
