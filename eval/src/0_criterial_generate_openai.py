"""
This script generates the core criteria of a problem.

Usage:
    python eval/src/0_criterial_generate_openai.py \
        --problem-dir MMBench/CPMCM/problem \
        --criterial-prompt eval/prompts/criterial_generate.yaml \
        --config-path config.yaml \
        --output-dir MMBench/CPMCM/criteria
        --subtask-dir subtask \
        --model-name gpt-5-mini
"""

import argparse
import logging
from pathlib import Path
from dotenv import load_dotenv

from eval_utils import (
    write_json,
    populate_template,
    load_json,
    clean_json_txt,
    find_task_id_from_path,
    load_yaml,
)
from llm.llm import LLM
from agent.coordinator import Coordinator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv(override=True)


def detect_subtask_finished(output_dir: Path, task_id: str, subtask_id: int) -> tuple[str, dict]:
    criteria_path = output_dir/f'{task_id}.json'
    if not criteria_path.exists():
        return 'unfinished', {}
    subtask_data = load_json(criteria_path)
    subtask_criteria = subtask_data.get('subtasks', {}).get(str(subtask_id), {})
    if not subtask_criteria:
        return 'unfinished', {}
    return 'finished', subtask_criteria

def run_dependency_analysis(llm: LLM, problem_str: str, task_descriptions: list[str], with_code: bool) -> dict[str, list[int]]:
    coordinator = Coordinator(llm)
    task_dependency_analysis = coordinator.analyze(
            len(task_descriptions),
            problem_str,
            task_descriptions,
            with_code
        )
    try:
        return coordinator.dag_construction(
            len(task_descriptions),
            problem_str,
            task_descriptions,
            task_dependency_analysis
        )
    except Exception as e:
        return {}

def generate_criterial(llm, problem: dict, task_descriptions: list[str], subtask_id: int, dependency: list[int], template:str, system: str = 'You are a helpful assistant.'):
    data = {
        'background': problem['background'],
        'question': problem['problem_requirement'],
        'subtask': task_descriptions[subtask_id - 1],
    }
    if dependency:
        dep = [task_descriptions[i-1] for i in dependency]
        previous_subtasks = '\n'.join(dep)
    else:
        previous_subtasks = None
    data['previous_subtasks'] = previous_subtasks

    prompt = populate_template(template, data)
    content = llm.generate(prompt, system, timeout=300)
    return clean_json_txt(content)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--problems", type=Path, nargs="+", default=None)
    parser.add_argument("--problem-dir", type=Path, default="MMBench/CPMCM/problem")
    parser.add_argument("--criterial-prompt", type=Path, default="eval/prompts/criterial_generate.yaml")
    parser.add_argument("--output-dir", type=Path, default="MMBench/CPMCM/criteria")
    parser.add_argument("--subtask-dir", type=Path, default="MMBench/CPMCM/subtask")
    parser.add_argument("--tmp-dir", type=Path, default="tmp/criteria")
    parser.add_argument('--model-name', type=str, default='Qwen2.5-32B-Instruct')
    return parser.parse_args()

def main(args):
    problem_paths = []
    if args.problems:
        problem_paths = args.problems
    else:
        for f in args.problem_dir.iterdir():
            problem_paths.append(f)

    output_dir: Path = args.output_dir
    subtask_dir = args.subtask_dir
    tmp_dir = args.tmp_dir
    tmp_dir.mkdir(exist_ok=True, parents=True)

    llm = LLM()
    criterial_prompt = load_yaml(args.criterial_prompt)
    user_prompt = criterial_prompt['math_modeling_criteria_generator']['zh']['eval_dimension']
    system_prompt = criterial_prompt['math_modeling_criteria_generator']['system']

    for problem_path in problem_paths:
        task_id = find_task_id_from_path(problem_path)
        if not task_id:
            raise ValueError(f"无法从 {problem_path} 中解析出 task_id")
        logger.info(f"🔎 开始处理问题：{problem_path}")

        subtasks_path = subtask_dir/f'{task_id}.json'
        subtasks = load_json(subtasks_path)
        if not subtasks:
            raise ValueError(f"无法从 {subtasks_path} 中解析出 subtasks") from None

        problem_str = subtasks.get('problem_str', '')
        task_descriptions = subtasks.get('task_descriptions', [])
        problem = subtasks.get('problem', {})
        with_code = len(problem.get('dataset_path', [])) > 0
        output_path = output_dir/f'{task_id}.json'
        if output_path.exists():
            result = load_json(output_path)
            if result:
                logger.info(f"✅  检测到已存在的结果文件 {output_path}")
                continue
        tmp_path = tmp_dir/f'{task_id}.json'
        dependency_dict = {}
        if tmp_path.exists() and tmp_path.stat().st_size > 0:
            result = load_json(tmp_path)
            dependency_dict = result.get('dependency_dict', {})
        if not dependency_dict:
            dependency_dict = run_dependency_analysis(llm, problem_str, task_descriptions, with_code)
            write_json({'dependency_dict': dependency_dict}, tmp_path)

        logger.info(f"dependency_dict: {dependency_dict}")
        logger.info(f"🔍 子任务依赖分析完成。")
        assert len(dependency_dict) == len(task_descriptions)
        dependency_dict = sorted(dependency_dict.items(), key=lambda x: int(x[0]))

        criterial_dict = {
            'task_id': task_id,
            'problem': problem['problem_requirement'],
            'subtask': {}
        }
        try:
            for id, dependencies in dependency_dict:
                id_int = int(id)
                dependencies = [int(dep) if not isinstance(dep, int) else dep for dep in dependencies]
                fin, _criteria = detect_subtask_finished(output_dir, task_id, id_int)
                if fin == 'finished':
                    criterial = _criteria
                else:
                    criterial = generate_criterial(
                        llm,
                        problem,
                        task_descriptions=task_descriptions,
                        subtask_id=id_int,
                        dependency=dependencies,
                        template=user_prompt,
                        system=system_prompt,
                    )
                logger.info(f"🔍 子任务 {id} 处理完成。")
                criterial_dict['subtask'][str(id_int)] = {
                    'subtask': task_descriptions[id_int - 1],
                    'criteria': criterial
                }
            logger.info("🔍 全流程运行完成。")
            write_json(criterial_dict, output_dir/f'{task_id}.json')
        except Exception as e:
            logger.error(f"🚫 处理子任务 {id} 时出错：{e}")
            write_json(criterial_dict, output_dir/f'{task_id}.json')

if __name__ == "__main__":
    args = parse_args()
    main(args)