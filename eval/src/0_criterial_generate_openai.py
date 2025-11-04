"""
This script generates the core criteria of a problem.

Usage:
    python eval/src/0_criterial_generate_openai.py \
        --problem-dir MMBench/CPMCM/problem \
        --criterial-prompt eval/prompts/criterial_generate.yaml \
        --config-path config.yaml \
        --output-dir MMBench/CPMCM/criteria
        --tmp-dir tmp \
        --model-name gpt-5-mini
"""

import json
import argparse
import logging
from pathlib import Path
import traceback
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    RetryCallState,
)
from dotenv import load_dotenv

from utils import (
    write_json,
    populate_template,
    load_json,
    clean_json_txt,
    find_task_id_from_path,
    load_yaml,
)

from agent.data_description import DataDescription
from agent.problem_analysis import ProblemUnderstanding
from agent.coordinator import Coordinator
from agent.problem_decompse import ProblemDecompose
from llm.llm import LLM
from prompt.template import PROBLEM_PROMPT

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv(override=True)


def get_problem(problem_path: Path, llm: LLM) -> tuple[str, dict]:
    problem = load_json(problem_path)
    data_description = problem.get('dataset_description', {})
    ds = DataDescription(llm)

    if data_description:
        data_path = problem['dataset_path'][:10]
        variable_description = problem['variable_description']
        data_summary = ds.summary(data_description=str(data_description) + '\n' + str(variable_description))
        data_summary = f'Dataset Path:\n{data_path}\n\nData Description:\n{data_summary}'
    else:
        data_summary = ''

    problem['data_summary'] = data_summary
    problem['data_description'] = data_description

    if problem.get('addendum', ''):
        addendum = f"Addendum: \n{problem['addendum']}"
    else:
        addendum = ''
    addendum = addendum[:300]

    problem_str = PROBLEM_PROMPT.format(
        problem_background=problem['background'],
        problem_requirement=problem['problem_requirement'],
        addendum=addendum,
        data_summary=data_summary
    ).strip()
    problem['problem_str'] = problem_str
    return problem_str, problem


def try_load_backup(output_dir: Path, task_id: str):
    """尝试加载最近一次备份"""
    backup_file = output_dir / f"{task_id}.json"
    if not backup_file.exists():
        return None
    logger.info(f"🧩 检测到备份文件，尝试从 {backup_file} 恢复。")

    with open(backup_file, "r", encoding="utf-8") as f:
        return json.load(f)


def backup_on_criteria_med(task_id:str, problem: dict, solution: dict,
                           dependency_dict:dict, output_dir: Path, error=None) -> None:
    """
    在任务处理过程中，根据给定条件将当前状态（包括问题、解法、依赖关系及错误信息）备份为 JSON 文件。

    该函数主要用于容错和调试：当任务成功完成或发生异常时，自动将关键上下文信息持久化到指定目录。
    若传入了 error（非 None），则记录为错误状态并输出警告日志；否则视为正常完成并记录信息日志。

    Args:
        task_id (str): 任务唯一标识符，用作备份文件名。
        problem (dict): 描述当前问题的字典。
        solution (dict): 当前生成的解决方案字典。
        dependency_dict (dict): 依赖关系字典。
        output_dir (Path): 备份文件保存的目录路径。
        error (Exception): 若任务出错，传入异常对象；若为 None 表示无错误。

    Results:
        - 在 output_dir 下创建以 task_id 命名的 .json 备份文件。
        - 记录 INFO 或 WARNING 级别日志。

    """
    backup_path = output_dir / f"{task_id}.json"
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    backup_data = {
        "task_id": task_id,
        "problem": problem,
        "solution": solution,
        "dependency_dict": dependency_dict,
        "error": str(error),
        "traceback": traceback.format_exc()
    }

    with open(backup_path, "w", encoding="utf-8") as f:
        json.dump(backup_data, f, ensure_ascii=False, indent=2)
    if error:
        logger.warning(f"⚠️ 出错，已自动备份到：{backup_path}")
    else:
        logger.info(f"🧩 任务完成，已自动备份到：{backup_path}")

def log_retry(retry_state: RetryCallState):
    logger.warning(f"🔁 {retry_state.fn.__name__} 第 {retry_state.attempt_number} 次尝试失败：{retry_state.outcome.exception()}")

def robust_retry(func):
    return retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=20),
        retry=retry_if_exception_type(Exception),
        reraise=True,
        after=log_retry
    )(func)

# 在每个阶段上单独应用
@robust_retry
def run_problem_understanding(llm, problem_str):
    pu = ProblemUnderstanding(llm)
    return pu.analysis(problem_str, round=0)

@robust_retry
def run_modeling(llm, problem_str, problem_analysis):
    pu = ProblemUnderstanding(llm)
    return pu.modeling(problem_str, problem_analysis, round=0)

@robust_retry
def run_decomposition(llm, problem_str, problem_analysis, modeling_solution, problem_type, config) -> list[str]:
    pd = ProblemDecompose(llm)
    return pd.decompose(problem_str, problem_analysis, modeling_solution, problem_type, config['tasknum'])

@robust_retry
def run_dependency_analysis(llm: LLM, problem_str, problem_analysis, modeling_solution, task_descriptions, with_code) -> dict[str, list[int]]:
    coordinator = Coordinator(llm)
    task_dependency_analysis = coordinator.analyze(
            len(task_descriptions),
            problem_str,
            problem_analysis,
            modeling_solution,
            task_descriptions,
            with_code
        )
    dependency_DAG_string = coordinator.dag_construction(
        len(task_descriptions),
        problem_str,
        problem_analysis,
        modeling_solution,
        task_descriptions,
        task_dependency_analysis
    )
    dependency_DAG = clean_json_txt(dependency_DAG_string)
    if not isinstance(dependency_DAG, dict):
        raise ValueError(
            f"依赖分析返回类型错误，期望dict但得到 {type(dependency_DAG)}，原始内容为：{str(dependency_DAG)}"
        )

    if not dependency_DAG:
        raise ValueError("依赖分析返回空字典，触发重试。")

    return dependency_DAG


def problem_analysis(llm: LLM, problem_path: Path, config: dict, tmp_dir:Path) -> tuple[dict, dict, bool, dict]:
    """
    对给定问题进行多阶段分析，包括问题理解、建模、任务分解和依赖分析，并支持从中断处恢复执行。

    该函数按顺序执行四个核心步骤，每步结果会缓存到全局状态（GLOBAL_STATE）并自动备份到临时目录，
    以便在失败后恢复。若存在已有备份，则从最后一个完成的阶段继续执行。

    Args:
        llm: MMAgent.llm.LLM
        problem_path: 问题文件路径，从中加载问题内容。
        config: 配置字典，包含任务分解等阶段所需的参数。
        tmp_dir (Path): 临时目录路径，用于读取或写入执行状态备份。

    Returns:
        problem (dict): 解析后的问题结构。
        dependency_dict (dict): 任务间的依赖关系字典。
        with_code (bool): 是否包含代码数据集（根据 problem['dataset_path'] 判断）。
        solution (dict): 各阶段生成的解决方案中间结果。
    """
    global GLOBAL_STATE
    problem_str, problem = get_problem(problem_path, llm)
    GLOBAL_STATE["problem"] = problem

    task_id = find_task_id_from_path(problem_path)
    if not task_id:
        raise ValueError(f"无法从 {problem_path} 中解析出 task_id。")
    problem_type = task_id.split('_')[-1]

    # 尝试恢复
    solution = GLOBAL_STATE.get("solution", {})
    if not solution:
        solution = {
            'problem_background': problem.get('background'),
            'problem_requirement': problem.get('problem_requirement'),
        }
        GLOBAL_STATE["solution"] = solution

    backup_data = try_load_backup(tmp_dir, task_id)
    if backup_data:
        GLOBAL_STATE.update(backup_data)
        solution = GLOBAL_STATE.get('solution', {})
        start_idx = detect_start_index(solution)
        logger.info(f"🔄 将从阶段 {start_idx} 恢复/开始执行。")
    else:
        start_idx = 0

    # === Step 1: Problem Understanding ===
    if start_idx <= 0:
        problem_analysis = run_problem_understanding(llm, problem_str)
        solution['problem_analysis'] = problem_analysis
        GLOBAL_STATE["solution"]["problem_analysis"] = problem_analysis
        logger.info('1️⃣  Step 1 finished.')

    # === Step 2: Modeling ===
    if start_idx <= 1:
        modeling_solution = run_modeling(llm, problem_str, problem_analysis)
        solution['modeling_solution'] = modeling_solution
        GLOBAL_STATE["solution"]["modeling_solution"] = modeling_solution
        logger.info('2️⃣  Step 2 finished.')

    # === Step 3: Decomposition ===
    if start_idx <= 2:
        task_descriptions = run_decomposition(llm, problem_str, problem_analysis, modeling_solution, problem_type, config)
        solution['task_descriptions'] = task_descriptions
        GLOBAL_STATE["solution"]['task_descriptions'] = task_descriptions
        logger.info('3️⃣  Step 3 finished.')

    # === Step 4: Dependency Analysis ===
    with_code = len(problem['dataset_path']) > 0
    if start_idx <= 3:
        dependency_dict = run_dependency_analysis(
            llm,
            problem_str,
            problem_analysis,
            modeling_solution,
            task_descriptions,
            with_code
        )
        GLOBAL_STATE["dependency_dict"] = dependency_dict
        logger.info('4️⃣  Step 4 finished.')
    else:
        dependency_dict = GLOBAL_STATE.get('dependency_dict', {})

    return problem, dependency_dict, with_code, solution

def detect_start_index(solution: dict) -> int:
    """
    根据 solution 内容决定从哪个阶段开始恢复
    返回阶段索引：
      0 - problem_understanding
      1 - modeling
      2 - decomposition
      3 - dependency_analysis
      4 - finished
    """
    if not solution:
        return 0
    if 'problem_analysis' not in solution:
        return 0
    if 'modeling_solution' not in solution:
        return 1
    if 'task_descriptions' not in solution:
        return 2
    # dependency_dict 存在于 GLOBAL_STATE，而不总在 solution 中
    if GLOBAL_STATE.get('dependency_dict') is None:
        return 3
    return 4

@robust_retry
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--problems", type=Path, nargs="+", default=None)
    parser.add_argument("--problem-dir", type=Path, default="MMBench/CPMCM/problem1")
    parser.add_argument("--criterial-prompt", type=Path, default="eval/prompts/criterial_generate.yaml")
    parser.add_argument("--config-path", type=Path, default="config.yaml")
    parser.add_argument("--output-dir", type=Path, default="MMBench/CPMCM/criteria")
    parser.add_argument("--tmp-dir", type=Path, default="tmp/criteria")
    parser.add_argument('--model-name', type=str, default='Qwen2.5-32B-Instruct')
    args = parser.parse_args()

    problem_paths = []
    if args.problems:
        problem_paths = args.problems
    else:
        for f in args.problem_dir.iterdir():
            problem_paths.append(f)

    config = load_yaml(args.config_path)
    output_dir: Path = args.output_dir
    tmp_dir = args.tmp_dir

    llm = LLM()
    criterial_prompt = load_yaml(args.criterial_prompt)
    user_prompt = criterial_prompt['math_modeling_criteria_generator']['zh']['eval_dimension']
    system_prompt = criterial_prompt['math_modeling_criteria_generator']['system']

    for problem_path in problem_paths:
        task_id = find_task_id_from_path(problem_path)
        if not task_id:
            raise ValueError(f"无法从 {problem_path} 中解析出 task_id")
        GLOBAL_STATE = {
            "task_id": task_id,
            "problem": None,
            "solution": {},
            "dependency_dict": None
        }
        logger.info(f"🔎 开始处理问题：{problem_path}")

        try:
            problem, dependency_dict, with_code, solution = problem_analysis(llm, problem_path, config, tmp_dir)
            backup_on_criteria_med(
                task_id,
                problem,
                solution,
                dependency_dict,
                tmp_dir,
                None,
            )
            assert len(dependency_dict) == len(solution['task_descriptions'])
            dependency_dict = sorted(dependency_dict.items(), key=lambda x: int(x[0]))

            criterial_path = output_dir / f'{task_id}.json'
            criterial_dict = {
                "task_id": GLOBAL_STATE.get("task_id"),
                "problem": problem['problem_requirement'],
                "subtask": []
            }

            if criterial_path.exists():
                logger.info(f"🔄 检测到已存在的结果文件 {criterial_path}，尝试断点恢复。")
                with open(criterial_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
                done_ids = {item['subtask_id'] for item in existing.get('subtask', [])}
                criterial_dict = existing
            else:
                done_ids = set()

            for id, dependencies in dependency_dict:
                id_int = int(id)
                if id_int in done_ids:
                    logger.info(f"⏩ 跳过已完成任务 {id_int}")
                    continue

                dependencies = [int(dep) if not isinstance(dep, int) else dep for dep in dependencies]
                criterial = generate_criterial(
                    llm,
                    problem,
                    task_descriptions=solution['task_descriptions'],
                    subtask_id=id_int,
                    dependency=dependencies,
                    template=user_prompt,
                    system=system_prompt,
                )
                logger.info(f"🔍 子任务 {id} 处理完成。")
                criterial_dict['subtask'].append({
                    "subtask_id": id_int,
                    "subtask": solution['task_descriptions'][id_int-1],
                    "criteria": criterial
                })
                write_json(criterial_dict, output_dir/f'{task_id}.json', )
            logger.info("🔍 全流程运行完成。")

        except Exception as e:
            logger.error(f"❌ 执行失败：{e}")
            backup_on_criteria_med(
                task_id,
                GLOBAL_STATE["problem"],
                GLOBAL_STATE["solution"],
                GLOBAL_STATE["dependency_dict"],
                tmp_dir,
                e
            )
            logger.error("程序已安全备份后退出。")
            exit(1)
