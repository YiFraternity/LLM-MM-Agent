import os
from pathlib import Path
import logging
import shutil

from utils.utils import (
    read_json_file,
    find_task_id_from_path,
    try_load_backup,
    backup_solution,
)
from agent.data_description import DataDescription
from agent.problem_analysis import ProblemUnderstanding
from agent.coordinator import Coordinator
from agent.problem_decompse import ProblemDecompose
from prompt.template import PROBLEM_PROMPT

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def get_problem(problem_path, llm):
    problem = read_json_file(problem_path)
    data_description = problem.get('dataset_description', {})
    ds = DataDescription(llm)

    if data_description:
        data_path = problem['dataset_path']
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

    problem_str = PROBLEM_PROMPT.format(
        problem_background=problem['background'],
        problem_requirement=problem['problem_requirement'],
        addendum=addendum,
        data_summary=data_summary
    ).strip()
    problem['problem_str'] = problem_str
    return problem_str, problem

def detect_start_index(state: dict) -> int:
    """
    根据 solution 内容决定从哪个阶段开始恢复
    返回阶段索引：
      0 - problem_analysis
      1 - modeling_solution
      2 - task_descriptions
      3 - dependency_analysis
      4 - finished
    """
    if not state:
        return 0
    if 'problem_analysis' not in state:
        return 1
    if 'modeling_solution' not in state:
        return 2
    if 'task_descriptions' not in state:
        return 3
    if 'order' not in state:
        return 4
    return 5


def problem_analysis(
    llm,
    problem_path,
    config,
    dataset_path,
    output_dir,
    tmp_dir: Path
) -> tuple[dict, dict, Coordinator]:
    """
    进行问题分析，包含以下步骤：
    1. 问题理解
    2. 高层次建模
    3. 问题分解
    4. 任务依赖分析

    Returns：
        problem: dict: 问题描述字典
        solution: dict: 包含问题分析各阶段结果的字典
        coordinator: 协调者实例
    """
    task_id = find_task_id_from_path(problem_path)
    if not task_id:
        raise ValueError(f"无法从 {problem_path} 中解析出 task_id。")
    tmp_path = tmp_dir / task_id / 'analysis.json'
    problem_type = os.path.splitext(os.path.basename(problem_path))[0].split('_')[-1]
    try:
        backup_data = try_load_backup(tmp_path)
        if backup_data:
            solution = backup_data
        else:
            solution = {
                'tasks': [],
                'problem_background': problem['background'],
                'problem_requirement': problem['problem_requirement'],
            }
        start_idx = detect_start_index(solution)

        # Get problem
        if start_idx <=0:
            logger.info('🔍  Loading problem from %s', problem_path)
            problem_str, problem = get_problem(problem_path, llm)
            solution['problem_str'] = problem_str
            solution['problem'] = problem
        else:
            problem_str = solution.get('problem_str', '')
            problem = solution.get('problem', {})

        if start_idx <= 1:
            pu = ProblemUnderstanding(llm)
            problem_analysis = pu.analysis(problem_str, round=config['problem_analysis_round'])
            solution['problem_analysis'] = problem_analysis
            logger.info('1️⃣  Analysis - Problem Understanding - finished.')
        else:
            problem_analysis = solution.get('problem_analysis', '')

        # High level probelm understanding modeling
        if start_idx <= 2:
            pu = ProblemUnderstanding(llm)
            modeling_solution = pu.modeling(problem_str, problem_analysis, round=config['problem_modeling_round'])
            solution['modeling_solution'] = modeling_solution
            logger.info('2️⃣  Analysis - Modeling - finished.')
        else:
            modeling_solution = solution.get('modeling_solution', '')

        # Problem Decomposition
        if start_idx <= 3:
            pd = ProblemDecompose(llm)
            task_descriptions = pd.decompose_and_refine(problem_str, problem_analysis, modeling_solution, problem_type, config['tasknum'])
            solution['task_descriptions'] = task_descriptions
            logger.info('3️⃣  Analysis - Problem Decomposition - finished.')
        else:
            task_descriptions = solution.get('task_descriptions', [])

        # Task Dependency Analysis
        with_code = len(problem['dataset_path']) > 0
        if with_code:
            shutil.copytree(dataset_path, os.path.join(output_dir,'code'), dirs_exist_ok=True)
        solution['with_code'] = with_code

        coordinator = Coordinator(llm)
        if start_idx <= 4:
            coordinator = Coordinator(llm)
            order = coordinator.analyze_dependencies(problem_str, problem_analysis, modeling_solution, task_descriptions, with_code)
            order = [int(i) for i in order]
            solution['order'] = order
            logger.info('4️⃣  Analysis - Task Dependency Analysis - finished.')
        else:
            order = solution.get('order', [])
        backup_solution(tmp_path, solution)
        return problem, task_id, solution, coordinator
    except Exception as e:
        backup_solution(tmp_path, solution, e)
        raise e
