from pathlib import Path
from typing import List
import logging

from torch.jit.annotations import Dict
from agent.retrieve_method import MethodRetriever
from agent.task_solving import TaskSolver
from prompt.template import (
    TASK_ANALYSIS_APPEND_PROMPT,
    TASK_FORMULAS_APPEND_PROMPT,
    TASK_MODELING_APPEND_PROMPT,
)
from utils.utils import (
    try_load_backup,
    backup_solution,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def detct_start_index(state: dict) -> int:
    """
    根据 solution 内容决定从哪个阶段开始恢复
    返回阶段索引：
      0 - task-analysis
      1 - retrieve-method
      2 - task-formulas
    """
    if not state or 'task_analysis' not in state:
        return 0
    if 'retrieve_content' not in state:
        return 1
    if 'task_modeling_formulas' not in state or 'task_modeling_method' not in state:
        return 2
    return 3

def get_dependency_prompt(with_code, coordinator, subtask_id):
    task_dependency = [int(i) for i in coordinator.DAG[str(subtask_id)]]
    dependent_file_prompt = ""
    if len(task_dependency) > 0:
        dependency_prompt = f"""\
这是任务 {subtask_id}，它依赖于以下任务: {task_dependency}。
该任务的依赖关系分析如下: {coordinator.task_dependency_analysis[subtask_id - 1]}
"""
        for id in task_dependency:
            dependency_prompt += f"""\
---
# 任务 {id} 的描述:
{coordinator.memory[str(id)]['task_description']}
# 任务 {id} 的建模方法:
{coordinator.memory[str(id)]['mathematical_modeling_process']}
"""
            if with_code:
                dependency_prompt += f"""\
# 任务 {id} 的代码结构:
{coordinator.code_memory[str(id)]}
# 任务 {id} 的结果:
{coordinator.memory[str(id)]['solution_interpretation']}
---
"""
                dependent_file_prompt += f"""\
# 任务 {id} 的代码生成文件:
{coordinator.code_memory[str(id)]}
"""
                try:
                    coordinator.code_memory[str(id)]['file_outputs']
                except:
                    print(f"警告: 未检测到任务 {id} 的文件输出。")
            else:
                dependency_prompt += f"""\
# 任务 {id} 的结果:
{coordinator.memory[str(id)]['solution_interpretation']}
---
"""

    if len(task_dependency) > 0:
        task_analysis_prompt = dependency_prompt + TASK_ANALYSIS_APPEND_PROMPT
        task_formulas_prompt = dependency_prompt + TASK_FORMULAS_APPEND_PROMPT
        task_modeling_prompt = dependency_prompt + TASK_MODELING_APPEND_PROMPT
    else:
        task_analysis_prompt = ""
        task_formulas_prompt = ""
        task_modeling_prompt = ""
    return task_analysis_prompt, task_formulas_prompt, task_modeling_prompt, dependent_file_prompt


def mathematical_modeling(llm, task_id: str, subtask_id: int, problem: dict, task_descriptions: List[str], config: Dict, coordinator, with_code: bool, tmp_dir: Path):
    """
    进行子任务的数学建模，包含以下步骤：
        1. 任务分析
        2. 分层建模方法检索
        3. 任务建模
    Args:
        task_id: 任务编号 (2004_B)
        subtask_id: 子任务编号 (1, 2, ...)
        problem: 问题描述字典
        task_descriptions: 任务描述列表
        llm: 大语言模型实例
        config: 配置字典
        coordinator: 协调者实例
        with_code: 是否生成代码
        tmp_dir: 临时文件目录路径
    Returns:
        solution: 包含子任务数学建模各阶段结果的字典
            {
                'task_description': 任务描述字符串,
                'task_analysis': 任务分析字符串,
                'task_modeling_formulas': 任务建模公式字符串,
                'task_modeling_method': 任务建模方法字符串
            }
        dependent_file_prompt: 依赖文件提示字符串
    """
    tmp_path = tmp_dir / task_id / "mathematical_modeling.json"
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    backup_data = try_load_backup(tmp_path)
    if backup_data:
        all_solution = backup_data
        solution = all_solution.get(str(subtask_id), {'task_description': task_descriptions[subtask_id - 1]})
    else:
        all_solution = {}
        solution = {
            'task_description': task_descriptions[subtask_id - 1]
        }
    all_solution[str(subtask_id)] = solution

    ts = TaskSolver(llm)
    mr = MethodRetriever(llm, embed_model=config['embed_model'])
    task_analysis_prompt, task_formulas_prompt, task_modeling_prompt, dependent_file_prompt = get_dependency_prompt(with_code, coordinator, subtask_id)

    # 任务分析
    task_description = solution['task_description']
    start_idx = detct_start_index(solution)
    try:
        if start_idx <= 0:
            logger.info(f"开始任务分析，subtask_id: {subtask_id}")
            task_analysis = ts.analysis(task_analysis_prompt, task_description)
            solution['task_analysis'] = task_analysis
        else:
            task_analysis = solution.get('task_analysis', '')
        # 分层建模方法检索
        if start_idx <= 1:
            logger.info(f"开始分层建模方法检索，subtask_id: {subtask_id}")
            description_and_analysis = f'## 任务描述\n{task_description}\n\n## 任务分析\n{task_analysis}'
            top_modeling_methods = mr.retrieve_methods(description_and_analysis, top_k=config['top_method_num'])
            solution['retrieve_content'] = top_modeling_methods
        else:
            top_modeling_methods = solution.get('retrieve_content', [])
            # 任务建模
        # 任务建模
        if start_idx <= 2:
            logger.info(f"开始任务建模，subtask_id: {subtask_id}")
            task_modeling_formulas, task_modeling_method = ts.modeling(
                task_formulas_prompt,
                task_modeling_prompt,
                problem['data_description'],
                task_description,
                task_analysis,
                top_modeling_methods,
                round=config['task_formulas_round']
            )
            solution['task_modeling_formulas'] = task_modeling_formulas
            solution['task_modeling_method'] = task_modeling_method
        else:
            task_modeling_formulas = solution.get('task_modeling_formulas', '')
            task_modeling_method = solution.get('task_modeling_method', '')
        backup_solution(tmp_path, all_solution)
        return solution, dependent_file_prompt
    except Exception as e:
        backup_solution(tmp_path, all_solution, e)
        raise e
