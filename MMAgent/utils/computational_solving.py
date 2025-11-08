import os
from pathlib import Path
from utils.utils import (
    save_solution,
    try_load_backup,
    backup_solution,
)
from agent.task_solving import TaskSolver
from agent.create_charts import ChartCreator

def detect_start_index_with_code(state: dict) -> int:
    """
    根据 solution 内容决定从哪个阶段开始恢复
    返回阶段索引：
      0 - coding
    """
    if not state:
        return 0
    if 'task_code' not in state:
        return 0
    if 'code_structure' not in state:
        return 1
    if 'task_result' not in state:
        return 2
    if 'task_answer' not in state:
        return 3
    return 4

def detect_start_index(state: dict) -> int:
    """
    根据 solution 内容决定从哪个阶段开始恢复
    返回阶段索引：
      0 - result
      1 - answer
    """
    if not state:
        return 0
    if 'task_result' not in state:
        return 0
    if 'task_answer' not in state:
        return 1
    return 2

def computational_solving(llm, task_id, coordinator, with_code, problem, subtask_id, task_description, task_analysis, task_modeling_formulas, task_modeling_method, dependent_file_prompt, config, solution, name, output_dir, tmp_dir: Path):
    ts = TaskSolver(llm)
    cc = ChartCreator(llm)
    code_template = open(os.path.join('MMAgent/code_template','main{}.py'.format(subtask_id))).read()
    save_path = os.path.join(output_dir,'code/main{}.py'.format(subtask_id))
    work_dir = os.path.join(output_dir,'code')
    script_name = 'main{}.py'.format(subtask_id)

    tmp_path = tmp_dir / task_id / "computational_solving.json"
    backup_data = try_load_backup(tmp_path)
    if backup_data:
        all_state = backup_data
        state = all_state.get(str(subtask_id), {})
    else:
        state = {}
        all_state[str(subtask_id)] = state

    if with_code:
        start_idx = detect_start_index_with_code(state)
        try:
            if start_idx <= 0:
                task_code, is_pass, execution_result = ts.coding(problem['dataset_path'], problem['data_description'], problem['variable_description'], task_description, task_analysis, task_modeling_formulas, task_modeling_method, dependent_file_prompt, code_template, script_name, work_dir)
            else:
                task_code = state.get('task_code', '')
                is_pass = state.get('is_pass', False)
                execution_result = state.get('execution_result', '')
            if start_idx <= 1:
                code_structure = ts.extract_code_structure(subtask_id, task_code, save_path)
                state['code_structure'] = code_structure
            else:
                code_structure = state.get('code_structure', {})
            if start_idx <= 2:
                task_result = ts.result(task_description, task_analysis, task_modeling_formulas, task_modeling_method, execution_result)
                state['task_result'] = task_result
            else:
                task_result = state.get('task_result', '')
            if start_idx <= 3:
                task_answer = ts.answer(task_description, task_analysis, task_modeling_formulas, task_modeling_method, task_result)
                state['task_answer'] = task_answer
            else:
                task_answer = state.get('task_answer', '')
            backup_solution(tmp_path, all_state)
        except Exception as e:
            backup_solution(tmp_path, all_state, e)
            raise e
        task_dict = {
            'task_description': task_description,
            'task_analysis': task_analysis,
            'preliminary_formulas': task_modeling_formulas,
            'mathematical_modeling_process': task_modeling_method,
            'task_code': task_code,
            'is_pass': is_pass,
            'execution_result': execution_result,
            'solution_interpretation': task_result,
            'subtask_outcome_analysis': task_answer
        }
        coordinator.code_memory[str(subtask_id)] = code_structure
    else:
        start_idx = detect_start_index(state)
        try:
            if start_idx <= 0:
                task_result = ts.result(task_description, task_analysis, task_modeling_formulas, task_modeling_method)
                state['task_result'] = task_result
            else:
                task_result = state.get('task_result', '')
            if start_idx <= 1:
                task_answer = ts.answer(task_description, task_analysis, task_modeling_formulas, task_modeling_method, task_result)
                state['task_answer'] = task_answer
            else:
                task_answer = state.get('task_answer', '')
            backup_solution(tmp_path, all_state)
        except Exception as e:
            backup_solution(tmp_path, all_state, e)
            raise e
        task_dict = {
            'task_description': task_description,
            'task_analysis': task_analysis,
            'preliminary_formulas': task_modeling_formulas,
            'mathematical_modeling_process': task_modeling_method,
            'solution_interpretation': task_result,
            'subtask_outcome_analysis': task_answer
        }
    coordinator.memory[str(subtask_id)] = task_dict
    if 'charts' in state:
        charts = state['charts']
    else:
        charts = cc.create_charts(str(task_dict), config['chart_num'])
        state['charts'] = charts
        backup_solution(tmp_path, all_state)
    task_dict['charts'] = charts
    solution['tasks'].append(task_dict)
    save_solution(solution, name, output_dir)
    return solution
