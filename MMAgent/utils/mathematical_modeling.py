from agent.retrieve_method import MethodRetriever
from agent.task_solving import TaskSolver
from prompt.template import (
    TASK_ANALYSIS_APPEND_PROMPT,
    TASK_FORMULAS_APPEND_PROMPT,
    TASK_MODELING_APPEND_PROMPT,
)


def get_dependency_prompt(with_code, coordinator, task_id):
    task_dependency = [int(i) for i in coordinator.DAG[str(task_id)]]
    dependent_file_prompt = ""
    if len(task_dependency) > 0:
        dependency_prompt = f"""\
这是任务 {task_id}，它依赖于以下任务: {task_dependency}。
该任务的依赖关系分析如下: {coordinator.task_dependency_analysis[task_id - 1]}
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


def mathematical_modeling(task_id, problem, task_descriptions, llm, config, coordinator, with_code):
    ts = TaskSolver(llm)
    mr = MethodRetriever(llm, embed_model=config['embed_model'])
    task_analysis_prompt, task_formulas_prompt, task_modeling_prompt, dependent_file_prompt = get_dependency_prompt(with_code, coordinator, task_id)

    # 任务分析
    task_description = task_descriptions[task_id - 1]
    task_analysis = ts.analysis(task_analysis_prompt, task_description)

    # 分层建模方法检索
    description_and_analysis = f'## 任务描述\n{task_description}\n\n## 任务分析\n{task_analysis}'
    top_modeling_methods = mr.retrieve_meethods(description_and_analysis, top_k=config['top_method_num'])

    # 任务建模
    task_modeling_formulas, task_modeling_method = ts.modeling(
        task_formulas_prompt,
        task_modeling_prompt,
        problem['data_description'],
        task_description,
        task_analysis,
        top_modeling_methods,
        round=config['task_formulas_round']
    )

    return task_description, task_analysis, task_modeling_formulas, task_modeling_method, dependent_file_prompt
