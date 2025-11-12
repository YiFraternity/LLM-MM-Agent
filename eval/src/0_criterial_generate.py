import os
import argparse
import copy
from pathlib import Path
from typing import List

from eval_utils import (
    find_task_id_from_path,
    clean_json_txt,
    load_yaml,
    load_json,
    write_jsonl,
    load_llm,
    prepare_batch_prompts,
)

from prompt.template import (
    PROBLEM_ANALYSIS_PROMPT,
    TASK_DECOMPOSE_PROMPT,
    PROBLEM_MODELING_PROMPT,
    PROBLEM_PROMPT,
    DATA_DESCRIPTION_PROMPT,
    TASK_DESCRIPTION_PROMPT,
    TASK_DEPENDENCY_ANALYSIS_WITH_CODE_PROMPT,
    TASK_DEPENDENCY_ANALYSIS_PROMPT,
)

def get_dataset_path(root_dir):
    dataset_path = []
    if not os.path.exists(root_dir):
        return dataset_path
    dataset_path = os.listdir(root_dir)
    return dataset_path

def postprocess_output(inputs:List[dict], outputs, key='output'):
    """
    Postprocess the output of the model.
    """
    assert len(inputs) == len(outputs)
    for _input, _output in zip(inputs, outputs):
        _input[key] = clean_json_txt(_output)
    return inputs

def get_phase1_variable(root_dir: List[Path]) -> List[dict]:
    result = []
    for task in root_dir:
        task_id = task.stem
        data = load_json(task)
        data['background'] = data['background']
        data['question'] = data['problem_requirement']
        data['task_id'] = task_id
        result.append(data)
    return result

def get_phase2_variable(root_dir: Path, problem_root: Path) -> List[dict]:
    result = []
    tasks = root_dir.iterdir()
    tasks = sorted(tasks)
    for task in tasks:
        task_id = task.stem
        task_content = load_json(task)
        try:
            data = task_content['output']
        except:
            print(f'task {task} is not valid')
            continue
        problem = problem_root / f'{task_id}.json'
        problem_content = load_json(problem)
        for key, value in data.items():
            if key.lower().startswith('subtask'):
                __data = dict()
                __data['task_id'] = task_id
                __data['subtask'] = value['description']
                __data['previous_subtasks'] = value['depend_on_prev_tasks']
                __data['background'] = problem_content['background']
                __data['question'] = problem_content['problem_requirement']
                result.append(__data)
    return result

def main(
    model,
    sampling_params,
    questions: List[dict],
    system_prompt='You are a helpful AI assistant.',
):
    """
    using MMAgent problem decompose
    1. perform problem analysis
    2. modeling solution based on problem analysis
    3. decompose problem
    """
    all_analysis_prompts = [[
        {'role': "system",'content': system_prompt},
        {"role": "user", "content": PROBLEM_ANALYSIS_PROMPT.format(
            modeling_problem = question['problem_str'],
            user_prompt = ''
        )}
    ] for question in questions]
    analysis_output = model.chat(all_analysis_prompts, sampling_params, use_tqdm=True)

    pred_lst = []
    for o_t in analysis_output:
        pred_lst.append(o_t.outputs[0].text)
    questions = postprocess_output(questions, pred_lst, 'analysis')

    all_modeling_prompts = [[
        {'role': "system",'content': system_prompt},
        {"role": "user", "content": PROBLEM_MODELING_PROMPT.format(
            modeling_problem = question['problem_str'],
            problem_analysis = question['analysis'],
            user_prompt = ''
        )}
    ] for question in questions]
    modeling_output = model.chat(all_modeling_prompts, sampling_params, use_tqdm=True)
    pred_lst = []
    for o_t in modeling_output:
        pred_lst.append(o_t.outputs[0].text)
    questions = postprocess_output(questions, pred_lst, 'modeling')

    decomposed_principles = load_json('MMAgent/prompt/decompose_prompt.json')

    all_decompose_prompts = []
    for question in questions:
        decomposed_principle = decomposed_principles.get(
            question['problem_type'], decomposed_principles['C']
        )
        decomposed_principle = decomposed_principle.get(
            str(question['tasknum']), decomposed_principle['4']
        )
        all_decompose_prompts.append([
            {'role': "system",'content': system_prompt},
            {"role": "user", "content": TASK_DECOMPOSE_PROMPT.format(
                modeling_problem = question['problem_str'],
                problem_analysis = question['analysis'],
                modeling_solution = question['modeling'],
                decomposed_principle = decomposed_principle,
                tasknum = question['tasknum'],
                user_prompt = ''
            )}
        ])
    decompose_output = model.chat(all_decompose_prompts, sampling_params, use_tqdm=True)
    pred_lst = []
    for o_t in decompose_output:
        pred_lst.append(o_t.outputs[0].text)
    questions = postprocess_output(questions, pred_lst, 'decompose')
    for question in questions:
        _decompose = question.get('decompose', '')
        question['decompose'] = [task.strip() for task in _decompose.split('---') if task.strip()]
    all_tasks_refine_prompts = []
    new_questions = []
    for question in questions:
        decomposed_subtasks = question.get('decompose', [])
        decomposed_subtasks_str = '\n'.join(decomposed_subtasks)
        for task_i in range(len(decomposed_subtasks)):
            all_tasks_refine_prompts.append([
                {'role': "system",'content': system_prompt},
                {'role': "user", 'content': TASK_DESCRIPTION_PROMPT.format(
                    modeling_problem = question['problem_str'],
                    problem_analysis = question['analysis'],
                    modeling_solution = question['modeling'],
                    decomposed_subtasks = decomposed_subtasks_str,
                    task_i = task_i + 1
                )}
            ])
            t = copy.deepcopy(question)
            t.update({'subtask_id': task_i + 1})
            new_questions.append(t)
    task_refine_output = model.chat(all_tasks_refine_prompts, sampling_params, use_tqdm=True)
    pred_lst = []
    for o_t in task_refine_output:
        pred_lst.append(o_t.outputs[0].text)
    new_questions = postprocess_output(new_questions, pred_lst, 'task_refine')

    analy_dependencies_prompt = []
    for question in new_questions:
        with_code = question.get('dataset_path') > 0
        if with_code:
            prompt_template = TASK_DEPENDENCY_ANALYSIS_WITH_CODE_PROMPT
        else:
            prompt_template = TASK_DEPENDENCY_ANALYSIS_PROMPT
        analy_dependencies_prompt.append([
            {'role': "system",'content': system_prompt},
            {'role': "user", 'content': prompt_template.format(
                tasknum = question['subtask_id'],
                modeling_problem = question['problem_str'],
                problem_analysis = question['analysis'],
                modeling_solution = question['modeling'],
                task_descriptions = question['decompose']
            )}
        ])
    return new_questions

def description_summary(model, sampling_params, questions: List[dict], system_prompt='You are a helpful AI assistant.'):
    """
    summary description
    """
    all_summary_prompts = [[
        {'role': "system",'content': system_prompt},
        {"role": "user", "content": DATA_DESCRIPTION_PROMPT.format(
            data_description = str(question.get('data_description', {})) + '\n' + str(question.get('variable_description', {}))
        )}
    ]for question in questions]
    summary_output = model.chat(all_summary_prompts, sampling_params, use_tqdm=True)
    pred_lst = []
    for o_t in summary_output:
        pred_lst.append(o_t.outputs[0].text)
    questions = postprocess_output(questions, pred_lst, 'data_summary')
    return questions

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate criteria and ensure placeholders for problems.')
    parser.add_argument('--criteria1', type=Path, default=Path('MMBench/CPMCM/criteria_1'),
                        help='Output directory for phase 1 results')
    parser.add_argument('--criteria2', type=Path, default=Path('MMBench/CPMCM/criteria_2'),
                        help='Output directory for phase 2 results')
    parser.add_argument('--problem-root', type=Path, default=Path('MMBench/CPMCM/problem'),
                        help='Root directory containing problem JSON files')
    parser.add_argument('--criteria-root', type=Path, default=Path('criteria-root'),
                        help='Directory to ensure criteria placeholders exist')
    parser.add_argument('--model-name-or-path', type=str, default='/group_homes/our_llm_domain/home/share/open_models/Qwen/Qwen2.5-32B-Instruct',
                        help='Path to the model')
    parser.add_argument('--gpu-num', type=int, default=2, help='Number of GPUs to use')
    parser.add_argument('--template-path', type=Path, default=Path('eval/prompts/criterial_generate.yaml'),
                        help='Path to the template file')
    parser.add_argument('--tmp-criteria1', type=Path, default=Path('tmp/criteria1'),
                        help='Temporary directory for phase 1 results')
    parser.add_argument('--tmp-criteria2', type=Path, default=Path('tmp/criteria2'),
                        help='Temporary directory for phase 2 results')
    args = parser.parse_args()

    task_ids = []

    template = load_yaml(args.template_path)
    problem_root = args.problem_root
    for file in problem_root.iterdir():
        task_id = file.stem
        task_ids.append(file)

    criteria1 = args.criteria1
    criteria2 = args.criteria2

    args.tmp_criteria1.mkdir(parents=True, exist_ok=True)
    args.tmp_criteria2.mkdir(parents=True, exist_ok=True)

    questions = []
    for task_id in task_ids:
        data = load_json(task_id)
        _task_id = find_task_id_from_path(task_id)
        problem_type = _task_id.split('_')[0]
        data['task_id'] = _task_id
        data['problem_type'] = problem_type

        if data.get('addendum', ''):
            addendum = f"Addendum: \n{data['addendum']}"
        else:
            addendum = ''
        data['addendum'] = addendum[:3000]
        data['tasknum'] = 4
        questions.append(data)

    model, sampling_params = load_llm(args.model_name_or_path, gpu_num=args.gpu_num)

    questions = description_summary(
        model,
        sampling_params,
        questions,
        system_prompt=template['math_modeling_criteria_generator']['system'],
    )

    for question in questions:
        question['problem_str'] = PROBLEM_PROMPT.format(
            problem_background=question['background'],
            problem_requirement=question['problem_requirement'],
            addendum=question['addendum'],
            data_summary=question['data_summary'],
        )

    questions = main(
        model,
        sampling_params,
        questions,
        system_prompt=template['math_modeling_criteria_generator']['system'],
    )

    for question in questions:
        task_id = question['task_id']
        output_path = args.tmp_criteria1 / f"{task_id}.jsonl"
        write_jsonl(question, output_path)
