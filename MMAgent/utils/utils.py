import json
from typing import Dict
import os
from datetime import datetime
import yaml


def read_text_file(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()


def read_json_file(file_path: str) -> Dict:
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)


def write_text_file(file_path: str, content: str):
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(content)


def write_json_file(file_path: str, data:dict) -> Dict:
    with open(file_path, "w", encoding="utf-8") as json_file:
        json_file.write(json.dumps(data, indent=4, ensure_ascii=False))


def parse_llm_output_to_json(output_text: str) -> dict:
    """
    Safely parse LLM output text into a Python dictionary.
    """
    start = output_text.find("{")
    end = output_text.rfind("}") + 1
    json_str = output_text[start:end]
    try:
        data = json.loads(json_str)
    except:
        raise
        data = {}
    return data

def json_to_markdown_en(paper):
    """
    Converts a paper dictionary to a Markdown string with multi-level headlines.

    Args:
        paper (dict): The paper dictionary containing problem details and tasks.

    Returns:
        str: A Markdown-formatted string representing the paper.
    """
    markdown_lines = []

    # Problem Background
    markdown_lines.append("## Problem Background")
    markdown_lines.append(paper.get('problem_background', 'No background provided.') + "\n")

    # Problem Requirement
    markdown_lines.append("## Problem Requirement")
    markdown_lines.append(paper.get('problem_requirement', 'No requirements provided.') + "\n")

    # Problem Analysis
    markdown_lines.append("## Problem Analysis")
    markdown_lines.append(paper.get('problem_analysis', 'No analysis provided.') + "\n")

    # Problem Modeling
    if 'problem_modeling' in paper:
        markdown_lines.append("## Problem Modeling")
        markdown_lines.append(paper.get('problem_modeling', 'No modeling provided.') + "\n")

    # Tasks
    tasks = paper.get('tasks', [])
    if tasks:
        markdown_lines.append("## Tasks\n")
        for idx, task in enumerate(tasks, start=1):

            markdown_lines.append(f"### Task {idx}")

            task_description = task.get('task_description', 'No description provided.')
            markdown_lines.append("#### Task Description")
            markdown_lines.append(task_description + "\n")

            # Task Analysis
            task_analysis = task.get('task_analysis', 'No analysis provided.')
            markdown_lines.append("#### Task Analysis")
            markdown_lines.append(task_analysis + "\n")

            # Mathematical Formulas
            task_formulas = task.get('mathematical_formulas', 'No formulas provided.')
            markdown_lines.append("#### Mathematical Formulas")
            if isinstance(task_formulas, list):
                for formula in task_formulas:
                    markdown_lines.append(f"$${formula}$$")
            else:
                markdown_lines.append(f"$${task_formulas}$$")
            markdown_lines.append("")  # Add an empty line

            # Mathematical Modeling Process
            task_modeling = task.get('mathematical_modeling_process', 'No modeling process provided.')
            markdown_lines.append("#### Mathematical Modeling Process")
            markdown_lines.append(task_modeling + "\n")

            # Result
            task_result = task.get('result', 'No result provided.')
            markdown_lines.append("#### Result")
            markdown_lines.append(task_result + "\n")

            # Answer
            task_answer = task.get('answer', 'No answer provided.')
            markdown_lines.append("#### Answer")
            markdown_lines.append(task_answer + "\n")

            # Charts
            charts = task.get('charts', [])
            if charts:
                markdown_lines.append("#### Charts")
                for i, chart in enumerate(charts, start=1):
                    markdown_lines.append(f"##### Chart {i}")
                    markdown_lines.append(chart + "\n")

    # Combine all lines into a single string
    markdown_str = "\n".join(markdown_lines)
    return markdown_str

def json_to_markdown(paper):
    """
    将论文字典转换为带有多级标题的 Markdown 字符串。

    参数:
        paper (dict): 包含问题详情和任务的论文字典。

    返回:
        str: 表示论文内容的 Markdown 格式字符串。
    """
    markdown_lines = []

    # 问题背景
    markdown_lines.append("## 问题背景")
    markdown_lines.append(paper.get('problem_background', '未提供问题背景。') + "\n")

    # 问题要求
    markdown_lines.append("## 问题要求")
    markdown_lines.append(paper.get('problem_requirement', '未提供问题要求。') + "\n")

    # 问题分析
    markdown_lines.append("## 问题分析")
    markdown_lines.append(paper.get('problem_analysis', '未提供问题分析。') + "\n")

    # 问题建模
    if 'problem_modeling' in paper:
        markdown_lines.append("## 问题建模")
        markdown_lines.append(paper.get('problem_modeling', '未提供问题建模。') + "\n")

    # 各任务
    tasks = paper.get('tasks', [])
    if tasks:
        markdown_lines.append("## 任务\n")
        for idx, task in enumerate(tasks, start=1):

            markdown_lines.append(f"### 任务 {idx}")

            task_description = task.get('task_description', '未提供任务描述。')
            markdown_lines.append("#### 任务描述")
            markdown_lines.append(task_description + "\n")

            # 任务分析
            task_analysis = task.get('task_analysis', '未提供任务分析。')
            markdown_lines.append("#### 任务分析")
            markdown_lines.append(task_analysis + "\n")

            # 数学公式
            task_formulas = task.get('mathematical_formulas', '未提供公式。')
            markdown_lines.append("#### 数学公式")
            if isinstance(task_formulas, list):
                for formula in task_formulas:
                    markdown_lines.append(f"$${formula}$$")
            else:
                markdown_lines.append(f"$${task_formulas}$$")
            markdown_lines.append("")  # 空行

            # 数学建模过程
            task_modeling = task.get('mathematical_modeling_process', '未提供建模过程。')
            markdown_lines.append("#### 数学建模过程")
            markdown_lines.append(task_modeling + "\n")

            # 结果
            task_result = task.get('result', '未提供结果。')
            markdown_lines.append("#### 结果")
            markdown_lines.append(task_result + "\n")

            # 答案
            task_answer = task.get('answer', '未提供答案。')
            markdown_lines.append("#### 答案")
            markdown_lines.append(task_answer + "\n")

            # 图表
            charts = task.get('charts', [])
            if charts:
                markdown_lines.append("#### 图表")
                for i, chart in enumerate(charts, start=1):
                    markdown_lines.append(f"##### 图表 {i}")
                    markdown_lines.append(chart + "\n")

    # 合并为一个字符串
    markdown_str = "\n".join(markdown_lines)
    return markdown_str


def json_to_markdown_general(json_data):
    """
    Convert a JSON object to a markdown format.

    Args:
    - json_data (str or dict): The JSON data to convert. It can be a JSON string or a dictionary.

    Returns:
    - str: The markdown formatted string.
    """

    if isinstance(json_data, str):
        json_data = json.loads(json_data)  # If input is a JSON string, parse it.

    def recursive_markdown(data, indent=0):
        markdown_str = ""
        indent_space = "  " * indent

        if isinstance(data, dict):
            for key, value in data.items():
                markdown_str += f"### {key}\n"
                markdown_str += recursive_markdown(value, indent + 1)
        elif isinstance(data, list):
            for index, item in enumerate(data):
                markdown_str += f"- **Item {index + 1}**\n"
                markdown_str += recursive_markdown(item, indent + 1)
        else:
            markdown_str += f"- {data}\n"

        return markdown_str

    markdown = recursive_markdown(json_data)
    return markdown


def save_solution(solution, name, path):
    write_json_file(f'{path}/json/{name}.json', solution)
    markdown_str = json_to_markdown(solution)
    write_text_file(f'{path}/markdown/{name}.md', markdown_str)


def mkdir(path):
    os.makedirs(path, exist_ok=True)
    os.makedirs(path + '/json', exist_ok=True)
    os.makedirs(path + '/markdown', exist_ok=True)
    os.makedirs(path + '/latex', exist_ok=True)
    os.makedirs(path + '/code', exist_ok=True)
    os.makedirs(path + '/usage', exist_ok=True)


def load_config(args, config_path='config.yaml'):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    config['model_name'] = args.model_name
    config['method_name'] = args.method_name
    return config


def get_info(args):
    model_name = os.getenv('MODEL_NAME')
    if model_name is None:
        raise ValueError("Please set the MODEL_NAME environment variable.")
    model_name = model_name.split('/')[-1]
    problem_path = 'MMBench/{}/problem/{}.json'.format(args.mm_dataset, args.task)
    config = load_config(args)
    dataset_dir = 'MMBench/{}/dataset/{}'.format(args.mm_dataset, args.task)
    output_dir = os.path.join(
        'output/{}/{}/{}'.format(model_name, args.mm_dataset, args.method_name),
        args.task + '_{}'.format(datetime.now().strftime('%Y%m%d-%H%M%S'))
    )
    mkdir(output_dir)
    print(f'Processing {problem_path}..., config: {config}')
    return problem_path, config, dataset_dir, output_dir
