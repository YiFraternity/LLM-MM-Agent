import json
import os
import re
from typing import Any, Dict

TASK_ID = '2014_E'
def clean_json_txt(text: str) -> Dict[str, Any]:
    """
    从给定文本中截取最后一个代码块并解析为 JSON。
    优先选择以 ```json 标注的最后一个代码块；若无或解析失败，
    尝试最后一个任意语言代码块中能被解析为 JSON 的内容。
    若都失败，尝试直接解析整段文本；仍失败则返回 {}。
    """
    if not isinstance(text, str):
        return {}

    # 匹配 ```lang\n ... \n``` 的代码块（lang 可空），跨行
    pattern = re.compile(r"```([a-zA-Z0-9_-]*)?[ \t]*\r?\n(.*?)```", re.DOTALL)
    matches = list(pattern.finditer(text))

    # 1) 先从最后一个显式标注为 json 的代码块开始尝试
    for m in reversed(matches):
        lang = (m.group(1) or "").lower()
        if lang == "json":
            candidate = m.group(2).strip()
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                # 继续向前找更早的 json 块或退到第 2 步
                pass

    # 2) 若没有 json 标注，或全都解析失败：从最后一个能正确解析的代码块返回
    for m in reversed(matches):
        candidate = m.group(2).strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    # 3) 兜底：尝试整段文本
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        return {}


def read_json_file(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        json_txt = f.read()
    return json.loads(json_txt)


def load_jsonl_file(jsonl_file) -> list:
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        jsonl = f.readlines()
    return [json.loads(json_txt) for json_txt in jsonl]


if __name__ == '__main__':
    read_dir = 'MMBench/CPMCM/problem_2'
    write_dir = 'MMBench/CPMCM/problem'
    for file in os.listdir(read_dir):
        task_id = file.split('.')[0]
        if task_id != TASK_ID:
            continue
        jsonl_file = os.path.join(read_dir, file)
        jsonl = load_jsonl_file(jsonl_file)
        task_id = file.split('.')[0]
        descs = {}
        dataset_paths = []
        variable_description = []
        for item in jsonl:
            # dataset_path = item.get('file_path')
            dataset_path = item.get('file_path') or item.get('dataset_path')
            dataset_path = dataset_path.split('/')[-1]
            dataset_paths.append(dataset_path)

            output = item['output']
            output_json = clean_json_txt(output)
            datapath = item.get('file_path') or item.get('dataset_path')
            datapath = datapath.split('/')[-1]
            datapath = '.'.join(datapath.split('.')[:-1])
            descs[datapath] = output_json.get('dataset_description', '')
            variable_description.append(output_json.get('variable_description', {}))
        task_final = read_json_file(os.path.join(write_dir, task_id + '.json'))
        task_dataset_path = task_final.get('dataset_path', [])
        data_index = [task_dataset_path.index(dataset_path) for dataset_path in dataset_paths]

        task_final['dataset_description'] = descs
        task_final['variable_description'] = [variable_description[i] for i in data_index]
        with open(os.path.join(write_dir, task_id + '.json'), 'w', encoding='utf-8') as f:
            json.dump(task_final, f, indent=2, ensure_ascii=False)