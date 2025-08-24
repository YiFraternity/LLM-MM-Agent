import json
import os

def read_json_file(file_path: str) -> dict:
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def write_json_file(data: dict, file_path: str) -> None:
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def parse_json_text(json_text: str) -> dict | str:
    json_text = json_text.strip()
    if json_text.startswith('```json'):
        json_text = json_text.split('```json')[1].split('```')[0].strip()
    elif json_text.startswith('```'):
        json_text = json_text.split('```')[1].split('```')[0].strip()
    try:
        return json.loads(json_text)
    except Exception as e:
        return json_text

if __name__ == '__main__':
    root_dir = 'MMBench/CPMCM/problem_1'
    rewrite_dir = 'MMBench/CPMCM/problem'
    os.makedirs(rewrite_dir, exist_ok=True)
    for file in os.listdir(root_dir):
        if file.endswith('.json'):
            file_path = os.path.join(root_dir, file)
            data = read_json_file(file_path)
            problem_content = data['output']
            problem_content = parse_json_text(problem_content)
            if isinstance(problem_content, dict):
                write_json_file(problem_content, os.path.join(rewrite_dir, file))
            else:
                with open('error.txt', 'a', encoding='utf-8') as f:
                    f.write(file + '\n')
