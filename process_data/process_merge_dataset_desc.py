import json
import os


def clean_json_txt(json_txt) -> dict:
    if '```json' in json_txt:
        json_txt = json_txt.replace('```json', '')
    elif '```' in json_txt:
        json_txt = json_txt.replace('```', '')
    json_txt = json_txt.replace('```', '').strip()
    try:
        json_txt = json.loads(json_txt)
    except json.decoder.JSONDecodeError:
        print(json_txt)
        return {}
    return json_txt


def read_json_file(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        json_txt = f.read()
    return json.loads(json_txt)


def read_jsonl_file(jsonl_file) -> list:
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        jsonl = f.readlines()
    return [json.loads(json_txt) for json_txt in jsonl]


if __name__ == '__main__':
    read_dir = 'MMBench/CPMCM/problem_2'
    write_dir = 'MMBench/CPMCM/problem'
    for file in os.listdir(read_dir):
        task_id = file.split('.')[0]
        if task_id != '2015_A':
            continue
        jsonl_file = os.path.join(read_dir, file)
        jsonl = read_jsonl_file(jsonl_file)
        task_id = file.split('.')[0]
        descs = []
        dataset_paths = []
        variable_description = []
        for item in jsonl:
            # dataset_path = item.get('file_path')
            dataset_path = item.get('file_path') or item.get('folder_path')
            dataset_path = dataset_path.split('/')[-1]
            dataset_paths.append(dataset_path)

            output = item['output']
            output_json = clean_json_txt(output)
            desc = output_json.get('dataset_description', '')
            descs.append(desc)
            variable_description.append(output_json.get('variable_description', {}))
        task_final = read_json_file(os.path.join(write_dir, task_id + '.json'))
        task_dataset_path = task_final.get('dataset_path', [])
        try:
            data_index = [task_dataset_path.index(dataset_path) for dataset_path in dataset_paths]

            task_final['dataset_description'] = '\n'.join([descs[i] for i in data_index])
            task_final['variable_description'] = [variable_description[i] for i in data_index]
            with open(os.path.join(write_dir, task_id + '.json'), 'w', encoding='utf-8') as f:
                json.dump(task_final, f, indent=2, ensure_ascii=False)
        except:
            print(f'error {task_id}')