import os
import re
import json
from pathlib import Path
from typing import List, Dict, Any


def clean_json_txt(json_txt: str) -> Dict[str, Any]:
    """
    从可能包含 ```json ... ``` 或 ``` ... ``` 的文本中提取 JSON 并解析为 dict。

    优先级：
      1) 第一个标记为 ```json 的代码块（忽略大小写）
      2) 第一个任意 ``` ... ``` 代码块
      3) 整个输入字符串

    解析失败时打印错误信息和出错行的上下文，返回 {} 。
    """
    if not isinstance(json_txt, str):
        raise TypeError("json_txt must be a str")

    # 1) 优先查找 ```json ... ```（不区分大小写）
    m = re.search(r'```(?:\s*json\b)[\r\n]*([\s\S]*?)```', json_txt, re.IGNORECASE)
    # 2) 若没有带 json 标记的，再查找任意 ``` ... ``` 代码块
    if not m:
        m = re.search(r'```[\r\n]*([\s\S]*?)```', json_txt)

    if m:
        payload = m.group(1).strip()
    else:
        # 没有 code fence，就用原始字符串
        payload = json_txt.strip()

    try:
        return json.loads(payload)
    except json.JSONDecodeError as e:
        # 打印调试信息：错误信息 + 错误附近上下文，便于定位
        print("JSONDecodeError:", e)
        lines = payload.splitlines()
        # JSONDecodeError 有 lineno, colno 属性（从1开始）
        lineno = getattr(e, "lineno", None)
        colno = getattr(e, "colno", None)
        if lineno is not None and lineno > 0:
            err_i = lineno - 1
            start = max(0, err_i - 3)
            end = min(len(lines), err_i + 3)
            print("---- payload context (lines {}..{}) ----".format(start+1, end))
            for i in range(start, end):
                prefix = ">>" if i == err_i else "  "
                print(f"{prefix} {i+1:4d}: {lines[i]}")
            if colno is not None and err_i < len(lines):
                # 在出错行下方画指针
                pointer = " " * (6 + colno) + "^"
                print(pointer)
            print("---- end context ----")
        else:
            # 如果没有行号信息，打印前 2000 字符以便检查
            print("Payload (first 2000 chars):")
            print(payload[:2000])
        return {}


def load_jsonl(file_path: str) -> List[Dict]:
    """加载 JSONL 文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def merge_outputs(file_path: str, output_file: str) -> None:
    """
    合并多个 JSONL 文件中的 output 内容，生成合并后的 JSON 文件。

    Args:
        file_path (str): 包含 JSONL 文件的目录路径。
        output_file (str): 合并后的 JSON 文件路径。
    """
    jsonl_data = load_jsonl(file_path)
    criteria = {}
    for entry in jsonl_data:
        subtask = entry.get("subtask", "")
        output = entry.get("output", "")
        number_of_subtask = entry.get("number_of_subtask", 0)
        dimension = entry.get("eval_dimension", "")
        subtask_id = f"subtask_{number_of_subtask}"
        if subtask_id not in criteria:
            criteria[subtask_id] = {
                "subtask": subtask,
                "criteria": {}
            }
        output = clean_json_txt(output)
        criteria[subtask_id]["criteria"][dimension] = output.get("evaluation_criteria", {})

    # 保存合并后的 JSON 文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(criteria, f, ensure_ascii=False, indent=2)
    print(f"合并完成，结果已保存到 {output_file}")

if __name__ == "__main__":
    # 输入目录和输出文件路径
    input_directory = "MMBench/CPMCM/criteria_2/2010_D.jsonl"
    output_file_path = os.path.join("MMBench/CPMCM/criteria", "2010_D.json")

    # 合并输出
    merge_outputs(input_directory, output_file_path)
