#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
import re
from collections import defaultdict

def process_latex_content(text):
    """
    处理LaTeX内容，如果包含```latex {content} ```格式，则只保留{content}部分
    忽略大小写
    """
    if not text:
        return ""

    # 匹配```latex {content} ```格式，忽略大小写
    pattern = r'```latex\s+(.*?)\s+```'
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)

    if match:
        # 只返回匹配到的内容部分
        return match.group(1).strip()

    # 如果没有匹配到特定格式，则返回原始文本
    return text.strip()

def merge_pages_by_task_and_paper():
    """
    读取images_2_latex.jsonl文件，按照task_id和paper_id整合所有页面的output
    为每个task_id/paper_id组合生成单独的合并文件
    只保留处理后的output内容
    """
    # 输入文件路径
    input_file = "/home/yhliu/MathModeling/LLM-MM-Agent/images_2_latex.jsonl"

    # 创建输出目录
    output_dir = "/home/yhliu/MathModeling/LLM-MM-Agent/merged_outputs"
    os.makedirs(output_dir, exist_ok=True)

    # 存储按task_id和paper_id分组的数据
    grouped_data = defaultdict(list)

    # 读取JSONL文件
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                # 提取task_id和paper_id作为分组键
                key = (data['task_id'], data['paper_id'])
                # 提取页码信息
                page_num = extract_page_number(data['image_path'])
                # 存储数据，包括页码信息用于后续排序
                grouped_data[key].append((page_num, data))
            except json.JSONDecodeError:
                print(f"警告: 无法解析JSON行: {line}")
            except KeyError as e:
                print(f"警告: 缺少必要的键: {e}, 行: {line}")

    # 为每个task_id/paper_id组合生成单独的合并文件
    processed_count = 0
    for key, pages in grouped_data.items():
        task_id, paper_id = key

        # 创建task_id目录
        task_dir = os.path.join(output_dir, task_id)
        os.makedirs(task_dir, exist_ok=True)

        # 输出文件路径
        output_file = os.path.join(task_dir, f"{paper_id}.tex")

        # 按页码排序
        pages.sort(key=lambda x: x[0])

        # 合并所有页面的output
        merged_output = ""

        for _, data in pages:
            # 处理output内容，提取LaTeX部分
            processed_output = process_latex_content(data.get('output', ''))

            # 合并处理后的output内容
            if merged_output and processed_output:
                merged_output += "\n\n" + processed_output
            elif processed_output:
                merged_output = processed_output

        # 只保存处理后的output内容到.tex文件
        if merged_output:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(merged_output)

            processed_count += 1
            print(f"已处理: {task_id}/{paper_id} - 合并了 {len(pages)} 页")

    print(f"合并完成! 共处理了 {processed_count} 个文档，结果保存到 {output_dir} 目录")
    return processed_count

def extract_page_number(image_path):
    """
    从图像路径中提取页码
    例如: /path/to/images/2004_A/1/page_1.jpg -> 1
    """
    match = re.search(r'page_(\d+)\.jpg', image_path)
    if match:
        return int(match.group(1))
    return 0  # 如果无法提取页码，默认为0

if __name__ == "__main__":
    merge_pages_by_task_and_paper()