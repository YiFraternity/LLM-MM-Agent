import json
import re
import pypandoc

# A sample Markdown string
markdown_text = """
# My Document

Some **bold** text here, and some *italic* text there.

- Bullet point 1
- Bullet point 2
"""


def markdown_to_latex(markdown_text):
    # Convert Markdown string to LaTeX
    latex_text = pypandoc.convert_text(markdown_text, to='latex', format='md')
    return latex_text


def markdown_to_json_method(markdown_text):
    # 初始化根节点和层级堆栈，初始层级设为 0，以便支持一级标题
    root = {"method_class": "root", "children": []}
    stack = [{"node": root, "level": 0}]  # 用堆栈跟踪层级关系

    lines = markdown_text.strip().split('\n')
    i = 0

    while i < len(lines):
        line = lines[i].strip()
        i += 1

        if not line:
            continue

        # 匹配标题
        if line.startswith('#'):
            match = re.match(r'^(#+)\s*(.*?)$', line)
            if not match:
                continue
            hashes, method_class = match.groups()
            current_level = len(hashes)

            # 创建新节点
            new_node = {"method_class": method_class, "children": [], "description": ""}

            # 寻找合适的父节点
            while stack and stack[-1]["level"] >= current_level:
                stack.pop()

            # 如果没有找到合适的父节点，则将 new_node 加入到 root 下
            if stack:
                parent = stack[-1]["node"]
            else:
                parent = root
            parent["children"].append(new_node)

            # 更新堆栈
            stack.append({"node": new_node, "level": current_level})

            # 查找紧随标题后的描述文本
            description_lines = []
            while i < len(lines) and lines[i].strip() and not lines[i].strip().startswith('#') and not lines[i].strip().startswith('-'):
                description_lines.append(lines[i].strip())
                i += 1

            if description_lines:
                new_node["description"] = " ".join(description_lines)

            # 回退一行，因为下一行可能是列表项或新标题
            if i < len(lines):
                i -= 1

        # 匹配列表项
        elif line.startswith('-'):
            item = {}
            if ': ' in line:
                method, description = line[1:].strip().split(': ', 1)
                description = description
                item = {"method": method.strip(), "description": description.strip()}
            else:
                item = {"method": line[1:].strip(), "description": ""}

            # 添加到当前层级的子节点；若无标题节点，则直接添加到 root
            if stack:
                current_node = stack[-1]["node"]
                current_node.setdefault("children", []).append(item)
            else:
                root.setdefault("children", []).append(item)

    # 返回所有解析到的顶级标题节点
    return root["children"]


def latex_to_json(latex_content):
    """
    解析LaTeX文件内容，并将其转换为树状结构的字典列表。

    Args:
        latex_content: 包含LaTeX源码的字符串。

    Returns:
        一个列表，其中每个元素都是一个代表section的字典。
    """

    # 定义LaTeX命令的层级
    hierarchy = {
        'section': 1,
        'subsection': 2,
        'subsubsection': 3,
        'paragraph': 4,
        'subparagraph': 5,
    }

    command_pattern = '|'.join(hierarchy.keys())
    pattern = re.compile(r'\\(' + command_pattern + r')\*?\s*\{([^}]+)\}')

    doc_match = re.search(r'\\begin\{document\}(.*?)\\end\{document\}', latex_content, re.DOTALL)
    if not doc_match:
        print("警告: 未找到 \\begin{document} 环境。将尝试解析整个文件。")
        doc_content = latex_content
    else:
        doc_content = doc_match.group(1)

    parts = pattern.split(doc_content)
    result_tree = []
    level_parents = {0: {"children": result_tree}}
    preamble_content = parts[0].strip()
    if preamble_content:
        preamble_node = {
            "title": "前言",
            "content": preamble_content,
            "children": []
        }
        result_tree.append(preamble_node)
        level_parents[1] = preamble_node

    for i in range(1, len(parts), 3):
        command = parts[i]
        title = parts[i+1].strip()
        content = parts[i+2].strip()

        level = hierarchy.get(command)
        if not level:
            continue

        new_node = {
            "title": title,
            "content": content,
            "children": []
        }

        parent_node = level_parents.get(level - 1)
        if not parent_node:
            parent_level = max(k for k in level_parents if k < level)
            parent_node = level_parents[parent_level]

        parent_node["children"].append(new_node)

        level_parents[level] = new_node
        keys_to_delete = [k for k in level_parents if k > level]
        for k in keys_to_delete:
            del level_parents[k]

    return result_tree


if __name__ == "__main__":
    with open("output/CPMCM/MM-Agent/2019_B_20250912-151230/latex/solution.tex", "r", encoding="utf-8") as f:
        latex_text = f.read()

    result = latex_to_json(latex_text)
    with open('test.json', 'w') as f:
        f.write(json.dumps(result, indent=2, ensure_ascii=False))


    # AIzaSyCfcnYh7jBDnjP7kex7HEj4rpUpHRxvM_0