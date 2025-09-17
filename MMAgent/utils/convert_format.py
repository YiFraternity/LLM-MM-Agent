import json
import re
import pypandoc
from pylatexenc.latexwalker import (
    LatexWalker,
    LatexCharsNode, LatexMacroNode, LatexEnvironmentNode,
    LatexGroupNode, LatexMathNode, LatexCommentNode
)

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

    # 构建一个正则表达式，用于匹配所有层级命令
    # - \\(section|subsection|...) 匹配命令本身
    # - \*? 匹配可选的星号 (例如 \section*{...})
    # - \s* 匹配0或多个空格
    # - \{([^}]+)\} 捕获花括号内的标题内容
    command_pattern = '|'.join(hierarchy.keys())
    pattern = re.compile(r'\\(' + command_pattern + r')\*?\s*\{([^}]+)\}')

    # 只处理 \begin{document} 和 \end{document} 之间的内容
    doc_match = re.search(r'\\begin\{document\}(.*?)\\end\{document\}', latex_content, re.DOTALL)
    if not doc_match:
        print("警告: 未找到 \\begin{document} 环境。将尝试解析整个文件。")
        doc_content = latex_content
    else:
        doc_content = doc_match.group(1)

    # 使用正则表达式分割文档内容
    # re.split会返回一个列表，其中包含被分割的文本和捕获的组
    # 格式为: [content_before_first_match, command1, title1, content1, command2, title2, content2, ...]
    parts = pattern.split(doc_content)

    # 最终的JSON输出结果
    result_tree = []

    # 使用一个字典来追踪每个层级的当前父节点
    # level_parents[0] 指向根节点列表 (result_tree)
    # level_parents[1] 指向当前 section 节点
    # level_parents[2] 指向当前 subsection 节点，依此类推
    level_parents = {0: {"children": result_tree}}

    # 处理文档开头到第一个section之间的内容（前言/摘要部分）
    preamble_content = parts[0].strip()
    if preamble_content:
        # 你可以自定义这部分的标题
        preamble_node = {
            "title": "前言",
            "content": preamble_content,
            "children": []
        }
        result_tree.append(preamble_node)
        # 如果前言部分存在，它将作为第一个1级节点
        level_parents[1] = preamble_node

    # 迭代处理分割后的部分，步长为3（command, title, content）
    for i in range(1, len(parts), 3):
        command = parts[i]
        title = parts[i+1].strip()
        content = parts[i+2].strip()

        level = hierarchy.get(command)
        if not level:
            continue

        # 创建新节点
        new_node = {
            "title": title,
            "content": content,
            "children": []
        }

        # 找到正确的父节点并添加新节点
        # 一个 level N 的节点，其父节点是 level N-1
        parent_node = level_parents.get(level - 1)
        if not parent_node:
            # 如果出现层级跳跃（例如section直接到subsubsection），则挂载到最近的上级
            # 查找比当前level小的最接近的父级
            parent_level = max(k for k in level_parents if k < level)
            parent_node = level_parents[parent_level]

        parent_node["children"].append(new_node)

        # 更新当前层级的父节点为新创建的节点
        level_parents[level] = new_node

        # 清理掉所有比当前层级更深的旧父节点，以确保层级正确
        # 例如，当遇到一个新的 section (level 1) 时，之前记录的 subsection (level 2) 应该被清除
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