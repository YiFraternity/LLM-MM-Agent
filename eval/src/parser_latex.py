import re

# 定义结构命令及其对应的层级
STRUCTURE_LEVELS = {
    "section": 1,
    "subsection": 2,
    "subsubsection": 3,
    "paragraph": 4, # 增加更多层级以测试
}

class Node:
    def __init__(self, level, title, content=""):
        self.level = level
        self.title = title
        self.content = content
        self.children = []

    def __repr__(self):
        return f"Node(Level={self.level}, Title='{self.title[:40]}...', Children={len(self.children)})"

    def to_dict(self):
        return {
            "level": self.level,
            "title": self.title,
            "content": self.content,
            "content_length": len(self.content),
            "children": [child.to_dict() for child in self.children]
        }


def parse_latex(file_content: str, target_level: int) -> Node:
    """
    解析 LaTeX 文件内容，构建结构树，并正确分配非叶子节点的content。

    Args:
        file_content: LaTeX 文件的全部内容字符串。
        target_level: 决定最终叶子节点内容的层级（例如 2 for \section, 3 for \subsection）。

    Returns:
        Node: 树的根节点 (Root Node)。
    """
    pattern = re.compile(
        r"\\(?P<cmd>" + "|".join(STRUCTURE_LEVELS.keys()) + r")(?P<star>\*?)\s*(?P<title>{.*?})",
        re.DOTALL
    )

    root = Node(level=-1, title="Document Root")
    node_stack = [root]

    matches = list(pattern.finditer(file_content))

    if matches:
        preamble_content = file_content[:matches[0].start()]
        if preamble_content.strip():
            preamble_node = Node(level=0, title="Preamble/Doc Class", content=preamble_content)
            root.children.append(preamble_node)
    else:
        root.content = file_content
        return root

    for i, match in enumerate(matches):
        cmd = match.group('cmd')
        title = match.group('title').strip().strip('{}')
        current_level = STRUCTURE_LEVELS.get(cmd, -1)
        start_index = match.end()

        if current_level > target_level:
            continue

        content_end_index = len(file_content)

        # 确定当前结构节点 (current_level) 的内容结束位置 (content_end_index)
        for j in range(i + 1, len(matches)):
            child_cmd = matches[j].group('cmd')
            child_level = STRUCTURE_LEVELS.get(child_cmd, -1)

            # 终止条件 1: 遇到同级或更高级的结构 (适用于所有节点)
            if child_level <= current_level:
                content_end_index = matches[j].start()
                break

            # 终止条件 2: 遇到子结构，且当前节点是非叶子节点 (current_level < target_level)
            #             并且该子结构在目标层级内 (child_level <= target_level)
            is_non_leaf = current_level < target_level
            is_valid_child = child_level > current_level and child_level <= target_level

            if is_non_leaf and is_valid_child:
                content_end_index = matches[j].start()
                break

        new_node = Node(level=current_level, title=title)
        new_node.content = file_content[start_index:content_end_index].strip()

        while node_stack[-1].level >= current_level:
            node_stack.pop()
            if not node_stack:
                node_stack.append(root)
                break

        parent_node = node_stack[-1]
        parent_node.children.append(new_node)

        if current_level < target_level:
            node_stack.append(new_node)

    return root

