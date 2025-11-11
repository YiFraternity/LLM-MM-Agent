"""
Academic Paper Generator

Generates academic papers in LaTeX format from structured JSON data using
language models to create content for each section.
"""

import json
from math import log
from pathlib import Path
import subprocess
import os
import re
import shutil
from typing import Dict, List, Any
from dataclasses import dataclass

import logging

# Import statements would be here in a real application
from prompt.template import (
    PAPER_CHAPTER_PROMPT,
    PAPER_CHAPTER_WITH_PRECEDING_PROMPT,
    PAPER_INFO_PROMPT,
    PAPER_NOTATION_PROMPT
)
from llm.llm import LLM
from utils.retry_utils import (
    retry_on_api_error,
    reflective_retry_on_logic_error,
    ensure_parsed_json_output,
    LogicError,
)
from .soluation_template import (
    SafeDict,
    add_figure,
    add_code,
    create_preamble,
    create_abstract
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
# --------------------------------
# Data Models
# --------------------------------


@dataclass
class Chapter:
    """Represents a chapter in the paper with its hierarchical structure and content."""
    path: List[str]  # Hierarchical path (e.g., ["Problem Analysis", "Task 1 Analysis"])
    content: str = ""
    title: str = ""
    is_generated: bool = False
    needs_content: bool = False

    @property
    def path_string(self) -> str:
        """Returns the full path as a string (e.g., 'Problem Analysis > Task 1 Analysis')"""
        return " > ".join(self.path)

    @property
    def depth(self) -> int:
        """Returns the heading level (depth in hierarchy)"""
        return len(self.path)

    @property
    def display_title(self) -> str:
        """Returns the chapter title to display (custom title or last path element)"""
        return self.title if self.title else self.path[-1]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "content": self.content,
            "title": self.title,
            "is_generated": self.is_generated,
            "needs_content": self.needs_content
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        return cls(**data)
# --------------------------------
# Language Model Interface
# --------------------------------

def escape_underscores_in_quotes(text):
    pattern = r'(".*?")|(\'.*?\')'
    def replace_underscores(match):
        content = match.group(0)[1:-1]
        escaped_content = content.replace('_', r'\_')
        return f'"{escaped_content}"' if match.group(0).startswith('"') else f"'{escaped_content}'"

    result = re.sub(pattern, replace_underscores, text, flags=re.DOTALL)
    return result


class ContentGenerator:
    """Interface for generating content using language models"""

    def __init__(self, llm: LLM):
        self.llm = llm

    def _extract_fenced_block(self, text: str) -> str:
        """
        从 Markdown 代码围栏中截取内容：
        1) 优先选取 ```latex / ```tex 的代码块
        2) 若没有，则取第一个代码块
        3) 若完全没有围栏，则返回原文（去首尾空白）
        """
        # 捕获所有 ```lang\n ... \n``` 形式的代码块
        blocks = list(re.finditer(r"```(?:\s*)(\w+)?\s*\n(.*?)\n```", text, re.DOTALL))
        if blocks:
            # 优先 latex/tex
            for m in blocks:
                lang = (m.group(1) or "").lower()
                if lang in ("latex", "tex"):
                    return m.group(2).strip()
            # 退回：第一个代码块
            return blocks[0].group(2).strip()

        # 兜底：宽松匹配（即使没有换行也尝试抓取）
        m = re.search(r"```(?:\w+)?(.*?)```", text, re.DOTALL)
        if m:
            return m.group(1).strip()

        return text.strip()

    @retry_on_api_error(max_attempts=3, wait_time=3)
    def generate_chapter_content(self, prompt: str) -> str:
        """Generate chapter content using the language model"""
        response = self.llm.generate(prompt)
        response = self._extract_fenced_block(response)
        response = escape_underscores_in_quotes(response)
        # return self._parse_latex_response(response)
        return response

    def _parse_latex_response(self, latex_string: str) -> Dict[str, str]:
        """Parse LLM response from LaTeX format"""
        pattern = r"```latex\s*\\chapter{\s*(.*?)\s*}\s*(.*)```"
        match = re.match(pattern, latex_string.strip(), re.DOTALL)

        if match:
            return {
                "title": match.group(1).strip(),
                "content": match.group(2).strip()
            }

        # Fallback if format doesn't match
        return {
            "title": "",
            "content": latex_string
        }

# --------------------------------
# Paper Structure
# --------------------------------

class OutlineGenerator:
    """创建论文的层级结构（内部键也使用中文）"""

    def create_outline(self, task_count: int) -> List[Chapter]:
        """根据任务数量创建完整的章节结构（输出中文日志）"""
        logger.info(f"正在为 {task_count} 个任务创建论文大纲")

        # 定义基础结构模板（内部键为中文）
        outline = self._create_base_outline(task_count)

        # 创建章节对象
        chapters = []
        for path in outline:
            # 若为叶子节点（没有子节点），则需要生成内容
            needs_content = not any(other[:len(path)] == path and len(other) > len(path)
                                   for other in outline)
            chapters.append(Chapter(path=path, needs_content=needs_content))

        content_chapters = sum(1 for c in chapters if c.needs_content)
        logger.info(f"共创建 {len(chapters)} 个章节，其中 {content_chapters} 个需要生成内容")
        for chapter in chapters:
            logger.info(" > ".join(chapter.path))
        return chapters

    def _create_base_outline(self, task_count: int) -> List[List[str]]:
        """定义论文的层级结构（内部键为中文）"""
        # 基础结构
        outline = [
            ["问题重述", "问题背景"],
            ["问题重述", "问题陈述"],
            ["模型假设"],
            ["假设说明"],
            ["问题分析"]
        ]

        # 各任务的分析章节
        for i in range(1, task_count + 1):
            outline.append(["问题分析", f"任务 {i} 分析"])

        # “问题求解”总章节
        outline.append(["问题求解"])

        # 各任务的求解章节
        for i in range(1, task_count + 1):
            outline.append(["问题求解", f"任务 {i} 解答", "模型建立：假设与链式模型"])
            outline.append(["问题求解", f"任务 {i} 解答", "模型计算"])

        # 结论、限制与符号说明
        outline.extend([
            ["模型结论", "模型优点"],
            ["模型结论", "模型不足"],
            ["符号说明"]
        ])

        return outline

    def generate_chapter_relevance_map(self, task_count: int) -> Dict[str, List[str]]:
        """
        根据任务数量动态生成章节相关性映射（内部键为中文）。
        返回的 key / value 均为由 ' > ' 连接的中文路径字符串。
        """
        relevance_map: Dict[str, List[str]] = {}

        for i in range(1, task_count + 1):
            setup_path = f"问题求解 > 任务 {i} 解答 > 模型建立：假设与链式模型"
            relevance_map[setup_path] = [f"问题分析 > 任务 {i} 分析"]

        for i in range(1, task_count + 1):
            calculation_path = f"问题求解 > 任务 {i} 解答 > 模型计算"
            relevance_map[calculation_path] = [
                f"问题分析 > 任务 {i} 分析",
                f"问题求解 > 任务 {i} 解答 > 模型建立：假设与链式模型",
            ]

        # 模型结论应汇总所有任务的求解结果
        task_solutions: List[str] = []
        for i in range(1, task_count + 1):
            task_solutions += [
                f"问题求解 > 任务 {i} 解答 > 模型计算",
                f"问题求解 > 任务 {i} 解答 > 模型建立：假设与链式模型"
            ]

        relevance_map["模型结论 > 模型优点"] = task_solutions.copy()
        relevance_map["模型结论 > 模型不足"] = task_solutions.copy()
        relevance_map["符号说明"] = task_solutions.copy()

        return relevance_map


# --------------------------------
# Context Extraction
# --------------------------------

class ContextExtractor:
    """从 JSON 中为每个章节抽取上下文（内部键为中文）"""

    def get_context_for_chapter(self, chapter: Chapter, data: Dict[str, Any]) -> Dict[str, Any]:
        """针对指定章节抽取 JSON 中的相关数据"""
        path = chapter.path

        # 顶层/二级章节
        if path == ["问题重述", "问题背景"]:
            return {"problem_background": data.get("problem_background", "")}

        elif path == ["问题重述", "问题陈述"]:
            return {"problem_requirement": data.get("problem_requirement", "")}

        elif path == ["模型假设"]:
            return self._get_assumptions_context(data)

        elif path == ["假设说明"]:
            return {}

        elif self._is_task_analysis(path):
            return self._get_task_analysis_context(path, data)

        elif self._is_model_setup(path):
            return self._get_model_setup_context(path, data)

        elif self._is_model_calculation(path):
            return self._get_model_calculation_context(path, data)

        # 其他章节默认空上下文
        return {}

    def _get_assumptions_context(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """模型假设章节上下文"""
        context = {"problem_analysis": data.get("problem_analysis", "")}

        # Extract task modeling information
        keys = ['task_description', 'task_analysis', 'mathematical_modeling_process']
        context["tasks"] = [
            {k: v for k, v in task.items() if k in keys}
            for task in data.get('tasks', [])
        ]

        return context

    def _get_task_analysis_context(self, path: List[str], data: Dict[str, Any]) -> Dict[str, Any]:
        """任务分析章节上下文"""
        task_num = self._extract_task_number(path[1])
        if not self._is_valid_task_index(task_num, data):
            return {}

        task_data = data["tasks"][task_num]
        keys = ['task_analysis', 'task_description']
        return {f'任务_{task_num+1}': {k: v for k, v in task_data.items() if k in keys}}

    def _get_model_setup_context(self, path: List[str], data: Dict[str, Any]) -> Dict[str, Any]:
        """模型建立章节上下文（假设与链式模型）"""
        task_num = self._extract_task_number(path[1])
        if not self._is_valid_task_index(task_num, data):
            return {}

        task_data = data["tasks"][task_num]
        keys = ['preliminary_formulas', 'mathematical_modeling_process']
        return {f'任务_{task_num+1}': {k: task_data.get(k, "") for k in keys}}

    def _get_model_calculation_context(self, path: List[str], data: Dict[str, Any]) -> Dict[str, Any]:
        """模型计算章节上下文"""
        task_num = self._extract_task_number(path[1])
        if not self._is_valid_task_index(task_num, data):
            return {}

        task_data = data["tasks"][task_num]
        keys = ['mathematical_modeling_process', 'execution_result', 'solution_interpretation', 'subtask_outcome_analysis']
        return {f'任务_{task_num+1}': {k: task_data.get(k, "") for k in keys}}

    # --------- 辅助判断（中文键）---------
    def _is_task_analysis(self, path: List[str]) -> bool:

        return (len(path) == 2 and
                path[0] == "问题分析" and
                re.match(r"^任务\s+\d+\s+分析$", path[1]) is not None)

    def _is_model_setup(self, path: List[str]) -> bool:

        return (len(path) == 3 and
                path[0] == "问题求解" and
                re.match(r"^任务\s+\d+\s+解答$", path[1]) is not None and
                path[2] == "模型建立：假设与链式模型")

    def _is_model_calculation(self, path: List[str]) -> bool:

        return (len(path) == 3 and
                path[0] == "问题求解" and
                re.match(r"^任务\s+\d+\s+解答$", path[1]) is not None and
                path[2] == "模型计算")

    def _extract_task_number(self, task_string: str) -> int:
        """从“任务 N 分析/解答”提取任务号（0 索引）"""
        m = re.search(r"任务\s+(\d+)\s+(分析|解答)", task_string)
        try:
            return int(m.group(1)) - 1 if m else -1
        except Exception:
            return -1

    def _is_valid_task_index(self, index: int, data: Dict[str, Any]) -> bool:

        return 0 <= index < len(data.get("tasks", []))

# --------------------------------
# Prompt Creation
# --------------------------------

class PromptCreator:
    """用于为大模型生成提示词（内部键为中文）"""

    def __init__(self):
        pass

    def create_prompt(self,
                     chapter: Chapter,
                     context: Dict[str, Any],
                     previous_chapters: List[Chapter]) -> str:
        """根据章节与上下文生成提示词"""
        json_str = json.dumps(context, indent=2, ensure_ascii=False)
        previous_text = self._format_previous_chapters(previous_chapters)

        # “符号说明”章节单独使用 PAPER_NOTATION_PROMPT
        if chapter.path == ["符号说明"]:
            return PAPER_NOTATION_PROMPT.format(
                previous_chapters=previous_text,
            )
        else:
            if json_str == '{}':
                return PAPER_CHAPTER_WITH_PRECEDING_PROMPT.format(
                    chapter_path=chapter.path_string,
                    previous_chapters=previous_text
                )
            else:
                # Build the prompt using the template
                return PAPER_CHAPTER_PROMPT.format(
                    chapter_path=chapter.path_string,
                    json_context=json_str,
                    previous_chapters=previous_text
                )

    def _format_previous_chapters(self, previous_chapters: List[Chapter]) -> str:
        """Format previously completed chapters for context"""
        if not previous_chapters:
            return ""

        text = ""
        for chapter in previous_chapters:
            text += f"Chapter: {chapter.path_string}\n"
            # text += f"Title: {chapter.display_title}\n"
            text += f"{chapter.content}\n\n"
        return text


# --------------------------------
# Document Assembly
# --------------------------------

class LatexDocumentAssembler:
    """组装最终的 LaTeX 文档（内部键为中文）"""

    def __init__(self, mathmodel_category='MCMICM') -> None:
        self.mathmodel_category = mathmodel_category

    def _tr(self, s: str) -> str:
        """标题兜底翻译（若仍出现英文键则转成中文；中文键则原样返回）"""
        import re as _re_local
        mapping = {
            "Problem Restatement": "问题重述",
            "Problem Background": "问题背景",
            "Problem Statement": "问题陈述",
            "Model Assumptions": "模型假设",
            "Explanation of Assumptions": "假设说明",
            "Problem Analysis": "问题分析",
            "Solution to the Problem": "问题求解",
            "Model Setup: Assumptions and Chain Models": "模型建立：假设与链式模型",
            "Model Calculation": "模型计算",
            "Model Conclusion": "模型结论",
            "Model Advantages": "模型优点",
            "Model Limitations": "模型不足",
            "Notation and Explanations": "符号说明",
            "References": "参考文献",
            "Appendix": "附录",
            "Python Code": "Python代码",
        }
        if s in mapping:
            return mapping[s]
        m = _re_local.match(r"Task\s+(\d+)\s+Analysis", s)
        if m:
            return f"任务{m.group(1)}分析"
        m = _re_local.match(r"Task\s+(\d+)\s+Solution", s)
        if m:
            return f"任务{m.group(1)}解答"
        # 中文键直接返回
        return s

    def create_document(self, chapters: List[Chapter], metadata: Dict[str, Any]) -> str:
        """创建完整 LaTeX 文档"""
        ordered_chapters = self._reorder_chapters(chapters)

        # Build document parts
        document_parts = [
            create_preamble(metadata, self.mathmodel_category),
            create_abstract(metadata),
            "\\maketitle",
            "\\renewcommand\\cfttoctitlefont{\\hfil\\Large\\bfseries}",
            "\\renewcommand{\\contentsname}{目录}",
            "\\tableofcontents",
            "\\newpage",
            self._create_body(ordered_chapters, metadata),
            "\\end{document}"
        ]

        return "\n\n".join(document_parts)

    def _reorder_chapters(self, chapters: List[Chapter]) -> List[Chapter]:
        """调整章节顺序：将“符号说明”移到“假设说明”之后"""
        reordered = []
        notation_chapter = next((ch for ch in chapters if ch.path == ["符号说明"]), None)

        for chapter in chapters:
            if chapter.path != ["符号说明"]:
                reordered.append(chapter)
                if notation_chapter and chapter.path == ["假设说明"]:
                    reordered.append(notation_chapter)

        return reordered


    def _create_body(self, chapters: List[Chapter], metadata: Dict[str, Any]) -> str:
        """拼装正文"""
        body_parts: List[str] = []
        current_path: List[str] = []

        for chapter in chapters:
            # 在 "模型结论 > 模型优点" 处插图（若有）
            if chapter.path == ["模型结论", "模型优点"] and metadata.get('figures', []):
                body_parts += add_figure(metadata['figures'])

            for i, section in enumerate(chapter.path):
                # If this path level is new or different
                if i >= len(current_path) or section != current_path[i]:
                    # Update current path
                    if len(current_path) <= i:
                        current_path.append(section)
                    else:
                        current_path[i] = section
                        current_path = current_path[:i+1]

                    # Use custom title if available for the last level
                    title = chapter.display_title if i == chapter.depth - 1 else section
                    title = self._tr(title)

                    # Add section heading at appropriate level
                    if i == 0:
                        body_parts.append(f"\\section{{{title}}}")
                    elif i == 1:
                        body_parts.append(f"\\subsection{{{title}}}")
                    elif i == 2:
                        body_parts.append(f"\\subsubsection{{{title}}}")

            # Add chapter content if generated
            if chapter.is_generated and chapter.content:
                body_parts.append(chapter.content)

        body_parts.append("\\section{参考文献}")
        body_parts += add_code(metadata['codes'])
        return "\n\n".join(body_parts)

# --------------------------------
# File Operations
# --------------------------------

class FileManager:
    """Handles file operations for saving papers and generating PDFs"""

    @staticmethod
    def save_to_file(content: str, filepath: str) -> None:
        """Save content to a file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            f.write(content)
        logger.info(f"文档已保存到 {filepath}")

    @staticmethod
    def generate_pdf(latex_path: str) -> None:
        """Generate a PDF from a LaTeX file"""
        logger.info(f"正在从 {latex_path} 生成PDF...")

        # Run pdflatex twice to ensure references and TOC are correct
        latex_dir = os.path.dirname(latex_path)
        subprocess.run(["xelatex", f"-output-directory={latex_dir}", "-interaction=nonstopmode", latex_path])
        subprocess.run(["xelatex", f"-output-directory={latex_dir}", "-interaction=nonstopmode", latex_path])

        # Clean up auxiliary files
        FileManager._clean_temp_files(latex_path)

        pdf_path = latex_path.replace('.tex', '.pdf')
        logger.info(f"PDF已生成: {pdf_path}")

    @staticmethod
    def _clean_temp_files(latex_path: str) -> None:
        """Clean up temporary files created during PDF generation"""
        for ext in ["aux", "log", "toc", "out"]:
            aux_file = latex_path.replace('.tex', f'.{ext}')
            if os.path.exists(aux_file):
                os.remove(aux_file)

# --------------------------------
# Main Paper Generator
# --------------------------------

class PaperGenerator:
    """Main class that orchestrates the paper generation process"""

    def __init__(self, llm, mathmodel_category='MCMICM', ckpt_path=Path('tmp/solution.json')):
        self.content_generator = ContentGenerator(llm)
        self.outline_generator = OutlineGenerator()
        self.context_extractor = ContextExtractor()
        self.prompt_creator = PromptCreator()
        self.document_assembler = LatexDocumentAssembler(mathmodel_category)
        self.file_manager = FileManager()
        self.llm = llm
        self.ckpt_path = ckpt_path
        self.ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    def _load_checkpoint(self) -> Dict[str, Any]:
        """Load checkpoint file if exists"""
        if self.ckpt_path.exists():
            with open(self.ckpt_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_checkpoint(self, data: Dict[str, Any]) -> None:
        """Save checkpoint file (after each chapter)"""
        with open(self.ckpt_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _detect_start_index(self, ckpt: dict) -> int:
        """Detect from which chapter to resume generation based on checkpoint"""
        if not ckpt.get("outline"):
            return -1
        chapters = [Chapter.from_dict(ch) for ch in ckpt["outline"]]
        for i, ch in enumerate(chapters):
            if not ch.is_generated and ch.needs_content:
                return i
        return len(chapters)

    def generate_paper(self,
                    json_data: Dict[str, Any],
                    metadata: Dict[str, Any],
                    output_dir: str,
                    filename: str) -> None:
        """Generate a complete academic paper from JSON data"""
        ckpt = self._load_checkpoint()
        start_idx = self._detect_start_index(ckpt)
        if start_idx < 0:
            logger.info("No checkpoint found, starting from the beginning")
            # 1. Create chapter structure
            task_count = len(json_data.get("tasks", []))
            logger.info(f"Starting paper generation with {task_count} tasks")
            chapters = self.outline_generator.create_outline(task_count)
            ckpt['outline'] =  [ch.to_dict() for ch in chapters]
        else:
            logger.info(f"Resuming from checkpoint, starting at chapter #{start_idx}")
            chapters = [Chapter.from_dict(ch) for ch in ckpt["outline"]]
            metadata = ckpt.get("metadata", metadata)
            task_count = len(json_data.get("tasks", []))

        # Generate chapter relevance map if not provided
        chapter_relevance_map = self.outline_generator.generate_chapter_relevance_map(task_count)

        # 2. Generate content for each chapter that needs it
        for idx in range(start_idx, len(chapters)):
            chapter = chapters[idx]
            if chapter.needs_content:
                try:
                    self._generate_chapter_content(chapter, json_data, chapters[:idx], chapter_relevance_map)
                    chapters[idx] = chapter
                    ckpt["outline"][idx] = chapter.to_dict()
                    logger.info(f"✅ Saved checkpoint after {chapter.path_string}")
                except Exception as e:
                    self._save_checkpoint(ckpt)
                    logger.error(f"❌ Error generating chapter {chapter.path_string}: {e}")
                    logger.info("💾 Partial progress saved. You can re-run to resume.")
                    return
        self._save_checkpoint(ckpt)  # Final save

        # 3. Complete metadata if needed
        if 'metadata' not in ckpt:
            complete_metadata = self._complete_metadata(chapters, metadata)
            ckpt['metadata'] = complete_metadata
        else:
            complete_metadata = ckpt['metadata']

        # 4. Assemble the final document
        document = self.document_assembler.create_document(chapters, complete_metadata)

        # 5. Save and convert to PDF
        latex_path = f"{output_dir}/{filename}.tex"
        self.file_manager.save_to_file(document, latex_path)
        self.file_manager.generate_pdf(latex_path)

    def _generate_chapter_content(self,
                            chapter: Chapter,
                            json_data: Dict[str, Any],
                            completed_chapters: List[Chapter],
                            chapter_relevance_map: Dict[str, List[str]]) -> None:
        """Generate content for a single chapter"""
        logger.info(f"正在生成内容: {chapter.path_string}")

        # Get relevant context data for this chapter
        context = self.context_extractor.get_context_for_chapter(chapter, json_data)

        # Get only the relevant completed chapters for context
        relevant_chapters = self._get_relevant_chapters(chapter, completed_chapters, chapter_relevance_map)

        # Create prompt and generate content
        prompt = self.prompt_creator.create_prompt(
            chapter, context, relevant_chapters
        )
        # Generate content
        response = self.content_generator.generate_chapter_content(prompt)

        # Update chapter with generated content
        # chapter.content = response['content']
        # chapter.title = self._format_title(chapter, response['title'])
        chapter.content = response
        chapter.title = ''
        chapter.is_generated = True

    def _get_relevant_chapters(self,
                         chapter: Chapter,
                         completed_chapters: List[Chapter],
                         chapter_relevance_map: Dict[str, List[str]]) -> List[Chapter]:
        """Filter completed chapters to only include those relevant to the current chapter"""
        # Get the path string for the current chapter
        current_path = chapter.path_string

        # If this chapter has specific relevant chapters defined in the map
        if current_path in chapter_relevance_map:
            relevant_paths = chapter_relevance_map[current_path]
            # Filter completed chapters to only include those in the relevant paths
            return [ch for ch in completed_chapters
                    if ch.path_string in relevant_paths]

        # Default: return all completed chapters if no specific relevance is defined
        return completed_chapters

    def _format_title(self, chapter: Chapter, generated_title: str) -> str:
        """Format title based on chapter type"""
        # Only use custom titles for certain chapter types
        if (chapter.path[0] == "Problem Analysis" or
            chapter.path[0] == "Solution to the Problem"):
            return generated_title
        return ''

    @retry_on_api_error(max_attempts=3, wait_time=3)
    @reflective_retry_on_logic_error(
        max_attempts=3,
        wait_time=3,
    )
    @ensure_parsed_json_output
    def _call_llm_for_metadata(self, prompt: str) -> dict:
        """内部封装的 LLM 调用 + 自动 JSON 提取 + 双层重试"""
        prompt_copy = prompt  # 防止原始 prompt 被修改
        logger.info("🔍 Calling LLM to generate paper metadata...")
        response = self.llm.generate(prompt_copy)
        if not response or not isinstance(response, str):
            raise LogicError("Empty or invalid LLM response.")
        return response  # ensure_parsed_json_output 会自动提取 JSON

    def _complete_metadata(self,
                        chapters: List[Chapter],
                        provided_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Complete paper metadata, generating missing fields if needed"""
        required_keys = ["title", "summary", "keywords"]
        missing_fields = [k for k in required_keys if k not in provided_metadata]

        if not missing_fields:
            logger.info("✅ Metadata already complete, skipping generation.")
            return provided_metadata

        logger.info(f"🧩 Missing metadata fields: {missing_fields}")

        chapters_text = "\n\n".join(
            f"Chapter: {ch.path_string}\n{ch.content}"
            for ch in chapters if ch.is_generated
        )
        prompt = PAPER_INFO_PROMPT.format(paper_chapters=chapters_text)

        try:
            generated_metadata = self._call_llm_for_metadata(prompt)
            if not generated_metadata:
                raise LogicError("No metadata generated or parsed.")

            logger.info("✅ Successfully generated metadata from LLM.")
        except Exception as e:
            logger.error(f"❌ Failed to generate metadata after retries: {e}")
            generated_metadata = {}

        # --- 合并已有与生成字段 ---
        complete_metadata = {**generated_metadata, **provided_metadata}
        logger.info(f"🧾 Final metadata keys: {list(complete_metadata.keys())}")
        return complete_metadata

# --------------------------------
# Main Function
# --------------------------------

def generate_paper_from_json(llm, mathmodel_category, json_data: dict, info: dict, output_dir: str, output_name: str, ckpt_path=Path('tmp/solution.json')) -> None:
    """Generate a paper from JSON data"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    generator = PaperGenerator(llm, mathmodel_category, ckpt_path)
    generator.generate_paper(json_data, info, output_dir, output_name)


def generate_paper(llm, output_dir, name, mathmodel_category='MCMICM', ckpt_path=Path('tmp/solution.json')):
    metadata = {
        "team": "Agent",
        "year": name.split('_')[0],
        "problem_type": name.split('_')[1]
    }
    json_file_path = f"{output_dir}/json/{name}.json"
    code_dir = f'{output_dir}/code'
    metadata['figures'] = [os.path.join(code_dir, f) for f in os.listdir(code_dir) if f.lower().split('.')[-1] in ['png', 'jpg', 'jpeg']]
    metadata['codes'] = sorted([os.path.join(code_dir, f) for f in os.listdir(code_dir) if f.lower().split('.')[-1] in ['py']])
    with open(json_file_path, 'r') as f:
        json_data = json.loads(f.read())
    json_data['tasks'] = json_data['tasks'][:]
    latex_template_dir = f'LaTeX_template/{mathmodel_category}'
    shutil.copytree(latex_template_dir, f"{output_dir}/latex/", dirs_exist_ok=True)

    # Generate paper with chapter relevance mapping
    generate_paper_from_json(llm, mathmodel_category, json_data, metadata, f"{output_dir}/latex", 'solution', ckpt_path)

