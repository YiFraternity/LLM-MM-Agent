import re
import os
import logging
import json
from pathlib import Path
import argparse
from typing import Dict, List, Tuple, Callable

from utils import (
    find_task_id_from_path,
    save_json,
    parse_tex,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Matches placeholders like [DISPLAYMATH:20], [TABLEENV:1], [FIGUREENV:3], [REFERENCES]
PLACEHOLDER_PATTERN = re.compile(r"\[(?:(TABLEENV|FIGUREENV|MATHENV|DISPLAYMATH|GRAPHIC):(\d+)|REFERENCES)\]")


def find_placeholders_in_text(text: str) -> List[Tuple[str, int]]:
    """
    Return list of (key, position) where key is like 'TABLEENV:1' or 'REFERENCES:1'.
    """
    out: List[Tuple[str, int]] = []
    for m in PLACEHOLDER_PATTERN.finditer(text):
        if m.group(1) and m.group(2):
            key = f"{m.group(1)}:{m.group(2)}"
        else:
            key = "REFERENCES:1"
        out.append((key, m.start()))
    return out


def _fuzzy_pattern(s: str) -> re.Pattern:
    # 1️⃣ 转义所有特殊符号
    esc = re.escape(s)

    # 2️⃣ 让空格 / 换行 / 制表符都等价
    esc = re.sub(r"\\\s+", r"\\s+", esc)
    esc = re.sub(r"\\\s", r"\\s+", esc)
    esc = re.sub(r"(?:\\ )+", r"\\s+", esc)

    # 3️⃣ 允许匹配 section / subsection / section*
    # 注意：re.escape(s) 之后，'\\section' 会变成 '\\\\section'
    esc = esc.replace(
        re.escape(r'\section'),
        r'\\(?:sub)*section\*?'  # 匹配 \section, \subsection, 带 * 或不带 *
    )

    # 4️⃣ 忽略括号差异：让 `{`、`}`、`[`、`]` 都是可选的
    esc = esc.replace(r'\{', r'\{?')
    esc = esc.replace(r'\}', r'\}?')
    esc = esc.replace(r'\[', r'\[?')
    esc = esc.replace(r'\]', r'\]?')

    # # 5️⃣ 允许匹配中文冒号
    # colon_flex = r'(?:\}?\s*[:：]\s*\{?)'
    # esc = esc.replace(':', colon_flex)
    # esc = esc.replace('：', colon_flex)
    # 6️⃣ 编译为正则
    return re.compile(esc, flags=re.DOTALL)


def _find_context_span(text: str, next_: str) -> Tuple[int, int, int, int]:
    """
    Find a span (prev_start, prev_end, next_start, next_end) in text where 'prev' occurs
    followed by 'next'. Whitespace is matched fuzzily. Returns (-1,-1,-1,-1) if not found.
    If prev is empty, search only next. If next is empty, search only prev.
    """
    T = text

    if next_:
        # Find the first occurrence of 'next' fuzzily; insert before it.
        nr = _fuzzy_pattern(next_)
        n = nr.search(T)
        if n:
            insert_at = n.start()
            return (insert_at, insert_at, n.start(), n.end())
        return (-1, -1, -1, -1)
    else:
        return (-1, -1, -1, -1)


def find_placeholders_from_text(text: str) -> List[Tuple[str, int]]:
    """
    Return list of (key, position) where key is like 'TABLEENV:1' or 'REFERENCES:1'.
    """
    out: List[Tuple[str, int]] = []
    for m in PLACEHOLDER_PATTERN.finditer(text):
        if m.group(1) and m.group(2):
            key = f"{m.group(1)}:{m.group(2)}"
        else:
            key = "REFERENCES:1"
        out.append((key, m.start()))
    return out

def load_json_mapping(json_path: Path) -> Dict[str, dict]:
    if not json_path.exists():
        return {}
    data = json.loads(json_path.read_text(encoding='utf-8'))
    mapping = data.get('placeholders') or {}
    return mapping


def replace_placeholders(text: str, mapping: Dict[str, dict], fuzzy_func: Callable[[str], re.Pattern]) -> str:
    """
    Replace placeholders in the LaTeX file using the given mapping and fuzzy matching function.
    """
    mapping = sorted(mapping.items(), key=lambda x: x[1].get("end"), reverse=True)

    end = len(text)
    start = 0
    for key, info in mapping:
        original = info.get("original") or ""
        next_ = info.get("next") or ""

        if next_:
            pattern = fuzzy_func(next_)
            m = pattern.search(text, start, end)
            if m:
                placeholder = find_placeholders_from_text(text[m.start()-20:m.start()])
                if len(placeholder) == 0:
                    logger.warning(f"No placeholder found for key '{key}'")
                    continue
                text = text.replace(f"[{placeholder[0][0]}]", original)
                continue
            else:
                text = text.replace(f"[{key}]", original)
                logger.warning(f"key '{key}' not found for key")

        # 默认：仅替换占位符自身
        # text = text[:start] + original + text[end:]

    # 清除残留占位符
    # text = PLACEHOLDER_PATTERN.sub("", text)
    return text

def process_task(latex_file: Path, latex_json: Dict[str, dict], output_file: Path, fuzzy_func=None) -> None:
    text = latex_file.read_text(encoding='utf-8', errors='replace')
    if latex_json and fuzzy_func:
        text = replace_placeholders(text, latex_json, fuzzy_func)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(text, encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Restore LaTeX placeholders using parsed JSON mapping (positioned by following context).')
    parser.add_argument('--latex-bak-dir', default='merged_outputs', type=Path,
                        help='Root dir of LaTeX sources from each page OCR tex')
    parser.add_argument('--json-root', default=None, type=Path,
                        help='Root dir of JSON mappings from each page OCR tex')
    parser.add_argument('--output-root', default='bestpaper_latex', type=Path,
                        help='Root dir of LaTeX mappings')
    parser.add_argument('--latex-dir', default=None, type=Path,
                        help='Need to replace the content in this dir')
    parser.add_argument('--latex-files', nargs='+', type=Path,
                        help='Need to replace the content in these files')
    args = parser.parse_args()

    latex_files = []
    if args.latex_dir is not None:
        latex_files = list(args.latex_dir.glob("**/*.tex"))
    elif args.latex_files is not None:
        latex_files = args.latex_files
    else:
        latex_files = [Path("best_paper_tex_clean_format/2006_A/1_cleaned.tex")]

    for tex_file in latex_files:
        if not tex_file.exists():
            logger.warning(f"File not found: {tex_file}")
            continue
        task_id = find_task_id_from_path(tex_file)
        latex_bak_file = args.latex_bak_dir / task_id / tex_file.name.replace("_cleaned", "")
        parsed = parse_tex(latex_bak_file)
        mapping = parsed.get('placeholders', {}) if isinstance(parsed, dict) else {}
        if args.json_root:
            json_file_path = args.json_root / task_id / f"{Path(tex_file.name).stem}.json"
            json_file_path.parent.mkdir(parents=True, exist_ok=True)
            save_json(parsed, json_file_path)

        output_path = args.output_root / task_id / tex_file.name.replace("_cleaned", "")
        process_task(
            latex_file=tex_file,
            latex_json=mapping,
            fuzzy_func=_fuzzy_pattern,
            output_file=output_path,
        )
