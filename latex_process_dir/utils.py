"""
# Core Functions

## [split_sections(text: str) -> List[Dict[str, Any]]]
Splits LaTeX document into sections based on `\section{}` commands.
- **Returns**: List of dicts with keys `title` and `content`

## [extract_subitems(content: str) -> List[Dict[str, Any]]]
Extracts subsections and subsubsections from section content.
- **Returns**: Nested structure of subsections with titles and content

## [extract_equations(text: str) -> List[str]]
Finds and extracts LaTeX equations.
- **Returns**: List of unique equation strings

## [extract_tables(text: str) -> List[str]]
Extracts `\begin{tabular}...\end{tabular}` blocks.
- **Returns**: List of table content strings

## [parse_tex(path: Path) -> dict]
Main parsing function that processes LaTeX files.
- **Returns**: Structured dict with sections, equations, tables, and metadata

## [clean_tex_content(text: str) -> Tuple[str, Dict[str, Any]]]
Identifies and records context around complex elements without modifying text.

## [find_unclosed_constructs(text: str) -> List[Dict[str, Any]]]
Detects common LaTeX parsing issues like unclosed environments.

## [save_json(data: dict, out_path: Path)]
Saves data to JSON file with proper formatting.

"""

import re
import json
from pathlib import Path
from typing import Dict, Tuple, List, Any
import yaml


def find_task_id_from_path(path: Path) -> str:
    task_id_pattern = re.compile(r'(\d{4}_[A-F])')
    for part in path.parts:
        match = task_id_pattern.search(part)
        if match:
            return match.group(0)

    match = task_id_pattern.search(str(path))
    if match:
        return match.group(0)

    return None


SECTION_RE = re.compile(r'\\section\{([^}]*)\}')
SUBSECTION_RE = re.compile(r'\\subsection\{([^}]*)\}')
SUBSUB_RE = re.compile(r'\\subsubsection\{([^}]*)\}')
EQUATION_ENV_RE = re.compile(r'\\begin\{equation\}(.*?)\\end\{equation\}', re.S)
DISPLAY_MATH_RE = re.compile(r'\\\[(.*?)\\\]', re.S)
INLINE_EQ_RE = re.compile(r'\$(.+?)\$', re.S)
TABULAR_RE = re.compile(r'\\begin\{tabular\}.*?\\end\{tabular\}', re.S)


# Patterns for cut-off triggers
TABLE_ENV_RE = re.compile(r"\\begin\{table\}", re.I)
TABULAR_RE = re.compile(r"\\begin\{tabular\}", re.I)
FIGURE_ENV_RE = re.compile(r"\\begin\{figure\}", re.I)
GRAPHICS_RE = re.compile(r"\\includegraphics", re.I)

# Common math environments and markers
MATH_ENVS = [
    r"equation\*?",
    r"align\*?",
    r"gather\*?",
    r"multline\*?",
    r"eqnarray\*?",
    r"displaymath",
]
MATH_ENV_RES: List[re.Pattern] = [re.compile(rf"\\begin\{{{env}\}}", re.I) for env in MATH_ENVS]
DISPLAY_BRACKET_RE = re.compile(r"\\\\\[", re.I)  # matches \\[
DISPLAY_PARENS_RE = re.compile(r"\\\\\(", re.I)  # matches \\(
DOLLAR_DOLLAR_RE = re.compile(r"\$\$")
INLINE_DOLLAR_RE = re.compile(r"(?<!\$)\$(?!\$)")  # single $

# References/Bibliography
REFERENCES_RE = re.compile(
    r"\\section\*?\{(?:References|Bibliography)\}|"  # \section{References}
    r"\\begin\{thebibliography\}|"                    # thebibliography env
    r"\\bibliography\{",                               # \bibliography{...}
    re.I,
)

TRIGGER_PATTERNS: List[re.Pattern] = [
    TABLE_ENV_RE,
    TABULAR_RE,
    FIGURE_ENV_RE,
    GRAPHICS_RE,
    *MATH_ENV_RES,
    DISPLAY_BRACKET_RE,
    DISPLAY_PARENS_RE,
    DOLLAR_DOLLAR_RE,
    INLINE_DOLLAR_RE,
    REFERENCES_RE,
]

def save_json(data: dict, out_path: Path):
    with out_path.open('w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def read_text(path: Path) -> str:
    with path.open('r', encoding='utf-8', errors='replace') as f:
        return f.read()

def split_sections(text: str):
    # 找到所有 section 的位置，并按顺序分段
    sections = []
    matches = list(SECTION_RE.finditer(text))
    if not matches:
        return [{'title': None, 'content': text}]
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i+1].start() if i+1 < len(matches) else len(text)
        title = m.group(1).strip()
        content = text[start:end].strip()
        sections.append({'title': title, 'content': content})
    return sections

def extract_subitems(content: str):
    subsections = []
    # split by subsection while keeping nested subsubsections inside content
    sub_matches = list(SUBSECTION_RE.finditer(content))
    if not sub_matches:
        return [{'title': None, 'content': content}]
    for i, m in enumerate(sub_matches):
        start = m.end()
        end = sub_matches[i+1].start() if i+1 < len(sub_matches) else len(content)
        title = m.group(1).strip()
        subcontent = content[start:end].strip()
        # extract subsubsections inside subcontent
        subsubs = []
        subsub_matches = list(SUBSUB_RE.finditer(subcontent))
        if not subsub_matches:
            subsubs = [{'title': None, 'content': subcontent}]
        else:
            for j, sm in enumerate(subsub_matches):
                sstart = sm.end()
                send = subsub_matches[j+1].start() if j+1 < len(subsub_matches) else len(subcontent)
                stitle = sm.group(1).strip()
                scontent = subcontent[sstart:send].strip()
                subsubs.append({'title': stitle, 'content': scontent})
        subsections.append({'title': title, 'subsubsections': subsubs})
    return subsections

def extract_equations(text: str):
    eqs = []
    for r in (EQUATION_ENV_RE, DISPLAY_MATH_RE, INLINE_EQ_RE):
        for m in r.finditer(text):
            eqs.append(m.group(1).strip())
    # 去重且保持顺序
    seen = set()
    out = []
    for e in eqs:
        if e not in seen:
            seen.add(e)
            out.append(e)
    return out

def extract_tables(text: str):
    tables = [t.strip() for t in TABULAR_RE.findall(text)]
    return tables

# 新增：通用占位符替换器，记录映射
def _replace_with_placeholders(text: str, pattern: str, label: str, mapping: Dict[str, Any]) -> str:
    """Generic helper to replace regex matches with indexed placeholders and record mapping.
    For each match store a dict: { original, prev, next, start, end } where prev/next are surrounding
    sentence-like contexts (try sentence punctuation, else fallback to fixed-window).
    """
    idx = 1

    def _find_last_of_any(s: str, chars: str) -> int:
        pos = -1
        for c in chars:
            p = s.rfind(c)
            if p > pos:
                pos = p
        return pos

    def _find_first_of_any(s: str, chars: str) -> int:
        for i, ch in enumerate(s):
            if ch in chars:
                return i
        return -1

    def repl(m: re.Match) -> str:
        nonlocal idx
        start = m.start()
        end = m.end()
        # 尝试按中/英文常见句末标点定位前一句和下一句
        sep_chars = '。.!?！？;；'
        prev_cut = _find_last_of_any(text[:start], sep_chars)
        if prev_cut == -1:
            prev_start = max(0, start - 200)
        else:
            prev_start = prev_cut + 1
        prev = text[prev_start:start].strip()

        next_rel = _find_first_of_any(text[end:end+200], sep_chars)
        if next_rel == -1:
            next_end = min(len(text), end + 200)
        else:
            next_end = end + next_rel + 1
        next_ctx = text[end:next_end].strip()

        key = f"{label}:{idx}"
        mapping[key] = {
            'original': m.group(0),
            'prev': prev,
            'next': next_ctx,
            'start': start,
            'end': end
        }
        idx += 1
        return f"[{key}]"

    return re.sub(pattern, repl, text, flags=re.DOTALL)

def load_yaml(yaml_path: str) -> dict[str, Any]:
    with open(yaml_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def clean_tex_content(text: str) -> Tuple[str, Dict[str, Any]]:
    """
    Scan LaTeX content and collect contexts for tables, figures/images, and display math
    without modifying the original text.

    For each matched object, record:
    - original: the matched LaTeX snippet
    - prev: up to 50 characters immediately before the object
    - next: up to 50 characters immediately after the object
    - start, end: character indices in the original text
    - label: category label

    Returns (original_text, mapping) where mapping maps synthetic keys to context dicts.
    """
    mapping: Dict[str, Any] = {}

    # Determine cutoff at the start of References/参考文献 and ignore anything after it
    refs_patterns = [
        r"\\begin\{thebibliography\}",
        r"\\bibliography\{.*?\}",
        r"\\bibliographystyle\{.*?\}",
        r"\\section\*?\{\s*(?:参考文献|References)[.:]?\s*\}",
        r"\\section\*?\{\s*[\d.．一二三四五六七八九十、]+\s*(?:参考文献|References)[.:]?\s*\}",
        r"\\chapter\*?\{\s*(?:参考文献|References)[.:]?\s*\}",
        r"\\chapter\*?\{\s*[\d.．一二三四五六七八九十、]+\s*(?:参考文献|References)[.:]?\s*\}"
    ]
    earliest = None
    for pat in refs_patterns:
        m = re.search(pat, text, flags=re.IGNORECASE | re.DOTALL)
        if m:
            pos = m.start()
            if earliest is None or pos < earliest:
                earliest = pos
    scan_text = text if earliest is None else text[:earliest]

    # Use fixed-width character windows instead of sentence segmentation
    WINDOW = 50

    counters: Dict[str, int] = {}

    def record_matches(pattern: str, label: str, collect_ranges: List[Tuple[int, int]] = None, skip_inside: List[Tuple[int, int]] = None):
        nonlocal mapping
        counters.setdefault(label, 0)

        # Compile pattern for potential performance boost
        compiled_pattern = re.compile(pattern, flags=re.DOTALL)

        for m in compiled_pattern.finditer(scan_text):
            start, end = m.start(), m.end()

            # If asked, skip matches that are fully inside any of the skip ranges (priority handling)
            if skip_inside and any(start >= rs and end <= re for rs, re in skip_inside):
                continue

            counters[label] += 1
            key = f"{label}:{counters[label]}"
            prev_start = max(0, start - WINDOW)
            next_end = min(len(scan_text), end + WINDOW)
            prev_ctx = scan_text[prev_start:start]
            next_ctx = scan_text[end:next_end]
            mapping[key] = {
                'original': m.group(0),
                'prev': prev_ctx,
                'next': next_ctx,
                'start': start,
                'end': end,
                'label': label,
            }
            if collect_ranges is not None:
                collect_ranges.append((start, end))

    # --- Priority Handling ---

    # 1. Capture outer TABLE environments (table, table*, longtable)
    table_outer_envs = ['table', 'table*', 'longtable']
    table_outer_ranges: List[Tuple[int, int]] = []
    for env in table_outer_envs:
        pattern = rf"\\begin\{{{re.escape(env)}\}}[\s\S]*?\\end\{{{re.escape(env)}\}}"
        record_matches(pattern, 'TABLEENV', collect_ranges=table_outer_ranges)

    # 2. Capture tabular-like environments only if NOT inside an outer table range (Implements: table > tabular)
    tabular_envs = ['tabular', 'tabular*']
    for env in tabular_envs:
        pattern = rf"\\begin\{{{re.escape(env)}\}}[\s\S]*?\\end\{{{re.escape(env)}\}}"
        record_matches(pattern, 'TABLEENV', skip_inside=table_outer_ranges)

    # 3. Capture FIGURE environments (figure, figure*)
    figure_envs = ['figure', 'figure*']
    figure_env_ranges: List[Tuple[int, int]] = []
    for env in figure_envs:
        pattern = rf"\\begin\{{{re.escape(env)}\}}[\s\S]*?\\end\{{{re.escape(env)}\}}"
        record_matches(pattern, 'FIGUREENV', collect_ranges=figure_env_ranges)

    # 4. Capture Graphics commands only if NOT inside a FIGURE environment (Implements: figure > includegraphics)
    # The patterns for includegraphics are slightly adjusted to be more robust.
    record_matches(r"\\includegraphics\*?\s*(?:\[[^\]]*\])?\s*\{[^\}]*\}", 'GRAPHIC', skip_inside=figure_env_ranges)


    # --- Remaining Environments and Constructs ---

    # 5. Math Environments
    math_envs = ['equation', 'equation*', 'align', 'align*', 'gather', 'multline', 'eqnarray', 'displaymath']
    for env in math_envs:
        pattern = rf"\\begin\{{{re.escape(env)}\}}[\s\S]*?\\end\{{{re.escape(env)}\}}"
        record_matches(pattern, 'MATHENV')

    # 6. Display math constructs
    record_matches(r"\$\$[\s\S]*?\$\$", 'DISPLAYMATH')
    record_matches(r"\\\[[\s\S]*?\\\]", 'DISPLAYMATH')


    # Return original text unchanged
    return text, mapping

def parse_tex(path: Path) -> dict:
    text = read_text(path)
    # 检测未闭合构造（基于原始文本，便于定位和修复）
    unclosed = find_unclosed_constructs(text)
    # 收集对象上下文，不修改原文
    cleaned_text, placeholders = clean_tex_content(text)
    # Determine pre-reference text for parsing
    refs_patterns = [
        r"\\begin\{thebibliography\}",
        r"\\bibliography\{.*?\}",
        r"\\bibliographystyle\{.*?\}",
        r"\\section\*?\{\s*(?:参考文献|References)[.:]?\s*\}",
        r"\\section\*?\{\s*[\d.．一二三四五六七八九十、]+\s*(?:参考文献|References)[.:]?\s*\}",
        r"\\chapter\*?\{\s*(?:参考文献|References)[.:]?\s*\}",
        r"\\chapter\*?\{\s*[\d.．一二三四五六七八九十、]+\s*(?:参考文献|References)[.:]?\s*\}"
    ]
    earliest = None
    for pat in refs_patterns:
        m = re.search(pat, cleaned_text, flags=re.IGNORECASE | re.DOTALL)
        if m:
            pos = m.start()
            if earliest is None or pos < earliest:
                earliest = pos
    pre_refs_text = cleaned_text if earliest is None else cleaned_text[:earliest]

    sections = split_sections(pre_refs_text)
    parsed = {
        'file': str(path),
        'sections': [],
        'placeholders': placeholders,  # now stores contexts with prev/next sentences
        'cleaned_text': cleaned_text,  # equals original text (kept intact)
        'unclosed': unclosed,
    }
    for sec in sections:
        sec_entry = {'title': sec['title'], 'raw': sec['content']}
        sec_entry['subsections'] = extract_subitems(sec['content'])
        sec_entry['equations'] = extract_equations(sec['content'])
        sec_entry['tables'] = extract_tables(sec['content'])
        parsed['sections'].append(sec_entry)
    # also include top-level equations/tables outside sections (ignore after references)
    parsed['top_equations'] = extract_equations(pre_refs_text)
    parsed['top_tables'] = extract_tables(pre_refs_text)
    return parsed


def find_unclosed_constructs(text: str) -> List[Dict[str, Any]]:
    """
    查找可能未闭合的 LaTeX 构造，返回一个问题列表，每项包含:
    - type: 'env'|'display_math_dollars'|'display_math_bracket'
    - env: env 名（仅 env 类型有）
    - start: 字符索引
    - line: 行号（从 1 开始）
    - preview: 从 start 开始的片段预览（最多 200 字符）
    """
    issues: List[Dict[str, Any]] = []

    # 1) 未闭合的 \begin{..} 环境
    for m in re.finditer(r'\\begin\{([^\}]+)\}', text):
        env = m.group(1)
        start_idx = m.start()
        end_pattern = rf'\\end\{{{re.escape(env)}\}}'
        if not re.search(end_pattern, text[m.end():], flags=re.DOTALL):
            line = text.count('\n', 0, start_idx) + 1
            preview = text[start_idx:start_idx+200].replace('\n', ' ')
            issues.append({'type': 'env', 'env': env, 'start': start_idx, 'line': line, 'preview': preview})

    # 2) 未配对的 $$ 表示的 display math（只检查偶数性）
    dollar_positions = [m.start() for m in re.finditer(r'\$\$', text)]
    if len(dollar_positions) % 2 == 1:
        pos = dollar_positions[-1]
        line = text.count('\n', 0, pos) + 1
        preview = text[pos:pos+200].replace('\n', ' ')
        issues.append({'type': 'display_math_dollars', 'start': pos, 'line': line, 'preview': preview})

    # 3) 未配对的 \[  与 \]
    opens = [m.start() for m in re.finditer(r'\\\[', text)]
    closes = [m.start() for m in re.finditer(r'\\\]', text)]
    if len(opens) > len(closes):
        # 多出的 opens 视为未闭合，按顺序列出多出的那部分位置
        for pos in opens[len(closes):]:
            line = text.count('\n', 0, pos) + 1
            preview = text[pos:pos+200].replace('\n', ' ')
            issues.append({'type': 'display_math_bracket', 'start': pos, 'line': line, 'preview': preview})

    return issues


def load_llm(model_name_or_path, tokenizer_name_or_path=None, gpu_num=1, lora_model_name_or_path=None, max_model_len=32768):
    """
    Load a VLLM model.
    """
    from vllm import LLM, SamplingParams
    kw_args = {
        "model": model_name_or_path,
        "tokenizer": tokenizer_name_or_path,
        "tokenizer_mode": "slow",
        "tensor_parallel_size" : gpu_num,
        "enable_lora": bool(lora_model_name_or_path),
        'max_model_len': max_model_len,
    }
    llm = LLM(**kw_args)
    kwargs={
        "n":1,
        "max_tokens": 4096,
        "top_p":1.0,
        # sampling
        "temperature":0,
        'top_k': 1,
    }
    sampling_params = SamplingParams(**kwargs)
    return llm, sampling_params


def populate_template(template: str, variables: dict) -> str:
    """
    Populate a Jinja template with variables.
    """
    from jinja2 import Template, StrictUndefined
    compiled_template = Template(template, undefined=StrictUndefined)
    try:
        return compiled_template.render(**variables)
    except Exception as e:
        raise Exception(f"Error during jinja template rendering: {type(e).__name__}: {e}")


def prepare_batch_prompts(prompts_kwargs: List[dict[str, Any]], prompt_template: str, system_prompt='') -> List[List[dict]]:
    """
    Prepare a batch of prompts for inference.
    """
    prompts = [populate_template(prompt_template, prompt_kwarg) for prompt_kwarg in prompts_kwargs]
    if system_prompt == '':
        sys_prompt = 'You are a helpful AI assistant.'
    else:
        sys_prompt = system_prompt
    system_prompts = [sys_prompt for _ in prompts]
    message_list = []
    for prompt, sys_prompt in zip(prompts, system_prompts):
        message_list.append([
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt}
            ])
    return message_list


def clean_json_txt(json_txt: str, standardize: bool = False) -> Union[str, Dict[str, Any], List[Dict[str, Any]]]:
    """
    从可能包含 ```json ... ``` 或 ``` ... ``` 的文本中提取 JSON 并解析为 dict。

    优先级：
      1) 第一个标记为 ```json 的代码块（忽略大小写）
      2) 第一个任意 ``` ... ``` 代码块
      3) 解析包含latex公式的json
      3) 整个输入字符串

    参数:
        json_txt: 要处理的JSON文本
        standardize: 是否尝试标准化非标准JSON

    返回:
        解析后的JSON对象，如果standardize为True且标准化失败，则返回空字典
    """
    if not isinstance(json_txt, str):
        raise TypeError("json_txt must be a str")

    m = re.search(r'```(?:\s*json\b)[\r\n]*([\s\S]*?)```', json_txt, re.IGNORECASE)
    if not m:
        m = re.search(r'```[\r\n]*([\s\S]*?)```', json_txt)

    if m:
        payload = m.group(1).strip()
    else:
        payload = json_txt.strip()

    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        # If it fails, check if the error is likely due to unescaped backslashes
        corrected_payload = re.sub(r'\\', r'\\\\', payload)
        try:
            return json.loads(corrected_payload)
        except json.JSONDecodeError:
            if standardize:
                return {}
            else:
                return json_txt
