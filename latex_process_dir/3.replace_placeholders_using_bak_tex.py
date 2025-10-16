import re
import os
import math
import json
from pathlib import Path
from typing import Dict, List, Tuple

PLACEHOLDER_PATTERN = re.compile(r"\[(?:([A-Z_]+):(\d+)|REFERENCES:1)\]")


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
    """Create a fuzzy regex pattern that matches the string s, allowing flexible whitespace.
    """
    # escape, then replace escaped whitespace sequences with \s+
    esc = re.escape(s)
    # Replace any escaped whitespace (including newlines, tabs, spaces) with a generic \s+
    esc = re.sub(r"\\\s+", r"\\s+", esc)
    esc = re.sub(r"\\\s", r"\\s+", esc)
    # Also collapse multiple escaped spaces into \s+
    esc = re.sub(r"(?:\\ )+", r"\\s+", esc)
    return re.compile(esc, flags=re.DOTALL)


def _find_context_span(text: str, prev: str, next_: str) -> Tuple[int, int, int, int]:
    """
    Find a span (prev_start, prev_end, next_start, next_end) in text where 'prev' occurs
    followed by 'next'. Whitespace is matched fuzzily. Returns (-1,-1,-1,-1) if not found.
    If prev is empty, search only next. If next is empty, search only prev.
    """
    T = text

    if prev and next_:
        pr = _fuzzy_pattern(prev)
        nr = _fuzzy_pattern(next_)
        last_candidate = (-1, -1, -1, -1)
        for m in pr.finditer(T):
            p_end = m.end()
            n = nr.search(T, p_end)
            if n:
                last_candidate = (m.start(), m.end(), n.start(), n.end())
                break
        return last_candidate
    elif prev:
        pr = _fuzzy_pattern(prev)
        m = pr.search(T)
        if not m:
            return (-1, -1, -1, -1)
        return (m.start(), m.end(), m.end(), m.end())
    elif next_:
        nr = _fuzzy_pattern(next_)
        n = nr.search(T)
        if not n:
            return (-1, -1, -1, -1)
        return (n.start(), n.start(), n.start(), n.end())
    else:
        return (-1, -1, -1, -1)


def load_json_mapping(json_path: Path) -> Dict[str, dict]:
    if not json_path.exists():
        return {}
    data = json.loads(json_path.read_text(encoding='utf-8'))
    mapping = data.get('placeholders') or {}
    return mapping


def process_task(
    latex_dir: Path,
    json_root: Path,
    output_root: Path,
    task_id: str=None,
    json_file: str=None,
) -> None:
    if task_id is None:
        task_ids = [d for d in sorted(os.listdir(json_root)) if (json_root / d).is_dir()]
    else:
        task_ids = [task_id]
    print(f"=== Task IDs: {task_ids}")
    for task_id in task_ids:
        json_task_dir = json_root / task_id
        tex_task_dir = latex_dir / task_id

        json_files = [p for p in sorted(os.listdir(json_task_dir)) if p.endswith('.json')]
        for json_f in json_files:
            if json_file is not None and json_f != json_file:
                continue
            json_path = json_task_dir / json_f
            tex_path = (tex_task_dir / Path(json_f).stem).with_suffix('.tex')
            if not tex_path.exists():
                continue

            print(f"=== File: {tex_path}")
            text = tex_path.read_text(encoding='utf-8', errors='replace')
            mapping = load_json_mapping(json_path)
            if not mapping:
                continue

            ops: List[Tuple[int, int, str]] = []
            for key, info in mapping.items():
                if isinstance(info, str):
                    continue
                else:
                    prev = info.get('prev', '') or ''
                    next_ = info.get('next', '') or ''
                    original = info.get('original', '') or ''
                if not original:
                    continue

                ps, pe, ns, ne = _find_context_span(text, prev, next_)
                if ps == -1:
                    continue
                replace_start = pe
                replace_end = ns
                if replace_end < replace_start:
                    continue
                ops.append((replace_start, replace_end, original))

            if not ops:
                print("- No applicable context matches found.")
                continue

            ops.sort(key=lambda x: x[0], reverse=True)
            new_text = text
            for rs, re_, rep in ops:
                new_text = new_text[:rs] + rep + new_text[re_:]

            if math.fabs(len(text) - len(new_text)) > 1000 :
                print(f"[WARN] Text length changed: {len(text)} -> {len(new_text)}")
                new_text = text

            backup_path = (output_root / task_id / Path(json_f).stem).with_suffix('.tex')
            os.makedirs(backup_path.parent, exist_ok=True)
            backup_path.write_text(new_text, encoding='utf-8')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Restore LaTeX placeholders using parsed JSON mapping.')
    parser.add_argument('--latex-root', default='best_paper_tex_clean_format', help='Root dir of LaTeX sources from openai generated tex')
    parser.add_argument('--json-root', default='bestpaper_latex.bak.json', help='Root dir of JSON mappings from each page OCR generated tex')
    parser.add_argument('--output-root', default='bestpaper_latex', help='Root dir of LaTeX mappings')
    parser.add_argument('--task-id', default='2012_A', help='Task ID to process')
    parser.add_argument('--json-file', default='2.json', help='JSON file to process')
    args = parser.parse_args()

    process_task(
        latex_dir=Path(args.latex_root),
        json_root=Path(args.json_root),
        output_root=Path(args.output_root),
        task_id=args.task_id,
        json_file=args.json_file,
    )
