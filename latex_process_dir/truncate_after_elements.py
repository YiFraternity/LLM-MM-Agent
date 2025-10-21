import re
import argparse
from pathlib import Path
from typing import Optional, List

from utils import (
    find_task_id_from_path,
)
def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def remove_elements(text: str, remove_inline_math: bool = False) -> str:
    """
    Remove LaTeX table/figure/equation content from the text.

    - Removes environments: table, table*, longtable, tabular, tabular*, figure, figure*,
      and math envs: equation, equation*, align, align*, gather, gather*, multline, multline*,
      eqnarray, eqnarray*, displaymath, math.
    - Removes display math forms: \[...\], $$...$$.
    - Optionally removes inline math delimited by single $...$ (default True).
    """
    # Build environment list
    table_envs = [
        'table', 'table*', 'longtable',
        'tabular', 'tabular*', 'tabularx'
    ]
    figure_envs = ['figure', 'figure*']
    math_envs = [
        'equation', 'equation*',
        'align', 'align*',
        'gather', 'gather*',
        'multline', 'multline*',
        'eqnarray', 'eqnarray*',
        'displaymath', 'math'
    ]

    # Remove environments
    def remove_env(text_in: str, env: str) -> str:
        pat = re.compile(rf"\\begin\{{{re.escape(env)}\}}[\s\S]*?\\end\{{{re.escape(env)}\}}", re.DOTALL | re.IGNORECASE)
        return pat.sub("", text_in)

    for env in table_envs + figure_envs + math_envs:
        text = remove_env(text, env)

    # Remove display math forms
    text = re.sub(r"\\\[[\s\S]*?\\\]", "", text, flags=re.DOTALL)  # \[ ... \]
    text = re.sub(r"\$\$[\s\S]*?\$\$", "", text, flags=re.DOTALL)            # $$ ... $$

    # Optionally remove inline math delimited by single $
    if remove_inline_math:
        # Match a single $ ... $ that is not $$ and allow escaped characters inside
        inline_pat = re.compile(r"(?<!\$)\$([\s\S]*?)(?<!\$)\$(?!\$)")
        text = inline_pat.sub("", text)

    return text


def truncate_at_first_trigger(text: str) -> str:
    # Deprecated behavior: keep for compatibility if referenced elsewhere
    return text


def process_file(in_path: Path, out_path: Optional[Path], in_place: bool = False) -> Path:
    text = read_text(in_path)
    truncated = remove_elements(text, remove_inline_math=False)

    if in_place:
        in_path.write_text(truncated, encoding="utf-8")
        return in_path

    if out_path is None:
        out_path = in_path.with_suffix("")
        out_path = out_path.with_name(out_path.name + ".truncated").with_suffix(".tex")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(truncated, encoding="utf-8")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Remove LaTeX tables, figures, and equations from files, preserving other content.")
    parser.add_argument("--raw-latex-dir", default=None, type=Path,
                        help="Path to input .tex file")
    parser.add_argument('--raw-latex-files', nargs='+', type=Path,
                        help="List of input .tex files")
    parser.add_argument("--truncated-latex-dir", default="bestpaper_latex.bak.truncated", type=Path,
                        help="Directory to write processed .tex files")
    parser.add_argument("--in-place", action="store_true", help="Modify the input file in place")
    parser.add_argument("--keep-inline", action="store_true", help="Keep inline $...$ math instead of removing it")
    args = parser.parse_args()

    raw_latex_files: List[Path] = []
    if args.raw_latex_dir is not None:
        in_path = Path(args.raw_latex_dir)
        if not in_path.exists():
            raise FileNotFoundError(f"Input file not found: {in_path}")
        raw_latex_files = list(in_path.glob("**/*.tex"))
    elif args.raw_latex_files is not None:
        raw_latex_files = args.raw_latex_files
    else:
        raise ValueError("Either --raw-latex-dir or --raw-latex-files must be specified")

    truncated_latex_dir = Path(args.truncated_latex_dir)
    truncated_latex_dir.mkdir(parents=True, exist_ok=True)

    for raw_latex_file in raw_latex_files:
        if not raw_latex_file.exists():
            print(f"Input file not found: {raw_latex_file}")
            continue
        task_id = find_task_id_from_path(raw_latex_file)
        out_base = truncated_latex_dir / task_id
        out_path = out_base / raw_latex_file.name
        # Respect keep-inline flag
        text = read_text(raw_latex_file)
        processed = remove_elements(text, remove_inline_math=not args.keep_inline)
        out_base.mkdir(parents=True, exist_ok=True)
        if args.in_place:
            raw_latex_file.write_text(processed, encoding="utf-8")
        else:
            out_path.write_text(processed, encoding="utf-8")


if __name__ == "__main__":
    main()
