"""
This code defines a Python module for parsing LaTeX `.tex` files,
specifically focusing on extracting the document's structure (sections, subsections),
embedded elements (equations, tables), and identifying potentially unclosed LaTeX constructs.
It also includes functionality to collect context around certain complex LaTeX elements
(like figures and display math)
without modifying the original text, aiming to aid in later content extraction or analysis.
storing the results in a structured JSON format.

The module defines several global regular expressions (`re.compile`) to identify key LaTeX constructs:

| Variable | Pattern | Description |
| :--- | :--- | :--- |
| `SECTION_RE` | `\\section\{([^}]*)\}` | Matches top-level sections. |
| `SUBSECTION_RE` | `\\subsection\{([^}]*)\}` | Matches subsections. |
| `SUBSUB_RE` | `\\subsubsection\{([^}]*)\}` | Matches subsubsections. |
| `EQUATION_ENV_RE` | `\\begin\{equation\}(.*?)\\end\{equation\}` | Matches equations in the `equation` environment. |
| `DISPLAY_MATH_RE` | `\\\[(.*?)\\\]` | Matches display math defined by `\[...\]`. |
| `INLINE_EQ_RE` | `\$(.+?)\$` | Matches inline math defined by `$ ... $`. |
| `TABULAR_RE` | `\\begin\{tabular\}.*?\\end\{tabular\}` | Matches `tabular` environments (often inside tables). |

"""
import os
import argparse
from pathlib import Path

from utils import (
    parse_tex,
    save_json,
)
def main():
    parser = argparse.ArgumentParser(description='Parse LaTeX .tex files into structured JSON')
    parser.add_argument('--path', default='bestpaper_latex.bak', help='.tex file or directory containing .tex files')
    parser.add_argument('--json', action='store_true', help='if set, save parsed output to .json files next to sources; otherwise print a concise summary')
    parser.add_argument('--json-dir', default='bestpaper_latex.bak.json', help='optional directory to save all json outputs (if set, overrides per-file location)')
    args = parser.parse_args()

    p = Path(args.path)
    task_ids = os.listdir(p)
    for task_id in task_ids:
        latex_files = os.listdir(p / task_id)
        out_base = Path(args.json_dir)/task_id
        for latex_file in latex_files:
            f = p / task_id / latex_file
            parsed = parse_tex(f)
            out = out_base / latex_file
            out = out.with_suffix('.json')
            out.parent.mkdir(parents=True, exist_ok=True)
            save_json(parsed, out)
            print(f'Wrote {out}')
    return 0

if __name__ == '__main__':
    main()