from pathlib import Path
import argparse

from utils import (
    split_sections,
    read_text,
)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--latex-file', type=Path, default=None,
                        help='Path to the LaTeX file')
    parser.add_argument('--latex-dir', type=Path, default='MMBench/CPMCM/BestPaper',
                        help='Path to the LaTeX dir')
    parser.add_argument('--tex-filename', type=str, default='2.tex',
                        help='Filename of the LaTeX file')
    args = parser.parse_args()

    file_list = []
    if args.latex_file:
        file_list.append(args.latex_file)
    elif args.latex_dir:
        for _dir in args.latex_dir.iterdir():
            if _dir.is_dir():
                file_list.extend(_dir.glob(args.tex_filename))
            else:
                file_list.append(_dir)

    file_list = sorted(file_list)

    for file in file_list:
        content = read_text(file)
        sections = split_sections(content)
        for section in sections:
            if section['title'] == '目录':
                content = content.replace(section['content'], '\\tableofcontents')
                content = content.replace('\\section{目录}', '')
                content = content.replace('\\section*{目录}', '')
                file.write_text(content)
                print(f"✅ Successfully processed: {file}")
                break
