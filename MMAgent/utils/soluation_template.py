from typing import Dict, List, Any


class SafeDict(dict):
    """安全的字典类，缺失的键返回空格"""
    def __missing__(self, key):
        return " "


CPMCM = r"""
\documentclass[bwprint]{{gmcmthesis}}
\usepackage[framemethod=TikZ]{{mdframed}}

\usepackage{{subfig}}
\usepackage{{colortbl}}
\usepackage{{palatino}}
\usepackage{{algorithm}}
\usepackage{{algorithmicx}}
\usepackage{{algpseudocode}}
\usepackage{{tocloft}}
\usepackage{{amsmath}}
\usepackage{{xcolor}}
\definecolor{{color1}}{{rgb}}{{0.78,0.88,0.99}}
\definecolor{{color2}}{{rgb}}{{0.36,0.62,0.84}}
\definecolor{{color3}}{{rgb}}{{0.88,0.92,0.96}}
\definecolor{{color4}}{{rgb}}{{0.96,0.97,0.98}}

\floatname{{algorithm}}{{算法}}
\renewcommand{{\algorithmicrequire}}{{\textbf{{输入:}}}}
\renewcommand{{\algorithmicensure}}{{\textbf{{输出:}}}}

\newcommand{{\red}}[1]{{\textcolor{{red}}{{#1}}}}
\newcommand{{\blue}}[1]{{\textcolor{{blue}}{{#1}}}}

\title{{{title}}}
\baominghao{{{baominghao}}}
\schoolname{{{schoolname}}}
\membera{{{membera}}}
\memberb{{{memberb}}}
\memberc{{{memberc}}}

\begin{{document}}
"""

MCMICM = r"""
\documentclass{{mcmthesis}}
\mcmsetup{{CTeX = false,
        tcn = {team}, problem = {problem_type},
        year = {year},
        sheet = true, titleinsheet = true, keywordsinsheet = true,
        titlepage = false, abstract = true}}

\usepackage{{palatino}}
\usepackage{{algorithm}}
\usepackage{{algpseudocode}}
\usepackage{{tocloft}}
\usepackage{{amsmath}}

\usepackage{{lastpage}}
\renewcommand{{\cftdot}}{{.}}
\renewcommand{{\cftsecleader}}{{\cftdotfill{{\cftdotsep}}}
\renewcommand{{\cftsubsecleader}}{{\cftdotfill{{\cftdotsep}}}
\renewcommand{{\cftsubsubsecleader}}{{\cftdotfill{{\cftdotsep}}}
\renewcommand{{\headset}}{{{year}\MCM/ICM\Summary Sheet}}
\title{{{title}}}

\begin{{document}}
"""


def add_figure(figures: List[str]) -> List[str]:
    """插入图片"""
    figure_str: List[str] = []
    for figure_path in figures:
        name = figure_path.split('/')[-1].split('.')[0].replace('_', '\\_')
        figure_str.append(f"""
\\begin{{figure}}[H]
\\centering
\\includegraphics[width=0.5\\textwidth]{{{figure_path}}}
\\caption{{图：{name}}}
\\end{{figure}}
""")
    return figure_str


def add_code(codes: List[str]) -> List[str]:
    """附录中包含 Python 代码清单"""
    code_str: List[str] = [
        "\\clearpage",
        "\\section{附录}",
    ]
    for code_path in codes:
        with open(code_path, 'r') as f:
            code = f.read()
        name = code_path.split('/')[-1].replace('_', '\\_')
        code_str.append(f"""
\\subsubsection*{{{name}}}

\\begin{{lstlisting}}[language=Python, frame=single, basicstyle=\\ttfamily\\small]
{code}
\\end{{lstlisting}}
""")
    return code_str


def create_preamble(metadata: Dict[str, Any], mathmodel_category: str) -> str:
    """LaTeX 导言区"""
    metadata['title'] = metadata.get("title", "paper_title")
    metadata['baominghao'] = metadata.get("baominghao", "0123")
    metadata['schoolname'] = metadata.get("schoolname", "Agent")
    metadata['membera'] = metadata.get("membera", "Agent")
    metadata['memberb'] = metadata.get("memberb", "Agent")
    metadata['memberc'] = metadata.get("memberc", "Agent")
    contest = eval(mathmodel_category)
    print(metadata)
    return contest.format_map(SafeDict(metadata))


def create_abstract(metadata: Dict[str, str]) -> str:
    """摘要与关键词（环境名仍为 abstract/keywords）"""
    return f"""\\begin{{abstract}}
{metadata.get('summary', '')}

\\begin{{keywords}}
{metadata.get('keywords', '')}
\\end{{keywords}}
\\end{{abstract}}"""
