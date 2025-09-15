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

