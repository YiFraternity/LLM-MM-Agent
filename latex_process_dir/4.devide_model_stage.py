import re
import shutil
from pathlib import Path
import argparse
import json
from typing import Dict, List

from utils import (
    find_task_id_from_path,
    split_sections,
    read_text,
    load_llm,
    prepare_batch_prompts,
    load_yaml,
    clean_json_txt,
)

def get_latex_sections(latex_file: Path) -> List[Dict[str, str]]:
    """
    Except Reference section and Extract sections from LaTeX content and categorize them by modeling stage.

    Args:
        latex_content: The full LaTeX document content

    Returns:
        List of sections
        {
            'title': section_title,
            'content': section_content
        }
    """
    latex_content = read_text(latex_file)

    ref_pattern = r'\\section\*?{.*(参考文献|References).*}'
    ref_match = re.search(ref_pattern, latex_content, re.IGNORECASE)
    if ref_match:
        content_before_refs = latex_content[:ref_match.start()]
    else:
        content_before_refs = latex_content

    sections = split_sections(content_before_refs)
    for section in sections:
        section['task_id'] = find_task_id_from_path(latex_file)
        section['sec_content'] = section['content'][:30000]
    return sections

def postprocess_output(inputs:List[dict], outputs):
    assert len(inputs) == len(outputs)
    for _input, _output in zip(inputs, outputs):
        section_class_lst = clean_json_txt(_output)
        if isinstance(section_class_lst, list):
            stages = []
            for _section_class in section_class_lst:
                stages.append(_section_class.get('modeling_stage', ''))
        else:
            stages = _output
        _input['stage'] = stages
        _input['tex'] = f"\\section{{{_input['title']}}}\n" + f"{_input['content']}\n\n"
    return inputs

def write_tex_file(jsonl_file: Path, TARGET_STAGES: set, output_dir: Path):
    """
    Read a .jsonl file where each line is a JSON object representing a section.
    Group sections by task_id and write one .tex file per task_id (overwriting if exists).
    Each section prefers the 'tex' field; if missing, it's constructed from 'title' and 'content'.
    """
    with jsonl_file.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            stage = obj.get('stage')
            section_class_lst = clean_json_txt(stage)
            if isinstance(section_class_lst, list):
                stages = set()
                for _section_class in section_class_lst:
                    stages.add(_section_class.get('modeling_stage', ''))
            else:
                stages = stage
            if isinstance(stages, set) and len(stages & TARGET_STAGES) > 0:
                with open(output_dir/f"{obj['task_id']}.tex", 'a', encoding='utf-8') as f:
                    tex = obj.get('tex', '')
                    f.write(tex + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--latex-file', type=Path, default=None,
                        help='Path to the LaTeX file')
    parser.add_argument('--latex-dir', type=Path, default='MMBench/CPMCM/BestPaper',
                        help='Path to the LaTeX dir')
    parser.add_argument('--tex-filename', type=str, default='2.tex',
                        help='Filename of the LaTeX file')
    parser.add_argument('--model-name-or-path', type=str, default='/group_homes/our_llm_domain/home/share/open_models/Qwen/Qwen2.5-32B-Instruct',
                        help='Path to the model')
    parser.add_argument('--gpu-num', type=int, default=2,
                        help='Number of GPUs to use')
    parser.add_argument('--prompt-template', type=Path, default='latex_process_dir/prompt/devide_model_stage.yaml',
                        help='Path to the prompt template')
    parser.add_argument('--tmp-dir', type=Path, default='tmp',
                        help='Path to the tmp dir')
    parser.add_argument('--output-dir', type=Path, default='BestPaper_only_model_stage',
                        help='Path to the output dir')
    args = parser.parse_args()

    prompt_template = load_yaml(args.prompt_template)
    # system_prompt = prompt_template['system_prompt']
    user_prompt = prompt_template['system_prompt']['zh']
    file_list = []
    if args.latex_file:
        file_list.append(args.latex_file)
    elif args.latex_dir:
        for _dir in args.latex_dir.iterdir():
            if _dir.is_dir():
                file_list.extend(_dir.glob(args.tex_filename))
            else:
                file_list.append(_dir)

    file_list = sorted(file_list, key=lambda x: x.stem)
    sections = []
    for file in file_list:
        sections.extend(get_latex_sections(file))
    all_prompts = prepare_batch_prompts(sections, user_prompt)
    print(all_prompts[0])

    model, sampling_params = load_llm(args.model_name_or_path, gpu_num=args.gpu_num)
    outputs_t = model.chat(all_prompts, sampling_params, use_tqdm=True)

    pred_lst = []
    for o_t in outputs_t:
        pred_lst.append(o_t.outputs[0].text)

    TARGET_STAGES = {
        "假设建立",
        "模型构建",
        "模型求解"
    }
    sections = postprocess_output(sections, pred_lst)

    shutil.rmtree(args.tmp_dir, ignore_errors=True)
    shutil.rmtree(args.output_dir, ignore_errors=True)

    args.tmp_dir.mkdir(exist_ok=True)
    args.output_dir.mkdir(exist_ok=True)

    for section in sections:
        with open(args.tmp_dir/f"{section['task_id']}.jsonl", 'a', encoding='utf-8') as f:
            f.write(json.dumps(section, ensure_ascii=False) + '\n')
        stages = section.get('stage')
        stages_set = set(stages)
        intersection = stages_set & TARGET_STAGES
        if len(intersection) == 0:
            continue
        with open(args.output_dir/f"{section['task_id']}.tex", 'a', encoding='utf-8') as f:
            tex = section.get('tex', '')
            f.write(tex + '\n')
    # for file in args.tmp_dir.iterdir():
    #     write_tex_file(file, TARGET_STAGES, args.output_dir)
