#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import argparse
from typing import List, Dict, Tuple, Optional
from dotenv import load_dotenv
import yaml

# OpenAI SDK (>=1.0)
import tiktoken
try:
    from openai import OpenAI
except Exception:
    OpenAI = None


load_dotenv(override=True)
PLACEHOLDER_LABELS = [
    'TABLEENV', 'FIGUREENV', 'MATHENV', 'DISPLAYMATH', 'GRAPHIC', 'REFERENCES'
]
PLACEHOLDER_REGEX = re.compile(
    r"\[(?:" + "|".join(PLACEHOLDER_LABELS) + r"):\d+\]"
)


def read_text(path: str, encoding: str = 'utf-8') -> str:
    with open(path, 'r', encoding=encoding, errors='ignore') as f:
        return f.read()


def write_text(path: str, content: str, encoding: str = 'utf-8') -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', encoding=encoding) as f:
        f.write(content)


def mask_placeholders(tex: str) -> Tuple[str, Dict[str, str]]:
    """
    Replace placeholders like [TABLEENV:1] with sentinel tokens that the model is
    unlikely to touch, then later restore them exactly.
    Returns: (masked_tex, mapping)
    """
    mapping: Dict[str, str] = {}
    def repl(m: re.Match) -> str:
        ph = m.group(0)
        key = f"__PH__{len(mapping)+1:06d}__"
        mapping[key] = ph
        return key
    masked = PLACEHOLDER_REGEX.sub(repl, tex)
    return masked, mapping


def unmask_placeholders(text: str, mapping: Dict[str, str]) -> str:
    for key, ph in mapping.items():
        text = text.replace(key, ph)
    return text


def load_prompts(config_path: str) -> Tuple[str, str]:
    with open(config_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    system_prompt: str = data.get('system_prompt', '').strip()
    user_prompt: str = data.get('user_prompt', '').strip()
    if not system_prompt or not user_prompt:
        raise ValueError('Config must contain system_prompt and user_prompt')
    return system_prompt, user_prompt


def build_messages(system_prompt: str, user_prompt: str, tex: str) -> list:
    user_content = user_prompt + "\n\n以下是需要修订的 LaTeX 文档：\n\n```latex\n" + tex + "\n```"
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


def extract_latex_from_response(text: str) -> Optional[str]:
    # Extract content inside a ```latex ... ``` code block
    code_block = re.search(r"```latex\s*([\s\S]*?)\s*```", text)
    if code_block:
        return code_block.group(1).strip()
    # Fallback: try generic triple backticks
    code_block = re.search(r"```\s*([\s\S]*?)\s*```", text)
    if code_block:
        return code_block.group(1).strip()
    return None


def call_openai(messages: list, model: str, temperature: float = 0.0, max_tokens: int = 8192) -> str:
    if OpenAI is None:
        raise RuntimeError("openai SDK not installed. pip install openai>=1.0.0")
    client = OpenAI(
        api_key=os.getenv('OPENAI_API_KEY'),
        base_url=os.getenv('OPENAI_API_BASE')
    )
    total_tokens = 0
    encoding = tiktoken.get_encoding("cl100k_base")
    for message in messages:
        for key, value in message.items():
            total_tokens += len(encoding.encode(value))
    print(f'total_tokens: {total_tokens}')
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content or ""

def process_single_file(input_path: str, output_path: str, system_prompt: str, user_prompt: str, args: argparse.Namespace) -> None:
    """Processes a single LaTeX file.
    
    Args:
        input_path: Path to the input LaTeX file
        output_path: Path where the processed file will be saved
        system_prompt: System prompt for the OpenAI API
        user_prompt: User prompt for the OpenAI API
        args: Command line arguments
    """
    # Skip if output file already exists
    if os.path.exists(output_path):
        print(f"Skipping (output exists): {output_path}")
        return
        
    print(f"Processing: {input_path}")
    src = read_text(input_path)
    # 1) Mask placeholders
    masked_tex, mapping = mask_placeholders(src)

    # 2) Call OpenAI
    messages = build_messages(system_prompt, user_prompt, masked_tex)
    response_text = call_openai(messages, model=args.model, temperature=args.temperature, max_tokens=args.max_tokens)

    # 3) Extract and unmask
    latex = extract_latex_from_response(response_text)
    if not latex:
        latex = response_text
    latex = unmask_placeholders(latex, mapping)

    # 4) Verify and restore placeholders
    orig_ph = sorted(PLACEHOLDER_REGEX.findall(src))
    new_ph = sorted(PLACEHOLDER_REGEX.findall(latex))
    if orig_ph != new_ph:
        missing = [ph for ph in orig_ph if ph not in new_ph]
        if missing:
            latex += "\n\n% Missing placeholders restored\n" + "\n".join(missing)

    write_text(output_path, latex)
    print(f"Saved formatted LaTeX to: {output_path}")

def process_directory(input_dir: str, output_dir: str, system_prompt: str, user_prompt: str, args: argparse.Namespace) -> None:
    """
    Recursively finds and processes all .tex files in a directory.
    """
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.tex'):
                input_path = os.path.join(root, file)

                # Determine the relative path to maintain directory structure
                rel_path = os.path.relpath(input_path, input_dir)
                output_path = os.path.join(output_dir, rel_path)

                # Ensure the output directory structure exists
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                try:
                    process_single_file(input_path, output_path, system_prompt, user_prompt, args)
                except Exception as e:
                    print(f"Failed to process {input_path}: {e}")

# --- Updated Main Function ---
def main():
    parser = argparse.ArgumentParser(description='Format LaTeX using OpenAI and preserve placeholders.')
    parser.add_argument('--input', '-i', type=str, required=True, help='Path to input .tex file or directory')
    parser.add_argument('--output', '-o', type=str, required=True, help='Path to save formatted .tex file or directory')
    parser.add_argument('--model', '-m', type=str, default='gpt-4o-mini', help='OpenAI model name')
    parser.add_argument('--config', '-c', type=str, default='', help='Path to YAML config with prompts')
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--max_tokens', type=int, default=32768)

    args = parser.parse_args()

    # Resolve config path
    config_path = args.config
    if not config_path:
        here = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(here, 'format_tex_openai.yaml')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f'Prompt config not found: {config_path}')

    system_prompt, user_prompt = load_prompts(config_path)

    # Check if input is a file or a directory
    if os.path.isdir(args.input):
        if not os.path.isdir(args.output):
            print("Output must be a directory when input is a directory. Creating it...")
            os.makedirs(args.output, exist_ok=True)
        process_directory(args.input, args.output, system_prompt, user_prompt, args)
    elif os.path.isfile(args.input):
        process_single_file(args.input, args.output, system_prompt, user_prompt, args)
    else:
        raise FileNotFoundError(f"Input path not found: {args.input}")

if __name__ == '__main__':
    main()