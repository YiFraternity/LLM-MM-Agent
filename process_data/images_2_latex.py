"""
DevRead: A Versatile File Processing Library for Text, PDFs, DOCX, JSON,
XML, YAML, HTML, Markdown, LaTeX, PPTX, Excel, Images, and Videos, etc.
"""

import os
import base64
from io import BytesIO
import json
import logging
from pathlib import Path
from typing import List, Any
import ijson
from PIL import Image
from rich.logging import RichHandler
from rich.console import Console
from tqdm import tqdm
from vllm import LLM, SamplingParams

from jinja2 import Template, StrictUndefined
from natsort import natsorted
import yaml

console = Console()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[RichHandler()],
)
logger = logging.getLogger(__name__)

def load_yaml(yaml_path: str) -> dict[str, Any]:
    with open(yaml_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def write_json(data: dict[str, Any], json_path: str) -> None:
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def populate_template(template: str, variables: dict[str, Any]) -> str:
    """
    Populate a Jinja template with variables.
    """
    compiled_template = Template(template, undefined=StrictUndefined)
    try:
        return compiled_template.render(**variables)
    except Exception as e:
        raise Exception(f"Error during jinja template rendering: {type(e).__name__}: {e}")


def load_llm(model_name_or_path, tokenizer_name_or_path=None, gpu_num=1, lora_model_name_or_path=None):
    """
    Load a VLLM model.
    """
    kw_args = {
        "model": model_name_or_path,
        # "tokenizer": tokenizer_name_or_path,
        "tokenizer_mode": "slow",
        "tensor_parallel_size" : gpu_num,
        # "enable_lora": bool(lora_model_name_or_path)
        'max_model_len': 32768,
    }
    llm = LLM(**kw_args)
    kwargs={
        "n":1,
        "max_tokens": 8192,
        "top_p":1.0,
        # sampling
        "temperature":0,
        'top_k': 1,
    }
    sampling_params = SamplingParams(**kwargs)
    return llm, sampling_params

def _encode_image(image_path: str) -> str:
    """优化的图像编码方法，支持 jpg/png 等格式"""
    with Image.open(image_path) as img:
        img = img.convert("RGB")  # 统一转换为RGB，保证兼容性
        buffered = BytesIO()
        img.save(buffered, format="PNG", quality=95)  # 保存为PNG，确保无损
        return base64.b64encode(buffered.getvalue()).decode("utf-8")


def prepare_batch_prompts(
    image_paths: List[dict],
    prompt_template: str,
    system_prompt: str = '',
) -> List[str]:
    """
    Prepare a batch of prompts for inference, with progress bar.
    """
    prompts_kwargs = [{} for _ in image_paths]
    prompts = [populate_template(prompt_template, prompt_kwarg) for prompt_kwarg in prompts_kwargs]

    if system_prompt == '':
        sys_prompt = 'You are a helpful AI assistant.'
    else:
        sys_prompt = system_prompt
    system_prompts = [sys_prompt for _ in prompts]

    message_list = []
    for prompt, sys_prompt, image_path in tqdm(
        zip(prompts, system_prompts, image_paths),
        total=len(image_paths),
        desc="Encoding images"
    ):
        image_url = _encode_image(image_path['image_path'])
        message_list.append([
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image;base64,{image_url}"}} ,
                {"type": "text", "text": prompt},
            ]}
        ])
    return message_list


def postprocess_output(inputs:List[dict], outputs):
    """
    Postprocess the output of the model.
    """
    assert len(inputs) == len(outputs)

    for _input, _output in zip(inputs, outputs):
        _input['output'] = _output
    return inputs

def main():
    source_dir = Path('/home/yhliu/MathModeling/LLM-MM-Agent/images')
    # Prepare prompts
    templates = load_yaml('process_data/images_2_latex.yaml')
    prompt_template = templates['user_prompt']
    system_prompt = templates['system_prompt']
    image_paths = []
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.endswith('.png') or file.endswith('.jpg'):
                image_path = os.path.join(root, file)
                task_id = os.path.basename(os.path.dirname(root))
                paper_id = os.path.basename(root)
                image_paths.append({'image_path': image_path, 'paper_id': paper_id, 'task_id': task_id})
    image_paths = natsorted(image_paths, key=lambda x: x['image_path'])
    model_name_or_path = '/home/share/models/modelscope/Qwen/Qwen2.5-VL-32B-Instruct'
    model, sampling_params = load_llm(model_name_or_path, gpu_num=4)

    # all_prompts = prepare_batch_prompts(image_paths, prompt_template, system_prompt)
    # with open('all_prompts_debug.json', 'w', encoding='utf-8') as f:
    #     json.dump(all_prompts, f, ensure_ascii=False, indent=2)
    # exit(0)
    pred_lst = []
    # 迭代读取 JSON 数组
    batch_size = 16  # 可以调整
    with open('all_prompts_debug.json', 'r', encoding='utf-8') as f_in, \
         open('images_2_latex.jsonl', 'w', encoding='utf-8') as f_out:

        prompts_iter = ijson.items(f_in, 'item')
        batch, batch_image_paths = [], []
        i = 0

        for prompt in tqdm(prompts_iter, desc="Processing prompts"):
            batch.append(prompt)
            batch_image_paths.append(image_paths[i])
            i += 1
            if len(batch) >= batch_size:
                outputs_t = model.chat(batch, sampling_params, use_tqdm=False)
                pred_lst = [o_t.outputs[0].text for o_t in outputs_t]

                results = postprocess_output(batch_image_paths, pred_lst)
                for result in results:
                    f_out.write(json.dumps(result, ensure_ascii=False) + '\n')

                f_out.flush()  # 及时写盘
                batch, batch_image_paths = [], []

        # 处理最后不足 batch 的部分
        if batch:
            outputs_t = model.chat(batch, sampling_params, use_tqdm=False)
            pred_lst = [o_t.outputs[0].text for o_t in outputs_t]

            results = postprocess_output(batch_image_paths, pred_lst)
            for result in results:
                f_out.write(json.dumps(result, ensure_ascii=False) + '\n')

            f_out.flush()


if __name__ == "__main__":
    main()
