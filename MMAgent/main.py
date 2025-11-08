
import os
from pathlib import Path
import time
import argparse
import logging

from dotenv import load_dotenv
from llm.llm import LLM
from utils.problem_analysis import problem_analysis
from utils.mathematical_modeling import mathematical_modeling
from utils.computational_solving import computational_solving
from utils.solution_reporting import generate_paper
from utils.utils import write_json_file, get_info


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

load_dotenv(override=True)
def run(problem_path, config, name, dataset_path, output_dir, tmp_dir: Path):
    # Initialize LLM
    llm = LLM()

    # Stage 1: Problem Analysis
    logger.info('➡️  Stage 1: Problem Analysis start ')
    problem, task_id, solution, coordinator = problem_analysis(
        llm,
        problem_path,
        config,
        dataset_path,
        output_dir,
        tmp_dir=tmp_dir,
    )
    task_descriptions = solution['task_descriptions']
    with_code = solution.get('with_code', False)
    order = solution['order']
    logger.info('✅  Stage 1: Problem Analysis finish')

    # Stage 2 & 3: Mathematical Modeling & Computational Solving
    logger.info('➡️  Stage 2 & 3: Mathematical Modeling & Computational Solving start ')
    for id in order:
        logger.info('▶️ Solving Task {} '.format(id))
        subtask_solution, dependent_file_prompt = mathematical_modeling(
            llm,
            task_id,
            id,
            problem,
            task_descriptions,
            config,
            coordinator,
            with_code,
            tmp_dir=tmp_dir,
        )
        task_description = subtask_solution['task_description']
        task_analysis = subtask_solution['task_analysis']
        task_modeling_formulas = subtask_solution['task_modeling_formulas']
        task_modeling_method = subtask_solution['task_modeling_method']
        solution = computational_solving(
            llm,
            task_id,
            coordinator,
            with_code,
            problem,
            id,
            task_description,
            task_analysis,
            task_modeling_formulas,
            task_modeling_method,
            dependent_file_prompt,
            config,
            solution,
            name,
            output_dir,
            tmp_dir=tmp_dir,
        )
    logger.info('✅  Stage 2 & 3: Mathematical Modeling & Computational Solving finish ')

    # optional
    ckpt_path = tmp_dir / task_id / "paper_solution.json"
    logger.info('➡️  Stage 4: Solution Reporting start ')
    generate_paper(llm, output_dir, name, config['mathmodel_category'], ckpt_path)
    logger.info('✅  Stage 4: Solution Reporting finish ')

    # print(solution)
    logger.info('Usage: %s', llm.get_total_usage())
    write_json_file(f'{output_dir}/usage/{name}.json', llm.get_total_usage())
    return solution


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mm-dataset', type=str, default='CPMCM')
    parser.add_argument('--method-name', type=str, default='MM-Agent')
    parser.add_argument('--task', type=str, default='2020_F')
    parser.add_argument('--tmp-dir', type=Path, default='tmp/output')
    parser.add_argument('--model-name', type=str, default=None,
                        help='Name of llm, if None, return os.MODEL_NAME')
    parser.add_argument('--embed-model', type=str, default=None,
                        help='Name of llm, if None, return os.EMBED_MODEL')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    args.model_name = args.model_name or os.getenv('MODEL_NAME')
    args.embed_model = args.embed_model or os.getenv('EMBED_MODEL')
    problem_path, config, dataset_dir, output_dir = get_info(args)
    config['mathmodel_category'] = args.mm_dataset
    start = time.time()
    logger.info('Config: %s', config)
    solution = run(
        problem_path=problem_path,
        config=config,
        name=args.task,
        dataset_path=dataset_dir,
        output_dir=output_dir,
        tmp_dir=args.tmp_dir,
    )
    end = time.time()
    with open(output_dir + '/usage/runtime.txt', 'w') as f:
        f.write("{:.2f}s".format(end - start))
