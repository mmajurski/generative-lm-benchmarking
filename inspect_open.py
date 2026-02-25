import argparse
import asyncio
import os
import copy
from inspect_ai import eval
from inspect_ai.model import GenerateConfig
from inspect_ai.model import get_model

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.scorer import model_graded_qa
from inspect_ai.solver import Generate, Solver, solver, TaskState
from inspect_ai.scorer._model import DEFAULT_MODEL_GRADED_FACT_TEMPLATE
import vllm_inspect_provider
import utils
import time
import json
import random

from dotenv import load_dotenv

load_dotenv()


# GRADER_MODEL = 'v_llm/gpt-oss-120b'
# GRADER_MODEL_BASE_URL = 'https://rchat.nist.gov/api'
# GRADER_MODEL_API_KEY = os.getenv("RCHAT_API_KEY")

GRADER_MODEL = 'v_llm/gpt120b'
GRADER_MODEL_BASE_URL = 'https://pn131285.nist.gov:8443/v1'
GRADER_MODEL_API_KEY=os.getenv("VLLM_API_KEY")

OE_QUESTIONS_DIR = './demo/oe_questions'


@solver
def open_ended_qa() -> Solver:

    template = r"""
Answer the following open ended short answer question. The last line of your response should be of the following format: 'ANSWER: $answer' (without quotes) where answer is the answer to the question. Think step by step before answering.

{question}
""".strip()

    async def solve(state: TaskState, generate: Generate) -> TaskState:

        state.user_prompt.text = template.format(question=state.user_prompt.text)
        state = await generate(state)

        val = state.output.completion
        val = val.strip()
        val = val.split('\n')
        if val[-1].startswith('ANSWER: '):
            val = val[-1]
            state.output.completion = val

        return state

    return solve


def base_task(lcl_fp, num_samples):

    config = GenerateConfig(max_connections=16)
    model = get_model(model=GRADER_MODEL, base_url=GRADER_MODEL_BASE_URL, config=config, api_key=GRADER_MODEL_API_KEY)

    with open(lcl_fp, 'r') as f:
        ds = json.load(f)

    random.shuffle(ds)
    samples = list()
    for row in ds:
        q = row['question']
        a = row['answer']
        if 'answer_index' in row:
            a = row['options'][row['answer_index']]
        samples.append(Sample(input=str(q), target=str(a)))

    samples = samples[:num_samples]

    return Task(
        dataset=samples,
        solver=open_ended_qa(),
        scorer=model_graded_qa(model=model, template=DEFAULT_MODEL_GRADED_FACT_TEMPLATE),
    )


def _make_task_fn(filepath: str):
    def fn(num_samples: int = 200):
        return base_task(filepath, num_samples)
    return fn


def _register_tasks_from_dir(directory: str) -> dict:
    registry = {}
    if not os.path.isdir(directory):
        return registry
    for fn in sorted(os.listdir(directory)):
        if not fn.endswith('.json'):
            continue
        name = os.path.splitext(fn)[0]
        filepath = os.path.abspath(os.path.join(directory, fn))
        task_fn = _make_task_fn(filepath)
        task_fn.__name__ = name
        registered = task(task_fn)
        globals()[name] = registered
        registry[name] = filepath
    return registry


_task_registry = _register_tasks_from_dir(OE_QUESTIONS_DIR)


def get_task_dir_dict(dataset_fldr: str) -> dict:
    result = {}
    if not os.path.isdir(dataset_fldr):
        return result
    for fn in sorted(os.listdir(dataset_fldr)):
        if fn.endswith('.json'):
            name = os.path.splitext(fn)[0]
            result[name] = os.path.abspath(os.path.join(dataset_fldr, fn))
    return result


def get_task(name: str, filepath: str, num_samples: int = 200):
    return base_task(filepath, num_samples)


def run(base_dir: str, models_config: list[dict], batch_size: int = 32, num_samples: int = 200,
        log_dir: str = None):
    """
    Args:
        models_config: list of dicts with keys: model_name, base_url, api_key
        log_dir: directory to store evaluation logs (defaults to <base_dir>/logs)
    """
    asyncio.set_event_loop(asyncio.new_event_loop())
    config = GenerateConfig(max_connections=batch_size, max_tokens=1024)
    models_dict = {
        m['model_name']: get_model(model=m['model_name'], base_url=m['base_url'], config=config, api_key=m['api_key'])
        for m in models_config
    }

    available_models = list(models_dict.keys())

    available_task_names_dict = get_task_dir_dict(base_dir)

    to_remove = [k for k, v in available_task_names_dict.items() if not os.path.exists(v)]
    for k in to_remove:
        print("missing task: ", k, " at ", available_task_names_dict[k])
        available_task_names_dict.pop(k)
    available_task_names = list(available_task_names_dict.keys())

    if log_dir is None:
        log_dir = os.path.join(base_dir, "logs")
    print("discovering completed logs...")
    completed_logs, completed_fns = utils.get_completed_logs(log_dir)

    unused_logs = copy.deepcopy(completed_logs)

    for task_name in available_task_names:
        print("--------------------------------")
        print(f"Processing folder {base_dir} task {task_name}")

        to_remove_models = []
        for log in completed_logs:
            if log['task'] == task_name:
                if log['model'].replace('v_llm/', '') in available_models:
                    to_remove_models.append(log['model'].replace('v_llm/', ''))
                    unused_logs.remove(log)
        work_models = list(set(available_models) - set(to_remove_models))

        if len(work_models) == 0:
            print(f"  No missing models")
            continue

        print(f"  Missing {len(work_models)} Models:")
        for model in work_models:
            print(f"    {model}")
        print("--------------------------------")

        models_list = [models_dict[model] for model in work_models]
        json_fp = available_task_names_dict[task_name]
        work_tasks = [get_task(task_name, json_fp, num_samples)]
        # display = 'full', 'plain', 'rich'
        eval(work_tasks, model=models_list, display='rich', log_format='json', no_log_images=True, no_log_samples=True, log_dir=log_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluates a MMLU style dataset using Inspect framework.')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--base_dir', type=str, default=OE_QUESTIONS_DIR, help='where the dataset json files are stored')
    parser.add_argument('--num_samples', type=int, default=200)
    parser.add_argument('--models_config', type=str, required=True,
                        help='path to a JSON file containing a list of model configs, '
                             'each with fields: model_name, base_url, api_key')
    parser.add_argument('--log_dir', type=str, default=None, help='directory to store evaluation logs (defaults to <base_dir>/logs)')

    args = parser.parse_args()

    with open(args.models_config, 'r') as f:
        models_config = json.load(f)

    run(base_dir=args.base_dir, models_config=models_config, batch_size=args.batch_size,
        num_samples=args.num_samples, log_dir=args.log_dir)
