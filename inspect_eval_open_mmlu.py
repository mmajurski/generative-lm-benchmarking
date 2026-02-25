
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

# Load environment variables from .env file
load_dotenv()



# GRADER_MODEL = 'v_llm/openai/gpt-oss-120b'
# GRADER_MODEL_BASE_URL = 'https://pn131285.nist.gov:8443/v1'
# GRADER_MODEL_API_KEY=os.getenv("VLLM_API_KEY")

GRADER_MODEL = 'v_llm/openai/gpt-oss-120b'
GRADER_MODEL_BASE_URL = 'https://rchat.nsit.gov/api'
GRADER_MODEL_API_KEY=os.getenv("RCHAT_API_KEY")


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
        dataset = samples,
        solver=open_ended_qa(),
        scorer=model_graded_qa(model=model, template=DEFAULT_MODEL_GRADED_FACT_TEMPLATE),
    )





@task
def mmlu_pro_biology(dataset_fldr, num_samples):
    return base_task(dataset_fldr, num_samples)

@task
def mmlu_pro_chemistry(dataset_fldr, num_samples):
    return base_task(dataset_fldr, num_samples)  

@task
def mmlu_pro_computer_science(dataset_fldr, num_samples):
    return base_task(dataset_fldr, num_samples)  

@task
def mmlu_pro_math(dataset_fldr, num_samples):
    return base_task(dataset_fldr, num_samples)  

@task
def mmlu_pro_physics(dataset_fldr, num_samples):
    return base_task(dataset_fldr, num_samples)

@task
def mmlu_pro_psychology(dataset_fldr, num_samples):
    return base_task(dataset_fldr, num_samples)

def get_task_dir_dict(dataset_fldr):
    return {
        'mmlu_pro_biology': os.path.abspath(os.path.join(dataset_fldr, 'mmlu_pro_biology.json')),
        'mmlu_pro_chemistry': os.path.abspath(os.path.join(dataset_fldr, 'mmlu_pro_chemistry.json')),
        'mmlu_pro_computer_science': os.path.abspath(os.path.join(dataset_fldr, 'mmlu_pro_computer_science.json')),
        'mmlu_pro_math': os.path.abspath(os.path.join(dataset_fldr, 'mmlu_pro_math.json')),
        'mmlu_pro_physics': os.path.abspath(os.path.join(dataset_fldr, 'mmlu_pro_physics.json')),
        'mmlu_pro_psychology': os.path.abspath(os.path.join(dataset_fldr, 'mmlu_pro_psychology.json')),
    }

def get_task(name: str, dataset_fldr: str, num_samples:int = 200):
    task_map = {
        'mmlu_pro_biology': mmlu_pro_biology,
        'mmlu_pro_chemistry': mmlu_pro_chemistry,
        'mmlu_pro_computer_science': mmlu_pro_computer_science,
        'mmlu_pro_math': mmlu_pro_math,
        'mmlu_pro_physics': mmlu_pro_physics,
        'mmlu_pro_psychology': mmlu_pro_psychology,
    }
    try:
        return task_map[name](dataset_fldr, num_samples)
    except KeyError:
        raise ValueError(f"Task {name} not found")



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Evaluates a MMLU style dataset using Inspect framework.')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--base_dir', type=str, required=True, help='where the dataset json files are stored')
    parser.add_argument('--num_samples', type=int, default=200)

    args = parser.parse_args()
    



    config = GenerateConfig(max_connections=args.batch_size, max_tokens=8192)
    models = list()


    models_dict = dict()

    models_dict['openai/gpt-oss-120b'] = get_model(model="v_llm/openai/gpt-oss-120b", base_url="https://<ip_address>:8443/v1", config=config, api_key=os.getenv("VLLM_API_KEY"))

    oai_config = GenerateConfig(max_connections=8, max_tokens=8192)
    models_dict['openai/gpt-4.1-mini'] = get_model(model="openai/gpt-4.1-mini", base_url="https://api.openai.com/v1", config=oai_config)
    
    



    
    

    available_models = list(models_dict.keys())

    
    dataset_fldr = args.base_dir

    available_task_names_dict = get_task_dir_dict(dataset_fldr)

    to_remove = list()
    for k in available_task_names_dict.keys():
        if not os.path.exists(available_task_names_dict[k]):
            to_remove.append(k)
            print("missing task: ", k, " at ", available_task_names_dict[k])
    for k in to_remove:
        available_task_names_dict.pop(k)
    available_task_names = list(available_task_names_dict.keys())



    log_dir = os.path.join(dataset_fldr, f"logs")
    print("discovering completed logs...")
    completed_logs, completed_fns = utils.get_completed_logs(log_dir)

    unused_logs = copy.deepcopy(completed_logs)

    for task_name in available_task_names:
        print("--------------------------------")
        print(f"Processing folder {dataset_fldr} task {task_name}")
        work_models = list(set(available_models))

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
        print("starting in 3...")
        time.sleep(1)
        print("            2...")
        time.sleep(1)
        print("            1...")
        time.sleep(1)

        models_list = [models_dict[model] for model in work_models]
        json_fp = available_task_names_dict[task_name]
        work_tasks = [get_task(task_name, json_fp, args.num_samples)]
        eval(work_tasks, model=models_list, display='full', log_format='json', no_log_images=True, no_log_samples=True, log_dir=log_dir)