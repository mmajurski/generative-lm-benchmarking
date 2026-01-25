
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



GRADER_MODEL = 'v_llm/openai/gpt-oss-120b'
GRADER_MODEL_BASE_URL = 'https://pn131285.nist.gov:8443/v1'
GRADER_MODEL_API_KEY=os.getenv("VLLM_API_KEY")



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




def base_task(lcl_fp):

    config = GenerateConfig(max_connections=128)
    model = get_model(model=GRADER_MODEL, base_url=GRADER_MODEL_BASE_URL, config=config, api_key=GRADER_MODEL_API_KEY)

    
    with open(lcl_fp, 'r') as f:
        ds = json.load(f)
    
    random.shuffle(ds)
    samples = list()
    for row in ds:
        samples.append(Sample(input=row['question'], target=str(row['answer'])))
    
    return Task(
        dataset = samples,
        solver=open_ended_qa(),
        scorer=model_graded_qa(model=model, template=DEFAULT_MODEL_GRADED_FACT_TEMPLATE),
    )





@task
def sec_qa(dataset_fldr):
    return base_task(dataset_fldr)

@task
def flashrag_2wikimultihopqa(dataset_fldr):
    return base_task(dataset_fldr)
    
@task
def flashrag_boolq(dataset_fldr):
    return base_task(dataset_fldr)

@task
def flashrag_fermi(dataset_fldr):
    return base_task(dataset_fldr)

@task
def flashrag_hotpotqa(dataset_fldr):
    return base_task(dataset_fldr)

@task
def flashrag_msmarcoqa(dataset_fldr):
    return base_task(dataset_fldr)

@task
def flashrag_musique(dataset_fldr):
    return base_task(dataset_fldr)

@task
def mrqa_HotpotQA(dataset_fldr):
    return base_task(dataset_fldr)

@task
def mrqa_NaturalQuestionsShort(dataset_fldr):
    return base_task(dataset_fldr)

@task
def mrqa_TriviaQA_web(dataset_fldr):
    return base_task(dataset_fldr)

@task
def natural_questions(dataset_fldr):
    return base_task(dataset_fldr)

@task
def pubmed_qa(dataset_fldr):
    return base_task(dataset_fldr)

@task
def squadv2(dataset_fldr):
    return base_task(dataset_fldr)

@task
def triva_qa(dataset_fldr):
    return base_task(dataset_fldr)

@task
def ai_plan(dataset_fldr):
    return base_task(dataset_fldr)

@task
def arXiv_2502_17521v1(dataset_fldr):
    return base_task(dataset_fldr)

@task
def annurev_control_071020_104336(dataset_fldr):
    return base_task(dataset_fldr)


def get_task_dir_dict(dataset_fldr):
    return {
        'flashrag_2wikimultihopqa': os.path.abspath(os.path.join(dataset_fldr, 'flashrag_2wikimultihopqa.json')),
        'flashrag_boolq': os.path.abspath(os.path.join(dataset_fldr, 'flashrag_boolq.json')),
        'flashrag_fermi': os.path.abspath(os.path.join(dataset_fldr, 'flashrag_fermi.json')),
        'flashrag_hotpotqa': os.path.abspath(os.path.join(dataset_fldr, 'flashrag_hotpotqa.json')),
        'flashrag_msmarcoqa': os.path.abspath(os.path.join(dataset_fldr, 'flashrag_msmarcoqa.json')),
        'flashrag_musique': os.path.abspath(os.path.join(dataset_fldr, 'flashrag_musique.json')),
        'mrqa_HotpotQA': os.path.abspath(os.path.join(dataset_fldr, 'mrqa_HotpotQA.json')),
        'mrqa_NaturalQuestionsShort': os.path.abspath(os.path.join(dataset_fldr, 'mrqa_NaturalQuestionsShort.json')),
        'mrqa_TriviaQA_web': os.path.abspath(os.path.join(dataset_fldr, 'mrqa_TriviaQA-web.json')),
        'pubmed_qa': os.path.abspath(os.path.join(dataset_fldr, 'pubmed_qa.json')),
        'squadv2': os.path.abspath(os.path.join(dataset_fldr, 'squadv2.json')),
        'triva_qa': os.path.abspath(os.path.join(dataset_fldr, 'triva_qa.json')),
        'sec_qa': os.path.abspath(os.path.join(dataset_fldr, 'sec_qa.json')),
        'ai-plan': os.path.abspath(os.path.join(dataset_fldr, 'ai-plan.json')),
        'arXiv_2502_17521v1': os.path.abspath(os.path.join(dataset_fldr, 'arXiv_2502_17521v1.json')),
        'annurev-control-071020-104336': os.path.abspath(os.path.join(dataset_fldr, 'annurev-control-071020-104336.json')),
    }

def get_task(name: str, dataset_fldr: str):
    task_map = {
        'flashrag_2wikimultihopqa': flashrag_2wikimultihopqa,
        'flashrag_boolq': flashrag_boolq,
        'flashrag_fermi': flashrag_fermi,
        'flashrag_hotpotqa': flashrag_hotpotqa,
        'flashrag_msmarcoqa': flashrag_msmarcoqa,
        'flashrag_musique': flashrag_musique,
        'mrqa_HotpotQA': mrqa_HotpotQA,
        'mrqa_NaturalQuestionsShort': mrqa_NaturalQuestionsShort,
        'mrqa_TriviaQA_web': mrqa_TriviaQA_web,
        'pubmed_qa': pubmed_qa,
        'squadv2': squadv2,
        'triva_qa': triva_qa,
        'sec_qa': sec_qa,
        'ai-plan': ai_plan,
        'arXiv_2502_17521v1': arXiv_2502_17521v1,
        'annurev-control-071020-104336': annurev_control_071020_104336,
    }
    try:
        return task_map[name](dataset_fldr)
    except KeyError:
        raise ValueError(f"Task {name} not found")


def record_to_sample(record):
    q_key = 'question'
    a_key = 'answer'
    
    # q_key = 'orig_question'
    # a_key = 'orig_answer'
    if not record[q_key] or not record[a_key]:
        print(record)
        raise ValueError(f"invalid record {record}")
    

    try:
        # return sample
        return Sample(
            input=record[q_key], target=record[a_key]
        )
    except Exception as e:
        print(record)
        raise e



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Evaluates a MMLU style dataset using Inspect framework.')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--reformat', action='store_true', help='Run reformat tasks instead of novel tasks')

    args = parser.parse_args()
    args.reformat = False
    



    config = GenerateConfig(max_connections=args.batch_size, max_tokens=8192)
    models = list()


    models_dict = dict()

    models_dict['openai/gpt-oss-120b'] = get_model(model="v_llm/openai/gpt-oss-120b", base_url="https://<ip_address>:8443/v1", config=config, api_key=os.getenv("VLLM_API_KEY"))

    oai_config = GenerateConfig(max_connections=8, max_tokens=8192)
    models_dict['openai/gpt-4.1-mini'] = get_model(model="openai/gpt-4.1-mini", base_url="https://api.openai.com/v1", config=oai_config)
    
    



    

    base_dir = './data-qa'
    datasets = [fn for fn in os.listdir(base_dir) if fn.startswith('oe-')]
    
    datasets.sort()
    if args.reformat:
        datasets = [fn for fn in datasets if 'orig' in fn]
    else:
        datasets = [fn for fn in datasets if 'reformat' not in fn]
    
    disp_type = 'full'

    available_models = list(models_dict.keys())


    for ds in datasets:
        print("--------------------------------")
        print(f"Processing folder {ds}")

        
        dataset_fldr = f"{base_dir}/{ds}"

        available_task_names_dict = get_task_dir_dict(dataset_fldr)

        if args.reformat:
            for key, val in available_task_names_dict.items():
                available_task_names_dict[key] = val.replace('.json','_orig.json')
        else:
            for key, val in available_task_names_dict.items():
                available_task_names_dict[key] = val.replace('.json','_novel.json')


        to_remove = list()
        for k in available_task_names_dict.keys():
            if not os.path.exists(available_task_names_dict[k]):
                to_remove.append(k)
                print("missing task: ", k, " at ", available_task_names_dict[k])
        for k in to_remove:
            available_task_names_dict.pop(k)
        available_task_names = list(available_task_names_dict.keys())



        log_dir = os.path.join(base_dir, f"logs-{ds}")
        print("discovering completed logs...")
        completed_logs, completed_fns = utils.get_completed_logs(log_dir)
        for l_idx, log in enumerate(completed_logs):
            d_fp = log['task_args']['dataset_fldr']
            if ds not in d_fp:
                print(f"Log in Wrong Place:  {completed_fns[l_idx]}")


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
            work_tasks = [get_task(task_name, json_fp)]
            eval(work_tasks, model=models_list, display=disp_type, log_format='json', no_log_images=True, no_log_samples=True, log_dir=log_dir)