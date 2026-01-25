
import os
import time
import copy
from inspect_ai import eval
from inspect_ai.model import GenerateConfig
from inspect_ai.model import get_model

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.scorer import choice
from inspect_ai.solver import multiple_choice
import vllm_inspect_provider
import utils

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def squad_task(lcl_fp):
    return Task(
        dataset=hf_dataset(
            path=lcl_fp,
            split="train",
            sample_fields=record_to_sample,
            revision="main",  # use this to prevent cache of the dataset
            shuffle_choices=True,
        ),
        solver=multiple_choice(),
        scorer=choice(),
    )



@task
def squadv2_reformat_mcq(dataset_fldr):
    import os
    return squad_task(os.path.abspath(os.path.join(dataset_fldr, 'squadv2_reformat')))

@task
def squadv2_novel_mcq(dataset_fldr):
    import os
    return squad_task(os.path.abspath(os.path.join(dataset_fldr, 'squadv2_novel')))

@task
def mrqa_HotpotQA_reformat_mcq(dataset_fldr):
    import os
    return squad_task(os.path.abspath(os.path.join(dataset_fldr, 'mrqa_HotpotQA_reformat')))

@task
def mrqa_HotpotQA_novel_mcq(dataset_fldr):
    import os
    return squad_task(os.path.abspath(os.path.join(dataset_fldr, 'mrqa_HotpotQA_novel')))

@task
def mrqa_NaturalQuestionsShort_reformat_mcq(dataset_fldr):
    import os
    return squad_task(os.path.abspath(os.path.join(dataset_fldr, 'mrqa_NaturalQuestionsShort_reformat')))

@task
def mrqa_NaturalQuestionsShort_novel_mcq(dataset_fldr):
    import os
    return squad_task(os.path.abspath(os.path.join(dataset_fldr, 'mrqa_NaturalQuestionsShort_novel')))

@task
def mrqa_TriviaQA_web_reformat_mcq(dataset_fldr):
    import os
    return squad_task(os.path.abspath(os.path.join(dataset_fldr, 'mrqa_TriviaQA-web_reformat')))

@task
def mrqa_TriviaQA_web_novel_mcq(dataset_fldr):
    import os
    return squad_task(os.path.abspath(os.path.join(dataset_fldr, 'mrqa_TriviaQA-web_novel')))

@task
def ucinlp_drop_reformat_mcq(dataset_fldr):
    import os
    return squad_task(os.path.abspath(os.path.join(dataset_fldr, 'ucinlp_drop_reformat')))

@task
def ucinlp_drop_novel_mcq(dataset_fldr):
    import os
    return squad_task(os.path.abspath(os.path.join(dataset_fldr, 'ucinlp_drop_novel')))

@task
def annurev_control_novel_mcq(dataset_fldr):
    import os
    return squad_task(os.path.abspath(os.path.join(dataset_fldr, 'annurev-control-071020-104336_novel')))

@task
def secqa_reformat_mcq(dataset_fldr):
    import os
    return squad_task(os.path.abspath(os.path.join(dataset_fldr, 'sec_qa_reformat')))

@task
def secqa_novel_mcq(dataset_fldr):
    import os
    return squad_task(os.path.abspath(os.path.join(dataset_fldr, 'sec_qa_novel')))

@task
def pubmed_qa_reformat_mcq(dataset_fldr):
    import os
    return squad_task(os.path.abspath(os.path.join(dataset_fldr, 'pubmed_qa_reformat')))

@task
def pubmed_qa_novel_mcq(dataset_fldr):
    import os
    return squad_task(os.path.abspath(os.path.join(dataset_fldr, 'pubmed_qa_novel')))

@task
def arXiv_2502_17521v1_novel_mcq(dataset_fldr):
    import os
    return squad_task(os.path.abspath(os.path.join(dataset_fldr, 'arXiv_2502_17521v1_novel')))


def get_task_dir_dict(dataset_fldr):
    return {
        'squadv2_reformat_mcq': os.path.abspath(os.path.join(dataset_fldr, 'squadv2_reformat')),
        'squadv2_novel_mcq': os.path.abspath(os.path.join(dataset_fldr, 'squadv2_novel')),
        'mrqa_HotpotQA_reformat_mcq': os.path.abspath(os.path.join(dataset_fldr, 'mrqa_HotpotQA_reformat')),
        'mrqa_HotpotQA_novel_mcq': os.path.abspath(os.path.join(dataset_fldr, 'mrqa_HotpotQA_novel')),
        'mrqa_NaturalQuestionsShort_reformat_mcq': os.path.abspath(os.path.join(dataset_fldr, 'mrqa_NaturalQuestionsShort_reformat')),
        'mrqa_NaturalQuestionsShort_novel_mcq': os.path.abspath(os.path.join(dataset_fldr, 'mrqa_NaturalQuestionsShort_novel')),
        'mrqa_TriviaQA_web_reformat_mcq': os.path.abspath(os.path.join(dataset_fldr, 'mrqa_TriviaQA-web_reformat')),
        'mrqa_TriviaQA_web_novel_mcq': os.path.abspath(os.path.join(dataset_fldr, 'mrqa_TriviaQA-web_novel')),
        'ucinlp_drop_reformat_mcq': os.path.abspath(os.path.join(dataset_fldr, 'ucinlp_drop_reformat')),
        'ucinlp_drop_novel_mcq': os.path.abspath(os.path.join(dataset_fldr, 'ucinlp_drop_novel')),
        'annurev_control_novel_mcq': os.path.abspath(os.path.join(dataset_fldr, 'annurev-control-071020-104336_novel')),
        'secqa_reformat_mcq': os.path.abspath(os.path.join(dataset_fldr, 'sec_qa_reformat')),
        'secqa_novel_mcq': os.path.abspath(os.path.join(dataset_fldr, 'sec_qa_novel')),
        'pubmed_qa_reformat_mcq': os.path.abspath(os.path.join(dataset_fldr, 'pubmed_qa_reformat')),
        'pubmed_qa_novel_mcq': os.path.abspath(os.path.join(dataset_fldr, 'pubmed_qa_novel')),
        'arXiv_2502_17521v1_novel_mcq': os.path.abspath(os.path.join(dataset_fldr, 'arXiv_2502_17521v1_novel')),
    }



def get_task(name: str, dataset_fldr: str):
    if name == 'squadv2_reformat_mcq':
        return squadv2_reformat_mcq(dataset_fldr)
    elif name == 'squadv2_novel_mcq':
        return squadv2_novel_mcq(dataset_fldr)
    elif name == 'mrqa_HotpotQA_reformat_mcq':
        return mrqa_HotpotQA_reformat_mcq(dataset_fldr)
    elif name == 'mrqa_HotpotQA_novel_mcq':
        return mrqa_HotpotQA_novel_mcq(dataset_fldr)
    elif name == 'mrqa_NaturalQuestionsShort_reformat_mcq':
        return mrqa_NaturalQuestionsShort_reformat_mcq(dataset_fldr)
    elif name == 'mrqa_NaturalQuestionsShort_novel_mcq':
        return mrqa_NaturalQuestionsShort_novel_mcq(dataset_fldr)
    elif name == 'mrqa_TriviaQA_web_reformat_mcq':
        return mrqa_TriviaQA_web_reformat_mcq(dataset_fldr)
    elif name == 'mrqa_TriviaQA_web_novel_mcq':
        return mrqa_TriviaQA_web_novel_mcq(dataset_fldr)
    elif name == 'ucinlp_drop_reformat_mcq':
        return ucinlp_drop_reformat_mcq(dataset_fldr)
    elif name == 'ucinlp_drop_novel_mcq':
        return ucinlp_drop_novel_mcq(dataset_fldr)
    elif name == 'annurev_control_novel_mcq':
        return annurev_control_novel_mcq(dataset_fldr)
    elif name == 'secqa_reformat_mcq':
        return secqa_reformat_mcq(dataset_fldr)
    elif name == 'secqa_novel_mcq':
        return secqa_novel_mcq(dataset_fldr)
    elif name == 'pubmed_qa_reformat_mcq':
        return pubmed_qa_reformat_mcq(dataset_fldr)
    elif name == 'pubmed_qa_novel_mcq':
        return pubmed_qa_novel_mcq(dataset_fldr)
    elif name == 'arXiv_2502_17521v1_novel_mcq':
        return arXiv_2502_17521v1_novel_mcq(dataset_fldr)
    else:
        raise ValueError(f"Task {name} not found")

def record_to_sample(record):
    # read the labels and text
    choices = record["choices"]
    target = record["answer"]

    if record['choices'] is None or record['answer'] is None or record['question'] is None:
        print(record)
        raise ValueError(f"choices is None for record {record}")

    try:
        # return sample
        return Sample(
            input=record["question"], choices=list(choices.values()), target=target
        )
    except Exception as e:
        print(record)
        raise e



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Evaluates a MMLU style dataset using Inspect framework.')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--run_reformat', action='store_true', help='Run reformat tasks instead of novel tasks')

    args = parser.parse_args()
    


    


    config = GenerateConfig(max_connections=args.batch_size, max_tokens=1024)
    models_dict = dict()

    models_dict['openai/gpt-oss-120b'] = get_model(model="v_llm/openai/gpt-oss-120b", base_url="https://<ip>:8443/v1", config=config, api_key=os.getenv("VLLM_API_KEY"))

    oai_config = GenerateConfig(max_connections=8, max_tokens=8192)
    models_dict['openai/gpt-4.1-mini'] = get_model(model="openai/gpt-4.1-mini", base_url="https://api.openai.com/v1", config=oai_config)
    
    
    disp_type = 'full'

    base_dir = './data-qa'
    datasets = [fn for fn in os.listdir(base_dir) if fn.startswith('mc-')]
    
    datasets.sort()
    if args.reformat:
        datasets = [fn for fn in datasets if 'orig' in fn]
    else:
        datasets = [fn for fn in datasets if 'reformat' not in fn]


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
