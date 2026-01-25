import numpy as np
import json
import os
import sys

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import answer_parser
from model_interface import SglModel
import prompts


def compute_meta_scores(dataset_fp, remote, model, force=False):
    with open(dataset_fp, 'r') as f:
        dataset = json.load(f)

    any_missing = False
    for d in dataset:
        if 'question_clarity_score' not in d or d['question_clarity_score'] is None:
            any_missing = True
        if 'question_difficulty_score' not in d or d['question_difficulty_score'] is None:
            any_missing = True
        if 'question_groundedness_score' not in d or d['question_groundedness_score'] is None:
            any_missing = True
    if not any_missing and not force:
        return
    
    print(f"Computing meta properties for {dataset_fp}")
    print("Dataset has %d contexts" % len(dataset))

    # build the prompts
    model_prompts = [prompts.META_PROPERTIES_PROMPT.format(context=d.get('context', ''), question=d['question'], answer=d['answer']) for d in dataset]

    model = SglModel(remote=remote, model=model)
    results, total_time = model.generate(model_prompts)
    print(f"in total took: {total_time} seconds")
    print(f"per question took: {total_time / len(results)} seconds for {len(results)} questions")

    total_input_tokens = sum([res['input_tokens'] for res in results])
    total_output_tokens = sum([res['output_tokens'] for res in results])
    total_tokens = sum([res['total_tokens'] for res in results])
    print(f"total input tokens: {total_input_tokens}")
    print(f"total output tokens: {total_output_tokens}")
    print(f"total tokens: {total_tokens}")

    for i in range(len(results)):
        res = results[i]
        if res['error'] is not None:
            raise Exception(f"Error: {res['error']}")
        else:
            parsed = answer_parser.parse_meta_properties_numbers(res['content'], valid_options=[1,2,3,4,5,6,7,8,9,10])
            if parsed is None:
                raise Exception(f"Failed to parse response: {res['content']}")
                dataset[i]['question_clarity_score'] = None  
                dataset[i]['question_difficulty_score'] = None
                dataset[i]['question_relevance_score'] = None
                dataset[i]['question_groundedness_score'] = None
            else:
                dataset[i]['question_clarity_score'] = parsed['clarity_score']
                dataset[i]['question_difficulty_score'] = parsed['difficulty_score']
                dataset[i]['question_groundedness_score'] = parsed['groundedness_score']


    scores = [d['question_clarity_score'] for d in dataset]
    print(f"Average question clarity score: {np.mean(scores)}")

    scores = [d['question_difficulty_score'] for d in dataset]
    print(f"Average question difficulty score: {np.mean(scores)}")

    scores = [d['question_groundedness_score'] for d in dataset]
    print(f"Average question groundedness score: {np.mean(scores)}")

    print(f"Saving {len(dataset)} questions to {dataset_fp}")
    with open(dataset_fp, 'w') as f:
        json.dump(dataset, f, indent=2)



def evaluate_dataset_relevance_features(ifp, remote='opchat', model='Llama-4-Maverick-17B-128E-Instruct-FP8'):

    fns = []
    for fn in os.listdir(ifp):
        if fn.endswith('.json') and ('novel' in fn or 'reformat' in fn):
            fns.append(fn)
    fns.sort()
    if len(fns) == 0:
        print(f"evaluate_dataset_relevance_features: No datasets found in {ifp}")
        return
    print(f"evaluate_dataset_relevance_features: Found {len(fns)} datasets")
    print(f"evaluate_dataset_relevance_features: Datasets: {fns}")



    force_flag = False
    print(f"evaluate_dataset_relevance_features: Computing squishy statistics for {len(fns)} datasets")

    for i, fn in enumerate(fns):
        dataset_fp = f"{ifp}/{fn}"
        print(f"Dataset: {dataset_fp}")
        print(f"Progress: {i+1}/{len(fns)}")
        compute_meta_scores(dataset_fp, remote, model, force=force_flag)
    



    avg_difficulty_score = {}
    avg_clarity_score = {}
    avg_groundedness_score = {}
    for fn in fns:
        dataset_fp = f"{ifp}/{fn}"
        with open(dataset_fp, 'r') as f:
            data = json.load(f)

        invalid_idx = list()
        for i in range(len(data)):
            d = data[i]
            if d.get('question_groundedness_score') is None or not (1 <= d['question_groundedness_score'] <= 10):
                invalid_idx.append(i)
            if d.get('question_difficulty_score') is None or not (1 <= d['question_difficulty_score'] <= 10):
                invalid_idx.append(i)
            if d.get('question_clarity_score') is None or not (1 <= d['question_clarity_score'] <= 10):
                invalid_idx.append(i)
        
        # Print invalid data entries
        if invalid_idx:
            print(f"Found {len(invalid_idx)} invalid entries in dataset {fn}:")
            for idx in invalid_idx:
                print(f"  Entry {idx}:")
                print(f"    Question: {data[idx]['question']}")
                print(f"    Question Groundedness: {data[idx].get('question_groundedness_score', 'missing')}")
                print(f"    Question Difficulty: {data[idx].get('question_difficulty_score', 'missing')}")
                print(f"    Question Clarity: {data[idx].get('question_clarity_score', 'missing')}")
                print()

            raise Exception(f"Found {len(invalid_idx)} invalid entries in dataset {fn}")
        
        
        avg_difficulty_score[fn] = np.mean([d['question_difficulty_score'] for d in data])
        avg_clarity_score[fn] = np.mean([d['question_clarity_score'] for d in data])
        avg_groundedness_score[fn] = np.mean([d['question_groundedness_score'] for d in data])

        print(f"Dataset: {os.path.basename(fn)}")
        print(f"  Average question groundedness score: {avg_groundedness_score[fn]:.4f}")
        print(f"  Average question difficulty score: {avg_difficulty_score[fn]:.4f}")
        print(f"  Average question clarity score: {avg_clarity_score[fn]:.4f}")
        print()



if __name__ == '__main__':
    ifp = './data-subset-1k-p3-v1'
    evaluate_dataset_relevance_features(ifp)