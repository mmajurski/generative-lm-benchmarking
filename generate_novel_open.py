import numpy as np
import os
import copy
import json
import time
import random

import answer_parser
from model_interface import SglModel
from generate_document_topics import extract_topics_per_context
import prompts









def build_questions(dataset: list[dict], remote:str, model_name:str, async_flag:bool=False):

    
    # Create a list to store model responses
    model_responses = list()
    # Copy over question (into orig_question), id, and context
    for item in dataset:
        response_item = {}
        if 'context' in item:
            response_item['context'] = item['context']
        else:
            raise ValueError("No context found in {item}")
        if 'topic' in item:
            response_item['topic'] = item['topic']
        else:
            raise ValueError("No topic found in {item}")
        response_item['failed'] = False
        model_responses.append(response_item)
    
    
    # build the prompts
    model_prompts = [prompts.QUESTION_GEN_OPEN_PROMPT.format(context=d['context'], topic=d['topic']) for d in model_responses]

    model = SglModel(remote=remote, model=model_name, sync_flag=(not async_flag))
    print(f"Generating questions against {model.url}")
    results, total_time = model.generate(model_prompts, reasoning_effort="high")
    print(f"in total took: {total_time} seconds")
    print(f"per question took: {total_time / len(results)} seconds for {len(results)} questions")

    total_input_tokens = sum([res['input_tokens'] for res in results])
    total_output_tokens = sum([res['output_tokens'] for res in results])
    total_tokens = sum([res['total_tokens'] for res in results])
    print(f"total input tokens: {total_input_tokens}")
    print(f"total output tokens: {total_output_tokens}")
    print(f"total tokens: {total_tokens}")
    print(f"total toks/sec: {total_tokens / total_time}")
    print(f"total output toks/sec: {total_output_tokens / total_time}")

    failed_responses = list()
    for i in range(len(results)):
        res = results[i]
        if res['error'] is not None:
            model_responses[i]['response'] = None
            model_responses[i]['failed'] = True
            model_responses[i]['error'] = res['error']
            failed_responses.append(model_responses[i])
        else:
            model_responses[i]['response'] = res['content']
            parsed = answer_parser.parse_generated_open(res['content'])
            if parsed is None:
                model_responses[i]['failed'] = True
                failed_responses.append(model_responses[i])
            else:
                model_responses[i]['question'] = parsed['question']
                model_responses[i]['answer'] = parsed['correct_answer']
                model_responses[i]['explanation'] = parsed['explanation']

    # remove failed responses
    model_responses = [d for d in model_responses if not d['failed']]
    # Assert that every entry in model_responses has 'question', 'context', and 'answer'
    for entry in model_responses:
        missing_keys = []
        for key in ['question', 'context', 'answer']:
            if key not in entry:
                missing_keys.append(key)
        if missing_keys:
            raise AssertionError(f"Missing keys {missing_keys} in entry: {entry}")
    # Remove the 'failed' key from each response
    for response in model_responses:
        if 'failed' in response:
            del response['failed']

    return model_responses, failed_responses
    



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Converts a jsonl dataset into MMLU format to be used as an open ended evaluation.')
    parser.add_argument('--dataset', type=str, default='squadv2.jsonl', help='dataset to generate, options: squadv2, ucinlp_drop')
    parser.add_argument('--src_dataset_dir', type=str, default='./data-subset-500', help='source dataset directory')
    parser.add_argument('--out_dataset_dir', type=str, required=True, help='output dataset directory')
    parser.add_argument('--remote', type=str, default="sierra")
    parser.add_argument('--model', type=str, default="meta-llama/Llama-3.3-70B-Instruct")
    parser.add_argument('--disable_async', action='store_false', dest='async_flag', help='Set to disable async processing (async is enabled by default)')
    parser.add_argument('--sample_count', type=int, default=-1, help='number of samples to generate, set to <=0 for all')
    parser.set_defaults(async_flag=True)

    args = parser.parse_args()
    print("Generating novel Open Ended MCQs")
    print(args)

    base_path = os.path.splitext(args.dataset)[0]  # Remove extension
    fn_basename = os.path.basename(base_path) + "_novel"
    fn_basename = fn_basename.replace('computer-systems-security-planning-for-success','sec_qa')
    os.makedirs(args.out_dataset_dir, exist_ok=True)
    output_fn = os.path.join(args.out_dataset_dir, f'{fn_basename}.json')
    if os.path.exists(output_fn):
        print(f"Output file {output_fn} already exists, skipping")
        exit()


    start_time = time.time()

    args.dataset = os.path.join(args.src_dataset_dir, args.dataset)

    with open(args.dataset, 'r') as f:
        dataset = json.load(f)
    
    sc = int(args.sample_count)
    if args.sample_count > 0 and sc < len(dataset):
        print(f"Sampling {sc} contexts from {len(dataset)}")
        # assume that each context has at least 3 questions
        dataset = random.sample(dataset, sc)

    # Keep only the "context" element for every item in the dataset
    for item in dataset:
        keys_to_remove = [key for key in item.keys() if key != 'context']
        for key in keys_to_remove:
            del item[key]

    # verify that each element in the dataset has the following keys: question, answer, context
    for item in dataset:
        if 'context' not in item:
            raise ValueError('each element in the dataset must have the following keys: context')

    print("Dataset has %d contexts" % len(dataset))
    contexts = [d['context'] for d in dataset]
    
    print("Evaluating the topics for each context")
    topic_remote = args.remote 
    topic_model = args.model
    topics_list, topic_extraction_responses = extract_topics_per_context(contexts, topic_remote, model=topic_model, async_flag=args.async_flag)

    

    
    new_dataset = []
    topic_counts = []
    num_empty_topics = 0
    for i in range(len(dataset)):
        topics = topics_list[i]
        if len(topics) == 0:
            num_empty_topics += 1
            continue
        topic_counts.append(len(topics))
        for topic in topics:
            dat = copy.deepcopy(dataset[i])
            dat['topic'] = topic
            new_dataset.append(dat)
    print(f"  {num_empty_topics} contexts had no topics extracted")
    print(f"  After topic expansion dataset has {len(new_dataset)} entries")
    random.shuffle(new_dataset)
    dataset = new_dataset
    print(f"  average number of topics per context: {np.mean(topic_counts)}")
    print("Dataset has %d contexts after expansion per unique topic" % len(dataset))
    if args.sample_count > 0 and len(dataset) > args.sample_count:
        print(f"Dataset has {len(dataset)} contexts, keeping a random {args.sample_count}")
        # print(f"This enables reasonable runtime for the model evaluation")
        dataset = random.sample(dataset, args.sample_count)


    print("Generating questions based on the topics")
    dataset, failed_responses = build_questions(dataset, args.remote, args.model, async_flag=args.async_flag)
    print(f"Generated {len(dataset)} initial questions based on the topics")
    print(f"Failed to generate {len(failed_responses)} questions")

    elapsed_time = time.time() - start_time


    # build the prompts
    print(f"Evaluating the completeness of the {len(dataset)} questions")
    model_prompts = [prompts.QUESTION_VALIDITY_PROMPT.format(context=d['context'], question=d['question'], answer=d['answer']) for d in dataset]


    model = SglModel(remote=args.remote, model=args.model)
    results, total_time = model.generate(model_prompts, reasoning_effort="medium")
    print(f"in total took: {total_time} seconds")
    print(f"per question took: {total_time / len(results)} seconds for {len(results)} questions")


    
    for i in range(len(results)):
        res = results[i]
        if res['error'] is not None:
            raise Exception(f"Error: {res['error']}")
        else:
            parsed = answer_parser.parse_question_completeness_numbers(res['content'], valid_options=[1,2,3,4,5,6,7,8,9,10])
            if parsed is None:
                raise Exception(f"Failed to parse response: {res['content']}")
            else:

                dataset[i]['question_completeness_score'] = parsed['question_completeness_score']
                dataset[i]['answer_completeness_score'] = parsed['answer_completeness_score']
                dataset[i]['completeness_response'] = res['content']

    fail_dataset =  [d for d in dataset if d['question_completeness_score'] < 5 and d['answer_completeness_score'] < 5]
    dataset =  [d for d in dataset if d['question_completeness_score'] >= 5 and d['answer_completeness_score'] >= 5]
    print(f"Filtered out {len(fail_dataset)} items with completeness score less than 5")
    if len(fail_dataset) > 0:
        with open(output_fn.replace('.json', '_completeness_fail.json'), 'w') as f:
            json.dump(fail_dataset, f, indent=2)
    





    # build the prompts
    print(f"Evaluating the meta properties of the {len(dataset)} questions")
    model_prompts = [prompts.META_PROPERTIES_PROMPT.format(context=d['context'], question=d['question'], answer=d['answer']) for d in dataset]

    model = SglModel(remote=args.remote, model=args.model)
    results, total_time = model.generate(model_prompts, reasoning_effort="medium")
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
            else:
                dataset[i]['question_clarity_score'] = parsed['clarity_score']
                dataset[i]['question_difficulty_score'] = parsed['difficulty_score']
                dataset[i]['question_groundedness_score'] = parsed['groundedness_score']

    # Remove dataset elements with a difficulty score less than 5
    # Keep only those elements whose question_difficulty_score is >= 5
    original_length = len(dataset)
    dataset = [d for d in dataset if d['question_difficulty_score'] >= 5]
    print(f"Filtered out {original_length - len(dataset)} items with difficulty score less than 5")
    original_length = len(dataset)
    invalid_dataset = [d for d in dataset if d['question_groundedness_score'] < 5]
    if len(invalid_dataset) > 0:
        with open(output_fn.replace('.json', '_grounding_fail.json'), 'w') as f:
            json.dump(invalid_dataset, f, indent=2)
    dataset = [d for d in dataset if d['question_groundedness_score'] >= 5]
    print(f"Filtered out {original_length - len(dataset)} items with groundedness score less than 5")


    scores = [d['question_clarity_score'] for d in dataset]
    print(f"Average question clarity score: {np.mean(scores)}")

    scores = [d['question_difficulty_score'] for d in dataset]
    print(f"Average question difficulty score: {np.mean(scores)}")

    scores = [d['question_groundedness_score'] for d in dataset]
    print(f"Average question groundedness score: {np.mean(scores)}")

    print(f"Saving {len(dataset)} questions to {output_fn}")
    with open(output_fn, 'w') as f:
        json.dump(dataset, f, indent=2)

