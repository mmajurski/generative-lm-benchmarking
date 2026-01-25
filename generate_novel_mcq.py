import numpy as np
import os
import copy
import json
import time
import random

import answer_parser
from similarity_filter import get_duplicate_contexts_embedding_cosine
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
    model_prompts = [prompts.QUESTION_GEN_PROMPT.format(context=d['context'], topic=d['topic']) for d in model_responses]

    model = SglModel(remote=remote, model=model_name, sync_flag=(not async_flag))
    print(f"Generating questions against {model.url}")
    results, total_time = model.generate(model_prompts)
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
            parsed = answer_parser.parse_generated_mcq(res['content'])
            if parsed is None:
                model_responses[i]['failed'] = True
                failed_responses.append(model_responses[i])
            else:
                model_responses[i]['question'] = parsed['question']
                model_responses[i]['choices'] = parsed['options']
                model_responses[i]['answer'] = parsed['correct_answer']
                model_responses[i]['explanation'] = parsed['explanation']

    # remove failed responses
    model_responses = [d for d in model_responses if not d['failed']]
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
    parser.add_argument('--out_dataset_dir', type=str, default='./data-subset-100', help='output dataset directory')
    parser.add_argument('--remote', type=str, default="sierra")
    parser.add_argument('--model', type=str, default="meta-llama/Llama-3.3-70B-Instruct")
    parser.add_argument('--disable_async', action='store_false', dest='async_flag', help='Set to disable async processing (async is enabled by default)')
    parser.add_argument('--sample_count', type=int, default=100, help='number of total questions to generate, set to <=0 for all')
    parser.set_defaults(async_flag=True)

    args = parser.parse_args()
    print("Generating novel MCQs")



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

    random.shuffle(dataset)
    
    sc = int(args.sample_count)
    if args.sample_count > 0 and sc < len(dataset):
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


    print("Extracting topics for each context. Will construct a question for each topic.")
    contexts = [d['context'] for d in dataset]
    topic_remote = args.remote 
    topic_model = args.model
    topics_list, topic_extraction_responses = extract_topics_per_context(contexts, topic_remote, model_name=topic_model, async_flag=args.async_flag)


    
    new_dataset = []
    topic_counts = []
    for i in range(len(dataset)):
        topics = topics_list[i]
        if len(topics) == 0:
            continue
        # to_delete_idx, _ = get_duplicate_contexts_embedding_cosine(topics, model=model)
        # topics = [topics[j] for j in range(len(topics)) if j not in to_delete_idx]
        topic_counts.append(len(topics))
        for topic in topics:
            dat = copy.deepcopy(dataset[i])
            dat['topic'] = topic
            new_dataset.append(dat)
    random.shuffle(new_dataset)
    dataset = new_dataset
    print(f"  average number of topics per context: {np.mean(topic_counts)}")
    print("Dataset has %d contexts after expansion per unique topic" % len(dataset))
    
    if args.sample_count > 0 and len(dataset) > args.sample_count:
        print(f"Dataset has more than {args.sample_count} contexts, keeping only the first {args.sample_count}")
        print(f"This enables reasonable runtime for the model evaluation")
        dataset = dataset[:args.sample_count]


    model_responses, failed_responses = build_questions(dataset, args.remote, args.model, async_flag=args.async_flag)

    elapsed_time = time.time() - start_time


    
    print(f"Saving {len(model_responses)} questions to {output_fn}")
    with open(output_fn, 'w') as f:
        json.dump(model_responses, f, indent=2)

    

