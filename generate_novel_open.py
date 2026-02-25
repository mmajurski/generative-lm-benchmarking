import numpy as np
import os
import copy
import json
import time
import random
import argparse

import answer_parser
from model_interface import SglModel
from generate_document_topics import extract_topics_per_context
import prompts


def build_questions(dataset: list[dict], remote: str, model_name: str, async_flag: bool = False):
    model_responses = list()
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

    model_prompts = [prompts.QUESTION_GEN_OPEN_PROMPT.format(context=d['context'], topic=d['topic']) for d in model_responses]

    model_obj = SglModel(remote=remote, model=model_name, sync_flag=(not async_flag))
    print(f"Generating questions against {model_obj.url}")
    results, total_time = model_obj.generate(model_prompts, reasoning_effort="high")
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

    model_responses = [d for d in model_responses if not d['failed']]
    for entry in model_responses:
        missing_keys = [key for key in ['question', 'context', 'answer'] if key not in entry]
        if missing_keys:
            raise AssertionError(f"Missing keys {missing_keys} in entry: {entry}")
    for response in model_responses:
        if 'failed' in response:
            del response['failed']

    return model_responses, failed_responses


def generate(input_filepath: str, output_filepath: str, remote: str,
                        model: str, async_flag: bool = True, sample_count: int = -1):
    os.makedirs(os.path.dirname(output_filepath) or '.', exist_ok=True)
    output_fn = output_filepath
    if os.path.exists(output_fn):
        print(f"Output file {output_fn} already exists, skipping")
        return

    start_time = time.time()

    with open(input_filepath, 'r') as f:
        data = json.load(f)

    sc = int(sample_count)
    if sample_count > 0 and sc < len(data):
        print(f"Sampling {sc} contexts from {len(data)}")
        data = random.sample(data, sc)

    for item in data:
        for key in [key for key in item.keys() if key != 'context']:
            del item[key]

    for item in data:
        if 'context' not in item:
            raise ValueError('each element in the dataset must have the following keys: context')

    print("Dataset has %d contexts" % len(data))
    contexts = [d['context'] for d in data]

    print("Evaluating the topics for each context")
    topics_list, _ = extract_topics_per_context(contexts, remote, model_name=model, async_flag=async_flag)

    new_dataset = []
    topic_counts = []
    num_empty_topics = 0
    for i in range(len(data)):
        topics = topics_list[i]
        if len(topics) == 0:
            num_empty_topics += 1
            continue
        topic_counts.append(len(topics))
        for topic in topics:
            dat = copy.deepcopy(data[i])
            dat['topic'] = topic
            new_dataset.append(dat)
    print(f"  {num_empty_topics} contexts had no topics extracted")
    print(f"  After topic expansion dataset has {len(new_dataset)} entries")
    random.shuffle(new_dataset)
    data = new_dataset
    print(f"  average number of topics per context: {np.mean(topic_counts)}")
    print("Dataset has %d contexts after expansion per unique topic" % len(data))
    if sample_count > 0 and len(data) > sample_count:
        print(f"Dataset has {len(data)} contexts, keeping a random {sample_count}")
        data = random.sample(data, sample_count)

    print("Generating questions based on the topics")
    data, failed_responses = build_questions(data, remote, model, async_flag=async_flag)
    print(f"Generated {len(data)} initial questions based on the topics")
    print(f"Failed to generate {len(failed_responses)} questions")

    elapsed_time = time.time() - start_time

    print(f"Evaluating the completeness of the {len(data)} questions")
    model_prompts = [prompts.QUESTION_VALIDITY_PROMPT.format(context=d['context'], question=d['question'], answer=d['answer']) for d in data]

    model_obj = SglModel(remote=remote, model=model)
    results, total_time = model_obj.generate(model_prompts, reasoning_effort="medium")
    print(f"in total took: {total_time} seconds")
    print(f"per question took: {total_time / len(results)} seconds for {len(results)} questions")

    for i in range(len(results)):
        res = results[i]
        if res['error'] is not None:
            raise Exception(f"Error: {res['error']}")
        parsed = answer_parser.parse_question_completeness_numbers(res['content'], valid_options=[1,2,3,4,5,6,7,8,9,10])
        if parsed is None:
            raise Exception(f"Failed to parse response: {res['content']}")
        data[i]['question_completeness_score'] = parsed['question_completeness_score']
        data[i]['answer_completeness_score'] = parsed['answer_completeness_score']
        data[i]['completeness_response'] = res['content']

    fail_dataset = [d for d in data if d['question_completeness_score'] < 5 and d['answer_completeness_score'] < 5]
    data = [d for d in data if d['question_completeness_score'] >= 5 and d['answer_completeness_score'] >= 5]
    print(f"Filtered out {len(fail_dataset)} items with completeness score less than 5")
    # if len(fail_dataset) > 0:
    #     with open(output_fn.replace('.json', '_completeness_fail.json'), 'w') as f:
    #         json.dump(fail_dataset, f, indent=2)

    print(f"Evaluating the meta properties of the {len(data)} questions")
    model_prompts = [prompts.META_PROPERTIES_PROMPT.format(context=d['context'], question=d['question'], answer=d['answer']) for d in data]

    model_obj = SglModel(remote=remote, model=model)
    results, total_time = model_obj.generate(model_prompts, reasoning_effort="medium")
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
        parsed = answer_parser.parse_meta_properties_numbers(res['content'], valid_options=[1,2,3,4,5,6,7,8,9,10])
        if parsed is None:
            raise Exception(f"Failed to parse response: {res['content']}")
        data[i]['question_clarity_score'] = parsed['clarity_score']
        data[i]['question_difficulty_score'] = parsed['difficulty_score']
        data[i]['question_groundedness_score'] = parsed['groundedness_score']

    original_length = len(data)
    data = [d for d in data if d['question_difficulty_score'] >= 5]
    print(f"Filtered out {original_length - len(data)} items with difficulty score less than 5")

    original_length = len(data)
    invalid_dataset = [d for d in data if d['question_groundedness_score'] < 5]
    # if len(invalid_dataset) > 0:
    #     with open(output_fn.replace('.json', '_grounding_fail.json'), 'w') as f:
    #         json.dump(invalid_dataset, f, indent=2)
    data = [d for d in data if d['question_groundedness_score'] >= 5]
    print(f"Filtered out {original_length - len(data)} items with groundedness score less than 5")

    print(f"Average question clarity score: {np.mean([d['question_clarity_score'] for d in data])}")
    print(f"Average question difficulty score: {np.mean([d['question_difficulty_score'] for d in data])}")
    print(f"Average question groundedness score: {np.mean([d['question_groundedness_score'] for d in data])}")

    print(f"Saving {len(data)} questions to {output_fn}")
    with open(output_fn, 'w') as f:
        json.dump(data, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Converts a jsonl dataset into MMLU format to be used as an open ended evaluation.')
    parser.add_argument('--input_filepath', type=str, required=True, help='dataset to generate, options: squadv2, ucinlp_drop')
    parser.add_argument('--output_filepath', type=str, required=True, help='source dataset directory')
    parser.add_argument('--remote', type=str, default="sierra")
    parser.add_argument('--model', type=str, default="meta-llama/Llama-3.3-70B-Instruct")
    parser.add_argument('--disable_async', action='store_false', dest='async_flag', help='Set to disable async processing (async is enabled by default)')
    parser.add_argument('--sample_count', type=int, default=-1, help='number of samples to generate, set to <=0 for all')
    parser.set_defaults(async_flag=True)

    args = parser.parse_args()
    print("Generating novel Open Ended MCQs")
    print(args)

    generate(
        input_filepath=args.input_filepath,
        output_filepath=args.output_filepath,
        remote=args.remote,
        model=args.model,
        async_flag=args.async_flag,
        sample_count=args.sample_count,
    )
